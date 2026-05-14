import pandas as pd
import numpy as np
import xgboost as xgb
import time
import psutil
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# ================= CONFIG =================
LOOK_BACK_PERIOD = 14
TRAIN_CSV = 'minija_complex_data_2024.csv'
TEST_CSV = 'minija_complex_data_test.csv'
DATE_COLUMN = 'timestamp'
K_DECAY = 0.85

# MODIFIED: Predict the change over 3 days
TARGET = 'target_change_3d'

# Apple M1 Max estimated CPU power
CPU_POWER_W = 30

np.random.seed(42)

process = psutil.Process(os.getpid())

def get_memory_mb():
    return process.memory_info().rss / (1024 * 1024)

# ================= FEATURE ENGINEERING =================
def prepare_features(df):
    df = df.copy().sort_index()

    # Average precipitation
    df['Pt'] = df[['precip_klaipedos-ams', 'precip_vezaiciu-ams']].mean(axis=1)

    # API
    api_vals, curr_api = [], 0
    for p in df['Pt']:
        curr_api = p + (K_DECAY * curr_api)
        api_vals.append(curr_api)
    df['API_t'] = api_vals
    df['API_norm'] = (df['API_t'] - df['API_t'].min()) / (df['API_t'].max() - df['API_t'].min())

    # Seasonality
    d = pd.to_datetime(df[DATE_COLUMN]).dt.dayofyear
    df['S_t'] = np.cos((2 * np.pi * d) / 365)

    # Snowmelt
    avg_t = df[['temp_klaipedos-ams', 'temp_vezaiciu-ams']].mean(axis=1)
    df['SMI_t'] = avg_t.apply(lambda x: max(0, x * 2.5) if x > 0 else 0)

    # Water level change
    df['delta_WL_t'] = df['water_level_cm'].diff().fillna(0)

    # MODIFIED: Target Change shifted by 3 days
    df['target_change_3d'] = df['water_level_cm'].shift(-3) - df['water_level_cm']

    return df.dropna()

# ================= LOAD DATA =================
print("Preparing training data...")
train_df = pd.read_csv(TRAIN_CSV)
train_df[DATE_COLUMN] = pd.to_datetime(train_df[DATE_COLUMN])
train_df = prepare_features(train_df)

print("Preparing test data...")
test_df = pd.read_csv(TEST_CSV)
test_df[DATE_COLUMN] = pd.to_datetime(test_df[DATE_COLUMN])
test_df = prepare_features(test_df)

FEATURES = ['API_norm', 'S_t', 'SMI_t', 'Pt', 'delta_WL_t']

X_train_raw = train_df[FEATURES].values
y_train_raw = train_df[TARGET].values.reshape(-1, 1)

X_test_raw = test_df[FEATURES].values
y_test_raw = test_df[TARGET].values.reshape(-1, 1)

# ================= SCALE =================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train_raw)
X_test_scaled = scaler_X.transform(X_test_raw)

y_train_scaled = scaler_y.fit_transform(y_train_raw)
y_test_scaled = scaler_y.transform(y_test_raw)

# ================= CREATE SEQUENCES =================
def create_flat_sequences(X, y, look_back):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:(i + look_back)].flatten())
        # Target corresponds to the LAST step in the sequence window
        ys.append(y[i + look_back - 1])
    return np.array(Xs), np.array(ys)

X_train, y_train = create_flat_sequences(X_train_scaled, y_train_scaled, LOOK_BACK_PERIOD)
X_test, y_test = create_flat_sequences(X_test_scaled, y_test_scaled, LOOK_BACK_PERIOD)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ================= BUILD IMPROVED XGBOOST =================
model = xgb.XGBRegressor(
    n_estimators=2000,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    min_child_weight=3,
    gamma=0.1,
    objective='reg:squarederror',
    early_stopping_rounds=50,
    random_state=42
)

# ================= TRAIN MODEL =================
print("\n--- Training XGBoost ---")

mem_before = get_memory_mb()
start_train = time.time()

model.fit(
    X_train, y_train.ravel(),
    eval_set=[(X_test, y_test.ravel())],
    verbose=False
)

train_time = time.time() - start_train
mem_after = get_memory_mb()

energy_joules = CPU_POWER_W * train_time

print("\n--- Training Resources ---")
print(f"Training Time: {train_time:.6f} sec")
print(f"Memory Usage: {mem_after - mem_before:.2f} MB")
print(f"Estimated Energy: {energy_joules:.4f} Joules")

# ================= INFERENCE =================
print("\n--- Running Inference ---")

mem_before = get_memory_mb()
start_pred = time.time()

y_pred_scaled = model.predict(X_test).reshape(-1, 1)

pred_time = time.time() - start_pred
mem_after = get_memory_mb()

energy_inference = CPU_POWER_W * pred_time

print("\n--- Inference Resources ---")
print(f"Inference Time: {pred_time:.6f} sec")
print(f"Time per Sample: {(pred_time/len(X_test))*1000:.6f} ms")
print(f"Memory Usage: {mem_after - mem_before:.2f} MB")
print(f"Estimated Energy: {energy_inference:.4f} Joules")

# ================= RECONSTRUCT WATER LEVEL =================
# Inverse transform the predicted delta changes
y_pred_change = scaler_y.inverse_transform(y_pred_scaled).flatten()
y_true_change = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Get the actual base water level corresponding to the last step of each sequence
wl_test_raw = test_df['water_level_cm'].values
wl_base = []
for i in range(len(test_df) - LOOK_BACK_PERIOD):
    wl_base.append(wl_test_raw[i + LOOK_BACK_PERIOD - 1])
wl_base = np.array(wl_base)

# Reconstruct actual absolute water levels for a fair comparison
wl_pred = wl_base + y_pred_change
wl_true = wl_base + y_true_change

# ================= EVALUATION =================
print("\n--- XGBoost Evaluation (3D Delta-aligned) ---")

mse = mean_squared_error(wl_true, wl_pred)
rmse = np.sqrt(mse)
r2 = r2_score(wl_true, wl_pred)
mae = mean_absolute_error(wl_true, wl_pred)
nrmse = rmse / (np.max(wl_true) - np.min(wl_true))

print(f"MSE:   {mse:.4f}")
print(f"RMSE:  {rmse:.4f} cm")
print(f"R2:    {r2:.4f}")
print(f"MAE:   {mae:.4f} cm")
print(f"NRMSE: {nrmse:.4f} ({nrmse * 100:.2f}%)")

# ================= EXPORT FOR DM TEST =================
# Align timestamps (we lose the first 'LOOK_BACK_PERIOD' days due to sequence creation)
aligned_timestamps = test_df[DATE_COLUMN].iloc[LOOK_BACK_PERIOD:].values

out_df = pd.DataFrame({
    "timestamp": aligned_timestamps,
    "WL_true": wl_true,
    "WL_pred": wl_pred
})

# MODIFIED: Changed export filename to prevent overwriting
out_df.to_csv("xgboost_predictions_3d.csv", index=False)
print("\nSaved predictions to 'xgboost_predictions_3d.csv'.")

# ================= PLOT =================
plt.figure(figsize=(14, 6))
plt.plot(wl_true, label="Actual WL (3 Days Ahead)")
plt.plot(wl_pred, linestyle="--", alpha=0.8, label="Predicted WL (3 Days Ahead)")
plt.title("XGBoost 3D Prediction of Water Level (Delta-aligned)")
plt.legend()
plt.grid(True)
plt.show()