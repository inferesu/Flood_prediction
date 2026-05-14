import pandas as pd
import numpy as np
import tensorflow as tf
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

FEATURES_LIST = ['API_norm', 'S_t', 'SMI_t', 'Pt', 'delta_WL_t']
# Updated target name for 5 days
TARGET = 'target_change_5d'

# Estimated M1 Max power consumption
CPU_POWER_W = 30

tf.random.set_seed(42)
np.random.seed(42)

process = psutil.Process(os.getpid())

def get_memory_mb():
    return process.memory_info().rss / (1024 * 1024)

# ================= FEATURE ENGINEERING =================
def prepare_complex_features(df: pd.DataFrame):

    df = df.copy().sort_index()

    df['Pt'] = df[['precip_klaipedos-ams', 'precip_vezaiciu-ams']].mean(axis=1)

    api_vals, curr_api = [], 0
    for p in df['Pt']:
        curr_api = p + (K_DECAY * curr_api)
        api_vals.append(curr_api)

    df['API_t'] = api_vals
    df['API_norm'] = (df['API_t'] - df['API_t'].min()) / (df['API_t'].max() - df['API_t'].min())

    d = pd.to_datetime(df[DATE_COLUMN]).dt.dayofyear
    df['S_t'] = np.cos((2 * np.pi * d) / 365)

    avg_t = df[['temp_klaipedos-ams', 'temp_vezaiciu-ams']].mean(axis=1)
    df['SMI_t'] = avg_t.apply(lambda x: max(0, x * 2.5) if x > 0 else 0)

    df['delta_WL_t'] = df['water_level_cm'].diff().fillna(0)

    # MODIFIED: Predict 5 days ahead instead of 3
    df['target_change_5d'] = df['water_level_cm'].shift(-5) - df['water_level_cm']

    return df.dropna()

# ================= SEQUENCE CREATION =================
def create_sequences(X, y, look_back):

    Xs, ys = [], []

    for i in range(len(X) - look_back):
        Xs.append(X[i:(i + look_back)])
        ys.append(y[i + look_back - 1])

    return np.array(Xs), np.array(ys)

# ================= LOAD DATA =================
print("Preparing training data...")
train_df = prepare_complex_features(pd.read_csv(TRAIN_CSV))

print("Preparing test data...")
test_df = prepare_complex_features(pd.read_csv(TEST_CSV))

X_train_raw = train_df[FEATURES_LIST].values
y_train_raw = train_df[TARGET].values.reshape(-1, 1)

X_test_raw = test_df[FEATURES_LIST].values
y_test_raw = test_df[TARGET].values.reshape(-1, 1)

# ================= SCALE =================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train_raw)
X_test_scaled = scaler_X.transform(X_test_raw)

y_train_scaled = scaler_y.fit_transform(y_train_raw)
y_test_scaled = scaler_y.transform(y_test_raw)

# ================= CREATE SEQUENCES =================
X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, LOOK_BACK_PERIOD)
X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, LOOK_BACK_PERIOD)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ================= BUILD LSTM =================
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True),
        input_shape=(LOOK_BACK_PERIOD, len(FEATURES_LIST))
    ),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.Huber()
)

model.summary()

# ================= CALLBACKS =================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        verbose=1
    )
]

# ================= TRAIN =================
print("\nTraining...")

mem_before = get_memory_mb()
start_train = time.time()

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=2
)

train_time = time.time() - start_train
mem_after = get_memory_mb()

energy_train = CPU_POWER_W * train_time

print("\n--- Training Resources ---")
print(f"Training Time : {train_time:.4f} sec")
print(f"Memory Usage  : {mem_after - mem_before:.2f} MB")
print(f"Estimated Energy : {energy_train:.4f} Joules")

# ================= INFERENCE =================
print("\nPredicting...")

mem_before = get_memory_mb()
start_pred = time.time()

y_pred_scaled = model.predict(X_test)

pred_time = time.time() - start_pred
mem_after = get_memory_mb()

energy_pred = CPU_POWER_W * pred_time

print("\n--- Inference Resources ---")
print(f"Inference Time : {pred_time:.4f} sec")
print(f"Time per Sample: {(pred_time / len(X_test)) * 1000:.6f} ms")
print(f"Memory Usage   : {mem_after - mem_before:.2f} MB")
print(f"Estimated Energy: {energy_pred:.4f} Joules")

# ================= RECONSTRUCT WATER LEVEL =================
y_pred_change = scaler_y.inverse_transform(y_pred_scaled).flatten()
y_true_change = scaler_y.inverse_transform(y_test).flatten()

wl_test_raw = test_df['water_level_cm'].values

wl_base = []
for i in range(len(test_df) - LOOK_BACK_PERIOD):
    # Base level is still the last day of the look_back period
    wl_base.append(wl_test_raw[i + LOOK_BACK_PERIOD - 1])

wl_base = np.array(wl_base)

# Calculate final predicted and true water levels based on the 5-day change
wl_pred = wl_base + y_pred_change
wl_true = wl_base + y_true_change

# ================= EVALUATION =================
print("\n--- Model Evaluation (5D) ---")

mse = mean_squared_error(wl_true, wl_pred)
rmse = np.sqrt(mse)
r2 = r2_score(wl_true, wl_pred)
mae = mean_absolute_error(wl_true, wl_pred)
nrmse = rmse / (np.max(wl_true) - np.min(wl_true))

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f} cm")
print(f"R2: {r2:.4f}")
print(f"MAE: {mae:.4f} cm")
print(f"NRMSE: {nrmse:.4f}")

print("\n--- Computational Cost ---")
print(f"Training Time : {train_time:.4f} sec")
print(f"Inference Time: {pred_time:.4f} sec")

# ================= EXPORT =================
# We align the timestamps to the predicted target day (+5 days from the end of the lookback sequence)
aligned_timestamps = test_df[DATE_COLUMN].iloc[LOOK_BACK_PERIOD:].values

out_df = pd.DataFrame({
    "timestamp": aligned_timestamps,
    "WL_true": wl_true,
    "WL_pred": wl_pred
})

# MODIFIED: Output file renamed for 5D model
out_df.to_csv("rnn_predictions_5d.csv", index=False)
print("\nSaved predictions to 'rnn_predictions_5d.csv'.")

# ================= PLOT =================
plt.figure(figsize=(14, 6))
plt.plot(wl_true, label="Actual WL (5 Days Ahead)")
plt.plot(wl_pred, linestyle="--", alpha=0.8, label="Predicted WL (5 Days Ahead)")
plt.title("LSTM 5D Prediction of Water Level")
plt.legend()
plt.grid(True)
plt.show()