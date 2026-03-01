import pandas as pd
import numpy as np
import tensorflow as tf
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
# MATCH ANFIS TARGET: Predict the change, not the absolute value
TARGET = 'target_change'

tf.random.set_seed(42)
np.random.seed(42)


# ================= FEATURE ENGINEERING =================
def prepare_complex_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()

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

    # Target Change (Eq. 21 analog) - ALIGNED WITH ANFIS
    df['target_change'] = df['water_level_cm'].shift(-1) - df['water_level_cm']

    return df.dropna()


# ================= SEQUENCE CREATION =================
def create_sequences(X, y, look_back):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:(i + look_back)])
        # Target corresponds to the LAST step in the sequence window
        ys.append(y[i + look_back - 1])
    return np.array(Xs), np.array(ys)


# ================= LOAD DATA =================
print("Preparing training data...")
train_df = prepare_complex_features(pd.read_csv(TRAIN_CSV))
X_train_raw = train_df[FEATURES_LIST].values
y_train_raw = train_df[TARGET].values.reshape(-1, 1)

print("Preparing test data...")
test_df = prepare_complex_features(pd.read_csv(TEST_CSV))
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

# ================= BUILD IMPROVED LSTM =================
model = tf.keras.Sequential([
    # Bidirectional LSTM captures context from both directions of the sequence
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
    loss=tf.keras.losses.Huber()  # More robust to extreme flood peaks than MSE
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
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=2
)

# ================= PREDICT & RECONSTRUCT =================
print("\nPredicting...")
y_pred_scaled = model.predict(X_test)

# Inverse transform the predicted delta changes
y_pred_change = scaler_y.inverse_transform(y_pred_scaled).flatten()
y_true_change = scaler_y.inverse_transform(y_test).flatten()

# Get the actual base water level corresponding to the last step of each sequence
wl_test_raw = test_df['water_level_cm'].values
wl_base = []
for i in range(len(test_df) - LOOK_BACK_PERIOD):
    wl_base.append(wl_test_raw[i + LOOK_BACK_PERIOD - 1])
wl_base = np.array(wl_base)

# Reconstruct actual absolute water levels for a fair 1:1 comparison with ANFIS
wl_pred = wl_base + y_pred_change
wl_true = wl_base + y_true_change

# ================= EVALUATION =================
print("\n--- Evaluation on absolute water_level_cm ---")

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

# ================= PLOT =================
plt.figure(figsize=(14, 6))
plt.plot(wl_true, label="Actual WL (Reconstructed)")
plt.plot(wl_pred, linestyle="--", alpha=0.8, label="Predicted WL (Reconstructed)")
plt.title("LSTM Prediction of Water Level (Delta-aligned)")
plt.legend()
plt.grid(True)
plt.show()