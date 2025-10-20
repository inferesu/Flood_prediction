import json
import random
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_

# --- Import ANFIS classes ---
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# --- Configuration & Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- File Paths ---
TRAIN_DATA_FILE = 'minija_multi_weather_data_2015-2022.csv'
MODEL_SAVE_PATH = "anfis_forecaster_24h.pth"
SCALER_X_PATH = "scaler_X_forecaster_24h.pkl"
SCALER_Y_PATH = "scaler_Y_forecaster_24h.pkl"
CONFIG_JSON_PATH = "config_forecaster_24h.json"

# --- Hyperparameters ---
BATCH_SIZE = 16
EPOCHS = 200
LR = 1e-3
# MOMENTUM is not used by the Adam optimizer
NUM_MFS = 2

# --- Feature Engineering for Forecasting ---
FEATURES_LIST = [
    'precip_klaipedos_lag_24h',
    'precip_vezaiciu_lag_24h',
    'temp_c',
    'precip_forecast_sum_12h',
    'precip_forecast_sum_24h',
    'temp_forecast_avg_12h',
    'temp_forecast_avg_24h',
    'thawing_hours_forecast_24h',
]
TARGET = 'target_change_24h'


def prepare_forecasting_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates features for a 24-hour ahead forecasting model.
    It uses future data to simulate a "perfect forecast".
    """
    df = df.copy()
    df.sort_index(inplace=True)
    df.interpolate(method='time', inplace=True)

    df['precip_klaipedos_lag_24h'] = df['precip_klaipedos-ams_mm'].rolling(24, min_periods=1).sum()
    df['precip_vezaiciu_lag_24h'] = df['precip_vezaiciu-ams_mm'].rolling(24, min_periods=1).sum()

    df['precip_forecast_sum_12h'] = df['precip_klaipedos-ams_mm'].rolling(12).sum().shift(-11)
    df['precip_forecast_sum_24h'] = df['precip_klaipedos-ams_mm'].rolling(24).sum().shift(-23)

    df['temp_forecast_avg_12h'] = df['temp_c'].rolling(12).mean().shift(-11)
    df['temp_forecast_avg_24h'] = df['temp_c'].rolling(24).mean().shift(-23)

    is_thawing = (df['temp_c'] > 0).astype(int)
    df['thawing_hours_forecast_24h'] = is_thawing.rolling(24).sum().shift(-23)

    df['target_change_24h'] = df['water_level_cm'].shift(-24) - df['water_level_cm']

    df.dropna(inplace=True)
    return df


def build_anfis(num_inputs: int, num_mfs: int) -> AnfisNet:
    """
    Creates an ANFIS model with a safer random initialization.
    """
    invardefs = []
    for i in range(num_inputs):
        # MODIFIED: Initialize 'a' and 'b' in a safer range [0.1, 1.1] to prevent them from starting near zero.
        # The center 'c' can still be anywhere in [0, 1].
        mfs = [BellMembFunc(a=0.1 + torch.rand(1), b=0.1 + torch.rand(1), c=torch.rand(1)) for _ in range(num_mfs)]
        invardefs.append((f'x{i}', mfs))
    return AnfisNet('Flood Forecaster Model', invardefs, ['y'], hybrid=True)


def clamp_membership_params(model):
    """
    A function to keep membership function parameters in a stable range.
    """
    for layer in model.layer.values():
        if hasattr(layer, 'varmfs'):
            for var in layer.varmfs.values():
                for mf in var.mfdefs.values():
                    if isinstance(mf, BellMembFunc):
                        mf.a.data.clamp_(min=1e-2)
                        mf.b.data.clamp_(min=1e-2)


def train_forecaster():
    """Main function to train the 24-hour forecaster model."""
    print("--- Step 1: Load and Prepare Training Data for Forecasting ---")
    df_train = pd.read_csv(TRAIN_DATA_FILE, parse_dates=['timestamp'], index_col='timestamp')
    df_train = prepare_forecasting_features(df_train)

    X_train = df_train[FEATURES_LIST].values
    y_train = df_train[TARGET].values

    print("\n--- Step 2: Scale Data and Save Scalers ---")
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)
    print(f"✅ Forecaster scalers saved to {SCALER_X_PATH} and {SCALER_Y_PATH}")

    x_train_tensor = torch.from_numpy(X_train_scaled).float()
    y_train_tensor = torch.from_numpy(y_train_scaled).float()
    train_dl = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)

    print("\n--- Step 3: Define and Build ANFIS Forecaster Model ---")
    num_inputs = len(FEATURES_LIST)
    model = build_anfis(num_inputs, NUM_MFS)
    print(f"✅ Model built successfully with {model.num_rules} rules.")
    # MODIFIED: Switched to the Adam optimizer for better stability.
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    print("\n--- Step 4: Training Loop ---")
    for epoch in range(EPOCHS):
        print(f'Starting Epoch {epoch + 1}/{EPOCHS}...')
        for x_batch, y_batch in train_dl:
            model.train()
            optimizer.zero_grad()
            y_pred_scaled = model(x_batch)

            if torch.isnan(y_pred_scaled).any():
                print("   - Warning: NaN detected in model output. Skipping batch.")
                continue

            loss = criterion(y_pred_scaled, y_batch)

            if torch.isnan(loss):
                print("   - Warning: Loss is NaN. Skipping backward pass.")
                continue

            loss.backward()

            clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            clamp_membership_params(model)

        with torch.no_grad():
            model.fit_coeff(x_train_tensor, y_train_tensor)

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                y_pred_final = model(x_train_tensor)
                epoch_loss = criterion(y_pred_final, y_train_tensor)
                print(f'   -> Epoch {epoch + 1} Loss: {epoch_loss.item():.4f}')

    print("\n--- Step 5: Save Forecaster Model and Configuration ---")
    torch.save({
        'model_state_dict': model.state_dict(),
        'consequent_coeffs': model.coeff,
    }, MODEL_SAVE_PATH)
    print(f"✅ Forecaster model saved to {MODEL_SAVE_PATH}")

    config = {
        "features_list": FEATURES_LIST,
        "num_inputs": len(FEATURES_LIST),
        "num_mfs": NUM_MFS,
        "target": TARGET
    }
    with open(CONFIG_JSON_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print(f"✅ Forecaster config saved to {CONFIG_JSON_PATH}")


if __name__ == "__main__":
    train_forecaster()