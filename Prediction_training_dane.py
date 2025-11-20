import json
import random
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

# Ensure these imports exist at runtime
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# --- Configuration & Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- File Paths (Updated with "dane") ---
TRAIN_DATA_FILE = 'dane-klaipeda_precip_data_2015-2024.csv'
MODEL_SAVE_PATH = "anfis_model_dane.pth"
SCALER_X_PATH = "scaler_X_dane.pkl"
SCALER_Y_PATH = "scaler_Y_dane.pkl"
CONFIG_JSON_PATH = "training_config_dane.json"

# --- Hyperparameters (matching original script) ---
BATCH_SIZE = 16
EPOCHS = 200
LR = 1e-3  # Keeping LR the same for now, as clipping is the primary fix
MOMENTUM = 0.9
NUM_MFS = 3
CLIP_GRAD_NORM = 1.0  # NEW: Gradient clipping maximum norm

# --- Feature Engineering (Updated for "dane" - Klaipedos only) ---
FEATURES_LIST = [
    'precip_klaipedos_lag_12h', 'precip_klaipedos_lag_24h',
    'precip_klaipedos_lag_48h', 'precip_klaipedos_lag_72h'
]
TARGET = 'target_change'


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_index(inplace=True)
    df.interpolate(method='time', inplace=True)

    # Feature engineering for Klaipeda
    df['precip_klaipedos_lag_12h'] = df['precip_klaipedos-ams_mm'].rolling(12, min_periods=1).sum()
    df['precip_klaipedos_lag_24h'] = df['precip_klaipedos-ams_mm'].rolling(24, min_periods=1).sum()
    df['precip_klaipedos_lag_48h'] = df['precip_klaipedos-ams_mm'].rolling(48, min_periods=1).sum()
    df['precip_klaipedos_lag_72h'] = df['precip_klaipedos-ams_mm'].rolling(72, min_periods=1).sum()

    df['target_change'] = df['water_level_cm'].shift(-1) - df['water_level_cm']
    df.dropna(inplace=True)
    return df


def build_anfis(num_inputs: int, num_mfs: int) -> AnfisNet:
    """
    Creates an ANFIS model with random initialization, matching the original script.
    """
    invardefs = []
    for i in range(num_inputs):
        # Using random initialization as per the first script
        mfs = [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(num_mfs)]
        invardefs.append((f'x{i}', mfs))

    return AnfisNet('Flood Prediction Model (Dane)', invardefs, ['y'], hybrid=True)


def train_and_save_model():
    """Main function to run the training and save artifacts."""
    print("--- Step 1: Load and Prepare Training Data ---")
    df_train = pd.read_csv(TRAIN_DATA_FILE, parse_dates=['timestamp'], index_col='timestamp')
    df_train = prepare_features(df_train)

    X_train = df_train[FEATURES_LIST].values
    y_train = df_train[TARGET].values

    print("\n--- Step 2: Scale Data and Save Scalers ---")
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

    # NEW: Check for NaNs in scaled data
    if np.isnan(X_train_scaled).any() or np.isnan(y_train_scaled).any():
        print("❌ Error: NaNs found in scaled training data. Aborting.")
        return

    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)
    print(f"Scalers saved to {SCALER_X_PATH} and {SCALER_Y_PATH}")

    x_train_tensor = torch.from_numpy(X_train_scaled).float()
    y_train_tensor = torch.from_numpy(y_train_scaled).float()
    train_dl = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)

    print("\n--- Step 3: Define and Build ANFIS Model ---")
    num_inputs = len(FEATURES_LIST)
    print(f"Building model with {num_inputs} inputs (from Klaipedos station only).")
    model = build_anfis(num_inputs, NUM_MFS)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    criterion = torch.nn.MSELoss()

    # Keep track of the last good model state
    last_good_model_state = model.state_dict()
    last_good_coeffs = model.coeff

    print("\n--- Step 4: Training Loop ---")
    nan_detected = False
    for epoch in range(EPOCHS):
        if nan_detected:
            break

        epoch_loss_total = 0.0
        num_batches = 0

        for x_batch, y_batch in train_dl:
            model.train()
            optimizer.zero_grad()
            y_pred_scaled = model(x_batch)

            # Check for NaNs in prediction
            if torch.isnan(y_pred_scaled).any():
                print(f"❌ NaN detected in model output during Epoch {epoch + 1}. Stopping gradient step.")
                nan_detected = True
                break

            loss = criterion(y_pred_scaled, y_batch)

            # Check for NaNs in loss
            if torch.isnan(loss):
                print(f"❌ NaN detected in loss during Epoch {epoch + 1}. Stopping gradient step.")
                nan_detected = True
                break

            loss.backward()

            # --- NEW: GRADIENT CLIPPING ---
            # Clip gradients to prevent them from exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)

            optimizer.step()

            epoch_loss_total += loss.item()
            num_batches += 1

        if nan_detected:
            print("NaN detected, stopping training.")
            break

        # --- MODIFIED: More stable hybrid learning step ---
        with torch.no_grad():
            model.fit_coeff(x_train_tensor, y_train_tensor)

            # --- NEW: Check for NaN after fit_coeff ---
            if torch.isnan(model.coeff).any():
                print(f"❌ NaN detected in model.coeff after fit_coeff in Epoch {epoch + 1}. Stopping training.")
                nan_detected = True
                break  # Exit the epoch loop
            else:
                # Update the last good state
                last_good_model_state = model.state_dict()
                last_good_coeffs = model.coeff

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                y_pred_final = model(x_train_tensor)
                epoch_loss = criterion(y_pred_final, y_train_tensor)
                print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss.item():.4f}')

    print("\n--- Step 5: Save Model and Configuration ---")
    if nan_detected:
        print("⚠️ WARNING: Training stopped due to NaN. Saving the *last good model state* before the error.")
    else:
        print("✅ Training completed successfully.")

    # Save the last good state
    # Save the last good state
    torch.save({
        'model_state_dict': last_good_model_state,
        'consequent_coeffs': last_good_coeffs,
    }, MODEL_SAVE_PATH)
    print(f"Model and coefficients saved to {MODEL_SAVE_PATH}")

    config = {
        "features_list": FEATURES_LIST,
        "num_inputs": len(FEATURES_LIST),
        "num_mfs": NUM_MFS,
        "target": TARGET
    }
    with open(CONFIG_JSON_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Training config saved to {CONFIG_JSON_PATH}")


if __name__ == "__main__":
    train_and_save_model()