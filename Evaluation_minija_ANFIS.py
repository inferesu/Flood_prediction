import json
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- Import ANFIS classes to be able to load the models ---
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# --- File Paths ---
ANFIS_MODEL_PATH = "anfis_model.pth"
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_Y.pkl"
CONFIG_JSON_PATH = "training_config.json"
TEST_DATA_FILE = 'tests2015-2022.csv'


# --- Model and Feature Definitions (Must be identical to training scripts) ---

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """A copy of the feature prep function from the ANFIS training script."""
    df = df.copy()
    df.sort_index(inplace=True)
    df.interpolate(method='time', inplace=True)
    df['precip_klaipedos_lag_12h'] = df['precip_klaipedos-ams_mm'].rolling(12, min_periods=1).sum()
    df['precip_klaipedos_lag_24h'] = df['precip_klaipedos-ams_mm'].rolling(24, min_periods=1).sum()
    df['precip_klaipedos_lag_48h'] = df['precip_klaipedos-ams_mm'].rolling(48, min_periods=1).sum()
    df['precip_klaipedos_lag_72h'] = df['precip_klaipedos-ams_mm'].rolling(72, min_periods=1).sum()
    df['precip_vezaiciu_lag_12h'] = df['precip_vezaiciu-ams_mm'].rolling(12, min_periods=1).sum()
    df['precip_vezaiciu_lag_24h'] = df['precip_vezaiciu-ams_mm'].rolling(24, min_periods=1).sum()
    df['precip_vezaiciu_lag_48h'] = df['precip_vezaiciu-ams_mm'].rolling(48, min_periods=1).sum()
    df['precip_vezaiciu_lag_72h'] = df['precip_vezaiciu-ams_mm'].rolling(72, min_periods=1).sum()
    df['target_change'] = df['water_level_cm'].shift(-1) - df['water_level_cm']
    df.dropna(inplace=True)
    return df


def build_anfis(num_inputs: int, num_mfs: int) -> AnfisNet:
    """Reconstructs the ANFIS model structure."""
    invardefs = []
    for i in range(num_inputs):
        mfs = [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(num_mfs)]
        invardefs.append((f'x{i}', mfs))
    return AnfisNet('Flood Prediction Model', invardefs, ['y'], hybrid=True)


def evaluate_anfis_model():
    """Main function to run the ANFIS evaluation."""
    print("--- Step 1: Loading Models and Artifacts ---")

    with open(CONFIG_JSON_PATH, "r") as f:
        config = json.load(f)
    features_list = config["features_list"]

    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    anfis_model = build_anfis(config["num_inputs"], config["num_mfs"])
    checkpoint = torch.load(ANFIS_MODEL_PATH, map_location="cpu")
    anfis_model.load_state_dict(checkpoint['model_state_dict'])
    anfis_model.coeff = checkpoint['consequent_coeffs']
    anfis_model.eval()
    print("✅ ANFIS model and scalers loaded.")

    print("\n--- Step 2: Preparing Test Data ---")
    df_test = pd.read_csv(TEST_DATA_FILE, parse_dates=['timestamp'], index_col='timestamp')
    df_test = prepare_features(df_test)
    X_test_unscaled = df_test[features_list].values
    y_true_change = df_test[config["target"]].values

    X_test_scaled = scaler_X.transform(X_test_unscaled)
    X_test_tensor = torch.from_numpy(X_test_scaled).float()

    print("\n--- Step 3: Performing Prediction ---")
    with torch.no_grad():
        anfis_preds_scaled = anfis_model(X_test_tensor).numpy().flatten()
        anfis_preds_change = scaler_y.inverse_transform(anfis_preds_scaled.reshape(-1, 1)).flatten()

    print("✅ Prediction complete.")

    print("\n--- Step 4: Final Evaluation ---")
    base_water_level = df_test['water_level_cm'].values
    predicted_level = base_water_level + anfis_preds_change
    actual_level = base_water_level + y_true_change

    # --- Metrics Calculation ---
    mse = mean_squared_error(actual_level, predicted_level)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_level, predicted_level)
    mae = mean_absolute_error(actual_level, predicted_level)

    # NRMSE calculation (normalized by the range of the actual data)
    nrmse = rmse / (actual_level.max() - actual_level.min())

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Normalized RMSE (NRMSE): {nrmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} cm")

    results_df = pd.DataFrame({
        'Timestamp': df_test.index,
        'Actual Water Level (cm)': actual_level,
        'ANFIS Prediction (cm)': predicted_level,
        'Error (cm)': actual_level - predicted_level
    })

    print("\n--- Predictions vs. Actuals ---")
    print(results_df.head(20).to_string(index=False))


if __name__ == "__main__":
    evaluate_anfis_model()