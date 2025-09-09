import json
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import r2_score, mean_absolute_error

from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# Paths
TEST_DATA_FILE = 'new_test.csv'
MODEL_SAVE_PATH = "anfis_model.pth"  # Correct path to the saved model bundle
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_Y.pkl"
CONFIG_JSON_PATH = "training_config.json"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares features for the ANFIS model."""
    df = df.copy()
    df.sort_index(inplace=True)
    df.interpolate(method='time', inplace=True)

    # Feature engineering for Klaipeda
    df['precip_klaipedos_lag_12h'] = df['precip_klaipedos-ams_mm'].rolling(12, min_periods=1).sum()
    df['precip_klaipedos_lag_24h'] = df['precip_klaipedos-ams_mm'].rolling(24, min_periods=1).sum()
    df['precip_klaipedos_lag_48h'] = df['precip_klaipedos-ams_mm'].rolling(48, min_periods=1).sum()
    df['precip_klaipedos_lag_72h'] = df['precip_klaipedos-ams_mm'].rolling(72, min_periods=1).sum()

    # Feature engineering for Vezaiciai
    df['precip_vezaiciu_lag_12h'] = df['precip_vezaiciu-ams_mm'].rolling(12, min_periods=1).sum()
    df['precip_vezaiciu_lag_24h'] = df['precip_vezaiciu-ams_mm'].rolling(24, min_periods=1).sum()
    df['precip_vezaiciu_lag_48h'] = df['precip_vezaiciu-ams_mm'].rolling(48, min_periods=1).sum()
    df['precip_vezaiciu_lag_72h'] = df['precip_vezaiciu-ams_mm'].rolling(72, min_periods=1).sum()

    df['target_change'] = df['water_level_cm'].shift(-1) - df['water_level_cm']
    df.dropna(inplace=True)
    return df


def build_anfis(num_inputs: int, num_mfs: int) -> AnfisNet:
    """
    Recreates the ANFIS model structure.
    The actual parameters will be overwritten by the loaded state_dict.
    """
    invardefs = []
    for i in range(num_inputs):
        # The initial values don't matter as they will be loaded from the state dict
        mfs = [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(num_mfs)]
        invardefs.append((f'x{i}', mfs))

    return AnfisNet('Flood Prediction Model', invardefs, ['y'], hybrid=True)


def evaluate_model():
    df_test = pd.read_csv(TEST_DATA_FILE, parse_dates=['timestamp'], index_col='timestamp')
    df_test = prepare_features(df_test)

    with open(CONFIG_JSON_PATH, "r") as f:
        config = json.load(f)
    features_list = config["features_list"]
    target = config["target"]

    X_test = df_test[features_list].values
    y_true_change = df_test[target].values

    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    print("Scalers loaded successfully.")

    X_test_scaled = scaler_X.transform(X_test)
    x_test_tensor = torch.from_numpy(X_test_scaled).float()

    model = build_anfis(num_inputs=config["num_inputs"], num_mfs=config["num_mfs"])

    # Load the checkpoint dictionary
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location="cpu")

    # Load the state_dict for parameters learned by backprop
    model.load_state_dict(checkpoint['model_state_dict'])

    # Manually set the consequent coefficients learned by LSE
    model.coeff = checkpoint['consequent_coeffs']

    model.eval()
    print("Model rebuilt and trained weights loaded.")

    print("\n--- Step 4: Make Predictions ---")
    with torch.no_grad():
        y_pred_scaled_tensor = model(x_test_tensor)

    y_pred_scaled = y_pred_scaled_tensor.numpy()
    y_pred_change = scaler_y.inverse_transform(y_pred_scaled).flatten()

    # Calculate final predicted water level
    base_water_level = df_test['water_level_cm'].values
    predicted_level = base_water_level + y_pred_change
    actual_level = base_water_level + y_true_change

    print("\n--- Final Evaluation Results ---")
    r2 = r2_score(actual_level, predicted_level)
    mae = mean_absolute_error(actual_level, predicted_level)
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} cm")

    results_df = pd.DataFrame({
        'Timestamp': df_test.index,
        'Actual Water Level (cm)': actual_level,
        'Predicted Water Level (cm)': predicted_level
    })
    print("\n--- Predictions vs. Actuals ---")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    evaluate_model()