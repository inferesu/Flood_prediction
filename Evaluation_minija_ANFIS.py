import json
import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- Import ANFIS classes ---
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# --- File Paths & Constants ---
ANFIS_MODEL_PATH = "anfis_model.pth"
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_Y.pkl"
CONFIG_JSON_PATH = "training_config.json"
TEST_DATA_FILE = 'tests2015-2022.csv'
K_DECAY = 0.85  # Decay coefficient k (Eq. 9)


def prepare_complex_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates hydrological indices (Eq. 9-23)"""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    df.sort_index(inplace=True)
    df.interpolate(method='time', inplace=True)

    # 1. Pt - Mean Precipitation (Using flexible column matching for _mm)
    p_cols = [c for c in df.columns if 'precip' in c and ('klaipedos' in c or 'vezaiciu' in c)]
    if not p_cols:
        raise KeyError(f"Could not find precipitation columns in {df.columns}")
    df['Pt'] = df[p_cols].mean(axis=1)

    # 2. API_t = Pt + k * API_t-1 (Eq. 9)
    api_values, current_api = [], 0
    for p in df['Pt']:
        current_api = p + (K_DECAY * current_api)
        api_values.append(current_api)
    df['API_t'] = api_values

    # 3. API Normalization (Eq. 10)
    # Formula: (API - API_min) / (API_max - API_min)
    api_min, api_max = df['API_t'].min(), df['API_t'].max()
    df['API_norm'] = (df['API_t'] - api_min) / (api_max - api_min)

    # 4. Seasonality (Eq. 11)
    # Season_cos = cos(2*pi*d / 365)
    df['S_t'] = np.cos((2 * np.pi * df.index.dayofyear) / 365)

    # 5. Snowmelt Index (Eq. 17)
    # SMI contributes to WL increase even in absence of precipitation
    t_cols = [c for c in df.columns if 'temp' in c]
    if t_cols:
        avg_temp = df[t_cols].mean(axis=1)
        df['SMI_t'] = avg_temp.apply(lambda x: max(0, x * 2.5) if x > 0 else 0)
    else:
        df['SMI_t'] = 0

    # 6. Trend Persistence (Eq. 21)
    df['delta_WL_t'] = df['water_level_cm'].diff().fillna(0)
    df['target_change'] = df['water_level_cm'].shift(-1) - df['water_level_cm']

    return df.dropna()


def build_anfis(num_inputs: int, num_mfs: int) -> AnfisNet:
    """Reconstructs ANFIS using 'x' prefix to match state_dict keys"""
    invardefs = []
    for i in range(num_inputs):
        mfs = [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(num_mfs)]
        invardefs.append((f'x{i}', mfs))
    return AnfisNet('Complex Flood Model', invardefs, ['y'], hybrid=True)


def evaluate_anfis_model():
    print("--- Step 1: Loading Models and Artifacts ---")
    with open(CONFIG_JSON_PATH, "r") as f:
        config = json.load(f)

    features_list = ['API_norm', 'S_t', 'SMI_t', 'Pt', 'delta_WL_t']

    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    anfis_model = build_anfis(len(features_list), config["num_mfs"])
    checkpoint = torch.load(ANFIS_MODEL_PATH, map_location="cpu")

    # Check for coefficient key naming
    coeff_key = 'consequent_coeffs' if 'consequent_coeffs' in checkpoint else 'coeff'

    anfis_model.load_state_dict(checkpoint['model_state_dict'])
    anfis_model.coeff = checkpoint[coeff_key]
    anfis_model.eval()
    print(f"✅ Model loaded using key: '{coeff_key}'")

    print("\n--- Step 2: Preparing Test Data ---")
    df_test_raw = pd.read_csv(TEST_DATA_FILE)
    df_test = prepare_complex_features(df_test_raw)

    X_test_scaled = scaler_X.transform(df_test[features_list].values)
    X_test_tensor = torch.from_numpy(X_test_scaled).float()

    print("\n--- Step 3: Predictions & Metrics ---")
    with torch.no_grad():
        preds_scaled = anfis_model(X_test_tensor).numpy().flatten()
        preds_change = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    actual_level = df_test['water_level_cm'].values + df_test['target_change'].values
    predicted_level = df_test['water_level_cm'].values + preds_change

    print(f"R-squared (R²): {r2_score(actual_level, predicted_level):.4f}")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(actual_level, predicted_level):.2f} cm")


if __name__ == "__main__":
    evaluate_anfis_model()