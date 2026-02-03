import json
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# ---------------- PATHS ----------------
ANFIS_MODEL_PATH = "anfis_model.pth"
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_y.pkl"
CONFIG_JSON_PATH = "training_config.json"
TEST_DATA_FILE = "minija_complex_data_test.csv"

# ---------------- CONSTANTS ----------------
K_DECAY = 0.85
T_MELT = 0.0

# ---------------- FEATURE ENGINEERING ----------------
def prepare_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Mean precipitation
    df["Pt"] = df[["precip_klaipedos-ams", "precip_vezaiciu-ams"]].mean(axis=1)

    # API
    api_vals = []
    val = 0.0
    for p in df["Pt"]:
        val = p + K_DECAY * val
        api_vals.append(val)
    df["API_norm"] = (api_vals - np.min(api_vals)) / (np.max(api_vals) - np.min(api_vals))

    # Seasonality
    doy = df["timestamp"].dt.dayofyear
    df["S_t"] = np.cos((2 * np.pi * doy) / 365.0)

    # Snowmelt Index
    T = df[["temp_klaipedos-ams", "temp_vezaiciu-ams"]].mean(axis=1)
    df["SMI_t"] = np.maximum(0, T - T_MELT)

    # Trend persistence
    df["delta_WL_t"] = df["water_level_cm"].diff().fillna(0)

    # Target
    df["target_change"] = df["water_level_cm"].shift(-1) - df["water_level_cm"]

    return df.dropna().reset_index(drop=True)


# ---------------- BUILD ANFIS ----------------
def build_anfis(n_inputs, n_mfs):
    invardefs = []
    for i in range(n_inputs):
        mfs = [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1))
               for _ in range(n_mfs)]
        invardefs.append((f"x{i}", mfs))
    return AnfisNet("Flood Model", invardefs, ["y"], hybrid=True)


# ---------------- EVALUATION ----------------
def evaluate():

    # Load config
    with open(CONFIG_JSON_PATH) as f:
        config = json.load(f)

    features = config["features_list"]
    num_inputs = config["num_inputs"]
    num_mfs = config["num_mfs"]
    target = config["target"]

    # Load scalers
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    # Load ANFIS
    model = build_anfis(num_inputs, num_mfs)
    ckpt = torch.load(ANFIS_MODEL_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.coeff = ckpt["coeff"]
    model.eval()

    # Load and prepare data
    df = prepare_features(pd.read_csv(TEST_DATA_FILE))
    X = df[features].values
    y_true = df[target].values

    # Scale inputs
    Xs = scaler_X.transform(X)
    preds_scaled = model(torch.tensor(Xs).float()).detach().numpy()
    preds = scaler_y.inverse_transform(preds_scaled).flatten()

    # Convert to water level
    wl = df["water_level_cm"].values
    wl_pred = wl + preds
    wl_true = wl + y_true

    # --------- METRICS ----------
    mse = mean_squared_error(wl_true, wl_pred)
    rmse = np.sqrt(mse)
    nrmse = rmse / (wl_true.max() - wl_true.min())
    mae = mean_absolute_error(wl_true, wl_pred)
    r2 = r2_score(wl_true, wl_pred)

    print("\n--- ANFIS Flood Model Performance ---")
    print(f"MSE   : {mse:.3f}")
    print(f"RMSE  : {rmse:.3f} cm")
    print(f"NRMSE : {nrmse:.4f}")
    print(f"MAE   : {mae:.3f} cm")
    print(f"R²    : {r2:.4f}")

    # Save results
    out = pd.DataFrame({
        "timestamp": df["timestamp"],
        "WL_true": wl_true,
        "WL_pred": wl_pred,
        "WL_error": wl_true - wl_pred
    })

    out.to_csv("anfis_predictions.csv", index=False)
    print("\nSaved to anfis_predictions.csv")
    print(out.head())


# ---------------- RUN ----------------
if __name__ == "__main__":
    evaluate()
