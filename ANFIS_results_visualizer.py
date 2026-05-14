import pandas as pd
import torch
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mpl_toolkits.mplot3d import Axes3D

from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# ---------------- PATHS ----------------
TEST_DATA_FILE = "minija_complex_data_test.csv"
TRAIN_DATA_FILE = "minija_complex_data_2024.csv"
MODEL_SAVE_PATH = "ANFIS_models/anfis_model.pth"
SCALER_X_PATH = "Scalers/scaler_X.pkl"
SCALER_Y_PATH = "scaler_y.pkl"
CONFIG_JSON_PATH = "Training_configs/training_config.json"

K_DECAY = 0.85

# ---------------- FEATURES ----------------
def prepare_complex_features(df):
    df = df.copy().sort_index()

    df['Pt'] = df[['precip_klaipedos-ams', 'precip_vezaiciu-ams']].mean(axis=1)

    api, curr = [], 0
    for p in df['Pt']:
        curr = p + K_DECAY * curr
        api.append(curr)
    df['API_t'] = api
    df['API_norm'] = (df['API_t'] - df['API_t'].min()) / (df['API_t'].max() - df['API_t'].min())

    d = pd.to_datetime(df['timestamp']).dt.dayofyear
    df['S_t'] = np.cos(2 * np.pi * d / 365)

    avg_t = df[['temp_klaipedos-ams', 'temp_vezaiciu-ams']].mean(axis=1)
    df['SMI_t'] = avg_t.apply(lambda x: max(0, x * 2.5) if x > 0 else 0)

    df['delta_WL_t'] = df['water_level_cm'].diff().fillna(0)
    df['target_change'] = df['water_level_cm'].shift(-1) - df['water_level_cm']

    return df.dropna()

# ---------------- MODEL ----------------
def build_anfis(num_inputs, num_mfs):
    invardefs = []
    for i in range(num_inputs):
        mfs = [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(num_mfs)]
        invardefs.append((f'x{i}', mfs))
    return AnfisNet("Flood Model", invardefs, ["y"], hybrid=True)

# ---------------- 3D SURFACE ----------------
def plot_surface(model, scaler_X, feature_names, f1, f2):
    idx1, idx2 = feature_names.index(f1), feature_names.index(f2)
    grid = np.linspace(0, 1, 30)
    Xg, Yg = np.meshgrid(grid, grid)

    Z = []
    for i in range(len(grid)):
        for j in range(len(grid)):
            row = np.zeros((1, len(feature_names)))
            row[0, idx1] = Xg[i, j]
            row[0, idx2] = Yg[i, j]
            Z.append(row)

    Z = np.vstack(Z)
    with torch.no_grad():
        pred = model(torch.tensor(Z).float()).numpy()

    Zp = pred.reshape(Xg.shape)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xg, Yg, Zp, cmap="viridis")
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_zlabel("Δ Water Level")
    plt.title("ANFIS Rule Surface")
    plt.show()

# ---------------- MAIN ----------------
def main():
    with open(CONFIG_JSON_PATH) as f:
        cfg = json.load(f)

    FEATURES = cfg["features_list"]

    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    model = build_anfis(cfg["num_inputs"], cfg["num_mfs"])
    ckpt = torch.load(MODEL_SAVE_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.coeff = ckpt["coeff"]
    model.eval()

    df = prepare_complex_features(pd.read_csv(TEST_DATA_FILE))
    X = scaler_X.transform(df[FEATURES])
    delta_true = df["target_change"].values
    wl_base = df["water_level_cm"].values

    with torch.no_grad():
        y_scaled = model(torch.tensor(X).float()).numpy()

    delta_pred = scaler_y.inverse_transform(y_scaled).flatten()

    # --- RECONSTRUCT ABSOLUTE WATER LEVEL ---
    y_true = wl_base + delta_true
    y_pred = wl_base + delta_pred

    # -------- METRICS --------
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    nrmse = rmse / (y_true.max() - y_true.min())
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"MSE={mse:.4f}, RMSE={rmse:.4f}, NRMSE={nrmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    # -------- PLOTS --------
    plt.figure(figsize=(12,5))
    plt.plot(y_true, label="Observed")
    plt.plot(y_pred, label="Predicted")
    plt.legend(); plt.title("Water Level Change"); plt.show()

    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("Observed"); plt.ylabel("Predicted"); plt.show()

    plt.hist(y_true - y_pred, bins=30)
    plt.title("Residual Distribution")
    plt.show()

    # -------- MEMBERSHIP --------
    fuzz = model.layer["fuzzify"]
    x = torch.linspace(0,1,400).unsqueeze(1)
    for mf in fuzz.varmfs["x0"].mfdefs.values():
        plt.plot(x.numpy(), mf(x).detach().numpy())
    plt.title("API_norm Memberships")
    plt.show()

    # -------- 3D RULE SURFACE --------
    plot_surface(model, scaler_X, FEATURES, "API_norm", "Pt")

    # -------- ABLATION: remove SMI --------
    X_no_smi = X.copy()
    smi_idx = FEATURES.index("SMI_t")
    X_no_smi[:, smi_idx] = 0

    with torch.no_grad():
        y_nosmi = scaler_y.inverse_transform(
            model(torch.tensor(X_no_smi).float()).numpy()
        ).flatten()

    plt.figure()
    plt.plot(y_pred, label="With SMI")
    plt.plot(y_nosmi, label="Without SMI")
    plt.legend()
    plt.title("SMI Ablation Effect")
    plt.show()

if __name__ == "__main__":
    main()
