import json
import numpy as np
import pandas as pd
import torch
import joblib
import time
import psutil
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc
import scipy.stats

# ---------------- PATHS ----------------
# Updated paths to point to the 3D model artifacts
ANFIS_MODEL_PATH = "ANFIS_models/anfis_model_3d.pth"
SCALER_X_PATH = "Scalers/scaler_X_3d.pkl"
SCALER_Y_PATH = "Scalers/scaler_y_3d.pkl"
CONFIG_JSON_PATH = "Training_configs/training_config_3d.json"  # Make sure this JSON reflects the new target!
TEST_DATA_FILE = "minija_complex_data_test.csv"

# ---------------- CONSTANTS & RESOURCE TRACKING ----------------
K_DECAY = 0.85
T_MELT = 0.0

# Apple M1 Max estimated CPU power
CPU_POWER_W = 30

process = psutil.Process(os.getpid())


def get_memory_mb():
    return process.memory_info().rss / (1024 * 1024)


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

    # Target: Modified for 3 days ahead
    df["target_change_3d"] = df["water_level_cm"].shift(-3) - df["water_level_cm"]

    return df.dropna().reset_index(drop=True)


# ---------------- BUILD ANFIS ----------------
def build_anfis(n_inputs, n_mfs):
    invardefs = []
    for i in range(n_inputs):
        mfs = [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1))
               for _ in range(n_mfs)]
        invardefs.append((f"x{i}", mfs))
    # Updated model name to match the 3D training script
    return AnfisNet("Flood Model 3D", invardefs, ["y"], hybrid=True)


# ---------------- EVALUATION ----------------
def evaluate():
    # Load config
    with open(CONFIG_JSON_PATH) as f:
        config = json.load(f)

    features = config["features_list"]
    num_inputs = config["num_inputs"]
    num_mfs = config["num_mfs"]
    target = config["target"] # Should be "target_change_3d" in your JSON

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

    # ================= INFERENCE & RESOURCE TRACKING =================
    print("\n--- Running Inference ---")

    mem_before = get_memory_mb()
    start_pred = time.time()

    preds_scaled = model(torch.tensor(Xs).float()).detach().numpy()

    pred_time = time.time() - start_pred
    mem_after = get_memory_mb()

    energy_inference = CPU_POWER_W * pred_time

    print("\n--- Inference Resources ---")
    print(f"Inference Time: {pred_time:.6f} sec")
    print(f"Time per Sample: {(pred_time / len(Xs)) * 1000:.6f} ms")
    print(f"Memory Usage: {mem_after - mem_before:.2f} MB")
    print(f"Estimated Energy: {energy_inference:.4f} Joules")
    # =================================================================

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

    print("\n--- ANFIS 3D Flood Model Performance ---")
    print(f"MSE   : {mse:.3f}")
    print(f"RMSE  : {rmse:.3f} cm")
    print(f"NRMSE : {nrmse:.4f}")
    print(f"MAE   : {mae:.3f} cm")
    print(f"R²    : {r2:.4f}")

    # ---- NEW: DIEBOLD-MARIANO TEST BLOCK ----
    # Updated competitor file expectation for the 3D baseline
    COMPETITOR_FILE = "rnn_predictions_3d.csv"
    try:
        print("\n--- Statistical Significance (Diebold-Mariano Test) ---")
        comp_df = pd.read_csv(COMPETITOR_FILE)

        # Convert timestamps to datetime for perfect alignment
        comp_df['timestamp'] = pd.to_datetime(comp_df['timestamp'])

        # Create a temporary dataframe for ANFIS results
        anfis_temp_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            'wl_true_anfis': wl_true,
            'wl_pred_anfis': wl_pred
        })

        # Merge them together based on the exact same days
        aligned_df = pd.merge(anfis_temp_df, comp_df, on='timestamp', how='inner')

        actual_aligned = aligned_df['wl_true_anfis'].values
        anfis_aligned = aligned_df['wl_pred_anfis'].values
        comp_aligned = aligned_df['WL_pred'].values

        # Test MSE differences on the perfectly aligned arrays
        dm_stat_mse, p_val_mse = diebold_mariano_test(actual_aligned, anfis_aligned, comp_aligned, loss='mse')

        print(f"Aligned test samples : {len(actual_aligned)} (truncated to match competitor look-back)")
        print(f"DM Statistic (MSE)   : {dm_stat_mse:.4f}")
        print(f"P-value (MSE)        : {p_val_mse:.4f}")

        if p_val_mse > 0.05:
            print(
                "Result: The difference in MSE between ANFIS 3D and the competitor is NOT statistically significant (p > 0.05).")
        else:
            print("Result: The difference in MSE IS statistically significant (p <= 0.05).")

    except FileNotFoundError:
        print(f"Note: '{COMPETITOR_FILE}' not found. Skipping DM test.")

    # ---- MODEL INTERPRETATION SECTION ----
    extract_membership_functions(model, features)
    print_sample_rules(model, features, num_mfs, n_show=5)
    rank_rules_by_activation(model, Xs, num_mfs, features, top_k=10)
    save_rules_to_csv(model, features, num_mfs)
    rank_least_activated_rules(model, Xs, num_mfs, features, bottom_k=10)
    rule_activation_statistics(model, Xs)
    flood_event_rule_analysis(model, Xs, wl_true,
                              num_mfs, features,
                              threshold_percentile=90,
                              top_k=5)
    check_membership_spread(model)
    feature_importance_via_coefficients(model, Xs, features)
    membership_overlap_index(model)
    rule_usage_entropy(model, Xs)

    # Save results
    out = pd.DataFrame({
        "timestamp": df["timestamp"],
        "WL_true": wl_true,
        "WL_pred": wl_pred,
        "WL_error": wl_true - wl_pred
    })

    # Output file adjusted for 3D predictions
    out.to_csv("anfis_predictions_3d.csv", index=False)
    print("\nSaved to anfis_predictions_3d.csv")
    print(out.head())


def decode_rule_index(rule_index, num_mfs, num_inputs):
    indices = []
    for _ in range(num_inputs):
        indices.append(rule_index % num_mfs)
        rule_index //= num_mfs
    return list(reversed(indices))


def extract_membership_functions(model, feature_names):
    print("\n=== MEMBERSHIP FUNCTIONS ===")

    fuzzify_layer = model.layer['fuzzify']
    variables = fuzzify_layer.varmfs  # OrderedDict

    for i, (var_name, var_obj) in enumerate(variables.items()):
        print(f"\nInput: {feature_names[i]}")

        for j, mf in enumerate(var_obj.mfdefs.values()):
            print(f"  MF_{j}: "
                  f"a={mf.a.item():.4f}, "
                  f"b={mf.b.item():.4f}, "
                  f"c={mf.c.item():.4f}")


def print_sample_rules(model, feature_names, num_mfs, n_show=5):
    print("\n=== SAMPLE FUZZY RULES ===")

    coeffs = model.coeff.detach().numpy()
    n_inputs = len(feature_names)
    n_rules = coeffs.shape[0]

    for r in range(min(n_show, n_rules)):
        mf_indices = decode_rule_index(r, num_mfs, n_inputs)

        print(f"\nRule {r}:")
        print("IF")

        for i, feature in enumerate(feature_names):
            print(f"   {feature} is MF_{mf_indices[i]}")

        print("THEN")

        # --- CASE 1: Zero-order Sugeno ---
        if coeffs.ndim == 2 and coeffs.shape[1] == 1:
            bias = coeffs[r, 0]
            print(f"   y = {bias:.4f}")

        # --- CASE 2: First-order Sugeno ---
        elif coeffs.ndim == 2:
            terms = []
            for i, feature in enumerate(feature_names):
                coef_value = coeffs[r, i]
                terms.append(f"{coef_value:.3f}*{feature}")

            bias = coeffs[r, -1]
            print("   y = " + " + ".join(terms) + f" + {bias:.3f}")

        else:
            print("Unexpected coefficient shape:", coeffs.shape)


def rank_rules_by_activation(model, X_scaled, num_mfs, feature_names, top_k=10):
    print("\n=== MOST ACTIVATED RULES ===")

    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled).float()
        fuzzified = model.layer['fuzzify'](X_tensor)
        firing_strengths = model.layer['rules'](fuzzified)

    avg_activation = firing_strengths.mean(dim=0).numpy()
    top_indices = np.argsort(avg_activation)[-top_k:][::-1]

    n_inputs = len(feature_names)

    for rank, rule_id in enumerate(top_indices):
        mf_indices = decode_rule_index(rule_id, num_mfs, n_inputs)

        print(f"\nRank {rank + 1} — Rule {rule_id} "
              f"(Avg activation={avg_activation[rule_id]:.6f})")

        for i, feature in enumerate(feature_names):
            print(f"   {feature} is MF_{mf_indices[i]}")


def save_rules_to_csv(model, feature_names, num_mfs):
    coeffs = model.coeff.detach().numpy()
    n_rules = coeffs.shape[0]
    n_inputs = len(feature_names)

    rows = []

    for r in range(n_rules):
        mf_indices = decode_rule_index(r, num_mfs, n_inputs)
        row = {"rule_id": r}

        for i, feature in enumerate(feature_names):
            row[f"{feature}_MF"] = mf_indices[i]

        row["rule_output_constant"] = coeffs[r, 0]
        rows.append(row)

    # Output file adjusted for 3D rules
    pd.DataFrame(rows).to_csv("anfis_rule_base_3d.csv", index=False)
    print("\nFull rule base saved to anfis_rule_base_3d.csv")


def rank_least_activated_rules(model, X_scaled, num_mfs, feature_names, bottom_k=10):
    print("\n=== LEAST ACTIVATED RULES ===")

    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled).float()
        fuzzified = model.layer['fuzzify'](X_tensor)
        firing_strengths = model.layer['rules'](fuzzified)

    avg_activation = firing_strengths.mean(dim=0).numpy()
    bottom_indices = np.argsort(avg_activation)[:bottom_k]

    n_inputs = len(feature_names)

    for rank, rule_id in enumerate(bottom_indices):
        mf_indices = decode_rule_index(rule_id, num_mfs, n_inputs)

        print(f"\nRank {rank + 1} — Rule {rule_id} "
              f"(Avg activation={avg_activation[rule_id]:.8f})")

        for i, feature in enumerate(feature_names):
            print(f"   {feature} is MF_{mf_indices[i]}")


def rule_activation_statistics(model, X_scaled):
    print("\n=== RULE ACTIVATION STATISTICS ===")

    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled).float()
        fuzzified = model.layer['fuzzify'](X_tensor)
        firing_strengths = model.layer['rules'](fuzzified)

    avg_activation = firing_strengths.mean(dim=0).numpy()

    print(f"Total rules: {len(avg_activation)}")
    print(f"Max activation: {avg_activation.max():.6f}")
    print(f"Mean activation: {avg_activation.mean():.6f}")
    print(f"Median activation: {np.median(avg_activation):.6f}")

    dead_rules = np.sum(avg_activation < 1e-4)
    print(f"Near-zero activation rules (<1e-4): {dead_rules}")


def rank_rules_by_contribution(model, X_scaled, num_mfs, feature_names, top_k=10):
    print("\n=== MOST INFLUENTIAL RULES (Activation × Output) ===")

    coeffs = model.coeff.detach().numpy()

    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled).float()
        fuzzified = model.layer['fuzzify'](X_tensor)
        firing_strengths = model.layer['rules'](fuzzified)

    avg_activation = firing_strengths.mean(dim=0).numpy()
    rule_outputs = coeffs[:, 0]

    contribution = np.abs(avg_activation * rule_outputs)
    top_indices = np.argsort(contribution)[-top_k:][::-1]

    n_inputs = len(feature_names)

    for rank, rule_id in enumerate(top_indices):
        mf_indices = decode_rule_index(rule_id, num_mfs, n_inputs)

        print(f"\nRank {rank + 1} — Rule {rule_id}")
        print(f"Contribution score: {contribution[rule_id]:.6f}")
        print(f"Output constant: {rule_outputs[rule_id]:.4f}")

        for i, feature in enumerate(feature_names):
            print(f"   {feature} is MF_{mf_indices[i]}")


def flood_event_rule_analysis(model, X_scaled, wl_true,
                              num_mfs, feature_names,
                              threshold_percentile=90,
                              top_k=5):
    print("\n=== FLOOD EVENT RULE ANALYSIS ===")

    threshold = np.percentile(wl_true, threshold_percentile)
    flood_mask = wl_true >= threshold

    print(f"Flood threshold (>{threshold_percentile}th percentile): {threshold:.2f} cm")
    print(f"Flood samples: {np.sum(flood_mask)}")

    if np.sum(flood_mask) == 0:
        print("No flood samples found.")
        return

    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled[flood_mask]).float()
        fuzzified = model.layer['fuzzify'](X_tensor)
        firing_strengths = model.layer['rules'](fuzzified)

    avg_activation = firing_strengths.mean(dim=0).numpy()
    top_indices = np.argsort(avg_activation)[-top_k:][::-1]

    n_inputs = len(feature_names)

    print("\nTop rules during floods:\n")

    for rank, rule_id in enumerate(top_indices):
        mf_indices = decode_rule_index(rule_id, num_mfs, n_inputs)

        print(f"Rank {rank + 1} — Rule {rule_id}")
        print(f"Avg flood activation: {avg_activation[rule_id]:.6f}")

        for i, feature in enumerate(feature_names):
            print(f"   {feature} is MF_{mf_indices[i]}")


def check_membership_spread(model):
    print("\n=== MEMBERSHIP FUNCTION SPREAD CHECK ===")

    fuzzify_layer = model.layer['fuzzify']
    variables = fuzzify_layer.varmfs

    for var_name, var_obj in variables.items():
        centers = [mf.c.item() for mf in var_obj.mfdefs.values()]
        spread = max(centers) - min(centers)

        print(f"{var_name}: center spread = {spread:.4f}")


def feature_importance_via_coefficients(model, X_scaled, feature_names):
    print("\n=== FEATURE IMPORTANCE (Activation × Coefficient) ===")

    coeffs = model.coeff.detach().numpy()
    print("Coefficient tensor shape:", coeffs.shape)

    if coeffs.ndim == 3:
        coeffs = coeffs[:, 0, :]

    n_rules, n_coeffs = coeffs.shape
    n_inputs = len(feature_names)

    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled).float()
        fuzzified = model.layer['fuzzify'](X_tensor)
        firing_strengths = model.layer['rules'](fuzzified)

    avg_activation = firing_strengths.mean(dim=0).numpy()
    importance = np.zeros(n_inputs)

    for r in range(n_rules):
        for i in range(n_inputs):
            importance[i] += abs(avg_activation[r] * coeffs[r, i])

    importance = importance / importance.sum()

    for i, feature in enumerate(feature_names):
        print(f"{feature}: {importance[i]:.4f}")


def membership_overlap_index(model, resolution=200):
    print("\n=== MEMBERSHIP OVERLAP INDEX ===")

    fuzzify_layer = model.layer['fuzzify']
    variables = fuzzify_layer.varmfs

    x_grid = np.linspace(0, 1, resolution)

    for var_name, var_obj in variables.items():
        mfs = list(var_obj.mfdefs.values())
        overlaps = []

        for i in range(len(mfs) - 1):
            mu1 = np.array([mfs[i](torch.tensor([x])).item() for x in x_grid])
            mu2 = np.array([mfs[i + 1](torch.tensor([x])).item() for x in x_grid])

            overlap = np.trapz(np.minimum(mu1, mu2), x_grid)
            overlaps.append(overlap)

        avg_overlap = np.mean(overlaps)
        print(f"{var_name}: average adjacent overlap = {avg_overlap:.4f}")


def rule_usage_entropy(model, X_scaled):
    print("\n=== RULE USAGE ENTROPY ===")

    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled).float()
        fuzzified = model.layer['fuzzify'](X_tensor)
        firing_strengths = model.layer['rules'](fuzzified)

    avg_activation = firing_strengths.mean(dim=0).numpy()
    total = avg_activation.sum()

    if total == 0:
        print("No rule activation detected.")
        return

    p = avg_activation / total
    p = p[p > 0]

    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(len(avg_activation))

    normalized_entropy = entropy / max_entropy

    print(f"Entropy: {entropy:.4f}")
    print(f"Normalized entropy: {normalized_entropy:.4f}")


# ---------------- STATISTICAL TESTING ----------------
def diebold_mariano_test(actual, pred1, pred2, loss='mse'):
    e1 = actual - pred1
    e2 = actual - pred2

    if loss == 'mse':
        d = (e1 ** 2) - (e2 ** 2)
    elif loss == 'mae':
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("Loss must be 'mse' or 'mae'")

    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1) / len(d)

    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(dm_stat)))

    return dm_stat, p_value


# ---------------- RUN ----------------
if __name__ == "__main__":
    evaluate()