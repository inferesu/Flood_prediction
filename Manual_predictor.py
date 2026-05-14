import os
import json
import numpy as np
import torch
import joblib
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# --- GET CURRENT DIRECTORY ---
# This forces Python to look in the exact folder where this script is saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- PATHS & CONSTANTS ----------------
ANFIS_MODEL_PATH = os.path.join(BASE_DIR, "anfis_model.pth")
SCALER_X_PATH = os.path.join(BASE_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(BASE_DIR, "scaler_y.pkl")
CONFIG_JSON_PATH = os.path.join(BASE_DIR, "training_config.json")

FEATURES_LIST = ['API_norm', 'S_t', 'SMI_t', 'Pt', 'delta_WL_t']
FRIENDLY_NAMES = ["Soil Saturation (API_norm)", "Seasonality (S_t)", "Snowmelt (SMI_t)", "Precipitation (Pt)", "Trend (delta_WL_t)"]


# ---------------- HELPER FUNCTIONS ----------------
def build_anfis(n_inputs, n_mfs):
    """Rebuilds the ANFIS network architecture."""
    invardefs = []
    for i in range(n_inputs):
        mfs = [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(n_mfs)]
        invardefs.append((f"x{i}", mfs))
    return AnfisNet("Flood Model", invardefs, ["y"], hybrid=True)


def decode_rule_index(rule_index, num_mfs, num_inputs):
    """Converts a flat rule index back into the specific MF combination."""
    indices = []
    for _ in range(num_inputs):
        indices.append(rule_index % num_mfs)
        rule_index //= num_mfs
    return list(reversed(indices))


def get_linguistic_term(mf_index, num_mfs):
    """Translates a Membership Function (MF) index into a human-readable word."""
    if num_mfs == 5:
        return ["Very Low", "Low", "Moderate", "High", "Very High"][mf_index]
    elif num_mfs == 3:
        return ["Low", "Moderate", "High"][mf_index]
    return f"Level {mf_index + 1}"


# ---------------- MAIN APP ----------------
def run_interactive_console():
    print("======================================================")
    print("🌊 ANFIS Flood Prediction - Interactive Console 🌊")
    print("======================================================")
    print("Loading model and scalers...")

    # Load Config
    try:
        with open(CONFIG_JSON_PATH) as f:
            config = json.load(f)
        num_mfs = config.get("num_mfs", 5)
        num_inputs = len(FEATURES_LIST)
    except FileNotFoundError:
        print(f"⚠️ Warning: {CONFIG_JSON_PATH} not found. Assuming 5 MFs.")
        num_mfs = 5
        num_inputs = len(FEATURES_LIST)

    # Load Scalers
    try:
        scaler_X = joblib.load(SCALER_X_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
    except FileNotFoundError:
        print("❌ Error: Scaler files not found. Cannot proceed.")
        return

    # Load Model
    try:
        model = build_anfis(num_inputs, num_mfs)
        ckpt = torch.load(ANFIS_MODEL_PATH, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.coeff = ckpt.get("coeff", ckpt.get("consequent_coeffs"))
        model.eval()
        print("✅ System Ready.\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    while True:
        print("-" * 54)
        print("Please enter the current values for the 5 features.")
        print("Type 'q' to quit at any time.\n")

        user_inputs = []
        try:
            for feat in FRIENDLY_NAMES:
                val = input(f"Enter {feat}: ")
                if val.lower() == 'q':
                    print("Exiting...")
                    return
                user_inputs.append(float(val))
        except ValueError:
            print("⚠️ Invalid input. Please enter numbers only.\n")
            continue

        # Format inputs for model
        input_array = np.array(user_inputs).reshape(1, -1)

        # Scale inputs
        try:
            X_scaled = scaler_X.transform(input_array)
            X_tensor = torch.tensor(X_scaled).float()
        except Exception as e:
            print(f"⚠️ Scaling Error: {e}")
            continue

        # Run Prediction
        with torch.no_grad():
            # 1. Get raw prediction
            pred_chg_scaled = model(X_tensor).numpy()
            pred_chg = scaler_y.inverse_transform(pred_chg_scaled)[0, 0]

            # 2. Extract fuzzy rules
            fuzzified = model.layer['fuzzify'](X_tensor)
            firing_strengths = model.layer['rules'](fuzzified)[0].numpy()

            top_rule_idx = np.argmax(firing_strengths)
            activation_strength = firing_strengths[top_rule_idx]

            mf_indices = decode_rule_index(top_rule_idx, num_mfs, num_inputs)

        # Build Linguistic Rule String
        rule_parts = []
        for i, feat_name in enumerate(["API", "Seasonality", "Snowmelt", "Precip", "Trend"]):
            term = get_linguistic_term(mf_indices[i], num_mfs)
            rule_parts.append(f"{feat_name} is {term}")

        rule_text = "IF (" + ") AND (".join(rule_parts) + ")"

        # Display Results
        print("\n" + "=" * 54)
        print("📊 PREDICTION RESULTS")
        print("=" * 54)
        print(f"Predicted Water Level Change : {pred_chg:+.2f} cm")
        print("\n🧠 FUZZY LOGIC INFERENCE")
        print(f"Primary Fired Rule (Rule #{top_rule_idx}, Activation: {activation_strength:.4f}):")
        print(rule_text)
        print(f"THEN Change is approx {pred_chg:+.2f} cm")
        print("======================================================\n")


if __name__ == "__main__":
    run_interactive_console()