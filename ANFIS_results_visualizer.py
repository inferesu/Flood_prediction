import pandas as pd
import torch
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt

# Import the necessary classes from the anfis folder in your project
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# --- Configuration (Paths must match the training script) ---
TEST_DATA_FILE = 'new_test.csv'
MODEL_SAVE_PATH = "anfis_model.pth"
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_Y.pkl"
CONFIG_JSON_PATH = "training_config.json"
# We still need the training data to get the real-world range for plotting
TRAIN_DATA_FILE = 'minija_multi_precip_data_2020-2023.csv'


# --- Feature Engineering (Must be IDENTICAL to the training script) ---
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares features for the ANFIS model."""
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
    """Recreates the ANFIS model structure based on the saved config."""
    invardefs = []
    for i in range(num_inputs):
        mfs = [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(num_mfs)]
        invardefs.append((f'x{i}', mfs))
    return AnfisNet('Flood Prediction Model', invardefs, ['y'], hybrid=True)


def visualize_results():
    """Loads a saved model and creates plots for its results and internal state."""
    print("--- Step 1: Loading All Saved Artifacts and Data ---")

    # --- Load Config and Scalers ---
    with open(CONFIG_JSON_PATH, "r") as f:
        config = json.load(f)
    features_list = config["features_list"]
    num_features = len(features_list)

    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    print("Config and scalers loaded successfully.")

    # --- Load Training Data (needed for MF plots) ---
    df_train = pd.read_csv(TRAIN_DATA_FILE, parse_dates=['timestamp'], index_col='timestamp')
    df_train = prepare_features(df_train)
    print("Training data loaded for plotting ranges.")

    # --- Build an initial (untrained) model for comparison ---
    initial_model = build_anfis(num_inputs=config["num_inputs"], num_mfs=config["num_mfs"])
    initial_fuzzify_layer = initial_model.layer['fuzzify']
    print("Initial untrained model built for comparison.")

    # --- Rebuild and Load the Trained Model ---
    trained_model = build_anfis(num_inputs=config["num_inputs"], num_mfs=config["num_mfs"])
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location="cpu")
    trained_model.load_state_dict(checkpoint['model_state_dict'])
    trained_model.coeff = checkpoint['consequent_coeffs']
    trained_model.eval()
    trained_fuzzify_layer = trained_model.layer['fuzzify']
    print("Trained model rebuilt and weights loaded successfully.")

    # --- Load Test Data and Make Predictions with the Trained Model ---
    df_test = pd.read_csv(TEST_DATA_FILE, parse_dates=['timestamp'], index_col='timestamp')
    df_test = prepare_features(df_test)
    X_test = df_test[features_list].values
    X_test_scaled = scaler_X.transform(X_test)
    x_test_tensor = torch.from_numpy(X_test_scaled).float()
    print("Test data loaded and prepared.")

    with torch.no_grad():
        y_pred_scaled_tensor = trained_model(x_test_tensor)
    y_pred_scaled = y_pred_scaled_tensor.numpy()
    y_pred_change = scaler_y.inverse_transform(y_pred_scaled).flatten()
    base_water_level = df_test['water_level_cm'].values
    predicted_level = base_water_level + y_pred_change
    actual_level = df_test['target_change'].values + base_water_level
    print("Predictions generated with trained model.")

    # --- Plot 1: Time Series of Predictions vs. Actuals ---
    print("-> Generating predictions vs. actuals plot...")
    plt.figure(figsize=(15, 7))
    plt.title('Trained Model Predictions vs. Actual Water Levels', fontsize=16)
    plt.plot(df_test.index, actual_level, label='Actual Water Level', color='blue', marker='o', linestyle='-', markersize=4)
    plt.plot(df_test.index, predicted_level, label='Predicted Water Level', color='red', marker='x', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Water Level (cm)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Side-by-Side Comparison of Membership Functions ---
    print("\n--- Generating side-by-side MF comparisons for each feature ---")
    for i, var_name in enumerate(features_list):
        # Create a figure with two subplots, side-by-side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        fig.suptitle(f'Membership Function Comparison for: {var_name}', fontsize=16)

        # Prepare the x-axis values for plotting
        min_val, max_val = df_train[var_name].min(), df_train[var_name].max()
        x_values = torch.linspace(min_val, max_val, 1000)
        dummy_for_scaling = np.zeros((len(x_values), num_features))
        dummy_for_scaling[:, i] = x_values.numpy()
        x_values_scaled = torch.from_numpy(scaler_X.transform(dummy_for_scaling)[:, i]).float().unsqueeze(1)

        # --- Plot on the LEFT: Initial (Untrained) MFs ---
        fuzzify_variable_initial = initial_fuzzify_layer.varmfs[f'x{i}']
        for j, mf in enumerate(fuzzify_variable_initial.mfdefs.values()):
            y_values = mf(x_values_scaled)
            ax1.plot(x_values.numpy(), y_values.detach().numpy(), label=f'Fuzzy Set {j + 1}')
        ax1.set_title('Initial (Untrained) State')
        ax1.set_xlabel('Feature Value (original scale)')
        ax1.set_ylabel('Degree of Membership')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        # --- Plot on the RIGHT: Learned (Trained) MFs ---
        fuzzify_variable_trained = trained_fuzzify_layer.varmfs[f'x{i}']
        for j, mf in enumerate(fuzzify_variable_trained.mfdefs.values()):
            y_values = mf(x_values_scaled)
            ax2.plot(x_values.numpy(), y_values.detach().numpy(), label=f'Fuzzy Set {j + 1}')
        ax2.set_title('Learned (Trained) State')
        ax2.set_xlabel('Feature Value (original scale)')
        ax2.set_ylabel('Degree of Membership')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    print("All plots displayed.")


if __name__ == "__main__":
    visualize_results()
