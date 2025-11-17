import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import r2_score, mean_absolute_error
from collections import deque

# --- Import ANFIS and RNN classes to be able to load the models ---
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# --- File Paths ---
ANFIS_MODEL_PATH = "anfis_model.pth"
RNN_MODEL_PATH = "rnn_corrector_multivariate.pth"
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_Y.pkl"
SCALER_ERROR_PATH = "scaler_error.pkl"
CONFIG_JSON_PATH = "training_config.json"
TEST_DATA_FILE = 'live_data.csv'

# --- RNN Configuration ---
SEQUENCE_LENGTH = 24
HIDDEN_SIZE = 60
NUM_LAYERS = 2

# --- Prediction Control ---
# REMOVED: The fixed dampening factor is no longer needed.
# CORRECTION_DAMPENING_FACTOR = 0.5
MAX_CORRECTION_ABS = 25.0

# --- Adaptive Correction Control ---
ADAPTIVE_ERROR_THRESHOLD = 15.0
ERROR_MONITORING_WINDOW = 12


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


class ErrorCorrectorRNN(nn.Module):
    """Reconstructs the new multivariate RNN model structure."""

    def __init__(self, input_size, hidden_size=50, num_layers=2, output_size=1):
        super(ErrorCorrectorRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def evaluate_hybrid_model_adaptive():
    """Main function to run the adaptive two-stage ANFIS+RNN evaluation."""
    print("--- Step 1: Loading All Models and Artifacts ---")

    with open(CONFIG_JSON_PATH, "r") as f:
        config = json.load(f)
    features_list = config["features_list"]
    num_features = len(features_list)

    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    anfis_model = build_anfis(config["num_inputs"], config["num_mfs"])
    checkpoint = torch.load(ANFIS_MODEL_PATH, map_location="cpu")
    anfis_model.load_state_dict(checkpoint['model_state_dict'])
    anfis_model.coeff = checkpoint['consequent_coeffs']
    anfis_model.eval()
    print("✅ ANFIS model and scalers loaded.")

    scaler_error = joblib.load(SCALER_ERROR_PATH)
    input_size = 1 + num_features
    rnn_model = ErrorCorrectorRNN(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    rnn_model.load_state_dict(torch.load(RNN_MODEL_PATH, map_location="cpu"))
    rnn_model.eval()
    print("✅ Multivariate RNN corrector model and scaler loaded.")

    print("\n--- Step 2: Preparing Test Data ---")
    df_test = pd.read_csv(TEST_DATA_FILE, parse_dates=['timestamp'], index_col='timestamp')
    df_test = prepare_features(df_test)
    X_test_unscaled = df_test[features_list].values
    y_true_change = df_test[config["target"]].values

    X_test_scaled = scaler_X.transform(X_test_unscaled)
    X_test_tensor = torch.from_numpy(X_test_scaled).float()

    print("\n--- Step 3: Performing Adaptive Hybrid Prediction Loop ---")
    final_predictions = []
    applied_corrections = []
    context_history = deque([np.zeros(1 + num_features)] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
    anfis_error_history = deque([0.0] * ERROR_MONITORING_WINDOW, maxlen=ERROR_MONITORING_WINDOW)

    with torch.no_grad():
        anfis_preds = scaler_y.inverse_transform(anfis_model(X_test_tensor).numpy()).flatten()

        for i in range(len(df_test)):
            anfis_prediction = anfis_preds[i]
            recent_anfis_mae = np.mean(np.abs(list(anfis_error_history)))
            final_correction = 0.0

            if recent_anfis_mae > ADAPTIVE_ERROR_THRESHOLD:
                context_np = np.array(context_history)
                context_to_scale = context_np.copy()
                context_to_scale[:, 0] = scaler_error.transform(context_to_scale[:, 0].reshape(-1, 1)).flatten()
                context_to_scale[:, 1:] = scaler_X.transform(context_to_scale[:, 1:])
                context_tensor = torch.from_numpy(context_to_scale).float().unsqueeze(0)

                correction_scaled = rnn_model(context_tensor).item()
                predicted_correction = scaler_error.inverse_transform([[correction_scaled]])[0, 0]

                # --- NEW: Adaptive Dampening Factor Logic ---
                # Define the range of the dampening factor
                min_damp = 0.3  # Be conservative when error is low
                max_damp = 0.8  # Be more aggressive when error is high
                # Define the error range over which to scale the factor
                # Start at the threshold, and cap at a high but reasonable error value
                error_scale_min = ADAPTIVE_ERROR_THRESHOLD
                error_scale_max = 75.0  # A high error, e.g., 75cm

                # Linearly scale the dampening factor based on the recent ANFIS error
                if recent_anfis_mae <= error_scale_min:
                    adaptive_dampening_factor = min_damp
                elif recent_anfis_mae >= error_scale_max:
                    adaptive_dampening_factor = max_damp
                else:
                    # Calculate the proportion of how far the error is into the scaling range
                    error_proportion = (recent_anfis_mae - error_scale_min) / (error_scale_max - error_scale_min)
                    adaptive_dampening_factor = min_damp + error_proportion * (max_damp - min_damp)

                dampened_correction = predicted_correction * adaptive_dampening_factor
                final_correction = np.clip(dampened_correction, -MAX_CORRECTION_ABS, MAX_CORRECTION_ABS)

            applied_corrections.append(final_correction)

            final_prediction = anfis_prediction + final_correction
            final_predictions.append(final_prediction)

            # Update histories
            true_final_error = y_true_change[i] - final_prediction
            current_features = X_test_unscaled[i]
            new_context = np.concatenate([[true_final_error], current_features])
            context_history.append(new_context)

            true_anfis_error = y_true_change[i] - anfis_prediction
            anfis_error_history.append(true_anfis_error)

    print("✅ Adaptive hybrid prediction loop complete.")
    final_predictions = np.array(final_predictions)

    print("\n--- Step 4: Final Evaluation ---")
    base_water_level = df_test['water_level_cm'].values
    predicted_level = base_water_level + final_predictions
    actual_level = base_water_level + y_true_change

    r2 = r2_score(actual_level, predicted_level)
    mae = mean_absolute_error(actual_level, predicted_level)
    print(f"R-squared (R²) for Adaptive Hybrid Model: {r2:.4f}")
    print(f"Mean Absolute Error (MAE) for Adaptive Hybrid Model: {mae:.2f} cm")

    results_df = pd.DataFrame({
        'Timestamp': df_test.index,
        'Actual Water Level (cm)': actual_level,
        'ANFIS Prediction (cm)': base_water_level + anfis_preds,
        'Applied Correction (cm)': applied_corrections,
        'Hybrid Final Prediction (cm)': predicted_level
    })
    print("\n--- Hybrid Predictions vs. Actuals ---")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    evaluate_hybrid_model_adaptive()
