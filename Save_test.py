# final_minimal_training_script.py
# This script uses the developer's official method to save a minimal model.

import numpy as np
import torch
from xanfis import Data, GdAnfisRegressor

# --- Configuration ---
MODEL_FILENAME = 'bug_report_model.pkl'  # Use .pkl as recommended
SCALER_FILE = 'bug_report_scalers.pth'


def create_and_save_minimal_model():
    """Creates and saves a minimal, reproducible model."""
    print("--- Step 1: Creating a simple, fake dataset ---")
    X_train = np.random.rand(100, 4)
    y_train = X_train[:, 0] + X_train[:, 1] + (np.random.rand(100) * 0.1)

    data = Data(X_train, y_train)
    data.X, scaler_X = data.scale(data.X, scaling_methods=("standard", "minmax"))
    data.y, scaler_y = data.scale(data.y.reshape(-1, 1), scaling_methods=("standard", "minmax"))
    data.y = data.y.flatten()
    print("Data created and scaled.")

    print("\n--- Step 2: Training a minimal model ---")
    model = GdAnfisRegressor(num_rules=5, mf_class="Trapezoidal", epochs=5, verbose=False)
    model.fit(X=data.X, y=data.y)
    print("Minimal model trained.")

    try:
        model.save_model(save_path=".", filename="bug_report_model.pkl")
    except Exception as e:
        print(f"An error occurred during saving: {e}")


if __name__ == "__main__":
    create_and_save_minimal_model()