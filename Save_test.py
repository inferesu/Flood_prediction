import numpy as np
import torch
from xanfis import Data, GdAnfisRegressor

MODEL_FILE = 'bug_report_model.json'
SCALER_FILE = 'bug_report_scalers.pth'

def create_and_save_minimal_model():
    # Create 100 rows of fake data with 4 input features.
    X_train = np.random.rand(100, 4)
    # Create fake answers that are roughly the sum of the first two inputs.
    y_train = X_train[:, 0] + X_train[:, 1] + (np.random.rand(100) * 0.1)
    print("Data created.")

    data = Data(X_train, y_train)
    data.X, scaler_X = data.scale(data.X, scaling_methods=("standard", "minmax"))
    data.y, scaler_y = data.scale(data.y.reshape(-1, 1), scaling_methods=("standard", "minmax"))
    data.y = data.y.flatten()
    print("Data scaled.")

    print("\nTraining a minimal model for a few epochs")
    model = GdAnfisRegressor(num_rules=5, mf_class="Trapezoidal", epochs=5, verbose=True)
    model.fit(X=data.X, y=data.y)
    print("Minimal model trained.")

    print(f"\n--- Step 3: Attempting to save model to '{MODEL_FILE}' ---")
    try:
        model.save_model(MODEL_FILE)
        print(f"Model config saved to '{MODEL_FILE}'")

        torch.save({'scaler_X': scaler_X, 'scaler_y': scaler_y}, SCALER_FILE)
        print(f"Scalers saved to '{SCALER_FILE}'")
    except Exception as e:
        print(f"❌ An error occurred during saving: {e}")

if __name__ == "__main__":
    create_and_save_minimal_model()