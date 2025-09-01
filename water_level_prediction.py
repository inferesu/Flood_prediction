import pandas as pd
import torch
import numpy as np
import os
from sklearn.metrics import r2_score, mean_absolute_error
# We need these imports so torch.load can reconstruct the saved objects
# MODIFIED: Corrected the import path for the internal CustomANFIS class.
from xanfis import GdAnfisRegressor, Data
from xanfis.core.anfis_model import CustomANFIS # This is the correct path to the internal model blueprint.

def evaluate_model():
    """Loads a model and evaluates it against a test data file."""
    model_file = 'anfis_model.pth'
    test_data_file = 'test_data_2024.csv' # Assumes you have created this file manually

    print("--- Step 1: Loading Model and Test Data ---")

    # --- Load the trained model and scalers ---
    if not os.path.exists(model_file):
        print(f"❌ FATAL ERROR: Model file '{model_file}' not found.")
        return
    try:
        saved_data = torch.load(model_file, weights_only=False)
        model = saved_data['model']
        scaler_X = saved_data['scaler_X']
        scaler_y = saved_data['scaler_y']
        print("✅ Model and scalers loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # --- Load and prepare the test data ---
    if not os.path.exists(test_data_file):
        print(f"❌ FATAL ERROR: Test data file '{test_data_file}' not found.")
        print("-> Please create this file with your manually collected 2024 data.")
        return

    df_test = pd.read_csv(test_data_file, parse_dates=['timestamp'], index_col='timestamp')
    print(f"✅ Successfully loaded '{test_data_file}' with {len(df_test)} rows.")

    # --- Step 2: Feature Engineering on Test Data ---
    # CRITICAL: We must perform the exact same feature engineering as on the training data.
    print("-> Applying feature engineering to the test data...")
    df_test.sort_index(inplace=True)
    df_test.interpolate(method='time', inplace=True)

    df_test['precip_klaipedos_lag_12h'] = df_test['precip_klaipedos-ams_mm'].rolling(window=12, min_periods=1).sum()
    df_test['precip_klaipedos_lag_24h'] = df_test['precip_klaipedos-ams_mm'].rolling(window=24, min_periods=1).sum()
    df_test['precip_vezaiciu_lag_12h'] = df_test['precip_vezaiciu-ams_mm'].rolling(window=12, min_periods=1).sum()
    df_test['precip_vezaiciu_lag_24h'] = df_test['precip_vezaiciu-ams_mm'].rolling(window=24, min_periods=1).sum()

    # Define the target (the actual, real values we want to compare against)
    df_test['target_water_level'] = df_test['water_level_cm'].shift(-6)
    df_test.dropna(inplace=True)
    print("✅ Test data features created.")

    # --- Step 3: Make Predictions on the Test Set ---
    features_list = [
        'water_level_cm',
        'precip_klaipedos_lag_12h',
        'precip_klaipedos_lag_24h',
        'precip_vezaiciu_lag_12h',
        'precip_vezaiciu_lag_24h'
    ]
    X_test = df_test[features_list].values
    y_true = df_test['target_water_level'].values

    # Scale the test features using the scaler from the training phase
    X_test_scaled = scaler_X.transform(X_test)

    print("-> Making predictions on the scaled test data...")
    y_pred_scaled = model.predict(X_test_scaled)

    # Inverse transform the scaled predictions to get the actual water level in cm
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    print("✅ Predictions generated and converted back to centimeters.")

    # --- Step 4: Evaluate and Display Results ---
    print("\n--- MODEL EVALUATION RESULTS ---")
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"\n- R-squared (R²): {r2:.4f}")
    print(f"- Mean Absolute Error (MAE): {mae:.2f} cm")
    print("\n(R² is a measure of how well the model's predictions match the real values. 1.0 is a perfect score.)")
    print("(MAE is the average difference between the predicted and actual water level, in cm.)")

    # --- Display a sample of predictions vs actual values ---
    results_df = pd.DataFrame({
        'Timestamp': df_test.index,
        'Actual Water Level (cm)': y_true,
        'Predicted Water Level (cm)': y_pred
    })
    print("\n--- Sample of Predictions vs. Actuals ---")
    print(results_df.head(10).to_string(index=False))
    print("----------------------------------------")

if __name__ == "__main__":
    evaluate_model()
