import pandas as pd
import torch
import numpy as np
import os
from xanfis import GdAnfisRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# --- Configuration ---
MODEL_FILE = 'anfis_model.json' # The model configuration and weights
SCALER_FILE = 'scalers.pth'     # The data scalers
TEST_DATA_FILE = 'quick_test_data.csv' # Your manually collected test data

def prepare_features(df):
    """Applies the same feature engineering to any dataframe."""
    df.sort_index(inplace=True)
    df.interpolate(method='time', inplace=True)
    df['precip_klaipedos_lag_12h'] = df['precip_klaipedos-ams_mm'].rolling(window=12, min_periods=1).sum()
    df['precip_klaipedos_lag_24h'] = df['precip_klaipedos-ams_mm'].rolling(window=24, min_periods=1).sum()
    df['precip_vezaiciu_lag_12h'] = df['precip_vezaiciu-ams_mm'].rolling(window=12, min_periods=1).sum()
    df['precip_vezaiciu_lag_24h'] = df['precip_vezaiciu-ams_mm'].rolling(window=24, min_periods=1).sum()
    df['target_water_level'] = df['water_level_cm'].shift(-1)
    df.dropna(inplace=True)
    return df

def evaluate_saved_model():
    """Loads a saved model and evaluates it on a test dataset."""
    print("--- Step 1: Loading Saved Model, Scalers, and Test Data ---")

    # --- Load Model and Scalers ---
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        print(f"❌ FATAL ERROR: Model or scaler file not found. Please run the training script first.")
        return
    try:
        # 1. Create a new, empty model object. This is a required step.
        model = GdAnfisRegressor()
        # 2. Use the library's dedicated function to load the saved parameters into it.
        model.load_model(MODEL_FILE)
        # 3. Load the scalers separately from their own file.
        scalers = torch.load(SCALER_FILE)
        scaler_X = scalers['scaler_X']
        scaler_y = scalers['scaler_y']
        print("✅ Model and scalers loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # --- Load Test Data ---
    if not os.path.exists(TEST_DATA_FILE):
        print(f"❌ FATAL ERROR: Test data file '{TEST_DATA_FILE}' not found.")
        return
    df_test = pd.read_csv(TEST_DATA_FILE, parse_dates=['timestamp'], index_col='timestamp')
    df_test = prepare_features(df_test)
    print(f"✅ Test data loaded and features prepared with {len(df_test)} rows.")

    if len(df_test) == 0:
        print("❌ Not enough test data to make a prediction (need at least 2 rows).")
        return

    # --- Step 2: Make Predictions ---
    features_list = [
        'water_level_cm', 'precip_klaipedos_lag_12h', 'precip_klaipedos_lag_24h',
        'precip_vezaiciu_lag_12h', 'precip_vezaiciu_lag_24h'
    ]
    X_test, y_true = df_test[features_list].values, df_test['target_water_level'].values
    X_test_scaled = scaler_X.transform(X_test)

    print("-> Making predictions on the scaled test data...")
    model.eval() # Put the model in evaluation mode
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    print("✅ Predictions generated and un-scaled.")

    # --- Step 3: Display Final Results ---
    print("\n--- FINAL EVALUATION RESULTS ---")
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n- R-squared (R²): {r2:.4f}")
    print(f"- Mean Absolute Error (MAE): {mae:.2f} cm")

    results_df = pd.DataFrame({
        'Timestamp': df_test.index,
        'Actual Water Level (cm)': y_true,
        'Predicted Water Level (cm)': y_pred
    })
    print("\n--- Predictions vs. Actuals ---")
    print(results_df.to_string(index=False))
    print("---------------------------------")

if __name__ == "__main__":
    evaluate_saved_model()
