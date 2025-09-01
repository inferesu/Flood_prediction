import pandas as pd
import torch
from xanfis import Data, GdAnfisRegressor
import os
from sklearn.metrics import r2_score, mean_absolute_error

# --- Configuration ---
TRAIN_DATA_FILE = 'minija_multi_precip_data_2020-2023.csv'
TEST_DATA_FILE = 'test_data_2024.csv' # Your manually collected test data

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

def run_full_evaluation():
    """Trains the model on the training set and evaluates it on the test set."""
    print("--- Step 1: Loading All Datasets ---")

    # --- Load Training Data ---
    if not os.path.exists(TRAIN_DATA_FILE):
        print(f"❌ FATAL ERROR: Training data file '{TRAIN_DATA_FILE}' not found.")
        return
    df_train = pd.read_csv(TRAIN_DATA_FILE, parse_dates=['timestamp'], index_col='timestamp')
    print(f"✅ Successfully loaded {len(df_train)} training rows.")

    # --- Load Test Data ---
    if not os.path.exists(TEST_DATA_FILE):
        print(f"❌ FATAL ERROR: Test data file '{TEST_DATA_FILE}' not found.")
        return
    df_test = pd.read_csv(TEST_DATA_FILE, parse_dates=['timestamp'], index_col='timestamp')
    print(f"✅ Successfully loaded {len(df_test)} test rows.")

    # --- Step 2: Prepare Features for Both Datasets ---
    print("\n--- Step 2: Preparing Features ---")
    df_train = prepare_features(df_train)
    df_test = prepare_features(df_test)

    if len(df_test) == 0:
        print("❌ Not enough test data to make a prediction (need at least 2 rows).")
        return

    features_list = [
        'water_level_cm', 'precip_klaipedos_lag_12h', 'precip_klaipedos_lag_24h',
        'precip_vezaiciu_lag_12h', 'precip_vezaiciu_lag_24h'
    ]
    target = 'target_water_level'

    X_train, y_train = df_train[features_list].values, df_train[target].values
    X_test, y_true = df_test[features_list].values, df_test[target].values

    # --- Step 3: Scale Data ---
    print("\n--- Step 3: Scaling Data ---")
    data_train = Data(X_train, y_train)
    data_train.X, scaler_X = data_train.scale(data_train.X, scaling_methods=("standard", "minmax"))
    data_train.y, scaler_y = data_train.scale(data_train.y.reshape(-1, 1), scaling_methods=("standard", "minmax"))
    data_train.y = data_train.y.flatten()

    # CRITICAL: Scale the test data using the scaler that was FIT on the training data
    X_test_scaled = scaler_X.transform(X_test)
    print("✅ Training and test data scaled correctly.")

    # --- Step 4: Train the Model ---
    print("\n--- Step 4: Training the ANFIS Model ---")
    model = GdAnfisRegressor(num_rules=50, mf_class="Gaussian", epochs=100, batch_size=32,
                             optim="Adam", optim_params={"lr": 0.001},
                             early_stopping=True, n_patience=10, verbose=True)
    model.fit(X=data_train.X, y=data_train.y)
    print("✅ Model training complete.")

    # --- Step 5: Evaluate the Model on the Test Set ---
    print("\n--- Step 5: Making Predictions on Unseen Test Data ---")
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    print("✅ Predictions generated.")

    # --- Step 6: Display Final Results ---
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
    run_full_evaluation()
