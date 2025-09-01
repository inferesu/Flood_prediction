import pandas as pd
import torch
from xanfis import Data, GdAnfisRegressor
import os

# --- Configuration ---
DATA_FILE = 'minija_multi_precip_data_2020-2023.csv'
MODEL_FILE = 'anfis_model.pth' # The trained model and scalers will be saved here.

def load_and_prepare_data(filepath):
    """Loads the CSV, handles errors, and creates features for the model."""
    print(f"--- Step 1: Loading and Preparing Data from '{filepath}' ---")
    if not os.path.exists(filepath):
        print(f"❌ FATAL ERROR: Data file '{filepath}' not found.")
        print("-> Please make sure your data file is in the same folder as this script.")
        return None

    df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
    print(f"✅ Successfully loaded '{filepath}' with {len(df)} rows.")

    df.sort_index(inplace=True)
    df.interpolate(method='time', inplace=True)

    # --- Feature Engineering ---
    df['precip_klaipedos_lag_12h'] = df['precip_klaipedos-ams_mm'].rolling(window=12, min_periods=1).sum()
    df['precip_klaipedos_lag_24h'] = df['precip_klaipedos-ams_mm'].rolling(window=24, min_periods=1).sum()
    df['precip_vezaiciu_lag_12h'] = df['precip_vezaiciu-ams_mm'].rolling(window=12, min_periods=1).sum()
    df['precip_vezaiciu_lag_24h'] = df['precip_vezaiciu-ams_mm'].rolling(window=24, min_periods=1).sum()

    # --- Define the Prediction Target ---
    # CRITICAL CHANGE: Predicting 1 day ahead is more useful for this system.
    df['target_water_level'] = df['water_level_cm'].shift(-1)
    df.dropna(inplace=True)

    print("✅ Data cleaned and features created for a 1-day-ahead prediction.")
    return df

def train_model(df):
    """Trains the ANFIS model on the entire dataset using the correct xanfis pattern."""
    print("\n--- Step 2: Training the ANFIS Model on the Full Dataset ---")

    features_list = [
        'water_level_cm',
        'precip_klaipedos_lag_12h',
        'precip_klaipedos_lag_24h',
        'precip_vezaiciu_lag_12h',
        'precip_vezaiciu_lag_24h'
    ]
    target = 'target_water_level'
    X_full = df[features_list].values
    y_full = df[target].values

    print(f"-> Using the full dataset of {len(X_full)} samples for training.")

    # 1. Create the xanfis Data object.
    data = Data(X_full, y_full)

    # 2. **CRITICAL STEP:** Scale the data. This is essential for the model to learn.
    print("-> Scaling data to improve model performance...")
    data.X, scaler_X = data.scale(data.X, scaling_methods=("standard", "minmax"))
    # The target variable must be reshaped for the scaler
    data.y, scaler_y = data.scale(data.y.reshape(-1, 1), scaling_methods=("standard", "minmax"))
    data.y = data.y.flatten() # Flatten it back to the required shape
    print("✅ Data scaling complete.")

    # 3. Create the ANFIS model with an adjusted learning rate.
    model = GdAnfisRegressor(num_rules=50,
                             mf_class="Gaussian",
                             epochs=100,
                             batch_size=32,
                             optim="Adam",
                             optim_params={"lr": 0.001}, # Reduced learning rate for stability
                             early_stopping=True,
                             n_patience=10,
                             verbose=True)

    # 4. Train the model using the .fit() method.
    print("-> Starting model training...")
    model.fit(X=data.X, y=data.y)
    print("✅ Model training complete.")

    # 5. Save the model AND the scalers together. This is crucial for making future predictions.
    model_and_scalers = {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }
    torch.save(model_and_scalers, MODEL_FILE)
    print(f"💾 Model and scalers saved to '{MODEL_FILE}'")
    print("\n--- Process Finished ---")

def main():
    """Main function to run the experiment."""
    data_frame = load_and_prepare_data(DATA_FILE)
    if data_frame is not None:
        train_model(data_frame)

if __name__ == "__main__":
    main()