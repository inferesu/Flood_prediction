import json
import time
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import requests
import torch

# Ensure these ANFIS classes are available in your project's path
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# --- Configuration ---
# MODIFIED: Changed file paths to match your original training script's output
MODEL_SAVE_PATH = "anfis_model.pth"
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_Y.pkl"
CONFIG_JSON_PATH = "training_config.json"

# API station codes
STATION_CODE_HYDRO = 'priekules-vms'
STATION_CODE_METEO_1 = 'klaipedos-ams'
STATION_CODE_METEO_2 = 'vezaiciu-ams'


# --- Core Functions (Must match training script) ---

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering. MUST BE IDENTICAL to the training script.
    MODIFIED: Reverted to the original precipitation-only features.
    """
    df = df.copy()
    df.sort_index(inplace=True)
    df.interpolate(method='time', inplace=True)

    # Feature engineering for Klaipeda
    df['precip_klaipedos_lag_12h'] = df['precip_klaipedos-ams_mm'].rolling(12, min_periods=1).sum()
    df['precip_klaipedos_lag_24h'] = df['precip_klaipedos-ams_mm'].rolling(24, min_periods=1).sum()
    df['precip_klaipedos_lag_48h'] = df['precip_klaipedos-ams_mm'].rolling(48, min_periods=1).sum()
    df['precip_klaipedos_lag_72h'] = df['precip_klaipedos-ams_mm'].rolling(72, min_periods=1).sum()

    # Feature engineering for Vezaiciai
    df['precip_vezaiciu_lag_12h'] = df['precip_vezaiciu-ams_mm'].rolling(12, min_periods=1).sum()
    df['precip_vezaiciu_lag_24h'] = df['precip_vezaiciu-ams_mm'].rolling(24, min_periods=1).sum()
    df['precip_vezaiciu_lag_48h'] = df['precip_vezaiciu-ams_mm'].rolling(48, min_periods=1).sum()
    df['precip_vezaiciu_lag_72h'] = df['precip_vezaiciu-ams_mm'].rolling(72, min_periods=1).sum()

    # NOTE: The 'target_change' column is not needed for live prediction
    return df


def build_anfis(num_inputs: int, num_mfs: int) -> AnfisNet:
    """Recreates the ANFIS model structure."""
    invardefs = []
    for i in range(num_inputs):
        mfs = [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(num_mfs)]
        invardefs.append((f'x{i}', mfs))
    return AnfisNet('Flood Prediction Model', invardefs, ['y'], hybrid=True)


# --- Live Data Fetching ---

def fetch_live_meteo_data(station_code, hours_needed=80):
    """Fetches the last N hours of meteorological data for a station."""
    print(f"-> Fetching last {hours_needed} hours of meteo data for {station_code}...")
    all_obs = []
    for i in range(int(hours_needed / 24) + 2):
        date_to_fetch = datetime.utcnow() - timedelta(days=i)
        date_str = date_to_fetch.strftime("%Y-%m-%d")
        url = f"https://api.meteo.lt/v1/stations/{station_code}/observations/{date_str}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                all_obs.extend(resp.json().get("observations", []))
        except Exception as e:
            print(f"   ⚠️  Could not fetch data for {date_str}: {e}")
        time.sleep(0.2)
    return all_obs


def fetch_live_water_level(station_code, hours_needed=80):
    """Fetches the last N hours of water level data."""
    print(f"-> Fetching last {hours_needed} hours of water level data for {station_code}...")
    all_obs = []
    for i in range(int(hours_needed / 24) + 2):
        date_to_fetch = datetime.utcnow() - timedelta(days=i)
        date_str = date_to_fetch.strftime("%Y-%m-%d")
        url = f"https://api.meteo.lt/v1/hydro-stations/{station_code}/observations/measured/{date_str}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                all_obs.extend(resp.json().get("observations", []))
        except Exception as e:
            print(f"   ⚠️  Could not fetch data for {date_str}: {e}")
        time.sleep(0.2)
    return all_obs


# --- Main Forecasting Logic ---

def run_forecast():
    """
    Main function to load the model, fetch live data, and generate a 24-hour forecast.
    """
    print("--- Step 1: Loading Forecaster Model and Artifacts ---")
    try:
        with open(CONFIG_JSON_PATH, "r") as f:
            config = json.load(f)
        features_list = config["features_list"]
        num_mfs = config["num_mfs"]
        num_inputs = config["num_inputs"]

        scaler_X = joblib.load(SCALER_X_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)

        model = build_anfis(num_inputs=num_inputs, num_mfs=num_mfs)
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.coeff = checkpoint['consequent_coeffs']
        model.eval()
        print("✅ Model and all artifacts loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ FATAL ERROR: Could not load a required file: {e}. Make sure all model files are present.")
        return

    print("\n--- Step 2: Fetching Live Data from API ---")
    meteo_data1 = fetch_live_meteo_data(STATION_CODE_METEO_1)
    meteo_data2 = fetch_live_meteo_data(STATION_CODE_METEO_2)
    water_data = fetch_live_water_level(STATION_CODE_HYDRO)

    if not all([meteo_data1, meteo_data2, water_data]):
        print("❌ FATAL ERROR: Failed to fetch sufficient live data from one or more sources. Aborting.")
        return

    # --- Step 3: Process and Combine Live Data ---
    df_meteo1 = pd.DataFrame(meteo_data1)[["observationTimeUtc", "precipitation"]].rename(
        columns={"precipitation": f"precip_{STATION_CODE_METEO_1}_mm"})
    df_meteo2 = pd.DataFrame(meteo_data2)[["observationTimeUtc", "precipitation"]].rename(
        columns={"precipitation": f"precip_{STATION_CODE_METEO_2}_mm"})
    df_water = pd.DataFrame(water_data)[["observationTimeUtc", "waterLevel"]].rename(
        columns={"waterLevel": "water_level_cm"})

    for df in [df_meteo1, df_meteo2, df_water]:
        df['timestamp'] = pd.to_datetime(df['observationTimeUtc'])
        df.set_index('timestamp', inplace=True)
        df.drop(columns=['observationTimeUtc'], inplace=True)

    df_live = pd.concat([df_water, df_meteo1, df_meteo2], axis=1).sort_index()
    df_live.drop_duplicates(inplace=True)
    df_live = df_live.interpolate(method='time').fillna(method='bfill')

    last_known_time = df_live.index[-1]
    last_known_level = df_live['water_level_cm'][-1]
    print(f"\n✅ Live data processed. Last known reading at {last_known_time}: {last_known_level:.2f} cm")

    print("\n--- Step 4: Generating 24-Hour Autoregressive Forecast ---")
    forecast_rows = []
    df_forecast = df_live.copy()

    for i in range(24):
        df_with_features = prepare_features(df_forecast)
        latest_features = df_with_features.iloc[-1][features_list].values.reshape(1, -1)

        latest_features_scaled = scaler_X.transform(latest_features)
        prediction_scaled = model(torch.from_numpy(latest_features_scaled).float())
        predicted_change = scaler_y.inverse_transform(prediction_scaled.detach().numpy()).flatten()[0]

        last_level = df_forecast.iloc[-1]['water_level_cm']
        new_level = last_level + predicted_change

        next_time = df_forecast.index[-1] + timedelta(hours=1)
        forecast_rows.append({'timestamp': next_time, 'predicted_level_cm': new_level})

        # Create the new row for the next forecast step
        new_row = df_forecast.iloc[-1:].copy()
        new_row.index = [next_time]
        new_row['water_level_cm'] = new_level

        # MODIFIED: Assume ZERO future precipitation for a more stable "runoff" forecast.
        # This prevents the model from running away with a single hour's weather.
        new_row[f'precip_{STATION_CODE_METEO_1}_mm'] = 0.0
        new_row[f'precip_{STATION_CODE_METEO_2}_mm'] = 0.0

        df_forecast = pd.concat([df_forecast, new_row])

    df_result = pd.DataFrame(forecast_rows)
    print("✅ Forecast generated successfully.")

    print("\n--- 🌊 MINIJOS RIVER - 24 HOUR WATER LEVEL FORECAST 🌊 ---")
    print(f"Forecast generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current Water Level ({last_known_time.strftime('%H:%M')}): {last_known_level:.2f} cm")
    print("-" * 55)
    print(df_result.to_string(index=False))
    print("-" * 55)


if __name__ == "__main__":
    run_forecast()