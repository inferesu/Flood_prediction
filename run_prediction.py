import json
import os
import time
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
import requests
import torch
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# --- Configuration ---
DATA_FILE = 'live_data.csv'
ANFIS_MODEL_PATH = "anfis_model.pth"
SCALER_X_PATH, SCALER_Y_PATH = "scaler_X.pkl", "scaler_Y.pkl"
CONFIG_JSON_PATH = "training_config.json"
PREDICTIONS_LOG_PATH = "predictions_log.json"

STATION_CODE_HYDRO = 'priekules-vms'
STATION_CODE_METEO_1 = 'klaipedos-ams'
STATION_CODE_METEO_2 = 'vezaiciu-ams'


# --- Data Collection Logic ---
def fetch_recent_water_level(station_code, date):
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.meteo.lt/v1/hydro-stations/{station_code}/observations/measured/{date_str}"
    print(f"-> Fetching water level for {date_str}...")
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        water_levels = [obs['waterLevel'] for obs in observations if obs.get('waterLevel') is not None]
        if water_levels:
            return round(sum(water_levels) / len(water_levels), 2)
    except requests.RequestException as e:
        print(f"   - ❌ Error fetching water level: {e}")
    return None


def fetch_recent_precipitation(station_code, date):
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.meteo.lt/v1/stations/{station_code}/observations/{date_str}"
    print(f"-> Fetching precipitation for '{station_code}' on {date_str}...")
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        daily_precip = sum(obs.get('precipitation', 0) for obs in observations if obs.get('precipitation') is not None)
        return round(daily_precip, 2)
    except requests.RequestException as e:
        print(f"   - ❌ Error fetching precipitation: {e}")
    return 0


def update_data_file():
    yesterday = datetime.now().date() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    if os.path.exists(DATA_FILE):
        df_existing = pd.read_csv(DATA_FILE)
        if yesterday_str in df_existing['timestamp'].values:
            print(f"✅ Data for {yesterday_str} already exists. No update needed.")
            return
    else:
        print(f"File '{DATA_FILE}' not found. Creating it.")
        pd.DataFrame(
            columns=['timestamp', 'water_level_cm', 'precip_klaipedos-ams_mm', 'precip_vezaiciu-ams_mm']).to_csv(
            DATA_FILE, index=False)
    print(f"--- Collecting data for {yesterday_str} ---")
    water_level = fetch_recent_water_level(STATION_CODE_HYDRO, yesterday)
    precip1 = fetch_recent_precipitation(STATION_CODE_METEO_1, yesterday)
    time.sleep(0.5)
    precip2 = fetch_recent_precipitation(STATION_CODE_METEO_2, yesterday)
    if water_level is None:
        print(f"❌ Could not fetch water level for {yesterday_str}. Aborting update.")
        return
    new_data = pd.DataFrame(
        [{'timestamp': yesterday_str, 'water_level_cm': water_level, 'precip_klaipedos-ams_mm': precip1,
          'precip_vezaiciu-ams_mm': precip2}])
    new_data.to_csv(DATA_FILE, mode='a', header=False, index=False)
    print(f"✅ --- Successfully added data for {yesterday_str} to '{DATA_FILE}'. ---")


# --- Feature Engineering ---
def prepare_features(df: pd.DataFrame):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.drop_duplicates(subset='timestamp', keep='last')
    df = df.set_index('timestamp').asfreq('D').ffill()
    for station in ['klaipedos', 'vezaiciu']:
        df[f'precip_{station}_lag_12h'] = df[f'precip_{station}-ams_mm'].rolling(window=1, min_periods=1).sum()
        df[f'precip_{station}_lag_24h'] = df[f'precip_{station}-ams_mm'].rolling(window=1, min_periods=1).sum()
        df[f'precip_{station}_lag_48h'] = df[f'precip_{station}-ams_mm'].rolling(window=2, min_periods=1).sum()
        df[f'precip_{station}_lag_72h'] = df[f'precip_{station}-ams_mm'].rolling(window=3, min_periods=1).sum()
    return df


def build_anfis(num_inputs: int, num_mfs: int):
    invardefs = [(f'x{i}', [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(num_mfs)]) for i in
                 range(num_inputs)]
    return AnfisNet('Flood Prediction Model', invardefs, ['y'], hybrid=True)


# --- Main Prediction Function ---
def run_prediction_job():
    print(f"--- Running hourly prediction job at {datetime.now()} ---")
    update_data_file()

    with open(CONFIG_JSON_PATH, "r") as f:
        config = json.load(f)
    features_list = config["features_list"]

    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    anfis_model = build_anfis(config["num_inputs"], config["num_mfs"])
    checkpoint = torch.load(ANFIS_MODEL_PATH, map_location="cpu")
    anfis_model.load_state_dict(checkpoint['model_state_dict'])
    anfis_model.coeff = checkpoint['consequent_coeffs']
    anfis_model.eval()

    print("--- Fetching today's live data for prediction ---")
    today = datetime.now().date()
    live_water_level = fetch_recent_water_level(STATION_CODE_HYDRO, today)
    live_precip1 = fetch_recent_precipitation(STATION_CODE_METEO_1, today)
    live_precip2 = fetch_recent_precipitation(STATION_CODE_METEO_2, today)

    if live_water_level is None:
        print("❌ Could not fetch live water level data. Skipping prediction.")
        return

    df_hist = pd.read_csv(DATA_FILE)
    live_row = pd.DataFrame([{'timestamp': today.strftime("%Y-%m-%d"), 'water_level_cm': live_water_level,
                              'precip_klaipedos-ams_mm': live_precip1, 'precip_vezaiciu-ams_mm': live_precip2}])
    df_combined = pd.concat([df_hist, live_row], ignore_index=True)

    features_df = prepare_features(df_combined)

    last_row = features_df.iloc[-1:]
    last_known_level = last_row['water_level_cm'].iloc[0]

    X_pred_unscaled = last_row[features_list].values
    X_pred_scaled = scaler_X.transform(X_pred_unscaled)
    X_pred_tensor = torch.from_numpy(X_pred_scaled).float()

    with torch.no_grad():
        predicted_change = scaler_y.inverse_transform(anfis_model(X_pred_tensor).numpy())[0, 0]

    predicted_next_day_level = last_known_level + predicted_change

    today_str = today.strftime("%Y-%m-%d")
    tomorrow_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    log_data = {}
    if os.path.exists(PREDICTIONS_LOG_PATH):
        with open(PREDICTIONS_LOG_PATH, 'r') as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = {}

    log_data.setdefault(today_str, {})['actual'] = last_known_level
    log_data.setdefault(tomorrow_str, {})['predicted'] = predicted_next_day_level

    with open(PREDICTIONS_LOG_PATH, 'w') as f:
        json.dump(log_data, f, indent=4)

    print(f"✅ Prediction complete. Log file '{PREDICTIONS_LOG_PATH}' updated.")
    print(f"   -> Today's Actual: {last_known_level:.2f} cm")
    print(f"   -> Tomorrow's Forecast: {predicted_next_day_level:.2f} cm")


if __name__ == "__main__":
    run_prediction_job()