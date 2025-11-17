import json
import os
import time
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import requests
import torch
from flask import Flask, jsonify, render_template

# --- Import ANFIS classes from your project ---
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration ---
DATA_FILE = 'live_data.csv'
ANFIS_MODEL_PATH = "anfis_model.pth"
SCALER_X_PATH, SCALER_Y_PATH = "scaler_X.pkl", "scaler_Y.pkl"
CONFIG_JSON_PATH = "training_config.json"
PREDICTIONS_LOG_PATH = "predictions_log.json"

# --- Constants for Data Fetching ---
STATION_CODE_HYDRO = 'priekules-vms'
STATION_CODE_METEO_1 = 'klaipedos-ams'
STATION_CODE_METEO_2 = 'vezaiciu-ams'


# --- Data Collection Logic (No changes) ---
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


# --- Feature Engineering (No changes) ---
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
    invardefs = [(f'x{i}', [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(num_mfs)]) for i in range(num_inputs)]
    return AnfisNet('Flood Prediction Model', invardefs, ['y'], hybrid=True)


# --- Main Route ---
@app.route('/')
def index():
    return render_template('index.html')


# --- API Endpoint ---
@app.route('/api/predict')
def predict_api():
    update_data_file()

    with open(CONFIG_JSON_PATH, "r") as f: config = json.load(f)
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
        return jsonify({"error": "Could not fetch live water level data."}), 500

    df_hist = pd.read_csv(DATA_FILE)
    live_row = pd.DataFrame([{'timestamp': today.strftime("%Y-%m-%d"), 'water_level_cm': live_water_level, 'precip_klaipedos-ams_mm': live_precip1, 'precip_vezaiciu-ams_mm': live_precip2}])
    df_combined = pd.concat([df_hist, live_row], ignore_index=True)

    features_df = prepare_features(df_combined)

    last_row = features_df.iloc[-1:]
    last_known_level = last_row['water_level_cm'].iloc[0]

    X_pred_unscaled = last_row[features_list].values
    X_pred_scaled = scaler_X.transform(X_pred_unscaled)
    X_pred_tensor = torch.from_numpy(X_pred_scaled).float()

    with torch.no_grad():
        predicted_change_scaled = anfis_model(X_pred_tensor)
        predicted_change = scaler_y.inverse_transform(predicted_change_scaled.numpy())[0, 0]

    predicted_next_day_level = last_known_level + predicted_change

    # --- THIS IS THE CORRECTED LOGIC ---
    today_str = today.strftime("%Y-%m-%d")
    tomorrow_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    log_data = {}
    if os.path.exists(PREDICTIONS_LOG_PATH):
        with open(PREDICTIONS_LOG_PATH, 'r') as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = {}

    # Update today's 'actual' value with the latest fetched data.
    log_data.setdefault(today_str, {})['actual'] = last_known_level
    print(f"✅ Updated 'actual' level for {today_str} to: {last_known_level:.2f} cm")

    # --- FIX ---
    # ALWAYS update (or create) the prediction for tomorrow.
    # The 'if' statement that protected manual entries has been removed.
    log_data.setdefault(tomorrow_str, {})['predicted'] = predicted_next_day_level
    print(f"✅ Updated 'predicted' level for {tomorrow_str} to: {predicted_next_day_level:.2f} cm")

    with open(PREDICTIONS_LOG_PATH, 'w') as f:
        json.dump(log_data, f, indent=4)
    # --- END OF CORRECTED LOGIC ---

    def get_safe_float(series, key, default=0.0):
        val = series.get(key)
        return default if val.empty or pd.isna(val.iloc[0]) else float(val.iloc[0])

    live_features_dict = {
        "precip_klaipedos_lag_24h": get_safe_float(last_row, 'precip_klaipedos_lag_24h'),
        "precip_vezaiciu_lag_24h": get_safe_float(last_row, 'precip_vezaiciu_lag_24h'),
        "precip_klaipedos_lag_72h": get_safe_float(last_row, 'precip_klaipedos_lag_72h'),
        "precip_vezaiciu_lag_72h": get_safe_float(last_row, 'precip_vezaiciu_lag_72h'),
    }

    if predicted_next_day_level < 250: risk_level = 'LOW'
    elif predicted_next_day_level < 400: risk_level = 'MODERATE'
    elif predicted_next_day_level < 550: risk_level = 'HIGH'
    else: risk_level = 'SEVERE'

    trend_diff = predicted_next_day_level - last_known_level
    if trend_diff > 10: trend_text = 'Rising'
    elif trend_diff < -10: trend_text = 'Falling'
    else: trend_text = 'Stable'

    historical_data = [{'date': k, **v} for k, v in sorted(log_data.items())]

    return jsonify({
        "lastKnownLevel": last_known_level,
        "predictedNextDayLevel": predicted_next_day_level,
        "liveFeatures": live_features_dict,
        "historicalData": historical_data,
        "risk": {"level": risk_level},
        "trend": {"text": trend_text},
        "lastUpdated": datetime.now().isoformat()
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)