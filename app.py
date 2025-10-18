import json
import os
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import joblib
from flask import Flask, jsonify, render_template

# --- Import ANFIS and RNN classes ---
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration ---
DATA_FILE = '2025new_test.csv'
ANFIS_MODEL_PATH = "anfis_model.pth"
SCALER_X_PATH, SCALER_Y_PATH = "scaler_X.pkl", "scaler_Y.pkl"
CONFIG_JSON_PATH = "training_config.json"
STATION_CODE_HYDRO, STATION_CODE_METEO_1, STATION_CODE_METEO_2 = 'priekules-vms', 'klaipedos-ams', 'vezaiciu-ams'
NUM_DAYS_TO_FETCH = 5


# --- Data Fetching Logic (No changes) ---
def fetch_recent_water_level(station_code, date):
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.meteo.lt/v1/hydro-stations/{station_code}/observations/measured/{date_str}"
    print(f"-> Fetching water level for {date_str}...")
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            observations = resp.json().get("observations", [])
            levels = [obs['waterLevel'] for obs in observations if obs.get('waterLevel') is not None]
            if levels: return round(sum(levels) / len(levels), 2)
    except Exception as e:
        print(f"   - ❌ Error fetching water level: {e}")
    return None


def fetch_recent_precipitation(station_code, date):
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.meteo.lt/v1/stations/{station_code}/observations/{date_str}"
    print(f"-> Fetching precipitation for '{station_code}' on {date_str}...")
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            observations = resp.json().get("observations", [])
            precip = sum(obs.get('precipitation', 0) for obs in observations if obs.get('precipitation') is not None)
            return round(precip, 2)
    except Exception as e:
        print(f"   - ❌ Error fetching precipitation: {e}")
    return 0


def update_and_overwrite_data():
    print(f"--- Starting data collection, including today's partial data ---")
    all_days_data = []
    for days_ago in range(NUM_DAYS_TO_FETCH - 1, -1, -1):
        current_date = datetime.now().date() - timedelta(days=days_ago)
        water_level = fetch_recent_water_level(STATION_CODE_HYDRO, current_date)
        time.sleep(0.3)
        precip1 = fetch_recent_precipitation(STATION_CODE_METEO_1, current_date)
        time.sleep(0.3)
        precip2 = fetch_recent_precipitation(STATION_CODE_METEO_2, current_date)
        if water_level is not None:
            all_days_data.append({
                'timestamp': current_date.strftime("%Y-%m-%d"),
                'water_level_cm': water_level,
                'precip_klaipedos-ams_mm': precip1,
                'precip_vezaiciu-ams_mm': precip2
            })
    if all_days_data:
        pd.DataFrame(all_days_data).to_csv(DATA_FILE, index=False)
        print(f"\n✅ --- Data file '{DATA_FILE}' updated successfully. ---")


# --- Feature Engineering & Model Classes ---
def prepare_features(df: pd.DataFrame):
    df = df.copy()
    for station in ['klaipedos', 'vezaiciu']:
        for lag in [12, 24, 48, 72]:
            df[f'precip_{station}_lag_{lag}h'] = df[f'precip_{station}-ams_mm'].rolling(lag, min_periods=1).sum()
    return df


def build_anfis(num_inputs: int, num_mfs: int):
    invardefs = [(f'x{i}', [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(num_mfs)]) for i in
                 range(num_inputs)]
    return AnfisNet('Flood Prediction Model', invardefs, ['y'], hybrid=True)


# --- Main Route to Serve the Dashboard ---
@app.route('/')
def index():
    return render_template('index.html')


# --- Main API Endpoint ---
@app.route('/api/predict')
def predict_api():
    update_and_overwrite_data()

    with open(CONFIG_JSON_PATH, "r") as f:
        config = json.load(f)
    features_list = config["features_list"]
    scalers = {id: joblib.load(path) for id, path in [('X', SCALER_X_PATH), ('y', SCALER_Y_PATH)]}
    anfis_model = build_anfis(config["num_inputs"], config["num_mfs"])
    checkpoint = torch.load(ANFIS_MODEL_PATH, map_location="cpu")
    anfis_model.load_state_dict(checkpoint['model_state_dict'])
    anfis_model.coeff = checkpoint['consequent_coeffs']
    anfis_model.eval()

    df_raw = pd.read_csv(DATA_FILE, parse_dates=['timestamp'], index_col='timestamp')
    df_hourly = df_raw.asfreq('H').ffill()

    forecast_values = []
    window_size = 72 + 1
    sim_window = df_hourly.iloc[-window_size:].copy()

    for _ in range(24):
        features_df = prepare_features(sim_window)
        last_row_features = features_df.iloc[-1:]
        X_pred_unscaled = last_row_features[features_list].values
        X_pred_scaled = scalers['X'].transform(X_pred_unscaled)
        X_pred_tensor = torch.from_numpy(X_pred_scaled).float()
        with torch.no_grad():
            predicted_change = scalers['y'].inverse_transform(anfis_model(X_pred_tensor).numpy())[0, 0]

        last_known_row = sim_window.iloc[-1]
        next_level = last_known_row['water_level_cm'] + predicted_change
        forecast_values.append(next_level)

        next_timestamp = last_known_row.name + pd.Timedelta(hours=1)
        next_row_data = {
            'water_level_cm': next_level,
            'precip_klaipedos-ams_mm': last_known_row['precip_klaipedos-ams_mm'] * 0.95,
            'precip_vezaiciu-ams_mm': last_known_row['precip_vezaiciu-ams_mm'] * 0.95
        }
        sim_window.loc[next_timestamp] = next_row_data
        sim_window = sim_window.iloc[1:]

    final_features_df = prepare_features(df_hourly)
    last_known_level = df_hourly['water_level_cm'].iloc[-1]
    live_features_series = final_features_df.iloc[-1]

    def get_safe_float(series, key, default=0.0):
        val = series.get(key, default)
        return default if pd.isna(val) else float(val)

    # --- THE FIX: Corrected spelling from 'vezaiciai' to 'vezaiciu' ---
    live_features_dict = {
        "precip_klaipedos_lag_24h": get_safe_float(live_features_series, 'precip_klaipedos_lag_24h'),
        "precip_vezaiciu_lag_24h": get_safe_float(live_features_series, 'precip_vezaiciu_lag_24h'),  # Corrected key
        "precip_klaipedos_lag_72h": get_safe_float(live_features_series, 'precip_klaipedos_lag_72h'),
        "precip_vezaiciu_lag_72h": get_safe_float(live_features_series, 'precip_vezaiciu_lag_72h'),  # Corrected key
    }

    max_forecast_level = max(forecast_values) if forecast_values else last_known_level
    if max_forecast_level < 250:
        risk_level_str = 'LOW'
    elif max_forecast_level < 400:
        risk_level_str = 'MODERATE'
    elif max_forecast_level < 550:
        risk_level_str = 'HIGH'
    else:
        risk_level_str = 'SEVERE'

    trend_value = forecast_values[-1] - last_known_level
    if trend_value > 10:
        trend_text_str = 'Rising'
    elif trend_value < -10:
        trend_text_str = 'Falling'
    else:
        trend_text_str = 'Stable'

    response_data = {
        "lastKnownLevel": last_known_level,
        "liveFeatures": live_features_dict,
        "forecast": forecast_values,
        "risk": {"level": risk_level_str},
        "trend": {"text": trend_text_str},
        "lastUpdated": datetime.now().isoformat()
    }

    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True)