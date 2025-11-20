import json
import os
import time
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import requests
import torch
from flask import Flask, jsonify, render_template, request

# --- Import ANFIS classes from your project ---
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# --- Flask App Initialization ---
app = Flask(__name__)

# --- -------------------------- ---
# --- MULTI-RIVER CONFIGURATION ---
# --- -------------------------- ---

MINIJA_CONFIG = {
    "name": "minija",
    "display_name": "Minija (Priekulė)",
    "data_file": "live_data.csv",
    "anfis_model_path": "anfis_model.pth",
    "scaler_x_path": "scaler_X.pkl",
    "scaler_y_path": "scaler_Y.pkl",
    "config_json_path": "training_config.json",
    "predictions_log_path": "predictions_log.json",
    "hydro_station": "priekules-vms",
    "meteo_stations_codes": ["klaipedos-ams", "vezaiciu-ams"],
    "meteo_stations_short": ["klaipedos", "vezaiciu"],  # For prepare_features
    "risk_levels": [250, 400, 550]  # [LOW, MODERATE, HIGH, SEVERE]
}

DANE_CONFIG = {
    "name": "dane",
    "display_name": "Danė (Klaipėda)",
    "data_file": "live_data_dane.csv",
    "anfis_model_path": "anfis_model_dane.pth",
    "scaler_x_path": "scaler_X_dane.pkl",
    "scaler_y_path": "scaler_Y_dane.pkl",
    "config_json_path": "training_config_dane.json",
    "predictions_log_path": "predictions_log_dane.json",
    "hydro_station": "klaipedos-vms",  # ASSUMPTION: 'klaipedos-vms' is the station code. Change if needed.
    "meteo_stations_codes": ["klaipedos-ams"],
    "meteo_stations_short": ["klaipedos"],  # For prepare_features
    "risk_levels": [150, 250, 350]  # EXAMPLE: [LOW, MODERATE, HIGH, SEVERE] - Adjust as needed
}

# Dictionary to easily access configs
RIVER_CONFIGS = {
    "minija": MINIJA_CONFIG,
    "dane": DANE_CONFIG
}


# --- Data Collection Logic (Generic) ---

def fetch_recent_water_level(station_code, date):
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.meteo.lt/v1/hydro-stations/{station_code}/observations/measured/{date_str}"
    print(f"-> Fetching water level for '{station_code}' on {date_str}...")
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


def update_data_file(config: dict):
    """
    Fetches and updates the live data CSV for a specific river configuration.
    """
    yesterday = datetime.now().date() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    data_file = config['data_file']
    hydro_station = config['hydro_station']
    meteo_station_codes = config['meteo_stations_codes']

    # Check if data for yesterday already exists
    if os.path.exists(data_file):
        df_existing = pd.read_csv(data_file)
        if yesterday_str in df_existing['timestamp'].values:
            print(f"✅ Data for {config['name']} ({yesterday_str}) already exists in '{data_file}'. No update needed.")
            return
    else:
        print(f"File '{data_file}' not found. Creating it.")
        columns = ['timestamp', 'water_level_cm'] + [f'precip_{code}_mm' for code in meteo_station_codes]
        pd.DataFrame(columns=columns).to_csv(data_file, index=False)

    print(f"--- Collecting data for {config['name']} ({yesterday_str}) ---")

    water_level = fetch_recent_water_level(hydro_station, yesterday)

    precip_data = {}
    for code in meteo_station_codes:
        precip_data[f'precip_{code}_mm'] = fetch_recent_precipitation(code, yesterday)
        time.sleep(0.5)  # Avoid rate-limiting

    if water_level is None:
        print(f"❌ Could not fetch water level for {config['name']}. Aborting update for this river.")
        return

    new_data_dict = {
        'timestamp': yesterday_str,
        'water_level_cm': water_level,
        **precip_data
    }

    new_data = pd.DataFrame([new_data_dict])
    new_data.to_csv(data_file, mode='a', header=False, index=False)
    print(f"✅ --- Successfully added data for {config['name']} to '{data_file}'. ---")


# --- Feature Engineering (Generic) ---

def prepare_features(df: pd.DataFrame, config: dict):
    """
    Prepares features based on the river configuration.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.drop_duplicates(subset='timestamp', keep='last')
    df = df.set_index('timestamp').asfreq('D').ffill()

    # Use 'meteo_stations_short' for feature generation
    for station in config['meteo_stations_short']:
        col_name = f'precip_{station}-ams_mm'  # Assumes column name matches this pattern
        if col_name in df.columns:
            df[f'precip_{station}_lag_12h'] = df[col_name].rolling(window=1, min_periods=1).sum()
            df[f'precip_{station}_lag_24h'] = df[col_name].rolling(window=1, min_periods=1).sum()
            df[f'precip_{station}_lag_48h'] = df[col_name].rolling(window=2, min_periods=1).sum()
            df[f'precip_{station}_lag_72h'] = df[col_name].rolling(window=3, min_periods=1).sum()
        else:
            print(f"Warning: Column {col_name} not found in DataFrame for feature engineering.")

    return df


def build_anfis(num_inputs: int, num_mfs: int):
    invardefs = [(f'x{i}', [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(num_mfs)]) for i in
                 range(num_inputs)]
    return AnfisNet('Flood Prediction Model', invardefs, ['y'], hybrid=True)


# --- Prediction Job (Generic) ---

def run_prediction_job(config: dict):
    """
    Runs the ANFIS prediction for a specific river configuration.
    """
    print(f"--- Running prediction job for {config['name']} ---")

    try:
        with open(config["config_json_path"], "r") as f:
            model_config = json.load(f)
        features_list = model_config["features_list"]

        scaler_X = joblib.load(config["scaler_x_path"])
        scaler_y = joblib.load(config["scaler_y_path"])

        anfis_model = build_anfis(model_config["num_inputs"], model_config["num_mfs"])
        checkpoint = torch.load(config["anfis_model_path"], map_location="cpu")
        anfis_model.load_state_dict(checkpoint['model_state_dict'])
        anfis_model.coeff = checkpoint['consequent_coeffs']
        anfis_model.eval()

        print(f"-> Fetching today's live data for {config['name']}...")
        today = datetime.now().date()
        live_water_level = fetch_recent_water_level(config["hydro_station"], today)

        live_precip_data = {}
        for code in config['meteo_stations_codes']:
            live_precip_data[f'precip_{code}_mm'] = fetch_recent_precipitation(code, today)

        if live_water_level is None:
            print(f"❌ Could not fetch live water level for {config['name']}. Skipping prediction.")
            return

        df_hist = pd.read_csv(config["data_file"])

        live_row_dict = {
            'timestamp': today.strftime("%Y-%m-%d"),
            'water_level_cm': live_water_level,
            **live_precip_data
        }

        live_row = pd.DataFrame([live_row_dict])
        df_combined = pd.concat([df_hist, live_row], ignore_index=True)

        features_df = prepare_features(df_combined, config)

        last_row = features_df.iloc[-1:]
        last_known_level = last_row['water_level_cm'].iloc[0]

        X_pred_unscaled = last_row[features_list].values

        # Handle potential NaN values from feature engineering
        if np.isnan(X_pred_unscaled).any():
            print(f"❌ NaN values detected in features for {config['name']}. Aborting prediction.")
            return

        X_pred_scaled = scaler_X.transform(X_pred_unscaled)
        X_pred_tensor = torch.from_numpy(X_pred_scaled).float()

        with torch.no_grad():
            predicted_change_scaled = anfis_model(X_pred_tensor)
            predicted_change = scaler_y.inverse_transform(predicted_change_scaled.numpy())[0, 0]

        predicted_next_day_level = last_known_level + predicted_change

        today_str = today.strftime("%Y-%m-%d")
        tomorrow_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        log_data = {}
        if os.path.exists(config["predictions_log_path"]):
            with open(config["predictions_log_path"], 'r') as f:
                try:
                    log_data = json.load(f)
                except json.JSONDecodeError:
                    log_data = {}

        log_data.setdefault(today_str, {})['actual'] = last_known_level
        log_data.setdefault(tomorrow_str, {})['predicted'] = predicted_next_day_level

        with open(config["predictions_log_path"], 'w') as f:
            json.dump(log_data, f, indent=4)

        print(f"✅ Prediction complete for {config['name']}. Log file updated.")
        print(f"   -> Today's Actual: {last_known_level:.2f} cm")
        print(f"   -> Tomorrow's Forecast: {predicted_next_day_level:.2f} cm")

    except FileNotFoundError as e:
        print(f"❌ ERROR for {config['name']}: Missing file {e.filename}. Skipping prediction.")
    except Exception as e:
        print(f"❌ An unexpected error occurred for {config['name']}: {e}")


# --- ---------------- ---
# --- FLASK API ROUTES ---
# --- ---------------- ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/run_predictions', methods=['POST'])
def run_predictions_api():
    """
    API endpoint triggered by the 'Refresh' button.
    Updates data and runs predictions for ALL rivers.
    """
    print("--- Received request to run all predictions ---")
    try:
        # Update historical data (yesterday)
        update_data_file(MINIJA_CONFIG)
        update_data_file(DANE_CONFIG)

        # Run new predictions (today)
        run_prediction_job(MINIJA_CONFIG)
        run_prediction_job(DANE_CONFIG)

        return jsonify({"status": "ok", "message": "Predictions updated for all rivers."})
    except Exception as e:
        print(f"Error during prediction run: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/data')
def get_data_api():
    """
    API endpoint to fetch the latest data for a specific river.
    Reads from the JSON log files. Does NOT trigger a new prediction.
    """
    river_name = request.args.get('river', 'minija')
    config = RIVER_CONFIGS.get(river_name)

    if not config:
        return jsonify({"error": "Invalid river name"}), 404

    try:
        # --- 1. Load Prediction Log ---
        log_data = {}
        if os.path.exists(config["predictions_log_path"]):
            with open(config["predictions_log_path"], 'r') as f:
                try:
                    log_data = json.load(f)
                except json.JSONDecodeError:
                    log_data = {}

        # --- 2. Load Model Config for features ---
        with open(config["config_json_path"], "r") as f:
            model_config = json.load(f)
        features_list = model_config["features_list"]

        # --- 3. Load Live Data for latest features ---
        df_hist = pd.read_csv(config["data_file"])
        today = datetime.now().date()
        live_water_level = fetch_recent_water_level(config["hydro_station"], today)

        live_precip_data = {}
        for code in config['meteo_stations_codes']:
            live_precip_data[f'precip_{code}_mm'] = fetch_recent_precipitation(code, today)

        if live_water_level is None:
            # Fallback to last known value from log
            today_str = today.strftime("%Y-%m-%d")
            last_known_level = log_data.get(today_str, {}).get('actual', 0)
            if last_known_level == 0 and len(df_hist) > 0:
                last_known_level = df_hist.iloc[-1]['water_level_cm']
            live_row_ok = False
        else:
            live_row_ok = True
            last_known_level = live_water_level

        features_df = pd.DataFrame()
        if live_row_ok:
            live_row_dict = {'timestamp': today.strftime("%Y-%m-%d"), 'water_level_cm': live_water_level,
                             **live_precip_data}
            live_row = pd.DataFrame([live_row_dict])
            df_combined = pd.concat([df_hist, live_row], ignore_index=True)
            features_df = prepare_features(df_combined, config)
        else:
            # Can't get live data, just use history
            features_df = prepare_features(df_hist, config)

        last_row = features_df.iloc[-1:]

        # --- 4. Get Data for UI ---

        # Get tomorrow's predicted level
        tomorrow_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        predicted_next_day_level = log_data.get(tomorrow_str, {}).get('predicted', 0)

        # Get latest 'actual' level (could be from log or live fetch)
        today_str = today.strftime("%Y-%m-%d")
        if live_row_ok:
            last_known_level = live_water_level
        else:
            last_known_level = log_data.get(today_str, {}).get('actual', df_hist.iloc[-1]['water_level_cm'] if len(
                df_hist) > 0 else 0)

        # Get live features for card
        def get_safe_float(series, key, default=0.0):
            val = series.get(key)
            return default if val is None or val.empty or pd.isna(val.iloc[0]) else float(val.iloc[0])

        live_features_dict = {}
        for station_short in config['meteo_stations_short']:
            live_features_dict[f"precip_{station_short}_lag_24h"] = get_safe_float(last_row,
                                                                                   f'precip_{station_short}_lag_24h')
            live_features_dict[f"precip_{station_short}_lag_72h"] = get_safe_float(last_row,
                                                                                   f'precip_{station_short}_lag_72h')

        # --- 5. Risk and Trend ---
        lvl_mod, lvl_high, lvl_severe = config['risk_levels']
        if predicted_next_day_level < lvl_mod:
            risk_level = 'LOW'
        elif predicted_next_day_level < lvl_high:
            risk_level = 'MODERATE'
        elif predicted_next_day_level < lvl_severe:
            risk_level = 'HIGH'
        else:
            risk_level = 'SEVERE'

        trend_diff = predicted_next_day_level - last_known_level
        if trend_diff > 10:
            trend_text = 'Rising'
        elif trend_diff < -10:
            trend_text = 'Falling'
        else:
            trend_text = 'Stable'

        # --- 6. Pad Historical Data for 30 Days ---
        padded_log_data = {}
        for i in range(30):  # 30 days into the past
            date_key = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            padded_log_data[date_key] = log_data.get(date_key, {})

        # Add tomorrow's prediction
        padded_log_data[tomorrow_str] = log_data.get(tomorrow_str, {})

        # Convert to sorted list for Chart.js
        historical_data = [{'date': k, **v} for k, v in sorted(padded_log_data.items())]

        return jsonify({
            "riverName": config['display_name'],
            "lastKnownLevel": last_known_level,
            "predictedNextDayLevel": predicted_next_day_level,
            "liveFeatures": live_features_dict,
            "historicalData": historical_data,
            "risk": {"level": risk_level},
            "trend": {"text": trend_text},
            "lastUpdated": datetime.now().isoformat()
        })

    except FileNotFoundError as e:
        return jsonify({"error": f"Missing data file: {e.filename}"}), 500
    except Exception as e:
        print(f"Error in /api/data for {river_name}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)