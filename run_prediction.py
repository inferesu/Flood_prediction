import json
import os
import time
from datetime import datetime, timedelta
import warnings

import joblib
import numpy as np
import pandas as pd
import requests
import torch

# --- LLM Integration Imports ---
from google import genai
from google.genai import types

# --- Import ANFIS classes from your project ---
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- GLOBAL LLM SETUP ---
GEMINI_MODEL = 'gemini-2.0-flash'
try:
    # Ensure GEMINI_API_KEY is set in your environment variables
    # Or replace os.environ.get(...) with your actual key string
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
except Exception as e:
    print(f"LLM Initialization Warning: {e}")
    client = None

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
    "meteo_stations_short": ["klaipedos", "vezaiciu"],
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
    "hydro_station": "klaipedos-vms",
    "meteo_stations_codes": ["klaipedos-ams"],
    "meteo_stations_short": ["klaipedos"],
    "risk_levels": [150, 250, 350]
}


# --- DATA FETCHING FUNCTIONS ---

def fetch_water_level_latest(station_code):
    """
    Fetches the absolute LATEST available water level observation.
    Uses the '/measured' endpoint (no date) which returns the last ~24h of data.
    We sort this list and pick the last one to guarantee the most recent reading.
    """
    url = f"https://api.meteo.lt/v1/hydro-stations/{station_code}/observations/measured"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])

        # Filter valid readings
        valid_obs = [obs for obs in observations if obs.get('waterLevel') is not None]

        if valid_obs:
            # Sort by time string (ISO format sorts alphabetically correctly)
            latest_obs = max(valid_obs, key=lambda x: x.get('observationTimeUTC', ''))
            level = latest_obs['waterLevel']
            print(
                f"   -> 💧 LATEST LIVE SNAPSHOT ({station_code}): {level} cm at {latest_obs.get('observationTimeUTC')}")
            return round(level, 2)

    except Exception as e:
        print(f"   - ❌ Error fetching latest level: {e}")
    return None


def fetch_water_level_average(station_code, date):
    """
    Fetches the DAILY AVERAGE for a specific date.
    Used for archiving historical data to CSV.
    """
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.meteo.lt/v1/hydro-stations/{station_code}/observations/measured/{date_str}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        levels = [obs['waterLevel'] for obs in observations if obs.get('waterLevel') is not None]
        if levels:
            avg = sum(levels) / len(levels)
            print(f"   -> 📊 Daily Average ({date_str}): {avg:.2f} cm")
            return round(avg, 2)
    except Exception:
        pass
    return None


def fetch_precipitation_sum(station_code, date):
    """Fetches the sum of precipitation for a given date."""
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.meteo.lt/v1/stations/{station_code}/observations/{date_str}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        daily_precip = sum(obs.get('precipitation', 0) for obs in observations if obs.get('precipitation') is not None)
        return round(daily_precip, 2)
    except Exception as e:
        print(f"   - ❌ Error fetching precipitation: {e}")
    return 0


# --- FILE MANAGEMENT (Archiving) ---

def update_data_file(config: dict):
    """
    Updates the CSV file with YESTERDAY'S data.
    We use the DAILY AVERAGE for the CSV to keep historical graphs smooth.
    """
    yesterday = datetime.now().date() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    data_file = config['data_file']

    # Check if data already exists
    if os.path.exists(data_file):
        df_existing = pd.read_csv(data_file)
        if yesterday_str in df_existing['timestamp'].values:
            print(f"✅ Data for {config['name']} ({yesterday_str}) already archives.")
            return
    else:
        print(f"File '{data_file}' not found. Creating it.")
        columns = ['timestamp', 'water_level_cm'] + [f'precip_{code}_mm' for code in config['meteo_stations_codes']]
        pd.DataFrame(columns=columns).to_csv(data_file, index=False)

    print(f"--- Archiving Average Data for {config['name']} ({yesterday_str}) ---")
    water_level = fetch_water_level_average(config['hydro_station'], yesterday)
    precip_data = {}
    for code in config['meteo_stations_codes']:
        precip_data[f'precip_{code}_mm'] = fetch_precipitation_sum(code, yesterday)
        time.sleep(0.5)

    if water_level is None:
        print(f"❌ Could not fetch average water level for yesterday. Skipping archive.")
        return

    new_data = pd.DataFrame([{'timestamp': yesterday_str, 'water_level_cm': water_level, **precip_data}])
    new_data.to_csv(data_file, mode='a', header=False, index=False)
    print(f"✅ Archived data for {yesterday_str}.")


# --- FEATURE ENGINEERING ---

def prepare_features(df: pd.DataFrame, config: dict):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Clean duplicates and ensure daily frequency for lag calculations
    df = df.drop_duplicates(subset='timestamp', keep='last')
    df = df.set_index('timestamp').asfreq('D').ffill()

    for station in config['meteo_stations_short']:
        col_name = f'precip_{station}-ams_mm'
        if col_name in df.columns:
            df[f'precip_{station}_lag_12h'] = df[col_name].rolling(window=1, min_periods=1).sum()
            df[f'precip_{station}_lag_24h'] = df[col_name].rolling(window=1, min_periods=1).sum()
            df[f'precip_{station}_lag_48h'] = df[col_name].rolling(window=2, min_periods=1).sum()
            df[f'precip_{station}_lag_72h'] = df[col_name].rolling(window=3, min_periods=1).sum()
    return df


def build_anfis(num_inputs: int, num_mfs: int):
    invardefs = [(f'x{i}', [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(num_mfs)]) for i in
                 range(num_inputs)]
    return AnfisNet('Flood Prediction Model', invardefs, ['y'], hybrid=True)


# --- LLM REPORTING ---

def generate_gemini_report(config: dict, current_level: float, predicted_change: float, projected_level: float,
                           rain_24h: float) -> str:
    if client is None:
        return "AI Analysis unavailable (Client not initialized)."

    lvl_mod, lvl_high, lvl_severe = config['risk_levels']
    risk_level = "LOW"
    if projected_level >= lvl_severe:
        risk_level = "SEVERE"
    elif projected_level >= lvl_high:
        risk_level = "HIGH"
    elif projected_level >= lvl_mod:
        risk_level = "MODERATE"

    prompt = f"""
    Role: Hydrology Analyst for the {config['display_name']} river system.

    LIVE TELEMETRY:
    - Current Water Level: {current_level:.2f} cm (Live Sensor Reading)
    - Rainfall (Last 24h): {rain_24h:.2f} mm
    - Predicted Change (24h): {predicted_change:+.2f} cm
    - Forecast Level: {projected_level:.2f} cm
    - Risk Level: {risk_level}

    Task: Write a short, professional status report (2-3 sentences) for local residents. 
    Explicitly state if the water is **rising**, **falling**, or **stable** based on the forecast.
    """

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI Analysis unavailable: {str(e)}"


# --- MAIN PREDICTION LOGIC ---

def run_prediction_job(config: dict):
    print(f"\n--- Running LIVE Prediction for {config['display_name']} ---")

    try:
        # 1. Load Model Config
        with open(config["config_json_path"], "r") as f:
            model_config = json.load(f)
        features_list = model_config["features_list"]

        # 2. Load Scalers and Model
        scaler_X = joblib.load(config["scaler_x_path"])
        scaler_y = joblib.load(config["scaler_y_path"])
        anfis_model = build_anfis(model_config["num_inputs"], model_config["num_mfs"])
        checkpoint = torch.load(config["anfis_model_path"], map_location="cpu")
        anfis_model.load_state_dict(checkpoint['model_state_dict'])
        anfis_model.coeff = checkpoint['consequent_coeffs']
        anfis_model.eval()

        # 3. Fetch LIVE Data (The exact snapshot)
        today = datetime.now().date()
        live_water_level = fetch_water_level_latest(config["hydro_station"])

        if live_water_level is None:
            print(f"❌ Could not fetch live water level for {config['name']}. Skipping prediction.")
            return

        live_precip_data = {}
        for code in config['meteo_stations_codes']:
            live_precip_data[f'precip_{code}_mm'] = fetch_precipitation_sum(code, today)

        # 4. Combine History + Live Row for Feature Engineering
        df_hist = pd.read_csv(config["data_file"])
        live_row_dict = {
            'timestamp': today.strftime("%Y-%m-%d"),
            'water_level_cm': live_water_level,
            **live_precip_data
        }
        live_row = pd.DataFrame([live_row_dict])
        df_combined = pd.concat([df_hist, live_row], ignore_index=True)
        features_df = prepare_features(df_combined, config)

        # 5. Extract Features for the Model
        last_row = features_df.iloc[-1:]
        X_pred_unscaled = last_row[features_list].values

        # --- CRITICAL: FORCE INJECTION ---
        # Overwrite the water level in the input array with the raw LIVE value.
        # This ensures the model sees "429.4" even if Pandas smoothed it out.
        if "water_level_cm" in features_list:
            idx = features_list.index("water_level_cm")
            print(f"   ⚠️  Force-Injecting Live Water Level into Model Input: {live_water_level} cm")
            X_pred_unscaled[0, idx] = live_water_level

        if np.isnan(X_pred_unscaled).any():
            print(f"❌ NaN values detected in features. Aborting.")
            return

        # 6. Run Numerical Prediction
        X_pred_scaled = scaler_X.transform(X_pred_unscaled)
        X_pred_tensor = torch.from_numpy(X_pred_scaled).float()

        with torch.no_grad():
            predicted_change_scaled = anfis_model(X_pred_tensor)
            predicted_change = scaler_y.inverse_transform(predicted_change_scaled.numpy())[0, 0]

        # Forecast is Live Level + Predicted Change
        predicted_next_day_level = live_water_level + predicted_change

        # 7. Run LLM Report
        rain_col = [c for c in features_list if 'lag_24h' in c]
        rain_24h = last_row[rain_col[0]].iloc[0] if rain_col else 0

        gemini_report = generate_gemini_report(
            config,
            live_water_level,
            predicted_change,
            predicted_next_day_level,
            rain_24h
        )

        # 8. Save to Log
        today_str = today.strftime("%Y-%m-%d")
        tomorrow_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")

        log_data = {}
        if os.path.exists(config["predictions_log_path"]):
            with open(config["predictions_log_path"], 'r') as f:
                try:
                    log_data = json.load(f)
                except json.JSONDecodeError:
                    log_data = {}

        log_data.setdefault(today_str, {})['actual'] = live_water_level
        log_data.setdefault(tomorrow_str, {})['predicted'] = predicted_next_day_level
        log_data.setdefault(tomorrow_str, {})['report'] = gemini_report

        with open(config["predictions_log_path"], 'w') as f:
            json.dump(log_data, f, indent=4)

        print(f"✅ Prediction Complete.")
        print(f"   -> Input Level: {live_water_level:.2f} cm")
        print(f"   -> Forecast Level: {predicted_next_day_level:.2f} cm")
        print(f"   -> Report saved.")

    except Exception as e:
        print(f"❌ An unexpected error occurred for {config['name']}: {e}")


if __name__ == "__main__":
    print(f"--- Running Daily Prediction Job at {datetime.now()} ---")

    # 1. Archive yesterday's data (Averages)
    print("\n>>> Archiving Historical Data...")
    update_data_file(MINIJA_CONFIG)
    update_data_file(DANE_CONFIG)

    # 2. Run Predictions based on LIVE snapshots
    print("\n>>> Running Live Predictions...")
    run_prediction_job(MINIJA_CONFIG)
    run_prediction_job(DANE_CONFIG)

    print(f"\n--- All jobs complete. ---")