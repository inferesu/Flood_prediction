import json
import os
import time
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
import requests
import torch
import warnings  # Added for suppressing PyTorch warnings

# --- LLM Integration Imports ---
from google import genai
from google.genai import types
# Set the environment variable for your API key (or set it here)

# --- Import ANFIS classes from your project ---
# Make sure this import works from your script's location
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# Suppress PyTorch warnings during model load/use
warnings.filterwarnings("ignore", category=UserWarning)

# --- GLOBAL LLM & MODEL SETUP ---
GEMINI_MODEL = 'gemini-2.5-flash'
try:
    # Client initialized here; automatically uses GEMINI_API_KEY env var
    client = genai.Client()
except Exception as e:
    print(f"LLM Initialization Error: {e}")
    print("WARNING: Gemini client failed to initialize. LLM reporting will be skipped.")
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
    "risk_thresholds": {"HIGH": 300, "MODERATE": 250},  # Hypothetical risk levels (cm)
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
    "risk_thresholds": {"HIGH": 200, "MODERATE": 180},  # Hypothetical risk levels (cm)
}


# --- Data Collection Logic (Generic) ---

def fetch_recent_water_level(station_code, date):
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.meteo.lt/v1/hydro-stations/{station_code}/observations/measured/{date_str}"
    # print(f"-> Fetching water level for '{station_code}' on {date_str}...")
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        # Find the latest observation
        latest_obs = max(observations, key=lambda x: datetime.strptime(x['timestamp'], "%Y-%m-%d %H:%M:%S"), default={})
        water_level = latest_obs.get('waterLevel')
        if water_level is not None:
            return round(water_level, 2)
    except requests.RequestException as e:
        print(f"   - ❌ Error fetching water level: {e}")
    return None


def fetch_recent_precipitation(station_code, date):
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.meteo.lt/v1/stations/{station_code}/observations/{date_str}"
    # print(f"-> Fetching precipitation for '{station_code}' on {date_str}...")
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
    # Fetch data for yesterday (D-1)
    yesterday = datetime.now().date() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    data_file = config['data_file']
    hydro_station = config['hydro_station']
    meteo_station_codes = config['meteo_stations_codes']

    # Check if data already exists
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
        # Note: If meteo API provides hourly, you might need a different aggregation logic
        # to get 12h/24h/48h/72h rollups more accurately in the future.
        precip_data[f'precip_{code}_mm'] = fetch_recent_precipitation(code, yesterday)
        time.sleep(0.5)

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
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Resample to daily frequency and forward-fill to ensure continuous index
    df = df.drop_duplicates(subset='timestamp', keep='last')
    df = df.set_index('timestamp').asfreq('D').ffill()

    # Feature calculation for each configured station
    for station in config['meteo_stations_short']:
        # The column name in the CSV is e.g., precip_klaipedos-ams_mm
        col_name = f'precip_{station}-ams_mm'
        if col_name in df.columns:
            # Re-evaluating the window size based on the likely daily sampling of the CSV
            # If the CSV is daily, a window of 1 is 1 day.
            df[f'precip_{station}_lag_12h'] = df[col_name].rolling(window=1, min_periods=1).sum()  # Daily accumulation
            df[f'precip_{station}_lag_24h'] = df[col_name].rolling(window=1, min_periods=1).sum()  # Daily accumulation
            df[f'precip_{station}_lag_48h'] = df[col_name].rolling(window=2, min_periods=1).sum()  # 2-day accumulation
            df[f'precip_{station}_lag_72h'] = df[col_name].rolling(window=3, min_periods=1).sum()  # 3-day accumulation
        else:
            print(f"Warning: Column {col_name} not found in DataFrame for feature engineering.")

    return df


def build_anfis(num_inputs: int, num_mfs: int):
    # This function must return an ANFIS structure matching the trained model
    invardefs = [(f'x{i}', [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(num_mfs)]) for i in
                 range(num_inputs)]
    return AnfisNet('Flood Prediction Model', invardefs, ['y'], hybrid=True)


# --- LLM Reporting Function ---

def generate_gemini_report(config: dict, current_level: float, predicted_change: float, projected_level: float,
                           rain_24h: float) -> str:
    """Uses the Gemini API to generate a human-readable report."""
    if client is None:
        return "❌ Gemini client not initialized. Cannot generate report."

    # Determine Risk Level based on configuration thresholds
    risk_level = "LOW"
    if projected_level >= config['risk_thresholds'].get("HIGH", 999):
        risk_level = "HIGH ALERT"
    elif projected_level >= config['risk_thresholds'].get("MODERATE", 999):
        risk_level = "MODERATE"

    # Prepare prompt with dynamic context
    prompt = f"""
    You are a professional Hydrology Analyst for the {config['display_name']} river system. Your task is to provide a brief, actionable report for city officials.

    ANFIS Model Telemetry:
    - Current Water Level (Today): {current_level:.2f} cm
    - Rainfall (Last 24h, accumulated): {rain_24h:.2f} mm
    - Predicted Water Level Change (Next 24h): {predicted_change:.2f} cm
    - Projected Water Level (Tomorrow): {projected_level:.2f} cm

    Risk Thresholds (for your context):
    - HIGH ALERT: >={config['risk_thresholds'].get("HIGH", 'N/A')} cm
    - MODERATE: >={config['risk_thresholds'].get("MODERATE", 'N/A')} cm

    Task: Write a short, professional status report (2-4 sentences). Explicitly state the **Risk Level ({risk_level})** and one key **Actionable Recommendation** based on the risk and projected level.
    """

    # Use System Instruction for better role adherence
    config_llm = types.GenerateContentConfig(
        system_instruction="You are a clear, concise, and professional Hydrology Analyst providing actionable intelligence. Do not provide numerical predictions, only the analysis and recommendation."
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config_llm
        )
        return response.text
    except Exception as e:
        return f"❌ Gemini API call failed: {e}. Check API key and quota."


# --- Prediction Job (Generic) ---

def run_prediction_job(config: dict):
    print(f"\n--- Running prediction job for {config['display_name']} ---")

    try:
        # Load Config
        with open(config["config_json_path"], "r") as f:
            model_config = json.load(f)
        features_list = model_config["features_list"]

        # Load Scalers and Model
        scaler_X = joblib.load(config["scaler_x_path"])
        scaler_y = joblib.load(config["scaler_y_path"])
        anfis_model = build_anfis(model_config["num_inputs"], model_config["num_mfs"])
        checkpoint = torch.load(config["anfis_model_path"], map_location="cpu")
        anfis_model.load_state_dict(checkpoint['model_state_dict'])
        anfis_model.coeff = checkpoint['consequent_coeffs']
        anfis_model.eval()

        # Fetch Live Data (Today)
        today = datetime.now().date()
        live_water_level = fetch_recent_water_level(config["hydro_station"], today)

        live_precip_data = {}
        for code in config['meteo_stations_codes']:
            # We assume this provides the daily cumulative precipitation up to the current time
            live_precip_data[f'precip_{code}_mm'] = fetch_recent_precipitation(code, today)

        if live_water_level is None:
            print(f"❌ Could not fetch live water level for {config['name']}. Skipping prediction.")
            return

        # Prepare Data for Prediction
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

        # Get 24h accumulated rainfall for the LLM report (using the engineered feature if possible, or the raw data)
        # Using the engineered 24h feature from the last row for consistency
        rain_24h = last_row[features_list[1]].iloc[0] if len(features_list) > 1 else 0

        X_pred_unscaled = last_row[features_list].values
        if np.isnan(X_pred_unscaled).any():
            print(f"❌ NaN values detected in features for {config['name']}. Aborting prediction.")
            return

        # Numerical Prediction
        X_pred_scaled = scaler_X.transform(X_pred_unscaled)
        X_pred_tensor = torch.from_numpy(X_pred_scaled).float()

        with torch.no_grad():
            predicted_change_scaled = anfis_model(X_pred_tensor)
            predicted_change = scaler_y.inverse_transform(predicted_change_scaled.numpy())[0, 0]

        predicted_next_day_level = last_known_level + predicted_change

        # LLM Reporting (New Step)
        gemini_report = generate_gemini_report(
            config,
            last_known_level,
            predicted_change,
            predicted_next_day_level,
            rain_24h
        )

        # Logging (Updated to include LLM report)
        today_str = today.strftime("%Y-%m-%d")
        tomorrow_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")

        log_data = {}
        if os.path.exists(config["predictions_log_path"]):
            with open(config["predictions_log_path"], 'r') as f:
                try:
                    log_data = json.load(f)
                except json.JSONDecodeError:
                    log_data = {}

        # Log prediction and report
        log_data.setdefault(today_str, {})['actual'] = last_known_level
        log_data.setdefault(tomorrow_str, {})['predicted'] = predicted_next_day_level
        log_data.setdefault(tomorrow_str, {})['report'] = gemini_report

        with open(config["predictions_log_path"], 'w') as f:
            json.dump(log_data, f, indent=4)

        print(f"\n✅ Prediction and Reporting complete for {config['display_name']}.")
        print(f"   -> Today's Actual: {last_known_level:.2f} cm")
        print(f"   -> Tomorrow's Forecast: {predicted_next_day_level:.2f} cm")
        print(f"\n*** Gemini Report Saved to Log ***\n{gemini_report}")

    except FileNotFoundError as e:
        print(f"❌ ERROR for {config['name']}: Missing file {e.filename}. Skipping prediction.")
    except Exception as e:
        print(f"❌ An unexpected error occurred for {config['name']}: {e}")


if __name__ == "__main__":
    print(f"--- Running Daily Prediction and Reporting Job at {datetime.now()} ---")

    print("\n--- Processing Minija ---")
    update_data_file(MINIJA_CONFIG)
    run_prediction_job(MINIJA_CONFIG)

    print("\n--- Processing Danė ---")
    update_data_file(DANE_CONFIG)
    run_prediction_job(DANE_CONFIG)

    print(f"\n--- All jobs complete. ---")