import json
import os
import time
import atexit
from datetime import datetime, timedelta
import warnings

import joblib
import numpy as np
import pandas as pd
import requests
import torch
from flask import Flask, jsonify, render_template, request

# --- Scheduler Imports ---
from apscheduler.schedulers.background import BackgroundScheduler

# --- LLM Integration Imports ---
from google import genai
from google.genai import types

# --- Import ANFIS classes ---
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Flask App Initialization ---
app = Flask(__name__)

# --- GLOBAL LLM SETUP ---
GEMINI_MODEL = 'gemini-2.5-flash'
try:
    # Ensure GEMINI_API_KEY is set in your environment variables
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
except Exception as e:
    print(f"LLM Initialization Warning: {e}")
    client = None

# --- CONFIGURATION ---
K_DECAY = 0.85  # Calibrated decay coefficient for Lithuania

MINIJA_CONFIG = {
    "name": "minija",
    "display_name": "Minija (Priekulė)",
    "data_file": "live_data_complex.csv",
    "anfis_model_path": "anfis_model.pth",
    "scaler_x_path": "scaler_X.pkl",
    "scaler_y_path": "scaler_Y.pkl",
    "config_json_path": "training_config.json",
    "predictions_log_path": "predictions_log.json",
    "hydro_station": "priekules-vms",
    "meteo_stations_codes": ["klaipedos-ams", "vezaiciu-ams"],
    "risk_levels": [250, 400, 550]
}

DANE_CONFIG = {
    "name": "dane",
    "display_name": "Danė (Klaipėda)",
    "data_file": "live_data_dane_complex.csv",
    "anfis_model_path": "anfis_model_dane.pth",
    "scaler_x_path": "scaler_X_dane.pkl",
    "scaler_y_path": "scaler_Y_dane.pkl",
    "config_json_path": "training_config_dane.json",
    "predictions_log_path": "predictions_log_dane.json",
    "hydro_station": "klaipedos-vms",
    "meteo_stations_codes": ["klaipedos-ams"],
    "risk_levels": [150, 250, 350]
}

RIVER_CONFIGS = {"minija": MINIJA_CONFIG, "dane": DANE_CONFIG}


# --- DATA FETCHING FUNCTIONS ---

def fetch_water_level_latest(station_code):
    """Fetches the latest measured water level."""
    print(f"   -> Fetching Hydro Snapshot: {station_code}")
    for days_back in [0, 1]:
        date_str = (datetime.now().date() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = f"https://api.meteo.lt/v1/hydro-stations/{station_code}/observations/measured/{date_str}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get("observations", [])
                valid = [x for x in data if x.get('waterLevel') is not None]
                if valid:
                    valid.sort(key=lambda x: x.get('observationTimeUtc', ''))
                    return round(valid[-1]['waterLevel'], 2)
        except Exception as e:
            print(f"   - Hydro Fetch error: {e}")
    return None


def fetch_meteo_daily_stats(station_code, date):
    """Fetches daily precip sum and avg temp for SMI and API calibration."""
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.meteo.lt/v1/stations/{station_code}/observations/{date_str}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        precip = sum(o.get('precipitation', 0) for o in obs if o.get('precipitation') is not None)
        temps = [o['airTemperature'] for o in obs if o.get('airTemperature') is not None]
        avg_temp = round(sum(temps) / len(temps), 2) if temps else 0
        return round(precip, 2), avg_temp
    except Exception as e:
        print(f"   - Meteo Error ({station_code}): {e}")
    return 0, 0


def fetch_water_level_average(station_code, date):
    """Historical average for archival."""
    url = f"https://api.meteo.lt/v1/hydro-stations/{station_code}/observations/measured/{date.strftime('%Y-%m-%d')}"
    try:
        resp = requests.get(url, timeout=20)
        obs = resp.json().get("observations", [])
        levels = [o['waterLevel'] for o in obs if o.get('waterLevel') is not None]
        if levels: return round(sum(levels) / len(levels), 2)
    except Exception:
        pass
    return None


# --- COMPLEX FEATURE ENGINEERING (Eq. 9-23) ---

def prepare_complex_features(df, config):
    """Implements recursive API, Seasonality, and Snowmelt logic."""
    df = df.copy().sort_values('timestamp')

    # 1. Pt - Mean Daily Precipitation
    p_cols = [c for c in df.columns if 'precip' in c]
    df['Pt'] = df[p_cols].mean(axis=1)

    # 2. API_t = Pt + k * API_t-1 (Eq. 9)
    api_vals, current_api = [], 0
    for p in df['Pt']:
        current_api = p + (K_DECAY * current_api)
        api_vals.append(current_api)
    df['API_t'] = api_vals

    # 3. API Normalization (Eq. 10)
    a_min, a_max = df['API_t'].min(), df['API_t'].max()
    df['API_norm'] = (df['API_t'] - a_min) / (a_max - a_min) if a_max != a_min else 0

    # 4. Seasonality S_t (Eq. 11)
    d = pd.to_datetime(df['timestamp']).dt.dayofyear
    df['S_t'] = np.cos((2 * np.pi * d) / 365)

    # 5. Snowmelt Index SMI_t (Eq. 17)
    t_cols = [c for c in df.columns if 'temp' in c]
    if t_cols:
        avg_temp = df[t_cols].mean(axis=1)
        df['SMI_t'] = avg_temp.apply(lambda x: max(0, x * 2.5) if x > 0 else 0)
    else:
        df['SMI_t'] = 0

    # 6. Trend delta_WL_t (Eq. 21)
    df['delta_WL_t'] = df['water_level_cm'].diff().fillna(0)

    return df


# --- FILE MANAGEMENT ---

def update_data_file(config: dict):
    """Archives yesterday's stats to maintain the API recursive 'memory'."""
    yesterday = datetime.now().date() - timedelta(days=1)
    y_str = yesterday.strftime("%Y-%m-%d")
    data_file = config['data_file']

    if os.path.exists(data_file):
        df_ex = pd.read_csv(data_file)
        if y_str in df_ex['timestamp'].values: return
    else:
        cols = ['timestamp', 'water_level_cm']
        for c in config['meteo_stations_codes']:
            cols += [f'precip_{c}_mm', f'temp_{c}_c']
        pd.DataFrame(columns=cols).to_csv(data_file, index=False)

    wl = fetch_water_level_average(config['hydro_station'], yesterday)
    row = {'timestamp': y_str, 'water_level_cm': wl}
    for c in config['meteo_stations_codes']:
        p, t = fetch_meteo_daily_stats(c, yesterday)
        row[f'precip_{c}_mm'], row[f'temp_{c}_c'] = p, t
        time.sleep(0.5)

    if wl is not None:
        pd.DataFrame([row]).to_csv(data_file, mode='a', header=False, index=False)
        print(f"✅ Archived History: {row}")


# --- MODEL & PREDICTION ---

def build_anfis(num_inputs, num_mfs):
    """Matches the 'x{i}' naming convention required by updated training."""
    invardefs = [(f'x{i}', [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(num_mfs)])
                 for i in range(num_inputs)]
    return AnfisNet('Flood Model', invardefs, ['y'], hybrid=True)


def run_prediction_job(config):
    print(f"--- Predicting {config['display_name']} ---")
    try:
        with open(config["config_json_path"], "r") as f:
            m_config = json.load(f)
        scaler_X = joblib.load(config["scaler_x_path"])
        scaler_y = joblib.load(config["scaler_y_path"])

        # Build model based on 5 complex features
        model = build_anfis(5, m_config["num_mfs"])
        ckpt = torch.load(config["anfis_model_path"], map_location="cpu")
        model.load_state_dict(ckpt['model_state_dict'])
        model.coeff = ckpt.get('consequent_coeffs') or ckpt.get('coeff')
        model.eval()

        today = datetime.now().date()
        live_wl = fetch_water_level_latest(config["hydro_station"])
        if live_wl is None: return

        live_row = {'timestamp': today.strftime("%Y-%m-%d"), 'water_level_cm': live_wl}
        for c in config['meteo_stations_codes']:
            p, t = fetch_meteo_daily_stats(c, today)
            live_row[f'precip_{c}_mm'], live_row[f'temp_{c}_c'] = p, t
            time.sleep(0.5)

        # Concatenate history and live row to calculate recursive API
        df_hist = pd.read_csv(config["data_file"])
        df_comb = pd.concat([df_hist, pd.DataFrame([live_row])], ignore_index=True)
        df_feats = prepare_complex_features(df_comb, config)

        last_row = df_feats.iloc[-1:]
        # locked feature order to match training
        feature_order = ['API_norm', 'S_t', 'SMI_t', 'Pt', 'delta_WL_t']
        X_unscaled = last_row[feature_order].values

        X_scaled = scaler_X.transform(X_unscaled)
        with torch.no_grad():
            pred_chg_scaled = model(torch.from_numpy(X_scaled).float()).numpy()
            pred_chg = scaler_y.inverse_transform(pred_chg_scaled)[0, 0]

        pred_level = live_wl + pred_chg
        report = generate_gemini_report(config, live_wl, pred_chg, pred_level, last_row['Pt'].iloc[0])

        # Log prediction
        today_s, tomorrow_s = today.strftime("%Y-%m-%d"), (today + timedelta(days=1)).strftime("%Y-%m-%d")
        log = {}
        if os.path.exists(config["predictions_log_path"]):
            with open(config["predictions_log_path"], 'r') as f: log = json.load(f)

        log.setdefault(today_s, {})['actual'] = live_wl
        log.setdefault(tomorrow_s, {})['predicted'] = round(pred_level, 2)
        log.setdefault(tomorrow_s, {})['report'] = report

        with open(config["predictions_log_path"], 'w') as f:
            json.dump(log, f, indent=4)
        print(f"✅ Prediction Success: {live_wl} -> {pred_level:.2f}")

    except Exception as e:
        print(f"❌ Error: {e}")


# --- AI & ROUTES ---

def generate_gemini_report(config, current_level, predicted_change, projected_level, rain_24h):
    if client is None: return "AI unavailable."
    risk = "LOW"
    l_mod, l_high, l_severe = config['risk_levels']
    if projected_level >= l_severe:
        risk = "SEVERE"
    elif projected_level >= l_high:
        risk = "HIGH"
    elif projected_level >= l_mod:
        risk = "MODERATE"

    prompt = f"Hydrology Report for {config['display_name']}. Current: {current_level}cm. Rain: {rain_24h}mm. Forecast: {projected_level}cm ({predicted_change:+.2f}cm change). Risk: {risk}. Provide a brief 2-sentence public safety summary."
    try:
        return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text
    except Exception:
        return "Report generation error."


@app.route('/')
def index(): return render_template('index.html')


@app.route('/api/run_predictions', methods=['POST'])
def run_predictions_api():
    scheduled_auto_prediction()
    return jsonify({"status": "ok"})


@app.route('/api/data')
def get_data_api():
    river = request.args.get('river', 'minija')
    config = RIVER_CONFIGS.get(river)
    if not config: return jsonify({"error": "Invalid river"}), 404

    # Logic to return processed JSON for UI...
    # (Same structure as your previous get_data_api, using prediction_log)
    return jsonify({"status": "data_ready"})


# --- SCHEDULER ---

def scheduled_auto_prediction():
    for cfg in RIVER_CONFIGS.values():
        update_data_file(cfg)
        run_prediction_job(cfg)


scheduler = BackgroundScheduler()
scheduler.add_job(func=scheduled_auto_prediction, trigger="cron", minute="05")
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

if __name__ == "__main__":
    app.run(debug=True, port=5002, use_reloader=False)