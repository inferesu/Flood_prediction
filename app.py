import json
import os
import time
import atexit
import logging
import warnings
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()

import joblib
import numpy as np
import pandas as pd
import requests
import torch
from flask import Flask, jsonify, render_template, request
from apscheduler.schedulers.background import BackgroundScheduler

# --- Import ANFIS classes ---
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('apscheduler')

app = Flask(__name__)

# --- HYDROLOGICAL CONSTANTS ---
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
    "meteo_stations": ["klaipedos-ams", "vezaiciu-ams"],
    "risk_levels": [250, 400, 550]
}

RIVER_CONFIGS = {"minija": MINIJA_CONFIG}


# --- DATA FETCHING ---

def fetch_water_level_latest(station_code):
    """Fetches the latest measured water level (WLt)."""
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
        except Exception:
            pass
    return None


def fetch_meteo_latest(station_code):
    """Fetches precip (Pt) and temperature for SMI calculation."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    url = f"https://api.meteo.lt/v1/stations/{station_code}/observations/{date_str}"
    try:
        resp = requests.get(url, timeout=10)
        obs = resp.json().get("observations", [])
        precip = sum(o.get('precipitation', 0) for o in obs if o.get('precipitation') is not None)
        temps = [o['airTemperature'] for o in obs if o.get('airTemperature') is not None]
        avg_t = sum(temps) / len(temps) if temps else 0
        return round(precip, 2), round(avg_t, 2)
    except Exception:
        return 0, 0


# --- COMPLEX FEATURE ENGINEERING (Eq. 9-23) ---

def prepare_complex_features(df, config):
    """
    Implements recursive API memory, Seasonal Cosine, and Snowmelt logic.
    """
    df = df.copy().sort_values('timestamp')
    p_cols = [c for c in df.columns if 'precip' in c]
    df['Pt'] = df[p_cols].mean(axis=1)  # Pt: current precipitation

    # API_t = Pt + k * API_t-1
    api_vals, current_api = [], 0
    for p in df['Pt']:
        current_api = p + (K_DECAY * current_api)
        api_vals.append(current_api)
    df['API_t'] = api_vals

    # API Normalization (Eq. 10)
    a_min, a_max = df['API_t'].min(), df['API_t'].max()
    df['API_norm'] = (df['API_t'] - a_min) / (a_max - a_min) if a_max != a_min else 0

    # Season_cos = cos(2*pi*d / 365)
    d = pd.to_datetime(df['timestamp']).dt.dayofyear
    df['S_t'] = np.cos((2 * np.pi * d) / 365)

    # Snowmelt Index (SMI_t)
    t_cols = [c for c in df.columns if 'temp' in c]
    if t_cols:
        avg_temp = df[t_cols].mean(axis=1)
        df['SMI_t'] = avg_temp.apply(lambda x: max(0, x * 2.5) if x > 0 else 0)
    else:
        df['SMI_t'] = 0

    # Trend Persistence (delta_WL_t)
    df['delta_WL_t'] = df['water_level_cm'].diff().fillna(0)
    return df


# --- PREDICTION JOB ---

def run_prediction_job(config):
    logger.info(f"--- Running Prediction for {config['display_name']} ---")
    try:
        # 1. Initialize data file if empty
        if not os.path.exists(config["data_file"]):
            pd.DataFrame(columns=['timestamp', 'water_level_cm']).to_csv(config["data_file"], index=False)

        # 2. Build ANFIS (Matches x0-x4 naming and 5-input logic)
        with open(config["config_json_path"], "r") as f:
            m_cfg = json.load(f)

        invardefs = [
            (f'x{i}', [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(m_cfg["num_mfs"])]) for
            i in range(5)]
        model = AnfisNet('Complex Flood Model', invardefs, ['y'], hybrid=True)

        ckpt = torch.load(config["anfis_model_path"], map_location="cpu")
        model.load_state_dict(ckpt['model_state_dict'])
        model.coeff = ckpt.get('consequent_coeffs') or ckpt.get('coeff')
        model.eval()

        # 3. Get Live Data
        live_wl = fetch_water_level_latest(config["hydro_station"])
        if live_wl is None: return

        live_row = {'timestamp': datetime.now().strftime("%Y-%m-%d"), 'water_level_cm': live_wl}
        for s in config['meteo_stations']:
            p, t = fetch_meteo_latest(s)
            live_row[f'precip_{s}_mm'], live_row[f'temp_{s}_c'] = p, t
            time.sleep(0.5)

        # 4. Feature Extraction
        df_hist = pd.read_csv(config["data_file"])
        df_comb = pd.concat([df_hist, pd.DataFrame([live_row])], ignore_index=True).drop_duplicates('timestamp')
        df_feats = prepare_complex_features(df_comb, config)

        # Vector: [API_norm, S_t, SMI_t, Pt, delta_WL_t]
        X_scaled = joblib.load(config["scaler_x_path"]).transform(
            df_feats.iloc[-1:][['API_norm', 'S_t', 'SMI_t', 'Pt', 'delta_WL_t']].values)

        with torch.no_grad():
            pred_chg = \
            joblib.load(config["scaler_y_path"]).inverse_transform(model(torch.from_numpy(X_scaled).float()).numpy())[
                0, 0]

        # 5. Save Log
        tomorrow_s = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        log_data = {}
        if os.path.exists(config["predictions_log_path"]):
            with open(config["predictions_log_path"], 'r') as f: log_data = json.load(f)

        log_data.setdefault(datetime.now().strftime("%Y-%m-%d"), {})['actual'] = live_wl
        log_data.setdefault(tomorrow_s, {})['predicted'] = round(live_wl + pred_chg, 2)
        log_data.setdefault(tomorrow_s, {})['report'] = f"Forecasted level is {live_wl + pred_chg:.2f} cm."

        with open(config["predictions_log_path"], 'w') as f:
            json.dump(log_data, f, indent=4)
        logger.info(f"✅ Prediction Archived: {live_wl + pred_chg:.2f}")

    except Exception as e:
        logger.error(f"Prediction Job Failed: {e}")


# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/data')
def get_data_api():
    river = request.args.get('river', 'minija')
    config = RIVER_CONFIGS.get(river)
    if not config: return jsonify({"error": "River config missing"}), 404

    log_data = {}
    if os.path.exists(config["predictions_log_path"]):
        with open(config["predictions_log_path"], 'r') as f: log_data = json.load(f)

    today_s = datetime.now().strftime("%Y-%m-%d")
    tomorrow_s = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    return jsonify({
        "riverName": config['display_name'],
        "lastKnownLevel": log_data.get(today_s, {}).get('actual', 0),
        "predictedNextDayLevel": log_data.get(tomorrow_s, {}).get('predicted', 0),
        "aiReport": log_data.get(tomorrow_s, {}).get('report', "No forecast available yet."),
        "historicalData": [{"date": k, "actual": v.get('actual'), "predicted": v.get('predicted')} for k, v in
                           sorted(log_data.items())][-30:],
        "lastUpdated": datetime.now().isoformat()
    })


# --- SCHEDULER & STARTUP ---

scheduler = BackgroundScheduler()
scheduler.add_job(func=lambda: run_prediction_job(MINIJA_CONFIG), trigger="cron", minute="05")
scheduler.start()

if __name__ == "__main__":
    # TRIGGERS A RUN IMMEDIATELY ON STARTUP TO POPULATE THE FRONTEND
    run_prediction_job(MINIJA_CONFIG)
    app.run(debug=True, port=5003, use_reloader=False)