import json
import os
import time
import logging
import warnings
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import joblib
import numpy as np
import pandas as pd
import requests
import torch
from flask import Flask, jsonify, render_template, request
from apscheduler.schedulers.background import BackgroundScheduler

# --- Import ANFIS classes ---
# Ensure these files exist in a folder named 'anfis'
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress the concat warning

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('flood_app')

app = Flask(__name__)

# --- HYDROLOGICAL CONSTANTS ---
K_DECAY = 0.85

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
    # Look back up to 2 days in case the sensor is temporarily offline
    for days_back in [0, 1, 2]:
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
        # Sum precipitation for the day so far
        precip = sum(o.get('precipitation', 0) for o in obs if o.get('precipitation') is not None)
        # Avg temperature
        temps = [o['airTemperature'] for o in obs if o.get('airTemperature') is not None]
        avg_t = sum(temps) / len(temps) if temps else 0
        return round(precip, 2), round(avg_t, 2)
    except Exception:
        return 0, 0


# --- COMPLEX FEATURE ENGINEERING ---
def prepare_complex_features(df):
    """ Implements recursive API memory, Seasonal Cosine, and Snowmelt logic. """
    df = df.copy().sort_values('timestamp')

    # Calculate Mean Precipitation across stations
    p_cols = [c for c in df.columns if 'precip' in c]
    if p_cols:
        df['Pt'] = df[p_cols].mean(axis=1)
    else:
        df['Pt'] = 0

    # API_t = Pt + k * API_t-1
    api_vals, current_api = [], 0
    for p in df['Pt']:
        current_api = p + (K_DECAY * current_api)
        api_vals.append(current_api)
    df['API_t'] = api_vals

    # API Normalization
    a_min, a_max = df['API_t'].min(), df['API_t'].max()
    if a_max != a_min:
        df['API_norm'] = (df['API_t'] - a_min) / (a_max - a_min)
    else:
        df['API_norm'] = 0

    # Season_cos
    if 'timestamp' in df.columns:
        d = pd.to_datetime(df['timestamp']).dt.dayofyear
        df['S_t'] = np.cos((2 * np.pi * d) / 365)
    else:
        df['S_t'] = 0

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
            cols = ['timestamp', 'water_level_cm']
            for s in config['meteo_stations']:
                cols.extend([f'precip_{s}_mm', f'temp_{s}_c'])
            pd.DataFrame(columns=cols).to_csv(config["data_file"], index=False)

        # 2. Get Live Data
        live_wl = fetch_water_level_latest(config["hydro_station"])

        if live_wl is None:
            logger.warning("Could not fetch live water level. Aborting prediction.")
            return

        today_str = datetime.now().strftime("%Y-%m-%d")
        live_row = {'timestamp': today_str, 'water_level_cm': live_wl}

        for s in config['meteo_stations']:
            p, t = fetch_meteo_latest(s)
            live_row[f'precip_{s}_mm'] = p
            live_row[f'temp_{s}_c'] = t
            time.sleep(0.2)  # Polite delay

        # 3. Load History & Combine
        df_hist = pd.read_csv(config["data_file"])

        # Create DataFrame for live row
        df_live = pd.DataFrame([live_row])

        # Combine (Avoid Future Warning by filtering out empty/NA before concat if needed)
        df_comb = pd.concat([df_hist, df_live], ignore_index=True)

        # Remove duplicates based on timestamp (keep last)
        df_comb = df_comb.drop_duplicates(subset='timestamp', keep='last')

        # 4. Feature Extraction
        df_feats = prepare_complex_features(df_comb)

        # 5. Load Model & Predict
        # Load Model Config
        with open(config["config_json_path"], "r") as f:
            m_cfg = json.load(f)

        # Initialize Model Structure
        invardefs = [
            (f'x{i}', [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(m_cfg["num_mfs"])])
            for i in range(5)
        ]
        model = AnfisNet('Complex Flood Model', invardefs, ['y'], hybrid=True)

        # Load Weights
        ckpt = torch.load(config["anfis_model_path"], map_location="cpu")
        model.load_state_dict(ckpt['model_state_dict'])
        model.coeff = ckpt.get('consequent_coeffs') or ckpt.get('coeff')
        model.eval()

        # Scale Input
        # Vector order: [API_norm, S_t, SMI_t, Pt, delta_WL_t]
        last_features = df_feats.iloc[-1:][['API_norm', 'S_t', 'SMI_t', 'Pt', 'delta_WL_t']].values
        X_scaled = joblib.load(config["scaler_x_path"]).transform(last_features)

        # Predict
        with torch.no_grad():
            pred_tensor = model(torch.from_numpy(X_scaled).float())
            # Inverse Transform Output (Predicted Change)
            pred_chg = joblib.load(config["scaler_y_path"]).inverse_transform(pred_tensor.numpy())[0, 0]

        predicted_level = round(live_wl + pred_chg, 2)

        # 6. Save Log
        tomorrow_s = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        log_data = {}
        if os.path.exists(config["predictions_log_path"]):
            with open(config["predictions_log_path"], 'r') as f:
                try:
                    log_data = json.load(f)
                except json.JSONDecodeError:
                    log_data = {}

        # Update Today (Actual)
        log_data.setdefault(today_str, {})
        log_data[today_str]['actual'] = live_wl

        # Update Tomorrow (Prediction)
        log_data.setdefault(tomorrow_s, {})
        log_data[tomorrow_s]['predicted'] = predicted_level
        log_data[tomorrow_s]['report'] = f"Forecast: {predicted_level:.2f} cm (Change: {pred_chg:+.2f} cm)"

        with open(config["predictions_log_path"], 'w') as f:
            json.dump(log_data, f, indent=4)

        logger.info(f"✅ Prediction Archived: {predicted_level:.2f}")

    except Exception as e:
        logger.error(f"Prediction Job Failed: {e}", exc_info=True)


# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/data')
def get_data_api():
    river = request.args.get('river', 'minija')
    config = RIVER_CONFIGS.get(river)

    if not config:
        return jsonify({"error": "River config missing"}), 404

    if not os.path.exists(config["predictions_log_path"]):
        return jsonify({"error": "No data available yet. Please wait for the first run."}), 202

    try:
        with open(config["predictions_log_path"], 'r') as f:
            log_data = json.load(f)

        # Get sorted dates
        sorted_dates = sorted(log_data.keys())
        if not sorted_dates:
            return jsonify({"error": "Log file is empty"}), 202

        # Find the latest "Actual" data point
        last_actual_date = sorted_dates[-1]
        last_known_level = log_data[last_actual_date].get('actual', 0)

        # Sometimes the latest date in the log is tomorrow (only prediction), so look back 1 day for actual
        if log_data[last_actual_date].get('actual') is None and len(sorted_dates) > 1:
            last_actual_date = sorted_dates[-2]
            last_known_level = log_data[last_actual_date].get('actual', 0)

        # Find the latest "Predicted" data point (usually tomorrow)
        tomorrow_s = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        pred_entry = log_data.get(tomorrow_s, {})

        # If tomorrow's prediction isn't generated yet, try to find the last available prediction
        predicted_val = pred_entry.get('predicted')
        report_val = pred_entry.get('report', "Forecast pending...")

        if predicted_val is None:
            # Fallback: look at the very last entry in the log
            latest_entry = log_data.get(sorted_dates[-1], {})
            if latest_entry.get('predicted'):
                predicted_val = latest_entry.get('predicted')
                report_val = latest_entry.get('report')

        # Prepare Historical Data for Chart
        chart_data = []
        for d in sorted_dates:
            item = log_data[d]
            # Only send if we have at least one value
            if item.get('actual') is not None or item.get('predicted') is not None:
                chart_data.append({
                    "date": d,
                    "actual": item.get('actual'),
                    "predicted": item.get('predicted')
                })

        return jsonify({
            "riverName": config['display_name'],
            "lastKnownLevel": last_known_level,
            "predictedNextDayLevel": predicted_val if predicted_val else 0,
            "aiReport": report_val,
            "historicalData": chart_data[-30:],  # Last 30 days
            "lastUpdated": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


# --- SCHEDULER & STARTUP ---
scheduler = BackgroundScheduler()
# Run every hour at minute 5
scheduler.add_job(func=lambda: run_prediction_job(MINIJA_CONFIG), trigger="cron", minute="05")
scheduler.start()

if __name__ == "__main__":
    # TRIGGERS A RUN IMMEDIATELY ON STARTUP TO POPULATE THE FRONTEND
    # Wrap in try/except so a startup crash doesn't stop the server from booting
    try:
        if not os.path.exists(MINIJA_CONFIG["predictions_log_path"]):
            print("Running initial prediction...")
            run_prediction_job(MINIJA_CONFIG)
    except Exception as e:
        print(f"Startup prediction failed: {e}")

    app.run(debug=True, port=5003, use_reloader=False)
