import json
import os
import time
import logging
import warnings
from datetime import datetime, timedelta
from dotenv import load_dotenv

import joblib
import numpy as np
import pandas as pd
import requests
import torch
from flask import Flask, jsonify, render_template, request
from apscheduler.schedulers.background import BackgroundScheduler

from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

load_dotenv()
warnings.filterwarnings("ignore")
logger = logging.getLogger('flood_app')
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

K_DECAY = 0.85

# --- MULTI-HORIZON CONFIGURATION ---
MINIJA_CONFIG = {
    "name": "minija",
    "display_name": "Minija (Priekulė)",
    "data_file": "live_data_complex.csv",
    "predictions_log_path": "predictions_log.json",
    "hydro_station": "priekules-vms",
    "lat": 55.5546,
    "lon": 21.3195,
    "meteo_stations": ["klaipedos-ams", "vezaiciu-ams"],
    "risk_levels": [250, 400, 550],
    "horizons": {
        "1d": {
            "model": "anfis_model.pth",
            "scaler_x": "scaler_X.pkl",
            "scaler_y": "scaler_Y.pkl",
            "config": "training_config.json",
            "label": "1-Day Forecast",
            "days_ahead": 1
        },
        "3d": {
            "model": "anfis_model_3d.pth",
            "scaler_x": "scaler_X_3d.pkl",
            "scaler_y": "scaler_y_3d.pkl",
            "config": "training_config_3d.json",
            "label": "3-Day Forecast",
            "days_ahead": 3
        },
        "5d": {
            "model": "anfis_model_5d.pth",
            "scaler_x": "scaler_X_5d.pkl",
            "scaler_y": "scaler_y_5d.pkl",
            "config": "training_config_5d.json",
            "label": "5-Day Forecast",
            "days_ahead": 5
        }
    }
}

RIVER_CONFIGS = {"minija": MINIJA_CONFIG}

# [MF_LABELS remains the same as your previous script]
MF_LABELS = {
    "API_norm": {0: "Low", 1: "Medium", 2: "Extreme", 3: "High", 4: "Very High"},
    "S_t": {0: "Late Winter", 1: "Mid Spring", 2: "Late Jan", 3: "Mid Spring", 4: "Early Spring"},
    "SMI_t": {0: "Extreme", 1: "Very High", 2: "Medium", 3: "Low", 4: "High"},
    "Pt": {0: "High", 1: "Medium", 2: "Extreme", 3: "Very High", 4: "Low"},
    "delta_WL_t": {0: "Very High", 1: "High", 2: "Medium", 3: "Extreme", 4: "Low"}
}


# --- DATA FETCHING & FEATURES ---
# [fetch_water_level_latest, fetch_meteo_latest, prepare_complex_features remain the same]
def fetch_water_level_latest(station_code):
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


def prepare_complex_features(df):
    df = df.copy().sort_values('timestamp')
    p_cols = [c for c in df.columns if 'precip' in c]
    df['Pt'] = df[p_cols].mean(axis=1) if p_cols else 0
    api_vals, current_api = [], 0
    for p in df['Pt']:
        current_api = p + (K_DECAY * current_api)
        api_vals.append(current_api)
    df['API_t'] = api_vals
    a_min, a_max = df['API_t'].min(), df['API_t'].max()
    df['API_norm'] = (df['API_t'] - a_min) / (a_max - a_min) if a_max != a_min else 0
    d = pd.to_datetime(df['timestamp']).dt.dayofyear
    df['S_t'] = np.cos((2 * np.pi * d) / 365)
    t_cols = [c for c in df.columns if 'temp' in c]
    if t_cols:
        avg_temp = df[t_cols].mean(axis=1)
        df['SMI_t'] = avg_temp.apply(lambda x: max(0, x * 2.5) if x > 0 else 0)
    else:
        df['SMI_t'] = 0
    df['delta_WL_t'] = df['water_level_cm'].diff().fillna(0)
    return df


def extract_strongest_rule(model, X_scaled, num_mfs):
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled).float()
        fuzzified = model.layer['fuzzify'](X_tensor)
        firing_strengths = model.layer['rules'](fuzzified)
    strengths = firing_strengths[0].numpy()
    rule_id = int(np.argmax(strengths))
    activation = float(strengths[rule_id])
    num_inputs = 5
    mf_indices = []
    temp = rule_id
    for _ in range(num_inputs):
        mf_indices.append(temp % num_mfs)
        temp //= num_mfs
    return rule_id, activation, list(reversed(mf_indices))


# --- UPDATED MULTI-PREDICTION JOB ---
def run_prediction_job(config):
    logger.info(f"--- Running Prediction Cycle for {config['display_name']} ---")
    try:
        live_wl = fetch_water_level_latest(config["hydro_station"])
        if live_wl is None: return

        # Prepare live row
        today_str = datetime.now().strftime("%Y-%m-%d")
        live_row = {'timestamp': today_str, 'water_level_cm': live_wl}
        for s in config['meteo_stations']:
            p, t = fetch_meteo_latest(s)
            live_row[f'precip_{s}_mm'], live_row[f'temp_{s}_c'] = p, t

        # History logic
        df_hist = pd.read_csv(config["data_file"]) if os.path.exists(config["data_file"]) else pd.DataFrame()
        df_comb = pd.concat([df_hist, pd.DataFrame([live_row])], ignore_index=True).drop_duplicates('timestamp')
        df_comb.to_csv(config["data_file"], index=False)
        df_feats = prepare_complex_features(df_comb)

        last_row = df_feats.iloc[-1]
        feature_display = {
            "Water Level": f"{live_wl} cm",
            "Precipitation (Pt)": f"{round(last_row['Pt'], 2)} mm",
            "Soil Saturation (API)": f"{round(last_row['API_t'], 2)} idx",
            "Snowmelt (SMI)": f"{round(last_row['SMI_t'], 2)} mm",
            "Seasonality": f"{round(last_row['S_t'], 3)}",
            "Trend": f"{round(last_row['delta_WL_t'], 2)} cm"
        }

        # Predict for each horizon
        all_preds = {}
        dominant_rule = ""

        input_data = last_row[['API_norm', 'S_t', 'SMI_t', 'Pt', 'delta_WL_t']].values.reshape(1, -1)

        for h_id, h_cfg in config["horizons"].items():
            # Load specific resources
            scaler_x = joblib.load(h_cfg["scaler_x"])
            scaler_y = joblib.load(h_cfg["scaler_y"])
            with open(h_cfg["config"], "r") as f:
                num_mfs = json.load(f).get("num_mfs", 5)

            X_scaled = scaler_x.transform(input_data)

            # Model Reconstruction
            invardefs = [(f'x{i}', [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(num_mfs)])
                         for i in range(5)]
            model = AnfisNet(f'ANFIS_{h_id}', invardefs, ['y'], hybrid=True)
            ckpt = torch.load(h_cfg["model"], map_location="cpu")
            model.load_state_dict(ckpt['model_state_dict'])
            model.coeff = ckpt.get('consequent_coeffs') or ckpt.get('coeff')
            model.eval()

            with torch.no_grad():
                pred_chg = scaler_y.inverse_transform(model(torch.from_numpy(X_scaled).float()).numpy())[0, 0]

            pred_chg = float(pred_chg)

            all_preds[h_id] = {
                "level": float(round(live_wl + pred_chg, 2)),
                "change": float(round(pred_chg, 2)),
                "label": h_cfg["label"]
            }

            # Capture 1d rule as the primary explanation
            if h_id == "1d":
                r_id, act, mf_i = extract_strongest_rule(model, X_scaled, num_mfs)
                parts = [f"{fn} is {MF_LABELS[fn][mf_i[i]]}" for i, fn in
                         enumerate(['API_norm', 'S_t', 'SMI_t', 'Pt', 'delta_WL_t'])]
                dominant_rule = f"Rule #{r_id} (Act: {act:.2f}): IF " + " AND ".join(parts)

        # Log Data
        log_data = {}
        # --- SAFE MULTI-HORIZON LOGGING ---

        log_path = config["predictions_log_path"]
        tmp_path = log_path + ".tmp"

        # Load existing log safely
        log_data = {}
        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    log_data = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Corrupted log detected. Resetting log file.")
                log_data = {}

        # Prepare entry for today
        log_data[today_str] = {
            "actual": float(live_wl),
            "features": feature_display,
            "fired_rule": dominant_rule,
            "horizons": all_preds
        }

        # Atomic write (prevents corruption)
        with open(tmp_path, "w") as f:
            json.dump(log_data, f, indent=4)

        os.replace(tmp_path, log_path)

    except Exception as e:
        logger.error(f"Job Failed: {e}", exc_info=True)


@app.route('/')
def index(): return render_template('index.html')


@app.route('/api/data')
def get_data_api():
    river = request.args.get('river', 'minija')
    config = RIVER_CONFIGS.get(river)

    path = config["predictions_log_path"]

    if not os.path.exists(path):
        return jsonify({"error": "No data"}), 202

    try:
        with open(path, "r") as f:
            log_data = json.load(f)
    except json.JSONDecodeError:
        return jsonify({"error": "Corrupted log"}), 500

    sorted_dates = sorted(log_data.keys())

    last_date = sorted_dates[-1]
    last_entry = log_data[last_date]

    chart_data = []

    for d in sorted_dates:
        entry = log_data[d]

        chart_data.append({
            "date": d,
            "actual": entry.get("actual"),
            "pred1d": entry.get("horizons", {}).get("1d", {}).get("level"),
            "pred3d": entry.get("horizons", {}).get("3d", {}).get("level"),
            "pred5d": entry.get("horizons", {}).get("5d", {}).get("level")
        })

    return jsonify({
        "riverName": config['display_name'],
        "lat": config["lat"],
        "lon": config["lon"],
        "lastKnownLevel": last_entry.get("actual"),
        "horizons": last_entry.get("horizons"),
        "currentFeatures": last_entry.get("features"),
        "firedRule": last_entry.get("fired_rule"),
        "historicalData": chart_data[-30:],
        "lastUpdated": datetime.now().isoformat()
    })


scheduler = BackgroundScheduler()
scheduler.add_job(func=lambda: run_prediction_job(MINIJA_CONFIG), trigger="cron", minute="05")
scheduler.start()

if __name__ == "__main__":
    run_prediction_job(MINIJA_CONFIG)
    app.run(debug=True, port=5000, use_reloader=False)