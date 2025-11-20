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
from flask import Flask, jsonify, render_template, request

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
GEMINI_MODEL = 'gemini-2.0-flash'
try:
    # REPLACE "os.environ.get..." WITH YOUR ACTUAL KEY STRING IF NEEDED
    # e.g. client = genai.Client(api_key="AIzaSy...")
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
except Exception as e:
    print(f"LLM Initialization Warning: {e}")
    client = None

# --- CONFIGURATION ---
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
    "risk_levels": [250, 400, 550]
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

RIVER_CONFIGS = {"minija": MINIJA_CONFIG, "dane": DANE_CONFIG}


# --- Data & Feature Logic ---

def fetch_recent_water_level(station_code, date):
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.meteo.lt/v1/hydro-stations/{station_code}/observations/measured/{date_str}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        water_levels = [obs['waterLevel'] for obs in observations if obs.get('waterLevel') is not None]
        if water_levels: return round(sum(water_levels) / len(water_levels), 2)
    except Exception as e:
        print(f"   - Error water level: {e}")
    return None


def fetch_recent_precipitation(station_code, date):
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.meteo.lt/v1/stations/{station_code}/observations/{date_str}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        return round(sum(obs.get('precipitation', 0) for obs in observations if obs.get('precipitation') is not None),
                     2)
    except Exception as e:
        print(f"   - Error precipitation: {e}")
    return 0


def update_data_file(config: dict):
    yesterday = datetime.now().date() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    data_file = config['data_file']

    if os.path.exists(data_file):
        df_existing = pd.read_csv(data_file)
        if yesterday_str in df_existing['timestamp'].values: return
    else:
        cols = ['timestamp', 'water_level_cm'] + [f'precip_{c}_mm' for c in config['meteo_stations_codes']]
        pd.DataFrame(columns=cols).to_csv(data_file, index=False)

    print(f"--- Collecting data for {config['name']} ---")
    wl = fetch_recent_water_level(config['hydro_station'], yesterday)
    precip = {f'precip_{c}_mm': fetch_recent_precipitation(c, yesterday) for c in config['meteo_stations_codes']}

    if wl is None: return
    pd.DataFrame([{'timestamp': yesterday_str, 'water_level_cm': wl, **precip}]).to_csv(data_file, mode='a',
                                                                                        header=False, index=False)


def prepare_features(df, config):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.drop_duplicates(subset='timestamp', keep='last').set_index('timestamp').asfreq('D').ffill()
    for s in config['meteo_stations_short']:
        c = f'precip_{s}-ams_mm'
        if c in df.columns:
            for win, lag in [(1, '12h'), (1, '24h'), (2, '48h'), (3, '72h')]:
                df[f'precip_{s}_lag_{lag}'] = df[c].rolling(window=win, min_periods=1).sum()
    return df


def build_anfis(num_inputs, num_mfs):
    invardefs = [(f'x{i}', [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(num_mfs)]) for i in
                 range(num_inputs)]
    return AnfisNet('Model', invardefs, ['y'], hybrid=True)


# --- LLM Generation ---
def generate_gemini_report(config, current_level, predicted_change, projected_level, rain_24h):
    if client is None: return "AI Analysis unavailable (Client not initialized)."

    lvl_mod, lvl_high, lvl_severe = config['risk_levels']
    risk_level = "LOW"
    if projected_level >= lvl_severe:
        risk_level = "SEVERE"
    elif projected_level >= lvl_high:
        risk_level = "HIGH"
    elif projected_level >= lvl_mod:
        risk_level = "MODERATE"

    prompt = f"""
    Act as a Hydrology Analyst for the {config['display_name']} river.
    Data:
    - Current Level: {current_level:.2f} cm
    - 24h Rain: {rain_24h:.2f} mm
    - Forecast Change: {predicted_change:+.2f} cm
    - Forecast Level: {projected_level:.2f} cm
    - Risk Category: {risk_level}

    Write a concise 2-sentence update for local residents. Mention the trend (rising/falling) and safety advice if needed.
    """
    try:
        res = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return res.text
    except Exception as e:
        return f"AI Analysis unavailable: {str(e)}"


# --- Prediction Job ---
def run_prediction_job(config):
    print(f"--- Prediction: {config['name']} ---")
    try:
        with open(config["config_json_path"], "r") as f:
            model_config = json.load(f)
        scaler_X = joblib.load(config["scaler_x_path"])
        scaler_y = joblib.load(config["scaler_y_path"])
        model = build_anfis(model_config["num_inputs"], model_config["num_mfs"])
        ckpt = torch.load(config["anfis_model_path"], map_location="cpu")
        model.load_state_dict(ckpt['model_state_dict'])
        model.coeff = ckpt['consequent_coeffs']
        model.eval()

        today = datetime.now().date()
        live_wl = fetch_recent_water_level(config["hydro_station"], today)
        live_precip = {f'precip_{c}_mm': fetch_recent_precipitation(c, today) for c in config['meteo_stations_codes']}

        if live_wl is None: return

        df_hist = pd.read_csv(config["data_file"])
        live_row = pd.DataFrame([{'timestamp': today.strftime("%Y-%m-%d"), 'water_level_cm': live_wl, **live_precip}])
        df_comb = pd.concat([df_hist, live_row], ignore_index=True)
        features_df = prepare_features(df_comb, config)

        last_row = features_df.iloc[-1:]
        X_unscaled = last_row[model_config["features_list"]].values
        if np.isnan(X_unscaled).any(): return

        X_scaled = scaler_X.transform(X_unscaled)
        with torch.no_grad():
            pred_chg = scaler_y.inverse_transform(model(torch.from_numpy(X_scaled).float()).numpy())[0, 0]

        pred_level = live_wl + pred_chg

        # Generate Report
        rain_col = [c for c in model_config["features_list"] if 'lag_24h' in c]
        rain_24h = last_row[rain_col[0]].iloc[0] if rain_col else 0
        report = generate_gemini_report(config, live_wl, pred_chg, pred_level, rain_24h)

        # Log
        today_str = today.strftime("%Y-%m-%d")
        tomorrow_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        log_data = {}
        if os.path.exists(config["predictions_log_path"]):
            with open(config["predictions_log_path"], 'r') as f: log_data = json.load(f)

        log_data.setdefault(today_str, {})['actual'] = live_wl
        log_data.setdefault(tomorrow_str, {})['predicted'] = pred_level
        log_data.setdefault(tomorrow_str, {})['report'] = report  # Save LLM report

        with open(config["predictions_log_path"], 'w') as f:
            json.dump(log_data, f, indent=4)
        print(f"✅ {config['name']} Updated. Report generated.")

    except Exception as e:
        print(f"❌ Error {config['name']}: {e}")


# --- Routes ---
@app.route('/')
def index(): return render_template('index.html')


@app.route('/api/run_predictions', methods=['POST'])
def run_predictions_api():
    update_data_file(MINIJA_CONFIG)
    update_data_file(DANE_CONFIG)
    run_prediction_job(MINIJA_CONFIG)
    run_prediction_job(DANE_CONFIG)
    return jsonify({"status": "ok"})


@app.route('/api/data')
def get_data_api():
    river = request.args.get('river', 'minija')
    config = RIVER_CONFIGS.get(river)
    if not config: return jsonify({"error": "Invalid river"}), 404

    try:
        log_data = {}
        if os.path.exists(config["predictions_log_path"]):
            with open(config["predictions_log_path"], 'r') as f: log_data = json.load(f)

        today = datetime.now().date()
        today_str = today.strftime("%Y-%m-%d")
        tomorrow_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")

        pred_level = log_data.get(tomorrow_str, {}).get('predicted', 0)
        ai_report = log_data.get(tomorrow_str, {}).get('report', "No report available.")
        actual_level = log_data.get(today_str, {}).get('actual', 0)

        # Re-read live features for UI (simplified)
        df = pd.read_csv(config["data_file"])
        df_feats = prepare_features(df, config)
        last_row = df_feats.iloc[-1:]

        feats = {}
        for s in config['meteo_stations_short']:
            feats[f"precip_{s}_lag_24h"] = float(last_row.get(f'precip_{s}_lag_24h', 0))
            feats[f"precip_{s}_lag_72h"] = float(last_row.get(f'precip_{s}_lag_72h', 0))

        lvl_mod, lvl_high, lvl_severe = config['risk_levels']
        risk = 'LOW'
        if pred_level >= lvl_severe:
            risk = 'SEVERE'
        elif pred_level >= lvl_high:
            risk = 'HIGH'
        elif pred_level >= lvl_mod:
            risk = 'MODERATE'

        trend = 'Rising' if (pred_level - actual_level) > 10 else 'Falling' if (
                                                                                           pred_level - actual_level) < -10 else 'Stable'

        hist_data = []
        for i in range(30):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            if d in log_data: hist_data.append({'date': d, **log_data[d]})
        hist_data.append({'date': tomorrow_str, **log_data.get(tomorrow_str, {})})
        hist_data.sort(key=lambda x: x['date'])

        return jsonify({
            "riverName": config['display_name'],
            "lastKnownLevel": actual_level,
            "predictedNextDayLevel": pred_level,
            "aiReport": ai_report,
            "liveFeatures": feats,
            "historicalData": hist_data,
            "risk": {"level": risk},
            "trend": {"text": trend},
            "lastUpdated": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5002)