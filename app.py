import json
import os
import time
import atexit
import logging
from datetime import datetime, timedelta
import warnings

# --- Load Environment Variables ---
from dotenv import load_dotenv

load_dotenv()

import joblib
import numpy as np
import pandas as pd
import requests
import torch
from flask import Flask, jsonify, render_template, request
from apscheduler.schedulers.background import BackgroundScheduler
from google import genai

# --- CUSTOM IMPORTS ---
from vector_engine import FloodVectorDB

# Ensure 'anfis' folder exists with __init__.py, anfis.py, membership.py
try:
    from anfis.anfis import AnfisNet
    from anfis.membership import BellMembFunc
except ImportError:
    print("⚠️ Warning: ANFIS modules not found.")

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig()
logging.getLogger('apscheduler').setLevel(logging.INFO)

app = Flask(__name__)

# --- LLM SETUP ---
GEMINI_MODEL = 'gemini-2.0-flash'
try:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
except Exception as e:
    print(f"LLM Error: {e}")
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


# --- DATA FETCHING ---

def fetch_water_level_latest(station_code):
    """Gets the absolute latest water level reading."""
    for days_back in [0, 1]:
        date = datetime.now().date() - timedelta(days=days_back)
        url = f"https://api.meteo.lt/v1/hydro-stations/{station_code}/observations/measured/{date}"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("observations", [])
                valid = [x for x in data if x.get('waterLevel') is not None]
                if valid:
                    valid.sort(key=lambda x: x['observationTimeUtc'])
                    return round(valid[-1]['waterLevel'], 2)
        except:
            pass
    return None


def fetch_precipitation_sum(station_code, date):
    """Fetches daily sum (Used for CSV logging)."""
    url = f"https://api.meteo.lt/v1/stations/{station_code}/observations/{date}"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            obs = resp.json().get("observations", [])
            return round(sum(o.get('precipitation', 0) for o in obs if o.get('precipitation')), 2)
    except:
        pass
    return 0


# --- FEATURE PREPARATION ---

def update_data_file(config):
    """Logs yesterday's data into CSV for history."""
    yesterday = datetime.now().date() - timedelta(days=1)
    if not os.path.exists(config['data_file']):
        cols = ['timestamp', 'water_level_cm'] + [f'precip_{c}_mm' for c in config['meteo_stations_codes']]
        pd.DataFrame(columns=cols).to_csv(config['data_file'], index=False)

    df = pd.read_csv(config['data_file'])
    if yesterday.strftime("%Y-%m-%d") in df['timestamp'].values: return

    wl = fetch_water_level_latest(config['hydro_station'])  # Simplification: using latest as daily avg approx
    precip = {f'precip_{c}_mm': fetch_precipitation_sum(c, yesterday) for c in config['meteo_stations_codes']}

    if wl is not None:
        row = {'timestamp': yesterday.strftime("%Y-%m-%d"), 'water_level_cm': wl, **precip}
        pd.DataFrame([row]).to_csv(config['data_file'], mode='a', header=False, index=False)


def prepare_features(df, config):
    """Calculates rolling averages (lags) for the ANFIS model."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.drop_duplicates('timestamp').set_index('timestamp').asfreq('D').ffill()

    for s in config['meteo_stations_short']:
        col = f'precip_{s}-ams_mm'
        if col in df.columns:
            # We calculate 24h (1 day) and 72h (3 days) rolling sums
            df[f'precip_{s}_lag_24h'] = df[col].rolling(window=1, min_periods=1).sum()
            df[f'precip_{s}_lag_72h'] = df[col].rolling(window=3, min_periods=1).sum()
    return df


# --- LLM REPORT ---

def generate_gemini_report(config, current, pred_anfis, pred_analog, risk):
    if not client: return "AI Analysis unavailable."

    prompt = f"""
    Act as a Hydrology Analyst for {config['display_name']}.
    Current Level: {current} cm.
    ANFIS Model Prediction (Mathematical): {pred_anfis:.1f} cm.
    Analog Model Prediction (Historical Pattern): {pred_analog} cm.
    Risk Level: {risk}.

    Compare the two models briefly. Give a 2-sentence advice to residents.
    """
    try:
        return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text
    except:
        return "AI Analysis failed."


# --- PREDICTION JOB (ANFIS + CHROMA) ---

def run_prediction_job(config):
    print(f"--- Running Job for {config['name']} ---")
    try:
        # 1. Update CSV History
        update_data_file(config)

        # 2. Update ChromaDB Index (Analog Model)
        vdb = FloodVectorDB(config['name'])
        vdb.update_index(config['data_file'])

        # 3. Load ANFIS Model
        with open(config["config_json_path"], "r") as f:
            model_cfg = json.load(f)
        scaler_X = joblib.load(config["scaler_x_path"])
        scaler_y = joblib.load(config["scaler_y_path"])
        model = AnfisNet('M', [(f'x{i}', [BellMembFunc(torch.tensor(1.), torch.tensor(1.), torch.tensor(1.)) for _ in
                                          range(model_cfg["num_mfs"])]) for i in range(model_cfg["num_inputs"])], ['y'],
                         hybrid=True)
        ckpt = torch.load(config["anfis_model_path"], map_location="cpu")
        model.load_state_dict(ckpt['model_state_dict'])
        model.coeff = ckpt['consequent_coeffs']
        model.eval()

        # 4. Prepare Live Data
        today = datetime.now().date()
        today_str = today.strftime("%Y-%m-%d")
        live_wl = fetch_water_level_latest(config["hydro_station"])
        live_precip = {f'precip_{c}_mm': fetch_precipitation_sum(c, today) for c in config['meteo_stations_codes']}

        if live_wl is None: return

        df = pd.read_csv(config["data_file"])
        live_row = pd.DataFrame([{'timestamp': today_str, 'water_level_cm': live_wl, **live_precip}])
        df_comb = pd.concat([df, live_row], ignore_index=True)

        # 5. Run ANFIS
        feats = prepare_features(df_comb, config)
        last_row_vals = feats.iloc[-1][model_cfg["features_list"]].values.reshape(1, -1)
        # Inject live water level if needed
        if "water_level_cm" in model_cfg["features_list"]:
            idx = model_cfg["features_list"].index("water_level_cm")
            last_row_vals[0, idx] = live_wl

        pred_change = \
        scaler_y.inverse_transform(model(torch.tensor(scaler_X.transform(last_row_vals)).float()).detach().numpy())[
            0, 0]
        pred_anfis = live_wl + pred_change

        # 6. Run Analog (Chroma)
        analog_pred_val = 0
        if len(df) >= 6:
            window = df['water_level_cm'].tail(6).astype(float).tolist()
            match = vdb.find_match(window)
            if match: analog_pred_val = match['predicted_level']

        # 7. Risk & Report
        risk = "LOW"
        if pred_anfis >= config['risk_levels'][2]:
            risk = "SEVERE"
        elif pred_anfis >= config['risk_levels'][1]:
            risk = "HIGH"
        elif pred_anfis >= config['risk_levels'][0]:
            risk = "MODERATE"

        report = generate_gemini_report(config, live_wl, pred_anfis, analog_pred_val, risk)

        # 8. Save Logs
        tomorrow_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        log = {}
        if os.path.exists(config["predictions_log_path"]):
            with open(config["predictions_log_path"], 'r') as f: log = json.load(f)

        log.setdefault(today_str, {})['actual'] = live_wl
        log.setdefault(tomorrow_str, {})['predicted'] = pred_anfis
        log.setdefault(tomorrow_str, {})['report'] = report

        with open(config["predictions_log_path"], 'w') as f:
            json.dump(log, f, indent=4)
        print(f"✅ {config['name']} Updated.")

    except Exception as e:
        print(f"❌ Error {config['name']}: {e}")


# --- SCHEDULER ---
def scheduled_task():
    run_prediction_job(MINIJA_CONFIG)
    run_prediction_job(DANE_CONFIG)


scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_task, 'cron', minute=5)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())


# --- API ROUTES ---

@app.route('/')
def index(): return render_template('index.html')


@app.route('/api/run_predictions', methods=['POST'])
def api_run():
    scheduled_task()
    return jsonify({"status": "ok"})


@app.route('/api/data')
def api_data():
    river = request.args.get('river', 'minija')
    config = RIVER_CONFIGS.get(river)
    if not config: return jsonify({"error": "Invalid river"}), 404

    try:
        today = datetime.now().date()
        today_str = today.strftime("%Y-%m-%d")
        tomorrow_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")

        # 1. Fetch Real-time
        wl = fetch_water_level_latest(config['hydro_station'])

        # 2. Get Logs
        log = {}
        if os.path.exists(config["predictions_log_path"]):
            with open(config["predictions_log_path"]) as f: log = json.load(f)

        if wl is None: wl = log.get(today_str, {}).get('actual', 0)
        pred_val = log.get(tomorrow_str, {}).get('predicted', 0)
        report = log.get(tomorrow_str, {}).get('report', "No report.")

        # 3. Get Analog Match (Live)
        vdb = FloodVectorDB(config['name'])
        df = pd.read_csv(config['data_file'])
        analog_res = None
        if len(df) >= 6:
            # We must use the CSV data for the window to match the index
            window = df['water_level_cm'].tail(6).astype(float).tolist()
            analog_res = vdb.find_match(window)

        # 4. Get Rain Features (Live calc for UI)
        live_precip = {f'precip_{c}_mm': fetch_precipitation_sum(c, today) for c in config['meteo_stations_codes']}
        live_row = pd.DataFrame([{'timestamp': today_str, 'water_level_cm': wl, **live_precip}])
        df_comb = pd.concat([df, live_row], ignore_index=True)
        feat_df = prepare_features(df_comb, config)
        last_row = feat_df.iloc[-1]

        feats = {}
        for s in config['meteo_stations_short']:
            feats[f"{s}_24h"] = float(last_row.get(f'precip_{s}_lag_24h', 0))
            feats[f"{s}_72h"] = float(last_row.get(f'precip_{s}_lag_72h', 0))

        # 5. History for Chart
        hist = []
        for i in range(30):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            if d in log: hist.append({'date': d, **log[d]})
        hist.append({'date': tomorrow_str, **log.get(tomorrow_str, {})})
        hist.sort(key=lambda x: x['date'])

        # Risk
        risk = "LOW"
        if pred_val >= config['risk_levels'][2]:
            risk = "SEVERE"
        elif pred_val >= config['risk_levels'][1]:
            risk = "HIGH"
        elif pred_val >= config['risk_levels'][0]:
            risk = "MODERATE"

        return jsonify({
            "riverName": config['display_name'],
            "lastKnownLevel": wl,
            "predictedNextDayLevel": pred_val,
            "analogForecast": analog_res,
            "aiReport": report,
            "liveFeatures": feats,
            "historicalData": hist,
            "risk": {"level": risk},
            "trend": {"text": "Rising" if pred_val > wl + 10 else "Falling" if pred_val < wl - 10 else "Stable"},
            "lastUpdated": datetime.now().isoformat()
        })
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)