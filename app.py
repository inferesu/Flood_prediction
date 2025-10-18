import json
import os
import time
from datetime import datetime, timedelta
from collections import deque

import joblib
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from flask import Flask, redirect, render_template, request, url_for

# --- Import ANFIS and RNN classes ---
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration & File Paths ---
DATA_FILE = '2025new_test.csv'
ANFIS_MODEL_PATH = "anfis_model.pth"
RNN_MODEL_PATH = "rnn_corrector_multivariate.pth"
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_Y.pkl"
SCALER_ERROR_PATH = "scaler_error.pkl"
CONFIG_JSON_PATH = "training_config.json"

# --- Data Fetching Configuration ---
STATION_CODE_HYDRO = 'priekules-vms'
STATION_CODE_METEO_1 = 'klaipedos-ams'
STATION_CODE_METEO_2 = 'vezaiciu-ams'
NUM_DAYS_TO_FETCH = 5


# --- Data Fetching Logic (No changes here) ---
def fetch_recent_water_level(station_code, date):
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.meteo.lt/v1/hydro-stations/{station_code}/observations/measured/{date_str}"
    print(f"-> Fetching water level for {date_str}...")
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            observations = resp.json().get("observations", [])
            water_levels = [obs['waterLevel'] for obs in observations if obs.get('waterLevel') is not None]
            if water_levels:
                avg_level = round(sum(water_levels) / len(water_levels), 2)
                print(f"   ✅ Success. Avg water level: {avg_level:.2f} cm")
                return avg_level
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
            daily_precip = sum(
                obs.get('precipitation', 0) for obs in observations if obs.get('precipitation') is not None)
            print(f"   ✅ Success. Total precipitation: {daily_precip:.2f} mm")
            return round(daily_precip, 2)
    except Exception as e:
        print(f"   - ❌ Error fetching precipitation: {e}")
    return 0


def update_and_overwrite_data():
    print(f"--- Starting data collection, including today's partial data ---")
    all_days_data = []
    for days_ago in range(NUM_DAYS_TO_FETCH - 1, -1, -1):
        current_date = datetime.now().date() - timedelta(days=days_ago)
        water_level = fetch_recent_water_level(STATION_CODE_HYDRO, current_date)
        time.sleep(0.5)
        precip1 = fetch_recent_precipitation(STATION_CODE_METEO_1, current_date)
        time.sleep(0.5)
        precip2 = fetch_recent_precipitation(STATION_CODE_METEO_2, current_date)

        if water_level is not None:
            daily_data = {'timestamp': current_date.strftime("%Y-%m-%d"), 'water_level_cm': water_level,
                          'precip_klaipedos-ams_mm': precip1, 'precip_vezaiciu-ams_mm': precip2}
            all_days_data.append(daily_data)
        else:
            print(f"   ⚠️ Skipping {current_date.strftime('%Y-%m-%d')} due to missing water level data.")

    if all_days_data:
        df_final = pd.DataFrame(all_days_data)
        df_final.to_csv(DATA_FILE, index=False)
        print(f"\n✅ --- Data file '{DATA_FILE}' updated successfully. ---")
    else:
        print("\n❌ --- Data collection failed. File not updated. ---")


# --- Model Definitions & Feature Engineering (No changes here) ---
def prepare_features_for_training(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_index(inplace=True)
    df = df.asfreq('H').interpolate(method='time')
    df['precip_klaipedos_lag_12h'] = df['precip_klaipedos-ams_mm'].rolling(12, min_periods=1).sum()
    df['precip_klaipedos_lag_24h'] = df['precip_klaipedos-ams_mm'].rolling(24, min_periods=1).sum()
    df['precip_klaipedos_lag_48h'] = df['precip_klaipedos-ams_mm'].rolling(48, min_periods=1).sum()
    df['precip_klaipedos_lag_72h'] = df['precip_klaipedos-ams_mm'].rolling(72, min_periods=1).sum()
    df['precip_vezaiciu_lag_12h'] = df['precip_vezaiciu-ams_mm'].rolling(12, min_periods=1).sum()
    df['precip_vezaiciu_lag_24h'] = df['precip_vezaiciu-ams_mm'].rolling(24, min_periods=1).sum()
    df['precip_vezaiciu_lag_48h'] = df['precip_vezaiciu-ams_mm'].rolling(48, min_periods=1).sum()
    df['precip_vezaiciu_lag_72h'] = df['precip_vezaiciu-ams_mm'].rolling(72, min_periods=1).sum()
    df['target_change'] = df['water_level_cm'].shift(-1) - df['water_level_cm']
    df.dropna(inplace=True)
    return df


def prepare_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_index(inplace=True)
    df = df.asfreq('H').interpolate(method='time')
    df['precip_klaipedos_lag_12h'] = df['precip_klaipedos-ams_mm'].rolling(12, min_periods=1).sum()
    df['precip_klaipedos_lag_24h'] = df['precip_klaipedos-ams_mm'].rolling(24, min_periods=1).sum()
    df['precip_klaipedos_lag_48h'] = df['precip_klaipedos-ams_mm'].rolling(48, min_periods=1).sum()
    df['precip_klaipedos_lag_72h'] = df['precip_klaipedos-ams_mm'].rolling(72, min_periods=1).sum()
    df['precip_vezaiciu_lag_12h'] = df['precip_vezaiciu-ams_mm'].rolling(12, min_periods=1).sum()
    df['precip_vezaiciu_lag_24h'] = df['precip_vezaiciu-ams_mm'].rolling(24, min_periods=1).sum()
    df['precip_vezaiciu_lag_48h'] = df['precip_vezaiciu-ams_mm'].rolling(48, min_periods=1).sum()
    df['precip_vezaiciu_lag_72h'] = df['precip_vezaiciu-ams_mm'].rolling(72, min_periods=1).sum()
    return df


def build_anfis(num_inputs: int, num_mfs: int) -> AnfisNet:
    invardefs = []
    for i in range(num_inputs):
        mfs = [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(num_mfs)]
        invardefs.append((f'x{i}', mfs))
    return AnfisNet('Flood Prediction Model', invardefs, ['y'], hybrid=True)


class ErrorCorrectorRNN(nn.Module):
    def __init__(self, input_size, hidden_size=60, num_layers=2, output_size=1):
        super(ErrorCorrectorRNN, self).__init__()
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


# --- Prediction Logic (With the date fix) ---
def get_latest_prediction():
    with open(CONFIG_JSON_PATH, "r") as f:
        config = json.load(f)
    features_list, num_features = config["features_list"], len(config["features_list"])
    scaler_X, scaler_y, scaler_error = joblib.load(SCALER_X_PATH), joblib.load(SCALER_Y_PATH), joblib.load(
        SCALER_ERROR_PATH)
    anfis_model = build_anfis(config["num_inputs"], config["num_mfs"])
    checkpoint = torch.load(ANFIS_MODEL_PATH, map_location="cpu")
    anfis_model.load_state_dict(checkpoint['model_state_dict'])
    anfis_model.coeff = checkpoint['consequent_coeffs']
    anfis_model.eval()
    rnn_model = ErrorCorrectorRNN(input_size=1 + num_features, hidden_size=60, num_layers=2)
    rnn_model.load_state_dict(torch.load(RNN_MODEL_PATH, map_location="cpu"))
    rnn_model.eval()

    try:
        df_raw = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
        df_raw.set_index('timestamp', inplace=True)
        df_hourly = df_raw.asfreq('H').ffill()
        df_features = prepare_features_for_prediction(df_hourly.copy())
        df_for_rnn_context = prepare_features_for_training(df_hourly.copy())
        if df_features.empty: return "Not enough data", "N/A", f"{df_raw.iloc[-1]['water_level_cm']:.2f} cm"
    except FileNotFoundError:
        return "Data file not found. Click the button to create it.", "N/A", "N/A"
    except Exception as e:
        print(f"An error during data preparation: {e}")
        return "Prep Error", "N/A", "N/A"

    X_pred_unscaled = df_features[features_list].iloc[-1:].values
    X_pred_scaled = scaler_X.transform(X_pred_unscaled)
    X_pred_tensor = torch.from_numpy(X_pred_scaled).float()
    X_context_unscaled = df_for_rnn_context[features_list].values
    y_context_true_change = df_for_rnn_context["target_change"].values
    SEQUENCE_LENGTH, ERROR_MONITORING_WINDOW, ADAPTIVE_ERROR_THRESHOLD, MAX_CORRECTION_ABS = 24, 12, 15.0, 25.0
    context_history = deque([np.zeros(1 + num_features)] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
    anfis_error_history = deque([0.0] * ERROR_MONITORING_WINDOW, maxlen=ERROR_MONITORING_WINDOW)

    with torch.no_grad():
        historical_anfis_preds = scaler_y.inverse_transform(
            anfis_model(torch.from_numpy(scaler_X.transform(X_context_unscaled)).float()).numpy()).flatten()
        for i in range(len(df_for_rnn_context)):
            anfis_prediction, true_anfis_error = historical_anfis_preds[i], y_context_true_change[i] - \
                                                                            historical_anfis_preds[i]
            anfis_error_history.append(true_anfis_error)
            new_context = np.concatenate([[y_context_true_change[i] - anfis_prediction], X_context_unscaled[i]])
            context_history.append(new_context)
        final_anfis_pred = scaler_y.inverse_transform(anfis_model(X_pred_tensor).numpy()).flatten()[0]
        final_correction = 0.0
        if np.mean(np.abs(list(anfis_error_history))) > ADAPTIVE_ERROR_THRESHOLD:
            context_np = np.array(context_history)
            context_to_scale = context_np.copy()
            context_to_scale[:, 0] = scaler_error.transform(context_to_scale[:, 0].reshape(-1, 1)).flatten()
            context_to_scale[:, 1:] = scaler_X.transform(context_to_scale[:, 1:])
            context_tensor = torch.from_numpy(context_to_scale).float().unsqueeze(0)
            predicted_correction = scaler_error.inverse_transform([[rnn_model(context_tensor).item()]])[0, 0]
            final_correction = np.clip(predicted_correction * 0.5, -MAX_CORRECTION_ABS, MAX_CORRECTION_ABS)

    current_level_value = df_features['water_level_cm'].iloc[-1]
    predicted_change = final_anfis_pred + final_correction
    final_predicted_level = current_level_value + predicted_change

    # --- THE FIX: Add one DAY, not one hour, to get tomorrow's date. ---
    prediction_date = df_features.index[-1] + timedelta(days=1)

    return (f"{final_predicted_level:.2f} cm", prediction_date.strftime('%Y-%m-%d %H:%M'),
            f"{current_level_value:.2f} cm")


# --- Flask Web Routes (No changes here) ---
@app.route('/')
def index():
    prediction = request.args.get('prediction', 'Click the button')
    pred_date = request.args.get('date', 'to get a prediction')
    current_level = request.args.get('current_level', 'N/A')
    return render_template('index.html', prediction_result=prediction, prediction_date=pred_date,
                           current_level_result=current_level)


@app.route('/predict')
def predict():
    update_and_overwrite_data()
    prediction, pred_date, current_level = get_latest_prediction()
    return redirect(url_for('index', prediction=prediction, date=pred_date, current_level=current_level))


# --- Main execution ---
if __name__ == "__main__":
    app.run(debug=True)