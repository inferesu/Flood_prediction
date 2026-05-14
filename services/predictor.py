"""Core prediction pipeline for each river station."""

import json
import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch

from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc
from services.data_fetcher import fetch_water_level_latest, fetch_meteo_latest
from services.features import (
    prepare_complex_features,
    extract_strongest_rule,
    format_rule_string,
)

logger = logging.getLogger("flood_app")


def _load_model(h_cfg: dict, num_mfs: int, horizon_id: str, display_name: str):
    """Load and return an ANFIS model from checkpoint."""
    centres = torch.linspace(0.1, 0.9, num_mfs)
    invardefs = [
        (
            f"x{i}",
            [
                BellMembFunc(
                    torch.tensor([1.0]),
                    torch.tensor([2.0]),
                    centres[j].unsqueeze(0),
                )
                for j in range(num_mfs)
            ],
        )
        for i in range(5)
    ]

    model = AnfisNet(f"ANFIS_{horizon_id}", invardefs, ["y"], hybrid=True)
    ckpt = torch.load(h_cfg["model"], map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    coeff = ckpt.get("coeff") or ckpt.get("consequent_coeffs")
    if coeff is None:
        logger.error(f"[{display_name}] {horizon_id}: no coeff in checkpoint.")
        return None

    if isinstance(coeff, np.ndarray):
        coeff = torch.tensor(coeff, dtype=torch.float32)
    model.coeff = coeff
    model.eval()
    return model


def run_prediction_job(config: dict):
    """Execute a full prediction cycle for a single river config."""
    display = config["display_name"]
    logger.info(f"--- Running Prediction Cycle for {display} ---")

    try:
        # 1. Fetch live water level
        live_wl = fetch_water_level_latest(config["hydro_station"])
        if live_wl is None:
            logger.warning(f"[{display}] No water level for '{config['hydro_station']}'. Skipping.")
            return

        logger.info(f"[{display}] Live water level: {live_wl} cm")

        # 2. Build today's data row
        today_str = datetime.now().strftime("%Y-%m-%d")
        live_row = {"timestamp": today_str, "water_level_cm": live_wl}
        for s in config["meteo_stations"]:
            p, t = fetch_meteo_latest(s)
            live_row[f"precip_{s}_mm"] = p
            live_row[f"temp_{s}_c"] = t

        # 3. Combine with history
        df_hist = (
            pd.read_csv(config["data_file"])
            if os.path.exists(config["data_file"])
            else pd.DataFrame()
        )
        df_comb = (
            pd.concat([df_hist, pd.DataFrame([live_row])], ignore_index=True)
            .drop_duplicates("timestamp")
        )
        df_comb.to_csv(config["data_file"], index=False)

        # 4. Feature engineering
        df_feats = prepare_complex_features(df_comb)
        last_row = df_feats.iloc[-1]

        feature_display = {
            "Water Level": f"{live_wl} cm",
            "Precipitation (Pt)": f"{round(last_row['Pt'], 2)} mm",
            "Soil Saturation (API)": f"{round(last_row['API_t'], 2)} idx",
            "Snowmelt (SMI)": f"{round(last_row['SMI_t'], 2)} mm",
            "Seasonality": f"{round(last_row['S_t'], 3)}",
            "Trend": f"{round(last_row['delta_WL_t'], 2)} cm",
        }

        input_data = last_row[["API_norm", "S_t", "SMI_t", "Pt", "delta_WL_t"]].values.reshape(1, -1)

        # 5. Run each horizon
        all_preds = {}
        dominant_rule = ""

        for h_id, h_cfg in config["horizons"].items():
            missing = [
                f
                for f in [h_cfg["model"], h_cfg["scaler_x"], h_cfg["scaler_y"], h_cfg["config"]]
                if not os.path.exists(f)
            ]
            if missing:
                logger.warning(f"[{display}] {h_id}: skipping — missing files: {missing}")
                continue

            scaler_x = joblib.load(h_cfg["scaler_x"])
            scaler_y = joblib.load(h_cfg["scaler_y"])

            with open(h_cfg["config"], "r") as f:
                num_mfs = json.load(f).get("num_mfs", 5)

            X_scaled = scaler_x.transform(input_data)

            model = _load_model(h_cfg, num_mfs, h_id, display)
            if model is None:
                continue

            with torch.no_grad():
                raw = model(torch.from_numpy(X_scaled).float())

            if torch.isnan(raw).any():
                logger.error(f"[{display}] {h_id}: model output NaN — skipping.")
                continue

            pred_chg = float(scaler_y.inverse_transform(raw.numpy())[0, 0])

            all_preds[h_id] = {
                "level": float(round(live_wl + pred_chg, 2)),
                "change": float(round(pred_chg, 2)),
                "label": h_cfg["label"],
            }

            # Extract dominant rule from the 1-day model
            if h_id == "1d":
                r_id, act, mf_i = extract_strongest_rule(model, X_scaled, num_mfs)
                dominant_rule = format_rule_string(r_id, act, mf_i)

        if not all_preds:
            logger.warning(f"[{display}] No valid predictions produced.")
            return

        # 6. Persist results
        _save_predictions(config, today_str, live_wl, feature_display, dominant_rule, all_preds)

    except Exception as e:
        logger.error(f"Job Failed for {display}: {e}", exc_info=True)


def _save_predictions(config, date_str, actual, features, rule, horizons):
    """Atomically write prediction results to the log file."""
    log_path = config["predictions_log_path"]
    tmp_path = log_path + ".tmp"

    log_data = {}
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                log_data = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Corrupted log detected. Resetting.")

    log_data[date_str] = {
        "actual": float(actual),
        "features": features,
        "fired_rule": rule,
        "horizons": horizons,
    }

    with open(tmp_path, "w") as f:
        json.dump(log_data, f, indent=4)
    os.replace(tmp_path, log_path)

    logger.info(f"[{config['display_name']}] Log written for {date_str}")
