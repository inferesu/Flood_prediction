"""Flask API routes for the flood prediction dashboard."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List

from flask import Blueprint, jsonify, render_template, request

from config.rivers import RIVER_CONFIGS
from config.infrastructure import INFRASTRUCTURE_DATA
from config.stations import MONITORING_STATIONS
from services.data_fetcher import fetch_station_forecast
from services.ai_summary import generate_summary

logger = logging.getLogger("flood_app")
api_bp = Blueprint("api", __name__)


@api_bp.route("/")
def index():
    return render_template("index.html")


@api_bp.route("/api/rivers")
def get_rivers():
    return jsonify([
        {
            "id": key,
            "display_name": cfg["display_name"],
            "lat": cfg["lat"],
            "lon": cfg["lon"],
        }
        for key, cfg in RIVER_CONFIGS.items()
    ])


@api_bp.route("/api/data")
def get_data():
    river = request.args.get("river", "minija")
    config = RIVER_CONFIGS.get(river)
    if not config:
        return jsonify({"error": "River not found"}), 404

    path = config["predictions_log_path"]
    if not os.path.exists(path):
        return jsonify({"error": "No prediction data yet for this river."}), 404

    try:
        with open(path, "r") as f:
            log_data = json.load(f)
    except json.JSONDecodeError:
        return jsonify({"error": "Corrupted log"}), 500

    sorted_dates = sorted(log_data.keys())
    last_date = sorted_dates[-1]
    last_entry = log_data[last_date]

    chart_data = _build_chart_data(log_data, sorted_dates)
    stations_data = _build_stations_data(river, last_entry)

    return jsonify({
        "riverName": config["display_name"],
        "riverKey": river,
        "lat": config["lat"],
        "lon": config["lon"],
        "riskLevels": config["risk_levels"],
        "lastKnownLevel": last_entry.get("actual"),
        "horizons": last_entry.get("horizons"),
        "currentFeatures": last_entry.get("features"),
        "firedRule": last_entry.get("fired_rule"),
        "historicalData": chart_data[-30:],
        "lastUpdated": datetime.now().isoformat(),
        "stations": stations_data,
        "infrastructure": INFRASTRUCTURE_DATA.get(river, []),
    })


@api_bp.route("/api/summary")
def get_ai_summary():
    try:
        result = generate_summary()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Gemini summary failed: {e}", exc_info=True)
        return jsonify({"summary": "AI summary unavailable.", "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_chart_data(log_data: Dict, sorted_dates: List[str]) -> List[Dict]:
    """Transform prediction log into chart-friendly time series."""
    chart_data = []
    for d in sorted_dates:
        entry = log_data[d]
        base_date = datetime.strptime(d, "%Y-%m-%d")

        chart_data.append({
            "date": d,
            "actual": entry.get("actual"),
            "pred1d": None,
            "pred3d": None,
            "pred5d": None,
        })

        horizons = entry.get("horizons", {})
        for h_id, days, key in [("1d", 1, "pred1d"), ("3d", 3, "pred3d"), ("5d", 5, "pred5d")]:
            if horizons.get(h_id):
                row = {k: None for k in ["date", "actual", "pred1d", "pred3d", "pred5d"]}
                row["date"] = (base_date + timedelta(days=days)).strftime("%Y-%m-%d")
                row[key] = horizons[h_id]["level"]
                chart_data.append(row)

    return chart_data


def _build_stations_data(river: str, last_entry: Dict) -> List[Dict]:
    """Build station-level current/forecast data for the map."""
    main_forecast = last_entry.get("horizons", {}).get("1d", {}).get("level", 200)
    stations_data = []

    for station in MONITORING_STATIONS.get(river, []):
        if station["is_main"]:
            current = last_entry.get("actual")
            forecast = main_forecast
        else:
            current, forecast = fetch_station_forecast(station["code"], main_forecast)

        stations_data.append({
            "name": station["name"],
            "lat": station["lat"],
            "lon": station["lon"],
            "current": current,
            "forecast": forecast,
        })

    return stations_data
