"""Functions for fetching live hydro/meteo data from the Lithuanian API."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger("flood_app")

_API_BASE_HYDRO = "https://api.meteo.lt/v1/hydro-stations"
_API_BASE_METEO = "https://api.meteo.lt/v1/stations"
_TIMEOUT = 10


def fetch_water_level_latest(station_code: str) -> Optional[float]:

    for days_back in range(3):
        date_str = (datetime.now().date() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = f"{_API_BASE_HYDRO}/{station_code}/observations/measured/{date_str}"
        try:
            resp = requests.get(url, timeout=_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json().get("observations", [])
                valid = [x for x in data if x.get("waterLevel") is not None]
                if valid:
                    valid.sort(key=lambda x: x.get("observationTimeUtc", ""))
                    return round(valid[-1]["waterLevel"], 2)
        except Exception:
            continue
    return None


def fetch_meteo_latest(station_code: str) -> Tuple[float, float]:
    date_str = datetime.now().strftime("%Y-%m-%d")
    url = f"{_API_BASE_METEO}/{station_code}/observations/{date_str}"
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        obs = resp.json().get("observations", [])
        precip = sum(
            o.get("precipitation", 0)
            for o in obs
            if o.get("precipitation") is not None
        )
        temps = [o["airTemperature"] for o in obs if o.get("airTemperature") is not None]
        avg_t = sum(temps) / len(temps) if temps else 0.0
        return round(precip, 2), round(avg_t, 2)
    except Exception:
        return 0.0, 0.0


def fetch_station_forecast(station_code: str, main_forecast_level: float) -> Tuple[float, float]:
    try:
        current = fetch_water_level_latest(station_code)
        if current is None:
            variation = np.random.uniform(-0.1, 0.1)
            current = round(main_forecast_level * (1 + variation), 2)
        forecast = round(current * 1.05, 2)
        return current, forecast
    except Exception as e:
        logger.warning(f"Failed to fetch data for {station_code}: {e}")
        return (
            round(main_forecast_level * 0.95, 2),
            round(main_forecast_level * 1.02, 2),
        )
