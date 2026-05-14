import time
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta

# --- Configuration ---
DATA_FILE = 'minija_kartena_complex_data_2024.csv'
STATION_CODE_HYDRO   = 'kartenos-vms'
STATION_CODE_METEO_1 = 'kretingos-ams'
STATION_CODE_METEO_2 = 'vezaiciu-ams'
K_DECAY = 0.85


def fetch_water_levels(station_code, start_date, end_date):
    all_obs = []
    today   = datetime.utcnow().date()
    current = start_date

    while current <= end_date and current.date() <= today:
        days_diff = (today - current.date()).days
        if days_diff <= 30:
            date_str = current.strftime("%Y-%m-%d")
            url      = f"https://api.meteo.lt/v1/hydro-stations/{station_code}/observations/measured/{date_str}"
            current += timedelta(days=1)
        else:
            date_str = current.strftime("%Y-%m")
            url      = f"https://api.meteo.lt/v1/hydro-stations/{station_code}/observations/historical/{date_str}"
            current  = (current.replace(day=1) + timedelta(days=32)).replace(day=1)

        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                for obs in resp.json().get("observations", []):
                    val = obs.get("waterLevel")
                    if val is not None:
                        ts = obs.get("observationTimeUtc") or obs.get("observationDateUtc")
                        all_obs.append({"timestamp": ts, "water_level_cm": val})
        except Exception:
            pass
        time.sleep(0.4)

    return all_obs


def fetch_meteo_data(station_code, start_date, end_date):
    all_obs = []
    today   = datetime.utcnow().date()
    current = start_date

    while current <= end_date and current.date() <= today:
        date_str = current.strftime("%Y-%m-%d")
        url      = f"https://api.meteo.lt/v1/stations/{station_code}/observations/{date_str}"
        print(f"  🌧️  Meteo [{station_code}] → {date_str}")
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                obs_list = resp.json().get("observations", [])
                daily_p  = sum(
                    o.get("precipitation", 0)
                    for o in obs_list
                    if o.get("precipitation") is not None
                )
                temps = [
                    o["airTemperature"]
                    for o in obs_list
                    if o.get("airTemperature") is not None
                ]
                avg_t = sum(temps) / len(temps) if temps else 0.0

                all_obs.append({
                    "timestamp":              date_str,
                    f"precip_{station_code}": round(daily_p, 2),
                    f"temp_{station_code}":   round(avg_t,   2),
                })
        except Exception:
            pass
        time.sleep(0.4)
        current += timedelta(days=1)

    return all_obs


def apply_hydrological_logic(df):
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # 1. Pt — average precipitation across both stations
    df["Pt"] = df[["precip_kretingos-ams", "precip_vezaiciu-ams"]].mean(axis=1)

    # 2. API_t = Pt + k * API_t-1  (Eq. 9)
    api_vals, current_api = [], 0.0
    for p in df["Pt"]:
        current_api = p + (K_DECAY * current_api)
        api_vals.append(current_api)
    df["API_t"] = api_vals

    # 3. API_norm  (Eq. 10)
    a_min, a_max = df["API_t"].min(), df["API_t"].max()
    df["API_norm"] = (
        (df["API_t"] - a_min) / (a_max - a_min)
        if a_max != a_min else 0.0
    )

    # 4. S_t — seasonality cosine  (Eq. 11)
    doy       = pd.to_datetime(df["timestamp"]).dt.dayofyear
    df["S_t"] = np.cos(2 * np.pi * doy / 365)

    # 5. SMI_t — snowmelt index  (Eq. 17)
    avg_temp   = df[["temp_kretingos-ams", "temp_vezaiciu-ams"]].mean(axis=1)
    df["SMI_t"] = avg_temp.apply(lambda x: max(0.0, x * 2.5) if x > 0 else 0.0)

    # 6. delta_WL_t — trend persistence  (Eq. 21)
    df["delta_WL_t"] = df["water_level_cm"].diff().fillna(0)

    return df


def main():
    start_date = datetime(2016, 4, 5)
    end_date   = datetime(2024, 12, 31)

    print("--- Step 1: Fetching Water Levels ---")
    w_data = fetch_water_levels(STATION_CODE_HYDRO, start_date, end_date)

    print("--- Step 2: Fetching Meteo Data ---")
    m1_data = fetch_meteo_data(STATION_CODE_METEO_1, start_date, end_date)
    m2_data = fetch_meteo_data(STATION_CODE_METEO_2, start_date, end_date)

    print("--- Step 3: Merging ---")
    df_w              = pd.DataFrame(w_data)
    df_w["timestamp"] = pd.to_datetime(df_w["timestamp"]).dt.date
    df_w              = df_w.groupby("timestamp")["water_level_cm"].mean().reset_index()

    df_m1              = pd.DataFrame(m1_data)
    df_m1["timestamp"] = pd.to_datetime(df_m1["timestamp"]).dt.date

    df_m2              = pd.DataFrame(m2_data)
    df_m2["timestamp"] = pd.to_datetime(df_m2["timestamp"]).dt.date

    df_final = (
        df_w
        .merge(df_m1, on="timestamp", how="left")
        .merge(df_m2, on="timestamp", how="left")
    )
    df_final.fillna(0, inplace=True)

    print("--- Step 4: Computing Features ---")
    df_final = apply_hydrological_logic(df_final)

    df_final.to_csv(DATA_FILE, index=False)
    print(f"✅ Saved → {DATA_FILE}  ({len(df_final)} rows)")


if __name__ == "__main__":
    main()
