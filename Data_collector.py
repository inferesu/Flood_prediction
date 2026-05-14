import time
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta

# --- Configuration ---
DATA_FILE = 'minija_complex_data_2024.csv'
STATION_CODE_HYDRO = 'priekules-vms'
STATION_CODE_METEO_1 = 'klaipedos-ams'
STATION_CODE_METEO_2 = 'vezaiciu-ams'
K_DECAY = 0.85  # Standard decay coefficient for API (Eq. 9)


def fetch_water_levels(station_code, start_date, end_date):
    """Fetches water level data from measured and historical endpoints."""
    all_obs = []
    today = datetime.utcnow().date()
    current = start_date

    while current <= end_date and current.date() <= today:
        days_diff = (today - current.date()).days
        if days_diff <= 30:
            obs_type, date_str = "measured", current.strftime("%Y-%m-%d")
            url = f"https://api.meteo.lt/v1/hydro-stations/{station_code}/observations/{obs_type}/{date_str}"
            current += timedelta(days=1)
        else:
            obs_type, date_str = "historical", current.strftime("%Y-%m")
            url = f"https://api.meteo.lt/v1/hydro-stations/{station_code}/observations/{obs_type}/{date_str}"
            current = (current.replace(day=1) + timedelta(days=32)).replace(day=1)

        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json().get("observations", [])
                for obs in data:
                    val = obs.get("waterLevel")
                    if val is not None:
                        ts = obs.get("observationTimeUtc") or obs.get("observationDateUtc")
                        all_obs.append({"timestamp": ts, "water_level_cm": val})
        except Exception:
            pass
        time.sleep(0.4)
    return all_obs


def fetch_meteo_data(station_code, start_date, end_date):
    """Fetches daily Precipitation (Pt) and Air Temperature (for SMIt)."""
    all_obs = []
    current = start_date
    while current <= end_date and current.date() <= datetime.utcnow().date():
        date_str = current.strftime("%Y-%m-%d")
        url = f"https://api.meteo.lt/v1/stations/{station_code}/observations/{date_str}"
        print(f"➡️ Fetching Meteo (P & T) for {station_code}: {date_str}")
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                obs_list = resp.json().get("observations", [])
                daily_p = sum(o.get('precipitation', 0) for o in obs_list if o.get('precipitation') is not None)
                temps = [o.get('airTemperature') for o in obs_list if o.get('airTemperature') is not None]
                avg_t = sum(temps) / len(temps) if temps else 0

                all_obs.append({
                    "timestamp": date_str,
                    f"precip_{station_code}": round(daily_p, 2),
                    f"temp_{station_code}": round(avg_t, 2)
                })
        except Exception:
            pass
        current += timedelta(days=1)
        time.sleep(0.4)
    return all_obs


def apply_hydrological_logic(df):
    """Computes API, SMI, and Seasonality per equations 9, 11, and 17."""
    df = df.copy().sort_values('timestamp')

    # 1. Pt (Average Precipitation across stations)
    df['Pt'] = df[['precip_klaipedos-ams', 'precip_vezaiciu-ams']].mean(axis=1)

    # 2. API_t = Pt + k * API_t-1 (Eq. 9)
    api_vals, current_api = [], 0
    for p in df['Pt']:
        current_api = p + (K_DECAY * current_api)
        api_vals.append(current_api)
    df['API_t'] = api_vals

    # 3. SMI_t (Snowmelt Index) (Eq. 17)
    # Simple degree-day: If temp > 0, melt occurs even without rain.
    avg_temp = df[['temp_klaipedos-ams', 'temp_vezaiciu-ams']].mean(axis=1)
    df['SMI_t'] = avg_temp.apply(lambda x: max(0, x * 2.5) if x > 0 else 0)

    # 4. Season_cos (Eq. 11)
    df['S_t'] = np.cos(2 * np.pi * pd.to_datetime(df['timestamp']).dt.dayofyear / 365)

    # 5. Trend/State (Eq. 20-21)
    df['delta_WL_t'] = df['water_level_cm'].diff().fillna(0)

    return df


def main():
    start_date = datetime(2016, 1, 1)
    end_date = datetime(2024, 12, 31)

    print("--- Step 1: Data Acquisition ---")
    w_data = fetch_water_levels(STATION_CODE_HYDRO, start_date, end_date)
    m1_data = fetch_meteo_data(STATION_CODE_METEO_1, start_date, end_date)
    m2_data = fetch_meteo_data(STATION_CODE_METEO_2, start_date, end_date)

    print("--- Step 2: Merging and Logic ---")
    df_w = pd.DataFrame(w_data)
    df_w['timestamp'] = pd.to_datetime(df_w['timestamp']).dt.date
    # Aggregate hourly water levels to daily mean
    df_w = df_w.groupby('timestamp')['water_level_cm'].mean().reset_index()

    df_m1 = pd.DataFrame(m1_data)
    df_m1['timestamp'] = pd.to_datetime(df_m1['timestamp']).dt.date

    df_m2 = pd.DataFrame(m2_data)
    df_m2['timestamp'] = pd.to_datetime(df_m2['timestamp']).dt.date

    df_final = df_w.merge(df_m1, on='timestamp', how='left').merge(df_m2, on='timestamp', how='left')
    df_final.fillna(0, inplace=True)

    df_final = apply_hydrological_logic(df_final)

    df_final.to_csv(DATA_FILE, index=False)
    print(f"✅ Success! Complex dataset saved to {DATA_FILE}")


if __name__ == "__main__":
    main()