import time
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta

# --- Configuration ---
DATA_FILE            = 'kanalas_lankupiu_complex_data_2024.csv'
STATION_CODE_HYDRO   = 'lankupiu-klaipedos-vms'
STATION_CODE_METEO_1 = 'ventes-ams'
STATION_CODE_METEO_2 = 'klaipedos-ams'
K_DECAY              = 0.85


def fetch_water_levels(station_code, start_date, end_date):
    all_obs = []
    today   = datetime.utcnow().date()
    current = start_date

    while current <= end_date and current.date() <= today:
        days_diff = (today - current.date()).days
        if days_diff <= 30:
            obs_type = "measured"
            date_str = current.strftime("%Y-%m-%d")
            url      = (
                f"https://api.meteo.lt/v1/hydro-stations/{station_code}"
                f"/observations/{obs_type}/{date_str}"
            )
            current += timedelta(days=1)
        else:
            obs_type = "historical"
            date_str = current.strftime("%Y-%m")
            url      = (
                f"https://api.meteo.lt/v1/hydro-stations/{station_code}"
                f"/observations/{obs_type}/{date_str}"
            )
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
            else:
                print(f"   ⚠️  [{obs_type.upper()}] {date_str}  |  HTTP {resp.status_code} — skipped")
        except Exception as e:
            print(f"   ❌ [{obs_type.upper()}] {date_str}  |  Error: {e}")

        time.sleep(0.4)

    print(f"   ✔️  Water level records fetched: {len(all_obs)}")
    return all_obs


def fetch_meteo_data(station_code, start_date, end_date):
    all_obs = []
    current = start_date

    while current <= end_date and current.date() <= datetime.utcnow().date():
        date_str = current.strftime("%Y-%m-%d")
        url      = (
            f"https://api.meteo.lt/v1/stations/{station_code}"
            f"/observations/{date_str}"
        )
        print(f"   ➡️  Fetching Meteo [{station_code}]: {date_str}")
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
            else:
                print(f"   ⚠️  {date_str}  |  HTTP {resp.status_code} — skipped")
        except Exception as e:
            print(f"   ❌ {date_str}  |  Error: {e}")

        current += timedelta(days=1)
        time.sleep(0.4)

    print(f"   ✔️  Meteo records fetched for {station_code}: {len(all_obs)}")
    return all_obs


def apply_hydrological_logic(df):
    df = df.copy().sort_values("timestamp")

    # 1. Pt — average precipitation across both meteo stations
    df["Pt"] = df[["precip_ventes-ams", "precip_klaipedos-ams"]].mean(axis=1)

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
    df["S_t"] = np.cos(2 * np.pi * pd.to_datetime(df["timestamp"]).dt.dayofyear / 365)

    # 5. SMI_t — snowmelt index  (Eq. 17)
    avg_temp    = df[["temp_ventes-ams", "temp_klaipedos-ams"]].mean(axis=1)
    df["SMI_t"] = avg_temp.apply(lambda x: max(0.0, x * 2.5) if x > 0 else 0.0)

    # 6. delta_WL_t — trend persistence  (Eq. 21)
    df["delta_WL_t"] = df["water_level_cm"].diff().fillna(0)

    return df


def main():
    start_date = datetime(2016, 1, 1)
    end_date   = datetime(2024, 12, 31)

    print("=" * 60)
    print("   🌊 KANALAS LANKUPIŲ DATA PIPELINE")
    print(f"   Output  : {DATA_FILE}")
    print(f"   Hydro   : {STATION_CODE_HYDRO}")
    print(f"   Meteo 1 : {STATION_CODE_METEO_1}")
    print(f"   Meteo 2 : {STATION_CODE_METEO_2}")
    print(f"   Period  : {start_date.date()} → {end_date.date()}")
    print("=" * 60)

    print("\n━━━ STEP 1: Fetching Water Levels ━━━━━━━━━━━━━━━━━━━━━━━")
    w_data  = fetch_water_levels(STATION_CODE_HYDRO,   start_date, end_date)

    print("\n━━━ STEP 2: Fetching Meteo Data ━━━━━━━━━━━━━━━━━━━━━━━━━")
    m1_data = fetch_meteo_data(STATION_CODE_METEO_1,   start_date, end_date)
    m2_data = fetch_meteo_data(STATION_CODE_METEO_2,   start_date, end_date)

    print("\n━━━ STEP 3: Merging DataFrames ━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if not w_data:
        print("❌ No water level data — check station code / date range.")
        return

    df_w              = pd.DataFrame(w_data)
    df_w["timestamp"] = pd.to_datetime(df_w["timestamp"]).dt.date
    df_w              = df_w.groupby("timestamp")["water_level_cm"].mean().reset_index()
    print(f"   ✔️  Water levels: {len(df_w)} daily rows")

    df_m1              = pd.DataFrame(m1_data)
    df_m1["timestamp"] = pd.to_datetime(df_m1["timestamp"]).dt.date

    df_m2              = pd.DataFrame(m2_data)
    df_m2["timestamp"] = pd.to_datetime(df_m2["timestamp"]).dt.date

    df_final = (
        df_w
        .merge(df_m1, on="timestamp", how="left")
        .merge(df_m2, on="timestamp", how="left")
    )
    missing = df_final.isnull().sum().sum()
    df_final.fillna(0, inplace=True)
    print(f"   ✔️  Merged: {len(df_final)} rows, {missing} NaN values filled with 0")

    print("\n━━━ STEP 4: Computing Hydrological Features ━━━━━━━━━━━━━")
    df_final = apply_hydrological_logic(df_final)

    print("\n━━━ STEP 5: Saving Output ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    df_final.to_csv(DATA_FILE, index=False)
    print(f"   ✔️  Saved → {DATA_FILE}  ({len(df_final)} rows)")
    print(f"   ✔️  Columns: {list(df_final.columns)}")
    print(f"   ✔️  Date range: {df_final['timestamp'].min()} → {df_final['timestamp'].max()}")

    print("\n" + "=" * 60)
    print(f"   ✅ Pipeline complete! → {DATA_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
