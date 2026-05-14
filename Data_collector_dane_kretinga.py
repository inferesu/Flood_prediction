import time
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta

# --- Configuration ---
DATA_FILE            = 'dane_kretinga_complex_data_2024.csv'
STATION_CODE_HYDRO   = 'kretingos-vms'
STATION_CODE_METEO_1 = 'kretingos-ams'
K_DECAY              = 0.85


def fetch_water_levels(station_code, start_date, end_date):
    all_obs       = []
    today         = datetime.utcnow().date()
    current       = start_date
    request_count = 0

    print(f"\n📡 Starting water level fetch for station: [{station_code}]")
    print(f"   Date range: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    print(f"   {'─' * 55}")

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
            resp          = requests.get(url, timeout=30)
            request_count += 1
            if resp.status_code == 200:
                data        = resp.json().get("observations", [])
                batch_count = 0
                for obs in data:
                    val = obs.get("waterLevel")
                    if val is not None:
                        ts = obs.get("observationTimeUtc") or obs.get("observationDateUtc")
                        all_obs.append({"timestamp": ts, "water_level_cm": val})
                        batch_count += 1
                print(
                    f"   💧 [{obs_type.upper():10}] {date_str}  |"
                    f"  +{batch_count:4} records  |  Total: {len(all_obs)}"
                )
            else:
                print(
                    f"   ⚠️  [{obs_type.upper():10}] {date_str}  |"
                    f"  HTTP {resp.status_code} — skipped"
                )
        except Exception as e:
            print(f"   ❌ [{obs_type.upper():10}] {date_str}  |  Error: {e}")

        time.sleep(0.4)

    print(f"   {'─' * 55}")
    print(
        f"   ✅ Water level fetch complete — "
        f"{len(all_obs)} total observations ({request_count} requests)\n"
    )
    return all_obs


def fetch_meteo_data(station_code, start_date, end_date):
    all_obs       = []
    current       = start_date
    request_count = 0
    total_days    = (end_date - start_date).days + 1

    print(f"\n🌤️  Starting meteo fetch for station: [{station_code}]")
    print(
        f"   Date range: {start_date.strftime('%Y-%m-%d')} → "
        f"{end_date.strftime('%Y-%m-%d')} ({total_days} days)"
    )
    print(f"   {'─' * 55}")

    while current <= end_date and current.date() <= datetime.utcnow().date():
        date_str   = current.strftime("%Y-%m-%d")
        url        = (
            f"https://api.meteo.lt/v1/stations/{station_code}"
            f"/observations/{date_str}"
        )
        days_done     = (current - start_date).days + 1
        progress_pct  = (days_done / total_days) * 100

        try:
            resp          = requests.get(url, timeout=30)
            request_count += 1
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
                print(
                    f"   ➡️  {date_str}  |  P={daily_p:5.2f}mm  T={avg_t:6.2f}°C"
                    f"  |  [{progress_pct:5.1f}%] ({days_done}/{total_days})"
                )
            else:
                print(
                    f"   ⚠️  {date_str}  |  HTTP {resp.status_code} — skipped"
                    f"  |  [{progress_pct:5.1f}%]"
                )
        except Exception as e:
            print(f"   ❌ {date_str}  |  Error: {e}")

        current += timedelta(days=1)
        time.sleep(0.4)

    print(f"   {'─' * 55}")
    print(
        f"   ✅ Meteo fetch complete — "
        f"{len(all_obs)} days collected ({request_count} requests)\n"
    )
    return all_obs


def apply_hydrological_logic(df):
    print(f"\n⚙️  Applying hydrological logic to {len(df)} rows...")

    df = df.copy().sort_values("timestamp")

    # 1. Pt — single station precipitation
    df["Pt"] = df["precip_kretingos-ams"]
    print(f"   ✔️  [Eq. 9]  Pt computed — mean={df['Pt'].mean():.2f}mm, max={df['Pt'].max():.2f}mm")

    # 2. API_t = Pt + k * API_t-1  (Eq. 9)
    api_vals, current_api = [], 0.0
    for p in df["Pt"]:
        current_api = p + (K_DECAY * current_api)
        api_vals.append(current_api)
    df["API_t"] = api_vals
    print(
        f"   ✔️  [Eq. 9]  API_t computed — "
        f"k={K_DECAY}, mean={df['API_t'].mean():.2f}, max={df['API_t'].max():.2f}"
    )

    # 3. API_norm  (Eq. 10)
    a_min, a_max = df["API_t"].min(), df["API_t"].max()
    df["API_norm"] = (
        (df["API_t"] - a_min) / (a_max - a_min)
        if a_max != a_min else 0.0
    )
    print(f"   ✔️  [Eq. 10] API_norm computed — range=[{df['API_norm'].min():.3f}, {df['API_norm'].max():.3f}]")

    # 4. S_t — seasonality cosine  (Eq. 11)
    doy       = pd.to_datetime(df["timestamp"]).dt.dayofyear
    df["S_t"] = np.cos(2 * np.pi * doy / 365)
    print(f"   ✔️  [Eq. 11] S_t computed — range=[{df['S_t'].min():.2f}, {df['S_t'].max():.2f}]")

    # 5. SMI_t — snowmelt index  (Eq. 17)
    df["SMI_t"] = df["temp_kretingos-ams"].apply(
        lambda x: max(0.0, x * 2.5) if x > 0 else 0.0
    )
    melt_days = (df["SMI_t"] > 0).sum()
    print(f"   ✔️  [Eq. 17] SMI_t computed — {melt_days} melt days, max={df['SMI_t'].max():.2f}")

    # 6. delta_WL_t — trend persistence  (Eq. 21)
    df["delta_WL_t"] = df["water_level_cm"].diff().fillna(0)
    rising  = (df["delta_WL_t"] > 0).sum()
    falling = (df["delta_WL_t"] < 0).sum()
    print(f"   ✔️  [Eq. 21] delta_WL_t computed — rising={rising} days, falling={falling} days")

    print(f"   {'─' * 55}")
    print("   ✅ Hydrological logic applied successfully\n")
    return df


def main():
    start_date = datetime(2021, 1, 1)
    end_date   = datetime(2024, 12, 31)

    print("=" * 60)
    print("   🌊 DANĖ–KRETINGA HYDROLOGICAL DATA PIPELINE")
    print(f"   Output : {DATA_FILE}")
    print(f"   Period : {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    print(f"   Hydro  : {STATION_CODE_HYDRO}")
    print(f"   Meteo  : {STATION_CODE_METEO_1}")
    print("=" * 60)

    # ── Step 1: Fetch ─────────────────────────────────────────────────────
    print("\n━━━ STEP 1: Data Acquisition ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    w_data  = fetch_water_levels(STATION_CODE_HYDRO,   start_date, end_date)
    m1_data = fetch_meteo_data(STATION_CODE_METEO_1,   start_date, end_date)

    # ── Step 2: Merge ─────────────────────────────────────────────────────
    print("\n━━━ STEP 2: Merging Data ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if not w_data:
        print("❌ No water level data returned — check station code / date range.")
        return

    print(f"   🔧 Building water level DataFrame ({len(w_data)} raw records)...")
    df_w              = pd.DataFrame(w_data)
    df_w["timestamp"] = pd.to_datetime(df_w["timestamp"]).dt.date
    df_w              = df_w.groupby("timestamp")["water_level_cm"].mean().reset_index()
    print(f"   ✔️  Water levels aggregated to {len(df_w)} daily rows")

    print(f"   🔧 Building meteo DataFrame ({len(m1_data)} records)...")
    df_m1              = pd.DataFrame(m1_data)
    df_m1["timestamp"] = pd.to_datetime(df_m1["timestamp"]).dt.date

    print("   🔧 Merging on timestamp...")
    df_final = df_w.merge(df_m1, on="timestamp", how="left")
    missing_before = df_final.isnull().sum().sum()
    df_final.fillna(0, inplace=True)
    print(
        f"   ✔️  Merge complete — {len(df_final)} rows, "
        f"{missing_before} NaN values filled with 0"
    )

    # ── Step 3: Features ──────────────────────────────────────────────────
    print("\n━━━ STEP 3: Hydrological Feature Engineering ━━━━━━━━━━━━━")
    df_final = apply_hydrological_logic(df_final)

    # ── Step 4: Save ──────────────────────────────────────────────────────
    print("\n━━━ STEP 4: Saving Output ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    df_final.to_csv(DATA_FILE, index=False)
    print(f"   💾 File saved  : {DATA_FILE}")
    print(f"   📊 Shape       : {df_final.shape[0]} rows × {df_final.shape[1]} columns")
    print(
        f"   📅 Date range  : "
        f"{df_final['timestamp'].min()} → {df_final['timestamp'].max()}"
    )
    print(f"   📋 Columns     : {list(df_final.columns)}")

    print("\n" + "=" * 60)
    print(f"   ✅ Pipeline complete! → {DATA_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
