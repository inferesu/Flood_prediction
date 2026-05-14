import time
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta

# --- Configuration ---
DATA_FILE            = 'marios_birstonas_complex_data_2024.csv'
STATION_CODE_HYDRO   = 'birstono-vms'
STATION_CODE_METEO_1 = 'birstono-ams'
K_DECAY              = 0.85


def fetch_water_levels(station_code, start_date, end_date):
    all_obs       = []
    today         = datetime.utcnow().date()
    current       = start_date
    skipped       = 0
    errors        = 0

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
                batch = 0
                for obs in resp.json().get("observations", []):
                    val = obs.get("waterLevel")
                    if val is not None:
                        ts = obs.get("observationTimeUtc") or obs.get("observationDateUtc")
                        all_obs.append({"timestamp": ts, "water_level_cm": val})
                        batch += 1
                if batch == 0:
                    print(f"   ⚠️  [{obs_type.upper()}] {date_str}  |  200 OK but 0 waterLevel values")
                    skipped += 1
            else:
                print(f"   ⚠️  [{obs_type.upper()}] {date_str}  |  HTTP {resp.status_code} — skipped")
                skipped += 1
        except Exception as e:
            print(f"   ❌ [{obs_type.upper()}] {date_str}  |  Error: {e}")
            errors += 1

        time.sleep(0.4)

    # ── Post-fetch validation ─────────────────────────────────────────────
    print(f"\n   📊 Water level fetch summary for [{station_code}]:")
    print(f"      Records fetched : {len(all_obs)}")
    print(f"      Skipped / empty : {skipped}")
    print(f"      Errors          : {errors}")

    if len(all_obs) == 0:
        print(f"   ❌ CRITICAL: No water level data at all — check station code '{station_code}'")
    elif len(all_obs) < 100:
        print(f"   ⚠️  WARNING: Only {len(all_obs)} records — suspiciously low, check date range")
    else:
        # Check for suspicious values
        vals = [o["water_level_cm"] for o in all_obs]
        neg  = sum(1 for v in vals if v < 0)
        if neg:
            print(f"   ⚠️  WARNING: {neg} negative water level values detected (will be clipped)")
        print(f"   ✔️  Range: {min(vals):.1f} cm → {max(vals):.1f} cm")

    return all_obs


def fetch_meteo_data(station_code, start_date, end_date):
    all_obs      = []
    current      = start_date
    skipped      = 0
    errors       = 0
    zero_precip  = 0

    while current <= end_date and current.date() <= datetime.utcnow().date():
        date_str = current.strftime("%Y-%m-%d")
        url      = f"https://api.meteo.lt/v1/stations/{station_code}/observations/{date_str}"
        print(f"➡️  Fetching Meteo [{station_code}]: {date_str}")
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                obs_list = resp.json().get("observations", [])

                if not obs_list:
                    print(f"   ⚠️  {date_str}  |  200 OK but no observations in response")
                    skipped += 1
                    current += timedelta(days=1)
                    time.sleep(0.4)
                    continue

                daily_p = sum(o.get("precipitation", 0) for o in obs_list if o.get("precipitation") is not None)
                temps   = [o.get("airTemperature") for o in obs_list if o.get("airTemperature") is not None]
                avg_t   = sum(temps) / len(temps) if temps else 0

                if not temps:
                    print(f"   ⚠️  {date_str}  |  No airTemperature values in {len(obs_list)} observations")

                if daily_p == 0:
                    zero_precip += 1

                all_obs.append({
                    "timestamp":              date_str,
                    f"precip_{station_code}": round(daily_p, 2),
                    f"temp_{station_code}":   round(avg_t,   2),
                })
            else:
                print(f"   ⚠️  {date_str}  |  HTTP {resp.status_code} — skipped")
                skipped += 1
        except Exception as e:
            print(f"   ❌ {date_str}  |  Error: {e}")
            errors += 1

        current += timedelta(days=1)
        time.sleep(0.4)

    # ── Post-fetch validation ─────────────────────────────────────────────
    print(f"\n   📊 Meteo fetch summary for [{station_code}]:")
    print(f"      Days fetched    : {len(all_obs)}")
    print(f"      Skipped / empty : {skipped}")
    print(f"      Errors          : {errors}")
    print(f"      Zero-precip days: {zero_precip} ({zero_precip/max(len(all_obs),1)*100:.1f}%)")

    if len(all_obs) == 0:
        print(f"   ❌ CRITICAL: No meteo data at all — check station code '{station_code}'")
    elif skipped > len(all_obs) * 0.2:
        print(f"   ⚠️  WARNING: {skipped} days skipped — more than 20% of the range has no data")
    else:
        temps_all  = [o[f"temp_{station_code}"]   for o in all_obs]
        precip_all = [o[f"precip_{station_code}"] for o in all_obs]
        print(f"   ✔️  Temp range   : {min(temps_all):.1f}°C → {max(temps_all):.1f}°C")
        print(f"   ✔️  Precip range : {min(precip_all):.1f}mm → {max(precip_all):.1f}mm")

    return all_obs


def apply_hydrological_logic(df):
    df = df.copy().sort_values("timestamp")

    df["precip_birstono-ams"] = df["precip_birstono-ams"].clip(lower=0)
    df["water_level_cm"]      = df["water_level_cm"].clip(lower=0)

    df["Pt"] = df["precip_birstono-ams"]

    api_vals, current_api = [], 0.0
    for p in df["Pt"]:
        current_api = p + (K_DECAY * current_api)
        api_vals.append(current_api)
    df["API_t"] = np.array(api_vals).clip(min=0)

    a_min, a_max = df["API_t"].min(), df["API_t"].max()
    df["API_norm"] = (
        ((df["API_t"] - a_min) / (a_max - a_min)).clip(0.0, 1.0)
        if a_max != a_min else 0.0
    )

    df["SMI_t"] = df["temp_birstono-ams"].apply(
        lambda x: max(0.0, x * 2.5) if x > 0 else 0.0
    )

    df["S_t"] = np.cos(2 * np.pi * pd.to_datetime(df["timestamp"]).dt.dayofyear / 365)

    df["delta_WL_t"] = df["water_level_cm"].diff().fillna(0)

    return df


def validate_merged_df(df, expected_days):
    """Runs sanity checks on the merged DataFrame before saving."""
    print(f"\n   🔍 Validating merged dataset...")
    passed = True

    # Row count
    if len(df) < expected_days * 0.8:
        print(f"   ⚠️  Only {len(df)} rows — expected ~{expected_days}. Many dates may be missing.")
        passed = False
    else:
        print(f"   ✔️  Row count : {len(df)} (expected ~{expected_days})")

    # NaN check (after fillna this should be 0 but worth confirming)
    nan_count = df.isnull().sum().sum()
    if nan_count:
        print(f"   ⚠️  {nan_count} NaN values remain after fillna")
        passed = False
    else:
        print(f"   ✔️  No NaN values")

    # Water level
    wl_zeros = (df["water_level_cm"] == 0).sum()
    if wl_zeros > len(df) * 0.1:
        print(f"   ⚠️  {wl_zeros} rows have water_level_cm = 0 (>{wl_zeros/len(df)*100:.0f}%)")
    else:
        print(f"   ✔️  Water level: min={df['water_level_cm'].min():.1f}, max={df['water_level_cm'].max():.1f}, mean={df['water_level_cm'].mean():.1f}")

    # Features
    for col in ["API_norm", "S_t", "SMI_t", "Pt", "delta_WL_t"]:
        if col in df.columns:
            nan_in_col = df[col].isnull().sum()
            if nan_in_col:
                print(f"   ⚠️  {col} has {nan_in_col} NaN values")
                passed = False

    if passed:
        print(f"   ✅ All validation checks passed")
    else:
        print(f"   ⚠️  Some checks failed — review warnings above before training")

    return passed


def main():
    start_date = datetime(2022, 1, 1)
    end_date   = datetime(2024, 12, 31)
    expected_days = (end_date - start_date).days + 1

    print("=" * 60)
    print("   🌊 BIRSTONAS DATA PIPELINE")
    print(f"   Output  : {DATA_FILE}")
    print(f"   Hydro   : {STATION_CODE_HYDRO}")
    print(f"   Meteo   : {STATION_CODE_METEO_1}")
    print(f"   Period  : {start_date.date()} → {end_date.date()} ({expected_days} days)")
    print("=" * 60)

    print("\n━━━ STEP 1: Data Acquisition ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    w_data  = fetch_water_levels(STATION_CODE_HYDRO,   start_date, end_date)
    m1_data = fetch_meteo_data(STATION_CODE_METEO_1,   start_date, end_date)

    # Hard stop if either dataset is empty
    if not w_data:
        print("\n❌ ABORTED: No water level data. Check station code and date range.")
        return
    if not m1_data:
        print("\n❌ ABORTED: No meteo data. Check station code and date range.")
        return

    print("\n━━━ STEP 2: Merging Data ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    df_w              = pd.DataFrame(w_data)
    df_w["timestamp"] = pd.to_datetime(df_w["timestamp"]).dt.date
    df_w              = df_w.groupby("timestamp")["water_level_cm"].mean().reset_index()
    print(f"   ✔️  Water levels: {len(df_w)} daily rows")

    df_m1              = pd.DataFrame(m1_data)
    df_m1["timestamp"] = pd.to_datetime(df_m1["timestamp"]).dt.date
    print(f"   ✔️  Meteo rows  : {len(df_m1)}")

    df_final       = df_w.merge(df_m1, on="timestamp", how="left")
    missing_before = df_final.isnull().sum().sum()
    df_final.fillna(0, inplace=True)
    print(f"   ✔️  Merged      : {len(df_final)} rows, {missing_before} NaN filled with 0")

    # Check how many meteo rows matched
    matched = len(df_final) - missing_before
    if missing_before > len(df_final) * 0.1:
        print(f"   ⚠️  WARNING: {missing_before} rows had no meteo match — gaps in meteo data?")

    print("\n━━━ STEP 3: Hydrological Feature Engineering ━━━━━━━━━━━━━")
    df_final = apply_hydrological_logic(df_final)

    print("\n━━━ STEP 4: Validation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    validate_merged_df(df_final, expected_days)

    print("\n━━━ STEP 5: Saving Output ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    df_final.to_csv(DATA_FILE, index=False)
    print(f"   💾 Saved  : {DATA_FILE}")
    print(f"   📊 Shape  : {df_final.shape[0]} rows × {df_final.shape[1]} columns")
    print(f"   📅 Range  : {df_final['timestamp'].min()} → {df_final['timestamp'].max()}")
    print(f"   📋 Columns: {list(df_final.columns)}")

    print("\n" + "=" * 60)
    print(f"   ✅ Pipeline complete! → {DATA_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
