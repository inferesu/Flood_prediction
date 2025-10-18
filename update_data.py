import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import os

# --- Configuration ---
DATA_FILE = '2025new_test.csv'
STATION_CODE_HYDRO = 'priekules-vms'
STATION_CODE_METEO_1 = 'klaipedos-ams'
STATION_CODE_METEO_2 = 'vezaiciu-ams'
# How many past days of data to fetch to ensure all features can be calculated.
# Since the longest lag is 72h (3 days), 5 days is a safe amount.
NUM_DAYS_TO_FETCH = 5


def fetch_recent_water_level(station_code, date):
    """Fetches the average water level for a single day."""
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
    """Fetches the total precipitation for a single day."""
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
    """
    Fetches data for the last NUM_DAYS_TO_FETCH days and overwrites the CSV file.
    """
    print(f"--- Starting data collection for the last {NUM_DAYS_TO_FETCH} days ---")

    all_days_data = []
    # Loop backwards from NUM_DAYS_TO_FETCH-1 to 0 to get chronological data
    # (e.g., 4 days ago, 3 days ago, ..., yesterday)
    for days_ago in range(NUM_DAYS_TO_FETCH - 1, -1, -1):
        current_date = datetime.now().date() - timedelta(days=days_ago)

        water_level = fetch_recent_water_level(STATION_CODE_HYDRO, current_date)
        # Be respectful of the API limit
        time.sleep(0.5)
        precip1 = fetch_recent_precipitation(STATION_CODE_METEO_1, current_date)
        time.sleep(0.5)
        precip2 = fetch_recent_precipitation(STATION_CODE_METEO_2, current_date)

        # Only add the day's data if we successfully got a water level reading
        if water_level is not None:
            daily_data = {
                'timestamp': current_date.strftime("%Y-%m-%d"),
                'water_level_cm': water_level,
                'precip_klaipedos-ams_mm': precip1,
                'precip_vezaiciu-ams_mm': precip2,
            }
            all_days_data.append(daily_data)
        else:
            print(f"   ⚠️ Skipping {current_date.strftime('%Y-%m-%d')} due to missing water level data.")

    # --- Overwrite the file with the newly collected data ---
    if all_days_data:
        df_final = pd.DataFrame(all_days_data)
        df_final.to_csv(DATA_FILE, index=False)
        print(f"\n✅ --- Update Complete ---")
        print(f"   -> Successfully saved last {len(all_days_data)} days of data to '{DATA_FILE}'.")
    else:
        print("\n❌ --- Update Failed ---")
        print("   -> No data was collected. The data file was not updated.")


if __name__ == "__main__":
    update_and_overwrite_data()