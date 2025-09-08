import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import os

# --- Configuration ---
# This is the file that will be updated.
DATA_FILE = '2025new_test.csv'

STATION_CODE_HYDRO = 'priekules-vms'
STATION_CODE_METEO_1 = 'klaipedos-ams'
STATION_CODE_METEO_2 = 'vezaiciu-ams'


def fetch_recent_water_level(station_code, date):
    """Fetches the water level for a single, recent day."""
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.meteo.lt/v1/hydro-stations/{station_code}/observations/measured/{date_str}"
    print(f"-> Fetching water level for {date_str}...")
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            observations = resp.json().get("observations", [])
            # For recent data, we average the hourly readings for consistency
            water_levels = [obs['waterLevel'] for obs in observations if obs.get('waterLevel') is not None]
            if water_levels:
                return round(sum(water_levels) / len(water_levels), 2)
    except Exception as e:
        print(f"   - ❌ Error fetching water level: {e}")
    return None


def fetch_recent_precipitation(station_code, date):
    """Fetches and sums the precipitation for a single, recent day."""
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.meteo.lt/v1/stations/{station_code}/observations/{date_str}"
    print(f"-> Fetching precipitation for '{station_code}' on {date_str}...")
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            observations = resp.json().get("observations", [])
            daily_precip = sum(
                obs.get('precipitation', 0) for obs in observations if obs.get('precipitation') is not None)
            return round(daily_precip, 2)
    except Exception as e:
        print(f"   - ❌ Error fetching precipitation: {e}")
    return 0  # Return 0 if fetching fails, as it's the most likely value


def collect_yesterdays_data():
    """
    Fetches the data for the previous day and appends it to the CSV file
    if it's not already present.
    """
    yesterday = datetime.now().date() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")

    # --- Step 1: Check if data for yesterday already exists ---
    if os.path.exists(DATA_FILE):
        df_existing = pd.read_csv(DATA_FILE)
        if yesterday_str in df_existing['timestamp'].values:
            print(f"✅ Data for {yesterday_str} already exists in '{DATA_FILE}'. Nothing to do.")
            return
    else:
        # Create the file with headers if it doesn't exist
        print(f"File '{DATA_FILE}' not found. Creating it now.")
        header_df = pd.DataFrame(
            columns=['timestamp', 'water_level_cm', 'precip_klaipedos-ams_mm', 'precip_vezaiciu-ams_mm'])
        header_df.to_csv(DATA_FILE, index=False)

    print(f"--- Collecting data for {yesterday_str} ---")

    # --- Step 2: Fetch data for yesterday ---
    water_level = fetch_recent_water_level(STATION_CODE_HYDRO, yesterday)
    precip1 = fetch_recent_precipitation(STATION_CODE_METEO_1, yesterday)
    time.sleep(0.5)  # Be respectful of the API limit
    precip2 = fetch_recent_precipitation(STATION_CODE_METEO_2, yesterday)

    if water_level is None:
        print(f"\n❌ Could not fetch water level data for {yesterday_str}. Aborting update.")
        return

    # --- Step 3: Append the new data to the CSV file ---
    new_data = {
        'timestamp': [yesterday_str],
        'water_level_cm': [water_level],
        'precip_klaipedos-ams_mm': [precip1],
        'precip_vezaiciu-ams_mm': [precip2],
    }
    df_new = pd.DataFrame(new_data)

    # Use mode='a' (append) and header=False to add the new row
    df_new.to_csv(DATA_FILE, mode='a', header=False, index=False)

    print(f"\n✅ --- Update Complete ---")
    print(f"   -> Successfully added data for {yesterday_str} to '{DATA_FILE}'.")


if __name__ == "__main__":
    collect_yesterdays_data()