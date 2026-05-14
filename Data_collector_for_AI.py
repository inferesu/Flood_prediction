import time
from datetime import datetime, timedelta

import pandas as pd
import requests

# --- Configuration ---
DATA_FILE = 'testAI.csv'

# Station codes based on API documentation and testing
STATION_CODE_METEO_1 = 'klaipedos-ams'
STATION_CODE_METEO_2 = 'vezaiciu-ams'


# --- Helper Functions ---

def test_station_availability(station_code, obs_type, test_date_str="2023-01-01"):
    """
    Performs a single request to check if a station is responsive for a given observation type.
    """
    print(f"-> Testing '{station_code}' for '{obs_type}' on {test_date_str}...")
    is_hydro = obs_type == "waterLevel"

    if is_hydro:
        # NOTE: This part is no longer used but kept for function integrity
        url = f"https://api.meteo.lt/v1/hydro-stations/{station_code}/observations/historical/{test_date_str[:7]}"
    else:  # is meteorological
        url = f"https://api.meteo.lt/v1/stations/{station_code}/observations/{test_date_str}"

    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200 and resp.json().get("observations"):
            print(f"   ✅ Success: Station '{station_code}' is responsive.")
            return True
        else:
            print(f"   ❌ Failed: Station '{station_code}' responded with status {resp.status_code} or no data.")
            return False
    except Exception as e:
        print(f"   ❌ Failed: Could not connect to station '{station_code}'. Error: {e}")
        return False


def fetch_precipitation(station_code, start_date, end_date):
    """
    Fetches daily precipitation data for a meteorological station.
    """
    all_obs = []
    current = start_date
    while current <= end_date and current.date() <= datetime.utcnow().date():
        date_str = current.strftime("%Y-%m-%d")
        url = f"https://api.meteo.lt/v1/stations/{station_code}/observations/{date_str}"
        print(f"➡️ Fetching precipitation for '{station_code}' on {date_str}")
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json().get("observations", [])
                # The meteorological API returns hourly data with a direct 'precipitation' key.
                # We sum the values from this key for all hours in the day.
                daily_precip = sum(obs.get('precipitation', 0) for obs in data if obs.get('precipitation') is not None)
                # Round the final sum to 2 decimal places to keep data clean.
                all_obs.append({"timestamp": date_str, f"precip_{station_code}_mm": round(daily_precip, 2)})
        except Exception:
            pass
        current += timedelta(days=1)
        time.sleep(0.5)
    return all_obs


def main():
    print("--- Step 1: Pre-check Station Availability ---")
    meteo1_ok = test_station_availability(STATION_CODE_METEO_1, "precipitation")
    meteo2_ok = test_station_availability(STATION_CODE_METEO_2, "precipitation")

    if not (meteo1_ok and meteo2_ok):
        print("\n❌ PRE-CHECK FAILED. One or more stations are unresponsive. Aborting.")
        return

    print("\n✅ Pre-check successful. All stations are available.")

    # Set the date range
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    print(f"\n--- Step 2: Starting Full Data Collection from {start_date.date()} to {end_date.date()} ---")

    # Fetch all data streams
    precip_data1 = fetch_precipitation(STATION_CODE_METEO_1, start_date, end_date)
    precip_data2 = fetch_precipitation(STATION_CODE_METEO_2, start_date, end_date)

    if not precip_data1 and not precip_data2:
        print("❌ No precipitation data could be collected. Aborting.")
        return

    # Process and combine data using pandas
    print("\n--- Step 3: Processing and Combining Data ---")

    df_precip1 = pd.DataFrame(precip_data1)
    df_precip1['timestamp'] = pd.to_datetime(df_precip1['timestamp']).dt.date

    df_precip2 = pd.DataFrame(precip_data2)
    df_precip2['timestamp'] = pd.to_datetime(df_precip2['timestamp']).dt.date

    # Merge the two precipitation dataframes on timestamp
    # Use an 'outer' merge to keep all records from both stations
    df_final = pd.merge(df_precip1, df_precip2, on='timestamp', how='outer')

    # Fill any missing precipitation values with 0
    df_final.fillna(0, inplace=True)
    df_final['timestamp'] = pd.to_datetime(df_final['timestamp'])

    df_final.sort_values('timestamp', inplace=True)
    df_final.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)

    print(f"\n✅ Collected and combined {len(df_final)} unique daily records.")
    df_final.to_csv(DATA_FILE, index=False)
    print(f"💾 Saved final dataset to {DATA_FILE}")


if __name__ == "__main__":
    main()