import time
from datetime import datetime, timedelta

import pandas as pd
import requests

# --- Configuration ---
# The output file for the complete historical dataset for training
DATA_FILE = 'minija_multi_weather_data_2015-2022.csv'

# Station codes for historical observations
STATION_CODE_HYDRO = 'priekules-vms'
STATION_CODE_METEO_1 = 'klaipedos-ams'
STATION_CODE_METEO_2 = 'vezaiciu-ams'


def fetch_water_levels(station_code, start_date, end_date):
    """
    Fetches historical water level data.
    Upsamples daily historical data to an hourly format to match meteorological data.
    """
    print(f"\n--- Fetching Water Levels for {station_code} ---")
    all_obs = []
    today = datetime.utcnow().date()
    current = start_date

    while current <= end_date and current.date() <= today:
        days_diff = (today - current.date()).days
        # Use measured (hourly) endpoint for recent data (last 30 days)
        if days_diff <= 30:
            date_str = current.strftime("%Y-%m-%d")
            url = f"https://api.meteo.lt/v1/hydro-stations/{station_code}/observations/measured/{date_str}"
            print(f"-> Fetching HOURLY measured water level for {date_str}")
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    data = resp.json().get("observations", [])
                    for obs in data:
                        if obs.get("waterLevel") is not None:
                            all_obs.append(
                                {"timestamp": obs["observationTimeUtc"], "water_level_cm": obs["waterLevel"]})
            except Exception as e:
                print(f"   - Warning: Could not fetch data for {date_str}. Error: {e}")
            current += timedelta(days=1)
        # Use historical (daily) endpoint for older data
        else:
            date_str = current.strftime("%Y-%m")
            url = f"https://api.meteo.lt/v1/hydro-stations/{station_code}/observations/historical/{date_str}"
            print(f"-> Fetching DAILY historical water level for {date_str}")
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    data = resp.json().get("observations", [])
                    for obs in data:
                        # Create 24 hourly records for each daily observation
                        if obs.get("waterLevel") is not None:
                            day_str = obs["observationDateUtc"]
                            for hour in range(24):
                                dt_str = f"{day_str} {hour:02d}:00:00"
                                all_obs.append({"timestamp": dt_str, "water_level_cm": obs["waterLevel"]})
            except Exception as e:
                print(f"   - Warning: Could not fetch data for {date_str}. Error: {e}")
            # Move to the next month
            current = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
        time.sleep(0.5)
    return all_obs


def fetch_hourly_meteo_data(station_code, start_date, end_date):
    """
    Fetches historical hourly precipitation and air temperature data.
    """
    print(f"\n--- Fetching Meteo Data for {station_code} ---")
    all_obs = []
    current = start_date
    while current <= end_date and current.date() <= datetime.utcnow().date():
        date_str = current.strftime("%Y-%m-%d")
        url = f"https://api.meteo.lt/v1/stations/{station_code}/observations/{date_str}"
        print(f"-> Fetching hourly meteo data for {date_str}")
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json().get("observations", [])
                for obs in data:
                    all_obs.append({
                        "timestamp": obs['observationTimeUtc'],
                        f"precip_{station_code}_mm": obs.get('precipitation'),
                        f"temp_{station_code}_c": obs.get('airTemperature')
                    })
        except Exception as e:
            print(f"   - Warning: Could not fetch data for {date_str}. Error: {e}")
        current += timedelta(days=1)
        time.sleep(0.5)
    return all_obs


def main():
    """Main function to collect, process, and save all historical data."""
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2022, 12, 31)
    print(f"--- Starting Full Historical Data Collection from {start_date.date()} to {end_date.date()} ---")

    # Fetch all data streams
    water_data = fetch_water_levels(STATION_CODE_HYDRO, start_date, end_date)
    meteo_data1 = fetch_hourly_meteo_data(STATION_CODE_METEO_1, start_date, end_date)
    meteo_data2 = fetch_hourly_meteo_data(STATION_CODE_METEO_2, start_date, end_date)

    if not water_data or not meteo_data1 or not meteo_data2:
        print("\n❌ One or more data streams could not be collected. Aborting.")
        return

    print("\n--- Processing and Combining All Hourly Data ---")
    df_water = pd.DataFrame(water_data)
    df_water['timestamp'] = pd.to_datetime(df_water['timestamp'])

    df_meteo1 = pd.DataFrame(meteo_data1)
    df_meteo1['timestamp'] = pd.to_datetime(df_meteo1['timestamp'])

    df_meteo2 = pd.DataFrame(meteo_data2)
    df_meteo2['timestamp'] = pd.to_datetime(df_meteo2['timestamp'])

    # Merge all dataframes on the timestamp
    df_final = pd.merge(df_water, df_meteo1, on='timestamp', how='outer')
    df_final = pd.merge(df_final, df_meteo2, on='timestamp', how='outer')

    # Create the final averaged temperature column and rename precip columns
    df_final['temp_c'] = df_final[[f'temp_{STATION_CODE_METEO_1}_c', f'temp_{STATION_CODE_METEO_2}_c']].mean(axis=1)
    df_final.rename(columns={
        f'precip_{STATION_CODE_METEO_1}_mm': 'precip_klaipedos-ams_mm',
        f'precip_{STATION_CODE_METEO_2}_mm': 'precip_vezaiciu-ams_mm',
    }, inplace=True)

    # Clean up the dataframe
    df_final.sort_values('timestamp', inplace=True)
    df_final.set_index('timestamp', inplace=True)

    # Create a complete hourly index and interpolate missing values
    full_hourly_index = pd.date_range(start=df_final.index.min(), end=df_final.index.max(), freq='H')
    df_final = df_final.reindex(full_hourly_index).interpolate(method='time')

    # Fill any remaining NaNs (usually at the very beginning)
    df_final.fillna(0, inplace=True)
    df_final.reset_index(inplace=True)
    df_final.rename(columns={'index': 'timestamp'}, inplace=True)

    # Select and order the final columns
    final_cols = ['timestamp', 'water_level_cm', 'precip_klaipedos-ams_mm', 'precip_vezaiciu-ams_mm', 'temp_c']
    df_final = df_final[final_cols]

    print(f"\n✅ Collected and combined {len(df_final)} unique hourly records.")
    df_final.to_csv(DATA_FILE, index=False, float_format='%.4f')
    print(f"💾 Saved final historical dataset to {DATA_FILE}")


if __name__ == "__main__":
    main()