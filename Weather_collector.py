import requests
import pandas as pd
from datetime import datetime

# --- Configuration ---
# The official place codes for Klaipėda and Vėžaičiai from the meteo.lt API
PLACE_CODES = ['klaipeda', 'vezaiciai']
# The output file where the latest forecast will be saved
OUTPUT_FILE = 'latest_forecast.csv'


def fetch_long_term_forecast(place_code: str) -> pd.DataFrame:
    """
    Fetches the long-term forecast for a single place code from the meteo.lt API.

    Args:
        place_code: The official code for the location (e.g., 'klaipeda').

    Returns:
        A pandas DataFrame with the forecast data or None if an error occurs.
    """
    print(f"--> Fetching long-term forecast for '{place_code}'...")
    all_forecasts = []
    try:
        url = f"https://api.meteo.lt/v1/places/{place_code}/forecasts/long-term"
        response = requests.get(url, timeout=20)
        # Raise an exception if the API returns an error (e.g., 404, 500)
        response.raise_for_status()
        data = response.json()

        forecast_timestamps = data.get('forecastTimestamps', [])
        if not forecast_timestamps:
            print(f"   - Warning: No forecast timestamps found for '{place_code}'.")
            return None

        for ts in forecast_timestamps:
            all_forecasts.append({
                'timestamp': ts.get('forecastTimeUtc'),
                # Use totalPrecipitation as it's the sum over the forecast interval
                f'precip_{place_code}_mm': ts.get('totalPrecipitation'),
                f'temp_{place_code}_c': ts.get('airTemperature'),
            })

        print(f"   - Successfully fetched {len(all_forecasts)} forecast hours for '{place_code}'.")
        return pd.DataFrame(all_forecasts)

    except requests.exceptions.RequestException as e:
        print(f"   - ❌ ERROR fetching forecast for '{place_code}': {e}")
        return None


def main():
    """
    Main function to collect, process, and save the latest weather forecasts.
    """
    print(f"--- Starting Weather Forecast Collection at {datetime.now()} ---")
    print(f"Fetching forecasts for places: {PLACE_CODES}")

    forecast_dfs = []
    for code in PLACE_CODES:
        df = fetch_long_term_forecast(code)
        if df is not None and not df.empty:
            forecast_dfs.append(df)

    if len(forecast_dfs) < len(PLACE_CODES):
        print("\n--- ❌ Could not fetch data for all required places. Aborting. ---")
        return

    print("\n--- Step 1: Combining and Processing Forecast Data ---")

    # Start with the first dataframe and iteratively merge the others
    df_final = forecast_dfs[0]
    for i in range(1, len(forecast_dfs)):
        df_final = pd.merge(df_final, forecast_dfs[i], on='timestamp', how='outer')

    df_final['timestamp'] = pd.to_datetime(df_final['timestamp'])

    # --- Step 2: Create Columns Compatible with Your Training Data ---
    # Average the temperature from both locations
    temp_cols = [col for col in df_final.columns if 'temp' in col]
    df_final['temp_c'] = df_final[temp_cols].mean(axis=1)

    # IMPORTANT: Rename precipitation columns to match the names used in your training script
    # This ensures the feature engineering function will work correctly.
    df_final.rename(columns={
        'precip_klaipeda_mm': 'precip_klaipedos-ams_mm',
        'precip_vezaiciai_mm': 'precip_vezaiciu-ams_mm',
    }, inplace=True)

    # Select only the columns needed for the forecasting model
    final_columns = ['timestamp', 'precip_klaipedos-ams_mm', 'precip_vezaiciu-ams_mm', 'temp_c']
    df_final = df_final[final_columns]

    # --- Step 3: Ensure a Clean, Hourly Timeseries ---
    df_final.set_index('timestamp', inplace=True)
    # The API forecast can have gaps (e.g., every 3 hours). We create a full
    # hourly index and interpolate to fill the missing hours smoothly.
    start_time = df_final.index.min()
    end_time = df_final.index.max()
    full_hourly_index = pd.date_range(start=start_time, end=end_time, freq='H')
    df_final = df_final.reindex(full_hourly_index).interpolate(method='time')

    df_final.reset_index(inplace=True)
    df_final.rename(columns={'index': 'timestamp'}, inplace=True)

    print(
        f"\n✅ Successfully processed forecast data from {df_final['timestamp'].min()} to {df_final['timestamp'].max()}")

    # Save the final, clean forecast data to a CSV file
    df_final.to_csv(OUTPUT_FILE, index=False, float_format='%.4f')
    print(f"💾 Saved latest forecast to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
