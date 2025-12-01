import pandas as pd
import requests
import time
from datetime import datetime, timedelta

# --- CONFIGURATION ---
# We mimic the structure expected by your app.py
STATIONS = [
    {
        "filename": "live_data.csv",
        "hydro_code": "priekules-vms",
        "meteo_codes": ["klaipedos-ams", "vezaiciu-ams"]
    },
    {
        "filename": "live_data_dane.csv",
        "hydro_code": "klaipedos-vms",
        "meteo_codes": ["klaipedos-ams"]
    }
]

DAYS_HISTORY = 90  # How many days back to fetch


def fetch_data(station_type, code, date_str):
    """Generic fetcher for Meteo.lt"""
    url = f"https://api.meteo.lt/v1/{station_type}/{code}/observations/{date_str}"
    if station_type == "hydro-stations":
        # Adjust URL for hydro measured data
        url = f"https://api.meteo.lt/v1/hydro-stations/{code}/observations/measured/{date_str}"

    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json().get("observations", [])
    except Exception as e:
        print(f"Error fetching {code} on {date_str}: {e}")
    return []


def run_seed():
    print(f"🌊 Starting Data Seeding for last {DAYS_HISTORY} days...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_HISTORY)

    for station in STATIONS:
        print(f"\n--- Processing {station['filename']} ---")
        all_rows = []

        # Loop through every single day
        for i in range(DAYS_HISTORY):
            current_date = start_date + timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")

            # 1. Fetch Hydro Data (Water Level)
            hydro_obs = fetch_data("hydro-stations", station['hydro_code'], date_str)
            # Filter for valid water levels and get daily average
            levels = [x['waterLevel'] for x in hydro_obs if x.get('waterLevel') is not None]

            if not levels:
                print(f"   Skipping {date_str}: No water level data.")
                continue

            avg_level = round(sum(levels) / len(levels), 2)

            # 2. Fetch Meteo Data (Precipitation)
            precip_data = {}
            for m_code in station['meteo_codes']:
                meteo_obs = fetch_data("stations", m_code, date_str)
                # Sum daily rain
                daily_rain = sum(x.get('precipitation', 0) for x in meteo_obs if x.get('precipitation') is not None)
                col_name = f"precip_{m_code}_mm"
                precip_data[col_name] = round(daily_rain, 2)

            # 3. Build Row
            row = {
                "timestamp": date_str,
                "water_level_cm": avg_level,
                **precip_data
            }
            all_rows.append(row)
            print(f"   ✅ {date_str}: Level {avg_level}cm")

            # Be nice to the API
            time.sleep(0.1)

        # 4. Save to CSV
        if all_rows:
            # Create DataFrame
            df = pd.DataFrame(all_rows)
            # Ensure columns are in the expected order
            cols = ['timestamp', 'water_level_cm'] + [f"precip_{c}_mm" for c in station['meteo_codes']]

            # Add missing columns if any (e.g. if a station had no rain data at all)
            for c in cols:
                if c not in df.columns:
                    df[c] = 0.0

            df = df[cols]  # Reorder
            df.to_csv(station['filename'], index=False)
            print(f"💾 Saved {len(df)} rows to {station['filename']}")
        else:
            print(f"⚠️ No data found for {station['filename']}")


if __name__ == "__main__":
    run_seed()