import pandas as pd
import os

# ================= CONFIGURATION =================
# 1. What is the name of YOUR existing file?
MY_SOURCE_FILE = "minija_multi_weather_data_2015-2022.csv"  # <--- Change this to your filename

# 2. Which river is this data for?
# Options: "minija" or "dane"
TARGET_RIVER = "minija"

# 3. Map YOUR column names to the APP's required names
# Format: "Your_Column_Name": "App_Required_Name"
COLUMN_MAPPING = {
    # CHANGE THE LEFT SIDE to match your CSV headers
    "Date": "timestamp",
    "WaterLevel": "water_level_cm",

    # If you have one rain column, map it to the main station
    "Precipitation": "precip_klaipedos-ams_mm",

    # If you have a second rain column (for Minija), map it here.
    # If you DON'T have a second rain column, comment this out.
    "Precipitation_Vezaiciai": "precip_vezaiciu-ams_mm"
}


# =================================================

def import_data():
    if not os.path.exists(MY_SOURCE_FILE):
        print(f"❌ Could not find file: {MY_SOURCE_FILE}")
        return

    print(f"📂 Reading {MY_SOURCE_FILE}...")
    df = pd.read_csv(MY_SOURCE_FILE)

    # 1. Rename Columns
    df = df.rename(columns=COLUMN_MAPPING)

    # 2. Fix Date Format (Ensure it is YYYY-MM-DD)
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"❌ Date Error: Make sure your date column is mapped correctly. {e}")
        return

    # 3. Handle Missing Columns (Crucial for Minija)
    # The app expects 'precip_vezaiciu-ams_mm' for Minija.
    # If you didn't have it, we fill it with 0 or copy the other rain column.
    if TARGET_RIVER == "minija":
        target_file = "live_data.csv"
        required_cols = ['timestamp', 'water_level_cm', 'precip_klaipedos-ams_mm', 'precip_vezaiciu-ams_mm']

        # If vezaiciai is missing, let's just copy klaipeda rain or set to 0 to prevent crash
        if 'precip_vezaiciu-ams_mm' not in df.columns:
            print("⚠️ 'precip_vezaiciu-ams_mm' missing. Filling with 0.")
            df['precip_vezaiciu-ams_mm'] = 0.0

    elif TARGET_RIVER == "dane":
        target_file = "live_data_dane.csv"
        required_cols = ['timestamp', 'water_level_cm', 'precip_klaipedos-ams_mm']

    # 4. Filter and Sort
    # Only keep the columns the app needs
    try:
        df = df[required_cols]
    except KeyError as e:
        print(f"❌ Column Missing: {e}")
        print(f"Your file columns are: {list(pd.read_csv(MY_SOURCE_FILE).columns)}")
        print("Check the COLUMN_MAPPING in the script.")
        return

    # Sort by date (Oldest to Newest)
    df = df.sort_values('timestamp')

    # 5. Save
    df.to_csv(target_file, index=False)
    print(f"✅ Success! Imported {len(df)} rows into '{target_file}'")
    print("🚀 You can now restart app.py and refresh the dashboard.")


if __name__ == "__main__":
    import_data()