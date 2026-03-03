import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ================= CONFIG =================
PERCENTILE_THRESHOLD = 0.90  # Look at the top 10% highest water levels
ANFIS_FILE = "anfis_predictions.csv"
RNN_FILE = "rnn_predictions.csv"
XGB_FILE = "xgboost_predictions.csv"

# ================= LOAD & MERGE DATA =================
print("Loading model predictions...")

try:
    # Load data
    df_anfis = pd.read_csv(ANFIS_FILE)
    df_rnn = pd.read_csv(RNN_FILE)
    df_xgb = pd.read_csv(XGB_FILE)

    # Convert timestamps to datetime objects for precise matching
    df_anfis['timestamp'] = pd.to_datetime(df_anfis['timestamp'])
    df_rnn['timestamp'] = pd.to_datetime(df_rnn['timestamp'])
    df_xgb['timestamp'] = pd.to_datetime(df_xgb['timestamp'])

    # Rename prediction columns so we know which is which
    df_anfis = df_anfis[['timestamp', 'WL_true', 'WL_pred']].rename(columns={'WL_pred': 'ANFIS_pred'})
    df_rnn = df_rnn[['timestamp', 'WL_pred']].rename(columns={'WL_pred': 'RNN_pred'})
    df_xgb = df_xgb[['timestamp', 'WL_pred']].rename(columns={'WL_pred': 'XGB_pred'})

    # Merge everything on the exact same timestamps
    # Inner merge automatically drops the 14 days ANFIS predicted that the others couldn't
    merged_df = df_anfis.merge(df_rnn, on='timestamp', how='inner')
    merged_df = merged_df.merge(df_xgb, on='timestamp', how='inner')

    print(f"Successfully aligned {len(merged_df)} daily predictions across all models.\n")

except FileNotFoundError as e:
    print(f"Error: Could not find one of the CSV files. Please make sure you ran the evaluation scripts. ({e})")
    exit()

# ================= PEAK EVENT FILTERING =================
# Find the threshold for the top 10% (or whatever PERCENTILE_THRESHOLD is set to)
flood_threshold = merged_df['WL_true'].quantile(PERCENTILE_THRESHOLD)
peak_events = merged_df[merged_df['WL_true'] >= flood_threshold]

print(f"--- PEAK EVENT ANALYSIS (Top {int((1 - PERCENTILE_THRESHOLD) * 100)}% of Water Levels) ---")
print(f"Flood Threshold   : >= {flood_threshold:.2f} cm")
print(f"Number of days    : {len(peak_events)} days")
print("-" * 60)

# ================= EVALUATE =================
models = ['ANFIS_pred', 'RNN_pred', 'XGB_pred']
results = []

actual_peaks = peak_events['WL_true'].values

for model_col in models:
    preds = peak_events[model_col].values

    mse = mean_squared_error(actual_peaks, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_peaks, preds)

    # We don't use R2 here because R2 behaves poorly on heavily truncated datasets
    results.append({
        'Model': model_col.replace('_pred', ''),
        'RMSE (cm)': round(rmse, 3),
        'MAE (cm)': round(mae, 3)
    })

# Create a nice Pandas DataFrame for the thesis table
results_df = pd.DataFrame(results).set_index('Model')
print(results_df)

# ================= VISUALIZATION =================
# 1. Bar Chart of Errors during Floods
plt.figure(figsize=(10, 6))
bars = plt.bar(results_df.index, results_df['MAE (cm)'], color=['blue', 'orange', 'green'], alpha=0.8)
plt.title(f"Mean Absolute Error During Floods (Top {int((1 - PERCENTILE_THRESHOLD) * 100)}% Water Levels)", fontsize=14)
plt.ylabel("Error in centimeters (Lower is better)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add exact numbers on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.2, f"{yval} cm", ha='center', va='bottom', fontweight='bold')

plt.show()

# 2. Time Series Snippet of the Highest Peak
# Find the index of the absolute highest water level
max_idx = peak_events['WL_true'].idxmax()

# Grab a 30-day window around the biggest flood event
window_start = max(0, max_idx - 15)
window_end = min(len(merged_df), max_idx + 15)
flood_window = merged_df.iloc[window_start:window_end]

plt.figure(figsize=(14, 6))
plt.plot(flood_window['timestamp'], flood_window['WL_true'], label='Actual Water Level', color='black', linewidth=3)
plt.plot(flood_window['timestamp'], flood_window['ANFIS_pred'], label='ANFIS', linestyle='--', linewidth=2)
plt.plot(flood_window['timestamp'], flood_window['RNN_pred'], label='RNN', linestyle='-.', linewidth=2)
plt.plot(flood_window['timestamp'], flood_window['XGB_pred'], label='XGBoost', linestyle=':', linewidth=2)

plt.axhline(flood_threshold, color='red', linestyle='-', alpha=0.3, label='Flood Threshold')
plt.title("Model Behavior During the Most Extreme Flood Event", fontsize=16)
plt.ylabel("Water Level (cm)", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()