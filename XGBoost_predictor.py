import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os

# --- 1. Configuration ---
LOOK_BACK_PERIOD = 3  # History window (same as your RNN)
TRAIN_CSV = 'minija_multi_precip_data_2015-2022.csv'
TEST_CSV = 'tests2015-2022.csv'
DATE_COLUMN = 'timestamp'
TARGET_COLUMN = 'water_level_cm'

# --- 2. Data Loading and Preprocessing ---

def load_and_preprocess_data(filepath, date_col, target_col):
    """Loads data and ensures the target is the first column."""
    try:
        df = pd.read_csv(filepath)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        raise

    # Reorder: Target first, then others
    cols = [target_col] + [col for col in df.columns if col != target_col]
    df = df[cols]
    return df.sort_index()

def create_flattened_sequences(data_array, look_back=1):
    """
    Converts data into sequences and flattens them for XGBoost.
    Resulting shape: [samples, look_back * n_features]
    """
    X, Y = [], []
    for i in range(len(data_array) - look_back):
        # Grab the window and flatten it into a 1D vector
        window = data_array[i:(i + look_back), :].flatten()
        X.append(window)
        # Target is the first column (index 0) of the next row
        Y.append(data_array[i + look_back, 0])
    return np.array(X), np.array(Y)

print("--- Loading and preparing data ---")
train_df = load_and_preprocess_data(TRAIN_CSV, DATE_COLUMN, TARGET_COLUMN)
test_df = load_and_preprocess_data(TEST_CSV, DATE_COLUMN, TARGET_COLUMN)

# Convert dataframes to numpy arrays
train_values = train_df.values
test_values = test_df.values

# Create sequences
X_train, y_train = create_flattened_sequences(train_values, LOOK_BACK_PERIOD)

# Prepend training tail to test data to handle the look_back for the first test point
combined_test_values = np.concatenate((train_values[-LOOK_BACK_PERIOD:], test_values))
X_test, y_test = create_flattened_sequences(combined_test_values, LOOK_BACK_PERIOD)

# --- 3. Build and Train the XGBoost Model ---

print("--- Building and training the XGBoost model ---")
# Using XGBRegressor for continuous values (water level)
model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

# Train the model
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# --- 4. Make Predictions ---

print("--- Making predictions on test data ---")
test_predictions = model.predict(X_test)
actual_water_levels = y_test  # In this script, y_test is already the unscaled target

# --- 5. Evaluate and Visualize ---

print("\n--- Evaluating XGBoost model performance ---")

mse = mean_squared_error(actual_water_levels, test_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(actual_water_levels, test_predictions)
mae = mean_absolute_error(actual_water_levels, test_predictions)
nrmse = rmse / (np.max(actual_water_levels) - np.min(actual_water_levels))

print(f'Test MSE:   {mse:.3f}')
print(f'Test RMSE:  {rmse:.3f}')
print(f'Test R2:    {r2:.3f}')
print(f'Test MAE:   {mae:.3f}')
print(f'Test NRMSE: {nrmse:.3f} ({(nrmse * 100):.2f}%)')

# Plotting the 2023-2024 range as requested in previous steps
plt.figure(figsize=(15, 7))
plt.plot(test_df.index, actual_water_levels, color='blue', label='Actual', alpha=0.7)
plt.plot(test_df.index, test_predictions, color='green', linestyle='--', label='Predicted (XGBoost)', alpha=0.8)

# Filtering the plot view specifically for 2023-2024 if dates exist
if test_df.index.max() > pd.Timestamp('2023-01-01'):
    plt.xlim(pd.Timestamp('2023-01-01'), pd.Timestamp('2024-12-31'))

plt.title('Water Level Prediction - XGBoost Baseline')
plt.xlabel('Date')
plt.ylabel('Water Level (cm)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()