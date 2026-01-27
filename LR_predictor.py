import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# --- 1. Configuration ---
LOOK_BACK_PERIOD = 3
TRAIN_CSV = 'minija_multi_precip_data_2015-2022.csv'
TEST_CSV = 'tests2015-2022.csv'
DATE_COLUMN = 'timestamp'
TARGET_COLUMN = 'water_level_cm'

# --- 2. Data Loading and Preprocessing ---

def load_and_preprocess_data(filepath, date_col, target_col):
    try:
        df = pd.read_csv(filepath)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        raise

    # Ensure target column is first for easier inverse scaling
    cols = [target_col] + [col for col in df.columns if col != target_col]
    df = df[cols]
    df = df.sort_index()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    return df, scaled_data, scaler

def create_flat_sequences(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        # Flattening 3D temporal data into 2D for Linear Regression
        window = dataset[i:(i + look_back), :].flatten()
        X.append(window)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

print("--- Loading and preparing data ---")
train_df, scaled_train_data, scaler = load_and_preprocess_data(TRAIN_CSV, DATE_COLUMN, TARGET_COLUMN)
N_FEATURES = len(train_df.columns)
X_train, y_train = create_flat_sequences(scaled_train_data, LOOK_BACK_PERIOD)

test_df, scaled_test_data, _ = load_and_preprocess_data(TEST_CSV, DATE_COLUMN, TARGET_COLUMN)
combined_data = np.concatenate((scaled_train_data[-LOOK_BACK_PERIOD:], scaled_test_data))
X_test, y_test = create_flat_sequences(combined_data, LOOK_BACK_PERIOD)

# --- 3. Build and Train the Linear Regression Model ---

print("--- Training Linear Regression model ---")
model = LinearRegression()
model.fit(X_train, y_train)

# --- 4. Make Predictions on Test Data ---

print("--- Making predictions on test data ---")
test_predictions_scaled = model.predict(X_test)

# Prepare for inverse transform
dummy_array_for_inverse = np.zeros((len(test_predictions_scaled), N_FEATURES))
dummy_array_for_inverse[:, 0] = test_predictions_scaled
test_predictions = scaler.inverse_transform(dummy_array_for_inverse)[:, 0]

actual_water_levels = test_df[TARGET_COLUMN].values

# --- 5. Evaluate and Visualize ---

print("\n--- Evaluating Linear Regression performance ---")

# Standard Metrics
mse = mean_squared_error(actual_water_levels, test_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(actual_water_levels, test_predictions)
mae = mean_absolute_error(actual_water_levels, test_predictions)

# NRMSE (Normalized Root Mean Squared Error)
# Formula: RMSE / (max_actual - min_actual)
data_range = np.max(actual_water_levels) - np.min(actual_water_levels)
nrmse = rmse / data_range

print(f'Test MSE:   {mse:.3f}')
print(f'Test RMSE:  {rmse:.3f}')
print(f'Test R2:    {r2:.3f}')
print(f'Test MAE:   {mae:.3f}')
print(f'Test NRMSE: {nrmse:.3f} ({(nrmse * 100):.2f}%)')

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(test_df.index, actual_water_levels, color='blue', label='Actual Water Level', alpha=0.7)
plt.plot(test_df.index, test_predictions, color='green', linestyle='--', label='LR Prediction', alpha=0.8)
plt.title('Minija River Flood Prediction - Linear Regression')
plt.xlabel('Date')
plt.ylabel('Water Level (cm)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()