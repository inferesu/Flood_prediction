import pandas as pd
import numpy as np
from sklearn.svm import SVR
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
        df = df.set_index(date_col).sort_index()

        # Ensure target is the first column for easier inverse scaling later
        cols = [target_col] + [col for col in df.columns if col != target_col]
        df = df[cols]

        # SVR works best with scaled data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)

        return df, scaled_data, scaler
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        raise


def create_flattened_sequences(dataset, look_back=1):
    """
    For SVR, we flatten the look_back window into a 1D vector of features.
    If look_back=3 and features=5, each input row becomes 15 features.
    """
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        # Grab the window and flatten it from (look_back, n_features) to (look_back * n_features,)
        window = dataset[i:(i + look_back), :].flatten()
        X.append(window)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


print("--- Loading and preparing data for SVR ---")
train_df, scaled_train_data, scaler = load_and_preprocess_data(TRAIN_CSV, DATE_COLUMN, TARGET_COLUMN)
N_FEATURES = train_df.shape[1]

X_train, y_train = create_flattened_sequences(scaled_train_data, LOOK_BACK_PERIOD)

test_df, scaled_test_data, _ = load_and_preprocess_data(TEST_CSV, DATE_COLUMN, TARGET_COLUMN)
# Combine end of train with test to ensure the first test rows have enough history
combined_data = np.concatenate((scaled_train_data[-LOOK_BACK_PERIOD:], scaled_test_data))
X_test, y_test = create_flattened_sequences(combined_data, LOOK_BACK_PERIOD)

# --- 3. Build and Train the SVR Model ---

print("--- Training SVR model ---")
# C: Regularization, epsilon: tube of insensitivity, gamma: kernel coefficient
model = SVR(kernel='rbf', C=100, epsilon=0.01, gamma='scale')
model.fit(X_train, y_train)

# --- 4. Make Predictions on Test Data ---

print("--- Making predictions ---")
test_predictions_scaled = model.predict(X_test)

# To inverse scale, we need an array of shape (n_samples, n_features)
dummy_array = np.zeros((len(test_predictions_scaled), N_FEATURES))
dummy_array[:, 0] = test_predictions_scaled
test_predictions = scaler.inverse_transform(dummy_array)[:, 0]

actual_water_levels = test_df[TARGET_COLUMN].values

# --- 5. Evaluate and Visualize ---

print("\n--- Evaluating SVR performance ---")

mse = mean_squared_error(actual_water_levels, test_predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_water_levels, test_predictions)
r2 = r2_score(actual_water_levels, test_predictions)
nrmse = rmse / (np.max(actual_water_levels) - np.min(actual_water_levels))

print(f'Test MAE:   {mae:.3f}')
print(f'Test MSE:   {mse:.3f}')
print(f'Test RMSE:  {rmse:.3f}')
print(f'Test NRMSE: {nrmse:.3f} ({(nrmse * 100):.2f}%)')
print(f'Test R2:    {r2:.3f}')

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(test_df.index, actual_water_levels, color='blue', label='Actual Water Level', alpha=0.6)
plt.plot(test_df.index, test_predictions, color='green', linestyle='--', label='SVR Prediction', alpha=0.8)
plt.title('Minija Water Level Prediction: SVR Model')
plt.xlabel('Date')
plt.ylabel('Water Level (cm)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()