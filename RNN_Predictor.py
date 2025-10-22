import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- 1. Configuration ---
# You can adjust these parameters
LOOK_BACK_PERIOD = 3  # Number of previous time steps to use as input to predict the next step
# N_FEATURES will be determined dynamically from the data
TRAIN_CSV = 'minija_multi_precip_data_2015-2022.csv'
TEST_CSV = 'new_test.csv'
DATE_COLUMN = 'timestamp'  # The name of the date/time column in your CSV
TARGET_COLUMN = 'water_level_cm'  # The name of the column we want to predict


# --- 2. Data Loading and Preprocessing ---

def load_and_preprocess_data(filepath, date_col, target_col):
    """
    Loads and preprocesses the time-series data from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        # Check if date and target columns exist
        if date_col not in df.columns:
            raise ValueError(f"Column '{date_col}' not found in {filepath}. Please check your CSV header.")
        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found in {filepath}. Please check your CSV header.")

        # Convert timestamp column to datetime objects
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        raise

    # Ensure target column is the first column for consistency in sequence creation
    cols = [target_col] + [col for col in df.columns if col != target_col]
    df = df[cols]

    # Ensure data is sorted by date
    df = df.sort_index()

    # We will scale the features to be between 0 and 1, which is good practice for NNs
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    return df, scaled_data, scaler


def create_sequences(dataset, look_back=1):
    """Converts an array of values into a dataset of sequences."""
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        # The input sequence (X) is the data for the look_back period
        X.append(dataset[i:(i + look_back), :])
        # The output (Y) is the water level of the next day
        Y.append(dataset[i + look_back, 0])  # Index 0 is always our target column ('water_level_cm')
    return np.array(X), np.array(Y)


print("--- Loading and preparing data ---")
# Load training data
train_df, scaled_train_data, scaler = load_and_preprocess_data(TRAIN_CSV, DATE_COLUMN, TARGET_COLUMN)
N_FEATURES = len(train_df.columns)  # Dynamically set number of features
X_train, y_train = create_sequences(scaled_train_data, LOOK_BACK_PERIOD)

# Load test data
test_df, scaled_test_data, _ = load_and_preprocess_data(TEST_CSV, DATE_COLUMN, TARGET_COLUMN)
# We need to prepend some training data to the test data to have enough history for the first test prediction
combined_data = np.concatenate((scaled_train_data[-LOOK_BACK_PERIOD:], scaled_test_data))
X_test, y_test = create_sequences(combined_data, LOOK_BACK_PERIOD)

# --- 3. Build and Train the RNN (LSTM) Model ---

print("--- Building and training the LSTM model ---")
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(LOOK_BACK_PERIOD, N_FEATURES)),
    tf.keras.layers.Dense(1)  # Output layer with one neuron to predict the single water_level value
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=2, validation_data=(X_test, y_test))

# --- 4. Make Predictions on Test Data ---

print("--- Making predictions on test data ---")
test_predictions_scaled = model.predict(X_test)

# We need to inverse the scaling to get the actual water level values
# The scaler was fitted on N_FEATURES columns, so we create a dummy array with the same shape
# to inverse the transform.
dummy_array_for_inverse = np.zeros((len(test_predictions_scaled), N_FEATURES))
dummy_array_for_inverse[:, 0] = test_predictions_scaled.ravel()
test_predictions = scaler.inverse_transform(dummy_array_for_inverse)[:, 0]

# Get the actual values for comparison. It should be the entire column.
actual_water_levels = test_df[TARGET_COLUMN].values

# --- 5. Evaluate and Visualize the Results ---

print("--- Evaluating model performance ---")
# Calculate Root Mean Squared Error
rmse = np.sqrt(mean_squared_error(actual_water_levels, test_predictions))
print(f'Test RMSE: {rmse:.3f}')

# Calculate R-squared (R2) score
r2 = r2_score(actual_water_levels, test_predictions)
print(f'Test R2 Score: {r2:.3f}')

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(test_df.index, actual_water_levels, color='blue', label='Actual Water Level')
plt.plot(test_df.index, test_predictions, color='red', linestyle='--', label='Predicted Water Level')
plt.title('Water Level Prediction - Test Set')
plt.xlabel('Date')
plt.ylabel('Water Level (cm)')
plt.legend()
plt.grid(True)
plt.show()