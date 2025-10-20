import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- 1. Configuration ---
# You can adjust these parameters
LOOK_BACK_PERIOD = 3  # Number of previous time steps to use as input to predict the next step
N_FEATURES = 2  # Number of features: 'water_level' and 'precipitation_forecast'
TRAIN_CSV = 'train_data.csv'
TEST_CSV = 'test_data.csv'


# --- 2. Data Loading and Preprocessing ---

def load_and_preprocess_data(filepath):
    """Loads and preprocesses the time-series data from a CSV file."""
    # Load the dataset
    df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')

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
        Y.append(dataset[i + look_back, 0])  # Index 0 corresponds to 'water_level'
    return np.array(X), np.array(Y)


print("--- Loading and preparing data ---")
# Load training data
train_df, scaled_train_data, scaler = load_and_preprocess_data(TRAIN_CSV)
X_train, y_train = create_sequences(scaled_train_data, LOOK_BACK_PERIOD)

# Load test data
test_df, scaled_test_data, _ = load_and_preprocess_data(TEST_CSV)
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
# The scaler was fitted on two columns, so we create a dummy array with the same shape
# to inverse the transform.
dummy_array_for_inverse = np.zeros((len(test_predictions_scaled), N_FEATURES))
dummy_array_for_inverse[:, 0] = test_predictions_scaled.ravel()
test_predictions = scaler.inverse_transform(dummy_array_for_inverse)[:, 0]

# Get the actual values for comparison
actual_water_levels = test_df['water_level'].values[LOOK_BACK_PERIOD:]

# --- 5. Evaluate and Visualize the Results ---

print("--- Evaluating model performance ---")
# Calculate Root Mean Squared Error
rmse = np.sqrt(mean_squared_error(actual_water_levels, test_predictions))
print(f'Test RMSE: {rmse:.3f}')

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(test_df.index[LOOK_BACK_PERIOD:], actual_water_levels, color='blue', label='Actual Water Level')
plt.plot(test_df.index[LOOK_BACK_PERIOD:], test_predictions, color='red', linestyle='--', label='Predicted Water Level')
plt.title('Water Level Prediction - Test Set')
plt.xlabel('Date')
plt.ylabel('Water Level (meters)')
plt.legend()
plt.grid(True)
plt.show()
