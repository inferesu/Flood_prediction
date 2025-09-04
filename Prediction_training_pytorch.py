import pandas as pd
import torch
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader

# CORRECTED IMPORTS: We now import the specific classes we need directly from their files.
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# --- Configuration ---
TRAIN_DATA_FILE = 'minija_multi_precip_data_2020-2023.csv'
TEST_DATA_FILE = 'new_test.csv'  # Your manually collected test data


def prepare_features(df):
    """Applies the same feature engineering to any dataframe."""
    df.sort_index(inplace=True)
    df.interpolate(method='time', inplace=True)
    # Short-term memory
    df['precip_klaipedos_lag_12h'] = df['precip_klaipedos-ams_mm'].rolling(window=12, min_periods=1).sum()
    df['precip_klaipedos_lag_24h'] = df['precip_klaipedos-ams_mm'].rolling(window=24, min_periods=1).sum()
    # Long-term memory - NEW FEATURES
    df['precip_klaipedos_lag_48h'] = df['precip_klaipedos-ams_mm'].rolling(window=48, min_periods=1).sum()
    df['precip_klaipedos_lag_72h'] = df['precip_klaipedos-ams_mm'].rolling(window=72, min_periods=1).sum()

    # Short-term memory
    df['precip_vezaiciu_lag_12h'] = df['precip_vezaiciu-ams_mm'].rolling(window=12, min_periods=1).sum()
    df['precip_vezaiciu_lag_24h'] = df['precip_vezaiciu-ams_mm'].rolling(window=24, min_periods=1).sum()
    # Long-term memory - NEW FEATURES
    df['precip_vezaiciu_lag_48h'] = df['precip_vezaiciu-ams_mm'].rolling(window=48, min_periods=1).sum()
    df['precip_vezaiciu_lag_72h'] = df['precip_vezaiciu-ams_mm'].rolling(window=72, min_periods=1).sum()

    df['target_change'] = df['water_level_cm'].shift(-1) - df['water_level_cm']
    df.dropna(inplace=True)
    return df


def run_full_evaluation_pytorch():
    """Trains and evaluates the model using the anfis-pytorch library."""
    print("--- Step 1: Loading All Datasets ---")
    df_train = pd.read_csv(TRAIN_DATA_FILE, parse_dates=['timestamp'], index_col='timestamp')
    df_test = pd.read_csv(TEST_DATA_FILE, parse_dates=['timestamp'], index_col='timestamp')
    print(f"✅ Loaded {len(df_train)} training rows and {len(df_test)} test rows.")

    print("\n--- Step 2: Preparing Features ---")
    df_train = prepare_features(df_train)
    df_test = prepare_features(df_test)

    # UPDATED: New, more comprehensive feature list
    features_list = [
        'precip_klaipedos_lag_12h', 'precip_klaipedos_lag_24h', 'precip_klaipedos_lag_48h', 'precip_klaipedos_lag_72h',
        'precip_vezaiciu_lag_12h', 'precip_vezaiciu_lag_24h', 'precip_vezaiciu_lag_48h', 'precip_vezaiciu_lag_72h'
    ]
    target = 'target_change'
    X_train, y_train = df_train[features_list].values, df_train[target].values
    X_test, y_true_change = df_test[features_list].values, df_test[target].values

    print("\n--- Step 3: Scaling Data ---")
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    X_test_scaled = scaler_X.transform(X_test)

    x_train_tensor = torch.from_numpy(X_train_scaled).float()
    y_train_tensor = torch.from_numpy(y_train_scaled).float()
    x_test_tensor = torch.from_numpy(X_test_scaled).float()
    train_dl = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=16, shuffle=True)
    print("✅ Data scaled and converted to Tensors.")

    print("\n--- Step 4: Defining the ANFIS Model ---")
    num_mfs = 3  # Reduced MFs slightly as we have more input features
    num_inputs = len(features_list)

    invardefs = []
    for i in range(num_inputs):
        mfs = [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(num_mfs)]
        invardefs.append((f'x{i}', mfs))

    outvars = ['y']
    model = AnfisNet('Flood Prediction Model', invardefs, outvars, hybrid=True)
    print("✅ Model defined successfully with Hybrid Learning enabled.")

    print("\n--- Step 5: Training the ANFIS Model ---")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    criterion = torch.nn.MSELoss()
    epochs = 200

    for epoch in range(epochs):
        for x_batch, y_batch in train_dl:
            model.train()
            optimizer.zero_grad()
            y_pred_scaled = model(x_batch)
            loss = criterion(y_pred_scaled, y_batch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.fit_coeff(x_train_tensor, y_train_tensor)

        if (epoch + 1) % 10 == 0:
            y_pred_final = model(x_train_tensor)
            epoch_loss = criterion(y_pred_final, y_train_tensor)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss.item():.4f}')
    print("✅ Model training complete.")

    print("\n--- Step 6: Evaluating Model on Test Data ---")
    model.eval()
    with torch.no_grad():
        y_pred_scaled_tensor = model(x_test_tensor)

    y_pred_scaled = y_pred_scaled_tensor.numpy()
    y_pred_change = scaler_y.inverse_transform(y_pred_scaled).flatten()

    base_water_level = df_test['water_level_cm'].values
    predicted_level = base_water_level + y_pred_change
    actual_level = base_water_level + y_true_change

    print("\n--- FINAL EVALUATION RESULTS ---")
    r2 = r2_score(actual_level, predicted_level)
    mae = mean_absolute_error(actual_level, predicted_level)
    print(f"\n- R-squared (R²): {r2:.4f}")
    print(f"- Mean Absolute Error (MAE): {mae:.2f} cm")

    results_df = pd.DataFrame({
        'Timestamp': df_test.index,
        'Actual Water Level (cm)': actual_level,
        'Predicted Water Level (cm)': predicted_level
    })
    print("\n--- Predictions vs. Actuals ---")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    run_full_evaluation_pytorch()
