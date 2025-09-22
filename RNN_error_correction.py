import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random

# --- Import ANFIS classes to be able to load the model ---
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# --- Configuration & Reproducibility ---
# ADDED: Defined SEED and set random states for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- File Paths ---
ANFIS_MODEL_PATH = "anfis_model.pth"
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_Y.pkl"
CONFIG_JSON_PATH = "training_config.json"
TRAIN_DATA_FILE = 'minija_multi_precip_data_2015-2022.csv'

RNN_MODEL_SAVE_PATH = "rnn_corrector_multivariate.pth"
SCALER_ERROR_PATH = "scaler_error.pkl"

# --- RNN Hyperparameters ---
SEQUENCE_LENGTH = 24
HIDDEN_SIZE = 60
NUM_LAYERS = 2
EPOCHS = 200  # Set a high number, early stopping will find the best one
BATCH_SIZE = 64
LR = 0.001

# --- Early Stopping Configuration ---
EARLY_STOPPING_PATIENCE = 10  # Stop if validation loss doesn't improve for 10 epochs
VALIDATION_SPLIT_SIZE = 0.2  # Use 20% of the data for validation


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """A copy of the feature prep function from the ANFIS training script."""
    df = df.copy()
    df.sort_index(inplace=True)
    df.interpolate(method='time', inplace=True)
    df['precip_klaipedos_lag_12h'] = df['precip_klaipedos-ams_mm'].rolling(12, min_periods=1).sum()
    df['precip_klaipedos_lag_24h'] = df['precip_klaipedos-ams_mm'].rolling(24, min_periods=1).sum()
    df['precip_klaipedos_lag_48h'] = df['precip_klaipedos-ams_mm'].rolling(48, min_periods=1).sum()
    df['precip_klaipedos_lag_72h'] = df['precip_klaipedos-ams_mm'].rolling(72, min_periods=1).sum()
    df['precip_vezaiciu_lag_12h'] = df['precip_vezaiciu-ams_mm'].rolling(12, min_periods=1).sum()
    df['precip_vezaiciu_lag_24h'] = df['precip_vezaiciu-ams_mm'].rolling(24, min_periods=1).sum()
    df['precip_vezaiciu_lag_48h'] = df['precip_vezaiciu-ams_mm'].rolling(48, min_periods=1).sum()
    df['precip_vezaiciu_lag_72h'] = df['precip_vezaiciu-ams_mm'].rolling(72, min_periods=1).sum()
    df['target_change'] = df['water_level_cm'].shift(-1) - df['water_level_cm']
    df.dropna(inplace=True)
    return df


def build_anfis(num_inputs: int, num_mfs: int) -> AnfisNet:
    """Reconstructs the ANFIS model structure."""
    invardefs = []
    for i in range(num_inputs):
        mfs = [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(num_mfs)]
        invardefs.append((f'x{i}', mfs))
    return AnfisNet('Flood Prediction Model', invardefs, ['y'], hybrid=True)


def create_sequences(data, seq_length):
    """Turns a 2D array of features into a 3D array of overlapping sequences."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0]  # Target is the error, which is the first column
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys).reshape(-1, 1)


class ErrorCorrectorRNN(nn.Module):
    """The multivariate RNN model structure."""

    def __init__(self, input_size, hidden_size=50, num_layers=2, output_size=1):
        super(ErrorCorrectorRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_rnn_with_early_stopping():
    """Main function to generate errors and train the RNN corrector with early stopping."""
    print("--- Step 1: Loading Trained ANFIS Model and Data ---")
    with open(CONFIG_JSON_PATH, "r") as f:
        config = json.load(f)
    features_list = config["features_list"]

    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    anfis_model = build_anfis(config["num_inputs"], config["num_mfs"])
    checkpoint = torch.load(ANFIS_MODEL_PATH, map_location="cpu")
    anfis_model.load_state_dict(checkpoint['model_state_dict'])
    anfis_model.coeff = checkpoint['consequent_coeffs']
    anfis_model.eval()
    print("✅ ANFIS model loaded successfully.")

    print("\n--- Step 2: Generating ANFIS Error and Context Dataset ---")
    df_train = pd.read_csv(TRAIN_DATA_FILE, parse_dates=['timestamp'], index_col='timestamp')
    df_train = prepare_features(df_train)

    X_train_unscaled = df_train[features_list].values
    y_train_actual = df_train[config["target"]].values

    X_train_scaled = scaler_X.transform(X_train_unscaled)
    with torch.no_grad():
        y_pred_scaled_tensor = anfis_model(torch.from_numpy(X_train_scaled).float())
    y_pred_anfis = scaler_y.inverse_transform(y_pred_scaled_tensor.numpy()).flatten()

    errors = y_train_actual - y_pred_anfis

    print("\n--- Step 3: Preparing Data for RNN ---")
    scaler_error = MinMaxScaler(feature_range=(-1, 1))
    errors_scaled = scaler_error.fit_transform(errors.reshape(-1, 1))
    joblib.dump(scaler_error, SCALER_ERROR_PATH)
    print(f"✅ Error scaler saved to {SCALER_ERROR_PATH}")

    context_data_scaled = np.hstack((errors_scaled, X_train_scaled))
    X_seq, y_seq = create_sequences(context_data_scaled, SEQUENCE_LENGTH)

    # --- Split into Training and Validation Sets ---
    X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
        X_seq, y_seq, test_size=VALIDATION_SPLIT_SIZE, random_state=SEED, shuffle=False
    )
    print(f"✅ Data split into {len(X_train_seq)} training sequences and {len(X_val_seq)} validation sequences.")

    X_train_t = torch.from_numpy(X_train_seq).float()
    y_train_t = torch.from_numpy(y_train_seq).float()
    X_val_t = torch.from_numpy(X_val_seq).float()
    y_val_t = torch.from_numpy(y_val_seq).float()

    train_data = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)

    print("\n--- Step 4: Training the RNN Error Corrector with Early Stopping ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = context_data_scaled.shape[1]
    rnn_model = ErrorCorrectorRNN(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=LR)

    # --- Early Stopping Logic ---
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        rnn_model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = rnn_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation phase
        rnn_model.eval()
        with torch.no_grad():
            val_outputs = rnn_model(X_val_t.to(device))
            val_loss = criterion(val_outputs, y_val_t.to(device))

        print(f'Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(rnn_model.state_dict(), RNN_MODEL_SAVE_PATH)
            print(f"   -> New best model saved with validation loss: {best_val_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"   -> No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("\n--- Early stopping triggered ---")
            break

    print("\n✅ Training complete. Best model saved to", RNN_MODEL_SAVE_PATH)


if __name__ == "__main__":
    train_rnn_with_early_stopping()
