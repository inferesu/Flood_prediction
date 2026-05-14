import json, random, torch, joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

# --- Config ---
SEED = 42
NUM_MFS = 5
NUM_EPOCHS = 300
random.seed(SEED);
np.random.seed(SEED);
torch.manual_seed(SEED)
K_DECAY = 0.85  #
TRAIN_DATA_FILE = '../minija_complex_data_2024.csv'
FEATURES_LIST = ['API_norm', 'S_t', 'SMI_t', 'Pt', 'delta_WL_t']
TARGET = 'target_change'


def prepare_complex_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()
    df['Pt'] = df[['precip_klaipedos-ams', 'precip_vezaiciu-ams']].mean(axis=1)  #

    # API calculation: APIt = Pt + k * APIt-1 (Eq. 9)
    api_vals, curr_api = [], 0
    for p in df['Pt']:
        curr_api = p + (K_DECAY * curr_api)
        api_vals.append(curr_api)
    df['API_t'] = api_vals

    # API Normalization (Eq. 10)
    df['API_norm'] = (df['API_t'] - df['API_t'].min()) / (df['API_t'].max() - df['API_t'].min())

    # Seasonality Encoding (Eq. 11)
    d = pd.to_datetime(df['timestamp']).dt.dayofyear
    df['S_t'] = np.cos((2 * np.pi * d) / 365)

    # Snowmelt Index (SMI_t) (Eq. 17)
    avg_t = df[['temp_klaipedos-ams', 'temp_vezaiciu-ams']].mean(axis=1)
    df['SMI_t'] = avg_t.apply(lambda x: max(0, x * 2.5) if x > 0 else 0)

    # Trend Persistence (Eq. 21)
    df['delta_WL_t'] = df['water_level_cm'].diff().fillna(0)
    df['target_change'] = df['water_level_cm'].shift(-1) - df['water_level_cm']

    return df.dropna()


def build_anfis(num_inputs, num_mfs):
    invardefs = []
    for i in range(num_inputs):
        # Bell functions to capture thresholds like theta_API (Eq. 15)
        mfs = [BellMembFunc(torch.rand(1), torch.rand(1), torch.rand(1)) for _ in range(num_mfs)]
        invardefs.append((f'x{i}', mfs))
    return AnfisNet('Flood Model', invardefs, ['y'], hybrid=True)


def train_and_save():
    df = prepare_complex_features(pd.read_csv(TRAIN_DATA_FILE))
    X, y = df[FEATURES_LIST].values, df[TARGET].values.reshape(-1, 1)

    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_scaled, y_scaled = scaler_X.fit_transform(X), scaler_y.fit_transform(y)

    joblib.dump(scaler_X, "../Scalers/scaler_X.pkl");
    joblib.dump(scaler_y, "scaler_y.pkl")

    model = build_anfis(len(FEATURES_LIST), NUM_MFS)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    criterion = torch.nn.MSELoss()

    x_t, y_t = torch.tensor(X_scaled).float(), torch.tensor(y_scaled).float()
    loader = DataLoader(TensorDataset(x_t, y_t), batch_size=16, shuffle=True)

    for epoch in range(300):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb);
            loss.backward();
            optimizer.step()
        with torch.no_grad():
            model.fit_coeff(x_t, y_t)
        if (epoch + 1) % 20 == 0: print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")

    torch.save({'model_state_dict': model.state_dict(), 'coeff': model.coeff}, "../ANFIS_models/anfis_model.pth")


if __name__ == "__main__":
    train_and_save()