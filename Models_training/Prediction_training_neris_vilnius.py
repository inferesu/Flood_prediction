import json
import random

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc

SEED       = 42
NUM_MFS    = 5
NUM_EPOCHS = 300
K_DECAY    = 0.85
LR         = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TRAIN_DATA_FILE = '../neris_vilnius_complex_data_2024.csv'  # ← changed
FEATURES_LIST   = ['API_norm', 'S_t', 'SMI_t', 'Pt', 'delta_WL_t']
TARGET          = 'target_change'

MODEL_FILE    = '../ANFIS_models/neris_vilnius_anfis_model.pth'  # ← changed
SCALER_X_FILE = '../Scalers/neris_vilnius_scaler_X.pkl'  # ← changed
SCALER_Y_FILE = '../Scalers/neris_vilnius_scaler_y.pkl'  # ← changed
CONFIG_FILE   = '../Training_configs/neris_vilnius_training_config.json'  # ← changed


def prepare_complex_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()

    df['precip_vilniaus-ams'] = df['precip_vilniaus-ams'].clip(lower=0)
    df['water_level_cm']      = df['water_level_cm'].clip(lower=0)

    df['Pt'] = df['precip_vilniaus-ams']

    api_vals, curr_api = [], 0
    for p in df['Pt']:
        curr_api = p + (K_DECAY * curr_api)
        api_vals.append(curr_api)
    df['API_t'] = np.array(api_vals).clip(min=0)

    api_min, api_max = df['API_t'].min(), df['API_t'].max()
    df['API_norm'] = (
        ((df['API_t'] - api_min) / (api_max - api_min)).clip(0.0, 1.0)
        if api_max != api_min else 0.0
    )

    d = pd.to_datetime(df['timestamp']).dt.dayofyear
    df['S_t'] = np.cos((2 * np.pi * d) / 365)

    df['SMI_t'] = df['temp_vilniaus-ams'].apply(
        lambda x: max(0, x * 2.5) if x > 0 else 0
    )

    df['delta_WL_t']    = df['water_level_cm'].diff().fillna(0)
    df['target_change'] = df['water_level_cm'].shift(-1) - df['water_level_cm']

    return df.dropna()


def build_anfis(num_inputs: int, num_mfs: int) -> AnfisNet:
    centres   = torch.linspace(0.1, 0.9, num_mfs)
    invardefs = []
    for i in range(num_inputs):
        mfs = [
            BellMembFunc(
                torch.tensor([1.0]),
                torch.tensor([2.0]),
                centres[j].unsqueeze(0),
            )
            for j in range(num_mfs)
        ]
        invardefs.append((f'x{i}', mfs))
    return AnfisNet('Neris Vilnius Flood Model', invardefs, ['y'], hybrid=True)  # ← changed


def has_nan_params(model: torch.nn.Module) -> bool:
    return any(torch.isnan(p).any().item() for p in model.parameters())


def train_and_save():
    print("=" * 60)
    print("   🌊 NERIS–VILNIUS ANFIS TRAINING PIPELINE")            # ← changed
    print(f"   Input  : {TRAIN_DATA_FILE}")
    print(f"   Model  : {MODEL_FILE}")
    print(f"   Station: vilniaus-ams / vilniaus-neris-vms")          # ← changed
    print(f"   LR     : {LR}")
    print("=" * 60)

    print("\n━━━ STEP 1: Loading & Preparing Features ━━━━━━━━━━━━━━━━")
    df = prepare_complex_features(pd.read_csv(TRAIN_DATA_FILE))
    print(f"   ✔️  Dataset loaded — {len(df)} rows after dropna")

    X = df[FEATURES_LIST].values
    y = df[TARGET].values.reshape(-1, 1)
    print(f"   ✔️  X shape: {X.shape}  |  y shape: {y.shape}")
    print(f"   ✔️  X range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"   ✔️  y range: [{y.min():.3f}, {y.max():.3f}]")

    print("\n━━━ STEP 2: Scaling ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    joblib.dump(scaler_X, SCALER_X_FILE)
    joblib.dump(scaler_y, SCALER_Y_FILE)
    print(f"   ✔️  Scalers saved → {SCALER_X_FILE}, {SCALER_Y_FILE}")

    print("\n━━━ STEP 3: Building ANFIS Model ━━━━━━━━━━━━━━━━━━━━━━━━")
    model        = build_anfis(len(FEATURES_LIST), NUM_MFS)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✔️  ANFIS built — inputs={len(FEATURES_LIST)}, MFs={NUM_MFS}, params={total_params}")

    assert not has_nan_params(model), "NaN in model parameters at init — check BellMembFunc"
    print("   ✔️  Parameter init verified — no NaN")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    x_t    = torch.tensor(X_scaled).float()
    y_t    = torch.tensor(y_scaled).float()
    loader = DataLoader(TensorDataset(x_t, y_t), batch_size=32, shuffle=True)

    print(f"\n━━━ STEP 4: Training ({NUM_EPOCHS} epochs) ━━━━━━━━━━━━━━━━━━━━")
    best_loss  = float('inf')
    best_state = None

    for epoch in range(NUM_EPOCHS):

        if has_nan_params(model):
            print(f"\n   ⚠️  NaN detected at epoch {epoch} — stopping early.")
            if best_state is not None:
                model.load_state_dict(best_state)
            break

        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)

            if torch.isnan(loss):
                print(f"   ⚠️  NaN loss at epoch {epoch} — skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        with torch.no_grad():
            model.fit_coeff(x_t, y_t)

        avg_loss = epoch_loss / max(len(loader), 1)

        if avg_loss < best_loss and not has_nan_params(model):
            best_loss  = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            nan_flag = " ⚠️ NaN params!" if has_nan_params(model) else ""
            print(f"   📉 Epoch [{epoch + 1:3d}/{NUM_EPOCHS}]  "
                  f"Loss: {avg_loss:.6f}  |  Best: {best_loss:.6f}{nan_flag}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n   ✔️  Best model state restored (loss={best_loss:.6f})")

    if has_nan_params(model):
        print("\n   ❌ Model still contains NaN — not saving.")
        return

    print(f"\n━━━ STEP 5: Saving Model & Config ━━━━━━━━━━━━━━━━━━━━━━")

    coeff = model.coeff
    if isinstance(coeff, np.ndarray):
        coeff = torch.tensor(coeff, dtype=torch.float32)

    if coeff is not None and torch.isnan(coeff).any():
        print("   ⚠️  coeff contains NaN — running fit_coeff one final time.")
        with torch.no_grad():
            model.fit_coeff(x_t, y_t)
        coeff = model.coeff
        if isinstance(coeff, np.ndarray):
            coeff = torch.tensor(coeff, dtype=torch.float32)

    torch.save({
        'model_state_dict': model.state_dict(),
        'coeff':            coeff,
        'features':         FEATURES_LIST,
        'num_mfs':          NUM_MFS,
        'k_decay':          K_DECAY,
    }, MODEL_FILE)
    print(f"   ✔️  Model saved → {MODEL_FILE}")

    with open(CONFIG_FILE, 'w') as f:
        json.dump({
            'num_mfs':    NUM_MFS,
            'features':   FEATURES_LIST,
            'k_decay':    K_DECAY,
            'num_epochs': NUM_EPOCHS,
        }, f, indent=4)
    print(f"   ✔️  Training config saved → {CONFIG_FILE}")

    print("\n" + "=" * 60)
    print(f"   ✅ Training complete!")
    print(f"   📉 Final best loss : {best_loss:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    train_and_save()
