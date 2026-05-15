"""Feature engineering for the ANFIS flood prediction model."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from config.constants import K_DECAY, MF_LABELS


def prepare_complex_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp")

    # Average precipitation across all precip columns
    p_cols = [c for c in df.columns if "precip" in c]
    df["Pt"] = df[p_cols].mean(axis=1) if p_cols else 0.0

    # Antecedent Precipitation Index
    api_vals, current_api = [], 0.0
    for p in df["Pt"]:
        current_api = p + (K_DECAY * current_api)
        api_vals.append(current_api)
    df["API_t"] = api_vals

    a_min, a_max = df["API_t"].min(), df["API_t"].max()
    df["API_norm"] = (
        (df["API_t"] - a_min) / (a_max - a_min) if a_max != a_min else 0.0
    )

    # Seasonality
    doy = pd.to_datetime(df["timestamp"]).dt.dayofyear
    df["S_t"] = np.cos((2 * np.pi * doy) / 365)

    # Snowmelt index from temperature
    t_cols = [c for c in df.columns if "temp" in c]
    if t_cols:
        avg_temp = df[t_cols].mean(axis=1)
        df["SMI_t"] = avg_temp.apply(lambda x: max(0, x * 2.5) if x > 0 else 0)
    else:
        df["SMI_t"] = 0.0

    # Water level trend
    df["delta_WL_t"] = df["water_level_cm"].diff().fillna(0)

    return df


def extract_strongest_rule(model, X_scaled: np.ndarray, num_mfs: int) -> Tuple[int, float, List[int]]:
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled).float()
        fuzzified = model.layer["fuzzify"](X_tensor)
        firing_strengths = model.layer["rules"](fuzzified)

    strengths = firing_strengths[0].numpy()
    rule_id = int(np.argmax(strengths))
    activation = float(strengths[rule_id])

    # Decode rule index into per-input MF indices
    mf_indices = []
    temp = rule_id
    for _ in range(5):
        mf_indices.append(temp % num_mfs)
        temp //= num_mfs

    return rule_id, activation, list(reversed(mf_indices))


def format_rule_string(rule_id: int, activation: float, mf_indices: List[int]) -> str:
    feature_names = ["API_norm", "S_t", "SMI_t", "Pt", "delta_WL_t"]
    parts = [
        f"{fn} is {MF_LABELS[fn][mf_indices[i]]}"
        for i, fn in enumerate(feature_names)
    ]
    return f"Rule #{rule_id} (Act: {activation:.2f}): IF " + " AND ".join(parts)
