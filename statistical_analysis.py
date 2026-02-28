# ==========================================
# HYDROMETEOROLOGICAL VARIABLE CLASSIFICATION
# Quintile-Based (5 Classes)
# Prints only threshold ranges
# ==========================================

import pandas as pd
import numpy as np

# ------------------------------------------
# 1. LOAD DATA
# ------------------------------------------
file_path = "minija_complex_data_2024.csv"
df = pd.read_csv(file_path)

# ------------------------------------------
# 2. VARIABLES TO CLASSIFY
# ------------------------------------------
variables = [
    "Pt",
    "API_t",
    "SMI_t",
    "S_t",
    "delta_WL_t"
]

# ------------------------------------------
# 3. ROBUST QUANTILE CLASSIFICATION FUNCTION
# ------------------------------------------
def classify_quintiles(series, n_classes=5):
    """
    Robust quantile classification.
    Automatically adjusts number of bins if duplicate edges occur.
    """

    series = series.dropna()

    # If column is not numeric, stop
    if not np.issubdtype(series.dtype, np.number):
        print(f"{series.name} is not numeric.")
        return None

    # Calculate quantile edges
    quantiles = np.linspace(0, 1, n_classes + 1)
    bin_edges = series.quantile(quantiles).unique()

    # Adjust number of classes if needed
    actual_classes = len(bin_edges) - 1

    if actual_classes < 2:
        print(f"Not enough variation in {series.name} to classify.")
        return None

    # Define labels dynamically
    base_labels = ["Low", "Medium", "High", "Very High", "Extreme"]
    labels = base_labels[:actual_classes]

    # Store thresholds
    thresholds = {}
    for i in range(actual_classes):
        thresholds[labels[i]] = (bin_edges[i], bin_edges[i + 1])

    return thresholds

def classify_precipitation(series):

    bins = [-0.01, 0, 2, 10, 30, series.max()]
    labels = ["No Rain", "Light", "Moderate", "Heavy", "Extreme"]

    thresholds = {}
    for i in range(len(labels)):
        thresholds[labels[i]] = (bins[i], bins[i+1])

    return thresholds
# ------------------------------------------
# 4. PRINT RESULTS
# ------------------------------------------

print("\nHYDROMETEOROLOGICAL VARIABLE CLASSIFICATION RESULTS\n")

for var in variables:

    if var not in df.columns:
        continue

    if var == "Pt":
        thresholds = classify_precipitation(df[var])
    else:
        thresholds = classify_quintiles(df[var])

    if thresholds is None:
        continue

    print("=" * 60)
    print(f"{var}")
    print("=" * 60)

    for category, (lower, upper) in thresholds.items():
        print(f"{category}: {round(lower, 2)} – {round(upper, 2)}")

    print("\n")