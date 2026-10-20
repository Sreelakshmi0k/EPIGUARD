from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = Path(__file__).resolve().parent

# Check standard file locations for your extracted 15-feature matrix
csv_candidates = [
    BASE_DIR / 'outputs' / 'eeg_engineered_features.csv',
    BASE_DIR / 'eeg_engineered_features.csv',
    BASE_DIR / 'eeg_engineered_features_ready_for_ml.csv',
    BASE_DIR / 'outputs' / 'eeg_engineered_features_ready_for_ml.csv',
]

csv_path = None
for candidate in csv_candidates:
    if candidate.exists():
        csv_path = candidate
        break

if csv_path is None:
    raise FileNotFoundError(
        "Could not find the engineered features CSV file. "
        "Please check the filename in your project folder."
    )

print(f"Loading data from: {csv_path.name}")
df = pd.read_csv(csv_path)

# Drop non-feature columns if present
drop_cols = [col for col in ['target', 'y', 'Unnamed: 0'] if col in df.columns]
X = df.drop(columns=drop_cols)

# Fit standardizer on the 15 features
scaler = StandardScaler()
scaler.fit(X)

# Export scaler.pkl directly to the project root
out_path = BASE_DIR / 'scaler.pkl'
joblib.dump(scaler, out_path)
print(f"Success: 'scaler.pkl' created at {out_path}")