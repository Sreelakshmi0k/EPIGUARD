import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

print("="*60)
print("[Step 1/4] Loading raw EEG dataset...")
print("="*60)

# Check file path
csv_path = 'data/Epileptic Seizure Recognition.csv'
if not os.path.exists(csv_path):
    csv_path = 'Epileptic Seizure Recognition.csv'

df = pd.read_csv(csv_path)
print(f"Dataset Loaded Successfully! Shape: {df.shape}")

# Drop identifier column if present
clean_df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col]).copy()

# Binarize label: 1 = Seizure (Ictal), 2-5 = Non-seizure
clean_df['target'] = (clean_df['y'] == 1).astype(int)
clean_df.drop(columns=['y'], inplace=True)
print(f"Target Distribution:\n{clean_df['target'].value_counts()}")

feature_cols = [f'X{i}' for i in range(1, 179)]

# Outlier treatment via Winsorization (0.5% and 99.5% quantiles)
q_low = clean_df[feature_cols].quantile(0.005)
q_high = clean_df[feature_cols].quantile(0.995)
clean_df[feature_cols] = clean_df[feature_cols].clip(lower=q_low, upper=q_high, axis=1)

print("\n" + "="*60)
print("[Step 2/4] Extracting statistical & sub-band frequency features...")
print("="*60)

def extract_features(row):
    mean_val = np.mean(row)
    std_val = np.std(row)
    var_val = np.var(row)
    peak_to_peak = np.ptp(row)
    skew_val = float(pd.Series(row).skew())
    kurt_val = float(pd.Series(row).kurt())
    rms_val = np.sqrt(np.mean(row**2))
    zero_crossings = np.sum(np.diff(np.sign(row)) != 0)
    
    # FFT sub-band decomposition
    fft_vals = np.abs(np.fft.rfft(row))[1:]
    total_energy = np.sum(fft_vals**2)
    prob = fft_vals / (np.sum(fft_vals) + 1e-12)
    spec_entropy = -np.sum(prob * np.log2(prob + 1e-12))
    
    delta_power = np.sum(fft_vals[0:4]**2)
    theta_power = np.sum(fft_vals[4:8]**2)
    alpha_power = np.sum(fft_vals[8:13]**2)
    beta_power  = np.sum(fft_vals[13:30]**2)
    gamma_power = np.sum(fft_vals[30:]**2)
    
    return [
        mean_val, std_val, var_val, peak_to_peak, skew_val, kurt_val, rms_val, zero_crossings,
        total_energy, spec_entropy, delta_power, theta_power, alpha_power, beta_power, gamma_power
    ]

feature_names = [
    'mean', 'std', 'variance', 'peak_to_peak', 'skewness', 'kurtosis', 'rms', 'zero_crossings',
    'total_energy', 'spectral_entropy', 'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power'
]

raw_feat_matrix = np.array([extract_features(row) for row in clean_df[feature_cols].values])

print("\n" + "="*60)
print("[Step 3/4] Scaling feature vectors (Z-score Standardization)...")
print("="*60)

scaler = StandardScaler()
scaled_feat_matrix = scaler.fit_transform(raw_feat_matrix)

engineered_df = pd.DataFrame(scaled_feat_matrix, columns=feature_names)
engineered_df['target'] = clean_df['target'].values

os.makedirs('outputs', exist_ok=True)
out_path = 'outputs/eeg_engineered_features.csv'
engineered_df.to_csv(out_path, index=False)

print("\n" + "="*60)
print(f"[Step 4/4] Completed! Feature matrix saved to: {out_path}")
print(f"Final Processed Dimensions: {engineered_df.shape}")
print("="*60)