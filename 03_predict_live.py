import os
import joblib
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier

# Check data and model paths
csv_path = 'outputs/eeg_engineered_features.csv'
if not os.path.exists(csv_path):
    csv_path = 'eeg_engineered_features_ready_for_ml.csv'

model_path = 'outputs/random_forest_eeg_model.pkl'

# Ensure model exists
if not os.path.exists(model_path):
    print("Initializing model...")
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['target'])
    y = df['target']
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X, y)
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(rf, model_path)

model = joblib.load(model_path)
raw_df = pd.read_csv(csv_path)

def analyze_eeg_epoch(features_df_row, test_id="Sample", patient_desc="Standard EEG Recording"):
    t0 = time.time()
    pred = model.predict(features_df_row)[0]
    prob = model.predict_proba(features_df_row)[0]
    latency = (time.time() - t0) * 1000
    
    print("\n" + "="*60)
    print(f"             PATIENT EEG TEST REPORT: {test_id}")
    print("="*60)
    print(f"  CLINICAL PROFILE : {patient_desc}")
    print("-" * 60)
    
    if pred == 1:
        print("  DIAGNOSIS        : [!] EPILEPSY SEIZURE DETECTED")
        print("  WAVE PATTERN     : Abnormal / Hypersynchronous Ictal Activity")
        print(f"  SEIZURE RISK     : {prob[1]*100:.1f}%")
        print("  RECOMMENDATION   : Immediate medical alert required!")
    else:
        print("  DIAGNOSIS        : [OK] NO SEIZURE DETECTED")
        print("  WAVE PATTERN     : Normal Physiological Rhythm")
        print(f"  NORMAL CONFIDENCE: {prob[0]*100:.1f}%")
        print("  RECOMMENDATION   : Brain activity within normal baseline limits.")
        
    print("-" * 60)
    print(f"  Analysis Latency : {latency:.2f} ms (Real-time Edge Classification)")
    print("="*60)

# 5 distinct sample test cases (3 Non-Seizure across baseline variations + 2 Active Seizures)
non_seizure_pool = raw_df[raw_df['target'] == 0].drop(columns=['target'])
seizure_pool = raw_df[raw_df['target'] == 1].drop(columns=['target'])

test_cases = [
    (non_seizure_pool.iloc[[0]],  "Patient A (Case 1)", "Routine Check-up / Relaxed Awake State"),
    (seizure_pool.iloc[[0]],      "Patient B (Case 2)", "Sudden Spike Discharge / Suspected Seizure"),
    (non_seizure_pool.iloc[[10]], "Patient C (Case 3)", "Resting Brainwave / Eyes Closed Baseline"),
    (seizure_pool.iloc[[15]],     "Patient D (Case 4)", "Acute Convulsive Episode / Severe Rhythmic Spiking"),
    (non_seizure_pool.iloc[[25]], "Patient E (Case 5)", "Mild Cognitive Task / Baseline Activity")
]

print("Running 5 Patient EEG Test Case Evaluations...")
for feat_row, case_name, desc in test_cases:
    analyze_eeg_epoch(feat_row, test_id=case_name, patient_desc=desc)