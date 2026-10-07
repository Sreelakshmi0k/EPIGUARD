import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_auc_score, roc_curve
)

print("="*60)
print("[Step 1/4] Loading engineered features...")
print("="*60)

feat_path = 'outputs/eeg_engineered_features.csv'
if not os.path.exists(feat_path):
    feat_path = 'eeg_engineered_features_ready_for_ml.csv'

df = pd.read_csv(feat_path)
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'SVM (RBF Kernel)': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
    'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5)
}

print("\n" + "="*60)
print("[Step 2/4] Training and benchmarking classifiers...")
print("="*60)

results = {}
for name, clf in models.items():
    print(f"Training {name}...")
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    
    t1 = time.time()
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    latency = (time.time() - t1) / len(X_test) * 1000
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    acc = accuracy_score(y_test, y_pred)
    prec, _, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    auc = roc_auc_score(y_test, y_prob)
    
    results[name] = {
        'Accuracy': acc, 'Sensitivity': sensitivity, 'Specificity': specificity,
        'Precision': prec, 'F1-Score': f1, 'ROC-AUC': auc, 'Latency': latency,
        'cm': (tn, fp, fn, tp), 'fpr_tpr': roc_curve(y_test, y_prob)
    }

print("\n" + "="*60)
print("BENCHMARK COMPARISON TABLE")
print("="*60)
summary_data = []
for name, r in results.items():
    summary_data.append({
        'Model': name,
        'Accuracy (%)': f"{r['Accuracy']*100:.2f}%",
        'Sensitivity (%)': f"{r['Sensitivity']*100:.2f}%",
        'Specificity (%)': f"{r['Specificity']*100:.2f}%",
        'Precision (%)': f"{r['Precision']*100:.2f}%",
        'F1-Score': f"{r['F1-Score']:.4f}",
        'ROC-AUC': f"{r['ROC-AUC']:.4f}",
        'Latency (ms)': f"{r['Latency']:.3f}"
    })
print(pd.DataFrame(summary_data).to_string(index=False))

print("\n" + "="*60)
print("[Step 3/4] Running 5-Fold Stratified Cross Validation...")
print("="*60)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(models['Random Forest'], X, y, cv=cv, scoring='roc_auc')
print(f"Random Forest 5-Fold ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print("\n" + "="*60)
print("[Step 4/4] Generating evaluation graphs...")
print("="*60)
os.makedirs('outputs', exist_ok=True)
plt.figure(figsize=(15, 4.5))

# Subplot 1: ROC Curves
plt.subplot(1, 3, 1)
for name, r in results.items():
    fpr, tpr, _ = r['fpr_tpr']
    plt.plot(fpr, tpr, label=f"{name} (AUC={r['ROC-AUC']:.3f})")
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curves')
plt.legend(loc='lower right', fontsize=8)
plt.grid(True, alpha=0.3)

# Subplot 2: Random Forest Confusion Matrix
plt.subplot(1, 3, 2)
tn, fp, fn, tp = results['Random Forest']['cm']
cm = np.array([[tn, fp], [fn, tp]])
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title('Random Forest Confusion Matrix')
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", 
                 color="white" if cm[i, j] > 1000 else "black", fontweight='bold')
plt.xticks([0, 1], ['Non-Seizure', 'Seizure'])
plt.yticks([0, 1], ['Non-Seizure', 'Seizure'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')

# Subplot 3: Feature Importance
plt.subplot(1, 3, 3)
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
idx = np.argsort(importances)
plt.barh(range(len(idx)), importances[idx], color='#2b6cb0')
plt.yticks(range(len(idx)), [X.columns[i] for i in idx], fontsize=8)
plt.xlabel('Gini Feature Importance')
plt.title('Top Predictive Features')

plt.tight_layout()
plt.savefig('outputs/ml_evaluation_results.png', dpi=300)
print("Graph saved successfully as: outputs/ml_evaluation_results.png")
print("="*60)