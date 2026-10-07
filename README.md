# 🧠 Automated Epileptic Seizure Detection from EEG Signals Using Machine Learning

## 📖 Project Overview

This project provides an automated, computer-aided diagnostic (CAD) software pipeline for **epileptic seizure recognition and continuous timeline monitoring** from electroencephalogram (EEG) recordings. 

Visual inspection of 24–72 hour long-term scalp EEG recordings by clinical neurologists is tedious, subjective, and prone to human diagnostic fatigue. This system automates the diagnostic pipeline by processing continuous EEG signal streams, extracting multi-domain biomarkers, pinpointing the exact onset, cessation, and duration of seizure episodes second by second, and escalating critical alerts through a 4-stage clinical triage system.

---

## 🔬 Core Methodology & Pipeline

1. **Dataset Ingestion & Structuring:**
   - Validated on the benchmark **University of Bonn / UCI Epileptic Seizure Recognition Dataset** (11,500 1-second epochs, 178 temporal voltage points per epoch sampled at $\approx 173.61$ Hz).
   - Ingests pre-segmented records or continuous 1D time-series signals, segmenting them into sequential 1-second analytical windows.

2. **Preprocessing & Artifact Mitigation:**
   - Removal of non-cerebral recording tags and metadata.
   - Diagnostic binarization: Ictal seizure events ($y = 1$) vs. Non-seizure physiological baselines ($y \in \{2, 3, 4, 5\}$).
   - Two-sided percentile Winsorization (clipping at 0.5% and 99.5%) to suppress electrode pop and muscle motion artifacts without distorting paroxysmal spike morphology.

3. **15 Multi-Domain Feature Extraction:**
   - **Time-Domain Statistics:** Mean, Standard Deviation, Variance, Peak-to-Peak Amplitude, RMS, Skewness, Kurtosis, Zero-Crossing Rate (ZCR).
   - **Spectral Dynamics:** Total Signal Energy, Spectral Entropy.
   - **Sub-Band Rhythms (FFT):** Delta ($<4$ Hz), Theta ($4\text{--}8$ Hz), Alpha ($8\text{--}13$ Hz), Beta ($13\text{--}30$ Hz), Gamma ($>30$ Hz).
   - **Standardization:** Harmonized using population Z-score normalization (`scaler.pkl`).

4. **Machine Learning Classifier:**
   - Balanced Random Forest Ensemble ($150$ estimators) with cost-sensitive class weighting to address the natural $4:1$ baseline-to-seizure distribution.
   - Achieves **98.26% Accuracy**, **95.65% Sensitivity (Seizure Recall)**, **98.91% Specificity**, and **0.9980 ROC-AUC** with an inference latency of **12–15 ms** per 1-second epoch.

5. **4-Stage Clinical Alert Triage & Interactive Dashboard:**
   - **Stage 1 & 2 (Normal Baseline):** Healthy physiological baselines (relaxed eyes-closed / alert eyes-open); silent operation, green indicator.
   - **Stage 3 (Orange Alert, Frequency $\le 5$):** Mild or isolated seizure spikes; amber status card and single 750 Hz buzzer tone.
   - **Stage 4 (Critical Red Emergency, Frequency $> 5$):** Acute continuous seizure attack; pulsing red banner, high-frequency multi-tone siren (1,800–3,200 Hz), and emergency first-aid instruction table.
   - **Timeline Inspection:** Interactive slider to examine raw voltage waveforms, spectral statistics, and confidence probabilities for any individual second.
