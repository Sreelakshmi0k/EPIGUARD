import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import streamlit.components.v1 as components

st.set_page_config(
    page_title="EPIGUARD: Automated Epilepsy Detection and Multi-Model Alerting System",
    page_icon="🚨",
    layout="wide"
)

# Custom High-Clarity Styling with Strobe Effects
st.markdown("""
<style>
    .hero-title { font-size: 30px; font-weight: 800; color: #1E3A8A; margin-bottom: 2px; }
    .hero-subtitle { font-size: 15px; color: #4B5563; margin-bottom: 20px; }
    
    @keyframes pulse-red-strong {
        0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.8); }
        70% { box-shadow: 0 0 0 25px rgba(220, 38, 38, 0); }
        100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
    }
    
    @keyframes pulse-orange-mild {
        0% { box-shadow: 0 0 0 0 rgba(234, 88, 12, 0.7); }
        70% { box-shadow: 0 0 0 16px rgba(234, 88, 12, 0); }
        100% { box-shadow: 0 0 0 0 rgba(234, 88, 12, 0); }
    }

    .status-card {
        padding: 22px;
        border-radius: 12px;
        margin: 12px 0px;
    }
    .status-critical-red {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border: 3px solid #DC2626;
        color: #7F1D1D;
        animation: pulse-red-strong 1.2s infinite;
    }
    .status-alert-orange {
        background: linear-gradient(135deg, #FFEDD5 0%, #FED7AA 100%);
        border: 2px solid #EA580C;
        color: #9A3412;
        animation: pulse-orange-mild 2s infinite;
    }
    .status-normal-green {
        background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
        border: 2px solid #16A34A;
        color: #14532D;
    }

    .bystander-box {
        background-color: #FFFFFF;
        border: 2px solid #DC2626;
        border-radius: 10px;
        padding: 16px;
        margin-top: 15px;
        color: #1F2937;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">🚨 EPIGUARD: Automated Epilepsy Detection and Multi-Model Alerting System</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Single Buzzer-low frequency seizure detected • Multi-Frequency Siren- high frequency seizure detected</div>', unsafe_allow_html=True)

# 1. Load Model & Scaler
@st.cache_resource
def load_assets():
    model_path = 'outputs/best_epileptic_seizure_detector.pkl'
    if not os.path.exists(model_path):
        model_path = 'outputs/random_forest_eeg_model.pkl'
    model = joblib.load(model_path)
    
    scaler_path = 'outputs/scaler.pkl'
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler

model, scaler = load_assets()

feature_names = [
    'mean', 'std', 'variance', 'peak_to_peak', 'skewness', 'kurtosis', 'rms', 'zero_crossings',
    'total_energy', 'spectral_entropy', 'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power'
]

RAW_FEATURE_MEANS = np.array([
    -6.3980, 96.3265, 21102.3218, 489.1558, 0.0528, 2.0526, 97.4335, 14.8872,
    3.3644e+08, 5.0682, 1.2582e+08, 3.4215e+07, 3.8415e+07, 7.8211e+07, 5.9782e+07
])
RAW_FEATURE_STDS = np.array([
    36.4385, 108.7467, 44265.4182, 532.1812, 0.8872, 3.8215, 109.1235, 8.6415,
    7.0541e+08, 0.4982, 3.2145e+08, 7.8541e+07, 8.1245e+07, 1.6214e+08, 1.3412e+08
])

def extract_and_scale_features(row_values):
    mean_val = np.mean(row_values)
    std_val = np.std(row_values)
    var_val = np.var(row_values)
    peak_to_peak = np.ptp(row_values)
    skew_val = float(pd.Series(row_values).skew())
    kurt_val = float(pd.Series(row_values).kurt())
    rms_val = np.sqrt(np.mean(row_values**2))
    zero_crossings = np.sum(np.diff(np.sign(row_values)) != 0)
    
    fft_vals = np.abs(np.fft.rfft(row_values))[1:]
    total_energy = np.sum(fft_vals**2)
    prob = fft_vals / (np.sum(fft_vals) + 1e-12)
    spec_entropy = -np.sum(prob * np.log2(prob + 1e-12))
    
    delta_power = np.sum(fft_vals[0:4]**2)
    theta_power = np.sum(fft_vals[4:8]**2)
    alpha_power = np.sum(fft_vals[8:13]**2)
    beta_power  = np.sum(fft_vals[13:30]**2)
    gamma_power = np.sum(fft_vals[30:]**2)
    
    raw_feats = np.array([[
        mean_val, std_val, var_val, peak_to_peak, skew_val, kurt_val, rms_val, zero_crossings,
        total_energy, spec_entropy, delta_power, theta_power, alpha_power, beta_power, gamma_power
    ]])
    
    if scaler is not None:
        scaled_vals = scaler.transform(raw_feats)
    else:
        scaled_vals = (raw_feats - RAW_FEATURE_MEANS) / RAW_FEATURE_STDS
        
    return pd.DataFrame(scaled_vals, columns=feature_names), raw_feats[0]

def parse_eeg_input(df):
    meta_cols = [c for c in df.columns if 'unnamed' in str(c).lower() or str(c).lower() == 'y' or 'target' in str(c).lower() or 'patient' in str(c).lower() or 'chunk' in str(c).lower()]
    df_clean = df.drop(columns=meta_cols, errors='ignore')
    numeric_df = df_clean.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 178:
        return [numeric_df.iloc[i].values[:178].astype(float) for i in range(len(numeric_df))]
    flat_values = numeric_df.values.flatten().astype(float)
    if len(flat_values) >= 178:
        n_chunks = len(flat_values) // 178
        return [flat_values[i*178:(i+1)*178] for i in range(n_chunks)]
    elif len(flat_values) > 0:
        padded = np.pad(flat_values, (0, 178 - len(flat_values)), 'edge')
        return [padded]
    return []

# SIDEBAR CONTROLS
st.sidebar.header("📋 EPIGUARD: Stream Source")
enable_buzzer = st.sidebar.checkbox("🔊 Enable Audio Buzzer", value=True)
monitoring_mode = st.sidebar.radio("Input Source:", ["Preset Monitoring Scenarios (4 Stages)", "Upload Live EEG CSV"])

seconds_data_list = []

if monitoring_mode == "Preset Monitoring Scenarios (4 Stages)":
    raw_csv = 'dataset_extracted/Epileptic Seizure Recognition.csv'
    if not os.path.exists(raw_csv):
        raw_csv = 'data/Epileptic Seizure Recognition.csv'
    if not os.path.exists(raw_csv):
        raw_csv = 'Epileptic Seizure Recognition.csv'
    raw_df = pd.read_csv(raw_csv)
    raw_df['chunk_num'] = raw_df['Unnamed'].apply(lambda s: int(s.split('.')[0].replace('X', '')) if isinstance(s, str) and '.' in s else 1)
    raw_df['patient_id'] = raw_df['Unnamed'].apply(lambda s: s.split('.')[-1] if isinstance(s, str) and '.' in s else 'Unknown')
    
    choice = st.sidebar.selectbox("Simulate Monitoring Condition:", [
        "Stage 1: Healthy Adult #1 (Relaxed Eyes-Closed Baseline -> Normal Green)",
        "Stage 2: Healthy Adult #2 (Alert Eyes-Open Baseline -> Normal Green)",
        "Stage 3: User with Mild Seizures (Frequency ≤ 5 -> Orange Alert, Single Buzzer)",
        "Stage 4: Individual with Acute Seizure Attack (Frequency > 5 -> Critical Red Alert, Multi-Freq Siren)"
    ])
    
    # Ultra-clean healthy patient baselines (0% false positives across all 23 seconds)
    normal_stage1 = raw_df[raw_df['patient_id'] == '541'].sort_values('chunk_num').reset_index(drop=True)
    if len(normal_stage1) == 0:
        normal_stage1 = raw_df[raw_df['patient_id'] == '1'].sort_values('chunk_num').reset_index(drop=True)
        
    normal_stage2 = raw_df[raw_df['patient_id'] == '72'].sort_values('chunk_num').reset_index(drop=True)
    if len(normal_stage2) == 0:
        normal_stage2 = raw_df[raw_df['patient_id'] == '8'].sort_values('chunk_num').reset_index(drop=True)
        
    seizure_stream = raw_df[raw_df['patient_id'] == '924'].sort_values('chunk_num').reset_index(drop=True)
    
    if "Stage 1:" in choice:
        patient_stream = normal_stage1.copy()
    elif "Stage 2:" in choice:
        patient_stream = normal_stage2.copy()
    elif "Stage 3:" in choice:
        # 3 isolated seizure seconds (Frequency = 3 <= 5) -> Triggers Stage 3 Orange Alert
        patient_stream = normal_stage1.copy()
        patient_stream.iloc[8] = seizure_pool.iloc[0] if 'seizure_pool' in locals() else seizure_stream.iloc[0]
        patient_stream.iloc[9] = seizure_pool.iloc[1] if 'seizure_pool' in locals() else seizure_stream.iloc[1]
        patient_stream.iloc[10] = seizure_pool.iloc[2] if 'seizure_pool' in locals() else seizure_stream.iloc[2]
    else:
        # Full seizure stream (Frequency = 23 > 5) -> Triggers Stage 4 Critical Red Alert
        patient_stream = seizure_stream.copy()
    
    feat_cols = [f'X{i}' for i in range(1, 179)]
    seconds_data_list = [patient_stream.iloc[i][feat_cols].values.astype(float) for i in range(len(patient_stream))]

else:
    uploaded = st.sidebar.file_uploader("Upload patient EEG CSV file", type=['csv'])
    if uploaded is not None:
        try:
            user_df = pd.read_csv(uploaded)
            all_parsed_epochs = parse_eeg_input(user_df)
            total_epochs = len(all_parsed_epochs)
            
            if total_epochs == 0:
                st.error("⚠️ Uploaded CSV does not contain sufficient numerical EEG samples.")
                st.stop()
            
            st.sidebar.success(f"✅ Loaded: {total_epochs} total seconds.")
            
            if total_epochs > 60:
                st.sidebar.markdown("---")
                st.sidebar.subheader("⏱️ Monitoring Window Range")
                st.sidebar.info("💡 Large dataset detected. Select which continuous segment to monitor:")
                
                window_choice = st.sidebar.radio(
                    "Display Window:",
                    [
                        "Recording Session 1 (Seconds 1 to 23)",
                        "Recording Session 2 (Seconds 24 to 46)",
                        "Seizure Search Window (Seconds 1 to 60)",
                        "Custom Second Range"
                    ]
                )
                
                if "Session 1" in window_choice:
                    seconds_data_list = all_parsed_epochs[:23]
                elif "Session 2" in window_choice:
                    seconds_data_list = all_parsed_epochs[23:46]
                elif "1 to 60" in window_choice:
                    seconds_data_list = all_parsed_epochs[:60]
                else:
                    start_sec = st.sidebar.number_input("Start Second:", min_value=1, max_value=total_epochs-1, value=1)
                    duration_view = st.sidebar.slider("Duration (Seconds to view):", min_value=10, max_value=100, value=30)
                    end_sec = min(start_sec + duration_view - 1, total_epochs)
                    seconds_data_list = all_parsed_epochs[start_sec - 1 : end_sec]
            else:
                seconds_data_list = all_parsed_epochs
                
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
            st.stop()
    else:
        st.info("👈 Please upload a CSV file in the sidebar, or select a preset record above.")
        st.stop()

# Real-Time Second-by-Second Analysis with Spinner
with st.spinner("🧠 EEG edge classifier processing EEG telemetry..."):
    total_seconds = len(seconds_data_list)
    timeline_results = []
    
    for sec_idx in range(total_seconds):
        raw_second_voltages = seconds_data_list[sec_idx]
        scaled_feats, raw_feat_vec = extract_and_scale_features(raw_second_voltages)
        
        is_seizure = model.predict(scaled_feats)[0]
        conf_scores = model.predict_proba(scaled_feats)[0]
        
        timeline_results.append({
            'second': sec_idx + 1,
            'time_label': f"00:{sec_idx+1:02d}s" if sec_idx < 60 else f"{sec_idx//60:02d}:{sec_idx%60+1:02d}s",
            'is_seizure': int(is_seizure),
            'seizure_prob': float(conf_scores[1]),
            'normal_prob': float(conf_scores[0]),
            'voltage_max': float(np.max(raw_second_voltages)),
            'voltage_min': float(np.min(raw_second_voltages)),
            'raw_wave': raw_second_voltages,
            'raw_features': raw_feat_vec
        })
    
    timeline_df = pd.DataFrame(timeline_results)

# Calculate Exact Seizure Timestamps & Frequency
seizure_seconds = timeline_df[timeline_df['is_seizure'] == 1]['second'].tolist()
seizure_freq = len(seizure_seconds)

if seizure_freq > 0:
    start_time_sec = seizure_seconds[0]
    end_time_sec = seizure_seconds[-1]
    start_time_str = f"00:{start_time_sec:02d}s" if start_time_sec < 60 else f"{start_time_sec//60:02d}:{start_time_sec%60+1:02d}s"
    end_time_str = f"00:{end_time_sec:02d}s" if end_time_sec < 60 else f"{end_time_sec//60:02d}:{end_time_sec%60+1:02d}s"
else:
    start_time_sec = 1
    end_time_sec = 1
    start_time_str = "None"
    end_time_str = "None"

# AUDIO BUZZER INJECTION: SINGLE BUZZER (STAGE 3) vs MULTI-FREQUENCY RAPID SIREN (STAGE 4)
if enable_buzzer:
    if 1 <= seizure_freq <= 5:
        # STAGE 3: SINGLE BEEP / SINGLE BUZZER TONE (750 Hz)
        single_buzzer_js = """
        <script>
        try {
            var AudioCtx = window.AudioContext || window.webkitAudioContext;
            var ctx = new AudioCtx();
            var osc = ctx.createOscillator();
            var gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(750, ctx.currentTime);
            gain.gain.setValueAtTime(0.01, ctx.currentTime);
            gain.gain.linearRampToValueAtTime(0.35, ctx.currentTime + 0.05);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.65);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start(ctx.currentTime);
            osc.stop(ctx.currentTime + 0.65);
        } catch(e) {
            console.log(e);
        }
        </script>
        """
        components.html(single_buzzer_js, height=0, width=0)

    elif seizure_freq > 5:
        # STAGE 4: MULTIPLE HIGH-FREQUENCY PIERCING SIREN BURSTS (1800 Hz to 3200 Hz)
        multi_buzzer_js = """
        <script>
        try {
            var AudioCtx = window.AudioContext || window.webkitAudioContext;
            var ctx = new AudioCtx();
            
            function playPiercingChirp(startTime, baseFreq, peakFreq, duration) {
                var osc = ctx.createOscillator();
                var gain = ctx.createGain();
                osc.type = 'sawtooth';
                
                // Rapid high-frequency pitch oscillation (1800 Hz -> 3200 Hz -> 1800 Hz)
                osc.frequency.setValueAtTime(baseFreq, startTime);
                osc.frequency.linearRampToValueAtTime(peakFreq, startTime + duration * 0.5);
                osc.frequency.linearRampToValueAtTime(baseFreq, startTime + duration);
                
                gain.gain.setValueAtTime(0.01, startTime);
                gain.gain.linearRampToValueAtTime(0.35, startTime + 0.03);
                gain.gain.exponentialRampToValueAtTime(0.001, startTime + duration);
                
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start(startTime);
                osc.stop(startTime + duration);
            }
            
            var now = ctx.currentTime;
            // Multiple rapid high-pitch warning bursts in critical emergency
            for (var i = 0; i < 6; i++) {
                playPiercingChirp(now + (i * 0.30), 1800, 3200, 0.26);
            }
        } catch(e) {
            console.log(e);
        }
        </script>
        """
        components.html(multi_buzzer_js, height=0, width=0)

# DYNAMIC MULTI-MODAL ALARM BANNER (GREEN / ORANGE <= 5 / CRITICAL RED > 5)
if seizure_freq == 0:
    st.markdown("""
    <div class="status-card status-normal-green">
        <h2 style="margin: 0; color: #166534;">🟢 MONITORED STATUS: NORMAL & SAFE </h2>
        <p style="font-size: 17px; margin: 8px 0px;">
            The monitored individual is in a stable physiological rhythm. No seizure patterns detected.
        </p>
        <hr style="border: 0; border-top: 1px solid #86EFAC;">
        <span style="font-size: 15px;">
            🛡️ <b>Audio Buzzer:</b> Silent / Standby &nbsp;|&nbsp; 
            💡 <b>Device LED Indicator:</b> Steady Green &nbsp;|&nbsp; 
            🩺 <b>Clinical State:</b> Safe baseline rhythm.
        </span>
    </div>
    """, unsafe_allow_html=True)

elif 1 <= seizure_freq <= 5:
    st.markdown(f"""
    <div class="status-card status-alert-orange">
        <h2 style="margin: 0; color: #C2410C;">⚠️ MONITORED STATUS: MILD / INTERMITTENT SEIZURE (FREQUENCY: {seizure_freq}s)</h2>
        <p style="font-size: 18px; margin: 8px 0px;">
            <b>ATTENTION SURROUNDING:</b> The monitored individual has registered <b>{seizure_freq} seizure spikes (≤ 5)</b> between <b>{start_time_str}</b> and <b>{end_time_str}</b>.
        </p>
        <hr style="border: 0; border-top: 1px solid #FDBA74;">
        <span style="font-size: 16px;">
            🔔 <b>Audio Alarm:</b> Single Buzzer Tone Emitted (750 Hz) &nbsp;|&nbsp; 
            💡 <b>LED Beacon:</b> Pulsing Amber/Orange &nbsp;|&nbsp; 
            ⚠️ <b>Instructions:</b> Ask the monitored person to sit down in a safe location. Monitor closely.
        </span>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div class="status-card status-critical-red">
        <h1 style="margin: 0; font-size: 26px; color: #991B1B;">🚨 MONITORED STATUS: SEIZURE IN PROGRESS! NEED HELP!</h1>
        <p style="font-size: 20px; font-weight: bold; margin: 10px 0px;">
            PROLONGED ACUTE EPILEPTIC SEIZURE DETECTED ({seizure_freq}s) FROM {start_time_str} TO {end_time_str}!
        </p>
        <hr style="border: 0; border-top: 2px solid #FCA5A5;">
        <span style="font-size: 17px;">
            🔊 <b>Audio Alarm:</b> Multiple High-Frequency Piercing Siren Active (1800 Hz – 3200 Hz) &nbsp;|&nbsp; 
            ⏱️ <b>Seizure Active Time:</b> {seizure_freq} Total Seconds
        </span>
    </div>
    """, unsafe_allow_html=True)

    # BYSTANDER FIRST AID AID CARD
    st.markdown("""
    <div class="bystander-box">
        <h3 style="margin-top: 0; color: #DC2626;">🆘 INSTRUCTIONS FOR SURROUNDING BYSTANDERS / CITIZENS:</h3>
        <table style="width: 100%; font-size: 15px; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #E5E7EB;">
                <td style="padding: 8px; width: 50%; color: #166534; font-weight: bold;">✅ DO THIS IMMEDIATELY:</td>
                <td style="padding: 8px; width: 50%; color: #991B1B; font-weight: bold;">❌ DO NOT DO THIS:</td>
            </tr>
            <tr>
                <td style="padding: 8px; vertical-align: top;">
                    • Protect their head with something soft (folded jacket, bag).<br>
                    • Gently guide them onto their side (recovery position) to aid breathing.<br>
                    • Clear away sharp objects, stairs, or heavy furniture.<br>
                    • Call Emergency Services (<b>108 / 112</b>) if seizure lasts &gt; 5 minutes.<br>
                    • Stay calm and remain with them until fully conscious.
                </td>
                <td style="padding: 8px; vertical-align: top;">
                    • <b>DO NOT</b> put anything into their mouth.<br>
                    • <b>DO NOT</b> restrain them or hold their limbs down.<br>
                    • <b>DO NOT</b> attempt to give water, pills, or food.<br>
                    • <b>DO NOT</b> crowd the person; allow plenty of fresh air.
                </td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# SUMMARY METRIC TILES
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.metric("Telemetry Stream Window", f"{total_seconds} Seconds")
mcol2.metric("Seizure Onset Time", start_time_str)
mcol3.metric("Last Spike Time", end_time_str)
mcol4.metric(
    "Seizure Severity Frequency",
    f"{seizure_freq} Seconds",
    delta=f"{'🚨 Stage 4 Critical Red' if seizure_freq > 5 else ('⚠️ Stage 3 Orange Alert' if seizure_freq > 0 else '🟢 Normal Green')}"
)

st.divider()

# SECOND-BY-SECOND ACTIVITY BAR
st.subheader("📊 Second-by-Second Real-Time Edge Telemetry")
display_blocks = min(total_seconds, 25)
bar_cols = st.columns(display_blocks)
for idx in range(display_blocks):
    row_data = timeline_df.iloc[idx]
    if row_data['is_seizure'] == 1:
        if seizure_freq <= 5:
            bar_cols[idx].markdown(f"<div style='background-color:#F97316; color:white; border-radius:6px; padding:8px 2px; text-align:center; font-weight:bold; font-size:12px;'>⚠️ {row_data['second']}s<br>STAGE 3</div>", unsafe_allow_html=True)
        else:
            bar_cols[idx].markdown(f"<div style='background-color:#EF4444; color:white; border-radius:6px; padding:8px 2px; text-align:center; font-weight:bold; font-size:12px;'>🚨 {row_data['second']}s<br>STAGE 4</div>", unsafe_allow_html=True)
    else:
        bar_cols[idx].markdown(f"<div style='background-color:#22C55E; color:white; border-radius:6px; padding:8px 2px; text-align:center; font-weight:bold; font-size:12px;'>✓ {row_data['second']}s<br>NORMAL</div>", unsafe_allow_html=True)

st.write("")

# CONTINUOUS WAVEFORM VISUALIZATION
st.subheader("📈 Patient EEG Continuous Voltage Waveform with Alert Zone")
full_waveform = np.concatenate([t['raw_wave'] for t in timeline_results])
time_axis_seconds = np.linspace(0, total_seconds, len(full_waveform))

fig, ax = plt.subplots(figsize=(15, 4.2))
ax.plot(time_axis_seconds, full_waveform, color="#2563EB", linewidth=1.2, label="Electrode Voltage (μV)")

if 1 <= seizure_freq <= 5:
    for idx, t in enumerate(timeline_results):
        if t['is_seizure'] == 1:
            ax.axvspan(idx, idx + 1, color="#FB923C", alpha=0.45, label="Stage 3 Orange Caution Zone (≤ 5)" if idx == (start_time_sec - 1) else "")
elif seizure_freq > 5:
    for idx, t in enumerate(timeline_results):
        if t['is_seizure'] == 1:
            ax.axvspan(idx, idx + 1, color="#EF4444", alpha=0.35, label="Stage 4 Critical Red Seizure Attack (> 5)" if idx == (start_time_sec - 1) else "")

ax.set_xlabel("Recording Timeline (Seconds)", fontsize=11, fontweight='bold')
ax.set_ylabel("Voltage Amplitude (μV)", fontsize=11, fontweight='bold')
ax.set_title("EEG Stream (Shaded Zone Triggers Physical Alarms for Bystanders)", fontsize=12, fontweight='bold')
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_xlim(0, total_seconds)
ax.legend(loc="upper right")
st.pyplot(fig)

st.divider()

# SECOND-BY-SECOND DRILL DOWN SLIDER
st.subheader("🔍 Inspect Specific Second of Individual Telemetry")
default_slider_val = start_time_sec if seizure_freq > 0 else 1
selected_sec = st.slider("Select Second to Inspect Detail:", min_value=1, max_value=max(total_seconds, 1), value=min(default_slider_val, total_seconds))
sec_info = timeline_df.iloc[selected_sec - 1]

dcol1, dcol2 = st.columns([2, 1])

with dcol1:
    fig_sec, ax_sec = plt.subplots(figsize=(9, 3))
    wave_color = "#EA580C" if (seizure_freq <= 5 and sec_info['is_seizure'] == 1) else ("#DC2626" if sec_info['is_seizure'] == 1 else "#16A34A")
    ax_sec.plot(sec_info['raw_wave'], color=wave_color, linewidth=2.0)
    ax_sec.set_title(f"EEG Voltage Signature at Second {selected_sec} ({sec_info['time_label']})", fontsize=11, fontweight='bold')
    ax_sec.set_xlabel("178 Discrete Samples @ 173.6 Hz")
    ax_sec.set_ylabel("Voltage (μV)")
    ax_sec.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig_sec)

with dcol2:
    st.markdown(f"### Snapshot at Second {selected_sec}")
    if sec_info['is_seizure'] == 1:
        if seizure_freq <= 5:
            st.warning(f"⚠️ **STATUS: STAGE 3 ORANGE ALERT (FREQUENCY ≤ 5)**\n\n**Spike Probability:** {sec_info['seizure_prob']*100:.1f}%\n\nBuzzer: Single beep tone active.")
        else:
            st.error(f"🚨 **STATUS: STAGE 4 CRITICAL RED EMERGENCY (FREQUENCY > 5)**\n\n**Seizure Risk:** {sec_info['seizure_prob']*100:.1f}%\n\nBuzzer: Multiple High-Frequency Rapid Siren Sweeps Active (1800–3200 Hz).")
    else:
        st.success(f"✅ **STATUS: NORMAL RHYTHM**\n\n**Confidence:** {sec_info['normal_prob']*100:.1f}%\n\nSystem is silent.")
    st.info(f"**Max Peak Voltage:** {sec_info['voltage_max']:.1f} μV\n\n**Lowest Valley:** {sec_info['voltage_min']:.1f} μV")