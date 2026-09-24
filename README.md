# WIRE WATCHER — RF Spectrum Monitoring, Signal Analysis, Channel Allocation and Availability Decision System

> **Electronics & Communication Engineering (ECE) / RF Spectrum Analysis Project**  
> An engineering instrument for RF spectrum monitoring, digital signal processing (DSP), time-domain waveform analysis, thermal noise floor evaluation, out-of-distribution (OOD) safety gating, channel allocation, temporal occupancy analytics, and operational availability decision-making.

---

## 📌 Executive Summary

**Wire Watcher** is an Electronics & Communication Engineering (ECE) system designed to assess radio frequency (RF) channel occupancy, operational spectrum availability, and channel candidate allocation across Very High Frequency (VHF) bands (70 MHz – 160 MHz).

The machine learning classifier (Random Forest v2, ROC-AUC ≈ 0.50) is **FROZEN** and serves as **ONE supplementary component** inside a multi-stage physical and digital signal processing pipeline. To maintain strict scientific integrity and avoid circular leakage, received signal strength is strictly quarantined from ML feature predictors.

### 🌟 Primary Objectives & Principles
- **Prominent RF Engineering & DSP**: Real physical equations, Johnson-Nyquist thermal noise calculations ($kTB$), free-space path loss ($FSPL$), electromagnetic wavelength ($\lambda = c/f$), antenna element dimensions ($\lambda/4, \lambda/2$), and Nyquist sampling rate checks ($f_s \ge 2B$).
- **Time-Domain & Spectral Analysis**: Hanning-windowed FFT Power Spectral Density (PSD), peak detection, and time-domain baseband complex I/Q waveform processing with derived RMS, Peak, and Crest Factor quality metrics.
- **Multi-Stage Decision Hierarchy**: Operational availability decisions (`AVAILABLE`, `OCCUPIED`, `UNCERTAIN`) combine empirical signal power detection, I/Q envelope DSP statistics, ML probability, and multivariate out-of-distribution (OOD) safety bounds.
- **Engineering Channel Allocation**: Automated channel candidate generation, guard-band analysis, inter-channel interference risk assessment, transparent 100-point scoring formula, and rejection trace.
- **RF Event & Temporal Occupancy Analytics**: Contiguous transmission burst event derivation, interval-merged temporal channel utilization %, and SQLite event persistence (`wire_watcher.db`).
- **Controlled Replay Engine**: Chronological playback across 164,160 authentic dataset observations with step-by-step pipeline execution (`Dataset → RF Observation → DSP → Activity Detector → ML Evidence → OOD Guard → Availability → Allocation`).
- **Explicit Data Provenance**: Every quantity in the system is labeled with its exact data provenance tag (`REAL_MEASURED`, `DATASET`, `DERIVED`, `INFERRED_RF_ACTIVITY`, `ML_EVIDENCE`, `APPLICATION_METADATA`).
- **Scientific Integrity**: Target label `inferred_rf_activity` is an **inferred pseudo-label**. Ground-truth spectrum occupancy is explicitly declared **UNVERIFIED**. Zero fake I/Q arrays or random events are fabricated when data is absent.

---

## 📐 Multi-Stage RF Decision & Allocation Hierarchy

```text
                   +---------------------------------------+
                   |            RF Observation             |
                   | (Freq, BW, Power, I/Q Sample Vector)  |
                   +---------------------------------------+
                                       |
                                       v
                   +---------------------------------------+
                   |  Time-Domain Waveform & DSP Analysis  |
                   | (I(t), Q(t), Magnitude, FFT, PSD)     |
                   +---------------------------------------+
                                       |
                                       v
                   +---------------------------------------+
                   |         Noise & Peak Analysis         |
                   |   (kTB Noise Floor & Peak Detection)  |
                   +---------------------------------------+
                                       |
                                       v
                   +---------------------------------------+
                   | RF Activity Det. + ML Activity Evid.  |
                   |  (Power Threshold + Random Forest v2) |
                   +---------------------------------------+
                                       |
                                       v
                   +---------------------------------------+
                   |            OOD Safety Guard           |
                   |   (1st/99th Percentile & Distance)    |
                   +---------------------------------------+
                                       |
                                       v
                   +---------------------------------------+
                   |         Availability Decision         |
                   |     AVAILABLE / OCCUPIED / UNCERTAIN  |
                   +---------------------------------------+
                                       |
                                       v
                   +---------------------------------------+
                   |       RF Event & Utilization Log      |
                   | (SQLite Event Burst Ingestion & % Util)|
                   +---------------------------------------+
                                       |
                                       v
                   +---------------------------------------+
                   |     Channel Candidate Allocation      |
                   | (Candidate Scoring & Guard-Band Risk) |
                   +---------------------------------------+
```

---

## 🛠️ System Modules

### 1. 📡 RF Engineering Intelligence Dashboard (Home Screen)
- **Top Instrument Banner**: Immediately communicates operational status, live frequency telemetry, signal strength, noise floor reference, and SNR.
- **Telemetry Cards**:
  - `RF Source Status`: Historical Dataset / IQ Replay observation (120.0 MHz, -75.0 dBm).
  - `Spectrum Availability`: `AVAILABLE` status with physical activity state (`NOT DETECTED`).
  - `Recommended Channel`: Optimal candidate allocation with engineering score (e.g. `95.0/100`).
  - `Event Analytics`: Total recorded events count and temporal channel utilization %.

### 2. 🌊 Time-Domain I/Q Waveform Visualizer
- **Complex Baseband Series**: Renders $I(t)$, $Q(t)$, and envelope magnitude $|x(t)|$.
- **Signal Quality Metrics**: Calculates Root Mean Square (RMS) magnitude, Peak magnitude, Variance, and Crest Factor.
- **Zero-Fake-IQ Policy**: Displays an explicit `TIME-DOMAIN WAVEFORM UNAVAILABLE` notice when baseband I/Q samples are not captured by the source.

### 3. 📊 Spectrum Analyzer & Relative PSD Visualizer
- **FFT Scope**: Renders Hanning-windowed FFT magnitude squared labeled accurately as `"Relative PSD (dBm)"` / `"Normalized Spectral Power"`.
- **Peak & Occupancy Detector**: Evaluates noise floor percentile, applies adaptive threshold ($Noise Floor + \Delta dB$), marks spectral peaks, and highlights occupied frequency bands.

### 4. 🎯 Availability & Channel Candidate Allocation Engine
- **Candidate Channel Generator**: Generates channel candidates across specified frequency spans (e.g. 70.0 – 160.0 MHz) with configurable bandwidths and guard bands.
- **Transparent Scoring Formula**:
  $$\text{Score} = 100 - \text{Penalty}_{\text{Activity}} - \text{Penalty}_{\text{Uncertainty}} - \text{Penalty}_{\text{OOD}} - \text{Penalty}_{\text{Noise}} - \text{Penalty}_{\text{GuardBand}} - \text{Penalty}_{\text{Interference}}$$
- **Guard-Band Analysis**: Evaluates protected lower/upper guard band boundaries and classifies status as `SAFE_MARGIN`, `MARGINAL`, or `CONFLICT`.
- **Interference Risk Assessment**: Categorizes inter-channel risk as `LOW`, `MODERATE`, or `HIGH`.
- **Calculation Trace Modal**: Provides full mathematical step-by-step breakdown for any candidate channel.

### 5. 📈 RF Event Analytics & Channel Utilization
- **Temporal Event Derivation**: Automatically derives contiguous active transmission bursts (`inferred_rf_activity == 1`) per frequency channel across 164,160 observations.
- **SQLite Event Persistence**: Stores derived events (`event_id`, `frequency_mhz`, `start_time`, `end_time`, `duration_seconds`, `peak_power_dbm`, `avg_power_dbm`, `provenance`) in `wire_watcher.db`.
- **Merged-Interval Utilization**: Merges overlapping event intervals to compute exact channel utilization % over the observation window.

### 6. 🎞️ Data Explorer & Controlled RF Replay Mode
- **Chronological Playback**: Navigates through 164,160 dataset observations (`0 / 164160`).
- **Interactive Controls**: `PLAY`, `PAUSE`, `RESET`, `STEP STEP >`, and playback speed toggles (`0.5x`, `1x`, `2x`, `5x`).
- **Pipeline Execution**: Reruns the complete RF processing pipeline on each step and updates telemetry.
- **Data Explorer Table**: Paginated browsing of authentic dataset records with frequency, bandwidth, signal strength, IQ availability, and inferred activity flags.

### 7. 🧮 RF & Electronics Engineering Calculator Suite
Interactive mathematical derivation modules with explicit display of `INPUT`, `FORMULA`, `SUBSTITUTION`, `RESULT`, and `UNIT`:
1. **Power Conversions**:
   $$P(\text{W}) = 10^{\frac{P_{\text{dBm}} - 30}{10}}, \quad P(\text{dBm}) = 10 \log_{10}(P(\text{W})) + 30, \quad P(\text{dBW}) = P(\text{dBm}) - 30$$
2. **Wavelength ($\lambda$)**:
   $$\lambda = \frac{c}{f}, \quad c = 299,792,458 \text{ m/s}$$
3. **Resonant Antenna Dimensions**:
   $$L_{\text{monopole}} = \frac{\lambda}{4}, \quad L_{\text{dipole}} = \frac{\lambda}{2}$$
4. **Bandwidth & Center Frequency**:
   $$BW = f_{\text{high}} - f_{\text{low}}, \quad f_0 = \frac{f_{\text{high}} + f_{\text{low}}}{2}, \quad FBW\% = \frac{BW}{f_0} \times 100\%$$
5. **Nyquist Sampling Rate**:
   $$f_s \ge 2B \quad (\text{Real Sampling PASS / FAIL Check})$$
6. **Johnson-Nyquist Thermal Noise Floor**:
   $$P_n = k_B \cdot T \cdot B, \quad N_0 \approx -174.00 \text{ dBm/Hz at } 290 \text{ K}$$
7. **Signal-to-Noise Ratio (SNR)**:
   $$SNR(\text{dB}) = P_{\text{signal}}(\text{dBm}) - P_{\text{noise}}(\text{dBm})$$
8. **Free-Space Path Loss (FSPL)**:
   $$FSPL(\text{dB}) = 20 \log_{10}(d_{\text{km}}) + 20 \log_{10}(f_{\text{MHz}}) + 32.44$$
9. **Link Budget Received Power ($P_r$)**:
   $$P_r(\text{dBm}) = P_t + G_t + G_r - L_{\text{path}} - L_{\text{misc}}$$

### 8. 📄 Automatic 10-Section RF Technical Report Generator
Generates printable/exportable engineering audit reports containing:
1. Observation Summary
2. DSP & Spectral Estimation
3. RF Engineering Parameters
4. Machine Learning Evidence
5. OOD Safety Guard Status
6. Spectrum Availability Assessment
7. Channel Allocation Recommendation
8. Event Analytics & Occupancy
9. Measurement Data Provenance
10. System Limitations & Disclaimers

### 9. 🛡️ ML Performance, Leakage Audit & Validation Dashboard
- **Model Architecture**: Frozen Random Forest Classifier v2 (60 trees, max depth 10) evaluated on a 60/20/20 chronological split (32,832 test samples).
- **Leakage Controls**: Received signal power (`signal_strength_dbm`) is strictly quarantined from ML predictor features.
- **Audit Findings**: Displays Gini impurity importance, permutation importance, feature group experiments (Exp A – Exp E), and future dataset improvement plans.

---

## 🚀 Quickstart & Setup Guide

### Prerequisites
- **Python**: 3.10 or higher
- **Node.js**: v18.0 or higher
- **Git**: Installed

### 1. Clone Repository
```bash
git clone https://github.com/Subham-Biswal3109/RFsignal.git
cd RFsignal
```

### 2. Backend Setup (Flask Server)
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the Flask backend server (port 5000)
python app.py
```
- **Backend API**: `http://localhost:5000/`
- **Health Check**: `http://localhost:5000/api/health`

### 3. Frontend Setup (React + Vite)
```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start Vite development server (port 5173)
npm run dev
```
- **Frontend App**: `http://localhost:5173/`

---

## 🧪 Testing & Verification

### Run Automated Backend Test Suite (pytest)
```bash
python -m pytest -v
```
> **Result**: 122 passed, 1 skipped (100% pass rate).

### Frontend Production Build
```bash
cd frontend
npm run build
```
> **Result**: `tsc && vite build` completed with 0 errors; clean production bundle.

---

## 📜 Scientific & Legal Notice

> **Scientific Notice:** This application provides an inferred operational spectrum availability decision based on real SDR signal measurements, derived Johnson-Nyquist thermal calculations, and leakage-safe ML feature analysis. It does NOT certify legal spectrum vacancy or live SDR hardware monitoring. Target labels are inferred pseudo-labels, not ground-truth occupancy.

