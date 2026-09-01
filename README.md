# WIRE WATCHER — RF Spectrum Monitoring, Signal Analysis and Availability Decision System

> **Electronics & Communication Engineering (ECE) / RF Spectrum Analysis Project**  
> An engineering instrument for RF spectrum monitoring, digital signal processing (DSP), thermal noise floor evaluation, out-of-distribution (OOD) safety gating, and operational channel availability decision-making.

---

## 📌 Executive Summary

**Wire Watcher** is an Electronics & Communication Engineering (ECE) system designed to assess radio frequency (RF) channel occupancy and operational spectrum availability across Very High Frequency (VHF) bands (70 MHz – 160 MHz).

The machine learning classifier (Random Forest, ROC-AUC ≈ 0.50) is **FROZEN** and serves as **ONE supplementary component** inside a multi-stage physical and digital signal processing pipeline. To maintain strict scientific integrity and avoid circular leakage, received signal strength is strictly quarantined from ML feature predictors.

### 🌟 Primary Objectives & Principles
- **Prominent RF Engineering & DSP**: Real physical equations, Johnson-Nyquist thermal noise calculations ($kTB$), free-space path loss ($FSPL$), electromagnetic wavelength ($\lambda = c/f$), antenna element dimensions ($\lambda/4, \lambda/2$), and Nyquist sampling rate checks ($f_s \ge 2B$).
- **Multi-Stage Decision Hierarchy**: Operational availability decisions (`AVAILABLE`, `OCCUPIED`, `UNCERTAIN`) combine empirical signal power detection, I/Q envelope DSP statistics, ML probability, and multivariate out-of-distribution (OOD) safety bounds.
- **Explicit Data Provenance**: Every quantity in the system is labeled with its exact data provenance tag (`REAL_MEASURED`, `DERIVED`, `INFERRED/PSEUDO-LABEL`, `ML_EVIDENCE`, `APPLICATION_METADATA`).
- **Scientific Integrity**: Target label `inferred_rf_activity` is an **inferred pseudo-label**. Ground-truth spectrum occupancy is explicitly declared **UNVERIFIED**.

---

## 📐 Multi-Stage RF Decision Hierarchy

```
                   +---------------------------------------+
                   |            RF Observation             |
                   | (Freq, BW, Power, I/Q Sample Vector)  |
                   +---------------------------------------+
                                       |
                                       v
                   +---------------------------------------+
                   |        RF Feature Extraction          |
                   |   (13 I/Q Envelope & FFT Features)    |
                   +---------------------------------------+
                                       |
                                       v
                   +---------------------------------------+
                   |               FFT / PSD               |
                   |      (Normalized Spectral Power)      |
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
                   |  (Power Threshold + Random Forest)    |
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
```

---

## 🛠️ System Modules

### 1. 📡 Spectrum Availability Analyzer (Home Screen)
- **Top Hero Instrument Banner**: Immediately communicates **WIRE WATCHER — RF SPECTRUM MONITORING & AVAILABILITY ANALYSIS**.
- **Live Spectrum Status Card**: Displays live carrier frequency, bandwidth, signal power, RF activity state (`NOT DETECTED` / `DETECTED`), and operational availability (`AVAILABLE` / `OCCUPIED` / `UNCERTAIN`).
- **Structured 5-Stage Diagnostic Card**:
  - `RF OBSERVATION` (`REAL_MEASURED`): Frequency, Bandwidth, Measured Signal Strength, I/Q Availability.
  - `DERIVED PHYSICS` (`DERIVED`): Wavelength $\lambda$, Power in Watts/nW, Johnson-Nyquist Thermal Noise Reference $kTB$ at 290 K.
  - `RF ANALYSIS` (`INFERRED/PSEUDO-LABEL`): Activity state, SNR margin over $kTB$, OOD guard flag.
  - `ML EVIDENCE` (`ML_EVIDENCE`): Model probability, confidence rating, signal power quarantine status.
  - `SYSTEM DECISION` (`APPLICATION_METADATA`): Inferred operational decision.
- **SQLite History**: Automatically persists observation records to SQLite database (`wire_watcher.db`).

### 2. 🧮 RF & Electronics Engineering Calculator Suite
Includes interactive mathematical derivation modules with explicit display of `INPUT`, `FORMULA`, `SUBSTITUTION`, `RESULT`, and `UNIT`:
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

*Inputs are strictly validated: rejects negative absolute temperature ($T \le 0$ K), zero/negative frequency ($f \le 0$ MHz), zero FSPL distance ($d \le 0$ km), invalid bandwidth ($BW \le 0$), invalid sampling rate ($f_s \le 0$), NaN, and Infinity.*

### 3. 📊 Spectrum Visualizer & DSP Peak Detector
- **FFT Scope**: Renders Hanning-windowed FFT magnitude squared labeled accurately as `"Relative PSD (dB)"` / `"Normalized Spectral Power"`.
- **Peak & Occupancy Detector**: Evaluates noise floor percentile, applies adaptive threshold ($Noise Floor + \Delta dB$), marks local maxima spectral peaks, and highlights occupied frequency channels.
- **Simulator Transfer**: Includes a **"Transfer to Activity Detector"** action for passing simulated channel parameters into the detector pipeline.

### 4. 🗃️ RF Dataset Explorer
Presents authentic statistics from the SDR acquisition dataset (`logged_data.csv`):
- **Observations**: 164,160 rows acquired at a 20-second cadence.
- **Carrier Frequencies**: 70, 90, 100, 120, 140, 160 MHz (VHF Band).
- **I/Q Sample Availability**: 109,324 rows (66.6%) with 100 complex I/Q samples; 54,836 rows (33.4%) missing I/Q.
- **Dynamic Signal Range**: -119.0 dBm to -25.0 dBm (94 dB span).
- **Visual Distributions**: Bar charts for frequency distribution, signal power bins, I/Q availability, and channel bandwidth (50 kHz).

### 5. 🤖 ML Architecture & Transparency
- **Model Architecture**: Frozen Random Forest Classifier (60 trees, max depth 10) evaluated on a 60/20/20 chronological split (32,832 test samples).
- **Performance**: ROC-AUC ≈ 0.50, PR-AUC ≈ 0.5058.
- **Target Leakage Control**: Received signal power (`signal_strength_dbm`) is strictly quarantined from ML predictor features.
- **Role Statement**: *"The ML classifier provides supplementary RF activity evidence. The final operational decision also incorporates physical RF activity detection and OOD protection."*

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

### Run Python Test Suite (pytest)
```bash
python -m pytest tests/test_rf_calculations.py tests/test_api.py tests/test_db.py tests/test_inference.py tests/test_leakage.py tests/ml/test_rf_signal_migration.py
```
> **Result**: 77 passed (100% pass rate).

### Frontend Type-Checking & Build
```bash
cd frontend
npm run type-check
npm run build
```
> **Result**: 0 TypeScript errors; clean production build.

### Live End-to-End Test
```bash
python tests/e2e_live_test.py
```
> **Result**: ALL PASSED across health, model-info, normal/strong/weak/OOD/IQ predict requests, and prediction history retrieval.

---

## 🌿 Repository Branch Structure

The repository [https://github.com/Subham-Biswal3109/RFsignal.git](https://github.com/Subham-Biswal3109/RFsignal.git) is structured into dedicated modular branches:

| Branch Name | Purpose |
| :--- | :--- |
| `main` | Primary production branch containing the complete ECE/RF Wire Watcher system |
| `dev` | Core development integration branch |
| `feature/rf-calculator` | 9-formula RF & Electronics Engineering Calculator suite |
| `feature/spectrum-visualizer` | Relative PSD FFT scope, peak detection, and DSP visualizer |
| `feature/dataset-explorer` | VHF dataset explorer & statistical distribution charts |
| `feature/ml-decision-engine` | Frozen ML classifier, OOD safety bounds, and decision flow |
| `release/v2.0.0` | Release tag branch for Wire Watcher v2.0.0 |

---

## 📜 Scientific & Legal Notice

> **Scientific Notice:** This application provides an inferred operational spectrum availability decision based on real SDR signal measurements, derived Johnson-Nyquist thermal calculations, and leakage-safe ML feature analysis. It does NOT certify legal spectrum vacancy or live SDR hardware monitoring. Target labels are inferred pseudo-labels, not ground-truth occupancy.
