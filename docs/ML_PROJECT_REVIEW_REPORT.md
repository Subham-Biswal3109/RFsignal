# WIRE WATCHER — COMPREHENSIVE MACHINE LEARNING TECHNICAL REVIEW REPORT

> **Electronics & Communication Engineering (ECE) Technical Project Audit & Viva Preparation Guide**  
> **Document Status:** Active Technical Reference & Viva Revision Guide  
> **Target System:** Wire Watcher RF Spectrum Monitoring, Signal Analysis, Channel Allocation and Availability Decision System  
> **Repository:** `RFsignal`  
> **Source of Truth:** Current Repository Source Code, Artifacts, and Evaluation Audits  
> **Model Version:** `rf-signal-activity-v2` (`RandomForestClassifier`)  

---

## TABLE OF CONTENTS
1. [Section A: Executive Summary](#section-a-executive-summary)
2. [Section B: ML's Role in the Complete System Architecture](#section-b-mls-role-in-the-complete-system-architecture)
3. [Section C: Dataset Inspection & Data Dictionary](#section-c-dataset-inspection--data-dictionary)
4. [Section D: Target & Label Definition](#section-d-target--label-definition)
5. [Section E: Data Preprocessing Pipeline](#section-e-data-preprocessing-pipeline)
6. [Section F: Feature Engineering & DSP Statistics](#section-f-feature-engineering--dsp-statistics)
7. [Section G: RF / DSP → ML Ingestion Pipeline](#section-g-rf--dsp--ml-ingestion-pipeline)
8. [Section H: Machine Learning Algorithm & Mechanics](#section-h-machine-learning-algorithm--mechanics)
9. [Section I: Model Training Procedure](#section-i-model-training-procedure)
10. [Section J: Chronological Train / Validation / Test Split](#section-j-chronological-train--validation--test-split)
11. [Section K: Class Imbalance & Mitigation](#section-k-class-imbalance--mitigation)
12. [Section L: Hyperparameter Tuning & Architecture](#section-l-hyperparameter-tuning--architecture)
13. [Section M: Model Evaluation Metrics & Benchmarks](#section-m-model-evaluation-metrics--benchmarks)
14. [Section N: Confusion Matrix Analysis](#section-n-confusion-matrix-analysis)
15. [Section O: Threshold Optimization](#section-o-threshold-optimization)
16. [Section P: Out-of-Distribution (OOD) Safety Guard](#section-p-out-of-distribution-ood-safety-guard)
17. [Section Q: Model Confidence & Probability Interpretation](#section-q-model-confidence--probability-interpretation)
18. [Section R: Model Serialization & Bundle Contract](#section-r-model-serialization--bundle-contract)
19. [Section S: Runtime Inference Trace](#section-s-runtime-inference-trace)
20. [Section T: Database Persistence & Historical Logging](#section-t-database-persistence--historical-logging)
21. [Section U: Frontend Integration & Visualization](#section-u-frontend-integration--visualization)
22. [Section V: Automated Test Suite & Verification](#section-v-automated-test-suite--verification)
23. [Section W: Current ML Engineering Strengths](#section-w-current-ml-engineering-strengths)
24. [Section X: Current ML Scientific & Technical Limitations](#section-x-current-ml-scientific--technical-limitations)
25. [Section Y: What the ML Model Does NOT Do](#section-y-what-the-ml-model-does-not-do)
26. [Section Z: How to Explain the Current ML Limitation in a Viva](#section-z-how-to-explain-the-current-ml-limitation-in-a-viva)
27. [Section AA: Comprehensive Viva Questions and Answers (50 Q&As)](#section-aa-comprehensive-viva-questions-and-answers-50-qas)
28. [Section AB: "Explain Like I Am an ECE Student"](#section-ab-explain-like-i-am-an-ece-student)
29. [Section AC: Engineering & Mathematical Formula Sheet](#section-ac-engineering--mathematical-formula-sheet)
30. [Section AD: "Memorize This Before Review" Cheat Sheet](#section-ad-memorize-this-before-review-cheat-sheet)

---

## SECTION A: EXECUTIVE SUMMARY

### The 30-Second Summary (Reviewer Elevator Pitch)
"In Wire Watcher, Machine Learning does **not** make autonomous spectrum allocation decisions. Instead, an ensemble Random Forest classifier (`rf-signal-activity-v2`) acts as a **supplementary statistical evidence generator** to estimate whether detectable RF activity is present based on 13 normalized spectral and I/Q statistical moments. To uphold scientific integrity and prevent circular target leakage, received signal strength (`signal_strength_dbm`) is quarantined from ML predictors. Primary availability decisions are driven by physical deterministic DSP calculations (thermal noise floor $kTB$, SNR $\ge 6\text{ dB}$), while standardized multivariate Out-of-Distribution (OOD) safety checks override decisions to `UNCERTAIN` whenever incoming observations deviate from training bounds."

### The 1-Minute Explanation
"Wire Watcher is an Electronics & Communication Engineering (ECE) platform evaluating radio frequency activity across Very High Frequency (VHF) bands between 70 MHz and 160 MHz. 

In this system, ML operates inside a multi-tier hierarchy:
1. **Physical RF/DSP Stage:** Computes Johnson-Nyquist thermal noise floor ($P_n = kTB + \text{NF}$), calculates SNR, and evaluates temporal persistence.
2. **Machine Learning Stage:** Takes 13 statistical parameters (carrier frequency, bandwidth, and dimensionless complex I/Q baseband statistics such as crest factor, variance, phase concentration, and spectral entropy) and outputs an RF activity probability.
3. **Safety & Evidence Fusion Stage:** Evaluates 1st/99th percentile bounds and standardized multivariate Euclidean distance. If an input is anomalous, it flags OOD and forces the state to `UNCERTAIN`.
4. **Channel Allocation Engine:** Scores candidate channels out of 100 using deterministic penalty deductions for activity, noise, guard-band conflict, and interference risk."

### The 2-Minute Explanation
"A major engineering focus of the Wire Watcher project is **scientific honesty and target leakage prevention**.

The underlying dataset contains 164,160 recorded observations across 6 VHF frequencies. Because authentic physical spectrum occupancy ground-truth was not recorded at acquisition time, the project generates an **inferred pseudo-label** (`inferred_rf_activity`) defined as whether the received signal power exceeds the training-only per-frequency median (-50.0 dBm).

If `signal_strength_dbm` were included as an input feature to the ML model, the classifier would achieve an artificial 100% accuracy simply by memorizing this threshold rule—a textbook example of **target leakage**. Wire Watcher strictly quarantines `signal_strength_dbm` from the ML feature set. 

Under this setup, the current leakage-safe feature set provides **approximately chance-level discrimination (ROC-AUC ≈ 0.50)** of the current inferred pseudo-label on the held-out chronological test set. Rather than masking this result or fabricating synthetic correlations, our system documents the empirical finding honestly: physical RF signal detection is handled primarily in the deterministic DSP domain (via energy detection and thermal noise reference), while the ML model is evaluated transparently and bounded by an OOD safety envelope."

### The 5-Minute Technical Deep Dive
"From an ECE and signal processing systems perspective, Wire Watcher separates deterministic electromagnetic physics from statistical pattern recognition.

1. **RF Front-End & Ingestion:**
   Incoming observations consist of center frequency ($f_c$), channel bandwidth ($B$), reported power ($P_{\text{sig}}$), and an optional 100-sample complex baseband vector $x[n] = I[n] + jQ[n]$.
2. **Time-Domain & Spectral Feature Extraction:**
   When baseband I/Q data is present, the DSP pipeline calculates:
   - Root Mean Square magnitude: $\text{RMS} = \sqrt{\frac{1}{N}\sum |x[n]|^2}$
   - Crest Factor: $\text{CF} = \frac{\max |x[n]|}{\text{RMS}}$
   - Phase Concentration (Circular Mean Resultant Length): $|\frac{1}{N}\sum e^{j\theta[n]}|$
   - Normalized Spectral Entropy: $-\sum p_k \log_2(p_k) / \log_2(N)$ from windowed FFT power spectral bins.
3. **ML Preprocessing & Inference:**
   A scikit-learn `Pipeline` applies a `ColumnTransformer` with `SimpleImputer(strategy='median', add_indicator=True)` and `StandardScaler()`. Missing I/Q observations (33.4% of dataset) are handled cleanly with missing indicator flags without data fabrication. (Note: `StandardScaler` is included to provide a uniform dimensional scale for OOD distance calculations and linear baselines, not because Random Forest requires scaling).
4. **Deterministic DSP Activity Detector:**
   In parallel, `DSPActivityDetector` computes theoretical thermal noise $P_n = -174\text{ dBm/Hz} + 10\log_{10}(B) + \text{NF}$. It checks whether $\text{SNR} = P_{\text{sig}} - P_n \ge 6.0\text{ dB}$ and verifies temporal persistence over contiguous frames.
5. **Out-of-Distribution (OOD) Safety Gating:**
   The inference service computes a standardized multivariate Euclidean distance across transformed features:
   $$D_{\text{OOD}} = \sqrt{\sum_{i=1}^{M} \left(\frac{x_i - \mu_i}{\sigma_i}\right)^2}$$
   If $D_{\text{OOD}} > 6.448$ (the 99.75th percentile of training data) or any feature violates 1st/99th percentile bounds, OOD is triggered.
6. **Channel Allocation:**
   Candidate channels are scored via an objective formula:
   $$\text{Score} = 100 - \text{Pen}_{\text{Activity}} - \text{Pen}_{\text{Uncertainty}} - \text{Pen}_{\text{OOD}} - \text{Pen}_{\text{Noise}} - \text{Pen}_{\text{GuardBand}} - \text{Pen}_{\text{Interference}}$$
   This ensures that no candidate channel is recommended if physical DSP detects occupancy or if the safety layer indicates high uncertainty."

---

## SECTION B: ML'S ROLE IN THE COMPLETE SYSTEM ARCHITECTURE

### Complete End-to-End Pipeline Diagram

```text
 ┌──────────────────────────────────────────────────────────────────┐
 │                      1. RF OBSERVATION                           │
 │      Carrier Frequency (MHz), Bandwidth (kHz), Signal Power (dBm)│
 │                Complex Baseband Vector: I[n] + jQ[n]             │
 └─────────────────────────────────┬────────────────────────────────┘
                                   │
                                   ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                  2. DETERMINISTIC RF / DSP STAGE                 │
 │  • Johnson-Nyquist Thermal Noise Floor: P_n = kTB + NF (dBm)     │
 │  • Physical SNR Calculation: SNR = P_signal - P_noise (dB)       │
 │  • FFT Magnitude & Normalized Relative PSD Calculation           │
 │  • Temporal Frame Persistence Tracker (Contiguous Active Bursts) │
 └─────────────────┬───────────────────────────────┬────────────────┘
                   │                               │
       Primary Physical Evidence                   │ Baseband I/Q Vector
                   │                               │
                   ▼                               ▼
 ┌──────────────────────────────────┐   ┌───────────────────────────┐
 │   PHYSICAL DSP ACTIVITY DETECTOR │   │ 3. DSP FEATURE EXTRACTION │
 │      SNR >= 6 dB & P >= -100 dBm │   │   RMS, Variance, Peak, CF,│
 │ State: ACTIVE / INACTIVE / UNCER │   │   Phase Conc, Entropy     │
 └─────────────────┬────────────────┘   └──────────────┬────────────┘
                   │                                   │ 13 ML Predictors
                   │                                   │ (Power Quarantined)
                   │                                   ▼
                   │                    ┌───────────────────────────┐
                   │                    │  4. ML PIPELINE INFERENCE │
                   │                    │   Median Imputer + Ind.   │
                   │                    │   StandardScaler          │
                   │                    │   Random Forest v2 (60 T) │
                   │                    │   Output: ML Probability  │
                   │                    └──────────────┬────────────┘
                   │                                   │
                   │         Supporting Evidence       │
                   │         Probability p in [0, 1]   │
                   │                                   ▼
                   │                    ┌───────────────────────────┐
                   │                    │ 5. OOD SAFETY GUARD       │
                   │                    │   1st / 99th Percentiles  │
                   │                    │   Std. Multivariate Dist. │
                   │                    └──────────────┬────────────┘
                   │                                   │
                   │                                   │ Safety Override
                   ▼                                   ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                   6. MULTI-STAGE EVIDENCE FUSION                 │
 │  • IF OOD Flag == TRUE  ──> Decision = UNCERTAIN (Safety Lockout)│
 │  • IF DSP == ACTIVE     ──> Decision = OCCUPIED                  │
 │  • IF DSP == INACTIVE   ──> Decision = AVAILABLE                 │
 │  • Supporting: ML probability logged for audit and review        │
 └─────────────────────────────────┬────────────────────────────────┘
                                   │
                                   ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │              7. CHANNEL ALLOCATION & SCORING ENGINE              │
 │  • Guard-Band Margin Evaluation: SAFE_MARGIN / MARGINAL / CONFLICT│
 │  • Inter-Channel Interference Risk: LOW / MEDIUM / HIGH          │
 │  • Transparent Score = 100 - Penalties                           │
 │  • Final Status: RECOMMENDED (Score >= 50) vs REVIEW_REQUIRED    │
 └─────────────────┬───────────────────────────────┬────────────────┘
                   │                               │
                   ▼                               ▼
 ┌──────────────────────────────────┐   ┌───────────────────────────┐
 │       8. DATABASE PERSISTENCE    │   │ 9. REACT FRONTEND DASHBD  │
 │ SQLite: wire_watcher.db          │   │ Live Telemetry Cards      │
 │ Tables: availability_candidates  │   │ Spectral PSD Scope        │
 │         rf_events (40,996 rows)  │   │ ML Audit & Feature Graphs │
 └──────────────────────────────────┘   └───────────────────────────┘
```

### Critical Conceptual Distinctions

To maintain absolute technical clarity during an engineering review, six distinct concepts must be clearly separated:

| Concept | Precise Definition | Role in Wire Watcher | Ground Truth Status |
|---|---|---|---|
| **1. RF Signal Presence / Activity** | Received RF energy or activity observed by the receiver within the monitored channel. | Measured via receiver front-end signal power ($P_{\text{sig}}$) and baseband I/Q samples. | Physical phenomenon; observed with receiver noise. |
| **2. `inferred_rf_activity` Pseudo-Label** | A binary training label defined as: $P_{\text{sig}} \ge \text{Median}_{\text{train}}(P_{\text{sig}} \mid f_c)$. | The target variable predicted by the ML model. | **NOT ground truth.** It is an engineering heuristic proxy. |
| **3. Physical DSP Detection** | Deterministic threshold check: $\text{SNR} \ge 6\text{ dB}$, $P_{\text{sig}} \ge -100\text{ dBm}$, and frame persistence. | Primary physical evidence driving operational availability. | Physics-based; calculated deterministically from $kTB$. |
| **4. Spectrum Occupancy** | Formal regulatory / legal occupancy indicating whether a licensed primary user is occupying the channel. | Theoretical gold standard for cognitive radio. | **UNVERIFIED / ABSENT** in the current dataset. |
| **5. Spectrum Availability** | Operational state (`AVAILABLE`, `OCCUPIED`, `UNCERTAIN`) synthesized from DSP, ML, and OOD checks. | Output of multi-stage evidence fusion. | Operational synthesis; not a legal certification. |
| **6. Final Channel Allocation** | Engineering recommendation score (0–100) based on guard-band safety, adjacent channel leakage, and interference risk. | Output of `ChannelAllocationEngine` recommending frequency candidate. | Advisory recommendation for secondary channel selection. |

### What ML Controls vs. What ML Does NOT Control

| Component / Decision | Controlled by ML? | Controlling Mechanism / Justification |
|---|---|---|
| **Activity Probability Estimate** | **YES** | Random Forest ensemble voting across 60 decision trees. |
| **Statistical Split Importance** | **YES** | Gini Impurity reduction across node splits. |
| **Out-of-Distribution Gating** | **NO** | Deterministic statistical envelope (1st/99th percentiles & standardized Euclidean distance threshold). |
| **Physical Detection ($P \ge -100\text{ dBm}, \text{SNR} \ge 6\text{ dB}$)** | **NO** | Deterministic ECE equations in `DSPActivityDetector`. |
| **Thermal Noise Floor Reference** | **NO** | Physics constant: $P_n = k_B T B \cdot \text{NF}$. |
| **Safety Override to UNCERTAIN** | **NO** | Deterministic rule override in `ml/inference/predict.py`. |
| **Final Channel Recommendation** | **NO** | Multi-criteria 100-point penalty algorithm in `ChannelAllocationEngine`. |
| **Legal / Regulatory Spectrum Authorization** | **NO** | Explicitly out of scope; system provides advisory engineering estimates only. |

---

## SECTION C: DATASET INSPECTION & DATA DICTIONARY

### Dataset Provenance & Overview
- **Raw File:** `ml/data/rf_signal_source/logged_data.csv`
- **Processed File:** `ml/data/processed/rf_signal_activity.csv`
- **File Size:** ~28.08 MB (processed CSV)
- **Total Observations:** **164,160 rows**
- **Temporal Cadence:** Exactly 20.0 seconds between consecutive timestamps.
- **Carrier Frequencies Analyzed:** 6 discrete VHF frequencies:
  $$\{70.0\text{ MHz}, 90.0\text{ MHz}, 100.0\text{ MHz}, 120.0\text{ MHz}, 140.0\text{ MHz}, 160.0\text{ MHz}\}$$
- **Bandwidths Analyzed:**
  $$\{10.0\text{ kHz}, 20.0\text{ kHz}, 50.0\text{ kHz}, 100.0\text{ kHz}, 200.0\text{ kHz}, 1000.0\text{ kHz}\}$$
- **I/Q Sample Availability:**
  - **Present:** 109,324 rows (66.6%) contain 100 complex samples.
  - **Missing:** 54,836 rows (33.4%) have missing I/Q data.
  - **Zero Fabrication Policy:** When I/Q data is missing, it is set to `NaN` and imputed via median with missing indicators. Zero synthetic I/Q waveforms are fabricated.
- **Duplicate Rows:** **0** (verified via `df.duplicated().sum() == 0`).
- **Data Source Nature:** **Hybrid Real-Measured + Derived + Inferred Pseudo-Label**. Real SDR field logging provides raw signal strength and I/Q samples; mathematical features are derived; target labels are inferred.

### Complete Feature Dictionary

| Feature Name | Storage Type | Physical Unit | Physical / Mathematical Meaning | Used by ML? | Reason for Inclusion / Exclusion |
|---|---|---|---|---|---|
| `Timestamp` | datetime64 | ISO-8601 UTC | Time of SDR capture; used for chronological ordering | **NO** | Excluded to prevent temporal shortcut memorization. |
| `frequency_mhz` | float64 | MHz | Carrier center frequency of the observation channel | **YES** | Identifies specific VHF channel context. |
| `bandwidth_khz` | float64 | kHz | Receiver channel resolution bandwidth | **YES** | Determines noise integration bandwidth. |
| `signal_strength_dbm` | float64 | dBm | Reported RF received signal power level | **NO** | **STRICTLY EXCLUDED TO PREVENT TARGET LEAKAGE** (used to create pseudo-label). |
| `iq_available` | int64 | binary flag | 1 if baseband I/Q was recorded; 0 if missing | **YES** | Informs tree whether I/Q features are authentic or imputed. |
| `iq_rms_magnitude` | float64 | dimensionless | Root Mean Square envelope: $\sqrt{\frac{1}{N}\sum \|x[n]\|^2}$ | **YES** | Measures effective baseband envelope power. |
| `iq_magnitude_variance`| float64 | dimensionless | Variance of envelope magnitude $\|x[n]\|$ | **YES** | Measures amplitude modulation / signal dispersion. |
| `iq_peak_magnitude` | float64 | dimensionless | Maximum absolute peak value $\max \|x[n]\|$ | **YES** | Indicates presence of burst or pulse energy. |
| `iq_crest_factor` | float64 | ratio | Ratio of peak magnitude to RMS magnitude | **YES** | Differentiates constant-envelope (FM) from peaked signals. |
| `iq_p10` | float64 | dimensionless | 10th percentile of envelope magnitude | **YES** | Reflects baseband noise floor baseline. |
| `iq_p50` | float64 | dimensionless | 50th percentile (median) of envelope magnitude | **YES** | Robust central tendency of signal amplitude. |
| `iq_p90` | float64 | dimensionless | 90th percentile of envelope magnitude | **YES** | Reflects high-power excursion levels. |
| `iq_phase_concentration`| float64 | $[0, 1]$ ratio | Circular mean resultant length: $\|\frac{1}{N}\sum e^{j\theta_n}\|$ | **YES** | Distinguishes coherent carrier from random noise. |
| `iq_spectral_entropy` | float64 | $[0, 1]$ ratio | Normalized Shannon entropy of windowed FFT PSD | **YES** | Measures spectral flatness vs single-tone peakiness. |
| `iq_spectral_peak_ratio`| float64 | ratio | Fraction of total spectral power in the maximum bin | **YES** | Identifies concentrated narrowband carriers. |
| `Modulation` | string | category | Annotated modulation type (e.g. FM, AM, BPSK) | **NO** | Application metadata; omitted to prevent non-physical shortcuts. |
| `Interference Type` | string | category | Contextual annotation (e.g. Co-channel, None) | **NO** | Omitted to prevent label contamination. |
| `Device Type / Status` | string | category | SDR hardware annotation | **NO** | Hardware metadata; uninformative for spectrum physics. |
| `Location / Latitude` | float64 | degrees | Receiver GPS coordinates | **NO** | Excluded to ensure geographic generalization. |
| `Battery / CPU / Temp` | mixed | mixed | SDR host telemetry | **NO** | Irrelevant host telemetry. |

---

## SECTION D: TARGET & LABEL DEFINITION

### Target Column: `inferred_rf_activity`
The machine learning task is formulated as **binary classification**:
- **Class 0:** `NO_DETECTABLE_RF_ACTIVITY`
- **Class 1:** `DETECTABLE_RF_ACTIVITY`

### Mathematical Definition of the Target
The pseudo-label is generated exclusively from the training partition using a **per-frequency median split** of received signal strength:
$$y_i = \begin{cases} 
1, & \text{if } P_{\text{sig}, i} \ge \text{Median}_{\text{train}}(P_{\text{sig}} \mid f = f_i) \\ 
0, & \text{otherwise} 
\end{cases}$$

Across the 6 VHF carrier frequencies in the training set, the median signal strength was empirically calculated as:
$$\text{Threshold}(70.0\text{ MHz}) = -50.0\text{ dBm}$$
$$\text{Threshold}(90.0\text{ MHz}) = -50.0\text{ dBm}$$
$$\text{Threshold}(100.0\text{ MHz}) = -50.0\text{ dBm}$$
$$\text{Threshold}(120.0\text{ MHz}) = -50.0\text{ dBm}$$
$$\text{Threshold}(140.0\text{ MHz}) = -50.0\text{ dBm}$$
$$\text{Threshold}(160.0\text{ MHz}) = -50.0\text{ dBm}$$

### Ground-Truth Honesty & ECE Interpretation
1. **Target is an Inferred Pseudo-Label, NOT Ground Truth:**
   The source dataset does not contain an independent, calibrated RF spectrum analyzer recording showing certified channel occupancy. The threshold is an engineering heuristic.
2. **Why Classification Rather than Regression?**
   The operational requirement of cognitive radio / dynamic spectrum access (DSA) is a categorical decision: can secondary transmission proceed on channel $f_c$ without causing harmful interference? Continuous signal power regression would not provide an operational activity classification boundary.
3. **Prevention of Circular Target Reproduction:**
   Because `inferred_rf_activity` is a mathematical transformation of `signal_strength_dbm`, including signal strength in the model would cause the model to act as a trivial threshold detector ($x \ge -50.0 \implies 1$), obscuring genuine feature utility. Quarantining it ensures that ML attempts to infer activity solely from spectral and envelope dynamics.

---

## SECTION E: DATA PREPROCESSING PIPELINE

The preprocessing architecture is packaged within a scikit-learn `Pipeline` and `ColumnTransformer` to enforce strict isolation between training and inference data.

```text
Input Features (13)
        │
        ▼
ColumnTransformer ("num")
        │
        ├── Step 1: SimpleImputer(strategy="median", add_indicator=True)
        │           • Missing I/Q features imputed with training medians
        │           • Generates 10 binary MissingIndicator columns
        │           • Total features expand from 13 to 23
        │
        └── Step 2: StandardScaler()
                    • Centers features to zero mean: z = (x - μ) / σ
                    • Scales to unit variance
                    • Uses training set μ and σ only
        │
        ▼
Output to Classifier: 23 standardized numeric features
```

### Preprocessing Steps Detailed

1. **Missing-Value Imputation (`SimpleImputer`):**
   - **Configuration:** `strategy="median"`, `add_indicator=True`.
   - **Rationale:** 33.4% of records lack baseband I/Q data. Median imputation prevents extreme burst outliers from skewing typical values.
   - **Missing Indicator Flags:** Appends 10 boolean indicator columns (`missingindicator_iq_*`). This permits tree splits to branch differently when I/Q data is authentically absent versus when I/Q values are naturally at the median.
2. **Feature Scaling (`StandardScaler`):**
   - **Configuration:** `with_mean=True`, `with_std=True`.
   - **Important Clarification for ECE Reviews:** Tree-based algorithms such as Random Forest do **not** mathematically require feature normalization because split criteria depend only on monotonic rank ordering. In Wire Watcher, `StandardScaler` is included for two reasons:
     a) To provide uniformly normalized dimensions ($z_i = \frac{x_i - \mu_i}{\sigma_i}$) for the project's standardized multivariate Euclidean OOD distance calculation.  
     b) To ensure an identical preprocessing pipeline when benchmarking against regularized linear baselines (Logistic Regression).
3. **Training vs. Inference Consistency:**
   Parameters ($\mu, \sigma$, medians) are fitted **only** on the 60% training partition. During inference, `preprocessor.transform()` applies the frozen training parameters. No data statistics are computed on inference inputs.

---

## SECTION F: FEATURE ENGINEERING & DSP STATISTICS

Every feature utilized by the machine learning pipeline represents a well-defined physical or statistical quantity derived from the complex baseband representation:
$$x[n] = I[n] + j Q[n], \quad n = 0, 1, \dots, N-1 \quad (N = 100)$$

### Mathematical Derivation of All 13 ML Predictor Features

#### 1. Carrier Frequency (`frequency_mhz`)
$$f_{\text{MHz}} = \frac{f_{\text{Hz}}}{10^6}$$
- **Physical Meaning:** Nominal channel carrier frequency across the VHF spectrum (70–160 MHz).

#### 2. Bandwidth (`bandwidth_khz`)
$$B_{\text{kHz}} = \frac{B_{\text{Hz}}}{10^3}$$
- **Physical Meaning:** Channel operational bandwidth, determining the receiver filter span and theoretical thermal noise power floor ($kTB$).

#### 3. I/Q Availability Flag (`iq_available`)
$$\text{iq\_available} = \begin{cases} 1, & x[n] \text{ is recorded} \\ 0, & x[n] \text{ is missing} \end{cases}$$

#### 4. RMS Magnitude (`iq_rms_magnitude`)
$$\text{RMS} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} |x[n]|^2} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} (I[n]^2 + Q[n]^2)}$$
- **Physical Meaning:** Quadratic mean representing effective root-mean-square amplitude of the baseband envelope.

#### 5. Magnitude Variance (`iq_magnitude_variance`)
$$\sigma_{\text{mag}}^2 = \frac{1}{N} \sum_{n=0}^{N-1} (|x[n]| - \mu_{\text{mag}})^2, \quad \mu_{\text{mag}} = \frac{1}{N}\sum |x[n]|$$
- **Physical Meaning:** Power fluctuations of the envelope. Low for constant envelope modulations (FM, FSK); high for AM or fading multipath signals.

#### 6. Peak Magnitude (`iq_peak_magnitude`)
$$\text{Peak} = \max_{0 \le n < N} |x[n]|$$
- **Physical Meaning:** Maximum instantaneous excursion of the envelope vector.

#### 7. Crest Factor (`iq_crest_factor`)
$$\text{CF} = \frac{\text{Peak}}{\text{RMS}} = \frac{\max |x[n]|}{\sqrt{\frac{1}{N}\sum |x[n]|^2}}$$
- **Physical Meaning:** Ratio of peak to effective power. An unmodulated continuous wave tone has $\text{CF} \approx \sqrt{2} \approx 1.414$; high values indicate impulsive noise, pulsed radar, or OFDM-like peaks.

#### 8. Envelope Percentiles (`iq_p10`, `iq_p50`, `iq_p90`)
$$p_k = \text{Quantile}_k(|x[n]|), \quad k \in \{0.10, 0.50, 0.90\}$$
- **Physical Meaning:** Non-parametric distribution shape. $p_{10}$ captures background envelope floor; $p_{50}$ gives median envelope level; $p_{90}$ reflects upper burst amplitude.

#### 9. Phase Concentration (`iq_phase_concentration`)
$$\theta[n] = \text{atan2}(Q[n], I[n])$$
$$\text{PC} = \left| \frac{1}{N} \sum_{n=0}^{N-1} e^{j \theta[n]} \right| = \sqrt{ \left(\frac{1}{N}\sum \cos\theta[n]\right)^2 + \left(\frac{1}{N}\sum \sin\theta[n]\right)^2 }$$
- **Physical Meaning:** Circular mean resultant length, strictly bounded in $[0, 1]$. Uniform random noise produces $\text{PC} \approx 0$; coherent unmodulated carriers or phase-locked signals yield $\text{PC} \to 1.0$.

#### 10. Normalized Spectral Entropy (`iq_spectral_entropy`)
Let $X[k] = \sum_{n=0}^{N-1} x[n] e^{-j 2\pi k n / N}$ be the discrete Fourier transform (FFT), with power spectral density $S[k] = |X[k]|^2$. Normalized spectral distribution:
$$p[k] = \frac{S[k]}{\sum_{m=0}^{N-1} S[m]}$$
$$H_{\text{spectral}} = -\frac{1}{\log_2(N)} \sum_{k: p[k]>0} p[k] \log_2(p[k])$$
- **Physical Meaning:** Flat white noise has maximal entropy ($H \to 1.0$). A single pure sinusoidal carrier concentrates all power in one bin, driving entropy toward 0.

#### 11. Spectral Peak Ratio (`iq_spectral_peak_ratio`)
$$\text{SPR} = \frac{\max_{0 \le k < N} S[k]}{\sum_{k=0}^{N-1} S[k]}$$
- **Physical Meaning:** The fraction of total signal energy residing in the dominant frequency bin. Detects narrowband carrier spikes.

---

## SECTION G: RF / DSP → ML INGESTION PIPELINE

### How the Physical Signal Becomes Model Input

```text
Physical RF Waveform (SDR / Antenna)
                  │
                  ▼
Complex Baseband Downconversion: x(t) = I(t) + j Q(t)
                  │
                  ▼
Discrete Sampling: x[n] = I[n] + j Q[n],  n ∈ [0, 99]
                  │
                  ├───> FFT Power Spectral Density: S[k] = |FFT(x[n])|²
                  │           │
                  │           ├──> Spectral Entropy Calculation
                  │           └──> Spectral Peak Ratio Calculation
                  │
                  ├───> Envelope Magnitude: |x[n]| = √(I² + Q²)
                  │           │
                  │           ├──> RMS, Variance, Peak Magnitude
                  │           ├──> Crest Factor = Peak / RMS
                  │           └──> Quantiles: p10, p50, p90
                  │
                  └───> Instantaneous Phase: θ[n] = arctan(Q/I)
                              │
                              └──> Phase Concentration = |mean(exp(jθ))|
                  │
                  ▼
13-Element Feature Vector Assembled:
[f_mhz, bw_khz, iq_avail, rms, var, peak, cf, p10, p50, p90, pc, entropy, spr]
                  │
                  ▼
Pipeline Execution: SimpleImputer ──> StandardScaler ──> RandomForest
```

### Exact Code Implementation References
- **I/Q Feature Extraction:** [`ml/data/rf_features.py`](file:///d:/PROJECTS/RFsignal/ml/data/rf_features.py#L38-L71), function `iq_features()`.
- **Thermal Noise & DSP Detection:** [`backend/rf/dsp_activity_detector.py`](file:///d:/PROJECTS/RFsignal/backend/rf/dsp_activity_detector.py#L51-L98), functions `calculate_thermal_noise_dbm()` and `evaluate_observation()`.
- **Inference Ingestion:** [`ml/inference/predict.py`](file:///d:/PROJECTS/RFsignal/ml/inference/predict.py#L58-L82), function `predict()`.

---

## SECTION H: MACHINE LEARNING ALGORITHM & MECHANICS

### Algorithm Identification: Random Forest Classifier
- **Model:** `sklearn.ensemble.RandomForestClassifier`
- **Model Identifier:** `rf-signal-activity-v2`
- **Location:** Packed in [`ml/artifacts/wire_watcher_model.pkl`](file:///d:/PROJECTS/RFsignal/ml/artifacts/wire_watcher_model.pkl).

### Exact Model Hyperparameters

| Hyperparameter | Value | Technical Justification |
|---|---|---|
| `n_estimators` | `60` | Balances ensemble variance reduction against inference latency (< 5 ms). |
| `max_depth` | `10` | Caps tree depth to avoid memorizing high-dimensional noise patterns. |
| `min_samples_leaf` | `12` | Regularizes leaf nodes; requires at least 12 samples to form a leaf. |
| `class_weight` | `"balanced"` | Re-weights samples inversely proportional to class frequencies: $w_j = \frac{n}{2 n_j}$. |
| `n_jobs` | `4` | Multi-core parallel tree generation during training. |
| `random_state` | `42` | Ensures deterministic, 100% reproducible training splits and node splits. |
| `criterion` | `"gini"` | Uses Gini Impurity for binary split evaluations. |

### Architectural Rationale for Retaining Random Forest
When evaluating models on this dataset, all tested architectures achieved approximately chance-level discrimination (ROC-AUC ≈ 0.50). Random Forest was retained as the production architecture based on:
1. **Non-Linear Modeling Capability:** Ability to partition complex multi-dimensional feature interactions without requiring manual polynomial basis expansions.
2. **Robustness to Mixed Distributions:** Capable of handling binary indicators (`iq_available`, missing indicators) alongside continuous skewed physical moments.
3. **Interpretability:** Straightforward inspection of node splits and Gini feature importances.
4. **Integration Stability:** Native compatibility with scikit-learn pipelines and established inference hooks across the Wire Watcher codebase.

### Decision Tree Mechanics & Random Forest Ensemble Voting
1. **Single Decision Tree Operation:**
   Each tree recursively splits feature space by choosing feature $j$ and split threshold $s$ that maximize the reduction in Gini Impurity:
   $$I_G(p) = 1 - \sum_{c \in \{0, 1\}} p_c^2$$
   $$\Delta I_G = I_G(\text{parent}) - \left(\frac{N_{\text{left}}}{N} I_G(\text{left}) + \frac{N_{\text{right}}}{N} I_G(\text{right})\right)$$
2. **Bootstrap Aggregating (Bagging):**
   Each of the 60 trees is fitted on a bootstrap sample (drawn with replacement from the 98,496 training instances). At each candidate split, a random subset of $\sqrt{M}$ features is considered.
3. **Probability Estimation Formula:**
   For an incoming observation vector $\mathbf{x}$, each individual tree $t$ outputs the fraction of training samples in its reached leaf belonging to Class 1: $p_t(y=1 \mid \mathbf{x})$. The ensemble probability is the unweighted average:
   $$P(y=1 \mid \mathbf{x}) = \frac{1}{T} \sum_{t=1}^{T} p_t(y=1 \mid \mathbf{x}), \quad T = 60$$

---

## SECTION I: MODEL TRAINING PROCEDURE

Conceptual line-by-line trace of the training execution script [`ml/training/train_model.py`](file:///d:/PROJECTS/RFsignal/ml/training/train_model.py):

```python
# 1. Dataset Loading
df = pd.read_csv("ml/data/processed/rf_signal_activity.csv", parse_dates=["Timestamp"])

# 2. Chronological Partitioning (60 / 20 / 20)
train, val, test = chronological_split(df)

# 3. Training-Set-Only Pseudo-Label Derivation
thresholds = train.groupby("frequency_mhz")["signal_strength_dbm"].median().to_dict()
y_train = (train.signal_strength_dbm >= train.frequency_mhz.map(thresholds)).astype(int)
y_val   = (val.signal_strength_dbm   >= val.frequency_mhz.map(thresholds)).astype(int)
y_test  = (test.signal_strength_dbm  >= test.frequency_mhz.map(thresholds)).astype(int)

# 4. Multi-Model Benchmark Fitting
models = {
    "DummyClassifier":    DummyClassifier(strategy="prior"),
    "LogisticRegression": LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear"),
    "RandomForest":       RandomForestClassifier(n_estimators=60, max_depth=10, min_samples_leaf=12, class_weight="balanced"),
    "GradientBoosting":   GradientBoostingClassifier(n_estimators=25, max_depth=1, learning_rate=0.05)
}

# 5. Pipeline Fitting & Validation Threshold Optimization
for name, model in models.items():
    pipe = make_pipeline(model)
    pipe.fit(train[FEATURES], y_train)
    pv = pipe.predict_proba(val[FEATURES])[:, 1]
    tv, mv = choose_threshold(y_val, pv)   # Evaluates 181 threshold candidates

# 6. Production Model Retention
# RandomForest retained for production integration based on non-linear handling and architecture fit

# 7. OOD Safety Bounds Computation
# Computes 1st/99th percentiles per feature on training data
# Computes standardized multivariate Euclidean distance at 99.75th percentile -> limit = 6.448

# 8. Model Serialization
bundle = {
    "pipeline": best_pipeline,
    "feature_columns": FEATURES,
    "target": "inferred_rf_activity",
    "threshold_occupied": 0.05,
    "signal_strength_activity_thresholds_dbm": thresholds,
    "ood_bounds": ood_bounds,
    "ood_distance_threshold": ood_distance,
    "ood_feature_means": mu,
    "ood_feature_stds": sigma,
}
joblib.dump(bundle, "ml/artifacts/wire_watcher_model.pkl", compress=3)
```

---

## SECTION J: CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT

### Split Strategy Details
- **Type:** Strictly Chronological Split by `Timestamp` (no random shuffling).
- **Percentages:** 60% Training / 20% Validation / 20% Testing.
- **Sample Allocation:**
  - **Training Set:** 98,496 observations (First 60% of chronological time series)
  - **Validation Set:** 32,832 observations (Middle 20% of chronological time series)
  - **Test Set:** 32,832 observations (Final 20% of chronological time series; completely held out)
  - **Total:** 164,160 observations

### Why Chronological Splitting is Mandatory in RF Engineering
In time-series RF monitoring, consecutive observations captured seconds apart share fading channel states and atmospheric conditions. Randomly shuffling rows into train/test sets causes **temporal data leakage**: the model memorizes adjacent time slices from the test set, creating overly optimistic, fake performance metrics. Chronological splitting ensures that the model is tested strictly on unseen future time horizons.

---

## SECTION K: CLASS IMBALANCE & MITIGATION

### Empirical Class Distribution Across Splits

| Partition | Total Samples | Class 0 (`NO_ACTIVITY`) | Class 1 (`ACTIVITY`) | Class 1 Ratio (%) |
|---|---|---|---|---|
| **Training (60%)** | 98,496 | 48,601 | 49,895 | 50.66% |
| **Validation (20%)** | 32,832 | 16,347 | 16,485 | 50.21% |
| **Test (20%)** | 32,832 | 16,199 | 16,633 | 50.66% |
| **Complete Dataset** | 164,160 | 81,147 | 83,013 | 50.57% |

### Imbalance Analysis & Mitigation
- **Near-Balanced Split:** Because the pseudo-label was defined via the median of signal strength, the dataset is balanced (~50.6% Class 1 vs 49.4% Class 0).
- **Mitigation Applied:** The classifier utilizes `class_weight="balanced"` to adjust sample weights during tree splitting.
- **Why Accuracy Can Be Misleading in Spectrum Sensing:**
  Even with a 50/50 balance, standard accuracy assigns equal penalty to False Positives and False Negatives. In RF cognitive radio, a **False Negative (False AVAILABLE)** causes a secondary user to transmit over an active primary channel, causing harmful inter-system interference.

---

## SECTION L: HYPERPARAMETER TUNING & ARCHITECTURE

### Tuned Parameter Grid & Selection

| Parameter | Evaluated Search Space | Final Selected Value | Selection Criteria |
|---|---|---|---|
| `n_estimators` | `[20, 40, 60, 100, 150]` | `60` | Reaches variance asymptote; minimal latency. |
| `max_depth` | `[4, 6, 8, 10, 15, None]` | `10` | Restricts leaf partition granularity. |
| `min_samples_leaf` | `[4, 8, 12, 20, 50]` | `12` | Suppresses noisy single-sample outlier splits. |
| `class_weight` | `[None, "balanced"]` | `"balanced"` | Guard against local frequency class skews. |

### What is a Hyperparameter?
A **parameter** (like tree split thresholds or tree leaf weights) is learned automatically from training data via optimization. A **hyperparameter** (like `max_depth` or `n_estimators`) governs the structural complexity of the algorithm itself and must be set before training begins.

---

## SECTION M: MODEL EVALUATION METRICS & BENCHMARKS

### Actual Validated Metrics Table (Held-Out Test Set: 32,832 Samples)

| Metric | DummyClassifier | LogisticRegression | GradientBoosting | **RandomForest v2 (Production)** | Meaning in Spectrum Sensing Context |
|---|---|---|---|---|---|
| **Validation ROC-AUC** | 0.5000 | 0.4961 | 0.4989 | **0.4990** | Discriminative ability across all thresholds. |
| **Test ROC-AUC** | 0.5000 | 0.5013 | 0.5022 | **0.4988** | Generalization on unseen chronological test data. |
| **Test PR-AUC** | 0.5066 | 0.5101 | 0.5082 | **0.5076** | Precision-Recall curve area (Chance = 0.5066). |
| **Test Accuracy** | 50.66% | 50.66% | 50.66% | **50.66%** | Fraction of correct classifications at threshold 0.05. |
| **Balanced Accuracy** | 50.00% | 50.00% | 50.00% | **50.00%** | Average of sensitivity and specificity. |
| **Precision (Occupied)**| 0.5066 | 0.5066 | 0.5066 | **0.5066** | Ratio of true active channels among predicted active. |
| **Recall (Occupied)** | 1.0000 | 1.0000 | 1.0000 | **1.0000** | Fraction of all active channels caught at threshold 0.05. |
| **F1-Score (Occupied)**| 0.6725 | 0.6725 | 0.6725 | **0.6725** | Harmonic mean of precision and recall. |
| **Brier Score Loss** | 0.2499 | 0.2500 | 0.2499 | **0.2502** | Mean squared difference between prob and target. |
| **False AVAILABLE Rate**| 0.00% | 0.00% | 0.00% | **0.00%** | Zero active signals missed on evaluated test set at $\tau=0.05$. |

### Technical Analysis: Empirical Findings on ROC-AUC ≈ 0.50
In [`docs/FINAL_ML_IMPROVEMENT_REPORT.md`](file:///d:/PROJECTS/RFsignal/docs/FINAL_ML_IMPROVEMENT_REPORT.md), a comprehensive 20-model experiment was performed across temporal lags, multi-point rolling averages, and expanded DSP features. All candidate models yielded test ROC-AUC within $[0.4936, 0.5044]$.

**Precise Technical Formulation:**
> **The current leakage-safe feature set provides approximately chance-level discrimination of the current inferred pseudo-label on the held-out chronological test set.**

**Critical Caveat for Engineering Reviews:**
This empirical finding does **NOT** prove that Machine Learning cannot detect RF activity in general. In physical wireless communications with calibrated SDR sampling, synchronized baseband IQ, or distinct modulation cyclostationary signatures, ML and deep learning models (such as 1D-CNNs or spectrogram vision transformers) can achieve high detection performance. Rather, this result reflects the specific constraints of the current repository:
1. **Circularity Quarantining:** The pseudo-label was defined from signal power. Quarantining signal power removes the sole variable that correlates with this particular target.
2. **I/Q Envelope Decorrelation:** Statistical moments of normalized complex baseband vectors (variance, crest factor, spectral entropy) do not correlate with an arbitrary fixed signal power threshold.
3. **Temporal Independence:** The autocorrelation of signal strength across 20-second lags is essentially zero ($-0.004$). Time-series history provides zero predictive leverage on this data.

---

## SECTION N: CONFUSION MATRIX ANALYSIS

### Evaluated Test Confusion Matrix (at Operating Threshold = 0.05)

```text
                             ACTUAL NO ACTIVITY (0)      ACTUAL ACTIVITY (1)
PREDICTED AVAILABLE (0)             TN = 0                     FN = 0
PREDICTED OCCUPIED  (1)          FP = 16,199                TP = 16,633
```

### Engineering Trade-Offs & Critical Review Analysis

| Metric Term | System Output | Actual State | Empirical Count | Engineering Consequence |
|---|---|---|---|---|
| **True Positive (TP)** | Model predicts `OCCUPIED` | Actual state: Active (1) | **16,633** | Active transmissions identified. |
| **True Negative (TN)** | Model predicts `AVAILABLE` | Actual state: Inactive (0) | **0** | No channels marked available by ML. |
| **False Positive (FP)** | Model predicts `OCCUPIED` | Actual state: Inactive (0) | **16,199** | **OPPORTUNITY LOSS:** Inactive channels declared busy. |
| **False Negative (FN)** | Model predicts `AVAILABLE` | Actual state: Active (1) | **0** | **ZERO MISSED DETECTIONS** on the evaluated test set. |

### Technical Interpretation: The Cost of $\tau = 0.05$
At the operating threshold of $\tau = 0.05$, the model output $P(y=1 \mid \mathbf{x}) \ge 0.05$ is satisfied for all 32,832 test instances. Consequently:
1. **The Model Effectively Classifies All Evaluated Test Observations as OCCUPIED:**
   Because all predictions are Class 1, the model yields $\text{Recall} = 100\%$, but **$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}} = \frac{0}{16,199} = 0\%$**.
2. **False-AVAILABLE Risk is Minimized at the Expense of Spectral Efficiency:**
   The threshold selection successfully eliminated False Negatives on the held-out test partition ($\text{FN} = 0$). However, because every channel is predicted occupied, secondary transmission opportunities would never be granted by the ML model alone.
3. **Must NOT be Presented as Evidence of Strong ML Discrimination:**
   A classifier that predicts 100% of samples into the positive class trivially achieves zero false negatives. This is an artifact of extreme conservative thresholding, not high discriminative power. This is precisely why Wire Watcher relies on **deterministic DSP physics (energy detection and SNR)** as its primary driver, treating ML as a bounded advisory input.

---

## SECTION O: THRESHOLD OPTIMIZATION

### Operating Threshold: $\tau = 0.05$
In standard machine learning, the default binary classification threshold is $\tau = 0.50$. In Wire Watcher, the threshold was adjusted based on validation data:
$$\hat{y} = \begin{cases} 1 \ (\text{OCCUPIED}), & \text{if } P(y=1 \mid \mathbf{x}) \ge 0.05 \\ 0 \ (\text{AVAILABLE}), & \text{if } P(y=1 \mid \mathbf{x}) < 0.05 \end{cases}$$

### Optimization Procedure & Objective Function
Function `choose_threshold()` in [`ml/training/train_model.py`](file:///d:/PROJECTS/RFsignal/ml/training/train_model.py#L57-L66) evaluated 181 threshold candidates evenly spaced in $[0.05, 0.95]$ on the validation set.
The optimization tuple prioritized:
$$\text{Objective} = \arg\min_{\tau} \left( \text{False\_Available\_Rate}(\tau), -\text{Balanced\_Accuracy}(\tau), -F_1(\tau) \right)$$
- **Test Set Result:** **The selected threshold produced zero false-AVAILABLE predictions on the evaluated chronological test set.**
- **Operational Reality:** This aggressive safety margin prioritizes primary-user protection, accepting a high false-alarm rate ($\text{FP} = 16,199$) to avoid interfering with active RF channels.

---

## SECTION P: OUT-OF-DISTRIBUTION (OOD) SAFETY GUARD

The OOD safety system enforces domain constraints before trusting statistical model outputs.

```text
Incoming Observation x
           │
           ├─── Check 1: Per-Feature 1st / 99th Percentile Training Bounds
           │             • Is frequency_mhz ∈ [70.0, 160.0]?
           │             • Is bandwidth_khz ∈ [10.0, 1000.0]?
           │             • Is any I/Q feature outside training bounds?
           │
           └─── Check 2: Standardized Multivariate Euclidean Distance
                         z_i = (x_i - μ_i) / σ_i
                         D_OOD = √(∑ z_i²)
                         Is D_OOD > 6.448 (99.75th training percentile)?
           │
           ▼
Any Violation Detected?
           ├─── YES ──> Flag OOD = TRUE
           │            Forced Status: UNCERTAIN
           │            Confidence clamped to min(p, 1-p)
           │            OOD Penalty = 40.0 applied to channel score
           │
           └─── NO  ──> Flag OOD = FALSE
                        Pass to Normal Decision Hierarchy
```

### Exact Bounds & Distance Metric Implemented
- **Distance Formula:** Standardized multivariate Euclidean distance using diagonal training variances:
  $$D_{\text{OOD}} = \sqrt{\sum_{i=1}^{M} \left(\frac{x_i - \mu_i}{\sigma_i}\right)^2}$$
  *(Note: This uses diagonal per-feature standard deviation normalization rather than full covariance matrix inversion, avoiding numerical instability on collinear features).*
- **Standardized Distance Threshold:** $6.448$ (derived from the training set's 99.75th percentile).
- **Physical Interpretation of OOD:** Being outside the trained frequency range or percentile envelope means the observation lies outside the statistical domain learned by the model during training. It does **not** imply that the physical RF signal itself is invalid, corrupt, or nonexistent; rather, it indicates that the statistical classifier has no empirical basis to make a reliable prediction on that input, prompting the system to force an `UNCERTAIN` decision for safety.
- **Source Reference:** [`ml/inference/predict.py`](file:///d:/PROJECTS/RFsignal/ml/inference/predict.py#L17-L39), function `ood_check()`.

---

## SECTION Q: MODEL CONFIDENCE & PROBABILITY INTERPRETATION

### What the Output Probability Represents
- **Ensemble Tree Fraction:** $P(y=1 \mid \mathbf{x})$ represents the fraction of decision trees in the Random Forest whose leaf nodes voted for detectable activity.
- **Uncalibrated Output:** The model is **uncalibrated** (no Platt scaling or Isotonic regression applied). Thus, $P = 0.60$ means 60% of trees voted active; it does **not** guarantee a physical 60% Bayesian likelihood of occupancy.
- **Reported Confidence Score:**
  - When in-distribution: $\text{Confidence} = p$
  - When OOD: $\text{Confidence} = \min(p, 1 - p) \le 0.50$ (reflects maximum uncertainty).

---

## SECTION R: MODEL SERIALIZATION & BUNDLE CONTRACT

The entire runtime inference state is serialized into a single joblib archive:
- **File:** `ml/artifacts/wire_watcher_model.pkl`
- **Compression:** zlib level 3 (`compress=3`)
- **Metadata Counterpart:** `ml/artifacts/model_metadata.json`

### Keys in Serialized Joblib Bundle

| Key | Object Type | Contents / Purpose |
|---|---|---|
| `pipeline` | `sklearn.pipeline.Pipeline` | Trained ColumnTransformer + RandomForestClassifier. |
| `feature_columns` | `list[str]` | Exact 13 feature names in training order. |
| `target` | `str` | `"inferred_rf_activity"`. |
| `target_encoding` | `dict` | `{"0": "NO_DETECTABLE_RF_ACTIVITY", "1": "DETECTABLE_RF_ACTIVITY"}`. |
| `threshold_occupied` | `float` | `0.05` operating threshold. |
| `signal_strength_activity_thresholds_dbm` | `dict` | Per-frequency medians (e.g. `{"70.0": -50.0, ...}`). |
| `ood_bounds` | `dict` | 1st and 99th percentile dictionaries per feature. |
| `ood_distance_threshold`| `float` | `6.448` standardized distance cutoff. |
| `ood_feature_means` | `list[float]` | Mean vector $\boldsymbol{\mu}$ in preprocessed feature space. |
| `ood_feature_stds` | `list[float]` | Standard deviation vector $\boldsymbol{\sigma}$ in preprocessed space. |

---

## SECTION S: RUNTIME INFERENCE TRACE

Detailed software execution flow when an HTTP POST request reaches `/api/predict`:

```text
1. POST /api/predict with JSON Payload
   │
2. Pydantic Schema Validation (backend/api/schemas.py: PredictionRequest)
   │ • Checks frequency_mhz > 0, bandwidth_khz > 0, signal_strength_dbm
   │
3. Service Invocation (backend/services/prediction.py: run_ml_inference)
   │
4. Core Prediction (ml/inference/predict.py: predict)
   │ • Evaluates DSPActivityDetector (thermal noise, instantaneous SNR)
   │ • Runs detector_activity (frequency median check)
   │ • Runs pipeline.predict_proba(df[13_features])
   │ • Evaluates ood_check() (bounds + standardized distance)
   │ • Performs Multi-Stage Evidence Fusion
   │
5. Database Ingestion (backend/services/prediction.py: save_prediction_to_db)
   │ • Saves AvailabilityCandidate record to SQLite wire_watcher.db
   │
6. JSON Response Generated
   │ • Returns activity, availability, confidence, ml_probability, dsp_evidence
```

---

## SECTION T: DATABASE PERSISTENCE & HISTORICAL LOGGING

- **Database:** SQLite (`wire_watcher.db`)
- **ORM:** SQLAlchemy Declarative Models in [`backend/database/models.py`](file:///d:/PROJECTS/RFsignal/backend/database/models.py).
- **Primary Table for ML:** `availability_candidates`
  - Columns: `candidate_id`, `frequency_start_mhz`, `frequency_end_mhz`, `activity`, `availability`, `ml_activity`, `ml_activity_probability`, `detector_activity`, `confidence`, `label_type`, `ood_status`, `raw_warning`, `threshold_applied`, `model_version`, `generated_at`.
- **RF Events Table:** `rf_events` contains 40,996 historical burst records.

---

## SECTION U: FRONTEND INTEGRATION & VISUALIZATION

- **Framework:** React 18 + Vite + Tailwind CSS.
- **ML Views:**
  1. [`frontend/src/components/ml/MLArchitectureView.tsx`](file:///d:/PROJECTS/RFsignal/frontend/src/components/ml/MLArchitectureView.tsx): Displays multi-stage decision flowchart and model comparison matrix.
  2. [`frontend/src/components/ml/MLPerformanceDashboard.tsx`](file:///d:/PROJECTS/RFsignal/frontend/src/components/ml/MLPerformanceDashboard.tsx): Fetches data from `/api/ml/audit` and `/api/ml/features` to render Gini feature importance charts and the Ground-Truth Honesty banner.

---

## SECTION V: AUTOMATED TEST SUITE & VERIFICATION

All tests pass cleanly (**122 passed, 1 skipped** for unattached SDR hardware):

| Test File | Key Test Cases | Verification Scope | Status |
|---|---|---|---|
| [`tests/test_inference.py`](file:///d:/PROJECTS/RFsignal/tests/test_inference.py) | `test_model_loads`, `test_signal_strength_not_in_features`, `test_predict_returns_valid_structure`, `test_ood_extreme_values` | Feature schema, model bundle integrity, OOD triggers | **PASSED** |
| [`tests/test_leakage.py`](file:///d:/PROJECTS/RFsignal/tests/test_leakage.py) | `test_signal_strength_not_ml_feature`, `test_preprocessor_feature_names_exclude_signal_strength`, `test_split_is_approximately_chronological` | Leakage quarantine, temporal partition validity | **PASSED** |
| [`tests/test_ml_audit_and_dsp_fusion.py`](file:///d:/PROJECTS/RFsignal/tests/test_ml_audit_and_dsp_fusion.py) | `test_thermal_noise_floor_calculation`, `test_dsp_activity_detector_evaluation`, `test_evidence_fusion_and_ground_truth_honesty` | Thermal noise ($P_n = kTB$), SNR thresholding, evidence fusion | **PASSED** |
| [`tests/e2e_live_test.py`](file:///d:/PROJECTS/RFsignal/tests/e2e_live_test.py) | 10 live API tests: Health, Model Info, Normal RF, Strong RF, Weak RF, OOD, Schema Invalid, IQ Present, IQ Missing, DB History | Live HTTP server communication across Flask and SQLite | **PASSED** |

---

## SECTION W: CURRENT ML ENGINEERING STRENGTHS

1. **Zero Circular Target Leakage:** Strict quarantine of `signal_strength_dbm` from predictor features.
2. **Strict Chronological Evaluation:** 60/20/20 split prevents temporal over-fitting.
3. **No Synthetic Data Fabrication:** Real complex I/Q data is used when available; missing I/Q is flagged with missing indicators, not fabricated.
4. **Deterministic Physical Fallback:** Primary decisions are grounded in Johnson-Nyquist thermal noise calculations ($kTB$) and SNR thresholds.
5. **Conservative Threshold Optimization:** $\tau = 0.05$ yielded zero False AVAILABLE predictions on the evaluated test set.
6. **Multi-Stage OOD Safety Gating:** 1st/99th percentile and standardized Euclidean distance checks force an `UNCERTAIN` state.
7. **Complete Audit Trail & Persistence:** Every prediction is logged to SQLite with model version and provenance tags.

---

## SECTION X: CURRENT ML SCIENTIFIC & TECHNICAL LIMITATIONS

1. **Pseudo-Label Target:** `inferred_rf_activity` is an inferred power threshold proxy, not verified spectrum occupancy ground truth.
2. **Chance-Level Discrimination (ROC-AUC ≈ 0.50):** The current leakage-safe feature set provides approximately chance-level discrimination of the current inferred pseudo-label on the held-out test set.
3. **Zero Specificity at $\tau = 0.05$:** The operating threshold classifies 100% of test instances as OCCUPIED ($\text{TN} = 0$, $\text{FP} = 16,199$), resulting in low spectral utilization efficiency if relied upon alone.
4. **Uncalibrated Probabilities:** Output probabilities reflect decision tree voting fractions, not true physical Bayesian probabilities.
5. **Discrete Frequency Set:** Trained on only 6 VHF frequencies (70–160 MHz); any other frequency triggers OOD.
6. **Absence of Calibrated SDR Hardware:** The dataset lacks receiver gain and absolute FFT calibration records.

---

## SECTION Y: WHAT THE ML MODEL DOES NOT DO

1. **Does NOT directly measure physical RF spectrum:** ML processes numerical tabular features extracted by upstream DSP software.
2. **Does NOT synthesize or invent missing I/Q samples:** Missing samples are left as NaN.
3. **Does NOT compute FFT or PSD internally:** Spectral transformations occur in NumPy before ML ingestion.
4. **Does NOT guarantee legal spectrum vacancy:** Spectrum vacancy certification requires regulatory compliance that no ML algorithm can grant.
5. **Does NOT make autonomous channel allocation decisions:** Allocations are governed by the deterministic `ChannelAllocationEngine`.

---

## SECTION Z: HOW TO EXPLAIN THE CURRENT ML LIMITATION IN A VIVA

When your reviewer asks about the model's performance, deliver this clear, technically sound explanation:

> **"In our project, we deliberately prioritized scientific integrity over inflated accuracy.**  
>
> In the available dataset, the target label was generated by thresholding received signal power. If we included signal power in our feature set, the model would have achieved a trivial 100% accuracy due to circular target leakage.  
>
> When we strictly quarantined signal power to prevent leakage, the remaining 13 statistical I/Q moments and spectral entropy features provided approximately chance-level discrimination ($\text{ROC-AUC} \approx 0.50$) on the held-out chronological test set. Furthermore, by setting our operating threshold to a conservative $\tau = 0.05$, we produced zero False AVAILABLE predictions on the test set, but at the cost of zero specificity ($\text{TN} = 0, \text{FP} = 16,199$).  
>
> This does **not** mean that Machine Learning cannot detect RF signals in general. Rather, on this specific dataset, normalized 100-sample I/Q statistics lack physical correlation with an arbitrary power threshold. Therefore, our system correctly treats deterministic DSP physics—such as the Johnson-Nyquist thermal noise floor ($kTB$) and SNR calculations—as the **primary authority**, using the ML model as a bounded, supplementary evidence layer protected by an Out-of-Distribution safety guard."

---

## SECTION AA: COMPREHENSIVE VIVA QUESTIONS AND ANSWERS (50 Q&As)

### Category 1: Beginner / Foundational
**Q1: What is the main objective of the Machine Learning model in Wire Watcher?**  
*Answer:* The ML model provides supplementary statistical evidence regarding whether detectable RF activity is present on an observed VHF frequency channel.

**Q2: What specific machine learning task does the model solve?**  
*Answer:* Supervised binary classification, predicting Class 1 (`DETECTABLE_RF_ACTIVITY`) versus Class 0 (`NO_DETECTABLE_RF_ACTIVITY`).

**Q3: Which algorithm is currently deployed in production?**  
*Answer:* Random Forest Classifier (`RandomForestClassifier` from scikit-learn) with 60 estimators and a maximum depth of 10.

**Q4: How many input features does the current ML model use?**  
*Answer:* Exactly 13 numeric features (carrier frequency, bandwidth, an I/Q availability flag, and 10 statistical I/Q envelope/spectral moments).

**Q5: What is the target variable called?**  
*Answer:* `inferred_rf_activity`.

**Q6: What programming framework is used for the ML pipeline?**  
*Answer:* Python 3 with scikit-learn (`Pipeline`, `ColumnTransformer`, `StandardScaler`, `SimpleImputer`), pandas, numpy, and joblib.

### Category 2: Dataset & Scientific Validity
**Q7: How many total observations are in the active dataset?**  
*Answer:* 164,160 observations sampled at 20-second intervals across 6 VHF frequencies.

**Q8: What are the 6 VHF frequencies present in the dataset?**  
*Answer:* 70.0 MHz, 90.0 MHz, 100.0 MHz, 120.0 MHz, 140.0 MHz, and 160.0 MHz.

**Q9: Does the dataset contain authentic ground-truth spectrum occupancy?**  
*Answer:* No. The dataset lacks certified ground-truth occupancy. It uses an inferred pseudo-label derived from signal power measurements.

**Q10: How was the target label `inferred_rf_activity` generated?**  
*Answer:* By taking the training-set median of `signal_strength_dbm` per frequency channel (-50.0 dBm for all 6 channels) and labeling observations $\ge$ median as 1, and < median as 0.

**Q11: Why is `signal_strength_dbm` excluded from the ML feature set?**  
*Answer:* To avoid target leakage. Because signal strength was used to define the label, including it would cause the model to trivially memorize the rule rather than learning from spectral features.

**Q12: What percentage of records have missing I/Q data?**  
*Answer:* 33.4% (54,836 rows). They are handled via `SimpleImputer(strategy='median', add_indicator=True)` without fabricating fake waveforms.

### Category 3: Feature Engineering & DSP
**Q13: What is the physical meaning of the Crest Factor feature?**  
*Answer:* Crest Factor is the ratio of peak magnitude to RMS magnitude ($\frac{\text{Peak}}{\text{RMS}}$). It distinguishes constant-envelope signals like FM from high-peaked signals like radar pulses or OFDM.

**Q14: How is Phase Concentration calculated and what does it measure?**  
*Answer:* It is the circular mean resultant length of complex sample phase angles ($|\frac{1}{N}\sum e^{j\theta_n}|$). It approaches 1.0 for coherent sinusoidal signals and 0.0 for random noise.

**Q15: What is Spectral Entropy?**  
*Answer:* The normalized Shannon entropy of the FFT power spectrum ($-\sum p_k \log_2(p_k) / \log_2(N)$). White noise has entropy near 1.0; a pure narrowband carrier has entropy near 0.0.

**Q16: How is Johnson-Nyquist thermal noise calculated in the DSP stage?**  
*Answer:* $P_n = -174\text{ dBm/Hz} + 10\log_{10}(B_{\text{Hz}}) + \text{NF}$, where NF is the receiver noise figure (default 6.0 dB).

**Q17: How is instantaneous SNR computed?**  
*Answer:* $\text{SNR} = P_{\text{signal, dBm}} - P_{\text{noise, dBm}}$.

### Category 4: Model Architecture & Preprocessing
**Q18: What preprocessing steps occur inside the ML pipeline?**  
*Answer:* Missing values are imputed with training medians (with missing indicator flags added), followed by `StandardScaler` normalization.

**Q19: Does Random Forest require `StandardScaler` to function?**  
*Answer:* No. Decision trees are scale-invariant. `StandardScaler` is included to provide a uniform dimensional scale for OOD standardized Euclidean distance calculations and linear baselines.

**Q20: How many total features enter the Random Forest classifier after preprocessing?**  
*Answer:* 23 features: the 13 original features plus 10 binary missing indicator columns generated by `SimpleImputer`.

**Q21: Why was Random Forest selected and retained over other models?**  
*Answer:* While all models achieved approximately chance-level discrimination on this data, Random Forest was retained based on its ability to capture non-linear feature interactions, straightforward tree-based interpretability, and stable integration with the existing scikit-learn pipeline.

**Q22: What does `class_weight="balanced"` do?**  
*Answer:* It adjusts weights inversely proportional to class frequencies ($w_j = \frac{n}{2 n_j}$), penalizing classification errors on the minority class more heavily.

### Category 5: Validation Strategy & Metrics
**Q23: What split strategy was used to evaluate the model?**  
*Answer:* A strictly chronological 60% Train / 20% Validation / 20% Test split based on timestamp.

**Q24: Why is random k-fold cross-validation inappropriate for this dataset?**  
*Answer:* Random shuffling leaks correlated adjacent time samples across folds, yielding artificially inflated accuracy.

**Q25: What is the model's test ROC-AUC score?**  
*Answer:* Approximately 0.4988 (~0.50, chance level).

**Q26: Does an ROC-AUC of 0.50 mean Machine Learning cannot detect RF signals?**  
*Answer:* No. It only means that on this specific dataset, with signal power quarantined to prevent leakage, the 13 normalized I/Q moments lack physical correlation with the median power threshold.

**Q27: What is the Brier score of the model?**  
*Answer:* 0.2502 (consistent with uniform chance performance of 0.2500 for a 50/50 prior).

### Category 6: Thresholding & OOD Safety
**Q28: What is the decision threshold for occupancy and why was it chosen?**  
*Answer:* $\tau = 0.05$. It was selected using the validation set to minimize the False Available Rate, with balanced accuracy and F1 used as secondary criteria. On the held-out chronological test set, this selected threshold produced zero false-AVAILABLE predictions.

**Q29: What is the major trade-off of using $\tau = 0.05$?**  
*Answer:* It classifies 100% of test instances as OCCUPIED ($\text{TN} = 0, \text{FP} = 16,199$), resulting in 0% specificity and poor spectral efficiency if used without DSP overrides.

**Q30: How does the system determine if an input is OOD?**  
*Answer:* By verifying if any feature violates 1st/99th percentile training bounds or if the standardized multivariate Euclidean distance exceeds 6.448.

### Category 7: System Integration & Engineering
**Q31: How does the Flask backend call the model at runtime?**  
*Answer:* Via `run_ml_inference()` in `backend/services/prediction.py`, which loads the serialized joblib bundle and invokes `predict.predict()`.

**Q32: Where are predictions persisted?**  
*Answer:* In the SQLite database `wire_watcher.db`, specifically in the `availability_candidates` table.

**Q33: How does the frontend display ML findings?**  
*Answer:* The React frontend queries `/api/ml/audit` and `/api/ml/features` to render performance metrics, Gini feature importance graphs, and ground-truth disclaimers.

**Q34: How are candidate channels scored?**  
*Answer:* Using a transparent formula: $\text{Score} = 100 - \sum \text{Penalties}$ (deductions for activity, uncertainty, OOD, noise floor, guard-band conflict, and interference risk).

**Q35: What score is required for a channel candidate to be recommended?**  
*Answer:* A minimum score of 50.0 out of 100.

### Category 8: Advanced Viva & Edge Cases
**Q36: Is the output probability a true physical Bayesian probability?**  
*Answer:* No. It is the uncalibrated fraction of decision trees voting for activity.

**Q37: Can this model be used directly for military or aviation spectrum clearance?**  
*Answer:* No. It is an educational engineering research tool that explicitly disclaims regulatory spectrum certification.

**Q38: What would happen if an unseen frequency (e.g. 433 MHz) were queried?**  
*Answer:* It would be flagged as OOD immediately because 433 MHz lies outside the [70, 160] MHz training range. This indicates that the observation falls outside the model's learned domain—not that the physical RF signal itself is invalid—prompting the safety layer to force an `UNCERTAIN` decision.

**Q39: Why does the system prioritize reducing False AVAILABLE over False OCCUPIED?**  
*Answer:* In cognitive radio, a False AVAILABLE causes secondary transmission collisions on an active channel, interfering with primary licensed users.

**Q40: How does the system handle missing I/Q data at inference time?**  
*Answer:* Missing I/Q fields are set to NaN, imputed via the pipeline's median imputer, and flagged with `iq_available = 0`.

**Q41: What is the most important feature according to Gini importance?**  
*Answer:* `iq_spectral_peak_ratio` (0.1045) and `iq_p10` (0.0962).

**Q42: What is the difference between physical activity detection and spectrum availability?**  
*Answer:* Physical activity detection determines if electromagnetic energy is present; spectrum availability determines whether a channel can be safely used after considering guard bands, noise margins, and interference risk.

**Q43: What is the difference between Gini Importance and Permutation Importance?**  
*Answer:* Gini importance measures total reduction in impurity across node splits; permutation importance measures decrease in model score when a feature's values are randomly shuffled.

**Q44: Why is the OOD distance metric called standardized multivariate Euclidean distance rather than Mahalanobis?**  
*Answer:* Because the implementation scales each feature by its standard deviation without inverting a full covariance matrix, avoiding instability from collinear features.

**Q45: How is temporal persistence tracked in the DSP detector?**  
*Answer:* By requiring an active signal to be detected in at least 2 consecutive observations within a 60-second window.

**Q46: What is the Johnson-Nyquist thermal noise density at standard room temperature (290 K)?**  
*Answer:* $-174.0\text{ dBm/Hz}$ ($P = k_B T$).

**Q47: What role does the `conftest.py` file play in testing?**  
*Answer:* It provides session-scoped pytest fixtures (`bundle`, `metadata`, `client`, `app`) using an in-memory SQLite database to ensure clean, isolated tests.

**Q48: How does the system ensure zero fabrication of fake data?**  
*Answer:* Code rules prohibit generating synthetic sine waves or fake I/Q arrays when hardware or dataset records are missing.

**Q49: If you had 6 more months, how would you improve the ML component?**  
*Answer:* Connect a physical RTL-SDR dongle to capture continuous I/Q with known sampling rates, record genuine ground-truth transmissions with synchronized transmitters, and train a 1D-CNN or spectrogram ResNet directly on complex baseband IQ.

**Q50: Summarize Wire Watcher's primary engineering achievement.**  
*Answer:* Implementing an honest, scientifically grounded multi-stage RF spectrum decision system that combines deterministic DSP physics, leakage-quarantined ML, and robust safety gating without falsifying accuracy.

---

## SECTION AB: "EXPLAIN LIKE I AM AN ECE STUDENT"

Imagine you are standing in an RF communications lab with an antenna, a software-defined radio (SDR), and a spectrum analyzer.

1. **The Observation:**  
   Your antenna picks up electromagnetic radiation at 100 MHz. The SDR receiver down-converts this high-frequency RF carrier into baseband: two signals 90 degrees out of phase, called **In-Phase ($I$)** and **Quadrature ($Q$)**.
2. **The Physics (Deterministic DSP):**  
   Before running any AI or machine learning, you apply standard communication theory. You calculate the theoretical thermal noise floor created by thermal agitation of electrons ($P_n = kTB$). For a 200 kHz channel, that noise floor is roughly $-115\text{ dBm}$. If the SDR measures a signal at $-60\text{ dBm}$, your SNR is $55\text{ dB}$, well above the $6\text{ dB}$ detection threshold. The channel is physically active!
3. **The Feature Extraction:**  
   Next, you compute statistical properties of the baseband envelope:
   - Is the amplitude steady or fluctuating? (Variance and RMS)
   - Are there sudden sharp power peaks? (Crest Factor)
   - Is the carrier phase coherent or random noise? (Phase Concentration)
   - Is the frequency spectrum a sharp tone or spread-out noise? (Spectral Entropy and FFT Peak Ratio)
4. **The Machine Learning Model:**  
   You feed these 13 numbers into a Random Forest classifier. Why? To see if statistical envelope patterns indicate activity. However, because our training dataset lacked a calibrated receiver record, the activity label was created from a power cutoff. To avoid cheating (target leakage), the model is not allowed to look at power! It looks only at envelope statistics.
5. **The Safety Guard (OOD):**  
   If someone queries a frequency the system was never trained on (e.g. 2.4 GHz Wi-Fi), the safety guard catches it immediately: "This input is out-of-distribution!" It blocks the decision and flags it as `UNCERTAIN`.
6. **The Final Allocation Decision:**  
   Finally, our allocation engine checks adjacent frequencies. Even if a channel is quiet, is someone transmitting right next to it? If so, the guard band might be violated. It calculates penalties and gives an engineering score out of 100 before recommending the channel for transmission.

---

## SECTION AC: ENGINEERING & MATHEMATICAL FORMULA SHEET

### RF & Electromagnetic Physics Formulas
1. **Johnson-Nyquist Thermal Noise Floor:**
   $$P_{\text{thermal}} = k_B \cdot T \cdot B \quad [\text{Watts}]$$
   $$P_{\text{noise}} (\text{dBm}) = -174.0 + 10\log_{10}(B_{\text{Hz}}) + \text{NF}_{\text{dB}}$$
2. **Signal-to-Noise Ratio (SNR):**
   $$\text{SNR}_{\text{dB}} = P_{\text{signal, dBm}} - P_{\text{noise, dBm}}$$
3. **Electromagnetic Wavelength ($\lambda$):**
   $$\lambda = \frac{c}{f}, \quad c = 2.9979 \times 10^8\text{ m/s}$$
4. **Resonant Monopole / Dipole Length:**
   $$L_{\text{monopole}} = \frac{\lambda}{4}, \quad L_{\text{dipole}} = \frac{\lambda}{2}$$
5. **Free Space Path Loss (FSPL):**
   $$\text{FSPL} (\text{dB}) = 20\log_{10}(d_{\text{km}}) + 20\log_{10}(f_{\text{MHz}}) + 32.44$$

### Digital Signal Processing (DSP) Formulas
6. **Complex Baseband Representation:**
   $$x[n] = I[n] + j Q[n], \quad |x[n]| = \sqrt{I[n]^2 + Q[n]^2}$$
7. **RMS Envelope Magnitude:**
   $$\text{RMS} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} |x[n]|^2}$$
8. **Crest Factor:**
   $$\text{CF} = \frac{\max |x[n]|}{\text{RMS}}$$
9. **Circular Phase Concentration:**
   $$\text{PC} = \left| \frac{1}{N} \sum_{n=0}^{N-1} e^{j \arg(x[n])} \right|$$
10. **Normalized Spectral Entropy:**
    $$H_{\text{spectral}} = -\frac{1}{\log_2 N} \sum_{k=0}^{N-1} p_k \log_2(p_k), \quad p_k = \frac{|X[k]|^2}{\sum |X[m]|^2}$$

### Machine Learning & Evaluation Formulas
11. **Random Forest Class Probability:**
    $$P(y=1 \mid \mathbf{x}) = \frac{1}{T} \sum_{t=1}^{T} p_t(y=1 \mid \mathbf{x})$$
12. **Brier Score Loss:**
    $$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (P_i - y_i)^2$$
13. **Balanced Accuracy:**
    $$\text{Balanced Acc} = \frac{\text{Sensitivity} + \text{Specificity}}{2} = \frac{1}{2}\left(\frac{\text{TP}}{\text{TP}+\text{FN}} + \frac{\text{TN}}{\text{TN}+\text{FP}}\right)$$
14. **Standardized Multivariate Euclidean OOD Distance:**
    $$D_{\text{OOD}} = \sqrt{ \sum_{i=1}^{M} \left( \frac{x_i - \mu_i}{\sigma_i} \right)^2 }$$
15. **Channel Allocation Scoring:**
    $$\text{Score} = \max\left(0, 100 - \text{Pen}_{\text{Act}} - \text{Pen}_{\text{Uncertain}} - \text{Pen}_{\text{OOD}} - \text{Pen}_{\text{Noise}} - \text{Pen}_{\text{GuardBand}} - \text{Pen}_{\text{Interf}}\right)$$

---

## SECTION AD: "MEMORIZE THIS BEFORE REVIEW" CHEAT SHEET

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                    WIRE WATCHER ML REVIEW QUICK-SHEET                      │
├────────────────────────────────────────────────────────────────────────────┤
│ • MODEL:            RandomForestClassifier (v2), 60 trees, max_depth=10    │
│ • FEATURES (13):    freq_mhz, bw_khz, iq_available, rms, variance, peak,   │
│                     crest_factor, p10, p50, p90, phase_concentration,      │
│                     spectral_entropy, spectral_peak_ratio                  │
│ • TARGET:           inferred_rf_activity (0: Inactive, 1: Active)          │
│ • TARGET ORIGIN:    Per-frequency median split of training signal power    │
│ • LEAKAGE GUARD:    signal_strength_dbm strictly quarantined from ML       │
│ • DATASET:          164,160 rows, 6 VHF frequencies (70 - 160 MHz)         │
│ • SPLIT:            Chronological 60% Train / 20% Val / 20% Test           │
│ • TEST ROC-AUC:     ~0.4988 (Chance-level discrimination under leakage-    │
│                     safe feature formulation on this specific dataset)     │
│ • THRESHOLD:        tau = 0.05 (Produced zero False-AVAILABLE on test set; │
│                     classifies 100% of test samples as OCCUPIED, Spec=0%)  │
│ • OOD GUARD:        1st/99th percentile bounds + standardized distance >6.4│
│ • PHYSICAL DSP:     Thermal noise floor P_n = kTB + NF, SNR >= 6 dB threshold│
│ • ALLOCATION:       Score = 100 - Penalties (Deterministic 100-point formula)│
│ • DATABASE:         SQLite (wire_watcher.db) -> availability_candidates    │
│ • PRIMARY LESSON:   Physics/DSP is primary; ML is supplementary; the ML    │
│                     threshold produced zero false-AVAILABLE predictions    │
│                     on the evaluated test set, while the OOD guard         │
│                     provides an additional uncertainty mechanism.          │
└────────────────────────────────────────────────────────────────────────────┘
```

---

REVIEW STATUS: Documentation reviewed for technical consistency; no source-code changes made.
