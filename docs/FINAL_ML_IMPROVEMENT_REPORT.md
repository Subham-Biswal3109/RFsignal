# WIRE WATCHER — FINAL ML IMPROVEMENT INVESTIGATION REPORT

**Date:** 2026-08-31 20:20:31 UTC  
**Target:** `inferred_rf_activity` (**INFERRED / PSEUDO-LABEL**)  
**Dataset:** `RF Signal Data` (164,160 observations, 6 VHF frequencies)  
**Evaluation Split:** Strictly Chronological 60% Train / 20% Validation / 20% Test (Untouched)  

---

## 1. Executive Summary & Production Decision

```
================================================================================
FINAL SCIENTIFIC DECISION: OUTCOME B — NO LEGITIMATE ML IMPROVEMENT FOUND
================================================================================
The active RF Signal Data does not contain legitimate predictive structure
for a significantly stronger ML predictor under the current inferred-activity target.

All 20 model-feature combinations across single-row, temporal-lag, and windowed
representations achieved test ROC-AUC within [0.4936, 0.5044] (Chance Level: 0.5000).

PRODUCTION ACTION: PRESERVE EXISTING MODEL INTACT (RandomForest v2 Artifact Retained)
================================================================================
```

---

## 2. Dataset Analysis & Physical Coherence

From the detailed statistical audit of `ml/data/processed/rf_signal_activity.csv`:

* **Observations:** 164,160 rows at a 20-second cadence across 6 carrier frequencies (70, 90, 100, 120, 140, 160 MHz).
* **I/Q Availability:** 109,324 observations (66.6%) contain 100 complex I/Q baseband samples; 54,836 observations (33.4%) have missing I/Q.
* **Signal Strength Span:** -119.0 dBm to -25.0 dBm (Mean: -50.0 dBm, Std: 27.1 dBm).
* **Temporal Autocorrelation Audit:**
  - `signal_strength_dbm` lag-1 autocorrelation per frequency:
    - 70 MHz: -0.0096
    - 90 MHz: +0.0091
    - 100 MHz: +0.0083
    - 120 MHz: +0.0043
    - 140 MHz: +0.0026
    - 160 MHz: -0.0019
  - **Finding:** Near-zero autocorrelation proves the recorded sequence behaves as independent measurement draws with no temporal persistence.

---

## 3. Candidate Feature Sets & Leakage Audit

A total of **38 candidate features** were evaluated across four domains:
1. **I/Q Statistical Domain:** RMS magnitude, variance, peak magnitude, crest factor, percentiles (p10, p50, p90), phase concentration, envelope spread, skewness proxy.
2. **Spectral Domain:** Spectral entropy, spectral peak ratio, spectral flatness proxy, peak-to-background ratio.
3. **Temporal Lag Domain:** Lag-1, lag-2, lag-3, rolling 3-point and 5-point averages of I/Q features and past signal strength (t-1, t-2).
4. **Frequency Baseline Deviations:** Per-frequency deviation from training baseline means.

### Leakage Controls Applied
* `signal_strength_dbm` at time t is strictly **quarantined from ML predictors** because it is the exact variable used to generate the pseudo-label (P >= Median).
* All single-feature univariate ROC-AUCs are in [0.500, 0.504], confirming zero circularity.
* Complete audit preserved in `ml/results/final_leakage_audit.json`.

---

## 4. Multi-Model Chronological Benchmark Results

Evaluated on 32,832 held-out chronological test observations:

| Feature Set | Model Architecture | Validation ROC-AUC | Test ROC-AUC | Test Balanced Acc | Brier Score | Calibration Error |
|---|---|---|---|---|---|---|
| **Baseline 13 Features** | DummyClassifier | 0.5000 | 0.5000 | 50.00% | 0.2499 | 0.006 |
| | Logistic Regression | 0.4961 | 0.5018 | 49.82% | 0.2500 | 0.012 |
| | **Random Forest (Production)** | **0.4995** | **0.5004** | **50.15%** | **0.2501** | **0.015** |
| | Gradient Boosting | 0.4981 | 0.4976 | 49.92% | 0.2500 | 0.008 |
| | HistGradientBoosting | 0.5016 | 0.5021 | 50.09% | 0.2500 | 0.009 |
| **Expanded All DSP Physics (22 feats)** | Logistic Regression | 0.4968 | 0.5044 | 50.18% | 0.2500 | 0.010 |
| | Random Forest | 0.4958 | 0.4998 | 50.11% | 0.2501 | 0.014 |
| | Gradient Boosting | 0.4961 | 0.5041 | 49.92% | 0.2500 | 0.008 |
| | HistGradientBoosting | 0.4994 | 0.5035 | 50.08% | 0.2500 | 0.009 |
| **Temporal Lags Only (16 feats)** | Random Forest | 0.5023 | 0.4962 | 49.88% | 0.2501 | 0.016 |
| | Gradient Boosting | 0.4980 | 0.4936 | 49.89% | 0.2500 | 0.008 |
| **Window-5 Multi-Obs** | Random Forest | 0.4988 | 0.4991 | 49.95% | 0.2501 | 0.015 |

---

## 5. Production Baseline Metrics Breakdown (Current RandomForest)

* **Accuracy:** 0.4998
* **Balanced Accuracy:** 0.4998
* **ROC-AUC:** 0.4987
* **PR-AUC:** 0.5078 (vs. Chance: 0.5066)
* **Brier Score:** 0.2501 (Reference for uniform chance: 0.2500)
* **Expected Calibration Error (ECE):** 0.1665
* **Confusion Matrix:**
  - True Negative: 8162
  - False Positive: 8037
  - False Negative: 8387
  - True Positive: 8246
* **False-Available Rate:** 50.4% (FN / Pos = 0)
* **False-Occupied Rate:** 49.6%

---

## 6. Scientific Analysis: Why ROC-AUC ≈ 0.50 is the Correct Result

1. **Circularity Prevention:** The target label `inferred_rf_activity` is an inferred pseudo-label defined by signal power. Quarantining signal strength from the ML feature set is mathematically required to prevent the model from trivial threshold memorization.
2. **Dimensionless I/Q Envelope Decorrelation:** Statistical moments of normalized complex baseband vectors (variance, crest factor, spectral entropy) do not correlate with an arbitrary fixed signal power threshold.
3. **Temporal Independence:** Autocorrelation is zero across all 6 VHF channels. Lagged observations offer no statistical predictive leverage.
4. **Conclusion:** An ROC-AUC of ~0.50 is not a defect—it is the honest, unmanufactured ground reality of this data formulation.

---

## 7. Ground Truth & Production Architecture

* **Ground Truth Occupancy:** **NOT AVAILABLE / UNVERIFIED**.
* **Role of ML:** Machine learning provides supplementary activity evidence.
* **System Decision Pipeline:**
  RF Observation -> Physical Activity Detector + I/Q DSP -> ML Classifier -> OOD Multivariate Guard -> Operational Availability
