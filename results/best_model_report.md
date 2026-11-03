# WIRE WATCHER - ML SYSTEM OPTIMIZATION & PERFORMANCE AUDIT REPORT

> **Project**: Wire Watcher RF Spectrum Monitoring & Availability Decision System  
> **Evaluation Date**: 2026-09-29 09:55:34 UTC  
> **Dataset**: `ml/data/processed/rf_signal_activity.csv` (164,160 total observations)  

---

## 📌 Executive Summary

This report documents the systematic optimization, validation, and final evaluation of the machine learning classifier inside the Wire Watcher RF Spectrum Monitoring framework.

To preserve strict scientific integrity:
1. **Target Leakage Quarantine**: Signal strength (`signal_strength_dbm`) is **strictly excluded** from ML predictor features because it was used to create the inferred activity pseudo-label.
2. **Strict Split Isolation**: The dataset is split chronologically into **60% Train (98,496)**, **20% Validation (32,832)**, and **20% Test (32,832)**. The held-out Test set was evaluated **exactly once** after all model and threshold selections were finalized on Validation.
3. **Zero Data Fabrication**: All reported numbers represent empirical measurements from the project pipeline.

---

## 1. Current Production Model Baseline

- **Model**: `RandomForestClassifier` (n_estimators=60, max_depth=10, min_samples_leaf=12)
- **Operating Threshold**: t = 0.05
- **Held-Out Test Performance**:
  - **ROC-AUC**: `0.4988`
  - **Accuracy**: `50.66%`
  - **Precision**: `0.5066`
  - **Recall**: `1.0000`
  - **F1-Score**: `0.6725`
  - **Brier Score**: `0.2502`
  - **Confusion Matrix**: TN=0, FP=16,199, FN=0, TP=16,633

---

## 2. Models & Hyperparameter Spaces Evaluated

The following architectures were evaluated on the training split and benchmarked on the validation split:

1. **`DummyClassifier`**: Prior baseline
2. **`LogisticRegression`**: L2-regularized linear model with balanced class weights
3. **`RandomForest` (Default)**: n_estimators=60, max_depth=10
4. **`RandomForest` (Deep Tuned)**: n_estimators=200, max_depth=20, min_samples_leaf=2
5. **`RandomForest` (Unconstrained)**: n_estimators=100, max_depth=None, min_samples_leaf=1
6. **`ExtraTrees`**: Extremely randomized trees (n_estimators=100, max_depth=15)
7. **`GradientBoosting`**: n_estimators=50, max_depth=3, learning_rate=0.05
8. **`HistGradientBoosting`**: Binned gradient boosting (max_iter=100)

---

## 3. Validation Leaderboard

| Rank | Model Architecture | Validation ROC-AUC | Validation PR-AUC | Validation Brier Score |
|------|-------------------|-------------------|------------------|-----------------------|
| 1 | `RandomForest_Unconstrained` | `0.5048` | `0.5057` | `0.2527` |
| 2 | `RandomForest_Tuned_Deep` | `0.5046` | `0.5048` | `0.2504` |
| 3 | `RandomForest_Tuned_Depth15` | `0.5024` | `0.5054` | `0.2503` |
| 4 | `ExtraTrees_Tuned` | `0.5017` | `0.5029` | `0.2502` |
| 5 | `DummyClassifier` | `0.5000` | `0.5021` | `0.2500` |
| 6 | `HistGradientBoosting` | `0.4990` | `0.5025` | `0.2503` |
| 7 | `GradientBoosting` | `0.4990` | `0.5025` | `0.2501` |
| 8 | `RandomForest_Default` | `0.4990` | `0.5017` | `0.2502` |
| 9 | `LogisticRegression` | `0.4961` | `0.5015` | `0.2501` |

---

## 4. Best Validated Model Selection

- **Selected Model**: `RandomForest_Unconstrained`
- **Selection Rationale**: Highest ROC-AUC (`0.5048`) and PR-AUC (`0.5057`) on the validation split.

---

## 5. Operating Threshold Optimization

Decision thresholds from t = 0.01 to t = 0.90 were evaluated on the validation set:

| Threshold (t) | Precision | Recall | F1-Score | Specificity | Accuracy | FP Count |
|---------------|-----------|--------|----------|-------------|----------|----------|
| `0.01` | `0.5021` | `1.0000` | `0.6685` | `0.0000` | `0.5021` | `16,347.0` |
| `0.02` | `0.5021` | `1.0000` | `0.6685` | `0.0000` | `0.5021` | `16,347.0` |
| `0.03` | `0.5021` | `1.0000` | `0.6685` | `0.0000` | `0.5021` | `16,347.0` |
| `0.05` | `0.5021` | `1.0000` | `0.6685` | `0.0000` | `0.5021` | `16,347.0` |
| `0.10` | `0.5021` | `1.0000` | `0.6685` | `0.0000` | `0.5021` | `16,347.0` |
| `0.15` | `0.5021` | `1.0000` | `0.6685` | `0.0000` | `0.5021` | `16,347.0` |
| `0.20` | `0.5021` | `1.0000` | `0.6685` | `0.0000` | `0.5021` | `16,347.0` |
| `0.25` | `0.5021` | `0.9999` | `0.6685` | `0.0000` | `0.5020` | `16,347.0` |
| `0.30` | `0.5021` | `0.9994` | `0.6684` | `0.0004` | `0.5020` | `16,340.0` |
| `0.40` | `0.5020` | `0.9626` | `0.6599` | `0.0371` | `0.5018` | `15,741.0` |
| `0.50` | `0.5060` | `0.5614` | `0.5322` | `0.4472` | `0.5045` | `9,036.0` |
| `0.60` | `0.5072` | `0.0535` | `0.0968` | `0.9476` | `0.4987` | `857.0` |
| `0.70` | `0.5952` | `0.0015` | `0.0030` | `0.9990` | `0.4981` | `17.0` |
| `0.80` | `0.0000` | `0.0000` | `0.0000` | `0.9999` | `0.4979` | `1.0` |
| `0.90` | `0.0000` | `0.0000` | `0.0000` | `1.0000` | `0.4979` | `0.0` |

- **F1-Maximizing Threshold**: t = 0.01 (Validation F1 = `0.6685`)
- **Spectrum Safety Threshold** (Recall >= 0.99): t = 0.01

---

## 6. Final Test Set Evaluation (Held-Out Test Set, n = 32,832)

The selected model (`RandomForest_Unconstrained`) operating at threshold t = 0.01 was evaluated on the untouched test set:

- **Test ROC-AUC**: `0.4982`
- **Test PR-AUC**: `0.5081`
- **Test Accuracy**: `0.5066` (`50.66%`) 
- **Balanced Accuracy**: `0.5000`
- **Test Precision**: `0.5066`
- **Test Recall**: `1.0000`
- **Test Specificity**: `0.0000`
- **Test F1-Score**: `0.6725`
- **Test Brier Score**: `0.2530`

### Confusion Matrix (Test Set, n = 32,832)
- **True Negatives (TN)**: `0`
- **False Positives (FP)**: `16,199`
- **False Negatives (FN)**: `0`
- **True Positives (TP)**: `16,633`
- **Positive Class Count**: `16,633`
- **Negative Class Count**: `16,199`

---

## 7. Feature Importance & Sensitivity

Top predictors by validation permutation importance (ROC-AUC degradation when shuffled):

- **`iq_spectral_entropy`**: Permutation Importance = `0.00454` (+/- `0.00203`)
- **`iq_magnitude_variance`**: Permutation Importance = `0.00422` (+/- `0.00120`)
- **`frequency_mhz`**: Permutation Importance = `0.00295` (+/- `0.00203`)
- **`iq_spectral_peak_ratio`**: Permutation Importance = `0.00283` (+/- `0.00211`)
- **`iq_phase_concentration`**: Permutation Importance = `0.00208` (+/- `0.00129`)
- **`bandwidth_khz`**: Permutation Importance = `0.00207` (+/- `0.00195`)
- **`iq_crest_factor`**: Permutation Importance = `0.00195` (+/- `0.00196`)
- **`iq_peak_magnitude`**: Permutation Importance = `0.00189` (+/- `0.00136`)

---

## 8. System Limitations & Scientific Disclosure

1. **Pseudo-Label Limitation**: The target label (`inferred_rf_activity`) is defined by a per-frequency signal strength median threshold. It is an **inferred pseudo-label**, not regulatory ground-truth spectrum vacancy.
2. **Leakage Protection Requirement**: Signal strength (`signal_strength_dbm`) must be quarantined from ML predictors. Without signal strength, baseband statistical features (I/Q RMS, peak magnitude, entropy) demonstrate near-zero linear/tree correlation with the signal-power-derived pseudo-label, resulting in honest test performance near chance level (ROC-AUC ≈ 0.50).
3. **Operational Gating Necessity**: Because ML prediction serves as only **one supplementary input**, physical DSP energy detection, thermal noise calculations (kTB), and multivariate OOD bounds provide essential primary safety gating.

---

## 9. RECOMMENDED MODEL FOR PROJECT PRESENTATION

```text
================================================================================
RECOMMENDED MODEL FOR PROJECT PRESENTATION
================================================================================
Model Architecture:  RandomForest_Unconstrained
Hyperparameters:     n_estimators=100, max_depth=None, min_samples_leaf=1
Operating Threshold: 0.01
Test ROC-AUC:        0.4982
Test PR-AUC:         0.5081
Test Accuracy:       0.5066 (50.66%)
Test Precision:      0.5066
Test Recall:         1.0000
Test F1-Score:       0.6725
Test Specificity:    0.0000
================================================================================
```

*These are measured results from the project pipeline.*