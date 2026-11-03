# Hybrid RF Dataset Methodology

## Data categories

### REAL_MEASURED
Values copied from the supplied `logged_data.csv`: `Timestamp`, `Frequency`, `Signal Strength`, `Bandwidth`, and I/Q samples where present. The project preserves the source-reported dBm unit and does not invent calibration.

### DERIVED
I/Q summaries are calculated deterministically: RMS magnitude, magnitude variance, peak magnitude, crest factor, P10/P50/P90 magnitude, phase concentration, normalized spectral entropy, and normalized spectral peak ratio. Because sampling-rate and receiver calibration metadata are incomplete, spectral features are relative/normalized only; no absolute FFT frequency axis is claimed.

### INFERRED/PSEUDO-LABEL
`inferred_rf_activity` is generated from a training-only per-frequency median of the recorded `signal_strength_dbm`. Values at or above that reference are `DETECTABLE_RF_ACTIVITY`; values below are `NO_DETECTABLE_RF_ACTIVITY`. This is an operational activity proxy, **not ground-truth occupancy**.

The signal-strength field is excluded from the ML predictor set because it is the target-generation variable. This prevents a circular classifier that simply learns the detector threshold.

### APPLICATION_METADATA
Modulation, interference, device, location, weather, and system fields are retained as source context but excluded from the primary activity model. They are not treated as occupancy ground truth.

## Runtime architecture

```text
RF observation
  -> RF feature extraction
  -> activity detector (measured signal-strength evidence)
  -> ML activity confidence
  -> OOD check
  -> AVAILABLE / OCCUPIED / UNCERTAIN
```

The detector is the operational activity decision because the dataset does not contain independent occupancy labels. The ML classifier is a secondary confidence/feature model and is not represented as proof of channel vacancy.

## Leakage controls
- No future observation is used.
- `signal_strength_dbm` generates the pseudo-label but is not an ML predictor.
- No target-derived state fields are used.
- Device/interference/location metadata are excluded from the primary model.
- Training/validation/test data are chronological.

## Limitations
- No verified occupied/unoccupied ground truth exists in the dataset.
- The pseudo-label is intentionally an RF activity proxy.
- The primary ML model may remain near chance because its predictors are deliberately independent of the label-generation measurement. That is preferable to circular high accuracy.
- Live SDR/API ingestion is not implemented.
