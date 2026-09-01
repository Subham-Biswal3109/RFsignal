# RF Signal Data Final Audit

## Verified source
- Dataset: RF Signal Data
- Source: `logged_data.csv`
- Provenance: **PARTIALLY VERIFIED** — published source evidence reports SDR/DragonOS acquisition; the exact CSV does not contain a complete calibration/acquisition record.

## Actual file
- Rows: 164,160
- Columns: 27
- Duplicate rows: 0
- Timestamp range: 2023-05-05 00:00:00 to 2023-06-11 23:59:40
- Timestamp cadence: source is ordered at 20-second intervals
- Frequencies (MHz): [70.0, 90.0, 100.0, 120.0, 140.0, 160.0]
- Bandwidths (kHz): [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
- Signal strength: -100.0 to 0.0 dBm as reported
- I/Q rows: 109,324
- I/Q missing: 54,836
- I/Q sample count: 100 complex samples in valid rows inspected

## Scientific status
There is no native occupied/unoccupied ground-truth label. The pipeline uses `inferred_rf_activity`, an **INFERRED/PSEUDO-LABEL** generated from a training-only per-frequency signal-strength threshold. It is not occupancy ground truth.

`signal_strength_dbm` is excluded from the ML predictor set because it is the variable used to generate the activity pseudo-label. This prevents the classifier from simply reproducing the label-generation threshold.

I/Q-derived features are calculated only when I/Q is present. No I/Q samples, sampling rate, receiver gain, calibration, or absolute FFT frequency axis is fabricated. FFT-derived values are normalized spectral summaries only.

## Limitation
This is RF activity detection, not proof of regulatory spectrum vacancy. Live SDR/API ingestion is not implemented.

## Dataset classifications
- REAL_MEASURED: source RF frequency, reported signal strength, bandwidth, timestamp, and recorded I/Q when present.
- DERIVED: all `iq_*` features and unit-normalized ML fields.
- INFERRED/PSEUDO-LABEL: `inferred_rf_activity`.
- APPLICATION_METADATA: modulation, interference, device, location, weather and system context.

## No-claim statement
The Kaggle dataset is not treated as containing occupancy ground truth. The application does not claim live RF monitoring.

## Leakage audit summary
- Signal-strength-only AUC: 1.000, intentionally rejected as circular because signal strength defines `inferred_rf_activity`.
- Primary ML feature set excludes signal strength.
- Modulation-only ablation AUC: approximately 0.505; excluded from the primary model as an annotation shortcut.
- Primary no-signal-strength ML AUC: approximately 0.502 on the chronological holdout.
