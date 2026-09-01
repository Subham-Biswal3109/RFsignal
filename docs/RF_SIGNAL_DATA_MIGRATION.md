# RF Signal Data Migration

## Status
Migrated from the prior synthetic temporal model to a hybrid RF-activity architecture.

## Dataset
`logged_data.csv` from the supplied RF Signal Data dataset. The actual file contains 164,160 observations and 27 source columns.

## Target
The source has no native occupancy label. The active ML target is `inferred_rf_activity`, an inferred/pseudo-label generated using a training-only per-frequency median of recorded signal strength. It is not ground truth occupancy.

## ML schema
The primary classifier excludes the signal-strength predictor because signal strength generates the pseudo-label. It uses frequency, bandwidth, I/Q availability, and deterministic I/Q-derived statistics.

## Evaluation
A chronological 60/20/20 split is used. The current model results are recorded in `ml/results/rf_signal_model_comparison.json`. The model is not selected or advertised on accuracy alone.

## Operational decision
The runtime availability layer uses measured signal-strength activity evidence plus OOD safeguards. `AVAILABLE` means no detectable activity according to the project's operational detector; it is not a guarantee of regulatory or physical vacancy. Ambiguous/OOD observations return `UNCERTAIN`.

## Legacy data
The original synthetic dataset and model remain under `legacy/synthetic/` for historical compatibility.

## Live monitoring
Live SDR/API ingestion is **NOT IMPLEMENTED**. The RF simulator, FFT, PSD, noise estimation, peak detection, and visualization remain available for demonstration/testing.

## Reproduction
```bash
PYTHONPATH=ml python ml/data/prepare_rf_signal_dataset.py
PYTHONPATH=ml python ml/training/train_model.py
python -m pytest tests/ml/test_rf_signal_migration.py
```

## Leakage/ablation evidence
The signal-strength-only classifier has ROC-AUC 1.000 because signal strength is the exact pseudo-label source; this is circular and is not a valid ML result. A modulation-only ablation produced ROC-AUC about 0.505, while the primary no-signal-strength ML feature set produced ROC-AUC about 0.502. These results support keeping the measured signal-strength detector separate from the ML confidence model.

Results are stored in `ml/results/rf_signal_leakage_audit.json` and `ml/results/rf_signal_ablation_results.json`.

## Runtime semantics
`signal_strength_dbm` is a REAL_MEASURED detector input and is not passed to the classifier. The classifier returns secondary ML activity probability. The final availability decision is conservative: OOD → `UNCERTAIN`; otherwise detector activity → `OCCUPIED`, detector inactivity → `AVAILABLE`. This is an operational RF-activity decision, not proof of legal/regulatory vacancy.
