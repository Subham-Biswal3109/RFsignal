"""ML migration smoke tests — verifying the RF Signal Data pipeline.

These tests confirm:
  - Processed dataset has the expected schema
  - signal_strength_dbm is excluded from ML features
  - Model metadata correctly describes the pseudo-label target
  - predict() returns valid outputs for known-frequency inputs
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

PROCESSED = ROOT / "ml" / "data" / "processed" / "rf_signal_activity.csv"
MODEL     = ROOT / "ml" / "artifacts" / "wire_watcher_model.pkl"
META      = ROOT / "ml" / "artifacts" / "model_metadata.json"


def test_processed_dataset_schema():
    required = {
        "Timestamp", "frequency_mhz", "bandwidth_khz",
        "signal_strength_dbm", "iq_available",
        "iq_rms_magnitude", "iq_spectral_entropy",
    }
    df = pd.read_csv(PROCESSED, nrows=10)
    assert required.issubset(df.columns), (
        f"Missing columns: {required - set(df.columns)}"
    )


def test_signal_strength_not_an_ml_feature():
    b = joblib.load(MODEL)
    assert "signal_strength_dbm" not in b["feature_columns"], (
        "signal_strength_dbm must be excluded from ML features."
    )


def test_model_metadata_is_hybrid_and_not_ground_truth():
    m = json.loads(META.read_text(encoding="utf-8"))
    assert m["target_is_ground_truth"] is False
    assert m["target"] == "inferred_rf_activity"
    assert "pseudo-label" in m["target_method"].lower()
    assert m["live_rf_ingestion"] is False


def test_model_predicts_activity_and_availability():
    b = joblib.load(MODEL)
    x = {c: None for c in b["feature_columns"]}
    x.update({
        "frequency_mhz": 120.0,
        "bandwidth_khz": 50.0,
        "iq_available": 0,
    })
    x["signal_strength_dbm"] = -60.0
    # Fill remaining NaN
    for k, v in x.items():
        if v is None:
            x[k] = float("nan")

    from inference.predict import predict
    result = predict(x, b)
    assert result["availability"] in {"AVAILABLE", "OCCUPIED", "UNCERTAIN"}
    assert result["activity"]     in {"DETECTED",  "NOT_DETECTED",  "UNCERTAIN"}
    assert 0.0 <= float(result["ml_activity_probability"]) <= 1.0
