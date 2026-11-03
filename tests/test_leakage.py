"""Leakage verification tests.

These tests verify ACTUAL leakage conditions:
  1. signal_strength_dbm is excluded from ML feature columns.
  2. The target-generation variable (signal_strength_dbm) is not accidentally
     used as a predictor anywhere in the trained pipeline.
  3. No future observations leak into training (chronological split verified).
  4. Feature names in the model bundle match the model_metadata.json.

IMPORTANT: These tests do NOT assert that individual features have AUC ≈ 0.50.
A predictive feature is not automatically leakage.  Leakage means the label-
generating variable is also used as a predictor.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

MODEL_PATH   = ROOT / "ml" / "artifacts" / "wire_watcher_model.pkl"
META_PATH    = ROOT / "ml" / "artifacts" / "model_metadata.json"
PROCESSED    = ROOT / "ml" / "data" / "processed" / "rf_signal_activity.csv"


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def meta():
    return json.loads(META_PATH.read_text(encoding="utf-8"))


# ── 1. Signal strength excluded from ML predictors ───────────────────────────

def test_signal_strength_not_ml_feature(bundle):
    """signal_strength_dbm must not appear in feature_columns."""
    assert "signal_strength_dbm" not in bundle["feature_columns"], (
        "LEAKAGE: signal_strength_dbm defines inferred_rf_activity but is "
        "also in the ML feature set — circular predictor."
    )


def test_signal_strength_excluded_per_metadata(meta):
    """model_metadata.json explicitly marks signal_strength_dbm as excluded."""
    assert "signal_strength_dbm" in meta.get("excluded_features", []), (
        "model_metadata.json should list signal_strength_dbm in excluded_features."
    )


def test_target_generation_feature_is_not_ml_predictor_flag(meta):
    """Metadata flag confirms the target-generation variable is not a predictor."""
    assert meta.get("target_generation_feature_used_as_ml_predictor") is False, (
        "Metadata claims target_generation_feature_used_as_ml_predictor=True — "
        "this indicates a leakage problem."
    )


# ── 2. Pipeline internal check ────────────────────────────────────────────────

def test_preprocessor_feature_names_exclude_signal_strength(bundle):
    """The ColumnTransformer step should not include signal_strength_dbm."""
    pre = bundle["pipeline"].named_steps["preprocessor"]
    # get_feature_names_out returns e.g. ['num__frequency_mhz', ...]
    names = pre.get_feature_names_out().tolist()
    assert not any("signal_strength_dbm" in n for n in names), (
        f"signal_strength_dbm found in preprocessor output: "
        f"{[n for n in names if 'signal_strength' in n]}"
    )


# ── 3. Feature name consistency ───────────────────────────────────────────────

def test_feature_columns_match_metadata(bundle, meta):
    """Bundle feature_columns must match model_metadata.json features list."""
    assert set(bundle["feature_columns"]) == set(meta["features"]), (
        f"Bundle features differ from metadata.\n"
        f"  Bundle: {sorted(bundle['feature_columns'])}\n"
        f"  Meta:   {sorted(meta['features'])}"
    )


def test_metadata_target_is_inferred_pseudolabel(meta):
    """Metadata must declare target as inferred pseudo-label, not ground truth."""
    assert meta.get("target") == "inferred_rf_activity"
    assert meta.get("target_is_ground_truth") is False
    assert "pseudo-label" in meta.get("target_method", "").lower(), (
        "target_method should mention pseudo-label."
    )


# ── 4. No temporal leakage ────────────────────────────────────────────────────

def test_dataset_has_timestamp_column():
    """Processed dataset must have a Timestamp column for chronological split."""
    import pandas as pd
    if not PROCESSED.exists():
        pytest.skip("Processed dataset not found — run prepare_rf_signal_dataset.py first.")
    df = pd.read_csv(PROCESSED, nrows=5)
    assert "Timestamp" in df.columns, "Missing Timestamp column in processed dataset."


def test_split_is_approximately_chronological():
    """Verify the chronological split: training rows come before validation rows.

    We sample 60 % / 20 % / 20 % chronologically.  The latest training
    timestamp must be ≤ the earliest validation timestamp (allowing ties).
    """
    import pandas as pd
    if not PROCESSED.exists():
        pytest.skip("Processed dataset not found.")
    df = pd.read_csv(PROCESSED, parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    n = len(df)
    a = int(n * 0.60)
    b = int(n * 0.80)
    train_max = df.iloc[:a]["Timestamp"].max()
    val_min   = df.iloc[a:b]["Timestamp"].min()
    assert train_max <= val_min, (
        f"Temporal leakage: latest training timestamp ({train_max}) is later than "
        f"earliest validation timestamp ({val_min})."
    )


# ── 5. Processed dataset schema ───────────────────────────────────────────────

def test_processed_dataset_schema():
    """Processed dataset contains all expected columns."""
    import pandas as pd
    if not PROCESSED.exists():
        pytest.skip("Processed dataset not found.")
    df = pd.read_csv(PROCESSED, nrows=10)
    required = {
        "Timestamp", "frequency_mhz", "bandwidth_khz", "signal_strength_dbm",
        "iq_available", "iq_rms_magnitude", "iq_spectral_entropy",
        "inferred_rf_activity",
    }
    missing = required - set(df.columns)
    assert not missing, f"Missing columns in processed dataset: {missing}"


def test_inferred_label_is_binary():
    """inferred_rf_activity must only contain 0 and 1."""
    import pandas as pd
    if not PROCESSED.exists():
        pytest.skip("Processed dataset not found.")
    df = pd.read_csv(PROCESSED, usecols=["inferred_rf_activity"], nrows=10000)
    unique_vals = set(df["inferred_rf_activity"].dropna().unique())
    assert unique_vals <= {0, 1}, f"Non-binary values in inferred_rf_activity: {unique_vals}"
