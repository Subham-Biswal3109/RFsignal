"""ML inference unit tests.

Tests cover:
  - Model and metadata loading
  - Feature contract (feature names, order)
  - Full predict() call returns valid structure
  - OOD check flags out-of-distribution inputs
  - Detector activity uses signal_strength_dbm correctly
  - Missing I/Q fields handled (pipeline imputes NaN)
  - Probability is always in [0, 1]
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from inference.predict import predict, ood_check, detector_activity, load_model


KNOWN_FREQS = [70.0, 90.0, 100.0, 120.0, 140.0, 160.0]


def _base_input(bundle: dict, freq: float = 120.0, signal: float = -60.0) -> dict:
    """Build a minimal valid input dict for the given bundle."""
    x: dict = {c: float("nan") for c in bundle["feature_columns"]}
    x.update({
        "frequency_mhz":    freq,
        "bandwidth_khz":    50.0,
        "iq_available":     0,
        "signal_strength_dbm": signal,
    })
    return x


# ── model loading ─────────────────────────────────────────────────────────────

def test_model_loads(bundle):
    """Model bundle is a dict with expected keys."""
    assert bundle is not None
    assert "pipeline" in bundle
    assert "feature_columns" in bundle
    assert "threshold_occupied" in bundle
    assert "ood_bounds" in bundle


def test_feature_columns_not_empty(bundle):
    """Feature column list is non-empty."""
    assert len(bundle["feature_columns"]) > 0


def test_signal_strength_not_in_features(bundle):
    """signal_strength_dbm must NOT be an ML feature (leakage guard)."""
    assert "signal_strength_dbm" not in bundle["feature_columns"], (
        "signal_strength_dbm must be excluded from ML features because it "
        "defines the inferred_rf_activity pseudo-label."
    )


def test_expected_features_present(bundle):
    """All expected feature columns are present."""
    expected = {
        "frequency_mhz", "bandwidth_khz", "iq_available",
        "iq_rms_magnitude", "iq_magnitude_variance", "iq_peak_magnitude",
        "iq_crest_factor", "iq_p10", "iq_p50", "iq_p90",
        "iq_phase_concentration", "iq_spectral_entropy", "iq_spectral_peak_ratio",
    }
    actual = set(bundle["feature_columns"])
    assert expected == actual, f"Feature mismatch. Extra: {actual-expected}, Missing: {expected-actual}"


# ── predict() outputs ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("freq", KNOWN_FREQS)
def test_predict_returns_valid_structure(bundle, freq):
    """predict() returns all required keys with valid types."""
    x = _base_input(bundle, freq=freq)
    result = predict(x, bundle)

    assert "availability"           in result
    assert "activity"               in result
    assert "ml_activity_probability" in result
    assert "ml_activity"            in result
    assert "detector_activity"      in result
    assert "ood"                    in result
    assert "ood_reasons"            in result
    assert "threshold"              in result

    assert result["availability"] in {"AVAILABLE", "OCCUPIED", "UNCERTAIN"}
    assert result["activity"]     in {"DETECTED",  "NOT_DETECTED",  "UNCERTAIN"}
    assert isinstance(result["ood"], bool)
    assert isinstance(result["ood_reasons"], list)


@pytest.mark.parametrize("freq", KNOWN_FREQS)
def test_probability_in_range(bundle, freq):
    """ML probability is always in [0, 1] and not NaN/Inf."""
    x = _base_input(bundle, freq=freq)
    p = predict(x, bundle)["ml_activity_probability"]
    assert 0.0 <= p <= 1.0, f"Probability out of range: {p}"
    assert not math.isnan(p) and not math.isinf(p)


def test_iq_missing_handled(bundle):
    """Missing I/Q (iq_available=0, all NaN) does not crash the pipeline."""
    x = _base_input(bundle)
    # Explicitly set all I/Q features to NaN (as they'd arrive without I/Q data).
    for col in bundle["feature_columns"]:
        if col.startswith("iq_") and col != "iq_available":
            x[col] = float("nan")
    result = predict(x, bundle)
    assert result["availability"] in {"AVAILABLE", "OCCUPIED", "UNCERTAIN"}


def test_iq_present_handled(bundle):
    """Providing I/Q features does not crash the pipeline."""
    x = _base_input(bundle)
    x.update({
        "iq_available": 1,
        "iq_rms_magnitude": 0.35,
        "iq_magnitude_variance": 0.02,
        "iq_peak_magnitude": 0.72,
        "iq_crest_factor": 2.06,
        "iq_p10": 0.12,
        "iq_p50": 0.31,
        "iq_p90": 0.58,
        "iq_phase_concentration": 0.45,
        "iq_spectral_entropy": 0.82,
        "iq_spectral_peak_ratio": 0.04,
    })
    result = predict(x, bundle)
    assert result["availability"] in {"AVAILABLE", "OCCUPIED", "UNCERTAIN"}


# ── OOD check ─────────────────────────────────────────────────────────────────

def test_ood_unknown_frequency(bundle):
    """An unknown frequency (no trained threshold) must trigger OOD/UNCERTAIN."""
    x = _base_input(bundle, freq=5000.0, signal=-20.0)
    result = predict(x, bundle)
    # The detector has no threshold for 5000 MHz → OOD must be True.
    assert result["ood"] is True
    assert result["availability"] == "UNCERTAIN"


def test_ood_extreme_values(bundle):
    """Extreme feature values should trigger OOD."""
    x = _base_input(bundle)
    x["frequency_mhz"] = 1e9   # far outside training range
    is_ood, reasons = ood_check(x, bundle)
    assert is_ood is True
    assert len(reasons) > 0


def test_ood_normal_input_not_always_ood(bundle):
    """A well-formed in-distribution input should NOT trigger per-feature OOD bounds."""
    x = _base_input(bundle)
    is_ood, reasons = ood_check(x, bundle)
    # The multivariate distance may still flag it; but per-feature bounds should be clean.
    feature_bound_reasons = [r for r in reasons if "outside training 1st-99th percentile" in r]
    assert len(feature_bound_reasons) == 0, \
        f"In-distribution input triggered per-feature OOD: {feature_bound_reasons}"


# ── RF detector ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("freq", KNOWN_FREQS)
def test_detector_activity_known_frequency(bundle, freq):
    """Detector returns a bool (not None) for all trained frequencies."""
    active, reason = detector_activity(-60.0, freq, bundle)
    assert active is not None, f"Detector returned None for freq={freq}: {reason}"
    assert isinstance(active, bool)


def test_detector_activity_unknown_frequency(bundle):
    """Detector returns None for an unknown frequency (never invents a threshold)."""
    active, reason = detector_activity(-60.0, 999.0, bundle)
    assert active is None
    assert reason is not None and len(reason) > 0
