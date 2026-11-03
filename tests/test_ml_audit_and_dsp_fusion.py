"""Wire Watcher — Automated Unit Tests for ML Audit & RF/DSP Evidence Fusion.

Tests:
  - Phase 5: Deterministic RF/DSP Activity Detector (Thermal noise floor, SNR >= 6dB threshold)
  - Phase 6: Temporal persistence tracking
  - Phase 7: Evidence Fusion layer (DSP primary + ML supporting + OOD safety override)
  - Phase 10: Ground-truth honesty (INFERRED_RF_ACTIVITY, UNVERIFIED)
  - Phase 11/12: API endpoints /api/ml/audit and /api/ml/features
"""

from __future__ import annotations
import math
import pytest
from pathlib import Path
from flask.testing import FlaskClient

from app import create_app
from backend.rf.dsp_activity_detector import (
    DSPActivityDetector,
    DEFAULT_SNR_THRESHOLD_DB,
    DEFAULT_NOISE_FIGURE_DB,
)
from backend.services.prediction import get_model_bundle, run_ml_inference


@pytest.fixture
def client() -> FlaskClient:
    app_instance = create_app()
    app_instance.config["TESTING"] = True
    with app_instance.test_client() as c:
        yield c


def test_thermal_noise_floor_calculation():
    """Verify physical thermal noise formula P_n = k * T * B + NF."""
    detector = DSPActivityDetector()
    # 200 kHz bandwidth with 6 dB Noise Figure
    noise_200khz = detector.calculate_thermal_noise_dbm(200.0, noise_figure_db=6.0)
    assert -120.0 <= noise_200khz <= -110.0

    # 1 MHz bandwidth should be 10*log10(5) ≈ 7 dB higher than 200 kHz
    noise_1mhz = detector.calculate_thermal_noise_dbm(1000.0, noise_figure_db=6.0)
    assert round(noise_1mhz - noise_200khz, 1) == 7.0


def test_dsp_activity_detector_evaluation():
    """Verify deterministic SNR activity thresholding (SNR >= 6 dB)."""
    detector = DSPActivityDetector(snr_threshold_db=6.0)

    # Strong signal (-60 dBm, noise floor ≈ -115 dBm -> SNR ≈ 55 dB)
    eval_strong = detector.evaluate_observation(
        frequency_mhz=70.0,
        bandwidth_khz=200.0,
        signal_strength_dbm=-60.0,
    )
    assert eval_strong["snr_db"] > 6.0
    assert eval_strong["instant_active"] is True

    # Weak signal (-130 dBm below noise floor -> negative SNR)
    eval_weak = detector.evaluate_observation(
        frequency_mhz=70.0,
        bandwidth_khz=200.0,
        signal_strength_dbm=-130.0,
    )
    assert eval_weak["snr_db"] < 6.0
    assert eval_weak["activity_state"] == "INACTIVE"


def test_temporal_persistence_tracking():
    """Verify persistence requirement across consecutive samples."""
    detector = DSPActivityDetector(min_persistence_count=2)

    # First observation (instant active, but 1 sample -> UNCERTAIN persistence)
    res1 = detector.evaluate_observation(
        frequency_mhz=90.0, bandwidth_khz=200.0, signal_strength_dbm=-50.0
    )
    assert res1["temporal_persistence"]["available"] is False

    # Second consecutive observation (2 samples -> ACTIVE)
    res2 = detector.evaluate_observation(
        frequency_mhz=90.0, bandwidth_khz=200.0, signal_strength_dbm=-50.0
    )
    assert res2["temporal_persistence"]["available"] is True
    assert res2["activity_state"] == "ACTIVE"


def test_evidence_fusion_and_ground_truth_honesty():
    """Verify evidence fusion output structure and label tagging."""
    sample_input = {
        "frequency_mhz": 70.0,
        "bandwidth_khz": 200.0,
        "signal_strength_dbm": -60.0,
        "iq_available": 1,
        "iq_rms_magnitude": 0.5,
        "iq_magnitude_variance": 0.05,
        "iq_peak_magnitude": 1.2,
        "iq_crest_factor": 2.4,
        "iq_p10": 0.1,
        "iq_p50": 0.5,
        "iq_p90": 0.9,
        "iq_phase_concentration": 0.8,
        "iq_spectral_entropy": 0.4,
        "iq_spectral_peak_ratio": 0.3,
    }

    result = run_ml_inference(sample_input)
    assert "dsp_evidence" in result
    assert result["label_type"] == "INFERRED_RF_ACTIVITY"
    assert result["ground_truth"] == "UNVERIFIED"
    assert "evidence_fusion_summary" in result


def test_api_ml_audit_endpoint(client: FlaskClient):
    """GET /api/ml/audit returns full ML audit results."""
    res = client.get("/api/ml/audit")
    assert res.status_code == 200
    data = res.get_json()
    assert "phase_1_ml_audit" in data
    assert "phase_2_leakage_audit" in data
    assert "phase_3_validation_redesign" in data
    assert "phase_4_baselines" in data
    assert "phase_12_feature_importance" in data
    assert "phase_13_ood_validation" in data


def test_api_ml_features_endpoint(client: FlaskClient):
    """GET /api/ml/features returns feature importance labeled MODEL FEATURE IMPORTANCE."""
    res = client.get("/api/ml/features")
    assert res.status_code == 200
    data = res.get_json()
    assert data.get("title") == "MODEL FEATURE IMPORTANCE"
    assert "gini_importance" in data
