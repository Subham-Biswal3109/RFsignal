"""API integration tests — 7 end-to-end scenarios.

Tests verify response structure and safe behaviour.
They do NOT hard-code expected probability values because the model is
approximately chance-level and outputs depend on the actual bundle.

Scenario coverage
──────────────────
1. Normal RF observation (known training-range frequency, moderate signal)
2. Strong RF activity  (high signal strength)
3. Weak RF activity    (low signal strength)
4. OOD input           (frequency far outside training distribution)
5. Invalid input       (missing required fields)
6. I/Q-present input   (all 13 I/Q features provided)
7. I/Q-missing input   (iq_available=0, all I/Q fields absent)
"""
from __future__ import annotations
import json
import math


# ── shared valid base request ────────────────────────────────────────────────
_BASE = {
    "frequency_mhz": 120.0,
    "bandwidth_khz": 50.0,
    "signal_strength_dbm": -60.0,
    "iq_available": 0,
}

_IQ_FIELDS = {
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
}

VALID_STATES = {"AVAILABLE", "OCCUPIED", "UNCERTAIN"}
VALID_ACTIVITIES = {"DETECTED", "NOT_DETECTED", "UNCERTAIN"}


def _check_structure(data: dict) -> None:
    """Assert all mandatory response fields are present and have valid types."""
    assert "availability"   in data, f"Missing 'availability': {data}"
    assert "activity"       in data, f"Missing 'activity': {data}"
    assert "probability"    in data, f"Missing 'probability': {data}"
    assert "ood_warning"    in data, f"Missing 'ood_warning': {data}"
    assert "data_source"    in data, f"Missing 'data_source': {data}"
    assert "label_type"     in data, f"Missing 'label_type': {data}"

    assert data["availability"] in VALID_STATES, \
        f"Unexpected availability: {data['availability']}"
    assert data["activity"] in VALID_ACTIVITIES, \
        f"Unexpected activity: {data['activity']}"

    prob = float(data["probability"])
    assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"
    assert not math.isnan(prob) and not math.isinf(prob), "Probability is NaN/Inf"

    assert isinstance(data["ood_warning"], bool)
    assert data["label_type"] == "INFERRED_RF_ACTIVITY"


def _check_ood_certain(data: dict) -> None:
    """When OOD, availability must be UNCERTAIN."""
    if data["ood_warning"]:
        assert data["availability"] == "UNCERTAIN", \
            f"OOD but availability={data['availability']} (should be UNCERTAIN)"


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — Normal RF observation
# ─────────────────────────────────────────────────────────────────────────────
def test_predict_normal_rf(client):
    """Normal RF input → valid structure, probability in [0,1]."""
    resp = client.post(
        "/api/predict",
        data=json.dumps(_BASE),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    _check_structure(data)
    _check_ood_certain(data)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — Strong RF activity (high signal strength)
# ─────────────────────────────────────────────────────────────────────────────
def test_predict_strong_rf(client):
    """Strong signal → valid response; we do NOT assert a specific prediction."""
    payload = {**_BASE, "signal_strength_dbm": -30.0}
    resp = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    _check_structure(data)
    _check_ood_certain(data)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 — Weak RF activity (low signal strength)
# ─────────────────────────────────────────────────────────────────────────────
def test_predict_weak_rf(client):
    """Weak signal → valid response; we do NOT assert a specific prediction."""
    payload = {**_BASE, "signal_strength_dbm": -110.0}
    resp = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    _check_structure(data)
    _check_ood_certain(data)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4 — OOD input (extreme / unknown frequency)
# ─────────────────────────────────────────────────────────────────────────────
def test_predict_ood_input(client):
    """Frequency far outside training distribution → OOD or UNCERTAIN."""
    payload = {**_BASE, "frequency_mhz": 5000.0, "signal_strength_dbm": 30.0}
    resp = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    _check_structure(data)
    # An unknown frequency has no trained detector threshold → must be UNCERTAIN.
    assert data["availability"] == "UNCERTAIN", \
        f"Expected UNCERTAIN for unknown frequency, got {data['availability']}"
    assert data["ood_warning"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5 — Invalid input
# ─────────────────────────────────────────────────────────────────────────────
def test_predict_invalid_input(client):
    """Missing required fields → 400 Bad Request."""
    payload = {"signal_strength_dbm": -60.0}   # missing frequency_mhz, bandwidth_khz
    resp = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data


def test_predict_invalid_frequency(client):
    """Non-positive frequency → 400."""
    payload = {**_BASE, "frequency_mhz": -10.0}
    resp = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_predict_empty_body(client):
    """Empty body → 400."""
    resp = client.post(
        "/api/predict",
        data="{}",
        content_type="application/json",
    )
    assert resp.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 6 — I/Q-present input
# ─────────────────────────────────────────────────────────────────────────────
def test_predict_with_iq(client):
    """I/Q features provided → valid response with ml_activity set."""
    payload = {
        **_BASE,
        "iq_available": 1,
        **_IQ_FIELDS,
    }
    resp = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    _check_structure(data)
    _check_ood_certain(data)
    assert "ml_activity" in data
    assert "ml_activity_probability" in data


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 7 — I/Q-missing input
# ─────────────────────────────────────────────────────────────────────────────
def test_predict_without_iq(client):
    """iq_available=0, no I/Q fields → valid response (pipeline imputes NaN)."""
    payload = {**_BASE, "iq_available": 0}
    resp = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    _check_structure(data)
    _check_ood_certain(data)


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary endpoints
# ─────────────────────────────────────────────────────────────────────────────
def test_health_endpoint(client):
    """/api/health returns status field."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "status"       in data
    assert "model_loaded" in data
    assert data["model_loaded"] is True


def test_model_info_endpoint(client):
    """/api/model-info returns expected fields."""
    resp = client.get("/api/model-info")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("target") == "inferred_rf_activity"
    assert data.get("ground_truth_occupancy") is False
    assert data.get("label_type") == "INFERRED/PSEUDO-LABEL"
    assert "features" in data
    assert isinstance(data["features"], list)
    assert len(data["features"]) > 0


def test_predictions_endpoint(client):
    """/api/predictions returns a list (may be empty on fresh DB)."""
    resp = client.get("/api/predictions")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)


def test_predictions_stored_after_predict(client):
    """After a valid prediction, GET /api/predictions should return ≥ 1 row."""
    client.post(
        "/api/predict",
        data=json.dumps(_BASE),
        content_type="application/json",
    )
    resp = client.get("/api/predictions")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["predictions"]) >= 1


def test_spectrum_analyze_endpoint(client):
    """POST /api/spectrum/analyze returns spectrum data and features."""
    payload = {
        "center_freq_mhz": 120.0,
        "bandwidth_mhz": 10.0,
        "signal_strength_dbm": -60.0,
        "noise_floor_dbm": -100.0,
    }
    resp = client.post(
        "/api/spectrum/analyze",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 200
    assert resp.content_type == "application/json"
    data = resp.get_json()
    assert "spectrum_data" in data
    assert "frequencies" in data["spectrum_data"]
    assert "power_dbm" in data["spectrum_data"]
    assert "noise_floor_dbm" in data
    assert "detected_signals" in data
    assert "extracted_features" in data
    assert len(data["spectrum_data"]["frequencies"]) > 0


def test_spectrum_analyze_invalid_body(client):
    """POST /api/spectrum/analyze with invalid data returns JSON 400 error."""
    resp = client.post(
        "/api/spectrum/analyze",
        data="invalid json",
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert resp.content_type == "application/json"
    data = resp.get_json()
    assert "error" in data


def test_allocation_recommend_endpoint(client):
    """POST /api/allocation/recommend returns structured allocation candidates."""
    payload = {
        "start_freq_mhz": 70.0,
        "end_freq_mhz": 160.0,
        "channel_bw_mhz": 0.2,
        "guard_band_mhz": 0.05,
        "noise_floor_dbm": -100.0,
        "observed_power_dbm": -75.0,
    }
    resp = client.post(
        "/api/allocation/recommend",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 200
    assert resp.content_type == "application/json"
    data = resp.get_json()
    assert "recommended_candidate" in data
    assert "candidates" in data or "candidate_channels" in data
    candidates_list = data.get("candidates") or data.get("candidate_channels")
    assert isinstance(candidates_list, list)
    assert len(candidates_list) > 0


def test_allocation_recommend_invalid_body(client):
    """POST /api/allocation/recommend with invalid payload returns JSON 400 error."""
    resp = client.post(
        "/api/allocation/recommend",
        data=json.dumps({"start_freq_mhz": "invalid"}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert resp.content_type == "application/json"
    data = resp.get_json()
    assert "error" in data


def test_rf_waveform_endpoint(client):
    """GET /api/rf/waveform returns waveform response."""
    resp = client.get("/api/rf/waveform")
    assert resp.status_code == 200
    assert resp.content_type == "application/json"
    data = resp.get_json()
    assert "iq_available" in data
    assert "provenance" in data


def test_analytics_events_endpoint(client):
    """GET /api/analytics/events returns events summary."""
    resp = client.get("/api/analytics/events")
    assert resp.status_code == 200
    assert resp.content_type == "application/json"
    data = resp.get_json()
    assert "total_rf_events" in data
    assert "events" in data


def test_analytics_utilization_endpoint(client):
    """GET /api/analytics/utilization returns channel utilization."""
    resp = client.get("/api/analytics/utilization")
    assert resp.status_code == 200
    assert resp.content_type == "application/json"
    data = resp.get_json()
    assert "utilization_percentage" in data or "utilization_status" in data


def test_technical_report_endpoint(client):
    """GET /api/report/rf-analysis returns 10-section report."""
    resp = client.get("/api/report/rf-analysis?frequency_mhz=120.0&signal_power_dbm=-75.0&noise_floor_dbm=-100.0")
    assert resp.status_code == 200
    assert resp.content_type == "application/json"
    data = resp.get_json()
    assert "sections" in data
    assert "10_limitations" in data["sections"]


def test_404_error_returns_json(client):
    """Unmatched route returns JSON 404 error, not HTML."""
    resp = client.get("/api/nonexistent-endpoint")
    assert resp.status_code == 404
    assert resp.content_type == "application/json"
    data = resp.get_json()
    assert "error" in data

