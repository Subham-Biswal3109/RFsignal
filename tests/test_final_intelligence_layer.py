"""Tests for the Final RF Engineering Intelligence Layer.

Covers:
1. Channel Allocation Engine (guard-band, interference, transparent scoring, recommendations)
2. RF Event Detector & Persistence
3. Channel Utilization & Occupancy Analytics
4. RF Replay Controller
5. Automatic RF Technical Report Generator
6. Flask API endpoints
"""
import pytest
from backend.services.allocation_service import ChannelAllocationEngine
from backend.rf.event_detector import RFEventDetector, get_rf_event_summary
from backend.services.occupancy_service import calculate_channel_utilization
from backend.rf.replay_controller import RFReplayController
from backend.services.report_service import generate_rf_technical_report
from backend.database.models import RFEventLog


@pytest.fixture(scope="function")
def db_session(app):
    """Return a SQLAlchemy session bound to the test app's in-memory DB."""
    import backend.database.connection as conn
    db = conn.SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_guard_band_and_interference_analysis():
    # 1. Safe margin
    lower, upper, gb_status = ChannelAllocationEngine.analyze_guard_band(
        center_freq_mhz=100.0,
        bandwidth_mhz=0.2,
        guard_band_mhz=0.05,
        active_peaks=[],
    )
    assert lower == pytest.approx(99.85)
    assert upper == pytest.approx(100.15)
    assert gb_status == "SAFE_MARGIN"

    # 2. Conflict (peak directly inside channel bandwidth)
    _, _, gb_status_conflict = ChannelAllocationEngine.analyze_guard_band(
        center_freq_mhz=100.0,
        bandwidth_mhz=0.2,
        guard_band_mhz=0.05,
        active_peaks=[{"frequency_mhz": 100.05}],
    )
    assert gb_status_conflict == "CONFLICT"

    # 3. Marginal (peak in protected guard band region)
    _, _, gb_status_marginal = ChannelAllocationEngine.analyze_guard_band(
        center_freq_mhz=100.0,
        bandwidth_mhz=0.2,
        guard_band_mhz=0.05,
        active_peaks=[{"frequency_mhz": 100.12}],
    )
    assert gb_status_marginal == "MARGINAL"

    # 4. Interference risk classification
    risk_high = ChannelAllocationEngine.analyze_interference_risk(
        center_freq_mhz=100.0,
        bandwidth_mhz=0.2,
        activity="DETECTED",
        guard_band_status="SAFE_MARGIN",
    )
    assert risk_high == "HIGH"

    risk_low = ChannelAllocationEngine.analyze_interference_risk(
        center_freq_mhz=100.0,
        bandwidth_mhz=0.2,
        activity="NOT_DETECTED",
        guard_band_status="SAFE_MARGIN",
    )
    assert risk_low == "LOW"


def test_channel_scoring_and_recommendation():
    res = ChannelAllocationEngine.generate_and_score_candidates(
        start_freq_mhz=100.0,
        end_freq_mhz=102.0,
        channel_bw_mhz=0.2,
        guard_band_mhz=0.05,
        noise_floor_dbm=-100.0,
        observed_power_dbm=-85.0,
    )

    assert "candidates" in res
    assert res["total_candidates"] > 0
    assert "scoring_weights" in res
    assert "disclaimer" in res
    cand = res["candidates"][0]
    assert "protected_lower_mhz" in cand
    assert "protected_upper_mhz" in cand
    assert "guard_band_status" in cand
    assert "interference_risk" in cand
    assert "assessment_explanation" in cand


def test_rf_event_detector_and_summary(app, db_session):
    detector = RFEventDetector(frequency_threshold_dbm=-80.0)

    # Process active start
    closed = detector.process_observation(
        frequency_mhz=120.0,
        power_dbm=-70.0,
        is_active=True,
        source_type="DATASET",
        db_session=db_session,
    )
    assert closed is None

    # Process inactive -> closes event
    closed = detector.process_observation(
        frequency_mhz=120.0,
        power_dbm=-95.0,
        is_active=False,
        source_type="DATASET",
        db_session=db_session,
    )
    assert closed is not None
    assert closed["frequency_mhz"] == 120.0
    assert closed["duration_seconds"] > 0

    summary = get_rf_event_summary(db_session=db_session)
    assert summary["total_rf_events"] >= 1
    assert summary["maximum_event_duration_s"] > 0


def test_occupancy_utilization_analytics(app, db_session):
    # Empty DB check
    util_empty = calculate_channel_utilization(target_frequency_mhz=999.0, db_session=db_session)
    assert util_empty["utilization_status"] == "UNAVAILABLE"
    assert util_empty["reason"] == "insufficient temporal observations"

    # Add a mock event
    event = RFEventLog(
        frequency_mhz=100.0,
        bandwidth_mhz=0.2,
        duration_seconds=180.0,
        peak_power_dbm=-70.0,
        avg_power_dbm=-75.0,
        activity_state="DETECTED",
        source_type="DATASET",
    )
    db_session.add(event)
    db_session.commit()

    util_res = calculate_channel_utilization(target_frequency_mhz=100.0, db_session=db_session)
    assert util_res["utilization_status"] == "AVAILABLE"
    assert util_res["event_count"] >= 1
    assert util_res["utilization_percentage"] > 0


def test_rf_replay_controller():
    controller = RFReplayController.get_instance()
    controller.control("play")
    status = controller.control("set_speed", speed=2.0)
    assert status["mode"] == "REPLAY_MODE"
    assert status["is_playing"] is True
    assert status["playback_speed"] == 2.0
    assert status["is_live"] is False
    assert "NOT LIVE" in status["display_warning"]


def test_automatic_rf_technical_report():
    report = generate_rf_technical_report(
        frequency_mhz=120.0,
        signal_power_dbm=-75.0,
        noise_floor_dbm=-100.0,
    )
    assert "report_id" in report
    assert "sections" in report
    sec = report["sections"]
    assert "1_observation" in sec
    assert "2_dsp" in sec
    assert "3_rf_engineering" in sec
    assert "4_ml" in sec
    assert "5_ood" in sec
    assert "6_availability" in sec
    assert "7_allocation" in sec
    assert "8_events" in sec
    assert "9_provenance" in sec
    assert "10_limitations" in sec


def test_api_routes(client):
    # GET /api/allocation/candidates
    resp = client.get("/api/allocation/candidates?start_freq_mhz=100.0&end_freq_mhz=102.0")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "candidates" in data

    # GET /api/allocation/recommended
    resp = client.get("/api/allocation/recommended?start_freq_mhz=100.0&end_freq_mhz=102.0")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "has_suitable_candidate" in data

    # GET /api/analytics/events
    resp = client.get("/api/analytics/events")
    assert resp.status_code == 200

    # GET /api/analytics/utilization
    resp = client.get("/api/analytics/utilization")
    assert resp.status_code == 200

    # GET /api/report/rf-analysis
    resp = client.get("/api/report/rf-analysis")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "sections" in data

    # POST /api/replay/control
    resp = client.post("/api/replay/control", json={"action": "play", "speed": 1.0})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["is_playing"] is True


def test_dataset_event_ingestion(db_session):
    from backend.rf.event_detector import derive_and_ingest_events_from_dataset
    res = derive_and_ingest_events_from_dataset(db_session=db_session, force_reingest=True)
    assert res["status"] == "success"
    assert res["events_created"] > 0
    assert res["observations_processed"] == 164160

    summary = get_rf_event_summary(db_session=db_session)
    assert summary["total_rf_events"] > 0
    assert summary["average_event_duration_s"] > 0


def test_replay_controller_dataset_total_samples():
    controller = RFReplayController.get_instance()
    status = controller.get_status()
    assert status["total_samples"] == 164160
    assert status["provenance"] == "DATASET"

