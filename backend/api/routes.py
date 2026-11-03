"""Wire Watcher Flask API routes.

Endpoints
─────────
  GET  /api/health          — liveness + model/db status
  GET  /api/model-info      — model metadata + KPIs from DB
  POST /api/predict         — RF activity / availability prediction
  GET  /api/predictions     — last 100 stored predictions
  POST /api/spectrum/analyze — spectrum visualisation (simulated RF)

All prediction values are ACTUAL computed results.  No values are hard-coded.

Data-provenance note
────────────────────
• signal_strength_dbm is a REAL_MEASURED detector input excluded from ML.
• ML features are derived I/Q statistics (DERIVED) + frequency/bandwidth.
• The target (inferred_rf_activity) is an INFERRED/PSEUDO-LABEL.
• The system does NOT claim ground-truth occupancy or live SDR monitoring.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from flask import Blueprint, request, jsonify
from pydantic import ValidationError
from sqlalchemy import text, func

from backend.api.schemas import PredictionRequest, SdrConfigRequest, SdrCaptureRequest, AllocationRequest
from backend.database.connection import get_db
from backend.database.models import AvailabilityCandidate
from backend.services.prediction import (
    get_model_bundle,
    get_model_metadata,
    run_ml_inference,
    save_prediction_to_db,
)
from backend.services.allocation_service import ChannelAllocationEngine
from backend.rf.rf_source import SimulatedRFSource
from backend.rf.spectrum_processor import compute_fft_psd
from backend.rf.noise_estimator import estimate_noise_floor
from backend.rf.peak_detector import detect_peaks, get_occupied_regions
from backend.rf.feature_extractor import extract_ml_features
from backend.rf.sources import RFSourceFactory, RtlSdrSource
from backend.rf.event_detector import get_rf_event_summary
from backend.services.occupancy_service import calculate_channel_utilization
from backend.rf.replay_controller import RFReplayController
from backend.services.report_service import generate_rf_technical_report
from backend.rf.waveform_processor import process_time_domain_waveform


api_bp = Blueprint("api_bp", __name__)


@api_bp.route("/", methods=["GET"])
@api_bp.route("/api", methods=["GET"])
def api_root():
    """Root endpoint: returns system status and available API route definitions."""
    return jsonify({
        "system": "Wire Watcher — RF Spectrum Monitoring & Availability Analysis",
        "status": "online",
        "version": "2.0.0",
        "documentation": "ECE / RF Spectrum Monitoring & Availability Decision System",
        "endpoints": {
            "GET /api/health": "Liveness probe & DB/model load status",
            "GET /api/model-info": "Model metadata, feature list, and DB KPIs",
            "POST /api/predict": "RF observation availability prediction",
            "GET /api/predictions": "Prediction history records from SQLite",
            "POST /api/spectrum/analyze": "FFT PSD spectral estimation & peak detection"
        }
    }), 200



# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _db_session():
    """Yield a DB session and clean up."""
    gen = get_db()
    db = next(gen)
    try:
        yield db
    finally:
        try:
            next(gen)
        except StopIteration:
            pass


def _confidence_label(probability: float, ood: bool) -> str:
    if ood:
        return "OOD / Unreliable"
    dist = abs(probability - 0.5)
    if dist > 0.15:
        return "High"
    if dist > 0.05:
        return "Medium"
    return "Low"


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/api/predict", methods=["POST"])
def run_prediction():
    """RF activity / availability prediction.

    Input: PredictionRequest (frequency_mhz, bandwidth_khz, signal_strength_dbm,
           optional I/Q features).
    Output: availability decision, ML probability, OOD status, etc.
    """
    bundle = get_model_bundle()
    if bundle is None:
        return jsonify({"error": "ML model failed to load on startup."}), 500

    # ── validate input ────────────────────────────────────────────────────────
    try:
        validated = PredictionRequest(**(request.json or {}))
    except ValidationError as exc:
        return jsonify({"error": "Invalid RF input", "details": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "Invalid JSON format", "details": str(exc)}), 400

    # ── build ML feature dict from validated request ───────────────────────────
    metadata = get_model_metadata()
    feature_columns = metadata.get("features", bundle.get("feature_columns", []))

    # Collect every feature the model expects; fall back to NaN for absent ones.
    ml_input: dict = {}
    for col in feature_columns:
        val = getattr(validated, col, None)
        ml_input[col] = float(val) if val is not None else float("nan")

    # signal_strength_dbm is needed by the RF detector inside predict().
    ml_input["signal_strength_dbm"] = float(validated.signal_strength_dbm)
    ml_input["frequency_mhz"]       = float(validated.frequency_mhz)

    # ── run inference ─────────────────────────────────────────────────────────
    try:
        result = run_ml_inference(ml_input)
    except Exception as exc:
        return jsonify({"error": "Prediction failed", "details": str(exc)}), 500

    # ── persist to DB ─────────────────────────────────────────────────────────
    data_source = metadata.get(
        "data_source",
        "RF Signal Data — real measured + derived + inferred"
    )
    save_prediction_to_db(validated, result, data_source)

    # ── build response ────────────────────────────────────────────────────────
    probability = float(result["ml_activity_probability"])
    ood         = bool(result["ood"])

    payload = {
        # ── primary decision ─────────────────────────────────────────────────
        "prediction":    result["availability"],
        "available":     result["availability"] == "AVAILABLE",
        "activity":      result["activity"],
        "availability":  result["availability"],
        # ── probabilities / confidence ────────────────────────────────────────
        "probability":           probability,
        "occupied_probability":  probability,
        "ml_activity_probability": probability,
        "confidence": _confidence_label(probability, ood),
        # ── detector + ML flags ───────────────────────────────────────────────
        "ml_activity":       result["ml_activity"],
        "detector_activity": result["detector_activity"],
        # ── OOD ──────────────────────────────────────────────────────────────
        "ood_warning": ood,
        "warning": "; ".join(result["ood_reasons"]) if result.get("ood_reasons") else None,
        # ── metadata ─────────────────────────────────────────────────────────
        "data_source":       data_source,
        "label_type":        "INFERRED_RF_ACTIVITY",
        "threshold":         float(result["threshold"]),
        "target_definition": metadata.get("target_method"),
        "model_version":     metadata.get("model_version"),
        "live_rf_ingestion": False,
        # ── transparency ─────────────────────────────────────────────────────
        "features_used":      {k: _safe_float(v) for k, v in ml_input.items()
                               if k != "signal_strength_dbm"},
        "important_features": list(metadata.get("feature_importances", {}).keys())[:5],
    }
    return jsonify(payload), 200



# ─────────────────────────────────────────────────────────────────────────────
# GET /api/predictions
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/api/predictions", methods=["GET"])
def get_predictions():
    """Return the last 100 stored predictions (newest first)."""
    for db in _db_session():
        try:
            rows = (
                db.query(AvailabilityCandidate)
                .order_by(AvailabilityCandidate.generated_at.desc())
                .limit(100)
                .all()
            )
            results = []
            for row in rows:
                results.append({
                    "id":                   row.candidate_id,
                    "start_frequency_mhz":  _safe_float(row.frequency_start_mhz),
                    "end_frequency_mhz":    _safe_float(row.frequency_end_mhz),
                    "bandwidth_mhz":        _safe_float(row.required_bandwidth_mhz),
                    "city":     row.district,
                    "state":    row.state,
                    "service_type": row.required_service,
                    "available":    bool(row.recommendation_status == "recommended")
                                    if row.recommendation_status else False,
                    "probability":  _safe_float(row.predicted_availability_probability),
                    "timestamp":    row.generated_at.isoformat() if row.generated_at else None,
                    "signal_power_dbm":  _safe_float(row.signal_power_dbm),
                    "noise_floor_dbm":   _safe_float(row.noise_floor_dbm),
                    "snr_db":            _safe_float(row.snr_db),
                    "ood_status":        row.ood_status == "1",
                    "data_source":       row.data_source,
                    "activity":          row.activity,
                    "availability":      row.availability,
                    "confidence":        row.confidence,
                    "label_type":        row.label_type,
                    "model_version":     row.model_version,
                })
            return jsonify({"predictions": results}), 200
        except Exception as exc:
            return jsonify({"error": "Failed to fetch predictions", "details": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/health
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/api/health", methods=["GET"])
def health_check():
    """Liveness probe: reports model load status and DB connectivity."""
    db_connected = False
    try:
        for db in _db_session():
            db.execute(text("SELECT 1"))
            db_connected = True
    except Exception:
        pass

    model_loaded = get_model_bundle() is not None
    status = "ok" if (model_loaded and db_connected) else "degraded"

    return jsonify({
        "status":             status,
        "model_loaded":       model_loaded,
        "database_connected": db_connected,
        "api":                "ok",
        "model":              "loaded" if model_loaded else "failed",
        "database":           "connected" if db_connected else "failed",
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/model-info
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/api/model-info", methods=["GET"])
def model_info():
    """Return model metadata and database KPIs."""
    bundle = get_model_bundle()
    if bundle is None:
        return jsonify({"error": "Model not loaded"}), 500

    meta = get_model_metadata()

    # ── DB KPIs ───────────────────────────────────────────────────────────────
    total = available = ood_count = 0
    avg_prob = 0.0
    try:
        for db in _db_session():
            stats = db.query(
                func.count(AvailabilityCandidate.candidate_id).label("total"),
                func.avg(AvailabilityCandidate.ml_activity_probability).label("avg_prob"),
                func.sum(
                    func.cast(AvailabilityCandidate.ood_status == "1", text("integer"))
                ).label("ood"),
                func.sum(
                    func.cast(AvailabilityCandidate.recommendation_status == "recommended",
                              text("integer"))
                ).label("available"),
            ).first()
            if stats and stats.total:
                total     = int(stats.total)
                available = int(stats.available or 0)
                avg_prob  = float(stats.avg_prob or 0.0)
                ood_count = int(stats.ood or 0)
    except Exception as exc:
        print(f"[model-info] DB KPI query failed: {exc}")

    return jsonify({
        "model_name":         "Wire Watcher RF Activity Model",
        "algorithm":          meta.get("final_model", "RandomForest"),
        "model_version":      meta.get("model_version", "rf-signal-activity-v2"),
        "training_date":      meta.get("training_date", ""),
        "dataset_name":       meta.get("dataset_name", "RF Signal Data / logged_data.csv"),
        "dataset_type":       meta.get("dataset_type", ""),
        "training_samples":   meta.get("training_samples", 0),
        "validation_samples": meta.get("validation_samples", 0),
        "test_samples":       meta.get("test_samples", 0),
        "features":           meta.get("features", []),
        "feature_count":      len(meta.get("features", [])),
        "target":             meta.get("target", "inferred_rf_activity"),
        "label_type":         "INFERRED/PSEUDO-LABEL",
        "target_method":      meta.get("target_method", ""),
        "target_is_ground_truth":   meta.get("target_is_ground_truth", False),
        "ground_truth_occupancy":   False,
        "live_rf_ingestion":        meta.get("live_rf_ingestion", False),
        "data_source":              meta.get("data_source", "RF Signal Data"),
        "threshold_occupied":       meta.get("threshold_occupied", 0.5),
        "threshold_available":      meta.get("threshold_available", 0.5),
        "feature_importances":      meta.get("feature_importances", {}),
        "limitations":              meta.get("limitations", []),
        "validation_metrics":       meta.get("validation_metrics", {}),
        "test_metrics":             meta.get("test_metrics", {}),
        "calibrated":               False,
        "kpis": {
            "total_predictions":     total,
            "available_predictions": available,
            "occupied_predictions":  total - available,
            "avg_probability":       avg_prob,
            "ood_count":             ood_count,
        },
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/spectrum/analyze
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/api/spectrum/analyze", methods=["POST"])
def analyze_spectrum():
    """Spectrum visualisation using a simulated RF signal.

    This endpoint is for VISUALISATION ONLY.  The ML model is NOT called
    here; no availability decision is made.  I/Q is not fabricated.
    """
    try:
        data = request.get_json(silent=True)
        if data is None and request.data:
            return jsonify({"error": "Invalid JSON in request body", "status": 400}), 400
        if data is None:
            data = {}
        center_freq_mhz    = float(data.get("center_freq_mhz", 1800.0))
        bandwidth_mhz      = float(data.get("bandwidth_mhz", 10.0))
        signal_strength_dbm = float(data.get("signal_strength_dbm", -75.0))
        noise_floor_dbm    = float(data.get("noise_floor_dbm", -100.0))

        source = SimulatedRFSource(
            center_freq_mhz    = center_freq_mhz,
            bandwidth_mhz      = bandwidth_mhz,
            signal_strength_dbm = signal_strength_dbm,
            noise_floor_dbm    = noise_floor_dbm,
        )
        rx_signal, sample_rate_mhz = source.get_signal()

        freqs_mhz, psd_dbm = compute_fft_psd(rx_signal, sample_rate_mhz, center_freq_mhz)
        estimated_noise     = estimate_noise_floor(psd_dbm)
        peaks               = detect_peaks(freqs_mhz, psd_dbm, estimated_noise)
        occupied            = get_occupied_regions(peaks)

        # Downsample for frontend JSON payload.
        factor = max(1, len(freqs_mhz) // 200)
        ml_features = extract_ml_features(
            center_freq_mhz, bandwidth_mhz, peaks, estimated_noise, location_info=data
        )

        return jsonify({
            "data_source": source.get_source_name(),
            "frequency_range": {
                "start_mhz": center_freq_mhz - bandwidth_mhz / 2,
                "end_mhz":   center_freq_mhz + bandwidth_mhz / 2,
            },
            "noise_floor_dbm":   round(estimated_noise, 2),
            "detected_signals":  peaks,
            "occupied_regions":  occupied,
            "available_regions": [],
            "spectrum_data": {
                "frequencies": [round(f, 3) for f in freqs_mhz[::factor].tolist()],
                "power_dbm":   [round(p, 2) for p in psd_dbm[::factor].tolist()],
            },
            "extracted_features": ml_features,
        }), 200
    except Exception as exc:
        return jsonify({"error": "Spectrum analysis failed", "details": str(exc)}), 500



# ─────────────────────────────────────────────────────────────────────────────
# RTL-SDR & RF Sources Endpoints (Phase 4A)
# ─────────────────────────────────────────────────────────────────────────────

_sdr_instance = RtlSdrSource()


@api_bp.route("/api/sdr/status", methods=["GET"])
def get_sdr_status():
    """Return status of physical RTL-SDR hardware adapter."""
    return jsonify(_sdr_instance.get_status()), 200


@api_bp.route("/api/sdr/configure", methods=["POST"])
def configure_sdr():
    """Configure RTL-SDR hardware parameters (center_freq_mhz, sample_rate_mhz, gain, buffer_size)."""
    try:
        req_data = SdrConfigRequest(**(request.json or {}))
    except ValidationError as exc:
        return jsonify({"error": "Invalid SDR configuration parameters", "details": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "Invalid JSON payload", "details": str(exc)}), 400

    try:
        updated = _sdr_instance.configure(
            center_freq_mhz=req_data.center_freq_mhz,
            sample_rate_mhz=req_data.sample_rate_mhz,
            gain=req_data.gain,
            buffer_size=req_data.buffer_size,
        )
        status = _sdr_instance.get_status()
        status["configuration"] = updated
        return jsonify(status), 200
    except ValueError as exc:
        return jsonify({"error": "Configuration rejected", "details": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "SDR configuration failed", "details": str(exc)}), 500


@api_bp.route("/api/sdr/capture", methods=["POST"])
def capture_sdr_iq():
    """Capture a controlled finite IQ sample buffer from physical RTL-SDR hardware.

    Returns HTTP 503 if hardware is unavailable. Zero fake data generated.
    """
    if not _sdr_instance.is_available():
        return jsonify({
            "error": "RTL-SDR hardware unavailable",
            "connected": False,
            "message": _sdr_instance.get_status()["message"],
            "provenance": "REAL_SDR",
        }), 503

    try:
        req_data = SdrCaptureRequest(**(request.json or {}))
    except ValidationError as exc:
        return jsonify({"error": "Invalid capture request", "details": str(exc)}), 400

    try:
        obs = _sdr_instance.get_observation(
            center_freq_mhz=req_data.center_freq_mhz,
            sample_rate_mhz=req_data.sample_rate_mhz,
            gain=req_data.gain,
            num_samples=req_data.num_samples,
        )
        return jsonify({
            "status": "success",
            "observation": obs.to_dict(),
        }), 200
    except Exception as exc:
        return jsonify({"error": "IQ capture failed", "details": str(exc)}), 500


@api_bp.route("/api/sources", methods=["GET"])
def list_rf_sources():
    """List all supported RF sources and their operational status."""
    return jsonify({
        "sources": RFSourceFactory.list_sources()
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# Channel Allocation & Recommendation Endpoints (Full Integration Phase)
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/api/allocation/recommend", methods=["POST"])
def recommend_channel_allocation():
    """Generate candidate channels across a frequency band, score them, and return allocation recommendations."""
    try:
        req_data = AllocationRequest(**(request.json or {}))
    except ValidationError as exc:
        return jsonify({"error": "Invalid allocation request parameters", "details": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "Invalid JSON format", "details": str(exc)}), 400

    try:
        res = ChannelAllocationEngine.generate_and_score_candidates(
            start_freq_mhz=req_data.start_freq_mhz,
            end_freq_mhz=req_data.end_freq_mhz,
            channel_bw_mhz=req_data.channel_bw_mhz,
            guard_band_mhz=req_data.guard_band_mhz,
            noise_floor_dbm=req_data.noise_floor_dbm,
            observed_power_dbm=req_data.observed_power_dbm,
            location=req_data.location,
        )
        return jsonify(res), 200
    except ValueError as exc:
        return jsonify({"error": "Allocation calculation failed", "details": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "Channel allocation engine failed", "details": str(exc)}), 500


@api_bp.route("/api/allocation/config", methods=["GET"])
def get_allocation_config():
    """Return allocation engine configuration and scoring formula weights."""
    return jsonify({
        "scoring_weights": {
            "base_score": 100.0,
            "occupied_penalty": 50.0,
            "ood_penalty": 40.0,
            "uncertainty_penalty": 25.0,
            "noise_baseline_dbm": -100.0,
            "noise_penalty_weight": 1.5,
            "max_noise_penalty": 30.0,
            "min_recommendation_score": 50.0,
        },
        "disclaimer": "Engineering availability assessment and allocation recommendation. Not a legal or regulatory transmission license.",
    }), 200


@api_bp.route("/api/allocation/candidates", methods=["GET"])
def get_allocation_candidates():
    """GET endpoint returning candidate channels list for allocation."""
    start_f = float(request.args.get("start_freq_mhz", 70.0))
    end_f = float(request.args.get("end_freq_mhz", 160.0))
    bw = float(request.args.get("channel_bw_mhz", 0.2))
    gb = float(request.args.get("guard_band_mhz", 0.05))
    noise = float(request.args.get("noise_floor_dbm", -100.0))

    res = ChannelAllocationEngine.generate_and_score_candidates(
        start_freq_mhz=start_f,
        end_freq_mhz=end_f,
        channel_bw_mhz=bw,
        guard_band_mhz=gb,
        noise_floor_dbm=noise,
    )
    return jsonify({"candidates": res["candidates"], "total_candidates": res["total_candidates"]}), 200


@api_bp.route("/api/allocation/recommended", methods=["GET"])
def get_recommended_channel():
    """GET endpoint returning top recommended channel candidate."""
    start_f = float(request.args.get("start_freq_mhz", 70.0))
    end_f = float(request.args.get("end_freq_mhz", 160.0))
    bw = float(request.args.get("channel_bw_mhz", 0.2))
    gb = float(request.args.get("guard_band_mhz", 0.05))
    noise = float(request.args.get("noise_floor_dbm", -100.0))

    res = ChannelAllocationEngine.generate_and_score_candidates(
        start_freq_mhz=start_f,
        end_freq_mhz=end_f,
        channel_bw_mhz=bw,
        guard_band_mhz=gb,
        noise_floor_dbm=noise,
    )
    return jsonify({
        "recommended_candidate": res["recommended_candidate"],
        "has_suitable_candidate": res["has_suitable_candidate"],
        "reason": res["reason"],
        "supporting_evidence": res["supporting_evidence"],
    }), 200


@api_bp.route("/api/allocation/analysis", methods=["GET"])
def get_allocation_analysis():
    """GET endpoint returning full allocation assessment analysis."""
    start_f = float(request.args.get("start_freq_mhz", 70.0))
    end_f = float(request.args.get("end_freq_mhz", 160.0))
    bw = float(request.args.get("channel_bw_mhz", 0.2))
    gb = float(request.args.get("guard_band_mhz", 0.05))
    noise = float(request.args.get("noise_floor_dbm", -100.0))

    res = ChannelAllocationEngine.generate_and_score_candidates(
        start_freq_mhz=start_f,
        end_freq_mhz=end_f,
        channel_bw_mhz=bw,
        guard_band_mhz=gb,
        noise_floor_dbm=noise,
    )
    return jsonify(res), 200


# ─────────────────────────────────────────────────────────────────────────────
# Event Analytics & Occupancy Timeline (Phase 4B & 4C)
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/api/events", methods=["GET"])
@api_bp.route("/api/analytics/events", methods=["GET"])
def get_events_analytics():
    """Return summary statistics of all detected RF events."""
    summary = get_rf_event_summary()
    return jsonify(summary), 200


@api_bp.route("/api/analytics/utilization", methods=["GET"])
def get_channel_utilization():
    """Return temporal channel utilization analytics across recorded events."""
    freq_param = request.args.get("target_frequency_mhz")
    target_freq = float(freq_param) if freq_param else None
    window_s = float(request.args.get("window_seconds", 3600.0))

    util_res = calculate_channel_utilization(
        target_frequency_mhz=target_freq,
        observation_window_seconds=window_s,
    )
    return jsonify(util_res), 200


# ─────────────────────────────────────────────────────────────────────────────
# Controlled RF Replay Mode (Section 13)
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/api/events/ingest", methods=["POST"])
def ingest_events_from_dataset():
    """Derive and ingest contiguous RF events from dataset into SQLite."""
    data = request.get_json(silent=True) or {}
    force = bool(data.get("force", True))

    from backend.rf.event_detector import derive_and_ingest_events_from_dataset
    res = derive_and_ingest_events_from_dataset(force_reingest=force)
    return jsonify(res), 200


@api_bp.route("/api/replay/control", methods=["POST"])
def control_rf_replay():
    """Control historical dataset/IQ observation replay (play, pause, reset, step, set_speed, set_index)."""
    data = request.json or {}
    action = data.get("action", "pause")
    speed = data.get("speed")
    target_index = data.get("index")

    controller = RFReplayController.get_instance()
    status = controller.control(action=action, speed=speed, target_index=target_index)
    if action == "step":
        obs = controller.step()
        status["current_observation"] = obs
    return jsonify(status), 200


@api_bp.route("/api/replay/status", methods=["GET"])
def get_rf_replay_status():
    """Get current RF replay state and provenance."""
    controller = RFReplayController.get_instance()
    return jsonify(controller.get_status()), 200


@api_bp.route("/api/replay/sample/<int:index>", methods=["GET"])
def get_replay_sample(index: int):
    """Get specific observation by index from dataset."""
    from backend.rf.sources.dataset_source import DatasetRFSource
    ds = DatasetRFSource()
    if not ds.is_available():
        return jsonify({"error": "Dataset unavailable", "status": 404}), 404
    if index < 0 or index >= ds.total_observations:
        return jsonify({"error": f"Index {index} out of bounds (0-{ds.total_observations-1})", "status": 400}), 400

    obs = ds.get_observation(index=index)
    return jsonify({
        "index": index,
        "total_samples": ds.total_observations,
        "observation": obs.to_dict(),
        "provenance": "DATASET",
    }), 200


@api_bp.route("/api/dataset/observations", methods=["GET"])
def get_dataset_observations():
    """Get paginated observation records from authentic dataset for Data Explorer."""
    from backend.rf.sources.dataset_source import DatasetRFSource
    ds = DatasetRFSource()
    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 50))
    freq_param = request.args.get("frequency_mhz")
    freq = float(freq_param) if freq_param else None

    res = ds.get_observations_page(offset=offset, limit=limit, frequency_mhz=freq)
    return jsonify(res), 200


# ─────────────────────────────────────────────────────────────────────────────
# Automatic RF Technical Report Generator (Section 19)
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/api/report/rf-analysis", methods=["GET"])
def get_rf_technical_report():
    """Generate comprehensive 10-section RF Technical Analysis Report."""
    freq_mhz = float(request.args.get("frequency_mhz", 120.0))
    power_dbm = float(request.args.get("signal_power_dbm", -75.0))
    noise_dbm = float(request.args.get("noise_floor_dbm", -100.0))
    source_type = request.args.get("source_type", "DATASET")

    report = generate_rf_technical_report(
        frequency_mhz=freq_mhz,
        signal_power_dbm=power_dbm,
        noise_floor_dbm=noise_dbm,
        source_type=source_type,
    )
    return jsonify(report), 200


# ─────────────────────────────────────────────────────────────────────────────
# RF Monitor & Waveform Endpoints (Section 40)
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route("/api/rf/observation", methods=["GET"])
def get_rf_observation():
    """GET current normalized RF observation metadata."""
    freq_mhz = float(request.args.get("frequency_mhz", 120.0))
    bw_mhz = float(request.args.get("bandwidth_mhz", 0.2))
    power_dbm = float(request.args.get("signal_power_dbm", -75.0))

    return jsonify({
        "status": "success",
        "observation": {
            "center_freq_mhz": freq_mhz,
            "bandwidth_khz": bw_mhz * 1000.0,
            "signal_strength_dbm": power_dbm,
            "sample_rate_mhz": 2.048,
            "source_type": "SIMULATION",
            "provenance_tag": "SIMULATION",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iq_available": 1,
        }
    }), 200


@api_bp.route("/api/rf/waveform", methods=["GET"])
def get_rf_waveform():
    """GET time-domain I/Q waveform series and quality metrics.

    If I/Q data is present, returns complex baseband series and derived metrics.
    If I/Q data is absent, returns explicit fallback status without generating fake data.
    """
    freq_mhz = float(request.args.get("frequency_mhz", 120.0))
    bw_mhz = float(request.args.get("bandwidth_mhz", 0.2))
    power_dbm = float(request.args.get("signal_power_dbm", -75.0))
    noise_dbm = float(request.args.get("noise_floor_dbm", -100.0))
    iq_flag = int(request.args.get("iq_available", 1))

    if iq_flag == 0:
        res = process_time_domain_waveform(
            iq_samples=None,
            sample_rate_mhz=2.048,
            center_freq_mhz=freq_mhz,
        )
        return jsonify(res), 200

    # Generate real complex baseband signal using SimulatedRFSource
    source = SimulatedRFSource(
        center_freq_mhz=freq_mhz,
        bandwidth_mhz=bw_mhz,
        signal_strength_dbm=power_dbm,
        noise_floor_dbm=noise_dbm,
        num_samples=1024,
    )
    rx_signal, sr_mhz = source.get_signal()

    res = process_time_domain_waveform(
        iq_samples=rx_signal,
        sample_rate_mhz=sr_mhz,
        center_freq_mhz=freq_mhz,
    )
    return jsonify(res), 200


@api_bp.route("/api/rf/spectrum", methods=["GET"])
def get_rf_spectrum():
    """GET FFT PSD spectral estimation and peak detection."""
    freq_mhz = float(request.args.get("frequency_mhz", 120.0))
    bw_mhz = float(request.args.get("bandwidth_mhz", 0.2))
    power_dbm = float(request.args.get("signal_power_dbm", -75.0))
    noise_dbm = float(request.args.get("noise_floor_dbm", -100.0))

    source = SimulatedRFSource(
        center_freq_mhz=freq_mhz,
        bandwidth_mhz=bw_mhz,
        signal_strength_dbm=power_dbm,
        noise_floor_dbm=noise_dbm,
    )
    rx_signal, sr_mhz = source.get_signal()
    freqs_mhz, psd_dbm = compute_fft_psd(rx_signal, sr_mhz, freq_mhz)
    estimated_noise = estimate_noise_floor(psd_dbm)
    peaks = detect_peaks(freqs_mhz, psd_dbm, estimated_noise)
    occupied = get_occupied_regions(peaks)

    factor = max(1, len(freqs_mhz) // 200)
    return jsonify({
        "data_source": "DSP Spectral Estimator",
        "frequency_range": {
            "start_mhz": freq_mhz - bw_mhz / 2,
            "end_mhz": freq_mhz + bw_mhz / 2,
        },
        "noise_floor_dbm": round(estimated_noise, 2),
        "detected_signals": peaks,
        "occupied_regions": occupied,
        "spectrum_data": {
            "frequencies": [round(f, 3) for f in freqs_mhz[::factor].tolist()],
            "power_dbm": [round(p, 2) for p in psd_dbm[::factor].tolist()],
        },
        "provenance": "DERIVED",
    }), 200


@api_bp.route("/api/rf/activity", methods=["GET"])
def get_rf_activity():
    """GET current RF activity state and signal-to-noise ratio."""
    power_dbm = float(request.args.get("signal_power_dbm", -75.0))
    noise_dbm = float(request.args.get("noise_floor_dbm", -100.0))

    is_active = power_dbm > (noise_dbm + 10.0)
    activity_state = "DETECTED" if is_active else "NOT_DETECTED"
    snr_db = power_dbm - noise_dbm

    return jsonify({
        "activity_state": activity_state,
        "signal_power_dbm": round(power_dbm, 2),
        "noise_floor_dbm": round(noise_dbm, 2),
        "snr_db": round(snr_db, 2),
        "provenance": "INFERRED_RF_ACTIVITY",
    }), 200


@api_bp.route("/api/ml/audit", methods=["GET"])
def get_ml_audit():
    """GET complete ML Audit, Leakage Audit, Validation Strategy comparison, and Baselines."""
    audit_file = Path(__file__).resolve().parents[2] / "ml" / "results" / "complete_ml_audit_results.json"
    if audit_file.exists():
        try:
            data = json.loads(audit_file.read_text(encoding="utf-8"))
            return jsonify(data), 200
        except Exception as exc:
            return jsonify({"error": f"Failed to load audit file: {exc}"}), 500
    
    # Fallback to dynamic audit run if file missing
    try:
        from ml.training.ml_audit_runner import run_full_ml_audit
        data = run_full_ml_audit()
        return jsonify(data), 200
    except Exception as exc:
        return jsonify({"error": f"Failed to run ML audit: {exc}"}), 500


@api_bp.route("/api/ml/features", methods=["GET"])
def get_ml_features():
    """GET ML Feature Importance (Gini & Permutation) labeled MODEL FEATURE IMPORTANCE."""
    audit_file = Path(__file__).resolve().parents[2] / "ml" / "results" / "complete_ml_audit_results.json"
    if audit_file.exists():
        try:
            data = json.loads(audit_file.read_text(encoding="utf-8"))
            feat_imp = data.get("phase_12_feature_importance", {})
            return jsonify(feat_imp), 200
        except Exception as exc:
            pass

    # Fall back to metadata if audit file unavailable
    metadata = get_model_metadata()
    return jsonify({
        "title": "MODEL FEATURE IMPORTANCE",
        "disclaimer": "Feature importance represents statistical model split contribution, NOT physical RF causality.",
        "gini_importance": metadata.get("feature_importances", {}),
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None





