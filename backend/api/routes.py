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
import numpy as np
from flask import Blueprint, request, jsonify
from pydantic import ValidationError
from sqlalchemy import text, func

from backend.api.schemas import PredictionRequest
from backend.database.connection import get_db
from backend.database.models import AvailabilityCandidate
from backend.services.prediction import (
    get_model_bundle,
    get_model_metadata,
    run_ml_inference,
    save_prediction_to_db,
)
from backend.rf.rf_source import SimulatedRFSource
from backend.rf.spectrum_processor import compute_fft_psd
from backend.rf.noise_estimator import estimate_noise_floor
from backend.rf.peak_detector import detect_peaks, get_occupied_regions
from backend.rf.feature_extractor import extract_ml_features

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
        data = request.json or {}
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
