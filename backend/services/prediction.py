"""Wire Watcher prediction service.

Loads the trained ML model bundle and metadata once at import time.
Provides:
  - get_model_bundle()   → the joblib bundle dict
  - get_model_metadata() → the model_metadata.json dict
  - run_ml_inference()   → calls ml/inference/predict.py
  - save_prediction_to_db() → persists a prediction row to SQLite

The ML feature contract is read from the bundle at runtime; it is never
hard-coded here.  signal_strength_dbm is used only by the RF detector, NOT
as a ML predictor.
"""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from backend.config import MODEL_PATH, METADATA_PATH
from backend.database.models import AvailabilityCandidate
from backend.database.connection import get_db

# ── add ml/ to path so inference.predict can be imported ─────────────────────
_ML_DIR = Path(__file__).resolve().parents[2] / "ml"
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

try:
    from inference import predict as _predict_module
    _model_bundle = _predict_module.load_model(MODEL_PATH)
except Exception as e:  # pragma: no cover
    print(f"[prediction.py] WARNING: Failed to load ML model: {e}")
    _model_bundle = None
    _predict_module = None  # type: ignore[assignment]

try:
    _model_metadata: dict = json.loads(Path(METADATA_PATH).read_text(encoding="utf-8"))
except Exception as e:  # pragma: no cover
    print(f"[prediction.py] WARNING: Failed to load model metadata: {e}")
    _model_metadata = {}


# ── public accessors ──────────────────────────────────────────────────────────

def get_model_bundle():
    """Return the loaded ML model bundle (or None if load failed)."""
    return _model_bundle


def get_model_metadata() -> dict:
    """Return the model_metadata.json dict."""
    return _model_metadata


# ── inference ─────────────────────────────────────────────────────────────────

def run_ml_inference(ml_input: dict) -> dict:
    """Run the full inference pipeline (ML + detector + OOD).

    Parameters
    ----------
    ml_input:
        Dict of feature values.  Must include all feature_columns from the
        bundle PLUS signal_strength_dbm (used by the RF detector only).

    Returns
    -------
    dict from ``predict.predict()`` with keys:
        activity, availability, confidence, ml_activity_probability,
        ml_activity, detector_activity, ood, ood_reasons, threshold.
    """
    if _model_bundle is None or _predict_module is None:
        raise RuntimeError(
            "ML model is not loaded.  Check ml/artifacts/wire_watcher_model.pkl."
        )
    return _predict_module.predict(ml_input, _model_bundle)


# ── database persistence ──────────────────────────────────────────────────────

def save_prediction_to_db(
    validated_data,
    result: dict,
    data_source: str,
) -> None:
    """Persist a prediction result to the database.

    Parameters
    ----------
    validated_data:
        Pydantic-validated PredictionRequest instance.
    result:
        Dict returned by ``run_ml_inference``.
    data_source:
        Provenance string (e.g. "RF Signal Data — real measured + derived + inferred").
    """
    db_gen = get_db()
    db = next(db_gen)
    try:
        half_bw = validated_data.bandwidth_khz / 2_000.0  # kHz → MHz

        availability = result.get("availability", "UNCERTAIN")
        is_available = availability == "AVAILABLE"

        c = AvailabilityCandidate(
            # ── frequency band ────────────────────────────────────────────────
            frequency_start_mhz    = validated_data.frequency_mhz - half_bw,
            frequency_end_mhz      = validated_data.frequency_mhz + half_bw,
            required_bandwidth_mhz = validated_data.bandwidth_khz / 1_000.0,
            # ── optional location metadata ────────────────────────────────────
            region    = validated_data.location,
            state     = None,
            district  = None,
            latitude  = validated_data.latitude,
            longitude = validated_data.longitude,
            required_service = None,
            # ── ML / detector results ─────────────────────────────────────────
            activity                        = result.get("activity"),
            availability                    = availability,
            ml_activity                     = result.get("ml_activity"),
            ml_activity_probability         = float(result.get("ml_activity_probability", 0.0)),
            detector_activity               = result.get("detector_activity"),
            confidence                      = str(result.get("confidence", "")),
            label_type                      = "INFERRED_RF_ACTIVITY",
            # ── legacy compat fields ──────────────────────────────────────────
            recommendation_status               = "recommended" if is_available else "review_required",
            predicted_availability_probability  = float(result.get("ml_activity_probability", 0.0)),
            # ── RF signal ────────────────────────────────────────────────────
            signal_power_dbm = validated_data.signal_strength_dbm,
            noise_floor_dbm  = None,   # not calculated in this path
            snr_db           = None,   # not calculated in this path
            # ── OOD ──────────────────────────────────────────────────────────
            ood_status        = "1" if result.get("ood") else "0",
            raw_warning       = "; ".join(result.get("ood_reasons", [])) or None,
            # ── metadata ─────────────────────────────────────────────────────
            threshold_applied = float(result.get("threshold", 0.5)),
            model_version     = _model_metadata.get("model_version", "rf-signal-activity-v2"),
            data_source       = data_source[:255],
            generated_at      = datetime.now(timezone.utc),
        )
        db.add(c)
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"[prediction.py] WARNING: Failed to save prediction to database: {exc}")
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass
