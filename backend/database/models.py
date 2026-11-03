"""ORM models for Wire Watcher.

AvailabilityCandidate stores every prediction made via POST /api/predict.

Design notes
────────────
• Historical synthetic records already in the database use the same table.
  They are NOT deleted; legacy rows simply have data_source = 'synthetic' or
  similar.
• Fields that the current application does not calculate (e.g. noise_floor_dbm,
  snr_db) are stored as NULL — no fake values are inserted.
• The schema is intentionally broad so that future hardware-level SDR sources
  can populate the optional numeric columns without a migration.
"""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, Text
)
from backend.database.connection import Base


class AvailabilityCandidate(Base):
    """One prediction / availability-assessment record."""

    __tablename__ = "availability_candidates"

    candidate_id = Column(Integer, primary_key=True, autoincrement=True)

    # ── Frequency band ────────────────────────────────────────────────────────
    frequency_start_mhz = Column(Float, nullable=True)
    frequency_end_mhz   = Column(Float, nullable=True)
    required_bandwidth_mhz = Column(Float, nullable=True)

    # ── Location (optional application metadata, not ML features) ─────────────
    region   = Column(String(255), nullable=True)
    state    = Column(String(128), nullable=True)
    district = Column(String(128), nullable=True)
    latitude  = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    # ── Service context ───────────────────────────────────────────────────────
    required_service = Column(String(128), nullable=True)

    # ── ML / detector outputs ────────────────────────────────────────────────
    activity                    = Column(String(32),  nullable=True)   # DETECTED / NOT_DETECTED / UNCERTAIN
    availability                = Column(String(32),  nullable=True)   # AVAILABLE / OCCUPIED / UNCERTAIN
    ml_activity                 = Column(Boolean,     nullable=True)
    ml_activity_probability     = Column(Float,       nullable=True)
    detector_activity           = Column(Boolean,     nullable=True)
    confidence                  = Column(String(64),  nullable=True)   # High / Medium / Low / OOD / Unreliable
    label_type                  = Column(String(64),  nullable=True,
                                         default="INFERRED_RF_ACTIVITY")

    # ── Legacy / compatibility fields used by the GET /api/predictions route ──
    # recommendation_status mirrors availability for backward compat.
    recommendation_status          = Column(String(64), nullable=True)
    predicted_availability_probability = Column(Float,  nullable=True)

    # ── RF signal fields (REAL_MEASURED when available) ──────────────────────
    signal_power_dbm = Column(Float, nullable=True)
    # noise_floor_dbm and snr_db are nullable; only set when the RF source
    # provides these values.  They are NOT fabricated.
    noise_floor_dbm  = Column(Float, nullable=True)
    snr_db           = Column(Float, nullable=True)

    # ── OOD ───────────────────────────────────────────────────────────────────
    ood_status = Column(String(4), nullable=True)   # '0' or '1' (legacy string)

    # ── Metadata ──────────────────────────────────────────────────────────────
    threshold_applied = Column(Float, nullable=True)
    model_version     = Column(String(64), nullable=True)
    data_source       = Column(String(255), nullable=True)
    generated_at      = Column(DateTime, default=datetime.utcnow, nullable=False)
    raw_warning       = Column(Text, nullable=True)   # OOD reason text

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<AvailabilityCandidate id={self.candidate_id} "
            f"freq={self.frequency_start_mhz}-{self.frequency_end_mhz} MHz "
            f"availability={self.availability}>"
        )


class RFEventLog(Base):
    """Stores detected RF active events over time for analytics & temporal occupancy."""

    __tablename__ = "rf_events"

    event_id = Column(Integer, primary_key=True, autoincrement=True)
    frequency_mhz = Column(Float, nullable=False)
    bandwidth_mhz = Column(Float, nullable=False, default=0.2)
    start_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    peak_power_dbm = Column(Float, nullable=False)
    avg_power_dbm = Column(Float, nullable=False)
    activity_state = Column(String(32), nullable=False, default="DETECTED")
    source_type = Column(String(64), nullable=False, default="DATASET")
    provenance = Column(String(64), nullable=False, default="INFERRED_RF_ACTIVITY")

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "frequency_mhz": round(self.frequency_mhz, 3),
            "bandwidth_mhz": round(self.bandwidth_mhz, 3),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": round(self.duration_seconds, 2) if self.duration_seconds is not None else None,
            "peak_power_dbm": round(self.peak_power_dbm, 2),
            "avg_power_dbm": round(self.avg_power_dbm, 2),
            "activity_state": self.activity_state,
            "source_type": self.source_type,
            "provenance": self.provenance,
        }

