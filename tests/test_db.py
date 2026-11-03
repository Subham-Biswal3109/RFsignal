"""Database layer tests.

Tests cover:
  - Table creation via init_db()
  - Inserting and retrieving AvailabilityCandidate rows
  - Nullable fields stored correctly (no fabricated values)
  - OOD status stored as string

Uses the `app` fixture from conftest.py which provides an isolated
in-memory SQLite database with tables already created.
"""
from __future__ import annotations
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="function")
def db_session(app):
    """Return a SQLAlchemy session bound to the test app's in-memory DB.

    Imports SessionLocal AFTER the app fixture has patched connection.engine,
    so this session targets the same in-memory DB that was populated with tables.
    """
    import backend.database.connection as conn
    db = conn.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── table creation ────────────────────────────────────────────────────────────

def test_init_db_creates_table(app, db_session):
    """AvailabilityCandidate table must exist after app startup."""
    from sqlalchemy import inspect
    import backend.database.connection as conn
    inspector = inspect(conn.engine)
    tables = inspector.get_table_names()
    assert "availability_candidates" in tables, \
        f"Table 'availability_candidates' not found. Got: {tables}"


# ── insert + retrieve ─────────────────────────────────────────────────────────

def test_insert_and_retrieve_prediction(app, db_session):
    """A prediction row can be inserted and retrieved with correct values."""
    from backend.database.models import AvailabilityCandidate

    row = AvailabilityCandidate(
        frequency_start_mhz=119.975,
        frequency_end_mhz=120.025,
        required_bandwidth_mhz=0.05,
        activity="DETECTED",
        availability="OCCUPIED",
        ml_activity=True,
        ml_activity_probability=0.65,
        detector_activity=True,
        confidence="High",
        label_type="INFERRED_RF_ACTIVITY",
        recommendation_status="review_required",
        predicted_availability_probability=0.65,
        signal_power_dbm=-50.0,
        noise_floor_dbm=None,     # not fabricated
        snr_db=None,              # not fabricated
        ood_status="0",
        threshold_applied=0.435,
        model_version="rf-signal-activity-v2",
        data_source="RF Signal Data — real measured + derived + inferred",
        generated_at=datetime.now(timezone.utc),
    )
    db_session.add(row)
    db_session.commit()
    db_session.refresh(row)

    assert row.candidate_id is not None
    assert row.availability == "OCCUPIED"
    assert row.ml_activity_probability == 0.65
    assert row.noise_floor_dbm is None    # must remain NULL
    assert row.snr_db is None             # must remain NULL
    assert row.label_type == "INFERRED_RF_ACTIVITY"


def test_nullable_fields_are_null(app, db_session):
    """Optional fields left unset should be NULL."""
    from backend.database.models import AvailabilityCandidate

    row = AvailabilityCandidate(
        frequency_start_mhz=90.0,
        frequency_end_mhz=90.05,
        availability="AVAILABLE",
        activity="NOT_DETECTED",
        ood_status="0",
        generated_at=datetime.now(timezone.utc),
    )
    db_session.add(row)
    db_session.commit()
    db_session.refresh(row)

    assert row.noise_floor_dbm is None
    assert row.snr_db          is None
    assert row.region          is None
    assert row.state           is None


def test_multiple_rows_retrievable(app, db_session):
    """Multiple rows can be inserted and retrieved."""
    from backend.database.models import AvailabilityCandidate

    rows = [
        AvailabilityCandidate(
            frequency_start_mhz=float(f),
            frequency_end_mhz=float(f) + 0.05,
            availability="UNCERTAIN",
            ood_status="1",
            generated_at=datetime.now(timezone.utc),
        )
        for f in [70, 90, 100]
    ]
    db_session.add_all(rows)
    db_session.commit()

    results = db_session.query(AvailabilityCandidate).all()
    assert len(results) >= 3


def test_ood_status_stored_as_string(app, db_session):
    """ood_status is stored as '0' or '1' (string format)."""
    from backend.database.models import AvailabilityCandidate

    row = AvailabilityCandidate(
        frequency_start_mhz=140.0,
        frequency_end_mhz=140.05,
        availability="UNCERTAIN",
        ood_status="1",
        generated_at=datetime.now(timezone.utc),
    )
    db_session.add(row)
    db_session.commit()
    db_session.refresh(row)

    assert row.ood_status == "1"
    assert isinstance(row.ood_status, str)
