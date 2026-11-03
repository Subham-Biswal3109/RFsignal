"""Pytest fixtures shared across the test suite.

Provides:
  - app      → Flask test application (isolated SQLite in-memory DB)
  - client   → Flask test client
  - bundle   → loaded ML model bundle (session-scoped)
  - metadata → model metadata dict (session-scoped)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest

# ── ensure project root is on sys.path ───────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_ML_DIR = ROOT / "ml"
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))


@pytest.fixture(scope="session")
def bundle():
    """Load the ML model bundle once per test session."""
    import joblib
    return joblib.load(ROOT / "ml" / "artifacts" / "wire_watcher_model.pkl")


@pytest.fixture(scope="session")
def metadata():
    """Load model_metadata.json once per test session."""
    return json.loads(
        (ROOT / "ml" / "artifacts" / "model_metadata.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="function")
def app():
    """Create a Flask test application with an isolated in-memory SQLite DB.

    Strategy
    ────────
    1. Patch backend.database.connection with a fresh in-memory engine.
    2. Create all tables on the new engine.
    3. Yield the Flask app.
    4. Drop all tables after the test.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    # Fresh in-memory engine for this test function.
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    # Patch the module-level engine and session factory BEFORE importing app.
    import backend.database.connection as conn
    original_engine  = conn.engine
    original_session = conn.SessionLocal

    conn.engine       = test_engine
    conn.SessionLocal = TestSession

    # Also patch the Base so create_all() targets the test engine.
    from backend.database.connection import Base
    import backend.database.models  # noqa: F401 — registers ORM classes
    Base.metadata.create_all(bind=test_engine)

    from app import create_app
    application = create_app()
    application.config["TESTING"] = True

    yield application

    # Teardown: drop tables and restore original engine.
    Base.metadata.drop_all(bind=test_engine)
    conn.engine       = original_engine
    conn.SessionLocal = original_session


@pytest.fixture(scope="function")
def client(app):
    """Return a Flask test client."""
    return app.test_client()
