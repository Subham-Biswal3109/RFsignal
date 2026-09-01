"""Wire Watcher application configuration.

All path and runtime settings are read from environment variables with safe
defaults.  The database URL uses SQLite because mysql.connector and psycopg2
are not installed in this environment.  The SQLAlchemy abstraction means
PostgreSQL or MySQL can be plugged in later by changing DB_URL alone.
"""
from __future__ import annotations
import os
from pathlib import Path

# ── repository root ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── ML artefact paths ─────────────────────────────────────────────────────────
ML_DIR = BASE_DIR / "ml"
MODEL_PATH = str(ML_DIR / "artifacts" / "wire_watcher_model.pkl")
METADATA_PATH = str(ML_DIR / "artifacts" / "model_metadata.json")

# ── database ──────────────────────────────────────────────────────────────────
# SQLite by default; override with DB_URL env-var for Postgres/MySQL later.
_DB_FILE = BASE_DIR / "wire_watcher.db"
DB_URL = os.getenv("DB_URL", f"sqlite:///{_DB_FILE}")

# ── Flask ─────────────────────────────────────────────────────────────────────
DEBUG = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
SECRET_KEY = os.getenv("SECRET_KEY", "wire-watcher-dev-key-change-in-production")
PORT = int(os.getenv("PORT", "5000"))
