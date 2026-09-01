"""Database initialisation helper.

Call ``init_db()`` once at application startup.  Uses CREATE TABLE IF NOT
EXISTS semantics (via SQLAlchemy's ``checkfirst=True``) so existing tables
and data are never dropped.
"""
from __future__ import annotations
from backend.database.connection import engine, Base
import backend.database.models as _models  # noqa: F401 — side-effect: registers ORM classes


def init_db() -> None:
    """Create all tables that do not already exist.

    Safe to call on every application start — existing tables and rows are
    preserved.
    """
    Base.metadata.create_all(bind=engine, checkfirst=True)


if __name__ == "__main__":
    init_db()
    print("Database initialised (SQLite tables created if missing).")
