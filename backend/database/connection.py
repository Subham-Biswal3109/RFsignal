"""SQLAlchemy engine + session factory.

Uses the DB_URL from config.  SQLite check_same_thread=False is required for
Flask's multi-threaded request handling.  For other databases that kwarg is
simply ignored.
"""
from __future__ import annotations
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from backend.config import DB_URL

# connect_args is only meaningful for SQLite; ignored by other dialects.
_connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}

engine = create_engine(
    DB_URL,
    connect_args=_connect_args,
    # Pool settings safe for SQLite and reasonable for Postgres/MySQL.
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


def get_db():
    """Yield a SQLAlchemy session and ensure it is closed on exit.

    Usage (Flask route)::

        db_gen = get_db()
        db = next(db_gen)
        try:
            ...
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
