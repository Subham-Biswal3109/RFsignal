"""Wire Watcher — Flask application entry point.

Usage
─────
    python app.py                    # development server (port 5000)
    flask run --port 5000            # via Flask CLI
    gunicorn "app:create_app()"      # production

The database (SQLite wire_watcher.db) is initialised automatically on first
run using CREATE TABLE IF NOT EXISTS — existing data is never dropped.
"""
from __future__ import annotations
import os
from flask import Flask
from flask_cors import CORS

from backend.config import DEBUG, PORT, SECRET_KEY
from backend.database.init_db import init_db
from backend.api.routes import api_bp


def create_app() -> Flask:
    """Application factory (allows pytest to create isolated test instances)."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = SECRET_KEY

    # Allow cross-origin requests from the React dev server (localhost:5173)
    # and production bundle.
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register the main API blueprint.
    app.register_blueprint(api_bp)

    # Initialise the database (safe: only creates tables that do not exist).
    with app.app_context():
        try:
            init_db()
        except Exception as exc:  # pragma: no cover
            print(f"[app.py] WARNING: DB initialisation failed: {exc}")

    return app


# ── development entry-point ───────────────────────────────────────────────────
if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=PORT, debug=DEBUG)
