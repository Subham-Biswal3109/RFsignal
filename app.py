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
from flask import Flask, jsonify
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
            from backend.rf.event_detector import derive_and_ingest_events_from_dataset
            derive_and_ingest_events_from_dataset(force_reingest=False)
        except Exception as exc:  # pragma: no cover
            print(f"[app.py] WARNING: DB initialisation or event ingestion failed: {exc}")

    @app.errorhandler(400)
    def bad_request_error(error):
        return jsonify({"error": "Validation failed", "message": str(error), "status": 400}), 400

    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({"error": "Resource not found", "message": "The requested API endpoint does not exist", "status": 404}), 404

    @app.errorhandler(405)
    def method_not_allowed_error(error):
        return jsonify({"error": "Method not allowed", "message": str(error), "status": 405}), 405

    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({"error": "Internal server error", "message": str(error), "status": 500}), 500

    return app


# ── development entry-point ───────────────────────────────────────────────────
if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=PORT, debug=DEBUG)
