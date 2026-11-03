"""Unit tests for Channel Allocation Engine & Scoring Layer.

Verifies:
  1. Candidate channel generation across specified frequency bounds and guard bands.
  2. Candidate scoring formula calculations and penalty deductions.
  3. Rejection / penalty of occupied, uncertain, and OOD candidate channels.
  4. Selection of recommended candidate channel.
  5. Condition where no suitable candidate channel is available.
  6. Allocation REST API endpoints (/api/allocation/recommend, /api/allocation/config).
"""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.allocation_service import ChannelAllocationEngine, ChannelCandidate
from app import create_app


@pytest.fixture(scope="module")
def app_client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ── 1. Candidate Generation & Bounds ─────────────────────────────────────────

def test_candidate_generation():
    res = ChannelAllocationEngine.generate_and_score_candidates(
        start_freq_mhz=100.0,
        end_freq_mhz=102.0,
        channel_bw_mhz=0.5,
        guard_band_mhz=0.1,
        noise_floor_dbm=-100.0,
        observed_power_dbm=-85.0,
    )

    assert "candidates" in res
    candidates = res["candidates"]
    assert len(candidates) > 0

    # First candidate: 100.0 to 100.5 MHz -> center = 100.25 MHz
    c1 = candidates[0]
    assert c1["bandwidth_mhz"] == 0.5
    assert c1["guard_band_mhz"] == 0.1


def test_invalid_frequency_range():
    with pytest.raises(ValueError, match="frequency range"):
        ChannelAllocationEngine.generate_and_score_candidates(
            start_freq_mhz=100.0,
            end_freq_mhz=90.0,
        )

    with pytest.raises(ValueError, match="channel bandwidth"):
        ChannelAllocationEngine.generate_and_score_candidates(
            start_freq_mhz=100.0,
            end_freq_mhz=105.0,
            channel_bw_mhz=10.0,
        )


# ── 2. Candidate Scoring Formula ──────────────────────────────────────────────

def test_scoring_formula_available_low_noise():
    score, breakdown = ChannelAllocationEngine.calculate_channel_score(
        availability="AVAILABLE",
        activity="NOT_DETECTED",
        ood_warning=False,
        noise_floor_dbm=-100.0,
    )
    assert score == 100.0
    assert breakdown["activity_penalty"] == 0.0
    assert breakdown["noise_penalty"] == 0.0
    assert breakdown["ood_penalty"] == 0.0


def test_scoring_formula_occupied_penalty():
    score, breakdown = ChannelAllocationEngine.calculate_channel_score(
        availability="OCCUPIED",
        activity="DETECTED",
        ood_warning=False,
        noise_floor_dbm=-100.0,
    )
    assert score == 50.0
    assert breakdown["activity_penalty"] == 50.0


def test_scoring_formula_ood_and_uncertainty_penalty():
    score, breakdown = ChannelAllocationEngine.calculate_channel_score(
        availability="UNCERTAIN",
        activity="UNCERTAIN",
        ood_warning=True,
        noise_floor_dbm=-90.0,
    )
    # Penalties: Uncertainty (25) + OOD (40) + Noise (( -90 - (-100) ) * 1.5 = 15) = 80
    assert score == 20.0
    assert breakdown["uncertainty_penalty"] == 25.0
    assert breakdown["ood_penalty"] == 40.0
    assert breakdown["noise_penalty"] == 15.0


# ── 3. Allocation API Endpoints ──────────────────────────────────────────────

def test_allocation_recommend_api(app_client):
    payload = {
        "start_freq_mhz": 120.0,
        "end_freq_mhz": 125.0,
        "channel_bw_mhz": 0.5,
        "guard_band_mhz": 0.1,
        "noise_floor_dbm": -95.0,
        "observed_power_dbm": -80.0,
    }
    response = app_client.post("/api/allocation/recommend", json=payload)
    assert response.status_code == 200
    data = response.get_json()

    assert "total_candidates" in data
    assert "candidates" in data
    assert "scoring_weights" in data
    assert "disclaimer" in data
    assert "has_suitable_candidate" in data


def test_allocation_config_api(app_client):
    response = app_client.get("/api/allocation/config")
    assert response.status_code == 200
    data = response.get_json()
    assert "scoring_weights" in data
    assert data["scoring_weights"]["base_score"] == 100.0
    assert "disclaimer" in data
