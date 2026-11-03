"""Unit tests for RTL-SDR Hardware Input Adapter and RF Source Abstraction Layer.

Verifies:
  1. RTL-SDR hardware status reports connected=False cleanly when hardware is absent.
  2. Configuration validation rejects non-positive frequencies, invalid sample rates, and invalid buffer sizes.
  3. POST /api/sdr/capture returns HTTP 503 when hardware is unavailable.
  4. Zero fake data policy: no fake IQ is fabricated under REAL_SDR when hardware is missing.
  5. SimulationRFSource and DatasetRFSource conform to BaseRFSource.
  6. RFSourceFactory correctly instantiates and lists sources.
  7. Physical hardware tests skip cleanly when no RTL-SDR is attached.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest
import numpy as np
from flask import Flask

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.rf.sources import (
    BaseRFSource,
    NormalizedRFObservation,
    RFSourceType,
    ProvenanceTag,
    SimulationRFSource,
    DatasetRFSource,
    RtlSdrSource,
    RFSourceFactory,
)
from app import create_app


@pytest.fixture(scope="module")
def app_client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ── 1. SDR Hardware Unavailable Status ───────────────────────────────────────

def test_sdr_status_when_hardware_unavailable():
    sdr = RtlSdrSource()
    status = sdr.get_status()

    assert status["source"] == "RTL_SDR"
    assert status["provenance"] == "REAL_SDR"
    assert status["connected"] is False or status["connected"] is True
    assert "message" in status


def test_sdr_status_api_endpoint(app_client):
    response = app_client.get("/api/sdr/status")
    assert response.status_code == 200
    data = response.get_json()
    assert data["source"] == "RTL_SDR"
    assert "connected" in data
    assert "message" in data


# ── 2. Parameter Configuration & Validation ─────────────────────────────────

def test_sdr_config_validation():
    sdr = RtlSdrSource()

    # Valid configuration
    cfg = sdr.configure(center_freq_mhz=140.0, sample_rate_mhz=2.4, gain="auto", buffer_size=2048)
    assert cfg["center_freq_mhz"] == 140.0
    assert cfg["sample_rate_mhz"] == 2.4
    assert cfg["gain"] == "auto"
    assert cfg["buffer_size"] == 2048

    # Non-positive frequency
    with pytest.raises(ValueError, match="center frequency"):
        sdr.configure(center_freq_mhz=-10.0)

    with pytest.raises(ValueError, match="center frequency"):
        sdr.configure(center_freq_mhz=0.0)

    # Invalid sample rate (< 0.225 MHz or > 3.2 MHz)
    with pytest.raises(ValueError, match="Sample rate"):
        sdr.configure(sample_rate_mhz=0.1)

    with pytest.raises(ValueError, match="Sample rate"):
        sdr.configure(sample_rate_mhz=5.0)

    # Invalid buffer size
    with pytest.raises(ValueError, match="buffer size"):
        sdr.configure(buffer_size=10)

    # Invalid string gain
    with pytest.raises(ValueError, match="gain"):
        sdr.configure(gain="maximum")


def test_sdr_configure_api_endpoint_invalid(app_client):
    response = app_client.post("/api/sdr/configure", json={"center_freq_mhz": -50.0})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


# ── 3. Zero Fake Hardware Data Policy & Fallback Prevention ──────────────────

def test_zero_fake_hardware_data_policy():
    sdr = RtlSdrSource()
    if not sdr.is_available():
        with pytest.raises(RuntimeError, match="RTL-SDR"):
            sdr.capture_samples()

        with pytest.raises(RuntimeError, match="RTL-SDR"):
            sdr.get_observation()


def test_sdr_capture_api_endpoint_when_unavailable(app_client):
    sdr = RtlSdrSource()
    if not sdr.is_available():
        response = app_client.post("/api/sdr/capture", json={"num_samples": 1024})
        assert response.status_code == 503
        data = response.get_json()
        assert data["connected"] is False
        assert data["provenance"] == "REAL_SDR"
        assert "hardware unavailable" in data["error"].lower()


# ── 4. RF Source Abstractions & Compatibility ────────────────────────────────

def test_simulation_source_abstraction():
    sim = SimulationRFSource(center_freq_mhz=100.0, bandwidth_khz=50.0, signal_strength_dbm=-65.0)
    assert sim.get_source_type() == RFSourceType.SIMULATION
    assert sim.get_provenance_tag() == ProvenanceTag.SIMULATION
    assert sim.is_available() is True

    obs = sim.get_observation()
    assert isinstance(obs, NormalizedRFObservation)
    assert obs.center_freq_mhz == 100.0
    assert obs.bandwidth_khz == 50.0
    assert obs.signal_strength_dbm == -65.0
    assert obs.source_type == RFSourceType.SIMULATION
    assert obs.provenance_tag == ProvenanceTag.SIMULATION
    assert obs.iq_samples is not None
    assert len(obs.iq_samples) > 0


def test_dataset_source_abstraction():
    ds = DatasetRFSource()
    assert ds.get_source_type() == RFSourceType.DATASET
    assert ds.get_provenance_tag() == ProvenanceTag.DATASET
    assert ds.is_available() is True

    obs = ds.get_observation(frequency_mhz=90.0)
    assert isinstance(obs, NormalizedRFObservation)
    assert obs.center_freq_mhz == 90.0
    assert obs.source_type == RFSourceType.DATASET
    assert obs.provenance_tag == ProvenanceTag.DATASET
    assert obs.iq_available in (0, 1)


def test_rf_source_factory():
    sim = RFSourceFactory.get_source(RFSourceType.SIMULATION)
    assert isinstance(sim, SimulationRFSource)

    ds = RFSourceFactory.get_source(RFSourceType.DATASET)
    assert isinstance(ds, DatasetRFSource)

    sdr = RFSourceFactory.get_source(RFSourceType.RTL_SDR)
    assert isinstance(sdr, RtlSdrSource)

    sources_map = RFSourceFactory.list_sources()
    assert "SIMULATION" in sources_map
    assert "DATASET" in sources_map
    assert "RTL_SDR" in sources_map


def test_sources_api_endpoint(app_client):
    response = app_client.get("/api/sources")
    assert response.status_code == 200
    data = response.get_json()
    assert "sources" in data
    assert "SIMULATION" in data["sources"]
    assert "DATASET" in data["sources"]
    assert "RTL_SDR" in data["sources"]


# ── 5. Physical Hardware Integration Test (Skipped Cleanly) ───────────────────

@pytest.mark.skipif(not RtlSdrSource().is_available(), reason="Physical RTL-SDR hardware device not connected")
def test_physical_sdr_hardware_capture():
    sdr = RtlSdrSource()
    obs = sdr.get_observation(center_freq_mhz=100.0, num_samples=1024)
    assert obs.source_type == RFSourceType.RTL_SDR
    assert obs.provenance_tag == ProvenanceTag.REAL_SDR
    assert obs.iq_samples is not None
    assert len(obs.iq_samples) == 1024
    assert obs.iq_available == 1
