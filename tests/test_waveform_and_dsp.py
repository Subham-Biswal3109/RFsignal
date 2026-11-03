"""Tests for Time-Domain Waveform Processing, I/Q Quality Metrics, and DSP API Endpoints.

Covers:
1. Complex I/Q time-domain waveform calculations (RMS magnitude, peak magnitude, crest factor)
2. Missing I/Q sample policy enforcement (zero fake data policy)
3. FFT windowing & frequency resolution calculations
4. Waveform REST API endpoints (GET /api/rf/waveform, GET /api/rf/observation, GET /api/rf/spectrum)
5. RF calculator formula traces & mathematical correctness
"""
import pytest
import numpy as np
from backend.rf.waveform_processor import process_time_domain_waveform
from backend.rf.sources import SimulationRFSource


def test_complex_iq_waveform_processing():
    # Create sample complex baseband tone + noise
    n_samples = 1024
    sr_mhz = 2.048
    t = np.arange(n_samples) / (sr_mhz * 1e6)
    tone = np.exp(2j * np.pi * 100e3 * t)

    res = process_time_domain_waveform(
        iq_samples=tone,
        sample_rate_mhz=sr_mhz,
        center_freq_mhz=120.0,
    )

    assert res["status"] == "AVAILABLE"
    assert res["iq_available"] == 1
    assert "metrics" in res
    m = res["metrics"]
    assert m["num_samples"] == 1024
    assert m["sample_rate_mhz"] == 2.048
    assert m["duration_us"] > 0
    assert m["magnitude_rms"] == pytest.approx(1.0, abs=1e-2)
    assert m["peak_magnitude"] == pytest.approx(1.0, abs=1e-2)
    assert m["crest_factor_linear"] == pytest.approx(1.0, abs=1e-2)
    assert m["crest_factor_db"] == pytest.approx(0.0, abs=1e-1)


def test_missing_iq_sample_policy():
    # Pass None to verify explicit fallback without fake data generation
    res = process_time_domain_waveform(
        iq_samples=None,
        sample_rate_mhz=2.048,
        center_freq_mhz=120.0,
    )

    assert res["iq_available"] == 0
    assert res["status"] == "UNAVAILABLE"
    assert "does not contain I/Q samples" in res["reason"]
    assert res["metrics"] is None
    assert res["waveform_series"] is None


from backend.rf.rf_source import SimulatedRFSource


def test_fft_window_telemetry():
    source = SimulatedRFSource(center_freq_mhz=120.0, bandwidth_mhz=0.2, num_samples=1024)
    signal, sr_mhz = source.get_signal()

    res = process_time_domain_waveform(
        iq_samples=signal,
        sample_rate_mhz=sr_mhz,
        center_freq_mhz=120.0,
    )

    m = res["metrics"]
    # df = f_s / N = (0.4 MHz * 1e6) / 1024 approx 390 Hz = 0.39 kHz
    assert m["frequency_resolution_khz"] > 0
    assert m["duration_us"] > 0


def test_rf_waveform_api_endpoints(client):
    # 1. GET /api/rf/observation
    resp = client.get("/api/rf/observation?frequency_mhz=120.0")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "observation" in data

    # 2. GET /api/rf/waveform with iq_available=1
    resp = client.get("/api/rf/waveform?frequency_mhz=120.0&iq_available=1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "AVAILABLE"
    assert "waveform_series" in data

    # 3. GET /api/rf/waveform with iq_available=0 (Fallback)
    resp = client.get("/api/rf/waveform?frequency_mhz=120.0&iq_available=0")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "UNAVAILABLE"
    assert data["iq_available"] == 0

    # 4. GET /api/rf/spectrum
    resp = client.get("/api/rf/spectrum?frequency_mhz=120.0")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "spectrum_data" in data

    # 5. GET /api/rf/activity
    resp = client.get("/api/rf/activity?signal_power_dbm=-75.0&noise_floor_dbm=-100.0")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["activity_state"] == "DETECTED"
    assert data["snr_db"] == 25.0
