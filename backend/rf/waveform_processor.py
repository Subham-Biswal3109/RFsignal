"""Time-Domain Waveform Processor for Wire Watcher.

Processes baseband complex I/Q samples x[n] = I[n] + j*Q[n]:
  - Extracts I(t), Q(t), Magnitude |x[n]|, and Phase phi[n] = atan2(Q, I)
  - Calculates derived I/Q quality metrics:
      * RMS Magnitude: x_rms = sqrt( (1/N) * sum(|x[n]|^2) )
      * Peak Magnitude: x_peak = max(|x[n]|)
      * Crest Factor: CF = x_peak / x_rms
      * Crest Factor (dB): CF_dB = 20 * log10(CF)
      * Component RMS: I_rms, Q_rms
  - Calculates FFT window telemetry:
      * Sample count N, duration T = N / f_s, frequency resolution df = f_s / N
  - Formats arrays safely for frontend JSON consumption without generating fake data.
"""
from __future__ import annotations
import math
from typing import Dict, Any, Optional, List, Tuple
import numpy as np


def process_time_domain_waveform(
    iq_samples: Optional[np.ndarray],
    sample_rate_mhz: float = 2.048,
    center_freq_mhz: float = 120.0,
    downsample_target: int = 500,
) -> Dict[str, Any]:
    """Process complex I/Q samples into time-domain waveform series and quality metrics."""
    if iq_samples is None or len(iq_samples) == 0:
        return {
            "iq_available": 0,
            "status": "UNAVAILABLE",
            "reason": "This RF observation does not contain I/Q samples.",
            "metrics": None,
            "waveform_series": None,
        }

    # Ensure complex numpy array
    samples = np.asarray(iq_samples, dtype=np.complex128)
    n_samples = len(samples)

    if n_samples == 0:
        return {
            "iq_available": 0,
            "status": "UNAVAILABLE",
            "reason": "I/Q sample buffer is empty.",
            "metrics": None,
            "waveform_series": None,
        }

    # Extract components
    i_series = np.real(samples)
    q_series = np.imag(samples)
    mag_series = np.abs(samples)
    phase_series = np.angle(samples)  # radians (-pi to pi)

    # Derived I/Q Metrics
    i_rms = float(np.sqrt(np.mean(i_series ** 2)))
    q_rms = float(np.sqrt(np.mean(q_series ** 2)))
    mag_rms = float(np.sqrt(np.mean(mag_series ** 2)))
    mag_peak = float(np.max(mag_series))

    if mag_rms > 1e-12:
        crest_factor = mag_peak / mag_rms
        crest_factor_db = 20.0 * math.log10(crest_factor)
    else:
        crest_factor = 1.0
        crest_factor_db = 0.0

    # FFT Telemetry
    sample_rate_hz = sample_rate_mhz * 1e6
    duration_sec = n_samples / sample_rate_hz if sample_rate_hz > 0 else 0.0
    freq_resolution_hz = sample_rate_hz / n_samples if n_samples > 0 else 0.0

    metrics = {
        "num_samples": n_samples,
        "sample_rate_mhz": round(sample_rate_mhz, 4),
        "center_freq_mhz": round(center_freq_mhz, 3),
        "duration_us": round(duration_sec * 1e6, 2),
        "frequency_resolution_khz": round(freq_resolution_hz / 1e3, 3),
        "i_rms": round(i_rms, 6),
        "q_rms": round(q_rms, 6),
        "magnitude_rms": round(mag_rms, 6),
        "peak_magnitude": round(mag_peak, 6),
        "crest_factor_linear": round(crest_factor, 4),
        "crest_factor_db": round(crest_factor_db, 2),
    }

    # Time axis in microseconds
    t_us = np.arange(n_samples) / (sample_rate_mhz * 1e6) * 1e6

    # Downsample if sample count exceeds target for clean rendering
    step = max(1, n_samples // downsample_target)
    indices = np.arange(0, n_samples, step)

    waveform_series = {
        "time_us": [round(float(t), 2) for t in t_us[indices]],
        "i": [round(float(val), 6) for val in i_series[indices]],
        "q": [round(float(val), 6) for val in q_series[indices]],
        "magnitude": [round(float(val), 6) for val in mag_series[indices]],
        "phase": [round(float(val), 4) for val in phase_series[indices]],
    }

    return {
        "iq_available": 1,
        "status": "AVAILABLE",
        "reason": "Derived from complex I/Q baseband samples",
        "metrics": metrics,
        "waveform_series": waveform_series,
        "provenance": "DERIVED",
    }
