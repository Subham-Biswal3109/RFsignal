"""Simple peak detector for the spectrum-analysis visualisation path.

Finds spectral peaks that exceed the noise floor by a configurable margin.
This is used ONLY for the /api/spectrum/analyze visualisation endpoint.
It is NOT used as an ML model input and does NOT determine availability.
"""
from __future__ import annotations
import numpy as np
from typing import Any


def detect_peaks(
    freqs_mhz: np.ndarray,
    psd_dbm: np.ndarray,
    noise_floor_dbm: float,
    min_snr_db: float = 6.0,
    min_separation_mhz: float = 0.1,
) -> list[dict[str, Any]]:
    """Detect spectral peaks above the noise floor.

    Parameters
    ----------
    freqs_mhz:
        Frequency axis in MHz.
    psd_dbm:
        Power spectral density in dBm.
    noise_floor_dbm:
        Estimated noise floor (dBm).
    min_snr_db:
        Minimum SNR (dB) above the noise floor for a peak to be reported.
    min_separation_mhz:
        Minimum spacing (MHz) between reported peaks.

    Returns
    -------
    List of dicts with keys: frequency_mhz, power_dbm, bandwidth_mhz, snr_db.
    """
    if len(psd_dbm) == 0:
        return []

    threshold_dbm = noise_floor_dbm + min_snr_db
    above = psd_dbm >= threshold_dbm

    peaks: list[dict[str, Any]] = []
    freq_arr = np.asarray(freqs_mhz, dtype=float)
    psd_arr  = np.asarray(psd_dbm,   dtype=float)

    # Find local maxima above threshold using a simple scan.
    n = len(psd_arr)
    for i in range(1, n - 1):
        if not above[i]:
            continue
        if psd_arr[i] >= psd_arr[i - 1] and psd_arr[i] >= psd_arr[i + 1]:
            f = float(freq_arr[i])
            p = float(psd_arr[i])
            snr = p - noise_floor_dbm
            # Suppress peaks too close to an already-found peak.
            if any(abs(f - pk["frequency_mhz"]) < min_separation_mhz for pk in peaks):
                continue
            # Estimate 3-dB bandwidth.
            half_power = p - 3.0
            left_i, right_i = i, i
            while left_i > 0  and psd_arr[left_i - 1] >= half_power:
                left_i -= 1
            while right_i < n - 1 and psd_arr[right_i + 1] >= half_power:
                right_i += 1
            bw = float(freq_arr[right_i] - freq_arr[left_i])
            peaks.append({
                "frequency_mhz": round(f, 4),
                "power_dbm": round(p, 2),
                "bandwidth_mhz": round(max(bw, 0.001), 4),
                "snr_db": round(snr, 2),
            })

    # Sort by power, descending.
    peaks.sort(key=lambda x: x["power_dbm"], reverse=True)
    return peaks


def get_occupied_regions(peaks: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Convert detected peaks into occupied frequency regions.

    Each region spans ±(bandwidth/2) around the peak frequency.
    """
    regions: list[dict[str, float]] = []
    for pk in peaks:
        half = pk["bandwidth_mhz"] / 2.0
        regions.append({
            "start_mhz": round(pk["frequency_mhz"] - half, 4),
            "end_mhz":   round(pk["frequency_mhz"] + half, 4),
        })
    return regions
