"""Shared RF feature extraction used by both training and inference.

The dataset's I/Q field contains 100 complex samples when present, but does not
provide a verified sampling rate, receiver gain, or calibration reference.
Therefore spectral features are dimensionless/normalized summaries only; no
absolute FFT frequency axis is claimed.
"""
from __future__ import annotations
import re
import numpy as np

FEATURE_COLUMNS = [
    "frequency_mhz", "bandwidth_khz", "signal_strength_dbm", "iq_available",
    "iq_rms_magnitude", "iq_magnitude_variance", "iq_peak_magnitude",
    "iq_crest_factor", "iq_p10", "iq_p50", "iq_p90",
    "iq_phase_concentration", "iq_spectral_entropy", "iq_spectral_peak_ratio",
]
ML_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c != "signal_strength_dbm"]

_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_PAIR_RE = re.compile(r"\((" + _NUM + r")(" + r"[+-](?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?" + r")j\)")

def parse_iq(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple)):
        vals = value
        arr = np.asarray(vals, dtype=np.complex128)
    else:
        pairs = _PAIR_RE.findall(str(value))
        if not pairs:
            return None
        arr = np.asarray([float(r) + 1j * float(i) for r, i in pairs], dtype=np.complex128)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr.real)) or not np.all(np.isfinite(arr.imag)):
        return None
    return arr

def iq_features(value):
    x = parse_iq(value)
    if x is None:
        return {"iq_available": 0, **{k: np.nan for k in FEATURE_COLUMNS if k.startswith("iq_") and k != "iq_available"}}
    mag = np.abs(x)
    phase = np.angle(x)
    rms = float(np.sqrt(np.mean(mag ** 2)))
    var = float(np.var(mag))
    peak = float(np.max(mag))
    cf = float(peak / rms) if rms > 0 else np.nan
    # Circular concentration: |mean(exp(j*phase))|, bounded [0,1].
    pc = float(np.abs(np.mean(np.exp(1j * phase))))
    spec = np.abs(np.fft.fft(x)) ** 2
    total = float(np.sum(spec))
    if total > 0:
        p = spec / total
        entropy = float(-np.sum(p[p > 0] * np.log2(p[p > 0])) / np.log2(len(p)))
        peak_ratio = float(np.max(spec) / total)
    else:
        entropy = np.nan
        peak_ratio = np.nan
    return {
        "iq_available": 1,
        "iq_rms_magnitude": rms,
        "iq_magnitude_variance": var,
        "iq_peak_magnitude": peak,
        "iq_crest_factor": cf,
        "iq_p10": float(np.percentile(mag, 10)),
        "iq_p50": float(np.percentile(mag, 50)),
        "iq_p90": float(np.percentile(mag, 90)),
        "iq_phase_concentration": pc,
        "iq_spectral_entropy": entropy,
        "iq_spectral_peak_ratio": peak_ratio,
    }

def build_features(row):
    out = {
        "frequency_mhz": float(row["Frequency"]) / 1e6,
        "bandwidth_khz": float(row["Bandwidth"]) / 1e3,
        "signal_strength_dbm": float(row["Signal Strength"]),
    }
    out.update(iq_features(row.get("I/Q Data")))
    return out
