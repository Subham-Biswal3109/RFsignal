"""Noise floor estimator for the spectrum analyser visualisation.

Uses the median of the lower power decile as a robust noise-floor estimate.
This is a heuristic for the simulated/visualisation path only; it is NOT
used as an ML model input.
"""
from __future__ import annotations
import numpy as np


def estimate_noise_floor(psd_dbm: np.ndarray, percentile: float = 10.0) -> float:
    """Estimate the noise floor from a PSD array.

    Takes the ``percentile``-th percentile of the PSD as a conservative
    estimate of the ambient noise level.

    Parameters
    ----------
    psd_dbm:
        Power spectral density in dBm.
    percentile:
        Which percentile to use (default 10).

    Returns
    -------
    float
        Estimated noise floor in dBm.
    """
    if psd_dbm is None or len(psd_dbm) == 0:
        return -100.0
    return float(np.percentile(psd_dbm, percentile))
