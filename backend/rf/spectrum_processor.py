"""FFT / Power Spectral Density computation for the spectrum analyser.

Accepts a complex baseband signal and returns frequency bins (MHz) plus
power in dBm.  These are SIMULATED values used only for visualisation and
the spectrum-analysis endpoint; they are NOT used as ML model inputs.
"""
from __future__ import annotations
import numpy as np


def compute_fft_psd(
    rx_signal: np.ndarray,
    sample_rate_mhz: float,
    center_freq_mhz: float,
    nfft: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one-sided FFT and convert to a dBm-referenced PSD.

    Parameters
    ----------
    rx_signal:
        Complex baseband samples.
    sample_rate_mhz:
        Sample rate in MHz.
    center_freq_mhz:
        RF centre frequency in MHz (used to shift the frequency axis).
    nfft:
        FFT size.  Defaults to ``len(rx_signal)``.

    Returns
    -------
    freqs_mhz : ndarray
        Frequency axis in MHz (absolute RF frequency).
    psd_dbm : ndarray
        Power spectral density in dBm per FFT bin.
    """
    n = len(rx_signal)
    nfft = nfft or n

    window = np.hanning(n)
    windowed = rx_signal * window

    spectrum = np.fft.fftshift(np.fft.fft(windowed, n=nfft))
    power = np.abs(spectrum) ** 2

    # Normalise to dBm; add small floor to avoid log10(0).
    power_normalised = power / (nfft ** 2)
    psd_dbm = 10.0 * np.log10(np.maximum(power_normalised, 1e-20)) + 30.0

    freqs_hz = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / (sample_rate_mhz * 1e6)))
    freqs_mhz = center_freq_mhz + freqs_hz / 1e6

    return freqs_mhz, psd_dbm
