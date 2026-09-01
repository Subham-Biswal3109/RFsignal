"""Simulated RF signal source for the spectrum analysis endpoint.

This module generates a synthetic baseband signal so that
POST /api/spectrum/analyze has something to work with when no hardware SDR
is attached.  It clearly labels its output as "Simulated RF" — not real
measured data.

Design constraints
──────────────────
• No I/Q data is fabricated for the ML model; iq_available is always 0 for
  simulated signals.
• Signal strength is a simulated value used only for the RF detector and
  spectrum visualisation; it is NOT passed as an ML predictor.
"""
from __future__ import annotations
import numpy as np


class SimulatedRFSource:
    """Generate a simple baseband-equivalent sine wave + AWGN.

    Parameters
    ----------
    center_freq_mhz:
        Centre frequency of the simulated channel (MHz).
    bandwidth_mhz:
        Simulated channel bandwidth (MHz) — used to set the sample rate.
    signal_strength_dbm:
        Desired received signal power in dBm.
    noise_floor_dbm:
        Ambient noise floor in dBm.
    num_samples:
        Length of the generated time-domain signal.
    """

    _SOURCE_NAME = "Simulated RF (no hardware SDR)"

    def __init__(
        self,
        center_freq_mhz: float = 1800.0,
        bandwidth_mhz: float = 10.0,
        signal_strength_dbm: float = -75.0,
        noise_floor_dbm: float = -100.0,
        num_samples: int = 2048,
    ) -> None:
        self.center_freq_mhz = float(center_freq_mhz)
        self.bandwidth_mhz = float(bandwidth_mhz)
        self.signal_strength_dbm = float(signal_strength_dbm)
        self.noise_floor_dbm = float(noise_floor_dbm)
        self.num_samples = int(num_samples)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _dbm_to_linear(dbm: float) -> float:
        """Convert dBm to linear power (milliwatts)."""
        return 10 ** (dbm / 10.0)

    def _sample_rate_mhz(self) -> float:
        """Nyquist-safe sample rate: 2× the bandwidth."""
        return max(self.bandwidth_mhz * 2.0, 1.0)

    # ── public API ────────────────────────────────────────────────────────────

    def get_signal(self) -> tuple[np.ndarray, float]:
        """Return a complex baseband signal and the sample rate in MHz.

        Returns
        -------
        rx_signal : ndarray of complex128
            Complex baseband samples.
        sample_rate_mhz : float
            Sample rate (MHz).
        """
        sample_rate = self._sample_rate_mhz()
        t = np.arange(self.num_samples) / (sample_rate * 1e6)

        # Tone at ¼ of the sample rate (arbitrary carrier within band).
        f_tone = sample_rate * 1e6 / 4.0
        signal_power_mw = self._dbm_to_linear(self.signal_strength_dbm)
        noise_power_mw = self._dbm_to_linear(self.noise_floor_dbm)

        signal = np.sqrt(signal_power_mw) * np.exp(2j * np.pi * f_tone * t)
        noise = (
            np.sqrt(noise_power_mw / 2)
            * (np.random.randn(self.num_samples) + 1j * np.random.randn(self.num_samples))
        )
        return signal + noise, sample_rate

    def get_source_name(self) -> str:  # pragma: no cover
        return self._SOURCE_NAME
