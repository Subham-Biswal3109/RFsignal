"""RTL-SDR Hardware Adapter Implementation.

Provides an isolated physical RTL-SDR hardware input adapter.

Design & Safety Constraints
───────────────────────────
• Optional Library Dependency: Gracefully handles missing `pyrtlsdr` package or `librtlsdr` C library.
• Zero Fake Data Policy: If hardware is unavailable or not connected, NO fake IQ is fabricated,
  no simulation is substituted under REAL_SDR, and an explicit hardware unavailable status is returned.
• Input Validation: Rejects non-positive frequencies, invalid sample rates, NaN/Inf values, and
  invalid capture buffer sizes.
"""
from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union
import numpy as np

from backend.rf.sources.base import (
    BaseRFSource,
    NormalizedRFObservation,
    RFSourceType,
    ProvenanceTag,
)

# Optional import of pyrtlsdr
_HAS_PYRTLSDR = False
_RtlSdrClass = None

try:
    from rtlsdr import RtlSdr as _RtlSdrClass  # type: ignore
    _HAS_PYRTLSDR = True
except Exception:
    _HAS_PYRTLSDR = False
    _RtlSdrClass = None


def compute_iq_dsp_features(iq_samples: np.ndarray) -> Dict[str, Optional[float]]:
    """Compute normalized statistical and spectral features from complex baseband samples."""
    if iq_samples is None or len(iq_samples) == 0:
        return {}

    mags = np.abs(iq_samples)
    rms = float(np.sqrt(np.mean(mags ** 2)))
    var = float(np.var(mags))
    peak = float(np.max(mags))
    crest = float(peak / rms) if rms > 1e-12 else 1.0

    p10 = float(np.percentile(mags, 10))
    p50 = float(np.percentile(mags, 50))
    p90 = float(np.percentile(mags, 90))

    # Phase concentration (mean circular magnitude)
    phases = np.angle(iq_samples)
    phase_conc = float(np.abs(np.mean(np.exp(1j * phases))))

    # Spectral entropy & peak ratio
    n = len(iq_samples)
    fft_mag = np.abs(np.fft.fft(iq_samples)) ** 2
    total_power = float(np.sum(fft_mag))
    if total_power > 1e-12:
        p_norm = fft_mag / total_power
        # Normalized Shannon entropy
        p_nonzero = p_norm[p_norm > 1e-12]
        entropy = float(-np.sum(p_nonzero * np.log2(p_nonzero)) / np.log2(n))
        peak_ratio = float(np.max(fft_mag) / total_power)
    else:
        entropy = 1.0
        peak_ratio = 0.0

    return {
        "iq_rms_magnitude": rms,
        "iq_magnitude_variance": var,
        "iq_peak_magnitude": peak,
        "iq_crest_factor": crest,
        "iq_p10": p10,
        "iq_p50": p50,
        "iq_p90": p90,
        "iq_phase_concentration": phase_conc,
        "iq_spectral_entropy": entropy,
        "iq_spectral_peak_ratio": peak_ratio,
    }


class RtlSdrSource(BaseRFSource):
    """Isolated RTL-SDR Hardware Adapter."""

    def __init__(
        self,
        center_freq_mhz: float = 120.0,
        sample_rate_mhz: float = 2.048,
        gain: Union[float, str] = "auto",
        buffer_size: int = 1024,
    ) -> None:
        self.center_freq_mhz = 120.0
        self.sample_rate_mhz = 2.048
        self.gain = "auto"
        self.buffer_size = 1024
        self._device = None

        # Validate initial configurations
        self.configure(
            center_freq_mhz=center_freq_mhz,
            sample_rate_mhz=sample_rate_mhz,
            gain=gain,
            buffer_size=buffer_size,
        )

    def get_source_type(self) -> RFSourceType:
        return RFSourceType.RTL_SDR

    def get_provenance_tag(self) -> ProvenanceTag:
        return ProvenanceTag.REAL_SDR

    def configure(
        self,
        center_freq_mhz: float | None = None,
        sample_rate_mhz: float | None = None,
        gain: Union[float, str] | None = None,
        buffer_size: int | None = None,
    ) -> Dict[str, Any]:
        """Configure and validate RTL-SDR hardware parameters."""
        if center_freq_mhz is not None:
            if not math.isfinite(center_freq_mhz) or center_freq_mhz <= 0:
                raise ValueError(f"Invalid center frequency: {center_freq_mhz}. Must be finite and > 0.")
            self.center_freq_mhz = float(center_freq_mhz)

        if sample_rate_mhz is not None:
            if not math.isfinite(sample_rate_mhz) or sample_rate_mhz <= 0:
                raise ValueError(f"Invalid sample rate: {sample_rate_mhz}. Must be finite and > 0.")
            if sample_rate_mhz < 0.225 or sample_rate_mhz > 3.2:
                raise ValueError(f"Sample rate {sample_rate_mhz} MHz outside RTL-SDR supported range [0.225, 3.2] MHz.")
            self.sample_rate_mhz = float(sample_rate_mhz)

        if gain is not None:
            if isinstance(gain, str):
                if gain.lower() != "auto":
                    raise ValueError(f"Invalid string gain '{gain}'. Only 'auto' is supported.")
                self.gain = "auto"
            else:
                if not math.isfinite(gain):
                    raise ValueError(f"Invalid numeric gain {gain}. Must be finite.")
                self.gain = float(gain)

        if buffer_size is not None:
            if not isinstance(buffer_size, int) or buffer_size < 256 or buffer_size > 262144:
                raise ValueError(f"Invalid buffer size {buffer_size}. Must be an integer between 256 and 262144.")
            self.buffer_size = int(buffer_size)

        return {
            "center_freq_mhz": self.center_freq_mhz,
            "sample_rate_mhz": self.sample_rate_mhz,
            "gain": self.gain,
            "buffer_size": self.buffer_size,
        }

    def is_available(self) -> bool:
        """Probe for connected RTL-SDR hardware."""
        if not _HAS_PYRTLSDR or _RtlSdrClass is None:
            return False

        try:
            # Probe device by attempting to open and query device count or index
            sdr = _RtlSdrClass(device_index=0)
            sdr.close()
            return True
        except Exception:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Return explicit SDR status without pretending hardware is attached."""
        if not _HAS_PYRTLSDR:
            return {
                "source": self.get_source_type().value,
                "provenance": self.get_provenance_tag().value,
                "connected": False,
                "device": None,
                "center_frequency_mhz": self.center_freq_mhz,
                "sample_rate_mhz": self.sample_rate_mhz,
                "gain": self.gain,
                "buffer_size": self.buffer_size,
                "message": "pyrtlsdr library is not installed in the Python environment",
            }

        hardware_connected = self.is_available()
        if not hardware_connected:
            return {
                "source": self.get_source_type().value,
                "provenance": self.get_provenance_tag().value,
                "connected": False,
                "device": None,
                "center_frequency_mhz": self.center_freq_mhz,
                "sample_rate_mhz": self.sample_rate_mhz,
                "gain": self.gain,
                "buffer_size": self.buffer_size,
                "message": "No RTL-SDR hardware device detected on USB bus",
            }

        return {
            "source": self.get_source_type().value,
            "provenance": self.get_provenance_tag().value,
            "connected": True,
            "device": "Generic RTL2832U OEM SDR",
            "center_frequency_mhz": self.center_freq_mhz,
            "sample_rate_mhz": self.sample_rate_mhz,
            "gain": self.gain,
            "buffer_size": self.buffer_size,
            "message": "RTL-SDR hardware device detected and ready",
        }

    def capture_samples(self, num_samples: int | None = None) -> np.ndarray:
        """Capture a controlled finite IQ sample buffer from physical RTL-SDR hardware.

        Raises RuntimeError if hardware is absent. NEVER fabricates fake data.
        """
        if not _HAS_PYRTLSDR or _RtlSdrClass is None:
            raise RuntimeError("RTL-SDR capture failed: pyrtlsdr package is not installed.")

        n_samples = int(num_samples if num_samples is not None else self.buffer_size)
        if n_samples < 256 or n_samples > 262144:
            raise ValueError(f"Invalid capture num_samples: {n_samples}. Must be between 256 and 262144.")

        try:
            sdr = _RtlSdrClass(device_index=0)
            sdr.sample_rate = self.sample_rate_mhz * 1e6
            sdr.center_freq = self.center_freq_mhz * 1e6
            if self.gain == "auto":
                sdr.gain = "auto"
            else:
                sdr.gain = float(self.gain)

            iq_samples = sdr.read_samples(n_samples)
            sdr.close()
            return np.asarray(iq_samples, dtype=np.complex128)
        except Exception as exc:
            raise RuntimeError(f"RTL-SDR hardware capture failed: {exc}")

    def get_observation(
        self,
        center_freq_mhz: float | None = None,
        sample_rate_mhz: float | None = None,
        gain: Union[float, str] | None = None,
        num_samples: int | None = None,
        **kwargs,
    ) -> NormalizedRFObservation:
        """Capture one normalized observation from physical hardware.

        Raises RuntimeError if hardware is not connected.
        """
        if center_freq_mhz is not None or sample_rate_mhz is not None or gain is not None:
            self.configure(
                center_freq_mhz=center_freq_mhz,
                sample_rate_mhz=sample_rate_mhz,
                gain=gain,
            )

        iq_samples = self.capture_samples(num_samples=num_samples)
        mags_sq = np.abs(iq_samples) ** 2
        mean_power_linear = float(np.mean(mags_sq))
        signal_power_dbm = 10.0 * math.log10(max(mean_power_linear, 1e-15)) + 30.0

        iq_features = compute_iq_dsp_features(iq_samples)
        bw_khz = (self.sample_rate_mhz * 1000.0) / 2.0  # Approximate channel bandwidth

        return NormalizedRFObservation(
            center_freq_mhz=self.center_freq_mhz,
            bandwidth_khz=bw_khz,
            signal_strength_dbm=signal_power_dbm,
            sample_rate_mhz=self.sample_rate_mhz,
            source_type=self.get_source_type(),
            provenance_tag=self.get_provenance_tag(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            iq_samples=iq_samples,
            iq_available=1,
            iq_features=iq_features,
            metadata={
                "device": "RTL2832U SDR",
                "gain_setting": self.gain,
                "num_samples": len(iq_samples),
            },
        )
