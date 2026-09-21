"""Simulation RF Source Implementation.

Wraps SimulatedRFSource to provide normalized RF observations generated via synthetic
complex baseband sine + AWGN.
"""
from __future__ import annotations
from datetime import datetime, timezone
import numpy as np

from backend.rf.sources.base import (
    BaseRFSource,
    NormalizedRFObservation,
    RFSourceType,
    ProvenanceTag,
)
from backend.rf.rf_source import SimulatedRFSource


class SimulationRFSource(BaseRFSource):
    """Simulation RF Source provider."""

    def __init__(
        self,
        center_freq_mhz: float = 120.0,
        bandwidth_khz: float = 50.0,
        signal_strength_dbm: float = -75.0,
        noise_floor_dbm: float = -100.0,
        num_samples: int = 2048,
    ) -> None:
        self.center_freq_mhz = float(center_freq_mhz)
        self.bandwidth_khz = float(bandwidth_khz)
        self.signal_strength_dbm = float(signal_strength_dbm)
        self.noise_floor_dbm = float(noise_floor_dbm)
        self.num_samples = int(num_samples)

    def get_source_type(self) -> RFSourceType:
        return RFSourceType.SIMULATION

    def get_provenance_tag(self) -> ProvenanceTag:
        return ProvenanceTag.SIMULATION

    def is_available(self) -> bool:
        return True

    def get_observation(
        self,
        center_freq_mhz: float | None = None,
        bandwidth_khz: float | None = None,
        signal_strength_dbm: float | None = None,
        num_samples: int | None = None,
        **kwargs,
    ) -> NormalizedRFObservation:
        f_c = float(center_freq_mhz if center_freq_mhz is not None else self.center_freq_mhz)
        bw_k = float(bandwidth_khz if bandwidth_khz is not None else self.bandwidth_khz)
        p_sig = float(signal_strength_dbm if signal_strength_dbm is not None else self.signal_strength_dbm)
        n_samples = int(num_samples if num_samples is not None else self.num_samples)

        sim = SimulatedRFSource(
            center_freq_mhz=f_c,
            bandwidth_mhz=bw_k / 1000.0,
            signal_strength_dbm=p_sig,
            noise_floor_dbm=self.noise_floor_dbm,
            num_samples=n_samples,
        )
        iq_signal, sample_rate_mhz = sim.get_signal()

        return NormalizedRFObservation(
            center_freq_mhz=f_c,
            bandwidth_khz=bw_k,
            signal_strength_dbm=p_sig,
            sample_rate_mhz=sample_rate_mhz,
            source_type=self.get_source_type(),
            provenance_tag=self.get_provenance_tag(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            iq_samples=iq_signal,
            iq_available=0,  # Simulated signals do not fabricate ML IQ features
            iq_features={},
            metadata={
                "description": "Synthetic baseband sine wave + AWGN",
                "noise_floor_dbm": self.noise_floor_dbm,
            },
        )
