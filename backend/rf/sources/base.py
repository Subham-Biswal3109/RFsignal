"""Common RF Source Interface and Data Structures.

Defines the normalized observation contract returned by all RF sources
(Simulation, Dataset, IQ Replay, RTL-SDR).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any, Dict
import numpy as np


class RFSourceType(str, Enum):
    """Enumeration of supported RF signal source types."""
    DATASET = "DATASET"
    SIMULATION = "SIMULATION"
    IQ_REPLAY = "IQ_REPLAY"
    RTL_SDR = "RTL_SDR"


class ProvenanceTag(str, Enum):
    """Data provenance classification tags for strict scientific tracking."""
    REAL_SDR = "REAL_SDR"
    SIMULATION = "SIMULATION"
    DATASET = "DATASET"
    IQ_REPLAY = "IQ_REPLAY"


@dataclass
class NormalizedRFObservation:
    """Standardized normalized structure for an RF observation across all sources."""
    center_freq_mhz: float
    bandwidth_khz: float
    signal_strength_dbm: float
    sample_rate_mhz: float
    source_type: RFSourceType
    provenance_tag: ProvenanceTag
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    iq_samples: Optional[np.ndarray] = None
    iq_available: int = 0
    iq_features: Dict[str, Optional[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert normalized observation to a dictionary suitable for API responses."""
        res = {
            "center_freq_mhz": float(self.center_freq_mhz),
            "bandwidth_khz": float(self.bandwidth_khz),
            "signal_strength_dbm": float(self.signal_strength_dbm),
            "sample_rate_mhz": float(self.sample_rate_mhz),
            "source_type": self.source_type.value if isinstance(self.source_type, Enum) else str(self.source_type),
            "provenance_tag": self.provenance_tag.value if isinstance(self.provenance_tag, Enum) else str(self.provenance_tag),
            "timestamp": self.timestamp,
            "iq_available": int(self.iq_available),
            "iq_features": self.iq_features,
            "metadata": self.metadata,
        }
        if self.iq_samples is not None:
            res["num_iq_samples"] = len(self.iq_samples)
        else:
            res["num_iq_samples"] = 0
        return res


class BaseRFSource(ABC):
    """Abstract Base Class for all RF signal sources."""

    @abstractmethod
    def get_source_type(self) -> RFSourceType:
        """Return the source type enum."""
        pass

    @abstractmethod
    def get_provenance_tag(self) -> ProvenanceTag:
        """Return the provenance classification tag."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the source is connected and operational."""
        pass

    @abstractmethod
    def get_observation(self, **kwargs) -> NormalizedRFObservation:
        """Capture or generate one normalized RF observation."""
        pass

    def get_status(self) -> Dict[str, Any]:
        """Return a status dictionary describing the source state."""
        return {
            "source": self.get_source_type().value,
            "provenance": self.get_provenance_tag().value,
            "available": self.is_available(),
        }
