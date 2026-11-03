"""Wire Watcher RF Sources package.

Provides a unified interface for RF observation data from multiple providers:
  - SIMULATION: Baseband sine + AWGN generator (SimulatedRFSource)
  - DATASET: Sampled records from active SDR dataset (logged_data.csv)
  - IQ_REPLAY: Baseband complex I/Q file replay provider
  - RTL_SDR: Physical USB RTL-SDR hardware adapter
"""
from backend.rf.sources.base import (
    BaseRFSource,
    NormalizedRFObservation,
    RFSourceType,
    ProvenanceTag,
)
from backend.rf.sources.simulation_source import SimulationRFSource
from backend.rf.sources.dataset_source import DatasetRFSource
from backend.rf.sources.rtlsdr_source import RtlSdrSource
from backend.rf.sources.factory import RFSourceFactory

__all__ = [
    "BaseRFSource",
    "NormalizedRFObservation",
    "RFSourceType",
    "ProvenanceTag",
    "SimulationRFSource",
    "DatasetRFSource",
    "RtlSdrSource",
    "RFSourceFactory",
]
