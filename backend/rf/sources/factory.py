"""RF Source Factory.

Instantiates and returns requested BaseRFSource implementations.
"""
from __future__ import annotations
from typing import Dict, Any, Union

from backend.rf.sources.base import BaseRFSource, RFSourceType
from backend.rf.sources.simulation_source import SimulationRFSource
from backend.rf.sources.dataset_source import DatasetRFSource
from backend.rf.sources.rtlsdr_source import RtlSdrSource


class RFSourceFactory:
    """Factory for managing and instantiating RF Sources."""

    _instances: Dict[RFSourceType, BaseRFSource] = {}

    @classmethod
    def get_source(
        self,
        source_type: Union[RFSourceType, str] = RFSourceType.SIMULATION,
        **kwargs,
    ) -> BaseRFSource:
        """Return or instantiate an RF source instance of the specified type."""
        if isinstance(source_type, str):
            try:
                st = RFSourceType(source_type.upper())
            except ValueError:
                raise ValueError(
                    f"Unknown source type '{source_type}'. "
                    f"Must be one of: {[s.value for s in RFSourceType]}"
                )
        else:
            st = source_type

        if st == RFSourceType.SIMULATION:
            return SimulationRFSource(**kwargs)
        elif st == RFSourceType.DATASET:
            return DatasetRFSource(**kwargs)
        elif st == RFSourceType.RTL_SDR:
            return RtlSdrSource(**kwargs)
        elif st == RFSourceType.IQ_REPLAY:
            # IQ File replay uses DatasetRFSource with specified file or Simulation as fallback
            return DatasetRFSource(**kwargs)
        else:
            raise ValueError(f"Unsupported RF source type: {st}")

    @classmethod
    def list_sources(cls) -> Dict[str, Dict[str, Any]]:
        """Return status information for all registered sources."""
        sources = {
            RFSourceType.SIMULATION: SimulationRFSource(),
            RFSourceType.DATASET: DatasetRFSource(),
            RFSourceType.RTL_SDR: RtlSdrSource(),
        }
        return {st.value: source.get_status() for st, source in sources.items()}
