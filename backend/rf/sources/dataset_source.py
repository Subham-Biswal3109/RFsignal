"""Dataset RF Source Implementation.

Provides normalized RF observations sampled directly from the authentic SDR dataset
(ml/data/processed/rf_signal_activity.csv).
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

from backend.rf.sources.base import (
    BaseRFSource,
    NormalizedRFObservation,
    RFSourceType,
    ProvenanceTag,
)

DATASET_PATH = Path(__file__).resolve().parents[3] / "ml" / "data" / "processed" / "rf_signal_activity.csv"


class DatasetRFSource(BaseRFSource):
    """Dataset RF Source provider."""

    def __init__(self, csv_path: Path | str = DATASET_PATH) -> None:
        self.csv_path = Path(csv_path)
        self._df: pd.DataFrame | None = None
        self._load_dataset()

    def _load_dataset(self) -> None:
        if self.csv_path.exists():
            try:
                self._df = pd.read_csv(self.csv_path)
            except Exception:
                self._df = None

    @property
    def df(self) -> pd.DataFrame | None:
        """Expose underlying dataset DataFrame."""
        return self._df

    @property
    def total_observations(self) -> int:
        """Return total row count of the dataset."""
        return len(self._df) if self._df is not None else 0

    def get_source_type(self) -> RFSourceType:
        return RFSourceType.DATASET

    def get_provenance_tag(self) -> ProvenanceTag:
        return ProvenanceTag.DATASET

    def is_available(self) -> bool:
        return self._df is not None and not self._df.empty

    def fetch_observation(self, *args, **kwargs) -> NormalizedRFObservation:
        """Alias for get_observation."""
        return self.get_observation(*args, **kwargs)

    def get_observation(
        self,
        index: int | None = None,
        frequency_mhz: float | None = None,
        **kwargs,
    ) -> NormalizedRFObservation:
        if not self.is_available() or self._df is None:
            raise RuntimeError(f"Dataset source unavailable. CSV not found at {self.csv_path}")

        df = self._df
        if frequency_mhz is not None:
            sub = df[df["frequency_mhz"] == float(frequency_mhz)]
            if not sub.empty:
                df = sub

        if index is not None and 0 <= index < len(df):
            row = df.iloc[index]
        else:
            row = df.sample(n=1).iloc[0]

        iq_avail = int(row.get("iq_available", 0))
        iq_feats = {}
        for col in [
            "iq_rms_magnitude", "iq_magnitude_variance", "iq_peak_magnitude",
            "iq_crest_factor", "iq_p10", "iq_p50", "iq_p90",
            "iq_phase_concentration", "iq_spectral_entropy", "iq_spectral_peak_ratio"
        ]:
            if col in row and pd.notnull(row[col]):
                iq_feats[col] = float(row[col])

        ts = str(row.get("Timestamp", datetime.now(timezone.utc).isoformat()))

        return NormalizedRFObservation(
            center_freq_mhz=float(row["frequency_mhz"]),
            bandwidth_khz=float(row.get("bandwidth_khz", 50.0)),
            signal_strength_dbm=float(row["signal_strength_dbm"]),
            sample_rate_mhz=float(row.get("bandwidth_khz", 50.0)) * 2.0 / 1000.0,
            source_type=self.get_source_type(),
            provenance_tag=self.get_provenance_tag(),
            timestamp=ts,
            iq_samples=None,  # Pre-computed statistics in dataset
            iq_available=iq_avail,
            iq_features=iq_feats,
            metadata={
                "dataset_name": "logged_data.csv",
                "inferred_rf_activity": int(row.get("inferred_rf_activity", 0)),
                "dataset_index": int(row.name) if hasattr(row, "name") and isinstance(row.name, (int, np.integer)) else (index or 0),
            },
        )

    def get_observations_page(
        self,
        offset: int = 0,
        limit: int = 50,
        frequency_mhz: float | None = None,
    ) -> dict:
        """Return paginated list of observation dicts."""
        if not self.is_available() or self._df is None:
            return {"total": 0, "offset": offset, "limit": limit, "observations": []}

        df = self._df
        if frequency_mhz is not None:
            df = df[df["frequency_mhz"] == float(frequency_mhz)]

        total = len(df)
        slice_df = df.iloc[offset : offset + limit]

        records = []
        for idx, row in slice_df.iterrows():
            records.append({
                "index": int(idx),
                "timestamp": str(row.get("Timestamp", "")),
                "frequency_mhz": float(row.get("frequency_mhz", 0.0)),
                "bandwidth_khz": float(row.get("bandwidth_khz", 0.0)),
                "signal_strength_dbm": float(row.get("signal_strength_dbm", 0.0)),
                "iq_available": int(row.get("iq_available", 0)),
                "inferred_rf_activity": int(row.get("inferred_rf_activity", 0)),
            })

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "observations": records,
        }

