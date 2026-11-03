"""RF Event Detector for Phase 4B.

Detects active RF transmission bursts from consecutive observations, computes event duration,
peak power, and persists events to the database for analytics.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import pandas as pd
from sqlalchemy.orm import Session

from backend.database.models import RFEventLog
from backend.database.connection import get_db


class RFEventDetector:
    """Detects active RF events and persists them to SQLite."""

    def __init__(self, frequency_threshold_dbm: float = -80.0):
        self.frequency_threshold_dbm = frequency_threshold_dbm
        self._active_events: Dict[str, Dict[str, Any]] = {}

    def process_observation(
        self,
        frequency_mhz: float,
        power_dbm: float,
        is_active: bool,
        source_type: str = "DATASET",
        bandwidth_mhz: float = 0.2,
        provenance: str = "INFERRED_RF_ACTIVITY",
        db_session: Optional[Session] = None,
    ) -> Optional[Dict[str, Any]]:
        """Process one sample point for a given frequency. Returns closed event if completed."""
        freq_key = f"{frequency_mhz:.3f}"
        now = datetime.now(timezone.utc)

        close_event_dict = None

        if is_active:
            if freq_key not in self._active_events:
                # Start new active event
                self._active_events[freq_key] = {
                    "frequency_mhz": frequency_mhz,
                    "bandwidth_mhz": bandwidth_mhz,
                    "start_time": now,
                    "peak_power_dbm": power_dbm,
                    "powers": [power_dbm],
                    "source_type": source_type,
                    "provenance": provenance,
                }
            else:
                # Continue active event
                ev = self._active_events[freq_key]
                ev["peak_power_dbm"] = max(ev["peak_power_dbm"], power_dbm)
                ev["powers"].append(power_dbm)
        else:
            if freq_key in self._active_events:
                # End active event
                ev = self._active_events.pop(freq_key)
                start_t = ev["start_time"]
                duration = max(0.1, (now - start_t).total_seconds())
                avg_power = sum(ev["powers"]) / len(ev["powers"])

                close_event_dict = {
                    "frequency_mhz": ev["frequency_mhz"],
                    "bandwidth_mhz": ev["bandwidth_mhz"],
                    "start_time": start_t,
                    "end_time": now,
                    "duration_seconds": duration,
                    "peak_power_dbm": ev["peak_power_dbm"],
                    "avg_power_dbm": avg_power,
                    "activity_state": "DETECTED",
                    "source_type": ev["source_type"],
                    "provenance": ev["provenance"],
                }

                # Persist event to DB
                session = db_session if db_session is not None else next(get_db())
                try:
                    db_obj = RFEventLog(
                        frequency_mhz=close_event_dict["frequency_mhz"],
                        bandwidth_mhz=close_event_dict["bandwidth_mhz"],
                        start_time=close_event_dict["start_time"],
                        end_time=close_event_dict["end_time"],
                        duration_seconds=close_event_dict["duration_seconds"],
                        peak_power_dbm=close_event_dict["peak_power_dbm"],
                        avg_power_dbm=close_event_dict["avg_power_dbm"],
                        activity_state="DETECTED",
                        source_type=close_event_dict["source_type"],
                        provenance=close_event_dict["provenance"],
                    )
                    session.add(db_obj)
                    session.commit()
                    close_event_dict["event_id"] = db_obj.event_id
                except Exception:
                    session.rollback()

        return close_event_dict


def get_rf_event_summary(db_session: Optional[Session] = None) -> Dict[str, Any]:
    """Retrieve summary statistics across all persisted RF events in SQLite."""
    session = db_session if db_session is not None else next(get_db())
    try:
        events = session.query(RFEventLog).all()
        if not events:
            return {
                "total_rf_events": 0,
                "average_event_duration_s": 0.0,
                "maximum_event_duration_s": 0.0,
                "average_peak_power_dbm": 0.0,
                "maximum_peak_power_dbm": 0.0,
                "most_active_frequency_mhz": None,
                "most_frequent_source": None,
                "events": [],
            }

        total_count = len(events)
        durations = [e.duration_seconds for e in events if e.duration_seconds is not None]
        peaks = [e.peak_power_dbm for e in events if e.peak_power_dbm is not None]

        avg_duration = sum(durations) / len(durations) if durations else 0.0
        max_duration = max(durations) if durations else 0.0
        avg_peak = sum(peaks) / len(peaks) if peaks else 0.0
        max_peak = max(peaks) if peaks else 0.0

        # Frequency breakdown
        freq_counts: Dict[float, int] = {}
        for e in events:
            freq_counts[e.frequency_mhz] = freq_counts.get(e.frequency_mhz, 0) + 1
        most_active_freq = max(freq_counts, key=freq_counts.get) if freq_counts else None

        # Source breakdown
        src_counts: Dict[str, int] = {}
        for e in events:
            src_counts[e.source_type] = src_counts.get(e.source_type, 0) + 1
        most_frequent_src = max(src_counts, key=src_counts.get) if src_counts else None

        return {
            "total_rf_events": total_count,
            "average_event_duration_s": round(avg_duration, 2),
            "maximum_event_duration_s": round(max_duration, 2),
            "average_peak_power_dbm": round(avg_peak, 2),
            "maximum_peak_power_dbm": round(max_peak, 2),
            "most_active_frequency_mhz": round(most_active_freq, 3) if most_active_freq else None,
            "most_frequent_source": most_frequent_src,
            "events": [e.to_dict() for e in events[:50]],  # Paginated top 50
        }
    except Exception as ex:
        return {
            "total_rf_events": 0,
            "error": str(ex),
            "events": [],
        }


def derive_and_ingest_events_from_dataset(
    dataset_source: Any = None,
    db_session: Optional[Session] = None,
    force_reingest: bool = False,
) -> Dict[str, Any]:
    """Derive contiguous active RF events per frequency channel from authentic dataset and persist to SQLite."""
    from datetime import timedelta
    from backend.rf.sources.dataset_source import DatasetRFSource

    if dataset_source is None:
        dataset_source = DatasetRFSource()

    if not dataset_source.is_available() or dataset_source.df is None:
        return {
            "status": "error",
            "message": "Dataset unavailable",
            "events_created": 0,
        }

    session = db_session if db_session is not None else next(get_db())

    # Check if events already exist in DB
    existing_count = session.query(RFEventLog).count()
    if existing_count > 0 and not force_reingest:
        return {
            "status": "already_ingested",
            "message": f"SQLite already contains {existing_count} RF events.",
            "events_created": 0,
            "existing_events": existing_count,
        }

    if force_reingest and existing_count > 0:
        session.query(RFEventLog).delete()
        session.commit()

    df = dataset_source.df.copy()

    # Parse timestamps safely
    if "Timestamp" in df.columns:
        df["dt"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    else:
        df["dt"] = [datetime.now(timezone.utc) + timedelta(seconds=i * 20) for i in range(len(df))]

    new_events: List[RFEventLog] = []

    # Process each frequency channel chronologically
    for freq, group in df.groupby("frequency_mhz"):
        group_sorted = group.sort_values("dt")

        in_event = False
        start_t = None
        last_t = None
        powers = []
        bw_khz = 200.0

        for _, row in group_sorted.iterrows():
            is_act = int(row.get("inferred_rf_activity", 0)) == 1
            pwr = float(row.get("signal_strength_dbm", -100.0))
            bw_khz = float(row.get("bandwidth_khz", 200.0))
            t = row["dt"] if pd.notnull(row["dt"]) else datetime.now(timezone.utc)

            if is_act:
                if not in_event:
                    in_event = True
                    start_t = t
                    powers = [pwr]
                else:
                    powers.append(pwr)
                last_t = t
            else:
                if in_event:
                    # Close event burst
                    end_t = last_t + timedelta(seconds=20)
                    dur = max(20.0, (end_t - start_t).total_seconds())
                    peak_pwr = max(powers)
                    avg_pwr = sum(powers) / len(powers)

                    ev = RFEventLog(
                        frequency_mhz=float(freq),
                        bandwidth_mhz=bw_khz / 1000.0,
                        start_time=start_t.to_pydatetime() if hasattr(start_t, "to_pydatetime") else start_t,
                        end_time=end_t.to_pydatetime() if hasattr(end_t, "to_pydatetime") else end_t,
                        duration_seconds=dur,
                        peak_power_dbm=peak_pwr,
                        avg_power_dbm=avg_pwr,
                        activity_state="DETECTED",
                        source_type="DATASET",
                        provenance="INFERRED_RF_ACTIVITY",
                    )
                    new_events.append(ev)
                    in_event = False
                    powers = []

        # Catch tail event if ongoing at end of dataset
        if in_event and start_t is not None and last_t is not None:
            end_t = last_t + timedelta(seconds=20)
            dur = max(20.0, (end_t - start_t).total_seconds())
            peak_pwr = max(powers)
            avg_pwr = sum(powers) / len(powers)

            ev = RFEventLog(
                frequency_mhz=float(freq),
                bandwidth_mhz=bw_khz / 1000.0,
                start_time=start_t.to_pydatetime() if hasattr(start_t, "to_pydatetime") else start_t,
                end_time=end_t.to_pydatetime() if hasattr(end_t, "to_pydatetime") else end_t,
                duration_seconds=dur,
                peak_power_dbm=peak_pwr,
                avg_power_dbm=avg_pwr,
                activity_state="DETECTED",
                source_type="DATASET",
                provenance="INFERRED_RF_ACTIVITY",
            )
            new_events.append(ev)

    try:
        session.bulk_save_objects(new_events)
        session.commit()
        return {
            "status": "success",
            "message": f"Successfully derived and persisted {len(new_events)} RF events in SQLite.",
            "events_created": len(new_events),
            "observations_processed": len(df),
        }
    except Exception as ex:
        session.rollback()
        return {
            "status": "error",
            "message": f"Failed to persist derived events: {str(ex)}",
            "events_created": 0,
        }

