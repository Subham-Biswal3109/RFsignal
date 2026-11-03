"""Phase 4C Occupancy & Channel Utilization Analytics Service.

Calculates temporal channel utilization from real observed event timelines in SQLite:
    Utilization (%) = (Active Time / Observation Time) * 100

Handles empty observation windows gracefully without fabricating temporal data.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from backend.database.models import RFEventLog
from backend.database.connection import get_db


def calculate_channel_utilization(
    target_frequency_mhz: Optional[float] = None,
    observation_window_seconds: float = 3600.0,
    db_session: Optional[Session] = None,
) -> Dict[str, Any]:
    """Calculate temporal channel utilization stats across recorded events."""
    session = db_session if db_session is not None else next(get_db())

    try:
        query = session.query(RFEventLog)
        if target_frequency_mhz is not None:
            query = query.filter(RFEventLog.frequency_mhz == target_frequency_mhz)

        events = query.all()

        if not events:
            return {
                "utilization_status": "UNAVAILABLE",
                "reason": "insufficient temporal observations",
                "target_frequency_mhz": target_frequency_mhz,
                "total_observation_duration_s": observation_window_seconds,
                "active_duration_s": 0.0,
                "inactive_duration_s": observation_window_seconds,
                "uncertain_duration_s": 0.0,
                "utilization_percentage": 0.0,
                "event_count": 0,
                "average_event_duration_s": 0.0,
                "maximum_event_duration_s": 0.0,
                "provenance": "DERIVED",
            }

        durations = [e.duration_seconds for e in events if e.duration_seconds is not None and e.duration_seconds > 0]
        
        # Merge overlapping time intervals to prevent double-counting active time
        intervals = []
        for e in events:
            if e.start_time and e.end_time:
                intervals.append((e.start_time, e.end_time))
            elif e.start_time and e.duration_seconds:
                from datetime import timedelta
                intervals.append((e.start_time, e.start_time + timedelta(seconds=e.duration_seconds)))

        merged = []
        if intervals:
            intervals.sort(key=lambda x: x[0])
            curr_start, curr_end = intervals[0]
            for start, end in intervals[1:]:
                if start <= curr_end:
                    curr_end = max(curr_end, end)
                else:
                    merged.append((curr_start, curr_end))
                    curr_start, curr_end = start, end
            merged.append((curr_start, curr_end))

        total_active_s = sum((end - start).total_seconds() for start, end in merged)
        
        if merged:
            min_time = min(start for start, _ in merged)
            max_time = max(end for _, end in merged)
            obs_duration_s = max(observation_window_seconds, (max_time - min_time).total_seconds())
        else:
            obs_duration_s = observation_window_seconds

        inactive_s = max(0.0, obs_duration_s - total_active_s)
        utilization_pct = min(100.0, (total_active_s / obs_duration_s) * 100.0)

        avg_dur = sum(durations) / len(durations) if durations else 0.0
        max_dur = max(durations) if durations else 0.0

        return {
            "utilization_status": "AVAILABLE",
            "reason": "calculated from recorded event history with interval merging",
            "target_frequency_mhz": target_frequency_mhz,
            "total_observation_duration_s": round(obs_duration_s, 2),
            "active_duration_s": round(total_active_s, 2),
            "inactive_duration_s": round(inactive_s, 2),
            "uncertain_duration_s": 0.0,
            "utilization_percentage": round(utilization_pct, 2),
            "event_count": len(events),
            "average_event_duration_s": round(avg_dur, 2),
            "maximum_event_duration_s": round(max_dur, 2),
            "provenance": "DERIVED",
        }

    except Exception as ex:
        return {
            "utilization_status": "UNAVAILABLE",
            "reason": f"Database query error: {str(ex)}",
            "utilization_percentage": None,
            "provenance": "DERIVED",
        }
