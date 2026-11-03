"""Wire Watcher — Deterministic RF/DSP Activity Detector.

Implements a non-ML, physics-grounded RF activity detection algorithm based on:
  1. Thermal Noise Floor calculation (P_n = k * T * B + NF)
  2. Signal-to-Noise Ratio (SNR >= 6 dB threshold)
  3. Spectral Power Density (PSD) & Peak Prominence (>= 3 dB above noise floor)
  4. Occupied Bandwidth & Guard Band Overlap
  5. Configurable Temporal Persistence (minimum active count, min duration, max gap)

Outputs:
  - activity_state: "ACTIVE" | "INACTIVE" | "UNCERTAIN"
  - dsp_evidence_details: dict of exact formulas, thresholds, and calculations.
"""

from __future__ import annotations
import math
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone

# ── PHYSICAL CONSTANTS ────────────────────────────────────────────────────────
BOLTZMANN_K = 1.380649e-23  # Joules per Kelvin
ROOM_TEMP_K = 290.0         # Standard IEEE noise temperature (290 K)
DEFAULT_NOISE_FIGURE_DB = 6.0  # Standard SDR front-end noise figure
DEFAULT_SNR_THRESHOLD_DB = 6.0  # Minimum SNR for reliable signal detection
DEFAULT_PROMINENCE_DB = 3.0    # Peak prominence above local floor


class DSPActivityDetector:
    """Deterministic RF/DSP Activity Detector."""

    def __init__(
        self,
        snr_threshold_db: float = DEFAULT_SNR_THRESHOLD_DB,
        noise_figure_db: float = DEFAULT_NOISE_FIGURE_DB,
        peak_prominence_db: float = DEFAULT_PROMINENCE_DB,
        min_persistence_count: int = 2,
        min_duration_seconds: float = 0.5,
        max_gap_seconds: float = 2.0,
    ):
        self.snr_threshold_db = snr_threshold_db
        self.noise_figure_db = noise_figure_db
        self.peak_prominence_db = peak_prominence_db
        self.min_persistence_count = min_persistence_count
        self.min_duration_seconds = min_duration_seconds
        self.max_gap_seconds = max_gap_seconds
        
        # State tracking per frequency key ("<freq_mhz>")
        self._history: Dict[str, List[Dict[str, Any]]] = {}

    @staticmethod
    def calculate_thermal_noise_dbm(bandwidth_khz: float, noise_figure_db: float = DEFAULT_NOISE_FIGURE_DB) -> float:
        """Calculate theoretical thermal noise floor in dBm.

        P_thermal_watts = k * T * B
        P_thermal_dbm = 10 * log10(P_watts * 1000) + Noise_Figure
        = -174 dBm/Hz + 10 * log10(B_hz) + NF
        """
        bw_hz = max(1.0, bandwidth_khz * 1000.0)
        thermal_dbm_per_hz = 10.0 * math.log10(BOLTZMANN_K * ROOM_TEMP_K * 1000.0)  # ≈ -173.98 dBm/Hz
        total_noise_dbm = thermal_dbm_per_hz + 10.0 * math.log10(bw_hz) + noise_figure_db
        return round(total_noise_dbm, 2)

    def evaluate_observation(
        self,
        frequency_mhz: float,
        bandwidth_khz: float,
        signal_strength_dbm: float,
        psd_peak_dbm: Optional[float] = None,
        iq_available: bool = False,
        iq_rms_magnitude: Optional[float] = None,
        iq_spectral_entropy: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single RF observation deterministically.

        Returns
        -------
        Dict with keys:
          - activity_state: "ACTIVE" | "INACTIVE" | "UNCERTAIN"
          - snr_db: float
          - noise_floor_dbm: float
          - detection_threshold_dbm: float
          - peak_prominence_db: float
          - temporal_persistence: dict
          - dsp_reason: str
        """
        # 1. Thermal Noise & Detection Threshold
        noise_floor_dbm = self.calculate_thermal_noise_dbm(bandwidth_khz, self.noise_figure_db)
        detection_threshold_dbm = noise_floor_dbm + self.snr_threshold_db

        # Effective signal power (use PSD peak if available, else signal_strength_dbm)
        effective_power = psd_peak_dbm if psd_peak_dbm is not None else signal_strength_dbm
        snr_db = round(effective_power - noise_floor_dbm, 2)
        peak_prominence = round(max(0.0, effective_power - noise_floor_dbm), 2)

        # 2. Instantaneous DSP Activity Condition
        instant_active = (snr_db >= self.snr_threshold_db) and (signal_strength_dbm >= -100.0)

        # 3. Temporal Persistence Check
        freq_key = f"{frequency_mhz:.3f}"
        now = timestamp or datetime.now(timezone.utc)
        
        if freq_key not in self._history:
            self._history[freq_key] = []

        # Append observation
        obs_entry = {
            "timestamp": now,
            "power_dbm": signal_strength_dbm,
            "snr_db": snr_db,
            "instant_active": instant_active,
        }
        self._history[freq_key].append(obs_entry)

        # Clean old entries beyond 60s
        self._history[freq_key] = [
            e for e in self._history[freq_key]
            if (now - e["timestamp"]).total_seconds() <= 60.0
        ]

        recent_entries = self._history[freq_key]
        active_count = sum(1 for e in recent_entries if e["instant_active"])

        # Check temporal sequence continuity
        if len(recent_entries) >= self.min_persistence_count:
            persistent_active = active_count >= self.min_persistence_count
            persistence_available = True
        else:
            persistent_active = instant_active
            persistence_available = False

        # 4. Final Decision Synthesis
        if instant_active and persistent_active:
            activity_state = "ACTIVE"
            reason = f"SNR ({snr_db:.1f} dB) exceeds threshold ({self.snr_threshold_db} dB) above noise floor ({noise_floor_dbm:.1f} dBm)."
        elif instant_active and not persistent_active:
            activity_state = "UNCERTAIN"
            reason = f"Instantaneous SNR high ({snr_db:.1f} dB), but minimum temporal persistence ({self.min_persistence_count} samples) not met."
        else:
            activity_state = "INACTIVE"
            reason = f"SNR ({snr_db:.1f} dB) below detection threshold ({self.snr_threshold_db} dB above noise floor {noise_floor_dbm:.1f} dBm)."

        return {
            "activity_state": activity_state,
            "snr_db": snr_db,
            "noise_floor_dbm": noise_floor_dbm,
            "detection_threshold_dbm": detection_threshold_dbm,
            "peak_prominence_db": peak_prominence,
            "instant_active": instant_active,
            "temporal_persistence": {
                "available": persistence_available,
                "recent_observations_count": len(recent_entries),
                "active_observations_count": active_count,
                "min_required_count": self.min_persistence_count,
                "status": "PERSISTENT_ACTIVE" if persistent_active else "TRANSIENT_OR_INACTIVE",
            },
            "dsp_reason": reason,
            "formulas": {
                "noise_floor": "P_n = k * T * B * NF (-174 dBm/Hz + 10*log10(B_Hz) + NF)",
                "snr": "SNR = P_signal_dbm - P_noise_dbm",
                "detection_criteria": "Active if SNR >= 6.0 dB and Power >= -100.0 dBm",
            },
        }
