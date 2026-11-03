"""RF Replay Controller for Controlled Historical RF Dataset Playback.

Provides chronological playback step-by-step through dataset or IQ replay observations.
Supports Play, Pause, Reset, and speed controls (0.5x, 1x, 2x, 5x).
Tags all outputs with IQ_REPLAY or DATASET provenance — NEVER claims live SDR monitoring.
"""
from __future__ import annotations
import time
from typing import Dict, Any, List, Optional
from backend.rf.sources.dataset_source import DatasetRFSource
from backend.services.prediction import run_ml_inference


class RFReplayController:
    """Singleton/Stateful replay controller for chronological observation replay."""

    _instance: Optional[RFReplayController] = None

    def __init__(self):
        self.is_playing: bool = False
        self.playback_speed: float = 1.0
        self.current_index: int = 0
        self.dataset_source: Optional[DatasetRFSource] = None
        self.last_observation: Optional[Dict[str, Any]] = None

    @classmethod
    def get_instance(cls) -> RFReplayController:
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.initialize_source()
        return cls._instance

    def initialize_source(self, dataset_path: Optional[str] = None):
        """Load dataset source for replay."""
        try:
            self.dataset_source = DatasetRFSource(csv_path=dataset_path) if dataset_path else DatasetRFSource()
        except Exception:
            self.dataset_source = None

    def control(self, action: str, speed: Optional[float] = None, target_index: Optional[int] = None) -> Dict[str, Any]:
        """Process replay control action: play, pause, reset, set_speed, step, set_index."""
        if self.dataset_source is None:
            self.initialize_source()

        action = action.lower()
        if action == "play":
            self.is_playing = True
        elif action == "pause":
            self.is_playing = False
        elif action == "reset":
            self.current_index = 0
            self.is_playing = False
            self.last_observation = None
        elif action == "step":
            self.step()
        elif action == "set_speed" and speed is not None:
            if float(speed) in (0.5, 1.0, 2.0, 5.0):
                self.playback_speed = float(speed)
        elif action == "set_index" and target_index is not None:
            total = self.dataset_source.total_observations if self.dataset_source else 0
            if 0 <= int(target_index) < total:
                self.current_index = int(target_index)
                self._load_current_observation()

        return self.get_status()

    def _load_current_observation(self) -> Optional[Dict[str, Any]]:
        """Fetch and execute full pipeline for current observation index."""
        if self.dataset_source is None or not self.dataset_source.is_available():
            return None

        total = self.dataset_source.total_observations
        if total == 0:
            return None

        if self.current_index >= total:
            self.current_index = total - 1

        obs = self.dataset_source.get_observation(index=self.current_index)
        obs_dict = obs.to_dict()

        # Run pipeline: ML evidence + OOD + Availability
        try:
            pred = run_ml_inference({
                "frequency_mhz": obs.center_freq_mhz,
                "bandwidth_khz": obs.bandwidth_khz,
                "signal_strength_dbm": obs.signal_strength_dbm,
                "iq_available": obs.iq_available,
                **obs.iq_features,
            })
            obs_dict["pipeline_result"] = pred
        except Exception as ex:
            obs_dict["pipeline_result"] = {"error": str(ex)}

        self.last_observation = obs_dict
        return self.last_observation

    def step(self) -> Optional[Dict[str, Any]]:
        """Advance one step in chronological replay."""
        if self.dataset_source is None:
            self.initialize_source()

        if self.dataset_source is None or not self.dataset_source.is_available():
            return None

        total = self.dataset_source.total_observations
        if total == 0:
            return None

        res = self._load_current_observation()

        # Increment index for next step
        if self.current_index + 1 < total:
            self.current_index += 1
        else:
            self.is_playing = False  # Reached end

        return res

    def get_status(self) -> Dict[str, Any]:
        """Return current replay state and provenance metadata."""
        if self.dataset_source is None:
            self.initialize_source()

        total = self.dataset_source.total_observations if self.dataset_source else 0
        if self.last_observation is None and total > 0:
            self._load_current_observation()

        return {
            "mode": "REPLAY_MODE",
            "is_playing": self.is_playing,
            "playback_speed": self.playback_speed,
            "current_index": self.current_index,
            "total_samples": total,
            "provenance": "DATASET",
            "is_live": False,
            "display_warning": "REPLAY MODE — Historical Dataset Replay Observation. NOT LIVE.",
            "last_observation": self.last_observation,
        }
