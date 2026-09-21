"""Wire Watcher Channel Allocation & Scoring Engine.

Provides engineering-based channel candidate generation, transparent scoring,
and candidate recommendation across an analyzed frequency band.

This is an engineering assessment and recommendation layer — NOT a regulatory
or legal spectrum authorization system.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from backend.services.prediction import run_ml_inference, get_model_bundle


# Default scoring formula weights and baseline constants
BASE_SCORE = 100.0
OCCUPIED_PENALTY = 50.0
OOD_PENALTY = 40.0
UNCERTAIN_PENALTY = 25.0
NOISE_BASELINE_DBM = -100.0
NOISE_PENALTY_WEIGHT = 1.5
MAX_NOISE_PENALTY = 30.0
MIN_RECOMMENDATION_SCORE = 50.0


@dataclass
class ChannelCandidate:
    """Represents a generated frequency channel candidate."""
    candidate_id: str
    center_freq_mhz: float
    start_freq_mhz: float
    end_freq_mhz: float
    bandwidth_mhz: float
    guard_band_mhz: float
    activity: str
    availability: str
    ml_probability: float
    detector_activity: Optional[bool]
    ood_warning: bool
    noise_floor_dbm: float
    snr_db: Optional[float]
    score: float
    scoring_breakdown: Dict[str, float]
    recommendation_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "center_freq_mhz": round(self.center_freq_mhz, 3),
            "start_freq_mhz": round(self.start_freq_mhz, 3),
            "end_freq_mhz": round(self.end_freq_mhz, 3),
            "bandwidth_mhz": round(self.bandwidth_mhz, 3),
            "bandwidth_khz": round(self.bandwidth_mhz * 1000.0, 1),
            "guard_band_mhz": round(self.guard_band_mhz, 3),
            "activity": self.activity,
            "availability": self.availability,
            "ml_probability": round(self.ml_probability, 4),
            "detector_activity": self.detector_activity,
            "ood_warning": self.ood_warning,
            "noise_floor_dbm": round(self.noise_floor_dbm, 2),
            "snr_db": round(self.snr_db, 2) if self.snr_db is not None else None,
            "score": round(self.score, 1),
            "scoring_breakdown": {k: round(v, 1) for k, v in self.scoring_breakdown.items()},
            "recommendation_status": self.recommendation_status,
        }


class ChannelAllocationEngine:
    """Engine for generating and scoring candidate channels for allocation."""

    @staticmethod
    def calculate_channel_score(
        availability: str,
        activity: str,
        ood_warning: bool,
        noise_floor_dbm: float,
    ) -> tuple[float, Dict[str, float]]:
        """Calculate a transparent candidate score from documented penalties."""
        breakdown = {
            "base_score": BASE_SCORE,
            "activity_penalty": 0.0,
            "noise_penalty": 0.0,
            "ood_penalty": 0.0,
            "uncertainty_penalty": 0.0,
        }

        # Activity penalty
        if availability == "OCCUPIED" or activity == "DETECTED":
            breakdown["activity_penalty"] = OCCUPIED_PENALTY

        # Uncertainty penalty
        if availability == "UNCERTAIN" or activity == "UNCERTAIN":
            breakdown["uncertainty_penalty"] = UNCERTAIN_PENALTY

        # OOD penalty
        if ood_warning:
            breakdown["ood_penalty"] = OOD_PENALTY

        # Noise penalty
        noise_diff = max(0.0, noise_floor_dbm - NOISE_BASELINE_DBM)
        breakdown["noise_penalty"] = min(MAX_NOISE_PENALTY, noise_diff * NOISE_PENALTY_WEIGHT)

        total_penalty = (
            breakdown["activity_penalty"]
            + breakdown["noise_penalty"]
            + breakdown["ood_penalty"]
            + breakdown["uncertainty_penalty"]
        )
        final_score = max(0.0, min(100.0, BASE_SCORE - total_penalty))
        return final_score, breakdown

    @classmethod
    def generate_and_score_candidates(
        cls,
        start_freq_mhz: float = 70.0,
        end_freq_mhz: float = 160.0,
        channel_bw_mhz: float = 0.2,
        guard_band_mhz: float = 0.05,
        noise_floor_dbm: float = -100.0,
        observed_power_dbm: float | None = None,
        location: str | None = None,
    ) -> Dict[str, Any]:
        """Generate candidate channels, evaluate inference + DSP, and return recommendations."""
        if start_freq_mhz <= 0 or end_freq_mhz <= 0 or end_freq_mhz <= start_freq_mhz:
            raise ValueError(f"Invalid frequency range: {start_freq_mhz} to {end_freq_mhz} MHz")
        if channel_bw_mhz <= 0 or channel_bw_mhz > (end_freq_mhz - start_freq_mhz):
            raise ValueError(f"Invalid channel bandwidth: {channel_bw_mhz} MHz")
        if guard_band_mhz < 0:
            raise ValueError(f"Invalid guard band: {guard_band_mhz} MHz")

        bundle = get_model_bundle()
        step = channel_bw_mhz + guard_band_mhz
        curr_start = start_freq_mhz
        candidates: List[ChannelCandidate] = []
        idx = 1

        while (curr_start + channel_bw_mhz) <= end_freq_mhz + 1e-6:
            curr_end = curr_start + channel_bw_mhz
            center_f = (curr_start + curr_end) / 2.0
            bw_khz = channel_bw_mhz * 1000.0

            # Signal power for evaluation
            power_dbm = observed_power_dbm if observed_power_dbm is not None else -75.0

            ml_input = {
                "frequency_mhz": center_f,
                "bandwidth_khz": bw_khz,
                "signal_strength_dbm": power_dbm,
                "iq_available": 0,
            }
            if bundle:
                for c in bundle.get("feature_columns", []):
                    ml_input.setdefault(c, float("nan"))

            if bundle:
                res = run_ml_inference(ml_input)
                activity = res.get("activity", "UNCERTAIN")
                availability = res.get("availability", "UNCERTAIN")
                prob = float(res.get("ml_activity_probability", 0.5))
                detector = res.get("detector_activity")
                ood = bool(res.get("ood", False))
            else:
                activity = "UNCERTAIN"
                availability = "UNCERTAIN"
                prob = 0.5
                detector = None
                ood = True

            snr_val = power_dbm - noise_floor_dbm if math.isfinite(noise_floor_dbm) else None

            score, breakdown = cls.calculate_channel_score(
                availability=availability,
                activity=activity,
                ood_warning=ood,
                noise_floor_dbm=noise_floor_dbm,
            )

            is_recommended = (
                score >= MIN_RECOMMENDATION_SCORE
                and availability == "AVAILABLE"
                and not ood
            )
            rec_status = "RECOMMENDED" if is_recommended else ("REVIEW_REQUIRED" if score >= 30 else "REJECTED")

            cand = ChannelCandidate(
                candidate_id=f"CH-{idx:03d}",
                center_freq_mhz=center_f,
                start_freq_mhz=curr_start,
                end_freq_mhz=curr_end,
                bandwidth_mhz=channel_bw_mhz,
                guard_band_mhz=guard_band_mhz,
                activity=activity,
                availability=availability,
                ml_probability=prob,
                detector_activity=detector,
                ood_warning=ood,
                noise_floor_dbm=noise_floor_dbm,
                snr_db=snr_val,
                score=score,
                scoring_breakdown=breakdown,
                recommendation_status=rec_status,
            )
            candidates.append(cand)

            curr_start += step
            idx += 1

        # Sort candidates by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)

        # Select best candidate
        recommended_candidate = next((c for c in candidates if c.recommendation_status == "RECOMMENDED"), None)

        return {
            "request_params": {
                "start_freq_mhz": start_freq_mhz,
                "end_freq_mhz": end_freq_mhz,
                "channel_bw_mhz": channel_bw_mhz,
                "guard_band_mhz": guard_band_mhz,
                "noise_floor_dbm": noise_floor_dbm,
                "location": location,
            },
            "scoring_weights": {
                "base_score": BASE_SCORE,
                "occupied_penalty": OCCUPIED_PENALTY,
                "ood_penalty": OOD_PENALTY,
                "uncertainty_penalty": UNCERTAIN_PENALTY,
                "noise_baseline_dbm": NOISE_BASELINE_DBM,
                "noise_penalty_weight": NOISE_PENALTY_WEIGHT,
                "max_noise_penalty": MAX_NOISE_PENALTY,
                "min_recommendation_score": MIN_RECOMMENDATION_SCORE,
            },
            "total_candidates": len(candidates),
            "recommended_candidate": recommended_candidate.to_dict() if recommended_candidate else None,
            "has_suitable_candidate": recommended_candidate is not None,
            "candidates": [c.to_dict() for c in candidates],
            "disclaimer": "Engineering availability assessment and allocation recommendation. Not a legal or regulatory transmission license.",
        }
