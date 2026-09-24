"""Wire Watcher Channel Allocation & Scoring Engine.

Provides engineering-based channel candidate generation, transparent scoring,
guard-band protected region analysis, interference risk assessment, and candidate
recommendations across an analyzed frequency band.

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
GUARD_BAND_CONFLICT_PENALTY = 20.0
GUARD_BAND_MARGINAL_PENALTY = 10.0
INTERFERENCE_HIGH_PENALTY = 25.0
INTERFERENCE_MEDIUM_PENALTY = 10.0
MIN_RECOMMENDATION_SCORE = 50.0


@dataclass
class ChannelCandidate:
    """Represents a generated frequency channel candidate with full assessment trace."""
    candidate_id: str
    center_freq_mhz: float
    start_freq_mhz: float
    end_freq_mhz: float
    bandwidth_mhz: float
    guard_band_mhz: float
    protected_lower_mhz: float
    protected_upper_mhz: float
    guard_band_status: str     # SAFE_MARGIN | MARGINAL | CONFLICT
    interference_risk: str      # LOW | MEDIUM | HIGH | UNKNOWN
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
    assessment_explanation: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "center_freq_mhz": round(self.center_freq_mhz, 3),
            "start_freq_mhz": round(self.start_freq_mhz, 3),
            "end_freq_mhz": round(self.end_freq_mhz, 3),
            "bandwidth_mhz": round(self.bandwidth_mhz, 3),
            "bandwidth_khz": round(self.bandwidth_mhz * 1000.0, 1),
            "guard_band_mhz": round(self.guard_band_mhz, 3),
            "protected_lower_mhz": round(self.protected_lower_mhz, 3),
            "protected_upper_mhz": round(self.protected_upper_mhz, 3),
            "guard_band_status": self.guard_band_status,
            "interference_risk": self.interference_risk,
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
            "assessment_explanation": self.assessment_explanation,
        }


class ChannelAllocationEngine:
    """Engine for generating, analyzing, scoring, and explaining candidate channel allocations."""

    @staticmethod
    def analyze_guard_band(
        center_freq_mhz: float,
        bandwidth_mhz: float,
        guard_band_mhz: float,
        active_peaks: List[Dict[str, float]] | None = None,
    ) -> tuple[float, float, str]:
        """Compute guard-band protected region and classify guard-band status."""
        channel_lower = center_freq_mhz - (bandwidth_mhz / 2.0)
        channel_upper = center_freq_mhz + (bandwidth_mhz / 2.0)
        protected_lower = channel_lower - guard_band_mhz
        protected_upper = channel_upper + guard_band_mhz

        if not active_peaks:
            return protected_lower, protected_upper, "SAFE_MARGIN"

        status = "SAFE_MARGIN"
        for peak in active_peaks:
            peak_f = peak.get("frequency_mhz", 0.0)
            if channel_lower <= peak_f <= channel_upper:
                status = "CONFLICT"
                break
            elif (protected_lower <= peak_f < channel_lower) or (channel_upper < peak_f <= protected_upper):
                if status != "CONFLICT":
                    status = "MARGINAL"

        return protected_lower, protected_upper, status

    @staticmethod
    def analyze_interference_risk(
        center_freq_mhz: float,
        bandwidth_mhz: float,
        activity: str,
        guard_band_status: str,
        active_peaks: List[Dict[str, float]] | None = None,
    ) -> str:
        """Calculate transparent interference risk metric based on observed measurements."""
        if activity == "UNCERTAIN" and not active_peaks:
            return "UNKNOWN"

        if activity == "DETECTED" or guard_band_status == "CONFLICT":
            return "HIGH"

        if guard_band_status == "MARGINAL":
            return "MEDIUM"

        if active_peaks:
            for peak in active_peaks:
                freq_sep = abs(center_freq_mhz - peak.get("frequency_mhz", 0.0))
                if freq_sep < bandwidth_mhz * 2.0:
                    return "MEDIUM"

        return "LOW"

    @classmethod
    def calculate_channel_score(
        cls,
        availability: str,
        activity: str,
        ood_warning: bool,
        noise_floor_dbm: float,
        guard_band_status: str = "SAFE_MARGIN",
        interference_risk: str = "LOW",
    ) -> tuple[float, Dict[str, float]]:
        """Calculate a transparent candidate score from documented penalties."""
        breakdown = {
            "base_score": BASE_SCORE,
            "activity_penalty": 0.0,
            "noise_penalty": 0.0,
            "ood_penalty": 0.0,
            "uncertainty_penalty": 0.0,
            "guard_band_penalty": 0.0,
            "interference_penalty": 0.0,
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

        # Guard-band penalty
        if guard_band_status == "CONFLICT":
            breakdown["guard_band_penalty"] = GUARD_BAND_CONFLICT_PENALTY
        elif guard_band_status == "MARGINAL":
            breakdown["guard_band_penalty"] = GUARD_BAND_MARGINAL_PENALTY

        # Interference penalty
        if interference_risk == "HIGH":
            breakdown["interference_penalty"] = INTERFERENCE_HIGH_PENALTY
        elif interference_risk == "MEDIUM":
            breakdown["interference_penalty"] = INTERFERENCE_MEDIUM_PENALTY

        total_penalty = sum(v for k, v in breakdown.items() if k != "base_score")
        final_score = max(0.0, min(100.0, BASE_SCORE - total_penalty))
        return final_score, breakdown

    @staticmethod
    def generate_explainable_assessment(
        center_freq_mhz: float,
        bandwidth_mhz: float,
        activity: str,
        noise_floor_dbm: float,
        interference_risk: str,
        guard_band_status: str,
        ood_warning: bool,
        availability: str,
        score: float,
        recommendation_status: str,
    ) -> Dict[str, Any]:
        """Generate explainable assessment text and engineering calculation trace."""
        if recommendation_status == "RECOMMENDED":
            summary = f"Candidate channel at {center_freq_mhz:.3f} MHz is preferred. Low noise ({noise_floor_dbm:.1f} dBm), inactive RF status, and safe guard-band margin."
            rec_text = "PREFERRED CANDIDATE"
        elif recommendation_status == "REVIEW_REQUIRED":
            summary = f"Candidate channel at {center_freq_mhz:.3f} MHz requires review due to moderate noise, marginal guard band, or OOD warning."
            rec_text = "REVIEW REQUIRED"
        else:
            summary = f"Candidate channel at {center_freq_mhz:.3f} MHz is rejected due to detected RF activity, high noise, or safety conflict."
            rec_text = "DO NOT SELECT"

        calculation_trace = {
            "inputs": {
                "center_frequency": f"{center_freq_mhz:.3f} MHz",
                "channel_bandwidth": f"{bandwidth_mhz * 1000.0:.1f} kHz",
                "guard_band": f"{0.05:.2f} MHz",
                "noise_floor": f"{noise_floor_dbm:.1f} dBm",
                "observed_activity": activity,
                "ood_flag": str(ood_warning),
            },
            "formula": "Score = 100 - ActivityPenalty(50) - UncertaintyPenalty(25) - OODPenalty(40) - NoisePenalty - GuardBandPenalty - InterferencePenalty",
            "substitution": f"Score = 100 - (Activity: {activity}) - (OOD: {ood_warning}) - (Noise: {noise_floor_dbm:.1f} dBm)",
            "result": f"{score:.1f} / 100.0",
            "unit": "Engineering Quality Index (0-100)",
            "engineering_note": "Candidate scores represent empirical RF activity and noise assessment only. Not a legal or regulatory transmission clearance.",
        }

        return {
            "headline": f"{center_freq_mhz:.3f} MHz / {bandwidth_mhz * 1000.0:.0f} kHz Assessment",
            "summary": summary,
            "recommendation": rec_text,
            "trace": calculation_trace,
        }

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
        active_peaks: List[Dict[str, float]] | None = None,
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

            # Guard band analysis
            prot_lower, prot_upper, gb_status = cls.analyze_guard_band(
                center_freq_mhz=center_f,
                bandwidth_mhz=channel_bw_mhz,
                guard_band_mhz=guard_band_mhz,
                active_peaks=active_peaks,
            )

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

            # Interference analysis
            interference_risk = cls.analyze_interference_risk(
                center_freq_mhz=center_f,
                bandwidth_mhz=channel_bw_mhz,
                activity=activity,
                guard_band_status=gb_status,
                active_peaks=active_peaks,
            )

            snr_val = power_dbm - noise_floor_dbm if math.isfinite(noise_floor_dbm) else None

            score, breakdown = cls.calculate_channel_score(
                availability=availability,
                activity=activity,
                ood_warning=ood,
                noise_floor_dbm=noise_floor_dbm,
                guard_band_status=gb_status,
                interference_risk=interference_risk,
            )

            is_recommended = (
                score >= MIN_RECOMMENDATION_SCORE
                and availability == "AVAILABLE"
                and not ood
                and gb_status != "CONFLICT"
                and interference_risk != "HIGH"
            )
            rec_status = "RECOMMENDED" if is_recommended else ("REVIEW_REQUIRED" if score >= 30 else "REJECTED")

            explanation = cls.generate_explainable_assessment(
                center_freq_mhz=center_f,
                bandwidth_mhz=channel_bw_mhz,
                activity=activity,
                noise_floor_dbm=noise_floor_dbm,
                interference_risk=interference_risk,
                guard_band_status=gb_status,
                ood_warning=ood,
                availability=availability,
                score=score,
                recommendation_status=rec_status,
            )

            cand = ChannelCandidate(
                candidate_id=f"CH-{idx:03d}",
                center_freq_mhz=center_f,
                start_freq_mhz=curr_start,
                end_freq_mhz=curr_end,
                bandwidth_mhz=channel_bw_mhz,
                guard_band_mhz=guard_band_mhz,
                protected_lower_mhz=prot_lower,
                protected_upper_mhz=prot_upper,
                guard_band_status=gb_status,
                interference_risk=interference_risk,
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
                assessment_explanation=explanation,
            )
            candidates.append(cand)

            curr_start += step
            idx += 1

        # Sort candidates by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)

        # Select best candidate
        recommended_candidate = next((c for c in candidates if c.recommendation_status == "RECOMMENDED"), None)
        rejected_candidates = [c.to_dict() for c in candidates if c.recommendation_status != "RECOMMENDED"][:5]

        reason = (
            f"Candidate {recommended_candidate.candidate_id} at {recommended_candidate.center_freq_mhz:.3f} MHz recommended with score {recommended_candidate.score:.1f}/100."
            if recommended_candidate
            else "NO SUITABLE CANDIDATE found matching safety criteria (score >= 50, AVAILABLE, no OOD, safe guard-band margin)."
        )

        supporting_evidence = (
            {
                "noise_floor_dbm": recommended_candidate.noise_floor_dbm,
                "activity": recommended_candidate.activity,
                "availability": recommended_candidate.availability,
                "guard_band_status": recommended_candidate.guard_band_status,
                "interference_risk": recommended_candidate.interference_risk,
                "ood_warning": recommended_candidate.ood_warning,
            }
            if recommended_candidate
            else {}
        )

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
                "guard_band_conflict_penalty": GUARD_BAND_CONFLICT_PENALTY,
                "guard_band_marginal_penalty": GUARD_BAND_MARGINAL_PENALTY,
                "interference_high_penalty": INTERFERENCE_HIGH_PENALTY,
                "interference_medium_penalty": INTERFERENCE_MEDIUM_PENALTY,
                "min_recommendation_score": MIN_RECOMMENDATION_SCORE,
            },
            "total_candidates": len(candidates),
            "recommended_candidate": recommended_candidate.to_dict() if recommended_candidate else None,
            "has_suitable_candidate": recommended_candidate is not None,
            "reason": reason,
            "supporting_evidence": supporting_evidence,
            "rejected_candidates": rejected_candidates,
            "candidates": [c.to_dict() for c in candidates],
            "disclaimer": "Engineering availability assessment and allocation recommendation. Not a legal or regulatory transmission license.",
        }
