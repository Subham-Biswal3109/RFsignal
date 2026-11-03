"""
Spectrum Simulation orchestrator.

Builds a simulated multi-channel RF environment and runs EACH channel
through the project's existing, unmodified signal-processing pipeline
(backend/rf/rf_source.py, spectrum_processor.py, noise_estimator.py,
peak_detector.py) to obtain an authentic "RF sensing" result, then runs
the EXISTING trained Random Forest model (backend/services/prediction.py)
on each channel's extracted features to obtain an independent
"ML availability estimation" result.

RF sensing and ML estimation are kept as two separate, clearly labelled
outputs per channel (see `rf_state` vs `ml_decision` below) — the module
never conflates "what the simulated spectrum contains" with "what the
model predicts", per the project's scientific-correctness requirement.

This module does not modify the existing model, its threshold, or the
existing /api/predict, /api/spectrum/analyze workflows in any way.
"""

from typing import List, Dict, Optional
import numpy as np

from backend.rf.simulation.channel_generator import generate_channel_grid
from backend.rf.simulation.signal_generator import assign_existing_signals
from backend.rf.rf_source import SimulatedRFSource
from backend.rf.spectrum_processor import compute_fft_psd
from backend.rf.noise_estimator import estimate_noise_floor
from backend.rf.peak_detector import detect_peaks
from backend.rf.feature_extractor import extract_ml_features
from backend.services.prediction import (
    get_model_bundle,
    get_model_metadata,
    run_ml_inference,
)

# Simulation-only tuning constants (documented; not physical measurements)
SIM_NUM_SAMPLES = 512          # samples per per-channel FFT (kept small: many channels per request)
SIM_SAMPLE_RATE_FACTOR = 4.0   # sample_rate_mhz = channel_bandwidth_mhz * this factor
UNOCCUPIED_SIGNAL_MARGIN_DB = 40.0  # how far below the noise floor an "empty" channel's carrier is set
SIM_WELCH_AVERAGES = 8          # number of independent PSD snapshots averaged per channel (Welch's method)


def _check_ood(ml_input: dict, model_metadata: dict) -> tuple[bool, List[str]]:
    """
    Same OOD bounds-check logic as backend/api/routes.py's /api/predict
    endpoint, duplicated here (read-only, no shared mutable state) so this
    new module never has to import or modify the existing route file.
    """
    ood_warning = False
    warnings_list: List[str] = []
    bounds = model_metadata.get("training_bounds", {})

    if "signal_power_dbm" in bounds:
        b = bounds["signal_power_dbm"]
        if ml_input["signal_power_dbm"] < b["p1"] or ml_input["signal_power_dbm"] > b["p99"]:
            ood_warning = True
            warnings_list.append(
                f"Signal Power ({ml_input['signal_power_dbm']} dBm) is outside typical range "
                f"({b['p1']:.1f} to {b['p99']:.1f})."
            )

    if "snr_db" in bounds:
        b = bounds["snr_db"]
        if ml_input["snr_db"] < b["p1"] or ml_input["snr_db"] > b["p99"]:
            ood_warning = True
            warnings_list.append(
                f"SNR ({ml_input['snr_db']} dB) is outside typical range "
                f"({b['p1']:.1f} to {b['p99']:.1f})."
            )

    return ood_warning, warnings_list


def _sense_channel(channel: Dict, noise_floor_dbm: float, assigned_power_dbm: Optional[float]):
    """
    Runs ONE channel through the existing FFT/PSD + peak-detection pipeline.

    A single 512-point periodogram of pure thermal noise has enough
    variance that its global maximum regularly lands >10 dB above the
    median (a ~48% false-positive rate in testing) — an artifact of
    single-snapshot FFT estimation, not a flaw in detect_peaks() itself.
    We apply Welch's method here (averaging SIM_WELCH_AVERAGES independent
    PSD snapshots in the linear power domain before detection) purely as a
    simulation-side measurement-quality improvement; this does not change
    backend/rf/peak_detector.py or affect the existing /api/spectrum/analyze
    endpoint, which still uses a single snapshot as before.

    Returns (freqs_mhz, psd_dbm, rf_signal_power_dbm, rf_noise_floor_dbm, rf_snr_db, rf_state, peaks).
    """
    carrier_power = (
        assigned_power_dbm if assigned_power_dbm is not None
        else noise_floor_dbm - UNOCCUPIED_SIGNAL_MARGIN_DB
    )

    sample_rate_mhz = max(channel["bandwidth_mhz"] * SIM_SAMPLE_RATE_FACTOR, 20.0)

    freqs_mhz = None
    linear_power_snapshots = []
    for _ in range(SIM_WELCH_AVERAGES):
        source = SimulatedRFSource(
            center_freq_mhz=channel["center_mhz"],
            bandwidth_mhz=channel["bandwidth_mhz"],
            signal_strength_dbm=carrier_power,
            noise_floor_dbm=noise_floor_dbm,
            num_samples=SIM_NUM_SAMPLES,
            sample_rate_mhz=sample_rate_mhz,
        )
        rx_signal, sr = source.get_signal()
        f_mhz, psd_snapshot_dbm = compute_fft_psd(rx_signal, sr, channel["center_mhz"])
        freqs_mhz = f_mhz
        linear_power_snapshots.append(10 ** (psd_snapshot_dbm / 10.0))

    averaged_linear_power = np.mean(linear_power_snapshots, axis=0)
    psd_dbm = 10 * np.log10(np.maximum(averaged_linear_power, 1e-20))

    estimated_noise = estimate_noise_floor(psd_dbm)
    peaks = detect_peaks(freqs_mhz, psd_dbm, estimated_noise, detection_margin_db=10.0)

    if peaks:
        primary = max(peaks, key=lambda p: p["power_dbm"])
        rf_signal_power_dbm = primary["power_dbm"]
        rf_snr_db = primary["snr_db"]
        rf_state = "OCCUPIED"
    else:
        rf_signal_power_dbm = estimated_noise
        rf_snr_db = 0.0
        rf_state = "AVAILABLE"

    return freqs_mhz, psd_dbm, rf_signal_power_dbm, estimated_noise, rf_snr_db, rf_state, peaks


def run_environment_simulation(
    start_frequency_mhz: float,
    end_frequency_mhz: float,
    channel_bandwidth_mhz: float,
    noise_floor_dbm: float,
    num_existing_users: int,
    seed: Optional[int] = None,
    state: str = "Maharashtra",
    city: str = "Mumbai",
    service_type: str = "4G LTE",
) -> Dict:
    """
    Generates the full simulated RF environment and analyzes every channel.

    Returns a dict with:
        channels: list of per-channel RF + ML results
        spectrum_data: {frequencies, power_dbm} composite curve for charting
        occupied_regions / available_regions: [{start_mhz, end_mhz}, ...]
        model_loaded: bool (ML section is skipped gracefully if the
            existing model failed to load, matching the rest of the app)
    """
    channels = generate_channel_grid(start_frequency_mhz, end_frequency_mhz, channel_bandwidth_mhz)
    assignments = assign_existing_signals(channels, num_existing_users, seed=seed)

    model_bundle = get_model_bundle()
    model_metadata = get_model_metadata()
    model_loaded = model_bundle is not None

    all_freqs: List[float] = []
    all_psd: List[float] = []
    results: List[Dict] = []

    for channel in channels:
        assigned_power = assignments.get(channel["channel_id"])
        freqs_mhz, psd_dbm, rf_power, rf_noise, rf_snr, rf_state, peaks = _sense_channel(
            channel, noise_floor_dbm, assigned_power
        )
        all_freqs.extend(freqs_mhz.tolist())
        all_psd.extend(psd_dbm.tolist())

        channel_result = {
            **channel,
            "rf_signal_power_dbm": round(float(rf_power), 2),
            "rf_noise_floor_dbm": round(float(rf_noise), 2),
            "rf_snr_db": round(float(rf_snr), 2),
            "rf_state": rf_state,
        }

        if model_loaded:
            ml_features = extract_ml_features(
                channel["center_mhz"], channel["bandwidth_mhz"], peaks, rf_noise,
                location_info={"state": state, "city": city, "service_type": service_type},
            )
            try:
                ml_result = run_ml_inference(ml_features)
                probability = float(ml_result["ml_activity_probability"])
                best_t = float(ml_result["threshold"])
                ood_warning = bool(ml_result["ood"])
                ood_reasons = list(ml_result["ood_reasons"])
                ml_decision = ml_result["availability"]
            except Exception as e:  # pragma: no cover - defensive, matches existing app style
                probability, best_t, ml_decision = None, model_metadata.get("best_threshold", 0.5), "UNKNOWN"
                ood_warning, ood_reasons = False, [f"ML inference failed: {e}"]

            channel_result.update({
                "ml_probability": round(float(probability), 4) if probability is not None else None,
                "ml_threshold": best_t,
                "ml_decision": ml_decision,
                "ml_ood_warning": ood_warning,
                "ml_ood_reasons": ood_reasons,
            })

            # Final state reconciles RF sensing (ground truth of this simulated
            # environment) with the ML estimate, without ever hiding either value.
            if rf_state == "OCCUPIED":
                final_state = "OCCUPIED"
            elif ml_decision == "AVAILABLE":
                final_state = "AVAILABLE"
            elif ml_decision == "OCCUPIED":
                final_state = "UNAVAILABLE"  # RF sensing sees it free, ML is not confident
            else:
                final_state = rf_state
        else:
            channel_result.update({
                "ml_probability": None,
                "ml_threshold": None,
                "ml_decision": "UNAVAILABLE_MODEL_NOT_LOADED",
                "ml_ood_warning": False,
                "ml_ood_reasons": [],
            })
            final_state = rf_state

        channel_result["state"] = final_state
        results.append(channel_result)

    # Sort composite spectrum curve by frequency (channels are generated in
    # order already, but this keeps the contract explicit and robust).
    order = np.argsort(all_freqs)
    sorted_freqs = np.array(all_freqs)[order]
    sorted_psd = np.array(all_psd)[order]

    downsample_factor = max(1, len(sorted_freqs) // 600)
    freqs_downsampled = sorted_freqs[::downsample_factor]
    psd_downsampled = sorted_psd[::downsample_factor]

    occupied_regions = [
        {"start_mhz": c["start_mhz"], "end_mhz": c["end_mhz"]}
        for c in results if c["rf_state"] == "OCCUPIED"
    ]
    available_regions = [
        {"start_mhz": c["start_mhz"], "end_mhz": c["end_mhz"]}
        for c in results if c["rf_state"] == "AVAILABLE"
    ]

    return {
        "channels": results,
        "spectrum_data": {
            "frequencies": [round(float(f), 3) for f in freqs_downsampled],
            "power_dbm": [round(float(p), 2) for p in psd_downsampled],
        },
        "occupied_regions": occupied_regions,
        "available_regions": available_regions,
        "model_loaded": model_loaded,
        "noise_floor_dbm": noise_floor_dbm,
    }


DEFAULT_SNR_SWEEP_VALUES_DB = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]


def run_snr_sweep(
    signal_power_dbm: float = -80.0,
    start_frequency_mhz: float = 120.0,
    end_frequency_mhz: float = 120.05,
    bandwidth_mhz: float = 0.05,
    hour_of_day: Optional[int] = None,
    day_of_week: Optional[int] = None,
    state: str = "Maharashtra",
    city: str = "Mumbai",
    service_type: str = "4G LTE",
    snr_values_db: Optional[List[float]] = None,
) -> Dict:
    """Run the existing SNR sweep against the new RF-activity architecture.

    The new ML predictor deliberately excludes signal strength because it is the
    pseudo-label-generation measurement. Consequently this experiment reports
    the detector response to signal level while the ML activity probability is
    retained as secondary evidence. No synthetic SNR/noise value is fed to ML.
    """
    model_bundle = get_model_bundle()
    if model_bundle is None:
        return {"model_loaded": False, "points": []}
    snr_values_db = snr_values_db or DEFAULT_SNR_SWEEP_VALUES_DB
    points=[]
    for snr_db in snr_values_db:
        noise_floor_dbm = signal_power_dbm - snr_db
        ml_input = {
            "frequency_mhz": float(start_frequency_mhz),
            "bandwidth_khz": float(bandwidth_mhz*1000.0),
            "signal_strength_dbm": float(signal_power_dbm),
            "iq_available": 0,
        }
        for c in model_metadata.get("features",[]): ml_input.setdefault(c,None)
        try:
            result=run_ml_inference(ml_input)
            points.append({
                "snr_db": snr_db, "noise_floor_dbm": round(noise_floor_dbm,2),
                "probability": round(float(result["ml_activity_probability"]),4),
                "decision": result["availability"], "threshold": result["threshold"],
                "ood_warning": result["ood"], "ood_reasons": result["ood_reasons"],
                "detector_activity": result["detector_activity"],
            })
        except Exception as e:
            points.append({"snr_db":snr_db,"noise_floor_dbm":round(noise_floor_dbm,2),"probability":None,"decision":"ERROR","error":str(e)})
    return {"model_loaded": True, "points": points}

