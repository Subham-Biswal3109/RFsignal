def extract_ml_features(center_freq_mhz: float, bandwidth_mhz: float, peaks: list[dict], noise_floor_dbm: float, location_info: dict = None) -> dict:
    """Map simulator sensing output to the new RF-activity model contract.

    The simulator does not have recorded I/Q, so no I/Q-derived feature is
    fabricated. Signal strength remains a measured/simulated detector input and
    is excluded from the ML predictor set by the trained pipeline.
    """
    if peaks:
        primary=max(peaks,key=lambda p:p["power_dbm"])
        signal_power=float(primary["power_dbm"])
    else:
        signal_power=float(noise_floor_dbm)
    return {
        "frequency_mhz": float(center_freq_mhz),
        "bandwidth_khz": float(bandwidth_mhz*1000.0),
        "signal_strength_dbm": signal_power,
        "iq_available": 0,
    }
