from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Union


class PredictionRequest(BaseModel):
    """Runtime contract for the RF activity/availability layer.

    Signal strength is a REAL_MEASURED detector input. It is intentionally not
    a predictor of the ML classifier because the pseudo-label is generated from
    a training-only signal-strength threshold.
    """
    frequency_mhz: float = Field(..., gt=0)
    bandwidth_khz: float = Field(..., gt=0)
    signal_strength_dbm: float
    iq_available: int = Field(0, ge=0, le=1)
    iq_rms_magnitude: Optional[float] = None
    iq_magnitude_variance: Optional[float] = None
    iq_peak_magnitude: Optional[float] = None
    iq_crest_factor: Optional[float] = None
    iq_p10: Optional[float] = None
    iq_p50: Optional[float] = None
    iq_p90: Optional[float] = None
    iq_phase_concentration: Optional[float] = None
    iq_spectral_entropy: Optional[float] = None
    iq_spectral_peak_ratio: Optional[float] = None
    timestamp: Optional[str] = None
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class SimulationUser(BaseModel):
    """One entry in a multi-user allocation request (Spectrum Simulation module)."""
    user_id: Optional[str] = None
    requested_bandwidth_mhz: float = Field(..., gt=0, description="Requested bandwidth in MHz")


class SimulationRequest(BaseModel):
    """
    Request schema for the new, separate Spectrum Simulation module
    (POST /api/simulation/run). This does not replace or alter
    PredictionRequest / /api/predict in any way.
    """
    start_frequency_mhz: float = Field(..., gt=0, description="Start of the simulated band in MHz")
    end_frequency_mhz: float = Field(..., gt=0, description="End of the simulated band in MHz")
    channel_bandwidth_mhz: float = Field(..., gt=0, description="Width of each simulated channel in MHz")
    noise_floor_dbm: float = Field(-100.0, description="Simulated ambient noise floor in dBm")
    num_existing_users: int = Field(0, ge=0, le=200, description="Number of pre-existing occupying signals to place")
    seed: Optional[int] = Field(None, description="Random seed for reproducible simulation runs")

    mode: str = Field("ml_assisted", description="basic | ml_assisted | multi_user")
    requested_bandwidth_mhz: Optional[float] = Field(
        None, gt=0, description="Bandwidth requested by a new single user (basic/ml_assisted modes)"
    )
    users: Optional[List[SimulationUser]] = Field(
        None, description="Sequential user requests for multi_user mode"
    )

    state: str = Field("Maharashtra", description="State passed through to the existing ML model's location features")
    city: str = Field("Mumbai", description="City passed through to the existing ML model's location features")
    service_type: str = Field("4G LTE", description="Service type passed through to the existing ML model")

    @model_validator(mode='after')
    def check_consistency(self):
        if self.end_frequency_mhz <= self.start_frequency_mhz:
            raise ValueError("end_frequency_mhz must be greater than start_frequency_mhz")
        if self.channel_bandwidth_mhz > (self.end_frequency_mhz - self.start_frequency_mhz):
            raise ValueError("channel_bandwidth_mhz cannot be larger than the total frequency range")
        if self.mode not in ("basic", "ml_assisted", "multi_user"):
            raise ValueError("mode must be one of: basic, ml_assisted, multi_user")
        if self.mode == "multi_user" and not self.users:
            raise ValueError("multi_user mode requires a non-empty 'users' list")
        if self.mode != "multi_user" and self.requested_bandwidth_mhz is None:
            raise ValueError("requested_bandwidth_mhz is required for basic/ml_assisted modes")
        return self


class SnrSweepRequest(BaseModel):
    """Request schema for POST /api/simulation/snr-sweep (section 18 experiment)."""
    signal_power_dbm: float = Field(-80.0, description="Fixed signal power in dBm used across the sweep")
    start_frequency_mhz: float = Field(1800.0, gt=0)
    end_frequency_mhz: float = Field(1810.0, gt=0)
    bandwidth_mhz: float = Field(10.0, gt=0)
    state: str = Field("Maharashtra")
    city: str = Field("Mumbai")
    service_type: str = Field("4G LTE")
    snr_values_db: Optional[List[float]] = Field(None, description="Custom SNR sweep points; defaults to 0-30dB in 5dB steps")

    @model_validator(mode='after')
    def check_consistency(self):
        if self.end_frequency_mhz <= self.start_frequency_mhz:
            raise ValueError("end_frequency_mhz must be greater than start_frequency_mhz")
        return self


class JammingSampleRequest(BaseModel):
    """
    Request schema for POST /api/jamming/predict — the RF Interference/
    Jamming Detector (a SEPARATE model from spectrum availability).

    Two ways to call this:
      1. sample_id: run inference on one of the held-out demo samples
         shipped with the model (ml/artifacts/jamming_detector_test_samples.json).
      2. features + band + scan_mode: run inference on a fully custom
         feature vector (advanced/API use).
    """
    sample_id: Optional[str] = Field(None, description="e.g. 'test_12345' from GET /api/jamming/samples")
    features: Optional[dict] = Field(None, description="Raw feature dict matching the model's expected stat columns")
    band: Optional[str] = Field(None, description="'2.4GHz' or '5GHz', required if 'features' is provided")
    scan_mode: Optional[str] = Field(None, description="'active' or 'passive', required if 'features' is provided")

    @model_validator(mode='after')
    def check_consistency(self):
        if not self.sample_id and not self.features:
            raise ValueError("Provide either 'sample_id' or 'features' (+ band + scan_mode)")
        if self.features and (not self.band or not self.scan_mode):
            raise ValueError("'band' and 'scan_mode' are required when providing 'features' directly")
        return self


class SdrConfigRequest(BaseModel):
    """Request schema for POST /api/sdr/configure."""
    center_freq_mhz: Optional[float] = Field(None, gt=0, description="Center frequency in MHz (gt 0)")
    sample_rate_mhz: Optional[float] = Field(None, ge=0.225, le=3.2, description="Sample rate in MHz [0.225, 3.2]")
    gain: Optional[Union[float, str]] = Field(None, description="Gain in dB or 'auto'")
    buffer_size: Optional[int] = Field(None, ge=256, le=262144, description="Capture buffer size [256, 262144]")

    @model_validator(mode='after')
    def check_gain_validity(self):
        if self.gain is not None:
            if isinstance(self.gain, str):
                if self.gain.lower() != "auto":
                    raise ValueError("String gain must be 'auto'")
            elif isinstance(self.gain, (int, float)):
                if not float("-inf") < float(self.gain) < float("inf"):
                    raise ValueError("Numeric gain must be finite")
            else:
                raise ValueError("Gain must be a float or string 'auto'")
        return self


class SdrCaptureRequest(BaseModel):
    """Request schema for POST /api/sdr/capture."""
    num_samples: Optional[int] = Field(1024, ge=256, le=262144, description="Number of complex IQ samples to capture [256, 262144]")
    center_freq_mhz: Optional[float] = Field(None, gt=0)
    sample_rate_mhz: Optional[float] = Field(None, ge=0.225, le=3.2)
    gain: Optional[Union[float, str]] = Field(None)


class AllocationRequest(BaseModel):
    """Request schema for POST /api/allocation/recommend."""
    start_freq_mhz: float = Field(70.0, gt=0, description="Start frequency of analyzed band in MHz")
    end_freq_mhz: float = Field(160.0, gt=0, description="End frequency of analyzed band in MHz")
    channel_bw_mhz: float = Field(0.2, gt=0, description="Bandwidth of each candidate channel in MHz")
    guard_band_mhz: float = Field(0.05, ge=0, description="Guard band between candidate channels in MHz")
    noise_floor_dbm: float = Field(-100.0, description="Observed or estimated noise floor in dBm")
    observed_power_dbm: Optional[float] = Field(None, description="Observed signal power in dBm")
    location: Optional[str] = Field(None, description="Optional region/service tag")

    @model_validator(mode='after')
    def validate_range(self):
        if self.end_freq_mhz <= self.start_freq_mhz:
            raise ValueError("end_freq_mhz must be greater than start_freq_mhz")
        if self.channel_bw_mhz > (self.end_freq_mhz - self.start_freq_mhz):
            raise ValueError("channel_bw_mhz cannot exceed total frequency range")
        return self



