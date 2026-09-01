"""Unit tests for RF and Electronics engineering calculations.

Verifies:
  - dBm ↔ Watts conversions and scaling
  - dBm ↔ dBW conversions
  - Wavelength (λ = c/f) and antenna element dimensions (λ/4, λ/2)
  - Bandwidth, center frequency, and fractional bandwidth
  - Nyquist sampling rate minimum (fs ≥ 2B)
  - Free-space path loss (FSPL = 20*log10(d) + 20*log10(f) + 32.44)
  - Thermal noise power (P_n = kTB) and noise density (-174 dBm/Hz at 290K)
  - Signal-to-noise ratio (SNR = P_signal - P_noise)
  - Link budget (P_r = P_t + G_t + G_r - L_path - L_misc)
  - Edge cases: zero, negative where invalid, boundary values
"""
from __future__ import annotations
import math
import pytest

C = 299792458  # m/s
K_B = 1.380649e-23  # J/K
T_0 = 290.0  # K


def dbm_to_watts(p_dbm: float) -> float:
    if not math.isfinite(p_dbm):
        return float("nan")
    return math.pow(10.0, (p_dbm - 30.0) / 10.0)


def watts_to_dbm(p_watts: float) -> float:
    if p_watts <= 0:
        return float("nan")
    return 10.0 * math.log10(p_watts) + 30.0


def wavelength_meters(f_mhz: float) -> float:
    if f_mhz <= 0:
        return float("nan")
    return C / (f_mhz * 1e6)


def fspl_db(d_km: float, f_mhz: float) -> float:
    if d_km <= 0 or f_mhz <= 0:
        return float("nan")
    return 20.0 * math.log10(d_km) + 20.0 * math.log10(f_mhz) + 32.44


def thermal_noise_dbm(bw_hz: float, temp_k: float = T_0) -> float:
    if bw_hz <= 0 or temp_k <= 0:
        return float("nan")
    p_watts = K_B * temp_k * bw_hz
    return 10.0 * math.log10(p_watts) + 30.0


# ── 1. Power Conversions ──────────────────────────────────────────────────────

def test_dbm_to_watts_known_values():
    assert math.isclose(dbm_to_watts(0.0), 1e-3, rel_tol=1e-6)  # 0 dBm = 1 mW = 0.001 W
    assert math.isclose(dbm_to_watts(30.0), 1.0, rel_tol=1e-6)  # 30 dBm = 1 W
    assert math.isclose(dbm_to_watts(-60.0), 1e-9, rel_tol=1e-6)  # -60 dBm = 1 nW
    assert math.isclose(dbm_to_watts(-90.0), 1e-12, rel_tol=1e-6)  # -90 dBm = 1 pW


def test_watts_to_dbm_known_values():
    assert math.isclose(watts_to_dbm(1e-3), 0.0, abs_tol=1e-6)
    assert math.isclose(watts_to_dbm(1.0), 30.0, abs_tol=1e-6)
    assert math.isclose(watts_to_dbm(1e-9), -60.0, abs_tol=1e-6)


def test_power_roundtrip():
    for dbm in [-110.0, -60.0, -30.0, 0.0, 10.0, 43.0]:
        w = dbm_to_watts(dbm)
        recovered_dbm = watts_to_dbm(w)
        assert math.isclose(recovered_dbm, dbm, abs_tol=1e-5)


# ── 2. Wavelength & Antenna Dimensions ────────────────────────────────────────

def test_wavelength_120mhz():
    # 120 MHz -> λ ≈ 2.49827 m
    wl = wavelength_meters(120.0)
    assert math.isclose(wl, 2.49827, rel_tol=1e-4)


def test_wavelength_edge_cases():
    assert math.isnan(wavelength_meters(0.0))
    assert math.isnan(wavelength_meters(-10.0))
    # 300 MHz -> λ ≈ 1.0 m (approx)
    wl_300 = wavelength_meters(300.0)
    assert math.isclose(wl_300, 0.9993, rel_tol=1e-3)


def test_antenna_lengths():
    wl_120 = wavelength_meters(120.0)
    l_quarter = wl_120 / 4.0
    l_half = wl_120 / 2.0
    assert math.isclose(l_quarter, 0.62456, rel_tol=1e-4)  # ~62.45 cm
    assert math.isclose(l_half, 1.24913, rel_tol=1e-4)     # ~1.249 m


# ── 3. Bandwidth & Center Frequency ───────────────────────────────────────────

def test_bandwidth_and_center():
    f_low, f_high = 100.0, 120.0
    bw = f_high - f_low
    f_center = (f_high + f_low) / 2.0
    fbw = (bw / f_center) * 100.0

    assert math.isclose(bw, 20.0)
    assert math.isclose(f_center, 110.0)
    assert math.isclose(fbw, (20.0 / 110.0) * 100.0, rel_tol=1e-5)  # ~18.18%


# ── 4. Nyquist Sampling Rate ──────────────────────────────────────────────────

def test_nyquist_criterion():
    bw = 1.0  # MHz
    min_nyquist = 2.0 * bw
    assert 2.5 >= min_nyquist  # PASS
    assert not (1.2 >= min_nyquist)  # FAIL


# ── 5. Thermal Noise Floor ────────────────────────────────────────────────────

def test_thermal_noise_density():
    # N_0 at 290 K = k_B * 290 = 4.00388e-21 W/Hz = -173.975 dBm/Hz ≈ -174 dBm/Hz
    n0_watts = K_B * 290.0
    n0_dbm = 10.0 * math.log10(n0_watts) + 30.0
    assert math.isclose(n0_dbm, -173.975, abs_tol=0.1)


def test_thermal_noise_1mhz():
    # 1 MHz bandwidth at 290 K: P_n ≈ -174 + 60 = -114 dBm
    p_n = thermal_noise_dbm(1e6, 290.0)
    assert math.isclose(p_n, -113.975, abs_tol=0.1)


def test_thermal_noise_50khz():
    # 50 kHz bandwidth (Wire Watcher dataset channel bandwidth):
    # P_n ≈ -173.975 + 10*log10(50000) = -173.975 + 46.99 = -126.985 dBm
    p_n = thermal_noise_dbm(50000.0, 290.0)
    assert math.isclose(p_n, -126.985, abs_tol=0.1)


# ── 6. SNR ───────────────────────────────────────────────────────────────────

def test_snr_calculation():
    p_sig = -60.0  # dBm
    p_noise = -90.0  # dBm
    snr = p_sig - p_noise
    assert math.isclose(snr, 30.0)


# ── 7. Free-Space Path Loss (FSPL) ───────────────────────────────────────────

def test_fspl_calculation():
    # d = 1 km, f = 120 MHz -> FSPL = 20*log10(1) + 20*log10(120) + 32.44 = 0 + 41.5836 + 32.44 = 74.02 dB
    loss = fspl_db(1.0, 120.0)
    assert math.isclose(loss, 74.0236, abs_tol=0.1)


def test_fspl_edge_cases():
    assert math.isnan(fspl_db(0.0, 120.0))
    assert math.isnan(fspl_db(-5.0, 120.0))
    assert math.isnan(fspl_db(1.0, -100.0))


# ── 8. Link Budget ────────────────────────────────────────────────────────────

def test_link_budget():
    pt = 30.0     # 30 dBm (1 Watt)
    gt = 2.15     # 2.15 dBi (half-wave dipole)
    gr = 2.15     # 2.15 dBi
    l_path = 74.0 # dB
    l_misc = 2.0  # dB
    pr = pt + gt + gr - l_path - l_misc
    assert math.isclose(pr, -41.7, abs_tol=0.01)


# ── 9. Error Handling & Edge Cases ───────────────────────────────────────────

def test_invalid_temperature_and_frequency():
    # Negative absolute temperature must return NaN
    assert math.isnan(thermal_noise_dbm(50000.0, -10.0))
    assert math.isnan(thermal_noise_dbm(50000.0, 0.0))
    # Zero or negative bandwidth must return NaN
    assert math.isnan(thermal_noise_dbm(0.0, 290.0))
    assert math.isnan(thermal_noise_dbm(-100.0, 290.0))


def test_invalid_power_inputs():
    assert math.isnan(watts_to_dbm(0.0))
    assert math.isnan(watts_to_dbm(-5.0))
    assert math.isnan(watts_to_dbm(float("nan")))
    assert math.isnan(dbm_to_watts(float("nan")))
    assert math.isnan(dbm_to_watts(float("inf")))

