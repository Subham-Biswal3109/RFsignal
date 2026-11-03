/**
 * Wire Watcher — RF & Electronics Engineering Calculation Engine
 *
 * Core Physical Constants:
 *  - SPEED_OF_LIGHT (c): 299,792,458 m/s (exact SI definition)
 *  - BOLTZMANN_CONSTANT (k_B): 1.380649e-23 J/K (exact SI definition)
 *  - STANDARD_TEMP_K (T_0): 290.0 K (IEEE standard reference temperature)
 */

export const SPEED_OF_LIGHT = 299792458; // m/s
export const BOLTZMANN_CONSTANT = 1.380649e-23; // J/K
export const STANDARD_TEMP_K = 290.0; // Kelvin (~16.85 °C / standard noise ref)

export interface CalculationStep {
  name: string;
  formula: string;
  substitution: string;
  result: string;
  numericResult: number;
  unit: string;
  notes?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Power Conversions (dBm ↔ Watts ↔ dBW ↔ Unit Scaling)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Convert power in dBm to Watts:
 * P(W) = 10^((P(dBm) - 30) / 10)
 */
export function dbmToWatts(pDbm: number): CalculationStep {
  if (!Number.isFinite(pDbm)) {
    return {
      name: "Power in Watts",
      formula: "P(W) = 10^((P_{dBm} - 30) / 10)",
      substitution: "Invalid input",
      result: "NaN",
      numericResult: NaN,
      unit: "W",
    };
  }
  const exponent = (pDbm - 30) / 10;
  const pWatts = Math.pow(10, exponent);

  let formattedResult: string;
  let scaledUnit: string;

  if (pWatts >= 1) {
    formattedResult = `${pWatts.toFixed(4)} W`;
    scaledUnit = "W";
  } else if (pWatts >= 1e-3) {
    formattedResult = `${(pWatts * 1e3).toFixed(4)} mW`;
    scaledUnit = "mW";
  } else if (pWatts >= 1e-6) {
    formattedResult = `${(pWatts * 1e6).toFixed(4)} µW`;
    scaledUnit = "µW";
  } else if (pWatts >= 1e-9) {
    formattedResult = `${(pWatts * 1e9).toFixed(4)} nW`;
    scaledUnit = "nW";
  } else if (pWatts >= 1e-12) {
    formattedResult = `${(pWatts * 1e12).toFixed(4)} pW`;
    scaledUnit = "pW";
  } else {
    formattedResult = `${pWatts.toExponential(4)} W`;
    scaledUnit = "W";
  }

  return {
    name: "dBm to Watts",
    formula: "P(W) = 10^{(P_{dBm} - 30) / 10}",
    substitution: `P = 10^{(${pDbm.toFixed(2)} - 30) / 10} = 10^{${exponent.toFixed(3)}}`,
    result: formattedResult,
    numericResult: pWatts,
    unit: scaledUnit,
    notes: `${pDbm} dBm is equivalent to ${pWatts.toExponential(4)} Watts (${scaledUnit})`,
  };
}

/**
 * Convert power in Watts to dBm:
 * P(dBm) = 10 * log10(P(W)) + 30
 */
export function wattsToDbm(pWatts: number): CalculationStep {
  if (!Number.isFinite(pWatts) || pWatts <= 0) {
    return {
      name: "Watts to dBm",
      formula: "P(dBm) = 10 \\cdot \\log_{10}(P_W) + 30",
      substitution: "Power must be > 0 W",
      result: "Invalid",
      numericResult: NaN,
      unit: "dBm",
    };
  }
  const pDbm = 10 * Math.log10(pWatts) + 30;
  return {
    name: "Watts to dBm",
    formula: "P(dBm) = 10 \\cdot \\log_{10}(P_W) + 30",
    substitution: `P = 10 \\cdot \\log_{10}(${pWatts.toExponential(4)}) + 30`,
    result: `${pDbm.toFixed(2)} dBm`,
    numericResult: pDbm,
    unit: "dBm",
  };
}

/**
 * Convert dBm to dBW:
 * P(dBW) = P(dBm) - 30
 */
export function dbmToDbw(pDbm: number): CalculationStep {
  if (!Number.isFinite(pDbm)) {
    return {
      name: "dBm to dBW",
      formula: "P(dBW) = P(dBm) - 30",
      substitution: "Invalid input",
      result: "Invalid",
      numericResult: NaN,
      unit: "dBW",
    };
  }
  const pDbw = pDbm - 30;
  return {
    name: "dBm to dBW",
    formula: "P(dBW) = P(dBm) - 30",
    substitution: `P = ${pDbm.toFixed(2)} - 30`,
    result: `${pDbw.toFixed(2)} dBW`,
    numericResult: pDbw,
    unit: "dBW",
  };
}


// ─────────────────────────────────────────────────────────────────────────────
// 2. Frequency & Wavelength & Antenna Dimensions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Calculate wavelength from frequency in MHz:
 * λ = c / f_Hz
 */
export function frequencyToWavelength(freqMhz: number): CalculationStep {
  if (!Number.isFinite(freqMhz) || freqMhz <= 0) {
    return {
      name: "Wavelength (λ)",
      formula: "\\lambda = \\frac{c}{f}",
      substitution: "Frequency must be > 0 MHz",
      result: "Invalid",
      numericResult: NaN,
      unit: "m",
    };
  }
  const fHz = freqMhz * 1e6;
  const lambda = SPEED_OF_LIGHT / fHz;

  let formattedResult: string;
  let unit: string;
  if (lambda >= 1) {
    formattedResult = `${lambda.toFixed(4)} m`;
    unit = "m";
  } else if (lambda >= 0.01) {
    formattedResult = `${(lambda * 100).toFixed(2)} cm`;
    unit = "cm";
  } else {
    formattedResult = `${(lambda * 1000).toFixed(2)} mm`;
    unit = "mm";
  }

  return {
    name: "Wavelength (λ)",
    formula: "\\lambda = \\frac{c}{f} = \\frac{299,792,458\\text{ m/s}}{f\\text{ (Hz)}}",
    substitution: `\\lambda = \\frac{299,792,458}{${freqMhz.toFixed(3)} \\times 10^6\\text{ Hz}} = \\frac{299,792,458}{${fHz.toLocaleString()}\\text{ Hz}}`,
    result: formattedResult,
    numericResult: lambda,
    unit: unit,
    notes: `At ${freqMhz} MHz (c = 299,792,458 m/s), electromagnetic wavelength λ is ${lambda.toFixed(4)} meters.`,
  };
}

/**
 * Approximate Quarter-Wave and Half-Wave antenna dimensions:
 * L_quarter ≈ λ / 4, L_half ≈ λ / 2
 */
export function calculateAntennaDimensions(freqMhz: number): {
  wavelength: CalculationStep;
  quarterWave: CalculationStep;
  halfWave: CalculationStep;
} {
  const wl = frequencyToWavelength(freqMhz);
  const lambda = wl.numericResult;

  if (isNaN(lambda)) {
    const invalidStep = (name: string, formula: string): CalculationStep => ({
      name,
      formula,
      substitution: "Invalid frequency",
      result: "Invalid",
      numericResult: NaN,
      unit: "m",
    });
    return {
      wavelength: wl,
      quarterWave: invalidStep("Quarter-Wave Monopole (λ/4)", "L \\approx \\lambda / 4"),
      halfWave: invalidStep("Half-Wave Dipole (λ/2)", "L \\approx \\lambda / 2"),
    };
  }

  const lQuarter = lambda / 4;
  const lHalf = lambda / 2;

  return {
    wavelength: wl,
    quarterWave: {
      name: "Quarter-Wave Monopole (λ/4)",
      formula: "L_{\\lambda/4} = \\frac{\\lambda}{4} = \\frac{c}{4f}",
      substitution: `L = \\frac{${lambda.toFixed(4)}\\text{ m}}{4}`,
      result: `${lQuarter >= 1 ? lQuarter.toFixed(4) + " m" : (lQuarter * 100).toFixed(2) + " cm"}`,
      numericResult: lQuarter,
      unit: lQuarter >= 1 ? "m" : "cm",
      notes: "Standard theoretical whip/monopole resonant element length before velocity factor corrections.",
    },
    halfWave: {
      name: "Half-Wave Dipole (λ/2)",
      formula: "L_{\\lambda/2} = \\frac{\\lambda}{2} = \\frac{c}{2f}",
      substitution: `L = \\frac{${lambda.toFixed(4)}\\text{ m}}{2}`,
      result: `${lHalf >= 1 ? lHalf.toFixed(4) + " m" : (lHalf * 100).toFixed(2) + " cm"}`,
      numericResult: lHalf,
      unit: lHalf >= 1 ? "m" : "cm",
      notes: "Standard theoretical center-fed dipole total resonant tip-to-tip length.",
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Bandwidth, Center Frequency & Fractional Bandwidth
// ─────────────────────────────────────────────────────────────────────────────

export interface BandwidthMetrics {
  bandwidthMhz: CalculationStep;
  centerFrequencyMhz: CalculationStep;
  fractionalBandwidthPercent: CalculationStep;
}

export function calculateBandwidthMetrics(fLowMhz: number, fHighMhz: number): BandwidthMetrics {
  if (!Number.isFinite(fLowMhz) || !Number.isFinite(fHighMhz) || fHighMhz <= fLowMhz || fLowMhz <= 0) {
    const err: CalculationStep = {
      name: "Bandwidth Metrics",
      formula: "\\text{BW} = f_{high} - f_{low}",
      substitution: "f_high must be strictly greater than f_low (> 0)",
      result: "Invalid",
      numericResult: NaN,
      unit: "MHz",
    };
    return { bandwidthMhz: err, centerFrequencyMhz: err, fractionalBandwidthPercent: err };
  }

  const bw = fHighMhz - fLowMhz;
  const fCenter = (fHighMhz + fLowMhz) / 2;
  const fbwFraction = bw / fCenter;
  const fbwPercent = fbwFraction * 100;

  return {
    bandwidthMhz: {
      name: "Bandwidth (BW)",
      formula: "\\text{BW} = f_{high} - f_{low}",
      substitution: `\\text{BW} = ${fHighMhz.toFixed(3)}\\text{ MHz} - ${fLowMhz.toFixed(3)}\\text{ MHz}`,
      result: `${bw.toFixed(3)} MHz (${(bw * 1000).toFixed(1)} kHz)`,
      numericResult: bw,
      unit: "MHz",
    },
    centerFrequencyMhz: {
      name: "Center Frequency (f_0)",
      formula: "f_0 = \\frac{f_{high} + f_{low}}{2}",
      substitution: `f_0 = \\frac{${fHighMhz.toFixed(3)} + ${fLowMhz.toFixed(3)}}{2}`,
      result: `${fCenter.toFixed(3)} MHz`,
      numericResult: fCenter,
      unit: "MHz",
    },
    fractionalBandwidthPercent: {
      name: "Fractional Bandwidth (FBW)",
      formula: "\\text{FBW} = \\frac{\\text{BW}}{f_0} \\times 100\\%",
      substitution: `\\text{FBW} = \\frac{${bw.toFixed(3)}\\text{ MHz}}{${fCenter.toFixed(3)}\\text{ MHz}} \\times 100\\%`,
      result: `${fbwPercent.toFixed(3)}%`,
      numericResult: fbwPercent,
      unit: "%",
      notes: fbwPercent < 1 ? "Narrowband channel (FBW < 1%)" : fbwPercent < 20 ? "Moderate bandwidth" : "Wideband/UWB signal",
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Nyquist Sampling Rate Check
// ─────────────────────────────────────────────────────────────────────────────

export interface NyquistCheckResult extends CalculationStep {
  passed: boolean;
  minNyquistRateMsps: number;
}

export function checkNyquistSampling(bandwidthMhz: number, sampleRateMsps: number): NyquistCheckResult {
  if (!Number.isFinite(bandwidthMhz) || !Number.isFinite(sampleRateMsps) || bandwidthMhz <= 0 || sampleRateMsps <= 0) {
    return {
      name: "Nyquist Sampling Criterion",
      formula: "f_s \\ge 2 \\times \\text{BW}",
      substitution: "Inputs must be positive numbers",
      result: "Invalid",
      numericResult: NaN,
      unit: "MS/s",
      passed: false,
      minNyquistRateMsps: NaN,
    };
  }

  const minNyquist = 2 * bandwidthMhz;
  const passed = sampleRateMsps >= minNyquist;

  return {
    name: "Nyquist Sampling Criterion",
    formula: "f_s \\ge 2 \\cdot \\text{BW} \\quad (\\text{Nyquist-Shannon theorem for real sampling})",
    substitution: `f_s = ${sampleRateMsps.toFixed(3)}\\text{ MS/s} \\; \\gtrless \\; 2 \\times ${bandwidthMhz.toFixed(3)}\\text{ MHz} = ${minNyquist.toFixed(3)}\\text{ MS/s}`,
    result: passed ? "PASS (Adequate Sampling)" : "FAIL (Aliasing / Insufficient Rate)",
    numericResult: minNyquist,
    unit: "MS/s",
    passed,
    minNyquistRateMsps: minNyquist,
    notes: passed
      ? `Sample rate ${sampleRateMsps} MS/s satisfies Nyquist minimum (${minNyquist} MS/s). For complex I/Q baseband, f_s ≥ BW is sufficient.`
      : `Sample rate ${sampleRateMsps} MS/s is below the 2× Nyquist minimum (${minNyquist} MS/s). Spectral aliasing will occur without oversampling.`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Thermal Noise Floor (Johnson-Nyquist Noise)
// ─────────────────────────────────────────────────────────────────────────────

export interface ThermalNoiseResult {
  noisePowerWatts: CalculationStep;
  noisePowerDbm: CalculationStep;
  noiseDensityDbmHz: CalculationStep;
}

export function calculateThermalNoise(bandwidthHz: number, tempKelvin: number = STANDARD_TEMP_K): ThermalNoiseResult {
  if (!Number.isFinite(bandwidthHz) || !Number.isFinite(tempKelvin) || bandwidthHz <= 0 || tempKelvin <= 0) {
    const err: CalculationStep = {
      name: "Thermal Noise",
      formula: "P_n = k_B \\cdot T \\cdot B",
      substitution: "Bandwidth and Temperature must be > 0",
      result: "Invalid",
      numericResult: NaN,
      unit: "dBm",
    };
    return { noisePowerWatts: err, noisePowerDbm: err, noiseDensityDbmHz: err };
  }

  // P_n (Watts) = k_B * T * B
  const pWatts = BOLTZMANN_CONSTANT * tempKelvin * bandwidthHz;
  // P_n (dBm) = 10 * log10(P_n / 1mW) = 10 * log10(k_B * T * B * 1000)
  const pDbm = 10 * Math.log10(pWatts) + 30;
  // Noise Spectral Density N_0 = k_B * T in dBm/Hz
  const n0Watts = BOLTZMANN_CONSTANT * tempKelvin;
  const n0DbmHz = 10 * Math.log10(n0Watts) + 30;

  return {
    noisePowerWatts: {
      name: "Thermal Noise Power (Watts)",
      formula: "P_n = k_B \\cdot T \\cdot B",
      substitution: `P_n = (1.380649 \\times 10^{-23}\\text{ J/K}) \\times ${tempKelvin.toFixed(1)}\\text{ K} \\times ${bandwidthHz.toLocaleString()}\\text{ Hz}`,
      result: `${pWatts.toExponential(4)} W`,
      numericResult: pWatts,
      unit: "W",
    },
    noisePowerDbm: {
      name: "Thermal Noise Floor (dBm)",
      formula: "P_{n(dBm)} = 10 \\cdot \\log_{10}(k_B T B / 10^{-3}) = N_0 + 10\\log_{10}(B)",
      substitution: `P_{n(dBm)} = ${n0DbmHz.toFixed(2)}\\text{ dBm/Hz} + 10\\log_{10}(${bandwidthHz.toLocaleString()}\\text{ Hz}) = ${pDbm.toFixed(2)}\\text{ dBm}`,
      result: `${pDbm.toFixed(2)} dBm`,
      numericResult: pDbm,
      unit: "dBm",
      notes: `Theoretical Johnson-Nyquist thermal noise floor at T = ${tempKelvin} K across ${bandwidthHz >= 1e6 ? (bandwidthHz / 1e6).toFixed(2) + " MHz" : (bandwidthHz / 1e3).toFixed(1) + " kHz"}.`,
    },
    noiseDensityDbmHz: {
      name: "Noise Spectral Density (N_0)",
      formula: "N_0 = 10 \\cdot \\log_{10}(k_B T \\times 1000)",
      substitution: `N_0 = 10\\log_{10}((1.380649 \\times 10^{-23}) \\times ${tempKelvin.toFixed(1)} \\times 10^3) = ${n0DbmHz.toFixed(2)}\\text{ dBm/Hz}`,
      result: `${n0DbmHz.toFixed(2)} dBm/Hz`,
      numericResult: n0DbmHz,
      unit: "dBm/Hz",
      notes: tempKelvin === 290 ? "Standard room-temperature reference density is approximately -174.00 dBm/Hz." : `At ${tempKelvin} K (${(tempKelvin - 273.15).toFixed(1)} °C).`,
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Signal-to-Noise Ratio (SNR)
// ─────────────────────────────────────────────────────────────────────────────

export function calculateSNR(pSignalDbm: number, pNoiseDbm: number): CalculationStep {
  if (!Number.isFinite(pSignalDbm) || !Number.isFinite(pNoiseDbm)) {
    return {
      name: "Signal-to-Noise Ratio (SNR)",
      formula: "\\text{SNR(dB)} = P_{signal(dBm)} - P_{noise(dBm)}",
      substitution: "Inputs must be valid numbers",
      result: "Invalid",
      numericResult: NaN,
      unit: "dB",
    };
  }

  const snrDb = pSignalDbm - pNoiseDbm;
  const snrLinear = Math.pow(10, snrDb / 10);

  return {
    name: "Signal-to-Noise Ratio (SNR)",
    formula: "\\text{SNR(dB)} = P_{signal(dBm)} - P_{noise(dBm)} = 10 \\cdot \\log_{10}\\left(\\frac{P_{signal(W)}}{P_{noise(W)}}\\right)",
    substitution: `\\text{SNR} = (${pSignalDbm.toFixed(2)}\\text{ dBm}) - (${pNoiseDbm.toFixed(2)}\\text{ dBm}) = ${snrDb.toFixed(2)}\\text{ dB}`,
    result: `${snrDb.toFixed(2)} dB (linear ratio: ${snrLinear >= 1000 ? snrLinear.toExponential(2) : snrLinear.toFixed(2)})`,
    numericResult: snrDb,
    unit: "dB",
    notes: snrDb < 0
      ? "Signal is submerged below noise floor (SNR < 0 dB)."
      : snrDb < 10
      ? "Low SNR (< 10 dB) — reliable demodulation requires robust coding / high processing gain."
      : "Adequate to high SNR (≥ 10 dB).",
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Free-Space Path Loss (FSPL)
// ─────────────────────────────────────────────────────────────────────────────

export function calculateFSPL(distanceKm: number, freqMhz: number): CalculationStep {
  if (!Number.isFinite(distanceKm) || !Number.isFinite(freqMhz) || distanceKm <= 0 || freqMhz <= 0) {
    return {
      name: "Free-Space Path Loss (FSPL)",
      formula: "\\text{FSPL(dB)} = 20\\log_{10}(d_{km}) + 20\\log_{10}(f_{MHz}) + 32.44",
      substitution: "Distance (km) and Frequency (MHz) must be > 0",
      result: "Invalid",
      numericResult: NaN,
      unit: "dB",
    };
  }

  // FSPL(dB) = 20 log10(d_km) + 20 log10(f_MHz) + 32.44 (exact constant: 20*log10(4*pi/c * 10^3 * 10^6) = 32.444)
  const termD = 20 * Math.log10(distanceKm);
  const termF = 20 * Math.log10(freqMhz);
  const fsplDb = termD + termF + 32.44;

  return {
    name: "Free-Space Path Loss (FSPL)",
    formula: "\\text{FSPL(dB)} = 20\\log_{10}(d_{\\text{km}}) + 20\\log_{10}(f_{\\text{MHz}}) + 32.44",
    substitution: `\\text{FSPL} = 20\\log_{10}(${distanceKm.toFixed(3)}) + 20\\log_{10}(${freqMhz.toFixed(3)}) + 32.44 = ${termD.toFixed(2)} + ${termF.toFixed(2)} + 32.44 = ${fsplDb.toFixed(2)}\\text{ dB}`,
    result: `${fsplDb.toFixed(2)} dB`,
    numericResult: fsplDb,
    unit: "dB",
    notes: "Theoretical isotropic line-of-sight propagation attenuation. Does not include multipath, shadowing, or atmospheric absorption.",
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Link Budget (Received Power P_r)
// ─────────────────────────────────────────────────────────────────────────────

export interface LinkBudgetInputs {
  ptDbm: number; // Transmit power (dBm)
  gtDbi: number; // Tx antenna gain (dBi)
  grDbi: number; // Rx antenna gain (dBi)
  lPathDb: number; // Path loss (dB)
  lMiscDb: number; // Cable/connector/miscellaneous loss (dB)
}

export function calculateLinkBudget(inputs: LinkBudgetInputs): CalculationStep {
  const { ptDbm, gtDbi, grDbi, lPathDb, lMiscDb } = inputs;

  if (
    !Number.isFinite(ptDbm) ||
    !Number.isFinite(gtDbi) ||
    !Number.isFinite(grDbi) ||
    !Number.isFinite(lPathDb) ||
    !Number.isFinite(lMiscDb)
  ) {
    return {
      name: "Link Budget Received Power",
      formula: "P_r = P_t + G_t + G_r - L_{path} - L_{misc}",
      substitution: "All parameters must be finite numbers",
      result: "Invalid",
      numericResult: NaN,
      unit: "dBm",
    };
  }

  const eirpDb = ptDbm + gtDbi;
  const prDbm = ptDbm + gtDbi + grDbi - lPathDb - lMiscDb;

  return {
    name: "Link Budget Received Power (P_r)",
    formula: "P_r(\\text{dBm}) = P_t(\\text{dBm}) + G_t(\\text{dBi}) + G_r(\\text{dBi}) - L_{\\text{path}}(\\text{dB}) - L_{\\text{misc}}(\\text{dB})",
    substitution: `P_r = ${ptDbm.toFixed(2)} + ${gtDbi.toFixed(2)} + ${grDbi.toFixed(2)} - ${lPathDb.toFixed(2)} - ${lMiscDb.toFixed(2)} = ${prDbm.toFixed(2)}\\text{ dBm}`,
    result: `${prDbm.toFixed(2)} dBm`,
    numericResult: prDbm,
    unit: "dBm",
    notes: `EIRP (Equivalent Isotropically Radiated Power) = ${eirpDb.toFixed(2)} dBm (${(Math.pow(10, (eirpDb - 30) / 10)).toFixed(3)} W). Theoretical link budget calculation.`,
  };
}
