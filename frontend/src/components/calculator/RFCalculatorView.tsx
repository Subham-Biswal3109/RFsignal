/**
 * Wire Watcher — RF & Electronics Engineering Calculator Suite
 *
 * Provides dedicated interactive calculators for:
 *  1. Signal Power Conversion (dBm ↔ Watts ↔ dBW ↔ Unit Scaling)
 *  2. Wavelength & Antenna Dimensions (λ = c/f, λ/4 Monopole, λ/2 Dipole)
 *  3. Bandwidth, Center Frequency & Fractional Bandwidth (FBW%)
 *  4. Nyquist-Shannon Sampling Rate Check (fs ≥ 2B)
 *  5. Thermal Noise Floor & Noise Density (P_n = kTB, N_0 ≈ -174 dBm/Hz)
 *  6. Signal-to-Noise Ratio (SNR = P_signal - P_noise)
 *  7. Free-Space Path Loss (FSPL = 20log10(d) + 20log10(f) + 32.44)
 *  8. Link Budget (P_r = P_t + G_t + G_r - L_path - L_misc)
 *
 * Every calculator explicitly presents:
 *  - INPUT
 *  - FORMULA
 *  - MATHEMATICAL SUBSTITUTION
 *  - RESULT WITH ENGINEERING UNITS
 *  - PHYSICAL NOTES & THEORETICAL ASSUMPTIONS
 */

import React, { useState } from "react";
import {
  dbmToWatts,
  wattsToDbm,
  dbmToDbw,
  frequencyToWavelength,
  calculateAntennaDimensions,
  calculateBandwidthMetrics,
  checkNyquistSampling,
  calculateThermalNoise,
  calculateSNR,
  calculateFSPL,
  calculateLinkBudget,
  CalculationStep,
} from "@/utils/rfCalculations";
import { Panel } from "@/components/wire/Panel";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import {
  Calculator,
  Radio,
  Zap,
  Activity,
  Gauge,
  Layers,
  Thermometer,
  Compass,
  ArrowRight,
  Info,
  CheckCircle2,
  AlertTriangle,
} from "lucide-react";

import { CalculationTraceModal, TraceData } from "./CalculationTraceModal";

interface StepDisplayProps {
  step: CalculationStep;
  badgeText?: string;
  badgeVariant?: "blue" | "green" | "amber";
  onOpenTrace?: (trace: TraceData) => void;
}

function StepDisplay({ step, badgeText, badgeVariant = "blue", onOpenTrace }: StepDisplayProps) {
  const badgeColor =
    badgeVariant === "green"
      ? "bg-emerald-100 text-emerald-800 border-emerald-300 dark:bg-emerald-950/40 dark:text-emerald-300"
      : badgeVariant === "amber"
      ? "bg-amber-100 text-amber-800 border-amber-300 dark:bg-amber-950/40 dark:text-amber-300"
      : "bg-sky-100 text-sky-800 border-sky-300 dark:bg-sky-950/40 dark:text-sky-300";

  const handleTraceClick = () => {
    if (onOpenTrace) {
      onOpenTrace({
        title: step.name,
        inputs: { Formula: step.formula },
        formula: step.formula,
        substitution: step.substitution,
        result: step.result,
        unit: step.unit || "SI Unit",
        interpretation: step.notes || "Mathematical calculation step.",
      });
    }
  };

  return (
    <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-900/60 p-4 space-y-3 font-mono text-xs">
      <div className="flex items-center justify-between font-sans">
        <span className="font-semibold text-sm text-slate-800 dark:text-slate-200">{step.name}</span>
        <div className="flex items-center gap-2">
          {badgeText && (
            <span className={`px-2 py-0.5 rounded text-[11px] font-semibold border ${badgeColor}`}>
              {badgeText}
            </span>
          )}
          {onOpenTrace && (
            <button
              onClick={handleTraceClick}
              className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-950 text-cyan-300 border border-cyan-800 hover:bg-cyan-900 transition"
            >
              Trace
            </button>
          )}
        </div>
      </div>

      <div className="space-y-1">
        <div className="text-slate-500 dark:text-slate-400 uppercase tracking-wider text-[10px]">Formula:</div>
        <div className="bg-white dark:bg-slate-950 p-2 rounded border border-slate-200 dark:border-slate-800 text-indigo-600 dark:text-indigo-400 font-bold overflow-x-auto">
          {step.formula}
        </div>
      </div>

      <div className="space-y-1">
        <div className="text-slate-500 dark:text-slate-400 uppercase tracking-wider text-[10px]">Substitution:</div>
        <div className="bg-white dark:bg-slate-950 p-2 rounded border border-slate-200 dark:border-slate-800 text-slate-700 dark:text-slate-300 overflow-x-auto">
          {step.substitution}
        </div>
      </div>

      <div className="flex items-center justify-between pt-1 border-t border-slate-200 dark:border-slate-800">
        <span className="text-slate-500 uppercase tracking-wider text-[10px]">Calculated Output:</span>
        <span className="text-base font-bold text-emerald-600 dark:text-emerald-400 font-mono">
          {step.result}
        </span>
      </div>

      {step.notes && (
        <p className="font-sans text-[11px] text-slate-500 dark:text-slate-400 italic pt-1">
          ℹ {step.notes}
        </p>
      )}
    </div>
  );
}

export function RFCalculatorView() {
  const [activeTab, setActiveTab] = useState<
    "power" | "wavelength" | "bandwidth" | "nyquist" | "thermal" | "snr" | "fspl" | "linkbudget"
  >("power");

  const [activeTrace, setActiveTrace] = useState<TraceData | null>(null);

  // 1. Power State
  const [inputDbm, setInputDbm] = useState<number>(-60);
  const [inputWatts, setInputWatts] = useState<number>(0.000000001); // 1 nW

  // 2. Frequency / Wavelength State
  const [freqMhz, setFreqMhz] = useState<number>(120);

  // 3. Bandwidth Metrics State
  const [fLowMhz, setFLowMhz] = useState<number>(100);
  const [fHighMhz, setFHighMhz] = useState<number>(120);

  // 4. Nyquist Check State
  const [nyquistBwMhz, setNyquistBwMhz] = useState<number>(1.0);
  const [sampleRateMsps, setSampleRateMsps] = useState<number>(2.5);

  // 5. Thermal Noise State
  const [tempKelvin, setTempKelvin] = useState<number>(290.0);
  const [thermalBwHz, setThermalBwHz] = useState<number>(50000); // 50 kHz

  // 6. SNR State
  const [snrSignalDbm, setSnrSignalDbm] = useState<number>(-60);
  const [snrNoiseDbm, setSnrNoiseDbm] = useState<number>(-90);

  // 7. FSPL State
  const [fsplDistKm, setFsplDistKm] = useState<number>(1.0);
  const [fsplFreqMhz, setFsplFreqMhz] = useState<number>(120.0);

  // 8. Link Budget State
  const [ptDbm, setPtDbm] = useState<number>(30.0); // 1 W
  const [gtDbi, setGtDbi] = useState<number>(2.15); // dipole
  const [grDbi, setGrDbi] = useState<number>(2.15);
  const [lPathDb, setLPathDb] = useState<number>(74.02);
  const [lMiscDb, setLMiscDb] = useState<number>(2.0);

  // Computed Values
  const powerWattsStep = dbmToWatts(inputDbm);
  const powerDbmStep = wattsToDbm(inputWatts);
  const powerDbwStep = dbmToDbw(inputDbm);

  const antennaSteps = calculateAntennaDimensions(freqMhz);
  const bwSteps = calculateBandwidthMetrics(fLowMhz, fHighMhz);
  const nyquistStep = checkNyquistSampling(nyquistBwMhz, sampleRateMsps);
  const thermalSteps = calculateThermalNoise(thermalBwHz, tempKelvin);
  const snrStep = calculateSNR(snrSignalDbm, snrNoiseDbm);
  const fsplStep = calculateFSPL(fsplDistKm, fsplFreqMhz);
  const linkBudgetStep = calculateLinkBudget({
    ptDbm,
    gtDbi,
    grDbi,
    lPathDb,
    lMiscDb,
  });

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="rounded-xl border border-sky-200 dark:border-sky-900/50 bg-gradient-to-r from-sky-50 to-indigo-50 dark:from-sky-950/20 dark:to-indigo-950/20 p-5">
        <div className="flex items-start gap-3">
          <div className="p-2.5 rounded-lg bg-sky-600 text-white shadow-sm shrink-0">
            <Calculator className="size-6" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              RF & Communications Engineering Calculator Suite
              <span className="text-xs px-2 py-0.5 rounded bg-sky-100 dark:bg-sky-900/60 text-sky-800 dark:text-sky-300 font-mono border border-sky-200 dark:border-sky-800">
                SI Exact Constants
              </span>
            </h2>
            <p className="text-xs text-slate-600 dark:text-slate-400 mt-1 max-w-3xl">
              Provides direct mathematical calculations of electrical quantities, RF link dynamics, thermal noise floors, and antenna dimensions.
              Every tool displays the governing physical formula, explicit step-by-step substitution, and SI scaled units.
            </p>
          </div>
        </div>
      </div>

      {/* Navigation Sub-Tabs */}
      <div className="flex flex-wrap gap-1.5 p-1 bg-slate-100 dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800 text-xs">
        {[
          { id: "power", label: "Power Conversions", icon: Zap },
          { id: "wavelength", label: "Wavelength & Antenna", icon: Radio },
          { id: "bandwidth", label: "Bandwidth & Center", icon: Activity },
          { id: "nyquist", label: "Nyquist Sampling", icon: Layers },
          { id: "thermal", label: "Thermal Noise Floor", icon: Thermometer },
          { id: "snr", label: "SNR Calculator", icon: Gauge },
          { id: "fspl", label: "Free-Space Path Loss", icon: Compass },
          { id: "linkbudget", label: "Link Budget (Pr)", icon: Calculator },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-1.5 px-3 py-2 rounded-md font-medium transition-colors ${
                isActive
                  ? "bg-white dark:bg-slate-800 text-sky-700 dark:text-sky-300 shadow-sm font-semibold"
                  : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-200/50 dark:hover:bg-slate-800/50"
              }`}
            >
              <Icon className="size-3.5" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* ─────────────────────────────────────────────────────────────────── */}
      {/* 1. POWER CONVERSIONS TAB */}
      {/* ─────────────────────────────────────────────────────────────────── */}
      {activeTab === "power" && (
        <div className="space-y-6">
          <Panel
            title="1. RF Signal Power Conversion Engine"
            subtitle="Converts logarithmic power levels (dBm, dBW) into absolute linear powers (W, mW, µW, nW, pW) with full mathematical derivation."
          >
            <div className="grid gap-6 lg:grid-cols-2">
              <div className="space-y-4">
                <div className="p-4 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 space-y-3">
                  <div className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
                    Input: Logarithmic Power in dBm
                  </div>
                  <div className="grid gap-2">
                    <Label>Power Level (dBm)</Label>
                    <div className="flex gap-2">
                      <Input
                        type="number"
                        step="any"
                        value={inputDbm}
                        onChange={(e) => setInputDbm(parseFloat(e.target.value) || 0)}
                      />
                      <span className="inline-flex items-center px-3 rounded bg-slate-100 dark:bg-slate-800 text-xs font-mono">
                        dBm
                      </span>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-1.5 pt-2">
                    <span className="text-[11px] text-slate-500 py-1">Quick Presets:</span>
                    {[
                      { label: "0 dBm (1 mW)", val: 0 },
                      { label: "30 dBm (1 W)", val: 30 },
                      { label: "-30 dBm (1 µW)", val: -30 },
                      { label: "-60 dBm (1 nW)", val: -60 },
                      { label: "-90 dBm (1 pW)", val: -90 },
                      { label: "-114 dBm (1 MHz noise)", val: -114 },
                    ].map((p) => (
                      <button
                        key={p.label}
                        onClick={() => setInputDbm(p.val)}
                        className="px-2 py-1 rounded text-[11px] bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700 font-mono"
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="p-4 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 space-y-3">
                  <div className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
                    Input: Linear Power in Watts
                  </div>
                  <div className="grid gap-2">
                    <Label>Power Level (Watts)</Label>
                    <div className="flex gap-2">
                      <Input
                        type="number"
                        step="any"
                        value={inputWatts}
                        onChange={(e) => setInputWatts(parseFloat(e.target.value) || 0)}
                      />
                      <span className="inline-flex items-center px-3 rounded bg-slate-100 dark:bg-slate-800 text-xs font-mono">
                        Watts
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <StepDisplay step={powerWattsStep} badgeText="dBm → Watts" badgeVariant="green" onOpenTrace={(t) => setActiveTrace(t)} />
                <StepDisplay step={powerDbwStep} badgeText="dBm → dBW" badgeVariant="blue" onOpenTrace={(t) => setActiveTrace(t)} />
                <StepDisplay step={powerDbmStep} badgeText="Watts → dBm" badgeVariant="amber" onOpenTrace={(t) => setActiveTrace(t)} />
              </div>
            </div>
          </Panel>
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────────────── */}
      {/* 2. WAVELENGTH & ANTENNA TAB */}
      {/* ─────────────────────────────────────────────────────────────────── */}
      {activeTab === "wavelength" && (
        <div className="space-y-6">
          <Panel
            title="2. Wavelength & Resonant Antenna Elements"
            subtitle="Calculates electromagnetic wavelength λ in free-space (c = 299,792,458 m/s) and fundamental theoretical antenna element dimensions."
          >
            <div className="grid gap-6 lg:grid-cols-2">
              <div className="space-y-4">
                <div className="p-4 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 space-y-3">
                  <div className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
                    Input: Operating Carrier Frequency
                  </div>
                  <div className="grid gap-2">
                    <Label>Carrier Frequency (MHz)</Label>
                    <div className="flex gap-2">
                      <Input
                        type="number"
                        step="any"
                        value={freqMhz}
                        onChange={(e) => setFreqMhz(parseFloat(e.target.value) || 0)}
                      />
                      <span className="inline-flex items-center px-3 rounded bg-slate-100 dark:bg-slate-800 text-xs font-mono">
                        MHz
                      </span>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-1.5 pt-2">
                    <span className="text-[11px] text-slate-500 py-1">Dataset Frequencies:</span>
                    {[70, 90, 100, 120, 140, 160].map((f) => (
                      <button
                        key={f}
                        onClick={() => setFreqMhz(f)}
                        className={`px-2.5 py-1 rounded text-xs font-mono border ${
                          freqMhz === f
                            ? "bg-sky-600 text-white border-sky-600 font-bold"
                            : "bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 border-slate-300 dark:border-slate-700"
                        }`}
                      >
                        {f} MHz
                      </button>
                    ))}
                  </div>

                  <div className="mt-4 p-3 rounded bg-sky-50 dark:bg-sky-950/30 border border-sky-200 dark:border-sky-900/40 text-xs text-sky-800 dark:text-sky-300 space-y-1">
                    <p className="font-semibold flex items-center gap-1.5">
                      <Info className="size-3.5" /> Physics Reference:
                    </p>
                    <p>
                      Electromagnetic wave speed in vacuum is exactly <strong>c = 299,792,458 m/s</strong>.
                      Lower frequencies (e.g. 70 MHz) produce longer physical waves (~4.28 m), requiring larger resonant antenna apertures.
                    </p>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <StepDisplay step={antennaSteps.wavelength} badgeText="Free-Space λ" badgeVariant="green" />
                <StepDisplay step={antennaSteps.quarterWave} badgeText="Monopole (λ/4)" badgeVariant="blue" />
                <StepDisplay step={antennaSteps.halfWave} badgeText="Dipole (λ/2)" badgeVariant="amber" />
              </div>
            </div>
          </Panel>
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────────────── */}
      {/* 3. BANDWIDTH & CENTER FREQUENCY TAB */}
      {/* ─────────────────────────────────────────────────────────────────── */}
      {activeTab === "bandwidth" && (
        <div className="space-y-6">
          <Panel
            title="3. Channel Bandwidth & Fractional Bandwidth (FBW)"
            subtitle="Computes channel span, center frequency f_0, and relative percentage fractional bandwidth (FBW)."
          >
            <div className="grid gap-6 lg:grid-cols-2">
              <div className="space-y-4">
                <div className="p-4 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 space-y-4">
                  <div className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
                    Channel Boundaries
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <Label>Lower Frequency (f_low)</Label>
                      <div className="flex gap-1 mt-1">
                        <Input
                          type="number"
                          step="any"
                          value={fLowMhz}
                          onChange={(e) => setFLowMhz(parseFloat(e.target.value) || 0)}
                        />
                        <span className="inline-flex items-center px-2 rounded bg-slate-100 dark:bg-slate-800 text-xs font-mono">
                          MHz
                        </span>
                      </div>
                    </div>
                    <div>
                      <Label>Upper Frequency (f_high)</Label>
                      <div className="flex gap-1 mt-1">
                        <Input
                          type="number"
                          step="any"
                          value={fHighMhz}
                          onChange={(e) => setFHighMhz(parseFloat(e.target.value) || 0)}
                        />
                        <span className="inline-flex items-center px-2 rounded bg-slate-100 dark:bg-slate-800 text-xs font-mono">
                          MHz
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-1.5 pt-2">
                    <span className="text-[11px] text-slate-500 py-1">Standard Presets:</span>
                    {[
                      { label: "VHF FM (88 - 108 MHz)", low: 88, high: 108 },
                      { label: "Aviation (118 - 137 MHz)", low: 118, high: 137 },
                      { label: "VHF Band III (174 - 230 MHz)", low: 174, high: 230 },
                      { label: "ISM 433 MHz (433.05 - 434.79 MHz)", low: 433.05, high: 434.79 },
                    ].map((p) => (
                      <button
                        key={p.label}
                        onClick={() => {
                          setFLowMhz(p.low);
                          setFHighMhz(p.high);
                        }}
                        className="px-2 py-1 rounded text-[11px] bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-slate-700 font-mono"
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <StepDisplay step={bwSteps.bandwidthMhz} badgeText="Bandwidth" badgeVariant="green" />
                <StepDisplay step={bwSteps.centerFrequencyMhz} badgeText="Center f_0" badgeVariant="blue" />
                <StepDisplay step={bwSteps.fractionalBandwidthPercent} badgeText="Fractional BW" badgeVariant="amber" />
              </div>
            </div>
          </Panel>
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────────────── */}
      {/* 4. NYQUIST SAMPLING TAB */}
      {/* ─────────────────────────────────────────────────────────────────── */}
      {activeTab === "nyquist" && (
        <div className="space-y-6">
          <Panel
            title="4. Nyquist-Shannon Sampling Rate Verification"
            subtitle="Tests whether an analog-to-digital converter (ADC) sampling rate f_s avoids aliasing according to the Nyquist criterion (f_s ≥ 2B for real signals)."
          >
            <div className="grid gap-6 lg:grid-cols-2">
              <div className="space-y-4">
                <div className="p-4 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 space-y-4">
                  <div className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
                    Sampling Parameters
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <Label>Signal Bandwidth (B)</Label>
                      <div className="flex gap-1 mt-1">
                        <Input
                          type="number"
                          step="any"
                          value={nyquistBwMhz}
                          onChange={(e) => setNyquistBwMhz(parseFloat(e.target.value) || 0)}
                        />
                        <span className="inline-flex items-center px-2 rounded bg-slate-100 dark:bg-slate-800 text-xs font-mono">
                          MHz
                        </span>
                      </div>
                    </div>
                    <div>
                      <Label>ADC Sample Rate (f_s)</Label>
                      <div className="flex gap-1 mt-1">
                        <Input
                          type="number"
                          step="any"
                          value={sampleRateMsps}
                          onChange={(e) => setSampleRateMsps(parseFloat(e.target.value) || 0)}
                        />
                        <span className="inline-flex items-center px-2 rounded bg-slate-100 dark:bg-slate-800 text-xs font-mono">
                          MS/s
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-1.5 pt-2">
                    <span className="text-[11px] text-slate-500 py-1">Quick Scenarios:</span>
                    {[
                      { label: "B = 1 MHz, fs = 2.5 MS/s (PASS)", bw: 1.0, fs: 2.5 },
                      { label: "B = 1 MHz, fs = 1.2 MS/s (FAIL)", bw: 1.0, fs: 1.2 },
                      { label: "B = 20 MHz, fs = 40 MS/s (Critically Sampled)", bw: 20.0, fs: 40.0 },
                      { label: "B = 50 kHz (Dataset), fs = 200 kS/s", bw: 0.05, fs: 0.2 },
                    ].map((p) => (
                      <button
                        key={p.label}
                        onClick={() => {
                          setNyquistBwMhz(p.bw);
                          setSampleRateMsps(p.fs);
                        }}
                        className="px-2 py-1 rounded text-[11px] bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-slate-700 font-mono"
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <div
                  className={`p-4 rounded-xl border flex items-center justify-between ${
                    nyquistStep.passed
                      ? "bg-emerald-50 border-emerald-300 text-emerald-900 dark:bg-emerald-950/40 dark:border-emerald-800 dark:text-emerald-300"
                      : "bg-red-50 border-red-300 text-red-900 dark:bg-red-950/40 dark:border-red-800 dark:text-red-300"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    {nyquistStep.passed ? (
                      <CheckCircle2 className="size-6 text-emerald-600 dark:text-emerald-400" />
                    ) : (
                      <AlertTriangle className="size-6 text-red-600 dark:text-red-400" />
                    )}
                    <div>
                      <div className="font-bold text-sm">
                        {nyquistStep.passed ? "NYQUIST CONDITION MET" : "INSUFFICIENT SAMPLING RATE"}
                      </div>
                      <div className="text-xs opacity-90">
                        Minimum Required: {nyquistStep.minNyquistRateMsps.toFixed(3)} MS/s | Provided: {sampleRateMsps.toFixed(3)} MS/s
                      </div>
                    </div>
                  </div>
                  <span
                    className={`px-3 py-1 rounded-full text-xs font-bold tracking-wider uppercase border ${
                      nyquistStep.passed
                        ? "bg-emerald-200 text-emerald-900 border-emerald-300 dark:bg-emerald-900 dark:text-emerald-100"
                        : "bg-red-200 text-red-900 border-red-300 dark:bg-red-900 dark:text-red-100"
                    }`}
                  >
                    {nyquistStep.passed ? "PASS" : "FAIL"}
                  </span>
                </div>

                <StepDisplay step={nyquistStep} badgeText="Nyquist Theorem" badgeVariant={nyquistStep.passed ? "green" : "amber"} />
              </div>
            </div>
          </Panel>
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────────────── */}
      {/* 5. THERMAL NOISE TAB */}
      {/* ─────────────────────────────────────────────────────────────────── */}
      {activeTab === "thermal" && (
        <div className="space-y-6">
          <Panel
            title="5. Johnson-Nyquist Thermal Noise Floor"
            subtitle="Calculates the fundamental theoretical thermal noise power generated by thermal agitation of electrons across a given bandwidth (P_n = k_B · T · B)."
          >
            <div className="grid gap-6 lg:grid-cols-2">
              <div className="space-y-4">
                <div className="p-4 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 space-y-4">
                  <div className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
                    Physical Parameters
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <Label>Temperature (Kelvin)</Label>
                      <div className="flex gap-1 mt-1">
                        <Input
                          type="number"
                          step="any"
                          value={tempKelvin}
                          onChange={(e) => setTempKelvin(parseFloat(e.target.value) || 0)}
                        />
                        <span className="inline-flex items-center px-2 rounded bg-slate-100 dark:bg-slate-800 text-xs font-mono">
                          K
                        </span>
                      </div>
                      <span className="text-[10px] text-slate-500">
                        = {(tempKelvin - 273.15).toFixed(1)} °C
                      </span>
                    </div>
                    <div>
                      <Label>Channel Bandwidth (Hz)</Label>
                      <div className="flex gap-1 mt-1">
                        <Input
                          type="number"
                          step="any"
                          value={thermalBwHz}
                          onChange={(e) => setThermalBwHz(parseFloat(e.target.value) || 0)}
                        />
                        <span className="inline-flex items-center px-2 rounded bg-slate-100 dark:bg-slate-800 text-xs font-mono">
                          Hz
                        </span>
                      </div>
                      <span className="text-[10px] text-slate-500">
                        = {(thermalBwHz / 1000).toFixed(1)} kHz
                      </span>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-1.5 pt-2">
                    <span className="text-[11px] text-slate-500 py-1">Standard Bandwidths:</span>
                    {[
                      { label: "1 Hz (PSD Base)", bw: 1 },
                      { label: "50 kHz (Wire Watcher)", bw: 50000 },
                      { label: "200 kHz (GSM)", bw: 200000 },
                      { label: "1.4 MHz (LTE 1.4)", bw: 1400000 },
                      { label: "20 MHz (WiFi/LTE)", bw: 20000000 },
                    ].map((p) => (
                      <button
                        key={p.label}
                        onClick={() => setThermalBwHz(p.bw)}
                        className="px-2 py-1 rounded text-[11px] bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-slate-700 font-mono"
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>

                  <div className="p-3 rounded bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900/40 text-xs text-amber-900 dark:text-amber-300 space-y-1">
                    <p className="font-semibold flex items-center gap-1.5">
                      <Info className="size-3.5" /> Engineering Clarification:
                    </p>
                    <p>
                      <strong>-174 dBm/Hz</strong> is the thermal noise spectral density at room temperature (T = 290 K).
                      It represents the absolute theoretical sensitivity floor for any receiver; real receivers exhibit an additional Noise Figure (NF).
                    </p>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <StepDisplay step={thermalSteps.noiseDensityDbmHz} badgeText="N_0 Density" badgeVariant="blue" />
                <StepDisplay step={thermalSteps.noisePowerDbm} badgeText="Total Noise Floor (dBm)" badgeVariant="green" />
                <StepDisplay step={thermalSteps.noisePowerWatts} badgeText="Power in Watts" badgeVariant="amber" />
              </div>
            </div>
          </Panel>
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────────────── */}
      {/* 6. SNR CALCULATOR TAB */}
      {/* ─────────────────────────────────────────────────────────────────── */}
      {activeTab === "snr" && (
        <div className="space-y-6">
          <Panel
            title="6. Signal-to-Noise Ratio (SNR) Analysis"
            subtitle="Calculates the ratio of received signal power to background noise power in decibels (SNR = P_signal - P_noise)."
          >
            <div className="grid gap-6 lg:grid-cols-2">
              <div className="space-y-4">
                <div className="p-4 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 space-y-4">
                  <div className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
                    Power Levels in dBm
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <Label>Signal Power (P_signal)</Label>
                      <div className="flex gap-1 mt-1">
                        <Input
                          type="number"
                          step="any"
                          value={snrSignalDbm}
                          onChange={(e) => setSnrSignalDbm(parseFloat(e.target.value) || 0)}
                        />
                        <span className="inline-flex items-center px-2 rounded bg-slate-100 dark:bg-slate-800 text-xs font-mono">
                          dBm
                        </span>
                      </div>
                    </div>
                    <div>
                      <Label>Noise Floor (P_noise)</Label>
                      <div className="flex gap-1 mt-1">
                        <Input
                          type="number"
                          step="any"
                          value={snrNoiseDbm}
                          onChange={(e) => setSnrNoiseDbm(parseFloat(e.target.value) || 0)}
                        />
                        <span className="inline-flex items-center px-2 rounded bg-slate-100 dark:bg-slate-800 text-xs font-mono">
                          dBm
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-1.5 pt-2">
                    <span className="text-[11px] text-slate-500 py-1">Examples:</span>
                    {[
                      { label: "Signal -60 dBm, Noise -90 dBm (30 dB)", s: -60, n: -90 },
                      { label: "Signal -80 dBm, Noise -85 dBm (5 dB)", s: -80, n: -85 },
                      { label: "Signal -95 dBm, Noise -90 dBm (-5 dB submerged)", s: -95, n: -90 },
                    ].map((p) => (
                      <button
                        key={p.label}
                        onClick={() => {
                          setSnrSignalDbm(p.s);
                          setSnrNoiseDbm(p.n);
                        }}
                        className="px-2 py-1 rounded text-[11px] bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-slate-700 font-mono"
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <StepDisplay step={snrStep} badgeText="Calculated SNR" badgeVariant="green" />
              </div>
            </div>
          </Panel>
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────────────── */}
      {/* 7. FREE-SPACE PATH LOSS TAB */}
      {/* ─────────────────────────────────────────────────────────────────── */}
      {activeTab === "fspl" && (
        <div className="space-y-6">
          <Panel
            title="7. Free-Space Path Loss (FSPL) Model"
            subtitle="Calculates theoretical electromagnetic attenuation between isotropic antennas over distance d in free space."
          >
            <div className="grid gap-6 lg:grid-cols-2">
              <div className="space-y-4">
                <div className="p-4 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 space-y-4">
                  <div className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
                    Link Parameters
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <Label>Distance (km)</Label>
                      <div className="flex gap-1 mt-1">
                        <Input
                          type="number"
                          step="any"
                          value={fsplDistKm}
                          onChange={(e) => setFsplDistKm(parseFloat(e.target.value) || 0)}
                        />
                        <span className="inline-flex items-center px-2 rounded bg-slate-100 dark:bg-slate-800 text-xs font-mono">
                          km
                        </span>
                      </div>
                    </div>
                    <div>
                      <Label>Frequency (MHz)</Label>
                      <div className="flex gap-1 mt-1">
                        <Input
                          type="number"
                          step="any"
                          value={fsplFreqMhz}
                          onChange={(e) => setFsplFreqMhz(parseFloat(e.target.value) || 0)}
                        />
                        <span className="inline-flex items-center px-2 rounded bg-slate-100 dark:bg-slate-800 text-xs font-mono">
                          MHz
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-1.5 pt-2">
                    <span className="text-[11px] text-slate-500 py-1">Distances:</span>
                    {[
                      { label: "100 m (0.1 km)", d: 0.1 },
                      { label: "1 km", d: 1.0 },
                      { label: "5 km", d: 5.0 },
                      { label: "20 km (Line-of-Sight)", d: 20.0 },
                    ].map((p) => (
                      <button
                        key={p.label}
                        onClick={() => setFsplDistKm(p.d)}
                        className="px-2 py-1 rounded text-[11px] bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-slate-700 font-mono"
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>

                  <div className="p-3 rounded bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-[11px] text-slate-600 dark:text-slate-400">
                    <strong>Note:</strong> FSPL assumes ideal line-of-sight propagation without terrain diffraction, multipath fading, foliage attenuation, or rain absorption.
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <StepDisplay step={fsplStep} badgeText="Theoretical Free-Space Loss" badgeVariant="blue" />
              </div>
            </div>
          </Panel>
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────────────── */}
      {/* 8. LINK BUDGET TAB */}
      {/* ─────────────────────────────────────────────────────────────────── */}
      {activeTab === "linkbudget" && (
        <div className="space-y-6">
          <Panel
            title="8. RF Link Budget & Received Power (P_r)"
            subtitle="Calculates the total end-to-end power balance across transmitter, antennas, propagation medium, and receiver."
          >
            <div className="grid gap-6 lg:grid-cols-2">
              <div className="space-y-4">
                <div className="p-4 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 space-y-4">
                  <div className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
                    Transmitter & Antenna Gains
                  </div>
                  <div className="grid grid-cols-3 gap-2">
                    <div>
                      <Label>P_tx (dBm)</Label>
                      <Input
                        type="number"
                        step="any"
                        value={ptDbm}
                        onChange={(e) => setPtDbm(parseFloat(e.target.value) || 0)}
                      />
                    </div>
                    <div>
                      <Label>G_tx (dBi)</Label>
                      <Input
                        type="number"
                        step="any"
                        value={gtDbi}
                        onChange={(e) => setGtDbi(parseFloat(e.target.value) || 0)}
                      />
                    </div>
                    <div>
                      <Label>G_rx (dBi)</Label>
                      <Input
                        type="number"
                        step="any"
                        value={grDbi}
                        onChange={(e) => setGrDbi(parseFloat(e.target.value) || 0)}
                      />
                    </div>
                  </div>

                  <div className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300 pt-2">
                    Losses in Path
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <Label>Path Loss L_path (dB)</Label>
                      <Input
                        type="number"
                        step="any"
                        value={lPathDb}
                        onChange={(e) => setLPathDb(parseFloat(e.target.value) || 0)}
                      />
                    </div>
                    <div>
                      <Label>Misc Losses L_misc (dB)</Label>
                      <Input
                        type="number"
                        step="any"
                        value={lMiscDb}
                        onChange={(e) => setLMiscDb(parseFloat(e.target.value) || 0)}
                      />
                    </div>
                  </div>

                  <div className="p-3 rounded bg-indigo-50 dark:bg-indigo-950/30 border border-indigo-200 dark:border-indigo-900/40 text-[11px] text-indigo-900 dark:text-indigo-300">
                    <strong>Formula:</strong> P_r = P_t + G_t + G_r - L_path - L_misc
                    <br />
                    EIRP = P_t + G_t = {(ptDbm + gtDbi).toFixed(2)} dBm
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <StepDisplay step={linkBudgetStep} badgeText="Theoretical Link Budget" badgeVariant="green" onOpenTrace={(t) => setActiveTrace(t)} />
              </div>
            </div>
          </Panel>
        </div>
      )}

      <CalculationTraceModal trace={activeTrace} onClose={() => setActiveTrace(null)} />
    </div>
  );
}
