/**
 * Wire Watcher — Engineering RF Observation & Prediction Result Card
 *
 * Displays:
 *  - Primary Operational Decision: AVAILABLE / OCCUPIED / UNCERTAIN
 *  - Real Measured RF Parameters [REAL_MEASURED]
 *  - Derived Electrical & Wave Properties [DERIVED_ELECTRICAL] (Wavelength, Power in nW, Thermal Noise Floor)
 *  - Activity Detection & ML Evidence [INFERRED_ACTIVITY, ML_EVIDENCE]
 *  - Out-of-Distribution (OOD) Guard Status [OOD_GUARD]
 */

import React from "react";
import {
  Database,
  Info,
  ShieldAlert,
  ShieldCheck,
  Shield,
  Zap,
  Radio,
  Activity,
  Layers,
  Thermometer,
  Cpu,
  AlertTriangle,
} from "lucide-react";
import { Panel } from "@/components/wire/Panel";
import type { PredictRequest, PredictResponse } from "@/types/wire-watcher";
import { formatProbability, formatTimestamp } from "@/utils/format";
import {
  frequencyToWavelength,
  dbmToWatts,
  calculateThermalNoise,
} from "@/utils/rfCalculations";
import { cn } from "@/lib/utils";

export function PredictionResultCard({
  result,
  request,
  receivedAt,
}: {
  result: PredictResponse;
  request: PredictRequest;
  receivedAt: string;
}) {
  const isAvailable = result.availability === "AVAILABLE";
  const isOccupied = result.availability === "OCCUPIED";
  const isUncertain = result.availability === "UNCERTAIN" || result.ood_warning;
  const ood = Boolean(result.ood_warning);
  const confidence = result.confidence;

  // Derived Electrical Quantities
  const wavelengthStep = frequencyToWavelength(request.frequency_mhz);
  const powerStep = dbmToWatts(request.signal_strength_dbm);
  const thermalNoiseStep = calculateThermalNoise(request.bandwidth_khz * 1000, 290.0);
  const thermalMarginDb = request.signal_strength_dbm - thermalNoiseStep.noisePowerDbm.numericResult;

  const getStatusColor = () => {
    if (isUncertain) return "border-amber-400 bg-amber-50/70 dark:bg-amber-950/20 text-amber-900 dark:text-amber-200";
    if (isAvailable) return "border-emerald-500 bg-emerald-50/70 dark:bg-emerald-950/20 text-emerald-900 dark:text-emerald-200";
    return "border-red-500 bg-red-50/70 dark:bg-red-950/20 text-red-900 dark:text-red-200";
  };

  const getStatusTextColor = () => {
    if (isUncertain) return "text-amber-600 dark:text-amber-400";
    if (isAvailable) return "text-emerald-600 dark:text-emerald-400";
    return "text-red-600 dark:text-red-400";
  };

  return (
    <div className="space-y-4">
      {/* Primary Engineering Decision Banner */}
      <section className={cn("panel rounded-xl border-2 p-6 shadow-sm", getStatusColor())}>
        <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-slate-800 text-white">
                OPERATIONAL DECISION
              </span>
              <span className="text-xs text-slate-500 font-mono">
                {formatTimestamp(receivedAt)}
              </span>
            </div>
            <h2 className={cn("text-4xl sm:text-5xl font-black font-mono tracking-tight mt-2", getStatusTextColor())}>
              {result.availability || (isAvailable ? "AVAILABLE" : isOccupied ? "OCCUPIED" : "UNCERTAIN")}
            </h2>
            <p className="text-xs mt-1 text-slate-600 dark:text-slate-400 max-w-xl">
              {isUncertain
                ? "Observation is flagged as UNCERTAIN due to out-of-distribution parameters or ambiguous RF features."
                : isAvailable
                ? "Channel RF activity is NOT DETECTED below the empirical median threshold."
                : "Channel RF activity is DETECTED at or above the empirical median threshold."}
            </p>
          </div>

          <div className="text-right flex flex-col sm:items-end gap-1.5 shrink-0">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-300 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 px-3 py-1 text-xs font-semibold text-slate-700 dark:text-slate-300 shadow-sm">
              <Radio className="size-3.5 text-sky-600" />
              {request.frequency_mhz} MHz Channel
            </span>
            <span className="text-[10px] font-mono text-slate-500">
              Source: {result.data_source || "RF Signal Data"}
            </span>
          </div>
        </div>

        {/* Electrical & RF Summary Grid — Provenance Sections */}
        <div className="mt-6 pt-5 border-t border-slate-200/80 dark:border-slate-800/80 grid gap-4 sm:grid-cols-2 lg:grid-cols-4 font-mono text-xs">
          {/* RF OBSERVATION Panel */}
          <div className="p-3.5 rounded-lg bg-white/80 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 space-y-1.5">
            <div className="flex items-center justify-between text-[10px] text-slate-500 uppercase font-bold">
              <span>RF OBSERVATION</span>
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-sky-100 text-sky-800 font-bold dark:bg-sky-950 dark:text-sky-300">REAL_MEASURED</span>
            </div>
            <div className="text-base font-bold text-slate-900 dark:text-slate-100">
              {request.frequency_mhz} MHz @ {(request.bandwidth_khz).toFixed(0)} kHz
            </div>
            <div className="text-[11px] text-sky-700 dark:text-sky-300 font-semibold">
              Power: {request.signal_strength_dbm.toFixed(1)} dBm
            </div>
            <div className="text-[10px] text-slate-500">
              I/Q Status: {request.iq_available === 1 ? "I/Q Available (100 samples)" : "I/Q Unavailable"}
            </div>
          </div>

          {/* DERIVED PHYSICS Panel */}
          <div className="p-3.5 rounded-lg bg-white/80 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 space-y-1.5">
            <div className="flex items-center justify-between text-[10px] text-slate-500 uppercase font-bold">
              <span>DERIVED PHYSICS</span>
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-indigo-100 text-indigo-800 font-bold dark:bg-indigo-950 dark:text-indigo-300">DERIVED</span>
            </div>
            <div className="text-base font-bold text-indigo-600 dark:text-indigo-400">
              λ = {wavelengthStep.result}
            </div>
            <div className="text-[11px] text-emerald-600 dark:text-emerald-400 font-semibold">
              Power: {powerStep.result}
            </div>
            <div className="text-[10px] text-slate-500">
              kTB Floor: {thermalNoiseStep.noisePowerDbm.result} (290 K)
            </div>
          </div>

          {/* RF ANALYSIS Panel */}
          <div className="p-3.5 rounded-lg bg-white/80 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 space-y-1.5">
            <div className="flex items-center justify-between text-[10px] text-slate-500 uppercase font-bold">
              <span>RF ANALYSIS</span>
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-purple-100 text-purple-800 font-bold dark:bg-purple-950 dark:text-purple-300">INFERRED/PSEUDO-LABEL</span>
            </div>
            <div className="text-base font-bold text-slate-900 dark:text-slate-100">
              Activity: {result.activity === "DETECTED" ? "DETECTED" : "NOT DETECTED"}
            </div>
            <div className="text-[11px] text-slate-700 dark:text-slate-300 font-semibold">
              SNR: +{thermalMarginDb.toFixed(1)} dB over kTB
            </div>
            <div className="text-[10px] text-slate-500">
              OOD Guard: {ood ? "TRUE (Safety Override)" : "FALSE (In-Distribution)"}
            </div>
          </div>

          {/* ML EVIDENCE Panel */}
          <div className="p-3.5 rounded-lg bg-white/80 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 space-y-1.5">
            <div className="flex items-center justify-between text-[10px] text-slate-500 uppercase font-bold">
              <span>ML EVIDENCE</span>
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 font-bold dark:bg-amber-950 dark:text-amber-300">ML_EVIDENCE</span>
            </div>
            <div className="text-base font-bold text-slate-900 dark:text-slate-100">
              P(Activity) = {formatProbability(result.ml_activity_probability ?? result.probability)}
            </div>
            <div className="text-[11px] text-slate-600 dark:text-slate-400">
              Confidence: {confidence}
            </div>
            <div className="text-[10px] text-slate-500">
              Quarantine: Signal power excluded from ML
            </div>
          </div>
        </div>
      </section>

      {/* OOD Alert Box */}
      {ood && (
        <div className="rounded-xl border border-amber-300 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/30 p-4 text-xs">
          <div className="flex items-center gap-2 font-bold text-amber-900 dark:text-amber-200">
            <AlertTriangle className="size-4 text-amber-600 dark:text-amber-400" />
            Out-of-Distribution (OOD) Safety Flag Triggered
          </div>
          <p className="mt-1 text-amber-800 dark:text-amber-300">
            {result.warning || "Input observations lie outside the model's 1st/99th percentile training distribution or exceeded multivariate Euclidean distance bounds."}
          </p>
          <div className="mt-2 text-[11px] text-amber-700 dark:text-amber-400 font-semibold">
            Operational action: Result overridden to <strong>UNCERTAIN</strong>.
          </div>
        </div>
      )}

      {/* Engineering Diagnostic Details Grid */}
      <Panel
        title="Multi-Stage Diagnostic Breakdown"
        subtitle="Individual evaluation outputs from the RF Activity Detector, I/Q Statistical DSP Extractor, and Machine Learning Model."
      >
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 font-mono text-xs">
          <div className="p-3 rounded bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
            <div className="text-[10px] text-slate-500 uppercase flex justify-between">
              <span>RF Activity Detector</span>
              <span className="text-[9px] text-sky-600 font-bold">REAL_MEASURED</span>
            </div>
            <div className="font-bold text-sm mt-1">
              {result.activity === "DETECTED" ? "DETECTED (≥ -50 dBm)" : result.activity === "NOT_DETECTED" ? "NOT DETECTED (< -50 dBm)" : "UNCERTAIN"}
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">Empirical training median comparison</div>
          </div>

          <div className="p-3 rounded bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
            <div className="text-[10px] text-slate-500 uppercase flex justify-between">
              <span>I/Q DSP Ingestion</span>
              <span className="text-[9px] text-indigo-600 font-bold">DERIVED</span>
            </div>
            <div className="font-bold text-sm mt-1">
              {request.iq_available === 1 ? "AVAILABLE (100 Samples)" : "UNAVAILABLE (Median Imputed)"}
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">Statistical envelope & entropy features</div>
          </div>

          <div className="p-3 rounded bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
            <div className="text-[10px] text-slate-500 uppercase flex justify-between">
              <span>Confidence Rating</span>
              <span className="text-[9px] text-amber-600 font-bold">ML_EVIDENCE</span>
            </div>
            <div className="font-bold text-sm mt-1">
              {confidence}
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">Calibrated distance from 0.5 decision boundary</div>
          </div>

          <div className="p-3 rounded bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
            <div className="text-[10px] text-slate-500 uppercase flex justify-between">
              <span>Label Provenance</span>
              <span className="text-[9px] text-teal-600 font-bold">INFERRED/PSEUDO</span>
            </div>
            <div className="font-bold text-sm mt-1 text-sky-600 dark:text-sky-400">
              INFERRED_RF_ACTIVITY
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">Not verified ground-truth occupancy</div>
          </div>
        </div>
      </Panel>

      {/* Scientific Transparency Disclaimer */}
      <div className="flex items-start gap-2 text-[11px] text-slate-500 dark:text-slate-400 px-1">
        <Info className="size-3.5 mt-0.5 shrink-0 text-slate-400" />
        <span>
          <strong>Scientific Notice:</strong> This output represents an inferred operational availability decision based on real SDR signal measurements,
          derived Johnson-Nyquist thermal calculations, and leakage-safe ML feature analysis. It does not certify regulatory spectrum vacancy or guaranteed clear spectrum.
        </span>
      </div>
    </div>
  );
}

