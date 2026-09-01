/// <reference types="vite/client" />
/**
 * Wire Watcher — Electronics & Communications Engineering (ECE)
 * RF Spectrum Monitoring and Availability Analysis System
 *
 * Primary Modules:
 *  1. Spectrum Availability Analyzer (Prediction Form + Engineering Observation Card + History)
 *  2. RF Engineering Calculator (Power, Wavelength, Bandwidth, Nyquist, Thermal Noise, SNR, FSPL, Link Budget)
 *  3. Spectrum & DSP Simulator (FFT, PSD Plot, Peak Detection, Adaptive Thresholding)
 *  4. RF Dataset Explorer (Real 164,160 observations, VHF frequencies, I/Q statistics)
 *  5. ML & Decision Engine (Model comparison, leakage protection, multi-stage decision pipeline)
 */

import React, { useCallback, useEffect, useState } from "react";
import { PredictionForm } from "@/components/wire/PredictionForm";
import { PredictionResultCard } from "@/components/wire/PredictionResultCard";
import { PredictionsTable } from "@/components/wire/PredictionsTable";
import { RFCalculatorView } from "@/components/calculator/RFCalculatorView";
import { SpectrumVisualizerView } from "@/components/spectrum/SpectrumVisualizerView";
import { RFDatasetExplorerView } from "@/components/dataset/RFDatasetExplorerView";
import { MLArchitectureView } from "@/components/ml/MLArchitectureView";

import type {
  PredictRequest,
  PredictResponse,
  PredictionRecord,
} from "@/types/wire-watcher";
import {
  Radio,
  Calculator,
  Activity,
  Database,
  Cpu,
  RefreshCw,
  Layers,
  Sparkles,
  Info,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:5000";

export default function App() {
  const [activeTab, setActiveTab] = useState<
    "analyzer" | "calculator" | "simulator" | "dataset" | "ml"
  >("analyzer");

  const [pending, setPending] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [request, setRequest] = useState<PredictRequest | null>(null);
  const [receivedAt, setReceivedAt] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [records, setRecords] = useState<PredictionRecord[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // ── fetch prediction history ───────────────────────────────────────────────
  const fetchHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/api/predictions`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const predictions: PredictionRecord[] = (data.predictions ?? []).map(
        (row: Record<string, unknown>) => ({
          id: String(row.id ?? ""),
          start_frequency_mhz: row.start_frequency_mhz as number | null,
          end_frequency_mhz: row.end_frequency_mhz as number | null,
          bandwidth_mhz: row.bandwidth_mhz as number | null,
          city: row.city as string | null,
          state: row.state as string | null,
          service_type: row.service_type as string | null,
          available: row.available as boolean | null,
          probability: row.probability as number | null,
          timestamp: row.timestamp as string | null,
          signal_power_dbm: row.signal_power_dbm as number | null,
          noise_floor_dbm: row.noise_floor_dbm as number | null,
          snr_db: row.snr_db as number | null,
          data_source: row.data_source as string | undefined,
          ood_status: row.ood_status as boolean | undefined,
          raw: row,
        }),
      );
      setRecords(predictions);
    } catch (err) {
      console.error("Failed to load prediction history:", err);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  // ── submit prediction ──────────────────────────────────────────────────────
  const handleSubmit = useCallback(
    async (input: PredictRequest) => {
      setPending(true);
      setError(null);
      setResult(null);
      setRequest(input);
      try {
        const resp = await fetch(`${API_BASE}/api/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(input),
        });
        const data = await resp.json();
        if (!resp.ok) {
          setError(data.error ?? `HTTP ${resp.status}`);
          return;
        }
        setResult(data as PredictResponse);
        setReceivedAt(new Date().toISOString());
        fetchHistory();
      } catch (err) {
        setError(err instanceof Error ? err.message : "Network error");
      } finally {
        setPending(false);
      }
    },
    [fetchHistory],
  );

  // ── handle transfer from simulator to analyzer ────────────────────────────
  const handleTransferFromSimulator = useCallback(
    (simRequest: PredictRequest) => {
      setActiveTab("analyzer");
      handleSubmit(simRequest);
    },
    [handleSubmit],
  );

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 flex flex-col font-sans">
      {/* ── TOP ECE INSTRUMENT HEADER ────────────────────────────────────────── */}
      <header className="border-b border-slate-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/90 backdrop-blur sticky top-0 z-50">
        <div className="mx-auto max-w-7xl px-4 py-3 flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="size-9 rounded-lg bg-sky-600 flex items-center justify-center text-white shadow-sm font-mono font-bold text-base">
              WW
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-black tracking-tight text-slate-900 dark:text-slate-100 font-mono">
                  WIRE WATCHER
                </h1>
                <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-sky-100 text-sky-800 dark:bg-sky-950/60 dark:text-sky-300 font-mono border border-sky-200 dark:border-sky-800">
                  RF Activity v2
                </span>
              </div>
              <p className="text-[11px] text-slate-500 font-medium">
                ECE RF Spectrum Monitoring, Analysis & Decision System
              </p>
            </div>
          </div>

          {/* Instrument Telemetry Badges */}
          <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
            <div className="px-2.5 py-1 rounded bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
              <span className="size-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span>BAND: 70 – 160 MHz (VHF)</span>
            </div>

            <div className="px-2.5 py-1 rounded bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300">
              DATASET: 164,160 OBS
            </div>

            <div className="px-2.5 py-1 rounded bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300">
              DECISION: INFERRED_ACTIVITY
            </div>
          </div>
        </div>

        {/* Primary Navigation Tabs */}
        <div className="border-t border-slate-200 dark:border-slate-800 px-4 bg-slate-50/70 dark:bg-slate-900/50">
          <div className="mx-auto max-w-7xl flex overflow-x-auto gap-2 py-1.5 text-xs">
            {[
              { id: "analyzer", label: "Spectrum Availability Analyzer", icon: Radio },
              { id: "calculator", label: "RF Engineering Calculator", icon: Calculator },
              { id: "simulator", label: "Spectrum & DSP Simulator", icon: Activity },
              { id: "dataset", label: "RF Dataset Explorer", icon: Database },
              { id: "ml", label: "ML & Decision Engine", icon: Cpu },
            ].map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex items-center gap-2 px-3.5 py-2 rounded-md font-medium whitespace-nowrap transition-all ${
                    isActive
                      ? "bg-white dark:bg-slate-800 text-sky-700 dark:text-sky-300 shadow-sm font-semibold border border-slate-200 dark:border-slate-700"
                      : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-200/50 dark:hover:bg-slate-800/50"
                  }`}
                >
                  <Icon className="size-4" />
                  {tab.label}
                </button>
              );
            })}
          </div>
        </div>
      </header>

      {/* ── MAIN CONTENT AREA ────────────────────────────────────────────────── */}
      <main className="mx-auto max-w-7xl w-full p-4 sm:p-6 space-y-8 flex-1">
        {/* ── TAB 1: SPECTRUM AVAILABILITY ANALYZER (HOME SCREEN) ── */}
        {activeTab === "analyzer" && (
          <div className="space-y-8">
            {/* Home Hero Banner */}
            <div className="rounded-xl border border-sky-300 dark:border-sky-800 bg-gradient-to-r from-slate-900 via-sky-950 to-slate-900 text-white p-6 shadow-md">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-0.5 rounded bg-sky-500/20 text-sky-300 text-[10px] font-mono font-bold tracking-wider uppercase border border-sky-400/30">
                      ECE / RF ENGINEERING INSTRUMENT
                    </span>
                  </div>
                  <h1 className="text-2xl sm:text-3xl font-black font-mono tracking-tight mt-1.5 text-white">
                    WIRE WATCHER
                  </h1>
                  <h2 className="text-base sm:text-lg font-bold font-mono text-sky-400 tracking-wide mt-0.5">
                    RF SPECTRUM MONITORING & AVAILABILITY ANALYSIS
                  </h2>
                  <p className="text-xs text-slate-300 max-w-2xl mt-2 leading-relaxed">
                    Digital signal processing, physical channel activity detection, Johnson-Nyquist thermal noise evaluation,
                    and out-of-distribution (OOD) safety gating for VHF spectrum availability.
                  </p>
                </div>

                {/* Live / Current Telemetry Summary Card */}
                <div className="p-3.5 rounded-lg bg-slate-950/80 border border-sky-500/30 font-mono text-xs space-y-2 shrink-0 min-w-[260px]">
                  <div className="text-[10px] uppercase font-bold text-sky-400 flex items-center justify-between border-b border-slate-800 pb-1.5">
                    <span className="flex items-center gap-1.5">
                      <span className="size-2 rounded-full bg-emerald-400 animate-pulse"></span>
                      Live Spectrum Status
                    </span>
                    <span className="text-[9px] text-slate-400">REAL-TIME</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-[11px]">
                    <div>
                      <span className="text-slate-400 block text-[9px] uppercase">Frequency</span>
                      <span className="font-bold text-white">{request ? `${request.frequency_mhz} MHz` : "120.0 MHz"}</span>
                    </div>
                    <div>
                      <span className="text-slate-400 block text-[9px] uppercase">Bandwidth</span>
                      <span className="font-bold text-white">{request ? `${request.bandwidth_khz} kHz` : "50 kHz"}</span>
                    </div>
                    <div>
                      <span className="text-slate-400 block text-[9px] uppercase">Signal Power</span>
                      <span className="font-bold text-white">{request ? `${request.signal_strength_dbm} dBm` : "-60.0 dBm"}</span>
                    </div>
                    <div>
                      <span className="text-slate-400 block text-[9px] uppercase">RF Activity</span>
                      <span className={`font-bold ${result?.activity === "DETECTED" ? "text-red-400" : "text-emerald-400"}`}>
                        {result?.activity ?? "NOT DETECTED"}
                      </span>
                    </div>
                  </div>
                  <div className="pt-1.5 border-t border-slate-800 flex items-center justify-between">
                    <span className="text-slate-400 text-[10px]">Decision:</span>
                    <span className={`font-bold px-2 py-0.5 rounded text-[10px] ${
                      result?.availability === "AVAILABLE" ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40" :
                      result?.availability === "OCCUPIED" ? "bg-red-500/20 text-red-300 border border-red-500/40" :
                      "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                    }`}>
                      {result?.availability ?? "AVAILABLE"}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Input Form Section */}
            <section>
              <div className="mb-4">
                <h2 className="text-base font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                  <Radio className="size-4 text-sky-600" />
                  RF Spectrum Observation Ingestion
                </h2>
                <p className="text-xs text-slate-500">
                  Enter real measured carrier frequency, bandwidth, and received signal strength.
                  Optional digital signal processing (DSP) I/Q envelope statistics can be provided when complex samples exist.
                </p>
              </div>
              <PredictionForm onSubmit={handleSubmit} pending={pending} />
            </section>

            {/* Error Banner */}
            {error && (
              <div className="rounded-lg border border-red-300 bg-red-50 dark:bg-red-950/40 p-4 text-xs text-red-700 dark:text-red-300">
                <strong>Error:</strong> {error}
              </div>
            )}

            {/* Prediction / Analysis Result Card */}
            {result && request && (
              <section className="space-y-2">
                <h2 className="text-base font-bold text-slate-900 dark:text-slate-100">
                  Observation Analysis & Electrical Evaluation
                </h2>
                <PredictionResultCard
                  result={result}
                  request={request}
                  receivedAt={receivedAt}
                />
              </section>
            )}

            {/* Prediction History Table */}
            <section className="space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-base font-bold text-slate-900 dark:text-slate-100">
                    Spectrum Availability Assessment History
                  </h2>
                  <p className="text-xs text-slate-500">
                    Recent records persisted in SQLite (<code className="font-mono">wire_watcher.db</code>).
                  </p>
                </div>
                <button
                  onClick={fetchHistory}
                  disabled={historyLoading}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 hover:bg-slate-50 text-slate-700 dark:text-slate-300 disabled:opacity-50"
                >
                  <RefreshCw className={`size-3.5 ${historyLoading ? "animate-spin" : ""}`} />
                  Refresh History
                </button>
              </div>

              {records.length === 0 ? (
                <div className="p-8 text-center rounded-xl border border-dashed border-slate-300 dark:border-slate-800 text-xs text-slate-500">
                  No prediction records stored yet. Ingest an RF observation above to populate history.
                </div>
              ) : (
                <PredictionsTable records={records} />
              )}
            </section>
          </div>
        )}

        {/* ── TAB 2: RF ENGINEERING CALCULATOR SUITE ── */}
        {activeTab === "calculator" && <RFCalculatorView />}

        {/* ── TAB 3: SPECTRUM & DSP SIMULATOR ── */}
        {activeTab === "simulator" && (
          <SpectrumVisualizerView onSendToAnalyzer={handleTransferFromSimulator} />
        )}

        {/* ── TAB 4: RF DATASET EXPLORER ── */}
        {activeTab === "dataset" && <RFDatasetExplorerView />}

        {/* ── TAB 5: ML & DECISION ENGINE ── */}
        {activeTab === "ml" && <MLArchitectureView />}
      </main>

      {/* ── FOOTER ───────────────────────────────────────────────────────────── */}
      <footer className="border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 py-6 text-xs text-slate-500">
        <div className="mx-auto max-w-7xl px-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4 font-mono text-[11px]">
          <div>
            <div className="font-bold text-slate-700 dark:text-slate-300 uppercase">Data Foundation</div>
            <p className="text-slate-500 font-sans mt-1">
              RF Signal Data (164,160 observations, 6 VHF frequencies, ~20s cadence).
            </p>
          </div>

          <div>
            <div className="font-bold text-slate-700 dark:text-slate-300 uppercase">Target Provenance</div>
            <p className="text-slate-500 font-sans mt-1">
              <code>inferred_rf_activity</code> is an INFERRED PSEUDO-LABEL based on training median signal strength.
            </p>
          </div>

          <div>
            <div className="font-bold text-slate-700 dark:text-slate-300 uppercase">Model Performance</div>
            <p className="text-slate-500 font-sans mt-1">
              ROC-AUC ≈ 0.50 with circular signal strength strictly quarantined from ML features.
            </p>
          </div>

          <div>
            <div className="font-bold text-slate-700 dark:text-slate-300 uppercase">Scientific Scope</div>
            <p className="text-slate-500 font-sans mt-1">
              Provides inferred operational availability decisions. Does not certify legal spectrum vacancy or live SDR hardware monitoring.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
