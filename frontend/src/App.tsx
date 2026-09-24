/// <reference types="vite/client" />
/**
 * Wire Watcher — Electronics & Communications Engineering (ECE)
 * RF Spectrum Monitoring, Analysis & Decision System
 *
 * 7-Module Clean Architecture:
 *  1. Dashboard Summary (High-level telemetry card grid & system state)
 *  2. RF Monitor (Spectrum Observation Ingestion, Time-Domain I/Q Waveform, Spectrum/PSD Visualizer)
 *  3. Availability & Channel Allocation (Availability Decision Engine, Allocation Heatmap, Candidate Table)
 *  4. RF Events & Utilization (Recorded Event History & Temporal Channel Utilization Timeline)
 *  5. RF Engineering (Calculator Suite with step-by-step mathematical trace modals)
 *  6. Data & Replay (RF Dataset Explorer & Controlled Historical Replay Mode)
 *  7. ML & Decision Engine (Random Forest Model Architecture, OOD Guard, Data Provenance Matrix)
 */

import React, { useCallback, useEffect, useState } from "react";
import { PredictionForm } from "@/components/wire/PredictionForm";
import { PredictionResultCard } from "@/components/wire/PredictionResultCard";
import { PredictionsTable } from "@/components/wire/PredictionsTable";
import { ChannelAllocationView } from "@/components/wire/ChannelAllocationView";
import { RFCalculatorView } from "@/components/calculator/RFCalculatorView";
import { SpectrumVisualizerView } from "@/components/spectrum/SpectrumVisualizerView";
import { RFDatasetExplorerView } from "@/components/dataset/RFDatasetExplorerView";
import { MLArchitectureView } from "@/components/ml/MLArchitectureView";

import { DashboardSummaryView } from "@/components/wire/DashboardSummaryView";
import { TimeDomainWaveformView } from "@/components/wire/TimeDomainWaveformView";
import { RFEventAnalyticsView } from "@/components/wire/RFEventAnalyticsView";
import { RFTechnicalReportView } from "@/components/wire/RFTechnicalReportView";
import { ReplayControllerView } from "@/components/wire/ReplayControllerView";
import { SourceComparisonTable } from "@/components/wire/SourceComparisonTable";

import type {
  PredictRequest,
  PredictResponse,
  PredictionRecord,
} from "@/types/wire-watcher";
import {
  LayoutDashboard,
  Radio,
  Layers,
  BarChart2,
  Calculator,
  Database,
  Cpu,
  RefreshCw,
  FileText,
  Film,
  ShieldCheck,
  Activity,
} from "lucide-react";
import { apiFetch } from "@/lib/api";

export default function App() {
  const [activeTab, setActiveTab] = useState<
    | "dashboard"
    | "monitor"
    | "allocation"
    | "events"
    | "calculator"
    | "data"
    | "ml"
    | "report"
  >("dashboard");

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
      const data = await apiFetch<any>("/api/predictions");
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
        const data = await apiFetch<PredictResponse>("/api/predict", {
          method: "POST",
          body: JSON.stringify(input),
        });
        setResult(data);
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
      setActiveTab("monitor");
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
                  RF System v2.0
                </span>
              </div>
              <p className="text-[11px] text-slate-500 font-medium">
                RF Spectrum Monitoring, Time-Domain Waveform & Channel Allocation System
              </p>
            </div>
          </div>

          {/* Telemetry Badges */}
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
              { id: "dashboard", label: "Dashboard Summary", icon: LayoutDashboard },
              { id: "monitor", label: "RF Monitor & Waveform", icon: Radio },
              { id: "allocation", label: "Availability & Channel Allocation", icon: Layers },
              { id: "events", label: "RF Events & Utilization", icon: BarChart2 },
              { id: "calculator", label: "RF Engineering Calculator", icon: Calculator },
              { id: "data", label: "Data Explorer & Replay", icon: Database },
              { id: "ml", label: "ML & Decision Engine", icon: Cpu },
              { id: "report", label: "Technical Analysis Report", icon: FileText },
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
        {/* MODULE 1: DASHBOARD SUMMARY */}
        {activeTab === "dashboard" && (
          <DashboardSummaryView onNavigateTab={(tab) => setActiveTab(tab as any)} />
        )}

        {/* MODULE 2: RF MONITOR & WAVEFORM */}
        {activeTab === "monitor" && (
          <div className="space-y-8">
            <section>
              <div className="mb-4">
                <h2 className="text-base font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                  <Radio className="size-4 text-sky-600" />
                  RF Spectrum Observation Ingestion & Availability Sensing
                </h2>
              </div>
              <PredictionForm onSubmit={handleSubmit} pending={pending} />
            </section>

            {error && (
              <div className="rounded-lg border border-red-300 bg-red-50 dark:bg-red-950/40 p-4 text-xs text-red-700 dark:text-red-300">
                <strong>Error:</strong> {error}
              </div>
            )}

            {/* TIME-DOMAIN I/Q WAVEFORM VISUALIZER */}
            <TimeDomainWaveformView
              centerFreqMhz={request?.frequency_mhz ?? 120.0}
              bandwidthMhz={(request?.bandwidth_khz ?? 200) / 1000.0}
              signalPowerDbm={request?.signal_strength_dbm ?? -75.0}
              noiseFloorDbm={-100.0}
              iqAvailable={request?.iq_available ?? 1}
            />

            {/* SPECTRUM & DSP SIMULATOR */}
            <SpectrumVisualizerView onSendToAnalyzer={handleTransferFromSimulator} />

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

            <section className="space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-base font-bold text-slate-900 dark:text-slate-100">
                    Spectrum Availability Assessment History
                  </h2>
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

        {/* MODULE 3: AVAILABILITY & CHANNEL ALLOCATION */}
        {activeTab === "allocation" && <ChannelAllocationView />}

        {/* MODULE 4: RF EVENTS & UTILIZATION */}
        {activeTab === "events" && <RFEventAnalyticsView />}

        {/* MODULE 5: RF ENGINEERING CALCULATOR */}
        {activeTab === "calculator" && <RFCalculatorView />}

        {/* MODULE 6: DATA EXPLORER & REPLAY */}
        {activeTab === "data" && (
          <div className="space-y-8">
            <ReplayControllerView />
            <SourceComparisonTable />
            <RFDatasetExplorerView />
          </div>
        )}

        {/* MODULE 7: ML & DECISION ENGINE */}
        {activeTab === "ml" && <MLArchitectureView />}

        {/* REPORT TAB */}
        {activeTab === "report" && <RFTechnicalReportView />}
      </main>

      {/* FOOTER */}
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
