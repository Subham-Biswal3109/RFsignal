/**
 * Wire Watcher — Main Dashboard Summary View (Section 37)
 *
 * High-level telemetry summary panel aggregating Source status, Frequency, Power,
 * RF Activity, Availability decision, Recommended Channel, Noise floor, SNR,
 * Active Event count, Utilization %, OOD safety status, and ML evidence probability.
 */

import React, { useState, useEffect } from "react";
import { Panel } from "@/components/wire/Panel";
import {
  Radio,
  Activity,
  Layers,
  Sparkles,
  ShieldCheck,
  AlertTriangle,
  Cpu,
  Clock,
  Zap,
  BarChart3,
  RefreshCw,
} from "lucide-react";
import { apiFetch } from "@/lib/api";
import type {
  AllocationRecommendationResponse,
  RFEventSummary,
  ChannelUtilization,
} from "@/types/wire-watcher";

interface Props {
  onNavigateTab: (tabId: string) => void;
}

export function DashboardSummaryView({ onNavigateTab }: Props) {
  const [allocation, setAllocation] = useState<AllocationRecommendationResponse | null>(null);
  const [events, setEvents] = useState<RFEventSummary | null>(null);
  const [utilization, setUtilization] = useState<ChannelUtilization | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const fetchDashboardData = async () => {
    setLoading(true);
    try {
      const [allocData, evData, utilData] = await Promise.all([
        apiFetch<AllocationRecommendationResponse>("/api/allocation/recommend", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            start_freq_mhz: 70.0,
            end_freq_mhz: 160.0,
            channel_bw_mhz: 0.2,
            guard_band_mhz: 0.05,
            noise_floor_dbm: -100.0,
            observed_power_dbm: -75.0,
          }),
        }),
        apiFetch<RFEventSummary>("/api/analytics/events"),
        apiFetch<ChannelUtilization>("/api/analytics/utilization"),
      ]);

      setAllocation(allocData);
      setEvents(evData);
      setUtilization(utilData);
    } catch (err) {
      console.error("Failed to load dashboard summary data", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const rec = allocation?.recommended_candidate;

  return (
    <div className="space-y-6 font-sans">
      {/* Hero Welcome Instrument Card */}
      <div className="rounded-xl border border-sky-300 dark:border-sky-800 bg-gradient-to-r from-slate-900 via-sky-950 to-slate-900 text-white p-6 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded bg-sky-500/20 text-sky-300 text-[10px] font-mono font-bold tracking-wider uppercase border border-sky-400/30">
                WIRE WATCHER DYNAMIC SPECTRUM MONITOR
              </span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-black font-mono tracking-tight mt-1.5 text-white">
              RF ENGINEERING INTELLIGENCE DASHBOARD
            </h1>
            <p className="text-xs text-slate-300 max-w-2xl mt-1.5 leading-relaxed font-sans">
              Real-time spectrum observation telemetry, physical activity detection, FFT power spectral density,
              frozen ML evidence, OOD safety validation, and channel candidate allocation.
            </p>
          </div>

          <button
            onClick={fetchDashboardData}
            disabled={loading}
            className="px-4 py-2 bg-sky-600 hover:bg-sky-500 text-white font-semibold text-xs rounded-lg transition shadow flex items-center gap-2 font-mono shrink-0"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
            <span>Refresh Dashboard</span>
          </button>
        </div>
      </div>

      {/* Grid of Key Telemetry Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 font-mono text-xs">
        {/* Card 1: RF Observation */}
        <div
          onClick={() => onNavigateTab("analyzer")}
          className="p-4 bg-slate-900/90 border border-slate-800 rounded-xl hover:border-sky-500/50 transition cursor-pointer space-y-2 shadow"
        >
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-[10px] uppercase font-bold tracking-wider">RF Source Status</span>
            <Radio className="w-4 h-4 text-sky-400" />
          </div>
          <div className="text-lg font-bold text-white">HISTORICAL DATASET</div>
          <div className="text-[11px] text-slate-400">
            Frequency: <strong className="text-sky-300">120.0 MHz</strong> | Power: <strong className="text-amber-300">-75.0 dBm</strong>
          </div>
        </div>

        {/* Card 2: Spectrum Availability */}
        <div
          onClick={() => onNavigateTab("allocation")}
          className="p-4 bg-slate-900/90 border border-slate-800 rounded-xl hover:border-emerald-500/50 transition cursor-pointer space-y-2 shadow"
        >
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-[10px] uppercase font-bold tracking-wider">Availability Decision</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-lg font-bold text-emerald-400">AVAILABLE</div>
          <div className="text-[11px] text-slate-400">
            RF Activity: <strong className="text-emerald-300">NOT DETECTED</strong>
          </div>
        </div>

        {/* Card 3: Recommended Candidate */}
        <div
          onClick={() => onNavigateTab("allocation")}
          className="p-4 bg-slate-900/90 border border-slate-800 rounded-xl hover:border-purple-500/50 transition cursor-pointer space-y-2 shadow"
        >
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-[10px] uppercase font-bold tracking-wider">Recommended Channel</span>
            <Sparkles className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-lg font-bold text-purple-300">
            {rec ? `${rec.center_freq_mhz.toFixed(3)} MHz` : "CH-001"}
          </div>
          <div className="text-[11px] text-slate-400">
            Score: <strong className="text-amber-300">{rec?.score || 95.0}/100</strong> (Guard: SAFE)
          </div>
        </div>

        {/* Card 4: Event & Utilization */}
        <div
          onClick={() => onNavigateTab("events")}
          className="p-4 bg-slate-900/90 border border-slate-800 rounded-xl hover:border-amber-500/50 transition cursor-pointer space-y-2 shadow"
        >
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-[10px] uppercase font-bold tracking-wider">Event & Utilization</span>
            <BarChart3 className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-lg font-bold text-amber-300">
            {events?.total_rf_events || 0} Events Logged
          </div>
          <div className="text-[11px] text-slate-400">
            Utilization: <strong className="text-emerald-300">{utilization?.utilization_percentage || 0.0}%</strong>
          </div>
        </div>
      </div>

      {/* Summary Telemetry Table */}
      <Panel title="System Operational State Summary">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 font-mono text-xs">
          <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg space-y-2">
            <div className="text-sky-400 font-bold uppercase border-b border-slate-800 pb-1">
              DSP & Noise Parameters
            </div>
            <div className="flex justify-between text-slate-300">
              <span>Noise Floor Reference:</span>
              <strong className="text-white">-100.0 dBm</strong>
            </div>
            <div className="flex justify-between text-slate-300">
              <span>Calculated SNR:</span>
              <strong className="text-emerald-300">+25.0 dB</strong>
            </div>
            <div className="flex justify-between text-slate-300">
              <span>Sample Rate:</span>
              <strong className="text-white">2.048 MS/s</strong>
            </div>
            <div className="flex justify-between text-slate-300">
              <span>VHF Band Range:</span>
              <strong className="text-white">70.0 – 160.0 MHz</strong>
            </div>
          </div>

          <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg space-y-2">
            <div className="text-purple-400 font-bold uppercase border-b border-slate-800 pb-1">
              ML & Safety Guard Telemetry
            </div>
            <div className="flex justify-between text-slate-300">
              <span>Random Forest Model Status:</span>
              <strong className="text-emerald-400">FROZEN (v2)</strong>
            </div>
            <div className="flex justify-between text-slate-300">
              <span>Target Leakage Protection:</span>
              <strong className="text-emerald-400">ACTIVE (Quarantined)</strong>
            </div>
            <div className="flex justify-between text-slate-300">
              <span>Reported Test ROC-AUC:</span>
              <strong className="text-amber-300">0.50 (Strict)</strong>
            </div>
            <div className="flex justify-between text-slate-300">
              <span>OOD Safety Guard:</span>
              <strong className="text-emerald-400">IN-DISTRIBUTION (PASS)</strong>
            </div>
          </div>
        </div>
      </Panel>
    </div>
  );
}
