/**
 * Wire Watcher — RF Spectrum Channel Allocation & Engineering Scoring View
 *
 * Demonstrates:
 *   - Visual Channel Allocation Heatmap across analyzed VHF band
 *   - Engineering channel candidate generation
 *   - Transparent scoring formula with Guard Band & Interference Risk analysis
 *   - Recommended Candidate Channel callout ("RECOMMENDED CANDIDATE" or "NO SUITABLE CANDIDATE")
 *   - Sortable Candidate Channels Table with data provenance tags & Calculation Trace Modal
 */

import React, { useState, useEffect } from "react";
import { Panel } from "@/components/wire/Panel";
import {
  Layers,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Radio,
  Sliders,
  Sparkles,
  ShieldAlert,
  ArrowUpDown,
  Info,
  RefreshCw,
  Calculator,
} from "lucide-react";
import type {
  AllocationRecommendationResponse,
  ChannelCandidate,
} from "@/types/wire-watcher";
import { cn } from "@/lib/utils";
import { apiFetch } from "@/lib/api";
import { ChannelAllocationHeatmap } from "./ChannelAllocationHeatmap";
import { CalculationTraceModal } from "./CalculationTraceModal";

export function ChannelAllocationView() {
  const [startFreqMhz, setStartFreqMhz] = useState<number>(70.0);
  const [endFreqMhz, setEndFreqMhz] = useState<number>(160.0);
  const [channelBwMhz, setChannelBwMhz] = useState<number>(0.2); // 200 kHz
  const [guardBandMhz, setGuardBandMhz] = useState<number>(0.05); // 50 kHz
  const [noiseFloorDbm, setNoiseFloorDbm] = useState<number>(-100.0);
  const [observedPowerDbm, setObservedPowerDbm] = useState<number>(-75.0);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<AllocationRecommendationResponse | null>(null);
  const [sortField, setSortField] = useState<"score" | "center_freq_mhz">("score");
  const [sortAsc, setSortAsc] = useState<boolean>(false);

  const [selectedCandidate, setSelectedCandidate] = useState<ChannelCandidate | null>(null);
  const [traceCandidate, setTraceCandidate] = useState<ChannelCandidate | null>(null);

  const fetchAllocation = async () => {
    setLoading(true);
    setError(null);
    try {
      const json = await apiFetch<AllocationRecommendationResponse>("/api/allocation/recommend", {
        method: "POST",
        body: JSON.stringify({
          start_freq_mhz: startFreqMhz,
          end_freq_mhz: endFreqMhz,
          channel_bw_mhz: channelBwMhz,
          guard_band_mhz: guardBandMhz,
          noise_floor_dbm: noiseFloorDbm,
          observed_power_dbm: observedPowerDbm,
        }),
      });
      setData(json);
      if (json.recommended_candidate) {
        setSelectedCandidate(json.recommended_candidate);
      }
    } catch (err: any) {
      setError(err.message || "Failed to calculate channel allocation");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAllocation();
  }, []);

  const handleSort = (field: "score" | "center_freq_mhz") => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(field === "center_freq_mhz");
    }
  };

  const sortedCandidates = data
    ? [...data.candidates].sort((a, b) => {
        const valA = a[sortField];
        const valB = b[sortField];
        if (valA < valB) return sortAsc ? -1 : 1;
        if (valA > valB) return sortAsc ? 1 : -1;
        return 0;
      })
    : [];

  return (
    <div className="space-y-6">
      {/* Header Instrument Banner */}
      <div className="rounded-xl border border-indigo-200 dark:border-indigo-900/50 bg-gradient-to-r from-indigo-50 via-sky-50 to-teal-50 dark:from-indigo-950/20 dark:via-sky-950/20 dark:to-teal-950/20 p-5">
        <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
          <div className="flex items-start gap-3">
            <div className="p-2.5 rounded-lg bg-indigo-600 text-white shadow-sm shrink-0">
              <Layers className="size-6" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                RF Spectrum Channel Allocation & Engineering Intelligence Layer
                <span className="text-xs px-2 py-0.5 rounded bg-indigo-100 dark:bg-indigo-900/60 text-indigo-800 dark:text-indigo-300 font-mono border border-indigo-200 dark:border-indigo-800">
                  Engineering Assessment
                </span>
              </h2>
              <p className="text-xs text-slate-600 dark:text-slate-400 mt-1 max-w-3xl">
                Divides analyzed spectrum bands into candidate channels, evaluates guard-band protected margins, calculates transparent interference metrics, and provides explainable calculation traces.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Control Input Form Panel */}
      <Panel
        title="Channel Allocation Parameters"
        subtitle="Configure frequency bounds, channel width, guard band, and noise floor reference."
      >
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 font-mono text-xs">
          <div>
            <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">
              Start Frequency (MHz)
            </label>
            <input
              type="number"
              step="1"
              value={startFreqMhz}
              onChange={(e) => setStartFreqMhz(parseFloat(e.target.value) || 70)}
              className="w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-3 py-2 text-slate-900 dark:text-slate-100"
            />
          </div>

          <div>
            <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">
              End Frequency (MHz)
            </label>
            <input
              type="number"
              step="1"
              value={endFreqMhz}
              onChange={(e) => setEndFreqMhz(parseFloat(e.target.value) || 160)}
              className="w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-3 py-2 text-slate-900 dark:text-slate-100"
            />
          </div>

          <div>
            <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">
              Channel Bandwidth (MHz)
            </label>
            <input
              type="number"
              step="0.05"
              value={channelBwMhz}
              onChange={(e) => setChannelBwMhz(parseFloat(e.target.value) || 0.2)}
              className="w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-3 py-2 text-slate-900 dark:text-slate-100"
            />
          </div>

          <div>
            <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">
              Guard Band (MHz)
            </label>
            <input
              type="number"
              step="0.01"
              value={guardBandMhz}
              onChange={(e) => setGuardBandMhz(parseFloat(e.target.value) || 0.05)}
              className="w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-3 py-2 text-slate-900 dark:text-slate-100"
            />
          </div>

          <div>
            <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">
              Noise Floor Reference (dBm)
            </label>
            <input
              type="number"
              step="1"
              value={noiseFloorDbm}
              onChange={(e) => setNoiseFloorDbm(parseFloat(e.target.value) || -100)}
              className="w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-3 py-2 text-slate-900 dark:text-slate-100"
            />
          </div>

          <div>
            <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">
              Observed Signal Power (dBm)
            </label>
            <input
              type="number"
              step="1"
              value={observedPowerDbm}
              onChange={(e) => setObservedPowerDbm(parseFloat(e.target.value) || -75)}
              className="w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-3 py-2 text-slate-900 dark:text-slate-100"
            />
          </div>
        </div>

        <div className="mt-4 flex justify-end">
          <button
            onClick={fetchAllocation}
            disabled={loading}
            className="inline-flex items-center gap-2 rounded-md bg-indigo-600 px-4 py-2 text-xs font-semibold text-white shadow hover:bg-indigo-500 disabled:opacity-50"
          >
            <RefreshCw className={cn("size-3.5", loading && "animate-spin")} />
            Calculate Channel Recommendations
          </button>
        </div>
      </Panel>

      {/* Visual Channel Allocation Heatmap */}
      {data && (
        <ChannelAllocationHeatmap
          candidates={data.candidates}
          recommendedCandidate={data.recommended_candidate}
          selectedCandidate={selectedCandidate}
          onSelectCandidate={(cand) => setSelectedCandidate(cand)}
          startFreqMhz={startFreqMhz}
          endFreqMhz={endFreqMhz}
        />
      )}

      {error && (
        <div className="rounded-xl border border-red-200 dark:border-red-900/50 bg-red-50 dark:bg-red-950/20 p-4 text-xs text-red-700 dark:text-red-300 font-mono">
          <strong>Allocation Engine Error:</strong> {error}
        </div>
      )}

      {/* Recommended Candidate Callout Card */}
      {data && (
        <div
          className={cn(
            "rounded-xl border-2 p-6 shadow-sm font-mono text-xs transition-colors",
            data.has_suitable_candidate && data.recommended_candidate
              ? "border-emerald-500 bg-emerald-50/70 dark:bg-emerald-950/20 text-emerald-900 dark:text-emerald-200"
              : "border-amber-500 bg-amber-50/70 dark:bg-amber-950/20 text-amber-900 dark:text-amber-200"
          )}
        >
          <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-slate-900 text-white">
                  RECOMMENDATION RESULT
                </span>
                <span className="text-xs text-slate-500">
                  {data.total_candidates} Candidate Channels Analyzed
                </span>
              </div>

              {data.has_suitable_candidate && data.recommended_candidate ? (
                <>
                  <h3 className="text-3xl sm:text-4xl font-black tracking-tight mt-2 text-emerald-600 dark:text-emerald-400">
                    RECOMMENDED CANDIDATE: {data.recommended_candidate.center_freq_mhz} MHz
                  </h3>
                  <p className="text-xs mt-1 text-slate-700 dark:text-slate-300 max-w-2xl font-sans">
                    Optimal candidate channel identified. Lowest observed RF activity, safe guard-band margin, low interference risk, and clean safety status.
                  </p>
                </>
              ) : (
                <>
                  <h3 className="text-3xl sm:text-4xl font-black tracking-tight mt-2 text-amber-600 dark:text-amber-400">
                    NO SUITABLE CANDIDATE
                  </h3>
                  <p className="text-xs mt-1 text-slate-700 dark:text-slate-300 max-w-2xl font-sans">
                    All analyzed channels are occupied, out-of-distribution, or fall below the minimum recommendation safety score threshold (50.0).
                  </p>
                </>
              )}
            </div>

            {data.recommended_candidate && (
              <div className="text-right flex flex-col sm:items-end gap-1 shrink-0">
                <div className="text-2xl font-black text-emerald-600 dark:text-emerald-400">
                  Score: {data.recommended_candidate.score} / 100
                </div>
                <button
                  onClick={() => setTraceCandidate(data.recommended_candidate)}
                  className="mt-1 inline-flex items-center gap-1.5 px-3 py-1 bg-emerald-600 hover:bg-emerald-500 text-white text-[11px] font-bold rounded shadow transition"
                >
                  <Calculator className="w-3.5 h-3.5" />
                  <span>View Calculation Trace</span>
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Candidate Channels Table */}
      {data && (
        <Panel
          title="Analyzed Candidate Channels Table"
          subtitle="Transparent scoring deductions for each generated frequency channel."
        >
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 text-left text-slate-500">
                  <th className="p-3">ID</th>
                  <th
                    className="p-3 cursor-pointer hover:text-slate-900 dark:hover:text-slate-100"
                    onClick={() => handleSort("center_freq_mhz")}
                  >
                    <div className="flex items-center gap-1">
                      Center Freq <ArrowUpDown className="size-3" />
                    </div>
                  </th>
                  <th className="p-3">Bandwidth</th>
                  <th className="p-3">RF Activity</th>
                  <th className="p-3">Noise Floor</th>
                  <th className="p-3">Interference</th>
                  <th className="p-3">Guard Band</th>
                  <th className="p-3">OOD</th>
                  <th
                    className="p-3 cursor-pointer hover:text-slate-900 dark:hover:text-slate-100"
                    onClick={() => handleSort("score")}
                  >
                    <div className="flex items-center gap-1">
                      Score <ArrowUpDown className="size-3" />
                    </div>
                  </th>
                  <th className="p-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                {sortedCandidates.map((c) => {
                  const isRec = c.recommendation_status === "RECOMMENDED";
                  const isRej = c.recommendation_status === "REJECTED";
                  return (
                    <tr
                      key={c.candidate_id}
                      className={cn(
                        "hover:bg-slate-50/50 dark:hover:bg-slate-900/50 transition-colors",
                        isRec && "bg-emerald-50/40 dark:bg-emerald-950/20 font-semibold",
                        selectedCandidate?.candidate_id === c.candidate_id && "ring-1 ring-cyan-400 bg-slate-900"
                      )}
                    >
                      <td className="p-3 font-bold text-slate-700 dark:text-slate-300">{c.candidate_id}</td>
                      <td className="p-3 text-sky-600 dark:text-sky-400 font-bold">{c.center_freq_mhz.toFixed(3)} MHz</td>
                      <td className="p-3 text-slate-600 dark:text-slate-400">{c.bandwidth_khz} kHz</td>
                      <td className="p-3 font-bold">
                        <span
                          className={cn(
                            "px-2 py-0.5 rounded text-[10px]",
                            c.activity === "DETECTED"
                              ? "bg-red-100 text-red-800 dark:bg-red-950/60 dark:text-red-300"
                              : c.activity === "NOT_DETECTED"
                              ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300"
                              : "bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300"
                          )}
                        >
                          {c.activity}
                        </span>
                      </td>
                      <td className="p-3 text-slate-700 dark:text-slate-300">{c.noise_floor_dbm} dBm</td>
                      <td className="p-3 font-semibold text-slate-300">{c.interference_risk || "LOW"}</td>
                      <td className="p-3">
                        <span className={cn(
                          "px-1.5 py-0.5 rounded text-[10px]",
                          c.guard_band_status === "CONFLICT" ? "bg-rose-950 text-rose-300" : (c.guard_band_status === "MARGINAL" ? "bg-amber-950 text-amber-300" : "bg-emerald-950 text-emerald-300")
                        )}>
                          {c.guard_band_status || "SAFE_MARGIN"}
                        </span>
                      </td>
                      <td className="p-3">
                        {c.ood_warning ? (
                          <span className="text-amber-600 font-bold">OOD</span>
                        ) : (
                          <span className="text-emerald-600">PASS</span>
                        )}
                      </td>
                      <td className="p-3 font-bold text-slate-900 dark:text-slate-100 text-sm">
                        {c.score.toFixed(1)}
                      </td>
                      <td className="p-3 text-right">
                        <button
                          onClick={() => setTraceCandidate(c)}
                          className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-cyan-300 text-[10px] font-mono rounded border border-slate-700 transition"
                        >
                          View Trace
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      {/* Trace Modal */}
      <CalculationTraceModal
        candidate={traceCandidate}
        onClose={() => setTraceCandidate(null)}
      />
    </div>
  );
}
