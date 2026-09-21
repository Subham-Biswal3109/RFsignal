/**
 * Wire Watcher — RF Spectrum Channel Allocation & Engineering Scoring View
 *
 * Demonstrates:
 *   - Engineering channel candidate generation across an analyzed VHF band
 *   - Transparent scoring formula: Score = 100 - ActivityPenalty - NoisePenalty - OODPenalty - UncertaintyPenalty
 *   - Recommended Candidate Channel callout ("RECOMMENDED CANDIDATE" or "NO SUITABLE CANDIDATE")
 *   - Sortable Candidate Channels Table with data provenance tags
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
} from "lucide-react";
import type {
  AllocationRecommendationResponse,
  ChannelCandidate,
} from "@/types/wire-watcher";
import { cn } from "@/lib/utils";

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

  const fetchAllocation = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch("/api/allocation/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          start_freq_mhz: startFreqMhz,
          end_freq_mhz: endFreqMhz,
          channel_bw_mhz: channelBwMhz,
          guard_band_mhz: guardBandMhz,
          noise_floor_dbm: noiseFloorDbm,
          observed_power_dbm: observedPowerDbm,
        }),
      });
      if (!resp.ok) {
        const errJson = await resp.json();
        throw new Error(errJson.error || errJson.details || "Allocation engine request failed");
      }
      const json: AllocationRecommendationResponse = await resp.json();
      setData(json);
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
                RF Spectrum Channel Allocation & Transparent Candidate Scoring
                <span className="text-xs px-2 py-0.5 rounded bg-indigo-100 dark:bg-indigo-900/60 text-indigo-800 dark:text-indigo-300 font-mono border border-indigo-200 dark:border-indigo-800">
                  Engineering Layer
                </span>
              </h2>
              <p className="text-xs text-slate-600 dark:text-slate-400 mt-1 max-w-3xl">
                Divides analyzed spectrum bands into candidate channels, evaluates observed RF activity, applies transparent penalty scoring,
                and recommends optimal candidate channels.
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
                    Optimal candidate channel identified. Lowest observed RF activity, acceptable noise floor, and clean in-distribution safety status.
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
                <div className="text-[11px] text-slate-600 dark:text-slate-400">
                  Bandwidth: {data.recommended_candidate.bandwidth_khz} kHz (Guard: {data.recommended_candidate.guard_band_mhz * 1000} kHz)
                </div>
              </div>
            )}
          </div>

          {/* Breakdown for Recommended Candidate */}
          {data.recommended_candidate && (
            <div className="mt-5 pt-4 border-t border-slate-200/80 dark:border-slate-800/80 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <div className="p-3 rounded bg-white/80 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 space-y-1">
                <div className="text-[10px] text-slate-500 uppercase font-bold">Frequency Range</div>
                <div className="text-sm font-bold text-slate-900 dark:text-slate-100">
                  {data.recommended_candidate.start_freq_mhz} - {data.recommended_candidate.end_freq_mhz} MHz
                </div>
              </div>

              <div className="p-3 rounded bg-white/80 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 space-y-1">
                <div className="text-[10px] text-slate-500 uppercase font-bold">RF Activity State</div>
                <div className="text-sm font-bold text-emerald-600 dark:text-emerald-400">
                  {data.recommended_candidate.activity} ({data.recommended_candidate.availability})
                </div>
              </div>

              <div className="p-3 rounded bg-white/80 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 space-y-1">
                <div className="text-[10px] text-slate-500 uppercase font-bold">Noise & SNR</div>
                <div className="text-sm font-bold text-indigo-600 dark:text-indigo-400">
                  {data.recommended_candidate.noise_floor_dbm} dBm (SNR: +{data.recommended_candidate.snr_db} dB)
                </div>
              </div>

              <div className="p-3 rounded bg-white/80 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 space-y-1">
                <div className="text-[10px] text-slate-500 uppercase font-bold">OOD Guard Status</div>
                <div className="text-sm font-bold text-emerald-600 dark:text-emerald-400">
                  PASS (In-Distribution)
                </div>
              </div>
            </div>
          )}
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
                  <th className="p-3">SNR</th>
                  <th className="p-3">OOD</th>
                  <th
                    className="p-3 cursor-pointer hover:text-slate-900 dark:hover:text-slate-100"
                    onClick={() => handleSort("score")}
                  >
                    <div className="flex items-center gap-1">
                      Score <ArrowUpDown className="size-3" />
                    </div>
                  </th>
                  <th className="p-3">Recommendation</th>
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
                        isRec && "bg-emerald-50/40 dark:bg-emerald-950/20 font-semibold"
                      )}
                    >
                      <td className="p-3 font-bold text-slate-700 dark:text-slate-300">{c.candidate_id}</td>
                      <td className="p-3 text-sky-600 dark:text-sky-400 font-bold">{c.center_freq_mhz} MHz</td>
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
                      <td className="p-3 text-slate-700 dark:text-slate-300">
                        {c.snr_db !== null ? `+${c.snr_db} dB` : "—"}
                      </td>
                      <td className="p-3">
                        {c.ood_warning ? (
                          <span className="text-amber-600 font-bold">OOD</span>
                        ) : (
                          <span className="text-emerald-600">PASS</span>
                        )}
                      </td>
                      <td className="p-3 font-bold text-slate-900 dark:text-slate-100 text-sm">
                        {c.score}
                      </td>
                      <td className="p-3">
                        <span
                          className={cn(
                            "px-2 py-0.5 rounded text-[10px] font-bold tracking-wider",
                            isRec
                              ? "bg-emerald-600 text-white"
                              : isRej
                              ? "bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-400"
                              : "bg-amber-500 text-white"
                          )}
                        >
                          {c.recommendation_status}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      {/* Engineering Scoring Formula Reference */}
      <Panel
        title="Candidate Channel Scoring Formula Rationale"
        subtitle="Transparent weight parameters used by the Channel Allocation Engine."
      >
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 font-mono text-xs">
          <div className="p-3 rounded border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/40">
            <div className="text-[10px] uppercase font-bold text-slate-500">Base Candidate Score</div>
            <div className="text-lg font-bold text-emerald-600 dark:text-emerald-400 mt-0.5">100.0 Points</div>
            <p className="text-[11px] text-slate-600 dark:text-slate-400 font-sans mt-1">Starting reference for clear spectrum.</p>
          </div>

          <div className="p-3 rounded border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/40">
            <div className="text-[10px] uppercase font-bold text-slate-500">Occupied Activity Penalty</div>
            <div className="text-lg font-bold text-red-600 dark:text-red-400 mt-0.5">-50.0 Points</div>
            <p className="text-[11px] text-slate-600 dark:text-slate-400 font-sans mt-1">Deducted if RF activity is DETECTED.</p>
          </div>

          <div className="p-3 rounded border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/40">
            <div className="text-[10px] uppercase font-bold text-slate-500">Out-of-Distribution Penalty</div>
            <div className="text-lg font-bold text-amber-600 dark:text-amber-400 mt-0.5">-40.0 Points</div>
            <p className="text-[11px] text-slate-600 dark:text-slate-400 font-sans mt-1">Deducted if input parameters trigger OOD bounds.</p>
          </div>

          <div className="p-3 rounded border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/40">
            <div className="text-[10px] uppercase font-bold text-slate-500">Uncertainty Penalty</div>
            <div className="text-lg font-bold text-purple-600 dark:text-purple-400 mt-0.5">-25.0 Points</div>
            <p className="text-[11px] text-slate-600 dark:text-slate-400 font-sans mt-1">Deducted when observation evidence is ambiguous.</p>
          </div>
        </div>
      </Panel>
    </div>
  );
}
