/**
 * Wire Watcher — Visual Channel Allocation Heatmap (Section 7)
 *
 * Visualizes frequency candidates across the spectrum band with state markers,
 * candidate scores, guard-band margins, and recommendation indicators.
 */

import React from "react";
import type { ChannelCandidate } from "@/types/wire-watcher";
import { cn } from "@/lib/utils";
import { Sparkles, AlertCircle, ShieldAlert, CheckCircle2 } from "lucide-react";

interface Props {
  candidates: ChannelCandidate[];
  recommendedCandidate: ChannelCandidate | null;
  selectedCandidate: ChannelCandidate | null;
  onSelectCandidate: (candidate: ChannelCandidate) => void;
  startFreqMhz: number;
  endFreqMhz: number;
}

export function ChannelAllocationHeatmap({
  candidates,
  recommendedCandidate,
  selectedCandidate,
  onSelectCandidate,
  startFreqMhz,
  endFreqMhz,
}: Props) {
  if (!candidates || candidates.length === 0) {
    return (
      <div className="p-6 text-center text-slate-400 bg-slate-900/50 rounded-lg border border-slate-800">
        No candidate channels available to visualize.
      </div>
    );
  }

  // Calculate statistics directly from the exact dataset rendering the heatmap
  const totalCandidates = candidates.length;
  const availableCandidates = candidates.filter((c) => c.availability === "AVAILABLE");
  const occupiedCandidates = candidates.filter((c) => c.availability === "OCCUPIED");
  const uncertainCandidates = candidates.filter((c) => c.availability === "UNCERTAIN");
  const rejectedCandidates = candidates.filter((c) => c.availability === "REJECTED");

  const availableCount = availableCandidates.length;
  const occupiedCount = occupiedCandidates.length;
  const uncertainCount = uncertainCandidates.length;
  const rejectedCount = rejectedCandidates.length;

  const availabilityPercentage =
    totalCandidates > 0 ? ((availableCount / totalCandidates) * 100).toFixed(1) : "0.0";
  const averageScore =
    totalCandidates > 0
      ? (candidates.reduce((sum, c) => sum + c.score, 0) / totalCandidates).toFixed(1)
      : "0.0";
  const recommendedCount = candidates.filter(
    (c) =>
      c.recommendation_status === "RECOMMENDED" ||
      recommendedCandidate?.candidate_id === c.candidate_id
  ).length;

  // Sort candidates by center frequency for linear spectrum layout
  const sortedByFreq = [...candidates].sort((a, b) => a.center_freq_mhz - b.center_freq_mhz);

  return (
    <div className="space-y-4 bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-xl">
      {/* Title & Metadata Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
              RF Channel Allocation Heatmap
            </h3>
            <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-indigo-900/80 text-indigo-300 border border-indigo-700">
              DEMONSTRATION DATA
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Spectrum Band: {startFreqMhz.toFixed(1)} MHz — {endFreqMhz.toFixed(1)} MHz ({totalCandidates} Candidate Channels)
          </p>
        </div>

        {/* Heatmap Legend */}
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5 text-emerald-400">
            <span className="w-3.5 h-3.5 rounded bg-emerald-950 border border-emerald-500" />
            <span className="font-semibold">AVAILABLE (Green)</span>
          </div>
          <div className="flex items-center gap-1.5 text-rose-400">
            <span className="w-3.5 h-3.5 rounded bg-rose-950 border border-rose-500" />
            <span className="font-semibold">OCCUPIED (Red)</span>
          </div>
          <div className="flex items-center gap-1.5 text-amber-400">
            <span className="w-3.5 h-3.5 rounded bg-amber-950 border border-amber-500" />
            <span className="font-semibold">UNCERTAIN (Orange)</span>
          </div>
          <div className="flex items-center gap-1.5 text-slate-300">
            <span className="w-3.5 h-3.5 rounded bg-slate-800 border border-slate-500" />
            <span className="font-semibold">REJECTED (Grey)</span>
          </div>
          <div className="flex items-center gap-1.5 text-purple-300">
            <span className="w-3.5 h-3.5 rounded bg-purple-950 border-2 border-purple-400 flex items-center justify-center">
              <Sparkles className="w-2.5 h-2.5 text-purple-200" />
            </span>
            <span className="font-semibold">RECOMMENDED (Purple Outline)</span>
          </div>
        </div>
      </div>

      {/* Step 6: Distribution Summary Above Heatmap */}
      <div className="space-y-2.5">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="p-3.5 rounded-lg border border-emerald-500/60 bg-emerald-950/40 flex flex-col shadow-sm">
            <span className="text-emerald-300 text-[11px] uppercase font-bold tracking-wider flex items-center justify-between">
              AVAILABLE
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
            </span>
            <span className="text-2xl sm:text-3xl font-black font-mono text-emerald-300 mt-1">
              {availableCount}
            </span>
            <span className="text-[10px] font-mono text-emerald-400/80 mt-0.5">
              Channels free for safe allocation
            </span>
          </div>

          <div className="p-3.5 rounded-lg border border-rose-500/60 bg-rose-950/40 flex flex-col shadow-sm">
            <span className="text-rose-300 text-[11px] uppercase font-bold tracking-wider flex items-center justify-between">
              OCCUPIED
              <span className="w-2 h-2 rounded-full bg-rose-400" />
            </span>
            <span className="text-2xl sm:text-3xl font-black font-mono text-rose-300 mt-1">
              {occupiedCount}
            </span>
            <span className="text-[10px] font-mono text-rose-400/80 mt-0.5">
              Active RF transmitter clusters
            </span>
          </div>

          <div className="p-3.5 rounded-lg border border-amber-500/60 bg-amber-950/40 flex flex-col shadow-sm">
            <span className="text-amber-300 text-[11px] uppercase font-bold tracking-wider flex items-center justify-between">
              UNCERTAIN
              <span className="w-2 h-2 rounded-full bg-amber-400" />
            </span>
            <span className="text-2xl sm:text-3xl font-black font-mono text-amber-300 mt-1">
              {uncertainCount}
            </span>
            <span className="text-[10px] font-mono text-amber-400/80 mt-0.5">
              Isolated weak / unverified channels
            </span>
          </div>

          <div className="p-3.5 rounded-lg border border-slate-600/70 bg-slate-800/60 flex flex-col shadow-sm">
            <span className="text-slate-300 text-[11px] uppercase font-bold tracking-wider flex items-center justify-between">
              REJECTED
              <span className="w-2 h-2 rounded-full bg-slate-400" />
            </span>
            <span className="text-2xl sm:text-3xl font-black font-mono text-slate-300 mt-1">
              {rejectedCount}
            </span>
            <span className="text-[10px] font-mono text-slate-400 mt-0.5">
              OOD safety & guard conflict channels
            </span>
          </div>
        </div>

        {/* Aggregate KPI Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs font-mono bg-slate-950/80 p-3 rounded-lg border border-slate-800">
          <div>
            <span className="text-slate-400 text-[10px] uppercase block">Total Candidates</span>
            <span className="text-base font-bold text-slate-200">{totalCandidates}</span>
          </div>
          <div>
            <span className="text-slate-400 text-[10px] uppercase block">Availability Percentage</span>
            <span className="text-base font-bold text-emerald-400">{availabilityPercentage}%</span>
          </div>
          <div>
            <span className="text-slate-400 text-[10px] uppercase block">Average Allocation Score</span>
            <span className="text-base font-bold text-cyan-300">{averageScore} pts</span>
          </div>
          <div>
            <span className="text-slate-400 text-[10px] uppercase block">Recommended Candidate Count</span>
            <span className="text-base font-bold text-purple-300">{recommendedCount} candidates</span>
          </div>
        </div>
      </div>

      {/* Heatmap Grid Section */}
      <div className="space-y-2 pt-2 border-t border-slate-800">
        <div className="flex justify-between text-[11px] font-mono text-slate-400 px-1">
          <span>{startFreqMhz.toFixed(1)} MHz (Low VHF)</span>
          <span>Spectrum Heatmap (70.0 — 160.0 MHz)</span>
          <span>{endFreqMhz.toFixed(1)} MHz (High VHF)</span>
        </div>

        {/* Step 4: Heatmap Cards Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8 2xl:grid-cols-10 gap-2 p-3 bg-slate-950/90 rounded-lg border border-slate-800 max-h-[580px] overflow-y-auto">
          {sortedByFreq.map((cand) => {
            const isRecommended =
              cand.recommendation_status === "RECOMMENDED" ||
              recommendedCandidate?.candidate_id === cand.candidate_id;
            const isSelected = selectedCandidate?.candidate_id === cand.candidate_id;

            // Step 4 Colors:
            // Green for AVAILABLE, Red for OCCUPIED, Orange for UNCERTAIN, Grey for REJECTED, Purple outline for RECOMMENDED
            let cardBg = "";
            let statusTextClass = "";

            if (cand.availability === "OCCUPIED") {
              cardBg = "bg-rose-950/85 border-rose-600/80 text-rose-200 hover:bg-rose-900/90";
              statusTextClass = "text-rose-300 bg-rose-900/40 border-rose-700/60";
            } else if (cand.availability === "UNCERTAIN") {
              cardBg = "bg-amber-950/85 border-amber-600/80 text-amber-200 hover:bg-amber-900/90";
              statusTextClass = "text-amber-300 bg-amber-900/40 border-amber-700/60";
            } else if (cand.availability === "REJECTED") {
              cardBg = "bg-slate-800/85 border-slate-600/80 text-slate-300 hover:bg-slate-700/90";
              statusTextClass = "text-slate-300 bg-slate-700/50 border-slate-600";
            } else {
              // AVAILABLE
              cardBg = "bg-emerald-950/85 border-emerald-500/80 text-emerald-200 hover:bg-emerald-900/90";
              statusTextClass = "text-emerald-300 bg-emerald-900/40 border-emerald-700/60";
            }

            return (
              <button
                key={cand.candidate_id}
                onClick={() => onSelectCandidate(cand)}
                className={cn(
                  "relative group flex flex-col justify-between p-2 rounded-lg border text-left transition-all cursor-pointer min-h-[74px] shadow-sm",
                  cardBg,
                  isRecommended && "ring-2 ring-purple-400 border-purple-400 shadow-md shadow-purple-500/30",
                  isSelected && "ring-2 ring-cyan-400 border-cyan-400"
                )}
                title={`Channel ${cand.candidate_id} | ${cand.center_freq_mhz.toFixed(3)} MHz | ${cand.availability} | ${cand.score} pts`}
              >
                {/* Recommended Badge Indicator */}
                {isRecommended && (
                  <span className="absolute -top-1.5 -right-1.5 px-1 py-0.2 rounded-full bg-purple-500 text-white text-[9px] font-bold flex items-center gap-0.5 shadow">
                    <Sparkles className="w-2.5 h-2.5" />
                    REC
                  </span>
                )}

                {/* Top Row: Channel ID & Score */}
                <div className="flex items-center justify-between w-full font-mono text-[11px] leading-tight">
                  <span className="font-bold text-slate-100">{cand.candidate_id}</span>
                  <span className="font-black px-1.5 py-0.5 rounded bg-black/60 text-white border border-white/10 shadow-sm text-[10px]">
                    {cand.score.toFixed(0)} pts
                  </span>
                </div>

                {/* Middle Row: Frequency */}
                <div className="font-mono text-[11px] text-slate-200 font-semibold my-0.5">
                  {cand.center_freq_mhz.toFixed(1)} MHz
                </div>

                {/* Bottom Row: State Badge */}
                <div className="flex items-center justify-between w-full mt-0.5">
                  <span
                    className={cn(
                      "text-[9px] font-mono font-bold uppercase px-1.5 py-0.5 rounded border",
                      statusTextClass
                    )}
                  >
                    {cand.availability}
                  </span>
                  <span className="text-[9px] font-mono text-slate-400">
                    {cand.bandwidth_khz}k
                  </span>
                </div>

                {/* Tooltip on Hover */}
                <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 hidden group-hover:block z-50 w-56 p-3 bg-slate-900 text-slate-200 text-left rounded-lg shadow-2xl border border-slate-700 pointer-events-none text-xs space-y-1 font-sans">
                  <div className="font-bold font-mono text-cyan-400 flex items-center justify-between">
                    <span>{cand.candidate_id}</span>
                    <span>{cand.center_freq_mhz.toFixed(3)} MHz</span>
                  </div>
                  <div className="text-[11px] text-slate-300 space-y-0.5 pt-1">
                    <div>
                      Status:{" "}
                      <span className="font-bold text-white">
                        {cand.availability}
                      </span>
                    </div>
                    <div>
                      Allocation Score:{" "}
                      <span className="font-bold text-amber-300">
                        {cand.score.toFixed(1)} / 100
                      </span>
                    </div>
                    <div>
                      Recommendation:{" "}
                      <span className="font-semibold text-purple-300">
                        {cand.recommendation_status}
                      </span>
                    </div>
                    <div>Noise Floor: {cand.noise_floor_dbm.toFixed(1)} dBm</div>
                    <div>Guard Band: {cand.guard_band_status}</div>
                    <div>Interference: {cand.interference_risk}</div>
                    <div>OOD Safety: {cand.ood_warning ? "WARNING" : "PASS"}</div>
                    {cand.assessment_explanation?.summary && (
                      <div className="text-[10px] text-slate-400 pt-1 border-t border-slate-800 line-clamp-2">
                        {cand.assessment_explanation.summary}
                      </div>
                    )}
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
