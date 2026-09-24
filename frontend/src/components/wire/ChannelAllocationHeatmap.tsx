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

  // Sort candidates by center frequency for linear spectrum layout
  const sortedByFreq = [...candidates].sort((a, b) => a.center_freq_mhz - b.center_freq_mhz);

  return (
    <div className="space-y-4 bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-xl">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            RF Channel Allocation Heatmap
          </h3>
          <p className="text-xs text-slate-400">
            Frequency Band: {startFreqMhz} MHz — {endFreqMhz} MHz ({candidates.length} Candidate Channels)
          </p>
        </div>

        {/* Heatmap Legend */}
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5 text-emerald-400">
            <span className="w-3 h-3 rounded bg-emerald-500/30 border border-emerald-500" />
            <span>FREE (Available)</span>
          </div>
          <div className="flex items-center gap-1.5 text-rose-400">
            <span className="w-3 h-3 rounded bg-rose-500/30 border border-rose-500" />
            <span>BUSY (Occupied)</span>
          </div>
          <div className="flex items-center gap-1.5 text-amber-400">
            <span className="w-3 h-3 rounded bg-amber-500/30 border border-amber-500" />
            <span>UNCERTAIN</span>
          </div>
          <div className="flex items-center gap-1.5 text-purple-300">
            <span className="w-3 h-3 rounded bg-purple-500/50 border border-purple-400 flex items-center justify-center">
              <Sparkles className="w-2.5 h-2.5 text-purple-200" />
            </span>
            <span>RECOMMENDED</span>
          </div>
        </div>
      </div>

      {/* Spectrum Heatmap Bar */}
      <div className="space-y-2">
        <div className="flex justify-between text-[11px] font-mono text-slate-400 px-1">
          <span>{startFreqMhz.toFixed(1)} MHz</span>
          <span>Center Spectrum</span>
          <span>{endFreqMhz.toFixed(1)} MHz</span>
        </div>

        <div className="grid grid-cols-12 sm:grid-cols-18 md:grid-cols-24 lg:grid-cols-30 gap-1.5 p-3 bg-slate-950/80 rounded-lg border border-slate-800/80 max-h-60 overflow-y-auto">
          {sortedByFreq.map((cand) => {
            const isRecommended = recommendedCandidate?.candidate_id === cand.candidate_id;
            const isSelected = selectedCandidate?.candidate_id === cand.candidate_id;

            let bgColor = "bg-slate-800 border-slate-700 text-slate-300";
            if (cand.availability === "AVAILABLE") {
              bgColor = "bg-emerald-950/80 border-emerald-500/60 text-emerald-300 hover:bg-emerald-900/90";
            } else if (cand.availability === "OCCUPIED") {
              bgColor = "bg-rose-950/80 border-rose-500/60 text-rose-300 hover:bg-rose-900/90";
            } else {
              bgColor = "bg-amber-950/80 border-amber-500/60 text-amber-300 hover:bg-amber-900/90";
            }

            return (
              <button
                key={cand.candidate_id}
                onClick={() => onSelectCandidate(cand)}
                className={cn(
                  "relative group flex flex-col items-center justify-center p-2 rounded-md border text-center transition-all cursor-pointer min-h-[64px]",
                  bgColor,
                  isRecommended && "ring-2 ring-purple-400 shadow-lg shadow-purple-500/20 bg-purple-950/60 border-purple-400",
                  isSelected && "ring-2 ring-cyan-400 border-cyan-400 shadow-md"
                )}
                title={`Channel ${cand.candidate_id}: ${cand.center_freq_mhz.toFixed(3)} MHz | Score: ${cand.score}/100 | ${cand.availability}`}
              >
                {isRecommended && (
                  <span className="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full bg-purple-500 text-white flex items-center justify-center shadow">
                    <Sparkles className="w-2.5 h-2.5" />
                  </span>
                )}
                <span className="text-[10px] font-mono font-bold">{cand.candidate_id}</span>
                <span className="text-[10px] font-mono text-slate-300 font-semibold mt-0.5">
                  {cand.center_freq_mhz.toFixed(1)}M
                </span>
                <span className="text-[9px] font-mono font-bold mt-1 px-1 rounded bg-black/40">
                  {cand.score.toFixed(0)} pts
                </span>

                {/* Tooltip on Hover */}
                <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 hidden group-hover:block z-50 w-48 p-2.5 bg-slate-900 text-slate-200 text-left rounded-lg shadow-xl border border-slate-700 pointer-events-none text-xs space-y-1 font-sans">
                  <div className="font-bold font-mono text-emerald-400 flex items-center justify-between">
                    <span>{cand.candidate_id}</span>
                    <span>{cand.center_freq_mhz.toFixed(3)} MHz</span>
                  </div>
                  <div className="text-[11px] text-slate-300">
                    <div>Availability: <span className="font-semibold">{cand.availability}</span></div>
                    <div>Score: <span className="font-bold text-amber-300">{cand.score}/100</span></div>
                    <div>Noise: {cand.noise_floor_dbm} dBm</div>
                    <div>Guard Band: {cand.guard_band_status || "SAFE_MARGIN"}</div>
                    <div>Interference: {cand.interference_risk || "LOW"}</div>
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
