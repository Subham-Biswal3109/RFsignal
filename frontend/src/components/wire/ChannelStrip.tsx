import React from "react";
import { cn } from "@/lib/utils";
import type { ChannelCandidate } from "@/types/wire-watcher";

interface ChannelStripProps {
  candidates: ChannelCandidate[];
  selectedId?: string;
  onSelect: (c: ChannelCandidate) => void;
}

export const ChannelStrip: React.FC<ChannelStripProps> = ({
  candidates,
  selectedId,
  onSelect,
}) => {
  const colorMap: Record<string, { bg: string; text: string; border: string }> = {
    FREE: {
      bg: "bg-emerald-600 hover:bg-emerald-500",
      text: "text-emerald-100",
      border: "border-emerald-500/80",
    },
    BUSY: {
      bg: "bg-rose-600 hover:bg-rose-500",
      text: "text-rose-100",
      border: "border-rose-500/80",
    },
    UNCERTAIN: {
      bg: "bg-amber-600 hover:bg-amber-500",
      text: "text-amber-100",
      border: "border-amber-500/80",
    },
    RECOMMENDED: {
      bg: "bg-purple-600 hover:bg-purple-500",
      text: "text-purple-100",
      border: "border-purple-400",
    },
    REJECTED: {
      bg: "bg-slate-700 hover:bg-slate-600",
      text: "text-slate-300",
      border: "border-slate-600",
    },
  };

  return (
    <div className="space-y-2">
      <div
        className="flex overflow-x-auto gap-1 p-2 rounded-lg bg-slate-950 border border-slate-800 scrollbar-thin"
        aria-label="Channel Allocation Strip"
      >
        {candidates.map((c) => {
          const isSelected = selectedId === c.candidate_id;
          const isRec = c.recommendation_status === "RECOMMENDED";
          const styling = colorMap[c.recommendation_status] ?? colorMap.FREE;

          return (
            <div
              key={c.candidate_id}
              className="relative group shrink-0"
            >
              <button
                type="button"
                onClick={() => onSelect(c)}
                className={cn(
                  "flex flex-col items-center justify-center px-1.5 py-1 text-[10px] font-mono rounded border transition-all cursor-pointer h-12",
                  styling.bg,
                  styling.text,
                  styling.border,
                  isSelected && "ring-2 ring-cyan-400 scale-105 z-10 shadow-lg",
                  isRec && "font-bold ring-1 ring-purple-300 shadow-purple-500/30 shadow-sm"
                )}
                style={{ minWidth: "48px" }}
                aria-label={`Channel ${c.candidate_id}: ${c.center_freq_mhz.toFixed(3)} MHz, Status ${c.recommendation_status}, Score ${c.score.toFixed(1)}`}
              >
                <span className="font-bold">{c.candidate_id}</span>
                <span className="text-[9px] opacity-90">{c.center_freq_mhz.toFixed(1)}M</span>
                <span className="text-[8px] opacity-80">{c.recommendation_status[0]}</span>
              </button>

              {/* Accessible Hover Tooltip */}
              <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 hidden group-hover:block z-50 w-56 p-2.5 bg-slate-900 text-slate-200 text-left rounded-lg shadow-xl border border-slate-700 pointer-events-none text-xs space-y-1 font-mono">
                <div className="font-bold text-sky-400 flex items-center justify-between border-b border-slate-800 pb-1">
                  <span>{c.candidate_id}</span>
                  <span>{c.center_freq_mhz.toFixed(3)} MHz</span>
                </div>
                <div className="text-[11px] space-y-0.5 pt-0.5">
                  <div>Status: <span className="font-bold text-slate-100">{c.recommendation_status}</span></div>
                  <div>Score: <span className="font-bold text-amber-300">{c.score.toFixed(1)} / 100</span></div>
                  <div>Bandwidth: {c.bandwidth_khz} kHz</div>
                  <div>RF Activity: {c.activity} ({c.availability})</div>
                  <div>Noise Floor Ref: {c.noise_floor_dbm} dBm</div>
                  {c.observed_power_dbm !== null && c.observed_power_dbm !== undefined && (
                    <div>Observed Power: {c.observed_power_dbm} dBm</div>
                  )}
                  {c.snr_db !== null && c.snr_db !== undefined && (
                    <div>Estimated SNR: {c.snr_db} dB</div>
                  )}
                  <div>Guard Band: {c.guard_band_status || "SAFE_MARGIN"}</div>
                  <div>Interference: {c.interference_risk || "LOW"}</div>
                  <div>OOD Safety: {c.ood_warning ? "OOD WARNING" : "PASSED"}</div>
                  {c.assessment_explanation?.summary && (
                    <div className="text-[10px] text-slate-400 pt-1 border-t border-slate-800 font-sans">
                      {c.assessment_explanation.summary}
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

