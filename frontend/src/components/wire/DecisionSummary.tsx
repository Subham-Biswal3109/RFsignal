import React from "react";
import { cn } from "@/lib/utils";
import type { ChannelCandidate } from "@/types/wire-watcher";
import { ShieldCheck, ShieldAlert, Sparkles, AlertCircle, Info } from "lucide-react";

interface DecisionSummaryProps {
  candidate?: ChannelCandidate;
  threshold: number;
}

export const DecisionSummary: React.FC<DecisionSummaryProps> = ({
  candidate,
  threshold,
}) => {
  if (!candidate) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-xs font-mono text-slate-400">
        Select a channel from the spectrum chart, strip, or table to view its decision summary.
      </div>
    );
  }

  const {
    candidate_id,
    center_freq_mhz,
    recommendation_status,
    score,
    activity,
    availability,
    ood_warning,
    guard_band_status,
    interference_risk,
    assessment_explanation,
  } = candidate;

  // Determine RF Status
  let rfStatus = "FREE";
  if (availability === "OCCUPIED" || activity === "DETECTED") {
    rfStatus = "BUSY";
  } else if (ood_warning || availability === "UNCERTAIN" || activity === "UNCERTAIN") {
    rfStatus = "UNCERTAIN";
  }

  // Recommendation Eligibility
  const meetsThreshold = score >= threshold;
  const passesSafety = !ood_warning && guard_band_status !== "CONFLICT" && interference_risk !== "HIGH";
  const isEligible = meetsThreshold && passesSafety && rfStatus === "FREE";

  // Decision explanation
  let explanation = assessment_explanation?.summary;
  if (!explanation) {
    if (recommendation_status === "RECOMMENDED") {
      explanation = "Recommended: passed all allocation checks, noise margins, and safety constraints.";
    } else if (recommendation_status === "FREE") {
      explanation = "Free: physical sensing confirms channel is unoccupied with safe guard-band margins.";
    } else if (rfStatus === "BUSY") {
      explanation = "Busy: classified as occupied by physical sensing / active transmission detection.";
    } else if (guard_band_status === "CONFLICT") {
      explanation = "Rejected: active transmitter frequency conflict within channel bandwidth.";
    } else if (ood_warning) {
      explanation = "Uncertain: out-of-distribution observation or insufficient evidence.";
    } else if (!meetsThreshold) {
      explanation = `Rejected: allocation score (${score.toFixed(1)}) falls below the minimum safety threshold (${threshold}).`;
    } else {
      explanation = "Rejected: failed one or more guard-band, noise, or interference safety criteria.";
    }
  }

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 font-mono text-xs shadow-md space-y-3">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div>
          <span className="text-[10px] text-slate-400 uppercase">Selected Channel</span>
          <h4 className="text-sm font-bold text-sky-400">
            {candidate_id} — {center_freq_mhz.toFixed(3)} MHz
          </h4>
        </div>
        <span
          className={cn(
            "px-2 py-0.5 rounded text-[10px] font-bold uppercase",
            recommendation_status === "RECOMMENDED"
              ? "bg-purple-900/80 text-purple-200 border border-purple-500"
              : recommendation_status === "FREE"
              ? "bg-emerald-900/80 text-emerald-200 border border-emerald-500"
              : recommendation_status === "BUSY"
              ? "bg-rose-900/80 text-rose-200 border border-rose-500"
              : recommendation_status === "REJECTED"
              ? "bg-slate-800 text-slate-300 border border-slate-600"
              : "bg-amber-900/80 text-amber-200 border border-amber-500"
          )}
        >
          {recommendation_status}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2 text-[11px]">
        <div className="p-2 rounded bg-slate-950 border border-slate-800/80">
          <span className="text-slate-400 text-[10px] block">RF Physical Status</span>
          <span
            className={cn(
              "font-bold",
              rfStatus === "FREE" ? "text-emerald-400" : (rfStatus === "BUSY" ? "text-rose-400" : "text-amber-400")
            )}
          >
            {rfStatus} ({activity})
          </span>
        </div>

        <div className="p-2 rounded bg-slate-950 border border-slate-800/80">
          <span className="text-slate-400 text-[10px] block">Allocation Score</span>
          <span className="font-bold text-slate-100">
            {score.toFixed(1)} <span className="text-slate-500 font-normal">/ 100</span>
          </span>
          <span className="text-[9px] text-slate-400 block">
            {meetsThreshold ? "✓ Meets &ge; 50 threshold" : "✗ Below 50 threshold"}
          </span>
        </div>

        <div className="p-2 rounded bg-slate-950 border border-slate-800/80">
          <span className="text-slate-400 text-[10px] block">Guard Band & Risk</span>
          <span className="font-semibold text-slate-300">
            {guard_band_status || "SAFE_MARGIN"} / {interference_risk || "LOW"}
          </span>
        </div>

        <div className="p-2 rounded bg-slate-950 border border-slate-800/80">
          <span className="text-slate-400 text-[10px] block">Eligibility Check</span>
          <span className={cn("font-bold", isEligible ? "text-emerald-400" : "text-amber-400")}>
            {isEligible ? "ELIGIBLE" : "INELIGIBLE"}
          </span>
        </div>
      </div>

      <div className="pt-2 border-t border-slate-800/80 font-sans text-xs text-slate-300">
        <strong className="block text-[11px] font-mono text-slate-400 uppercase mb-1">
          Decision Explanation:
        </strong>
        <p className="leading-relaxed bg-slate-950/60 p-2.5 rounded border border-slate-800">
          {explanation}
        </p>
      </div>
    </div>
  );
};

