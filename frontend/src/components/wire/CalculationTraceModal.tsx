/**
 * Wire Watcher — Engineering Calculation Trace Modal (Section 10)
 *
 * Displays exact mathematical calculation traces formatted into:
 * INPUT, FORMULA, SUBSTITUTION, RESULT, UNIT, and ENGINEERING NOTE.
 */

import React from "react";
import { X, Calculator, ShieldCheck, Info } from "lucide-react";
import type { ChannelCandidate } from "@/types/wire-watcher";

interface Props {
  candidate: ChannelCandidate | null;
  onClose: () => void;
}

export function CalculationTraceModal({ candidate, onClose }: Props) {
  if (!candidate) return null;

  const trace = candidate.assessment_explanation?.trace;
  const breakdown = candidate.scoring_breakdown;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fade-in">
      <div className="relative w-full max-w-2xl bg-slate-900 border border-slate-700 rounded-xl shadow-2xl overflow-hidden text-slate-200">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 bg-slate-950 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <Calculator className="w-5 h-5 text-cyan-400" />
            <h3 className="text-base font-semibold text-slate-100 font-mono">
              Engineering Trace — Candidate {candidate.candidate_id} ({candidate.center_freq_mhz.toFixed(3)} MHz)
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-6 space-y-5 max-h-[80vh] overflow-y-auto font-sans">
          {/* Headline & Recommendation Banner */}
          <div className="p-4 bg-slate-950/80 rounded-lg border border-slate-800 flex items-center justify-between">
            <div>
              <div className="text-xs uppercase tracking-wider text-slate-400 font-mono">Final Decision</div>
              <div className="text-lg font-bold font-mono text-cyan-300">
                {candidate.assessment_explanation?.recommendation || candidate.recommendation_status}
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs uppercase tracking-wider text-slate-400 font-mono">Channel Score</div>
              <div className="text-xl font-extrabold font-mono text-amber-400">{candidate.score.toFixed(1)} / 100</div>
            </div>
          </div>

          {/* Mathematical Trace Box */}
          <div className="space-y-3 bg-slate-950 p-4 rounded-lg border border-slate-800 font-mono text-xs">
            <div className="text-slate-400 uppercase tracking-wider font-bold border-b border-slate-800 pb-1">
              Mathematical Derivation Steps
            </div>

            {/* 1. INPUT */}
            <div>
              <span className="text-cyan-400 font-bold">1. INPUT:</span>
              <div className="ml-4 mt-1 grid grid-cols-2 gap-2 text-slate-300">
                <div>Center Freq: <span className="text-white font-semibold">{candidate.center_freq_mhz.toFixed(3)} MHz</span></div>
                <div>Bandwidth: <span className="text-white font-semibold">{candidate.bandwidth_khz} kHz</span></div>
                <div>Noise Floor: <span className="text-white font-semibold">{candidate.noise_floor_dbm} dBm</span></div>
                <div>Activity State: <span className="text-white font-semibold">{candidate.activity}</span></div>
                <div>Guard Band: <span className="text-white font-semibold">{candidate.guard_band_status || "SAFE_MARGIN"}</span></div>
                <div>Interference: <span className="text-white font-semibold">{candidate.interference_risk || "LOW"}</span></div>
              </div>
            </div>

            {/* 2. FORMULA */}
            <div>
              <span className="text-cyan-400 font-bold">2. FORMULA:</span>
              <div className="ml-4 mt-1 text-slate-300 bg-slate-900/80 p-2 rounded border border-slate-800 text-[11px]">
                {trace?.formula || "Score = BaseScore(100) - ActivityPenalty - UncertaintyPenalty - OODPenalty - NoisePenalty - GuardBandPenalty - InterferencePenalty"}
              </div>
            </div>

            {/* 3. SUBSTITUTION */}
            <div>
              <span className="text-cyan-400 font-bold">3. SUBSTITUTION:</span>
              <div className="ml-4 mt-1 text-slate-300 bg-slate-900/80 p-2 rounded border border-slate-800 text-[11px] space-y-1">
                <div>Base Score: +100.0</div>
                {breakdown.activity_penalty > 0 && <div className="text-rose-400">Activity Penalty: -{breakdown.activity_penalty.toFixed(1)}</div>}
                {breakdown.uncertainty_penalty > 0 && <div className="text-amber-400">Uncertainty Penalty: -{breakdown.uncertainty_penalty.toFixed(1)}</div>}
                {breakdown.ood_penalty > 0 && <div className="text-purple-400">OOD Penalty: -{breakdown.ood_penalty.toFixed(1)}</div>}
                {breakdown.noise_penalty > 0 && <div className="text-sky-400">Noise Penalty: -{breakdown.noise_penalty.toFixed(1)}</div>}
                {(breakdown.guard_band_penalty ?? 0) > 0 && <div className="text-orange-400">Guard Band Penalty: -{(breakdown.guard_band_penalty ?? 0).toFixed(1)}</div>}
                {(breakdown.interference_penalty ?? 0) > 0 && <div className="text-red-400">Interference Penalty: -{(breakdown.interference_penalty ?? 0).toFixed(1)}</div>}
              </div>
            </div>

            {/* 4. RESULT */}
            <div>
              <span className="text-cyan-400 font-bold">4. RESULT:</span>
              <div className="ml-4 mt-1 text-slate-100 font-bold text-sm">
                Candidate Score = {candidate.score.toFixed(1)} / 100.0
              </div>
            </div>

            {/* 5. UNIT */}
            <div>
              <span className="text-cyan-400 font-bold">5. UNIT:</span>
              <div className="ml-4 mt-1 text-slate-300">
                {trace?.unit || "Normalized Engineering Quality Score (0–100)"}
              </div>
            </div>

            {/* 6. ENGINEERING NOTE */}
            <div>
              <span className="text-cyan-400 font-bold">6. ENGINEERING NOTE:</span>
              <div className="ml-4 mt-1 text-slate-400 text-[11px] italic bg-slate-900/60 p-2 rounded border border-slate-800/80">
                {trace?.engineering_note || "Candidate scores represent empirical RF activity and noise assessment only. Not a legal or regulatory transmission clearance."}
              </div>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 bg-slate-950 border-t border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg transition"
          >
            Close Trace
          </button>
        </div>
      </div>
    </div>
  );
}
