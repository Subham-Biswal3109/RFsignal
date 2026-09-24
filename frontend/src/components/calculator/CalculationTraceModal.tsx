/**
 * Wire Watcher — Generic Calculation Trace Modal for RF Calculator (Section 38)
 *
 * Displays step-by-step mathematical calculation traces formatted into:
 * INPUT, FORMULA, SUBSTITUTION, RESULT, UNIT, and ENGINEERING INTERPRETATION.
 */

import React from "react";
import { X, Calculator, Info } from "lucide-react";

export interface TraceData {
  title: string;
  inputs: Record<string, string>;
  formula: string;
  substitution: string;
  result: string;
  unit: string;
  interpretation: string;
}

interface Props {
  trace: TraceData | null;
  onClose: () => void;
}

export function CalculationTraceModal({ trace, onClose }: Props) {
  if (!trace) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-fade-in font-sans">
      <div className="relative w-full max-w-2xl bg-slate-900 border border-slate-700 rounded-xl shadow-2xl overflow-hidden text-slate-200">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 bg-slate-950 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <Calculator className="w-5 h-5 text-cyan-400" />
            <h3 className="text-base font-semibold text-slate-100 font-mono">
              Calculation Trace — {trace.title}
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-5 max-h-[80vh] overflow-y-auto">
          <div className="space-y-4 bg-slate-950 p-5 rounded-xl border border-slate-800 font-mono text-xs">
            {/* 1. INPUT */}
            <div>
              <span className="text-cyan-400 font-bold uppercase tracking-wider">1. INPUT:</span>
              <div className="ml-4 mt-1.5 grid grid-cols-2 gap-2 text-slate-300 bg-slate-900/80 p-3 rounded border border-slate-800">
                {Object.entries(trace.inputs).map(([k, v]) => (
                  <div key={k}>
                    <span className="text-slate-400">{k}:</span>{" "}
                    <span className="text-white font-bold">{v}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* 2. FORMULA */}
            <div>
              <span className="text-cyan-400 font-bold uppercase tracking-wider">2. FORMULA:</span>
              <div className="ml-4 mt-1.5 text-slate-200 bg-slate-900/90 p-3 rounded border border-slate-800 text-xs font-bold text-amber-300">
                {trace.formula}
              </div>
            </div>

            {/* 3. SUBSTITUTION */}
            <div>
              <span className="text-cyan-400 font-bold uppercase tracking-wider">3. SUBSTITUTION:</span>
              <div className="ml-4 mt-1.5 text-slate-300 bg-slate-900/80 p-3 rounded border border-slate-800 text-xs">
                {trace.substitution}
              </div>
            </div>

            {/* 4. RESULT */}
            <div>
              <span className="text-cyan-400 font-bold uppercase tracking-wider">4. RESULT:</span>
              <div className="ml-4 mt-1.5 text-emerald-400 font-bold text-sm bg-emerald-950/40 p-3 rounded border border-emerald-800/60">
                {trace.result}
              </div>
            </div>

            {/* 5. UNIT */}
            <div>
              <span className="text-cyan-400 font-bold uppercase tracking-wider">5. UNIT:</span>
              <div className="ml-4 mt-1 text-slate-300 font-bold">
                {trace.unit}
              </div>
            </div>

            {/* 6. ENGINEERING INTERPRETATION */}
            <div>
              <span className="text-cyan-400 font-bold uppercase tracking-wider">6. ENGINEERING INTERPRETATION:</span>
              <div className="ml-4 mt-1.5 text-slate-300 text-[11px] italic bg-slate-900/60 p-3 rounded border border-slate-800/80 font-sans leading-relaxed">
                {trace.interpretation}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 bg-slate-950 border-t border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg transition"
          >
            Close Calculation Trace
          </button>
        </div>
      </div>
    </div>
  );
}
