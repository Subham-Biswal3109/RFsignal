/**
 * Wire Watcher — RF Source Comparison Table & Measurement Quality Panel (Sections 14 & 15)
 *
 * Compares actual system capabilities across Dataset, Simulation, IQ Replay, and RTL-SDR sources,
 * and provides clear provenance tags and calibration notices.
 */

import React from "react";
import { Panel } from "@/components/wire/Panel";
import { Check, X, ShieldAlert, Cpu, Radio, Database, Sparkles } from "lucide-react";

export function SourceComparisonTable() {
  return (
    <div className="space-y-6">
      {/* Section 15: Measurement Quality Panel */}
      <Panel title="RF Measurement Quality & Data Provenance Panel (Section 15)">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-xs">
          <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg space-y-2">
            <div className="text-cyan-400 font-bold uppercase tracking-wider flex items-center gap-1.5">
              <Radio className="w-4 h-4 text-cyan-400" />
              Physical Hardware Status
            </div>
            <div className="text-slate-300">Source: <span className="font-semibold text-white">RTL-SDR (R820T2)</span></div>
            <div className="text-slate-300">IQ Sample Rate: <span className="font-semibold text-white">2.048 MS/s</span></div>
            <div className="text-slate-300">Calibration: <span className="font-bold text-amber-400">NOT CALIBRATED</span></div>
            <div className="text-slate-300">Absolute Power Accuracy: <span className="font-bold text-rose-400">NOT GUARANTEED</span></div>
            <div className="text-[10px] text-slate-500 italic mt-1">
              Uncalibrated receiver gain levels yield relative power measurements in dBm.
            </div>
          </div>

          <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg space-y-2">
            <div className="text-purple-400 font-bold uppercase tracking-wider flex items-center gap-1.5">
              <Database className="w-4 h-4 text-purple-400" />
              Historical Dataset Provenance
            </div>
            <div className="text-slate-300">Source: <span className="font-semibold text-white">Historical Dataset</span></div>
            <div className="text-slate-300">Dataset Size: <span className="font-semibold text-white">164,160 rows</span></div>
            <div className="text-slate-300">Frequency Range: <span className="font-semibold text-white">70 – 160 MHz</span></div>
            <div className="text-slate-300">Provenance Tag: <span className="font-bold text-purple-300">DATASET</span></div>
            <div className="text-[10px] text-slate-500 italic mt-1">
              Provides verified historical observations without live hardware connection requirements.
            </div>
          </div>

          <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg space-y-2">
            <div className="text-emerald-400 font-bold uppercase tracking-wider flex items-center gap-1.5">
              <Cpu className="w-4 h-4 text-emerald-400" />
              Provenance Category Taxonomy
            </div>
            <div className="text-[11px] text-slate-400 space-y-1">
              <div><strong className="text-cyan-300">REAL_MEASURED:</strong> Raw SDR hardware observation</div>
              <div><strong className="text-purple-300">IQ_REPLAY:</strong> Stored IQ binary capture playback</div>
              <div><strong className="text-amber-300">DERIVED:</strong> FFT PSD, SNR, noise, channel score</div>
              <div><strong className="text-emerald-300">INFERRED_RF_ACTIVITY:</strong> Activity & Availability decision</div>
            </div>
          </div>
        </div>
      </Panel>

      {/* Section 14: Source Capability Matrix */}
      <Panel title="RF Source Capability Matrix (Section 14)">
        <div className="overflow-x-auto rounded-lg border border-slate-800">
          <table className="w-full text-xs font-mono text-left text-slate-200">
            <thead className="bg-slate-950 text-slate-400 uppercase text-[10px] border-b border-slate-800">
              <tr>
                <th className="px-4 py-3">Capability</th>
                <th className="px-4 py-3 text-center">Dataset Source</th>
                <th className="px-4 py-3 text-center">Simulation Source</th>
                <th className="px-4 py-3 text-center">IQ Replay Source</th>
                <th className="px-4 py-3 text-center">RTL-SDR Source</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 bg-slate-900/50">
              <tr>
                <td className="px-4 py-3 font-semibold text-slate-300">RF Observations</td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-semibold text-slate-300">Raw I/Q Samples</td>
                <td className="px-4 py-3 text-center text-slate-500">Based on file</td>
                <td className="px-4 py-3 text-center text-emerald-400">Generated sim</td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-semibold text-slate-300">DSP / FFT / PSD Processing</td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-semibold text-slate-300">RF Activity Detection</td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-semibold text-slate-300">ML Evidence & OOD Guard</td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-semibold text-slate-300">RF Event Detection & Persistence</td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-semibold text-slate-300">Channel Allocation & Scoring</td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
                <td className="px-4 py-3 text-center text-emerald-400"><Check className="w-4 h-4 mx-auto" /></td>
              </tr>
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
