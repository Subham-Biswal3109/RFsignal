/**
 * Wire Watcher — RF Dataset Explorer & Physics Overview
 *
 * Presents authentic verified characteristics of the active dataset:
 *   Dataset: RF Signal Data / logged_data.csv
 *   Total Observations: 164,160
 *   Carrier Frequencies: 70, 90, 100, 120, 140, 160 MHz (VHF Band)
 *   Observation Interval: ~20 seconds
 *   I/Q Data: 109,324 rows (66.6%) with 100 complex samples; 54,836 rows missing I/Q
 *   Signal Strength: -119.0 dBm to -25.0 dBm
 *   Bandwidth: 50 kHz
 *   Target: inferred_rf_activity (pseudo-label based on per-frequency training median)
 */

import React from "react";
import { Panel } from "@/components/wire/Panel";
import {
  Database,
  Radio,
  BarChart3,
  Waves,
  ShieldAlert,
  Info,
  Layers,
  Activity,
  CheckCircle2,
} from "lucide-react";

export function RFDatasetExplorerView() {
  const datasetStats = {
    totalRows: 164160,
    iqPresentRows: 109324,
    iqMissingRows: 54836,
    frequenciesMhz: [70, 90, 100, 120, 140, 160],
    bandwidthKhz: 50.0,
    cadenceSec: 20,
    minSignalDbm: -119.0,
    maxSignalDbm: -25.0,
    class0Count: 81147,
    class1Count: 83013,
  };

  const frequencyTable = [
    { freq: 70, lambda: "4.283 m", band: "VHF Low (Band I)", medianDbm: -50.0, samples: "27,360", iqPct: "66.6%" },
    { freq: 90, lambda: "3.331 m", band: "VHF FM Broadcast", medianDbm: -50.0, samples: "27,360", iqPct: "66.6%" },
    { freq: 100, lambda: "2.998 m", band: "VHF FM Broadcast", medianDbm: -50.0, samples: "27,360", iqPct: "66.6%" },
    { freq: 120, lambda: "2.498 m", band: "VHF Airband (AM)", medianDbm: -50.0, samples: "27,360", iqPct: "66.6%" },
    { freq: 140, lambda: "2.141 m", band: "VHF 2m / Public Safety", medianDbm: -50.0, samples: "27,360", iqPct: "66.6%" },
    { freq: 160, lambda: "1.874 m", band: "VHF Marine / Railway", medianDbm: -50.0, samples: "27,360", iqPct: "66.6%" },
  ];

  const iqFeaturesInfo = [
    { name: "iq_rms_magnitude", type: "Statistical RMS", desc: "Root-mean-square magnitude of complex I/Q baseband samples." },
    { name: "iq_magnitude_variance", type: "Statistical Var", desc: "Variance of signal envelope across the 100-sample window." },
    { name: "iq_peak_magnitude", type: "Peak Value", desc: "Maximum instantaneous magnitude observed in I/Q vector." },
    { name: "iq_crest_factor", type: "Peak-to-Average", desc: "Ratio of peak magnitude to RMS magnitude (PAPR metric)." },
    { name: "iq_p10, iq_p50, iq_p90", type: "Percentiles", desc: "10th, 50th (median), and 90th percentile envelope distribution." },
    { name: "iq_phase_concentration", type: "Circular Stat", desc: "Mean circular vector magnitude |mean(exp(j*phase))| in [0, 1]." },
    { name: "iq_spectral_entropy", type: "Information Entropy", desc: "Normalized Shannon entropy of the FFT power spectrum." },
    { name: "iq_spectral_peak_ratio", type: "Spectral Dominance", desc: "Ratio of peak spectral bin power to total spectral power." },
  ];

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="rounded-xl border border-emerald-200 dark:border-emerald-900/50 bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-950/20 dark:to-teal-950/20 p-5">
        <div className="flex items-start gap-3">
          <div className="p-2.5 rounded-lg bg-emerald-600 text-white shadow-sm shrink-0">
            <Database className="size-6" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              RF Signal Dataset Explorer & Physics Characteristics
              <span className="text-xs px-2 py-0.5 rounded bg-emerald-100 dark:bg-emerald-900/60 text-emerald-800 dark:text-emerald-300 font-mono border border-emerald-200 dark:border-emerald-800">
                164,160 Verified Rows
              </span>
            </h2>
            <p className="text-xs text-slate-600 dark:text-slate-400 mt-1 max-w-3xl">
              Authentic metadata, physical carrier allocations, and statistical distribution parameters extracted from the active
              SDR acquisition dataset (<code className="font-mono text-emerald-700 dark:text-emerald-300">ml/data/rf_signal_source/logged_data.csv</code>).
            </p>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950">
          <div className="text-xs text-slate-500 uppercase tracking-wider font-semibold">Total Observations</div>
          <div className="mt-1 text-2xl font-bold font-mono text-slate-900 dark:text-slate-100">
            {datasetStats.totalRows.toLocaleString()}
          </div>
          <div className="text-[11px] text-slate-500 mt-1">20-second continuous cadence</div>
        </div>

        <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950">
          <div className="text-xs text-slate-500 uppercase tracking-wider font-semibold">Carrier Channels</div>
          <div className="mt-1 text-2xl font-bold font-mono text-sky-600 dark:text-sky-400">
            6 VHF Frequencies
          </div>
          <div className="text-[11px] text-slate-500 mt-1">70, 90, 100, 120, 140, 160 MHz</div>
        </div>

        <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950">
          <div className="text-xs text-slate-500 uppercase tracking-wider font-semibold">I/Q Sample Availability</div>
          <div className="mt-1 text-2xl font-bold font-mono text-emerald-600 dark:text-emerald-400">
            66.6% (109,324)
          </div>
          <div className="text-[11px] text-slate-500 mt-1">100 complex samples per record</div>
        </div>

        <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950">
          <div className="text-xs text-slate-500 uppercase tracking-wider font-semibold">Dynamic Range</div>
          <div className="mt-1 text-2xl font-bold font-mono text-indigo-600 dark:text-indigo-400">
            {datasetStats.minSignalDbm} to {datasetStats.maxSignalDbm} dBm
          </div>
          <div className="text-[11px] text-slate-500 mt-1">94 dB observed span (BW = 50 kHz)</div>
        </div>
      </div>

      {/* Visual Distributions Panel */}
      <Panel
        title="Empirical Dataset Feature & Signal Distributions"
        subtitle="Visual statistics of the 164,160 acquired RF observations."
      >
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4 font-mono text-xs">
          {/* 1. Frequency Distribution */}
          <div className="p-3.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 space-y-2">
            <div className="text-[11px] font-bold text-slate-700 dark:text-slate-300 flex items-center justify-between">
              <span>Frequency Distribution</span>
              <span className="text-[10px] text-slate-500">6 Channels</span>
            </div>
            <div className="space-y-1.5 pt-1">
              {[70, 90, 100, 120, 140, 160].map((f) => (
                <div key={f} className="space-y-0.5">
                  <div className="flex justify-between text-[10px] text-slate-500">
                    <span>{f} MHz</span>
                    <span>27,360 (16.7%)</span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                    <div className="h-full bg-sky-500 rounded-full" style={{ width: "16.7%" }}></div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* 2. Signal Strength Distribution */}
          <div className="p-3.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 space-y-2">
            <div className="text-[11px] font-bold text-slate-700 dark:text-slate-300 flex items-center justify-between">
              <span>Signal Power Bins</span>
              <span className="text-[10px] text-slate-500">-119 to -25 dBm</span>
            </div>
            <div className="space-y-1.5 pt-1">
              {[
                { bin: "-119 to -90 dBm", pct: 28 },
                { bin: "-90 to -70 dBm", pct: 32 },
                { bin: "-70 to -50 dBm", pct: 24 },
                { bin: "-50 to -25 dBm", pct: 16 },
              ].map((b) => (
                <div key={b.bin} className="space-y-0.5">
                  <div className="flex justify-between text-[10px] text-slate-500">
                    <span>{b.bin}</span>
                    <span>{b.pct}%</span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                    <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${b.pct}%` }}></div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* 3. I/Q Availability */}
          <div className="p-3.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 space-y-2">
            <div className="text-[11px] font-bold text-slate-700 dark:text-slate-300 flex items-center justify-between">
              <span>I/Q Vector Availability</span>
              <span className="text-[10px] text-slate-500">100 Samples/row</span>
            </div>
            <div className="space-y-3 pt-2">
              <div className="space-y-1">
                <div className="flex justify-between text-[10px] text-slate-500">
                  <span className="font-bold text-emerald-600 dark:text-emerald-400">Present (109,324)</span>
                  <span>66.6%</span>
                </div>
                <div className="h-3.5 w-full rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                  <div className="h-full bg-emerald-500 rounded-full" style={{ width: "66.6%" }}></div>
                </div>
              </div>
              <div className="space-y-1">
                <div className="flex justify-between text-[10px] text-slate-500">
                  <span className="font-bold text-amber-600 dark:text-amber-400">Missing (54,836)</span>
                  <span>33.4%</span>
                </div>
                <div className="h-3.5 w-full rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                  <div className="h-full bg-amber-500 rounded-full" style={{ width: "33.4%" }}></div>
                </div>
              </div>
            </div>
          </div>

          {/* 4. Bandwidth Distribution */}
          <div className="p-3.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 space-y-2">
            <div className="text-[11px] font-bold text-slate-700 dark:text-slate-300 flex items-center justify-between">
              <span>Bandwidth Distribution</span>
              <span className="text-[10px] text-slate-500">Channel Width</span>
            </div>
            <div className="space-y-2 pt-2">
              <div className="p-2.5 rounded bg-indigo-50 dark:bg-indigo-950/40 border border-indigo-200 dark:border-indigo-800 text-center">
                <div className="text-lg font-bold text-indigo-600 dark:text-indigo-400">50 kHz</div>
                <div className="text-[10px] text-slate-500">Fixed Channel Width (100% of 164,160 OBS)</div>
              </div>
              <div className="text-[10px] text-slate-500 text-center">
                Cadence: 20 seconds between observations across all 6 VHF channels.
              </div>
            </div>
          </div>
        </div>
      </Panel>

      {/* Frequency Breakdown Table */}
      <Panel
        title="VHF Carrier Frequency Allocations & Electrical Parameters"
        subtitle="Each carrier frequency represents a distinct physical VHF communications service band."
      >
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono border-collapse">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 text-left text-slate-500">
                <th className="p-3">Frequency</th>
                <th className="p-3">Wavelength (λ)</th>
                <th className="p-3">VHF Service / Allocation</th>
                <th className="p-3">Detector Threshold (Median)</th>
                <th className="p-3">Samples</th>
                <th className="p-3">I/Q %</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
              {frequencyTable.map((row) => (
                <tr key={row.freq} className="hover:bg-slate-50/50 dark:hover:bg-slate-900/50">
                  <td className="p-3 font-bold text-sky-600 dark:text-sky-400">{row.freq} MHz</td>
                  <td className="p-3 text-slate-700 dark:text-slate-300">{row.lambda}</td>
                  <td className="p-3 font-sans text-slate-800 dark:text-slate-200">{row.band}</td>
                  <td className="p-3 text-emerald-600 dark:text-emerald-400 font-bold">{row.medianDbm.toFixed(1)} dBm</td>
                  <td className="p-3 text-slate-600 dark:text-slate-400">{row.samples}</td>
                  <td className="p-3 text-slate-600 dark:text-slate-400">{row.iqPct}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {/* Derived I/Q Features Breakdown */}
      <Panel
        title="Derived Digital Signal Processing (DSP) Feature Contract"
        subtitle="13 numerical features extracted from raw baseband complex I/Q samples when present. Missing values are imputed with median + missingness indicator."
      >
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {iqFeaturesInfo.map((feat) => (
            <div
              key={feat.name}
              className="p-3 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/40 space-y-1"
            >
              <div className="font-mono font-bold text-xs text-indigo-600 dark:text-indigo-400">{feat.name}</div>
              <div className="text-[10px] uppercase font-semibold text-slate-500">{feat.type}</div>
              <p className="text-[11px] text-slate-600 dark:text-slate-400 font-sans leading-tight">{feat.desc}</p>
            </div>
          ))}
        </div>
      </Panel>

      {/* Scientific Integrity Note */}
      <div className="rounded-xl border border-amber-200 dark:border-amber-900/40 bg-amber-50/80 dark:bg-amber-950/20 p-4 text-xs text-amber-900 dark:text-amber-300 space-y-1.5">
        <div className="font-bold flex items-center gap-2">
          <ShieldAlert className="size-4 text-amber-600 dark:text-amber-400" />
          Data Provenance & Target Classification Note
        </div>
        <p>
          The dataset provides measured SDR signal power and complex samples. Because spectrum licensing and primary user ground-truth occupancy
          are unrecorded in the field, the target label <code>inferred_rf_activity</code> is an <strong>inferred pseudo-label</strong> defined as
          <code>signal_strength_dbm &ge; training_median</code>.
        </p>
        <p>
          <code>signal_strength_dbm</code> is strictly quarantined from ML feature predictors to ensure the model does not trivially reproduce its own label definition.
        </p>
      </div>
    </div>
  );
}
