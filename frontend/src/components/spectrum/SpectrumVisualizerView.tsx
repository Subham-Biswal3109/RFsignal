/**
 * Wire Watcher — Spectrum Visualizer & DSP Peak Detection View
 *
 * Demonstrates the Signal Processing & Spectrum Sensing Pipeline:
 *   Simulated / Recorded Baseband Signal
 *       ↓
 *   Hanning Windowing & FFT Spectral Estimation
 *       ↓
 *   Relative Power Spectral Density (PSD)
 *       ↓
 *   Noise Floor Estimation (10th Percentile)
 *       ↓
 *   Adaptive Thresholding (Noise Floor + Δ dB)
 *       ↓
 *   Peak Detection & Occupied Region Extraction
 */

import React, { useState, useEffect, useCallback } from "react";
import { Panel } from "@/components/wire/Panel";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import {
  Activity,
  Radio,
  Sliders,
  Sparkles,
  TrendingUp,
  Layers,
  ArrowRight,
  Info,
  RefreshCw,
  Loader2,
} from "lucide-react";
import type { AnalyzeSpectrumResponse, PredictRequest } from "@/types/wire-watcher";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:5000";

interface SpectrumVisualizerViewProps {
  onSendToAnalyzer?: (req: PredictRequest) => void;
}

export function SpectrumVisualizerView({ onSendToAnalyzer }: SpectrumVisualizerViewProps) {
  const [centerFreqMhz, setCenterFreqMhz] = useState<number>(120.0);
  const [bandwidthMhz, setBandwidthMhz] = useState<number>(10.0);
  const [signalPowerDbm, setSignalPowerDbm] = useState<number>(-60.0);
  const [noiseFloorDbmInput, setNoiseFloorDbmInput] = useState<number>(-100.0);
  const [snrThresholdDeltaDb, setSnrThresholdDeltaDb] = useState<number>(6.0);

  const [loading, setLoading] = useState<boolean>(false);
  const [spectrumData, setSpectrumData] = useState<AnalyzeSpectrumResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runAnalysis = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`${API_BASE}/api/spectrum/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          center_freq_mhz: centerFreqMhz,
          bandwidth_mhz: bandwidthMhz,
          signal_strength_dbm: signalPowerDbm,
          noise_floor_dbm: noiseFloorDbmInput,
        }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data: AnalyzeSpectrumResponse = await resp.json();
      setSpectrumData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to compute spectrum");
    } finally {
      setLoading(false);
    }
  }, [centerFreqMhz, bandwidthMhz, signalPowerDbm, noiseFloorDbmInput]);

  useEffect(() => {
    runAnalysis();
  }, [runAnalysis]);

  // SVG Plot Scaling Helpers
  const freqs = spectrumData?.spectrum_data?.frequencies ?? [];
  const powers = spectrumData?.spectrum_data?.power_dbm ?? [];
  const noiseFloor = spectrumData?.noise_floor_dbm ?? -100;
  const detectionThreshold = noiseFloor + snrThresholdDeltaDb;

  const minFreq = freqs.length > 0 ? Math.min(...freqs) : centerFreqMhz - bandwidthMhz / 2;
  const maxFreq = freqs.length > 0 ? Math.max(...freqs) : centerFreqMhz + bandwidthMhz / 2;
  const minPower = -120;
  const maxPower = -20;

  const svgWidth = 700;
  const svgHeight = 240;
  const padding = { top: 20, right: 30, bottom: 35, left: 55 };

  const plotWidth = svgWidth - padding.left - padding.right;
  const plotHeight = svgHeight - padding.top - padding.bottom;

  const getX = (f: number) => {
    if (maxFreq === minFreq) return padding.left;
    return padding.left + ((f - minFreq) / (maxFreq - minFreq)) * plotWidth;
  };

  const getY = (p: number) => {
    const clamped = Math.max(minPower, Math.min(maxPower, p));
    return padding.top + (1 - (clamped - minPower) / (maxPower - minPower)) * plotHeight;
  };

  // Build SVG Path for PSD curve
  let psdPathD = "";
  if (freqs.length > 0 && powers.length > 0) {
    psdPathD = freqs.reduce((acc, f, idx) => {
      const x = getX(f);
      const y = getY(powers[idx]);
      return idx === 0 ? `M ${x} ${y}` : `${acc} L ${x} ${y}`;
    }, "");
  }

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="rounded-xl border border-indigo-200 dark:border-indigo-900/50 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-950/20 dark:to-purple-950/20 p-5">
        <div className="flex items-start gap-3">
          <div className="p-2.5 rounded-lg bg-indigo-600 text-white shadow-sm shrink-0">
            <Activity className="size-6" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              Spectrum Analyzer & DSP Peak Detection Visualizer
              <span className="text-xs px-2 py-0.5 rounded bg-indigo-100 dark:bg-indigo-900/60 text-indigo-800 dark:text-indigo-300 font-mono border border-indigo-200 dark:border-indigo-800">
                Hanning FFT PSD
              </span>
            </h2>
            <p className="text-xs text-slate-600 dark:text-slate-400 mt-1 max-w-3xl">
              Demonstrates digital signal processing (DSP) spectral estimation, noise floor evaluation, adaptive peak detection,
              and channel occupancy region segmentation.
            </p>
          </div>
        </div>
      </div>

      {/* Main Grid: Controls + Visualizer */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Controls Column */}
        <div className="space-y-4 lg:col-span-1">
          <Panel
            title="Sensing & Signal Parameters"
            subtitle="Simulated RF channel environment parameters passed to the baseband spectral synthesizer."
          >
            <div className="space-y-4">
              <div>
                <Label>Center Frequency (f_0)</Label>
                <div className="flex gap-2 mt-1">
                  <Input
                    type="number"
                    step="any"
                    value={centerFreqMhz}
                    onChange={(e) => setCenterFreqMhz(parseFloat(e.target.value) || 0)}
                  />
                  <span className="inline-flex items-center px-3 rounded bg-slate-100 dark:bg-slate-800 text-xs font-mono">
                    MHz
                  </span>
                </div>
              </div>

              <div>
                <Label>Channel Span / Bandwidth</Label>
                <div className="flex gap-2 mt-1">
                  <Input
                    type="number"
                    step="any"
                    value={bandwidthMhz}
                    onChange={(e) => setBandwidthMhz(parseFloat(e.target.value) || 0.1)}
                  />
                  <span className="inline-flex items-center px-3 rounded bg-slate-100 dark:bg-slate-800 text-xs font-mono">
                    MHz
                  </span>
                </div>
              </div>

              <div>
                <Label>Simulated Signal Power</Label>
                <div className="flex gap-2 mt-1">
                  <Input
                    type="number"
                    step="any"
                    value={signalPowerDbm}
                    onChange={(e) => setSignalPowerDbm(parseFloat(e.target.value) || -100)}
                  />
                  <span className="inline-flex items-center px-3 rounded bg-slate-100 dark:bg-slate-800 text-xs font-mono">
                    dBm
                  </span>
                </div>
              </div>

              <div>
                <Label>Simulated Noise Floor</Label>
                <div className="flex gap-2 mt-1">
                  <Input
                    type="number"
                    step="any"
                    value={noiseFloorDbmInput}
                    onChange={(e) => setNoiseFloorDbmInput(parseFloat(e.target.value) || -120)}
                  />
                  <span className="inline-flex items-center px-3 rounded bg-slate-100 dark:bg-slate-800 text-xs font-mono">
                    dBm
                  </span>
                </div>
              </div>

              <div>
                <Label>Peak Threshold Margin (Δ dB above Noise Floor)</Label>
                <div className="flex gap-2 mt-1">
                  <Input
                    type="number"
                    step="0.5"
                    value={snrThresholdDeltaDb}
                    onChange={(e) => setSnrThresholdDeltaDb(parseFloat(e.target.value) || 0)}
                  />
                  <span className="inline-flex items-center px-3 rounded bg-slate-100 dark:bg-slate-800 text-xs font-mono">
                    dB
                  </span>
                </div>
                <p className="text-[10px] text-slate-500 mt-1">
                  Threshold = Noise Floor + Δ dB = {(noiseFloor + snrThresholdDeltaDb).toFixed(1)} dBm
                </p>
              </div>

              <div className="pt-2 flex gap-2">
                <Button onClick={runAnalysis} disabled={loading} className="w-full gap-1.5">
                  {loading ? <Loader2 className="size-4 animate-spin" /> : <RefreshCw className="size-4" />}
                  Compute Spectrum FFT
                </Button>
              </div>

              {/* Dataset Presets */}
              <div className="pt-2 border-t border-slate-200 dark:border-slate-800">
                <span className="text-[11px] font-semibold text-slate-600 dark:text-slate-400">
                  Quick VHF Dataset Presets:
                </span>
                <div className="flex flex-wrap gap-1 mt-1.5">
                  {[70, 90, 100, 120, 140, 160].map((f) => (
                    <button
                      key={f}
                      onClick={() => {
                        setCenterFreqMhz(f);
                        setBandwidthMhz(10);
                        setSignalPowerDbm(-60);
                      }}
                      className="px-2 py-0.5 rounded text-[11px] bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300 font-mono border border-slate-300 dark:border-slate-700"
                    >
                      {f} MHz
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </Panel>
        </div>

        {/* Visualizer & Results Column */}
        <div className="space-y-4 lg:col-span-2">
          <Panel
            title="RF Spectrum Analyzer & Relative PSD Visualizer"
            subtitle="Hanning-windowed FFT spectral estimation. Displays Relative PSD / Normalized Spectral Power. Solid cyan: Relative PSD | Dashed orange: Noise Floor | Dotted yellow: Detection Threshold."
          >
            {error ? (
              <div className="p-4 rounded-lg bg-red-50 text-red-800 border border-red-200 text-xs">
                {error}
              </div>
            ) : (
              <div className="space-y-4">
                {/* SVG Spectrum Scope */}
                <div className="rounded-xl border border-slate-800 bg-slate-950 p-3 shadow-inner overflow-hidden">
                  <div className="flex items-center justify-between text-[11px] font-mono text-emerald-400 pb-2 border-b border-slate-800 px-1">
                    <span>SPAN: {bandwidthMhz.toFixed(2)} MHz</span>
                    <span>CENTER: {centerFreqMhz.toFixed(3)} MHz</span>
                    <span>EST NOISE: {noiseFloor.toFixed(1)} dB (Rel)</span>
                    <span>THR: {detectionThreshold.toFixed(1)} dB (Rel)</span>
                  </div>

                  <div className="relative w-full overflow-x-auto">
                    <svg
                      viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                      className="w-full h-auto max-h-[300px] select-none"
                    >
                      {/* Axis Label */}
                      <text
                        x={padding.left + 5}
                        y={padding.top + 10}
                        fill="#06b6d4"
                        fontSize="9"
                        fontFamily="monospace"
                        fontWeight="bold"
                      >
                        Normalized Spectral Power (Relative PSD dB)
                      </text>

                      {/* Grid Lines */}
                      {[-100, -80, -60, -40].map((pLevel) => (
                        <g key={pLevel}>
                          <line
                            x1={padding.left}
                            y1={getY(pLevel)}
                            x2={svgWidth - padding.right}
                            y2={getY(pLevel)}
                            stroke="#1e293b"
                            strokeDasharray="3,3"
                            strokeWidth="1"
                          />
                          <text
                            x={padding.left - 8}
                            y={getY(pLevel) + 3}
                            fill="#64748b"
                            fontSize="9"
                            fontFamily="monospace"
                            textAnchor="end"
                          >
                            {pLevel} dB
                          </text>
                        </g>
                      ))}

                      {/* Frequency Grid Lines */}
                      {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
                        const fVal = minFreq + ratio * (maxFreq - minFreq);
                        const x = getX(fVal);
                        return (
                          <g key={ratio}>
                            <line
                              x1={x}
                              y1={padding.top}
                              x2={x}
                              y2={svgHeight - padding.bottom}
                              stroke="#1e293b"
                              strokeDasharray="3,3"
                              strokeWidth="1"
                            />
                            <text
                              x={x}
                              y={svgHeight - padding.bottom + 15}
                              fill="#64748b"
                              fontSize="9"
                              fontFamily="monospace"
                              textAnchor="middle"
                            >
                              {fVal.toFixed(2)} MHz
                            </text>
                          </g>
                        );
                      })}

                      {/* Occupied Regions Shading */}
                      {(spectrumData?.occupied_regions ?? []).map((reg, idx) => {
                        const x1 = getX(reg.start_mhz);
                        const x2 = getX(reg.end_mhz);
                        const width = Math.max(2, x2 - x1);
                        return (
                          <rect
                            key={idx}
                            x={x1}
                            y={padding.top}
                            width={width}
                            height={plotHeight}
                            fill="rgba(239, 68, 68, 0.18)"
                            stroke="rgba(239, 68, 68, 0.5)"
                            strokeWidth="1"
                            strokeDasharray="2,2"
                          />
                        );
                      })}

                      {/* Noise Floor Line */}
                      <line
                        x1={padding.left}
                        y1={getY(noiseFloor)}
                        x2={svgWidth - padding.right}
                        y2={getY(noiseFloor)}
                        stroke="#f97316"
                        strokeDasharray="5,3"
                        strokeWidth="1.5"
                      />

                      {/* Detection Threshold Line */}
                      <line
                        x1={padding.left}
                        y1={getY(detectionThreshold)}
                        x2={svgWidth - padding.right}
                        y2={getY(detectionThreshold)}
                        stroke="#eab308"
                        strokeDasharray="2,2"
                        strokeWidth="1.5"
                      />

                      {/* PSD Curve */}
                      {psdPathD && (
                        <path
                          d={psdPathD}
                          fill="none"
                          stroke="#06b6d4"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      )}

                      {/* Detected Peaks Markers */}
                      {(spectrumData?.detected_signals ?? []).map((peak, idx) => {
                        const x = getX(peak.frequency_mhz);
                        const y = getY(peak.power_dbm);
                        return (
                          <g key={idx}>
                            <circle cx={x} cy={y} r="4" fill="#ef4444" stroke="#ffffff" strokeWidth="1.5" />
                            <text
                              x={x}
                              y={y - 8}
                              fill="#fca5a5"
                              fontSize="10"
                              fontFamily="monospace"
                              fontWeight="bold"
                              textAnchor="middle"
                            >
                              ▼ {peak.power_dbm.toFixed(1)} dBm
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                  </div>
                </div>

                {/* Detected Signals Table */}
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="p-3.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950">
                    <div className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300 mb-2 flex items-center justify-between">
                      <span>Detected Spectral Peaks ({spectrumData?.detected_signals?.length ?? 0})</span>
                      <span className="text-[10px] text-slate-500 font-mono">
                        SNR &gt; {snrThresholdDeltaDb} dB
                      </span>
                    </div>

                    {(spectrumData?.detected_signals ?? []).length === 0 ? (
                      <p className="text-xs text-slate-500 py-3 text-center">
                        No peaks detected above threshold ({detectionThreshold.toFixed(1)} dBm).
                      </p>
                    ) : (
                      <div className="space-y-2">
                        {(spectrumData?.detected_signals ?? []).map((pk, i) => (
                          <div
                            key={i}
                            className="p-2 rounded bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs font-mono flex items-center justify-between"
                          >
                            <div>
                              <span className="font-bold text-sky-600 dark:text-sky-400">
                                {pk.frequency_mhz.toFixed(3)} MHz
                              </span>
                              <span className="text-[10px] text-slate-500 block">
                                3-dB BW: {(pk.bandwidth_mhz * 1000).toFixed(1)} kHz
                              </span>
                            </div>
                            <div className="text-right">
                              <span className="font-bold text-red-600 dark:text-red-400">
                                {pk.power_dbm.toFixed(1)} dBm
                              </span>
                              <span className="text-[10px] text-emerald-600 dark:text-emerald-400 block font-semibold">
                                SNR: +{pk.snr_db.toFixed(1)} dB
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  <div className="p-3.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 space-y-3">
                    <div className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
                      DSP Pipeline Explanation
                    </div>
                    <div className="text-xs text-slate-600 dark:text-slate-400 space-y-1.5">
                      <p>
                        <strong>1. Windowing & FFT:</strong> Baseband time-domain samples are multiplied by a Hanning window to suppress spectral leakage before computing the FFT magnitude squared.
                      </p>
                      <p>
                        <strong>2. Noise Floor:</strong> Estimated robustly using the 10th percentile of the power spectrum ({noiseFloor.toFixed(1)} dBm).
                      </p>
                      <p>
                        <strong>3. Peak Detection:</strong> Identifies local maxima that exceed the adaptive threshold ({detectionThreshold.toFixed(1)} dBm).
                      </p>
                    </div>

                    {onSendToAnalyzer && spectrumData?.extracted_features && (
                      <Button
                        variant="secondary"
                        onClick={() => onSendToAnalyzer(spectrumData.extracted_features)}
                        className="w-full text-xs gap-1.5 mt-2 font-medium"
                      >
                        Transfer to Activity Detector <ArrowRight className="size-3.5" />
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}
