/**
 * Wire Watcher — Time-Domain I/Q Waveform Visualizer (Sections 2–7 & 11)
 *
 * Displays raw baseband complex I/Q samples in the time domain: I(t), Q(t), Magnitude |x(t)|.
 * Provides derived I/Q quality metrics (RMS, Peak, Crest Factor), window controls,
 * and an explicit UNAVAILABLE fallback when I/Q samples do not exist in the observation.
 */

import React, { useState, useEffect, useMemo } from "react";
import { Panel } from "@/components/wire/Panel";
import { Activity, Sliders, AlertCircle, RefreshCw, ZoomIn, ZoomOut, RotateCcw, Info } from "lucide-react";
import type { WaveformResponse } from "@/types/wire-watcher";
import { cn } from "@/lib/utils";
import { apiFetch } from "@/lib/api";

interface Props {
  centerFreqMhz?: number;
  bandwidthMhz?: number;
  signalPowerDbm?: number;
  noiseFloorDbm?: number;
  iqAvailable?: number;
}

export function TimeDomainWaveformView({
  centerFreqMhz = 120.0,
  bandwidthMhz = 0.2,
  signalPowerDbm = -75.0,
  noiseFloorDbm = -100.0,
  iqAvailable = 1,
}: Props) {
  const [channelMode, setChannelMode] = useState<"I" | "Q" | "IQ" | "MAG">("IQ");
  const [sampleWindow, setSampleWindow] = useState<number>(512);
  const [data, setData] = useState<WaveformResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchWaveform = async () => {
    setLoading(true);
    setError(null);
    try {
      const json = await apiFetch<WaveformResponse>(
        `/api/rf/waveform?frequency_mhz=${centerFreqMhz}&bandwidth_mhz=${bandwidthMhz}&signal_power_dbm=${signalPowerDbm}&noise_floor_dbm=${noiseFloorDbm}&iq_available=${iqAvailable}`
      );
      setData(json);
    } catch (err: any) {
      setError(err.message || "Waveform capture failed");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchWaveform();
  }, [centerFreqMhz, bandwidthMhz, signalPowerDbm, noiseFloorDbm, iqAvailable]);

  // Windowed points for SVG plot
  const windowedPoints = useMemo(() => {
    if (!data?.waveform_series) return null;
    const series = data.waveform_series;
    const limit = Math.min(sampleWindow, series.time_us.length);
    return {
      time: series.time_us.slice(0, limit),
      i: series.i.slice(0, limit),
      q: series.q.slice(0, limit),
      magnitude: series.magnitude.slice(0, limit),
    };
  }, [data, sampleWindow]);

  // SVG Polyline paths calculation
  const svgPaths = useMemo(() => {
    if (!windowedPoints || windowedPoints.time.length < 2) return { pathI: "", pathQ: "", pathMag: "" };

    const width = 800;
    const height = 220;
    const padding = 20;

    const times = windowedPoints.time;
    const tMin = times[0];
    const tMax = times[times.length - 1] || 1;

    // Find amplitude scaling
    let maxAmp = 0.001;
    for (let idx = 0; idx < times.length; idx++) {
      maxAmp = Math.max(
        maxAmp,
        Math.abs(windowedPoints.i[idx]),
        Math.abs(windowedPoints.q[idx]),
        windowedPoints.magnitude[idx]
      );
    }
    maxAmp = maxAmp * 1.15; // padding factor

    const mapX = (t: number) => padding + ((t - tMin) / (tMax - tMin)) * (width - 2 * padding);
    const mapY = (val: number) => (height / 2) - (val / maxAmp) * ((height - 2 * padding) / 2);

    let pathI = "";
    let pathQ = "";
    let pathMag = "";

    for (let k = 0; k < times.length; k++) {
      const x = mapX(times[k]).toFixed(1);
      const yI = mapY(windowedPoints.i[k]).toFixed(1);
      const yQ = mapY(windowedPoints.q[k]).toFixed(1);
      const yMag = mapY(windowedPoints.magnitude[k]).toFixed(1);

      if (k === 0) {
        pathI = `M ${x} ${yI}`;
        pathQ = `M ${x} ${yQ}`;
        pathMag = `M ${x} ${yMag}`;
      } else {
        pathI += ` L ${x} ${yI}`;
        pathQ += ` L ${x} ${yQ}`;
        pathMag += ` L ${x} ${yMag}`;
      }
    }

    return { pathI, pathQ, pathMag };
  }, [windowedPoints]);

  return (
    <div className="space-y-6">
      <Panel
        title="TIME-DOMAIN I/Q WAVEFORM"
        subtitle="Baseband complex samples x[n] = I[n] + j*Q[n] captured from active RF observation."
      >
        {/* State 1: Fallback when I/Q is unavailable */}
        {data?.iq_available === 0 || iqAvailable === 0 ? (
          <div className="p-8 bg-slate-950/90 border border-slate-800 rounded-xl text-center space-y-3 font-mono">
            <div className="inline-flex items-center justify-center p-3 rounded-full bg-amber-950/80 text-amber-400 border border-amber-800">
              <AlertCircle className="w-6 h-6" />
            </div>
            <div className="text-base font-bold text-amber-300">TIME-DOMAIN WAVEFORM UNAVAILABLE</div>
            <div className="text-xs text-slate-400 max-w-md mx-auto">
              Reason: {data?.reason || "This RF observation does not contain I/Q samples."}
            </div>
            <div className="text-[11px] text-slate-500 italic max-w-lg mx-auto border-t border-slate-900 pt-3">
              Scientific Policy: Zero fake I/Q arrays are generated from signal power when complex baseband samples are not captured by the source.
            </div>
          </div>
        ) : (
          <div className="space-y-4 font-mono">
            {/* Visualizer Header Toolbar */}
            <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-slate-950 border border-slate-800 rounded-lg text-xs">
              {/* Channel Selectors */}
              <div className="flex items-center gap-1.5">
                <span className="text-slate-400 text-[11px] mr-1">Channel:</span>
                {[
                  { id: "IQ", label: "I + Q" },
                  { id: "I", label: "I(t) Only" },
                  { id: "Q", label: "Q(t) Only" },
                  { id: "MAG", label: "Magnitude |x(t)|" },
                ].map((mode) => (
                  <button
                    key={mode.id}
                    onClick={() => setChannelMode(mode.id as any)}
                    className={cn(
                      "px-2.5 py-1 rounded text-xs font-bold transition",
                      channelMode === mode.id
                        ? "bg-cyan-600 text-white shadow"
                        : "bg-slate-900 text-slate-400 hover:text-slate-200 hover:bg-slate-800"
                    )}
                  >
                    {mode.label}
                  </button>
                ))}
              </div>

              {/* Sample Window Selection */}
              <div className="flex items-center gap-1.5">
                <span className="text-slate-400 text-[11px] mr-1">Sample Window:</span>
                {[256, 512, 1024, 2048].map((win) => (
                  <button
                    key={win}
                    onClick={() => setSampleWindow(win)}
                    className={cn(
                      "px-2 py-0.5 rounded text-xs font-mono transition",
                      sampleWindow === win
                        ? "bg-slate-800 text-cyan-300 font-bold border border-cyan-500/50"
                        : "bg-slate-900 text-slate-400 hover:text-slate-200"
                    )}
                  >
                    {win}
                  </button>
                ))}

                <button
                  onClick={fetchWaveform}
                  disabled={loading}
                  className="ml-2 p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded transition"
                  title="Refresh Waveform"
                >
                  <RefreshCw className={cn("w-3.5 h-3.5", loading && "animate-spin")} />
                </button>
              </div>
            </div>

            {/* Time-Domain SVG Waveform Plot */}
            <div className="relative bg-slate-950 p-4 rounded-xl border border-slate-800 shadow-inner overflow-hidden">
              {/* Legend & Telemetry Overlay */}
              <div className="flex items-center justify-between text-[11px] mb-2 px-1">
                <div className="flex items-center gap-4">
                  {(channelMode === "I" || channelMode === "IQ") && (
                    <span className="flex items-center gap-1.5 text-cyan-400 font-bold">
                      <span className="w-3 h-0.5 bg-cyan-400 rounded-full" /> I(t) In-Phase
                    </span>
                  )}
                  {(channelMode === "Q" || channelMode === "IQ") && (
                    <span className="flex items-center gap-1.5 text-amber-400 font-bold">
                      <span className="w-3 h-0.5 bg-amber-400 rounded-full" /> Q(t) Quadrature
                    </span>
                  )}
                  {channelMode === "MAG" && (
                    <span className="flex items-center gap-1.5 text-emerald-400 font-bold">
                      <span className="w-3 h-0.5 bg-emerald-400 rounded-full" /> |x(t)| Magnitude
                    </span>
                  )}
                </div>

                <div className="text-slate-500 text-[10px]">
                  Showing {windowedPoints?.time.length || 0} samples ({data?.metrics?.duration_us} µs)
                </div>
              </div>

              {/* SVG Canvas */}
              <div className="w-full overflow-x-auto">
                <svg viewBox="0 0 800 220" className="w-full h-56 bg-slate-900/60 rounded-lg border border-slate-800/80">
                  {/* Grid Lines */}
                  <line x1="20" y1="110" x2="780" y2="110" stroke="#334155" strokeWidth="1" strokeDasharray="4 4" />
                  <line x1="20" y1="55" x2="780" y2="55" stroke="#1e293b" strokeWidth="1" strokeDasharray="2 2" />
                  <line x1="20" y1="165" x2="780" y2="165" stroke="#1e293b" strokeWidth="1" strokeDasharray="2 2" />

                  {/* Polylines */}
                  {(channelMode === "I" || channelMode === "IQ") && (
                    <path d={svgPaths.pathI} fill="none" stroke="#22d3ee" strokeWidth="1.8" strokeLinecap="round" />
                  )}
                  {(channelMode === "Q" || channelMode === "IQ") && (
                    <path d={svgPaths.pathQ} fill="none" stroke="#fbbf24" strokeWidth="1.8" strokeLinecap="round" />
                  )}
                  {channelMode === "MAG" && (
                    <path d={svgPaths.pathMag} fill="none" stroke="#34d399" strokeWidth="2.0" strokeLinecap="round" />
                  )}
                </svg>
              </div>

              <div className="flex justify-between text-[10px] text-slate-500 mt-2 font-mono px-1">
                <span>0 µs</span>
                <span>Time Axis (t) →</span>
                <span>{data?.metrics?.duration_us} µs</span>
              </div>
            </div>

            {/* Derived I/Q Quality Panel (Section 6 & 7) */}
            {data?.metrics && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2">
                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
                  <div className="text-[10px] text-slate-500 uppercase font-bold">Samples & Rate</div>
                  <div className="text-sm font-bold text-slate-200 mt-0.5">
                    {data.metrics.num_samples} samples @ {data.metrics.sample_rate_mhz} MS/s
                  </div>
                </div>

                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
                  <div className="text-[10px] text-slate-500 uppercase font-bold">Component RMS</div>
                  <div className="text-sm font-bold text-cyan-300 mt-0.5">
                    I: {data.metrics.i_rms} | Q: {data.metrics.q_rms}
                  </div>
                </div>

                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
                  <div className="text-[10px] text-slate-500 uppercase font-bold">Magnitude RMS & Peak</div>
                  <div className="text-sm font-bold text-emerald-300 mt-0.5">
                    RMS: {data.metrics.magnitude_rms} | Peak: {data.metrics.peak_magnitude}
                  </div>
                </div>

                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
                  <div className="text-[10px] text-slate-500 uppercase font-bold">Crest Factor</div>
                  <div className="text-sm font-bold text-amber-300 mt-0.5">
                    {data.metrics.crest_factor_linear} ({data.metrics.crest_factor_db} dB)
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </Panel>
    </div>
  );
}
