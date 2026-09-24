/**
 * Wire Watcher — Controlled RF Replay Mode View (Section 13)
 *
 * Provides controls for Play, Pause, Reset, and playback speed (0.5x, 1x, 2x, 5x)
 * through historical dataset observations.
 */

import React, { useState, useEffect } from "react";
import { Panel } from "@/components/wire/Panel";
import { Play, Pause, RotateCcw, FastForward, Film, AlertTriangle } from "lucide-react";
import { apiFetch } from "@/lib/api";
import type { RFReplayStatus } from "@/types/wire-watcher";

export function ReplayControllerView() {
  const [status, setStatus] = useState<RFReplayStatus | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const fetchStatus = async () => {
    try {
      const data = await apiFetch<RFReplayStatus>("/api/replay/status");
      setStatus(data);
    } catch (err) {
      console.error("Failed to fetch replay status", err);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  useEffect(() => {
    if (!status?.is_playing) return;
    const intervalMs = Math.max(200, 1000 / (status.playback_speed || 1.0));
    const timer = setInterval(() => {
      handleAction("step");
    }, intervalMs);
    return () => clearInterval(timer);
  }, [status?.is_playing, status?.playback_speed]);

  const handleAction = async (action: string, speed?: number) => {
    setLoading(true);
    try {
      const data = await apiFetch<RFReplayStatus>("/api/replay/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, speed }),
      });
      setStatus(data);
    } catch (err) {
      console.error("Replay action failed", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <Film className="w-5 h-5 text-purple-400" />
            Controlled RF Replay Mode
          </h2>
          <p className="text-xs text-slate-400">
            Chronological playback of historical RF observations from stored dataset / IQ files.
          </p>
        </div>

        {/* Warning Badge: Never Claim Live */}
        <div className="px-3 py-1.5 bg-amber-950/80 border border-amber-800 rounded-lg text-amber-300 text-xs font-mono font-semibold flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-amber-400" />
          <span>REPLAY MODE — NOT LIVE</span>
        </div>
      </div>

      {/* Control Panel */}
      <Panel title="Historical Playback Controls">
        <div className="space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4 bg-slate-950 p-4 rounded-xl border border-slate-800">
            {/* Playback Action Buttons */}
            <div className="flex items-center gap-2">
              {status?.is_playing ? (
                <button
                  onClick={() => handleAction("pause")}
                  disabled={loading}
                  className="px-4 py-2 bg-amber-600 hover:bg-amber-500 text-white font-semibold text-xs rounded-lg transition flex items-center gap-1.5 shadow"
                >
                  <Pause className="w-4 h-4" />
                  <span>PAUSE</span>
                </button>
              ) : (
                <button
                  onClick={() => handleAction("play")}
                  disabled={loading}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs rounded-lg transition flex items-center gap-1.5 shadow"
                >
                  <Play className="w-4 h-4" />
                  <span>PLAY</span>
                </button>
              )}

              <button
                onClick={() => handleAction("reset")}
                disabled={loading}
                className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-xs rounded-lg transition flex items-center gap-1.5"
              >
                <RotateCcw className="w-4 h-4" />
                <span>RESET</span>
              </button>

              <button
                onClick={() => handleAction("step")}
                disabled={loading}
                className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-xs rounded-lg transition"
              >
                STEP STEP &gt;
              </button>
            </div>

            {/* Speed Buttons */}
            <div className="flex items-center gap-1.5 bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs">
              <span className="text-slate-400 px-2 font-mono text-[11px]">Speed:</span>
              {[0.5, 1.0, 2.0, 5.0].map((s) => (
                <button
                  key={s}
                  onClick={() => handleAction("set_speed", s)}
                  className={`px-2.5 py-1 rounded text-xs font-mono font-bold transition ${
                    status?.playback_speed === s
                      ? "bg-purple-600 text-white shadow"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
                  }`}
                >
                  {s}x
                </button>
              ))}
            </div>
          </div>

          {/* Replay Flow Pipeline Trace */}
          <div className="space-y-2">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 font-mono">
              Replay Execution Flow Pipeline
            </h4>
            <div className="flex items-center justify-between gap-1 overflow-x-auto p-3 bg-slate-950 rounded-xl border border-slate-800 text-[11px] font-mono text-slate-300">
              <span className="px-2 py-1 bg-slate-900 border border-slate-800 rounded">Dataset / IQ File</span>
              <span>→</span>
              <span className="px-2 py-1 bg-slate-900 border border-slate-800 rounded">RF Observation</span>
              <span>→</span>
              <span className="px-2 py-1 bg-slate-900 border border-slate-800 rounded">DSP / FFT</span>
              <span>→</span>
              <span className="px-2 py-1 bg-slate-900 border border-slate-800 rounded">Activity Detector</span>
              <span>→</span>
              <span className="px-2 py-1 bg-slate-900 border border-slate-800 rounded">ML Evidence</span>
              <span>→</span>
              <span className="px-2 py-1 bg-slate-900 border border-slate-800 rounded">OOD Guard</span>
              <span>→</span>
              <span className="px-2 py-1 bg-slate-900 border border-slate-800 rounded">Availability</span>
              <span>→</span>
              <span className="px-2 py-1 bg-purple-950/80 border border-purple-800 text-purple-300 rounded font-bold">Allocation</span>
            </div>
          </div>

          {/* Current Replay Observation Status */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 font-mono text-xs">
            <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
              <div className="text-slate-400 uppercase">Provenance</div>
              <div className="text-purple-400 font-bold mt-0.5">{status?.provenance || "DATASET"}</div>
            </div>
            <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
              <div className="text-slate-400 uppercase">Current Sample Index</div>
              <div className="text-slate-100 font-bold mt-0.5">
                {status?.current_index} / {status?.total_samples}
              </div>
            </div>
            <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
              <div className="text-slate-400 uppercase">Playback Speed</div>
              <div className="text-amber-400 font-bold mt-0.5">{status?.playback_speed}x</div>
            </div>
            <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
              <div className="text-slate-400 uppercase">Live SDR Connection</div>
              <div className="text-rose-400 font-bold mt-0.5">FALSE (Replay Only)</div>
            </div>
          </div>

          {/* Current Replayed Observation Pipeline Telemetry */}
          {status?.last_observation && (
            <div className="p-4 bg-slate-950 border border-purple-900/50 rounded-xl space-y-3 font-mono text-xs">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="text-purple-400 font-bold uppercase">
                  Replayed Observation #{status.last_observation.metadata?.dataset_index ?? status.current_index} Telemetry
                </span>
                <span className="text-slate-400 text-[11px]">{status.last_observation.timestamp}</span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div>Frequency: <strong className="text-cyan-300">{status.last_observation.center_freq_mhz} MHz</strong></div>
                <div>Bandwidth: <strong className="text-white">{status.last_observation.bandwidth_khz} kHz</strong></div>
                <div>Signal Power: <strong className="text-amber-300">{status.last_observation.signal_strength_dbm} dBm</strong></div>
                <div>I/Q Availability: <strong className="text-emerald-400">{status.last_observation.iq_available ? "PRESENT" : "MISSING"}</strong></div>
              </div>
              {status.last_observation.pipeline_result && (
                <div className="pt-2 border-t border-slate-800 flex flex-wrap items-center justify-between gap-2 text-[11px]">
                  <div>Availability: <strong className="text-emerald-400">{status.last_observation.pipeline_result.availability}</strong></div>
                  <div>Activity: <strong className="text-cyan-300">{status.last_observation.pipeline_result.activity}</strong></div>
                  <div>ML Prob: <strong className="text-amber-300">{status.last_observation.pipeline_result.probability}</strong></div>
                  <div>OOD Warning: <strong className={status.last_observation.pipeline_result.ood_warning ? "text-rose-400" : "text-emerald-400"}>{status.last_observation.pipeline_result.ood_warning ? "YES" : "PASS"}</strong></div>
                </div>
              )}
            </div>
          )}
        </div>
      </Panel>
    </div>
  );
}
