/**
 * Wire Watcher — RF Event Analytics & Channel Utilization View (Sections 11 & 12)
 *
 * Displays summary statistics of detected RF events and calculates temporal
 * channel utilization: Utilization % = (Active Time / Observation Time) * 100.
 */

import React, { useState, useEffect } from "react";
import { Panel } from "@/components/wire/Panel";
import { Activity, Clock, Zap, Radio, BarChart3, AlertCircle, ShieldAlert } from "lucide-react";
import { apiFetch } from "@/lib/api";
import type { RFEventSummary, ChannelUtilization } from "@/types/wire-watcher";

export function RFEventAnalyticsView() {
  const [eventsSummary, setEventsSummary] = useState<RFEventSummary | null>(null);
  const [utilization, setUtilization] = useState<ChannelUtilization | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [evData, utData] = await Promise.all([
        apiFetch<RFEventSummary>("/api/analytics/events"),
        apiFetch<ChannelUtilization>("/api/analytics/utilization"),
      ]);

      setEventsSummary(evData);
      setUtilization(utData);
    } catch (err: any) {
      setError(err.message || "Failed to load RF event analytics");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleRefresh = async () => {
    setLoading(true);
    setError(null);
    try {
      // Trigger deterministic event ingestion from dataset
      await apiFetch("/api/events/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force: true }),
      });
      await fetchData();
    } catch (err: any) {
      setError(err.message || "Failed to refresh event data");
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <Activity className="w-5 h-5 text-cyan-400" />
            RF Event Analytics & Channel Utilization
          </h2>
          <p className="text-xs text-slate-400">
            Temporal spectrum observation analysis derived from real logged RF events in SQLite.
          </p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={loading}
          className="px-3.5 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold rounded-lg transition shadow disabled:opacity-50"
        >
          {loading ? "Ingesting Data..." : "Refresh Event Data"}
        </button>
      </div>

      {error && (
        <div className="p-4 bg-rose-950/80 border border-rose-800 rounded-xl text-rose-300 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Utilization Metric Section */}
      <Panel title="Channel Utilization Analytics (Section 11)">
        {utilization?.utilization_status === "UNAVAILABLE" ? (
          <div className="p-6 bg-slate-950/80 border border-slate-800 rounded-lg text-center space-y-2">
            <div className="text-sm font-bold text-amber-400 font-mono">UTILIZATION: UNAVAILABLE</div>
            <div className="text-xs text-slate-400">Reason: {utilization.reason}</div>
            <div className="text-[11px] text-slate-500 italic">
              Temporal utilization is calculated exclusively from real observed timelines. Zero values are fabricated.
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg">
              <div className="text-xs text-slate-400 uppercase font-mono">Channel Utilization</div>
              <div className="text-2xl font-extrabold text-emerald-400 font-mono mt-1">
                {utilization?.utilization_percentage.toFixed(1)}%
              </div>
              <div className="text-[10px] text-slate-500 mt-1">Active Time / Observation Time</div>
            </div>

            <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg">
              <div className="text-xs text-slate-400 uppercase font-mono">Observation Window</div>
              <div className="text-xl font-bold text-slate-200 font-mono mt-1">
                {utilization?.total_observation_duration_s.toFixed(0)} s
              </div>
              <div className="text-[10px] text-slate-500 mt-1">Total recorded duration</div>
            </div>

            <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg">
              <div className="text-xs text-slate-400 uppercase font-mono">Active Duration</div>
              <div className="text-xl font-bold text-rose-400 font-mono mt-1">
                {utilization?.active_duration_s.toFixed(1)} s
              </div>
              <div className="text-[10px] text-slate-500 mt-1">Cumulative active RF state</div>
            </div>

            <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg">
              <div className="text-xs text-slate-400 uppercase font-mono">Event Count</div>
              <div className="text-xl font-bold text-cyan-400 font-mono mt-1">
                {utilization?.event_count}
              </div>
              <div className="text-[10px] text-slate-500 mt-1">Recorded transmissions</div>
            </div>
          </div>
        )}
      </Panel>

      {/* Summary Statistics Section */}
      <Panel title="RF Event Summary Statistics (Section 12)">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg">
            <div className="text-xs text-slate-400 uppercase font-mono">Total RF Events</div>
            <div className="text-2xl font-bold text-slate-100 font-mono mt-1">
              {eventsSummary?.total_rf_events || 0}
            </div>
          </div>

          <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg">
            <div className="text-xs text-slate-400 uppercase font-mono">Avg Event Duration</div>
            <div className="text-2xl font-bold text-amber-400 font-mono mt-1">
              {eventsSummary?.average_event_duration_s.toFixed(2) || "0.00"} s
            </div>
          </div>

          <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg">
            <div className="text-xs text-slate-400 uppercase font-mono">Max Event Duration</div>
            <div className="text-2xl font-bold text-rose-400 font-mono mt-1">
              {eventsSummary?.maximum_event_duration_s.toFixed(2) || "0.00"} s
            </div>
          </div>

          <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg">
            <div className="text-xs text-slate-400 uppercase font-mono">Max Peak Power</div>
            <div className="text-2xl font-bold text-cyan-400 font-mono mt-1">
              {eventsSummary?.maximum_peak_power_dbm || "0.0"} dBm
            </div>
          </div>
        </div>

        {/* Logged Events Table */}
        <div className="mt-5">
          <h4 className="text-xs font-semibold text-slate-300 uppercase font-mono mb-2">
            Recorded RF Event History ({eventsSummary?.events.length || 0} Events)
          </h4>
          {eventsSummary?.events && eventsSummary.events.length > 0 ? (
            <div className="overflow-x-auto rounded-lg border border-slate-800">
              <table className="w-full text-xs text-left font-mono text-slate-300">
                <thead className="bg-slate-950 text-slate-400 uppercase text-[10px] border-b border-slate-800">
                  <tr>
                    <th className="px-3 py-2">ID</th>
                    <th className="px-3 py-2">Frequency</th>
                    <th className="px-3 py-2">Start Time</th>
                    <th className="px-3 py-2">Duration</th>
                    <th className="px-3 py-2">Peak Power</th>
                    <th className="px-3 py-2">Avg Power</th>
                    <th className="px-3 py-2">Source</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 bg-slate-900/50">
                  {eventsSummary.events.map((ev) => (
                    <tr key={ev.event_id} className="hover:bg-slate-800/50">
                      <td className="px-3 py-2 text-slate-400">#{ev.event_id}</td>
                      <td className="px-3 py-2 font-bold text-cyan-300">{ev.frequency_mhz.toFixed(3)} MHz</td>
                      <td className="px-3 py-2 text-slate-400">{ev.start_time ? new Date(ev.start_time).toLocaleTimeString() : "—"}</td>
                      <td className="px-3 py-2 text-amber-300">{ev.duration_seconds ? `${ev.duration_seconds}s` : "Ongoing"}</td>
                      <td className="px-3 py-2 text-rose-300">{ev.peak_power_dbm} dBm</td>
                      <td className="px-3 py-2 text-slate-300">{ev.avg_power_dbm} dBm</td>
                      <td className="px-3 py-2 text-slate-400">{ev.source_type}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="p-4 bg-slate-950 text-slate-500 text-xs text-center rounded border border-slate-800">
              No RF events recorded in SQLite yet.
            </div>
          )}
        </div>
      </Panel>
    </div>
  );
}
