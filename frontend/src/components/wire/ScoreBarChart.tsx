import React from "react";
import {
  BarChart,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
  Bar as RechartsBar,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import type { ChannelCandidate } from "@/types/wire-watcher";

// Bypass React 18 JSX typing incompatibility with Recharts 2.x
const Tooltip = RechartsTooltip as any;
const Bar = RechartsBar as any;

interface ScoreBarChartProps {
  candidates: ChannelCandidate[];
  threshold: number;
  mode: "eligible" | "top";
}

export const ScoreBarChart: React.FC<ScoreBarChartProps> = ({
  candidates,
  threshold,
  mode,
}) => {
  const eligibleOnly = candidates.filter(
    (c) =>
      c.score >= threshold &&
      (c.recommendation_status === "RECOMMENDED" || c.recommendation_status === "FREE") &&
      !c.ood_warning &&
      c.guard_band_status !== "CONFLICT"
  );

  const displayList = mode === "eligible" ? eligibleOnly.slice(0, 10) : [...candidates].sort((a, b) => b.score - a.score).slice(0, 10);

  if (mode === "eligible" && displayList.length === 0) {
    return (
      <div className="p-6 text-center text-amber-400 bg-amber-950/20 border border-amber-800/60 rounded-lg text-xs font-mono">
        No eligible candidates meet all safety checks and the minimum recommendation score threshold ({threshold}).
      </div>
    );
  }

  const chartData = displayList.map((c) => ({
    name: `${c.candidate_id} (${c.center_freq_mhz.toFixed(1)}M)`,
    score: c.score,
    status: c.recommendation_status,
    freq: c.center_freq_mhz,
    candidate_id: c.candidate_id,
  }));

  const getBarColor = (status: string) => {
    switch (status) {
      case "RECOMMENDED":
        return "#a855f7"; // purple
      case "FREE":
        return "#10b981"; // emerald
      case "BUSY":
        return "#ef4444"; // red
      case "UNCERTAIN":
        return "#f59e0b"; // amber
      default:
        return "#64748b"; // slate
    }
  };

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <div className="rounded border border-slate-700 bg-slate-900 p-2 text-xs font-mono text-slate-200 space-y-1 shadow-lg">
          <div className="font-bold text-sky-400">{item.candidate_id} — {item.freq.toFixed(3)} MHz</div>
          <div>Score: <span className="font-bold text-amber-300">{item.score.toFixed(1)} / 100</span></div>
          <div>Status: <span className="font-semibold" style={{ color: getBarColor(item.status) }}>{item.status}</span></div>
          <div>Eligibility: {item.score >= threshold ? "Passes (≥50)" : "Fails (<50)"}</div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-slate-950/60 rounded-lg border border-slate-800 p-3">
      <ResponsiveContainer width="100%" height={260}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 10, right: 30, left: 90, bottom: 10 }}
        >
          {/* @ts-ignore */}
          <XAxis
            type="number"
            domain={[0, 100]}
            stroke="#64748b"
            tick={{ fill: "#94a3b8", fontSize: 11, fontFamily: "monospace" }}
            label={{ value: "Allocation Score (0–100)", position: "insideBottom", offset: -5, fill: "#94a3b8", fontSize: 10 }}
          />
          {/* @ts-ignore */}
          <YAxis
            dataKey="name"
            type="category"
            stroke="#64748b"
            tick={{ fill: "#cbd5e1", fontSize: 11, fontFamily: "monospace" }}
            width={110}
          />
          <Tooltip content={<CustomTooltip />} />
          {/* @ts-ignore */}
          <ReferenceLine
            x={threshold}
            stroke="#f59e0b"
            strokeDasharray="4 4"
            strokeWidth={1.5}
            label={{ value: "Threshold: 50.0", fill: "#f59e0b", fontSize: 10, position: "top" }}
          />
          <Bar dataKey="score" radius={[0, 4, 4, 0]}>
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getBarColor(entry.status)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

