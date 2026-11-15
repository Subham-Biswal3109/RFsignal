import React from "react";
import {
  ComposedChart,
  Line as RechartsLine,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
  CartesianGrid,
  ReferenceLine,
  ReferenceArea as RechartsReferenceArea,
  ResponsiveContainer,
} from "recharts";
import type { ChannelCandidate } from "@/types/wire-watcher";

// Bypass React 18 JSX typing incompatibility with Recharts 2.x
const Tooltip = RechartsTooltip as any;
const Line = RechartsLine as any;
const ReferenceArea = RechartsReferenceArea as any;

interface SpectrumChartProps {
  candidates: ChannelCandidate[];
  noiseFloorDbm: number;
  selectedCandidate?: ChannelCandidate;
  onSelect?: (c: ChannelCandidate) => void;
}

export const SpectrumChart: React.FC<SpectrumChartProps> = ({
  candidates,
  noiseFloorDbm,
  selectedCandidate,
  onSelect,
}) => {
  // Transform candidates for plotting
  const data = candidates.map((c) => ({
    freq: c.center_freq_mhz,
    power: c.observed_power_dbm ?? (c.snr_db !== null && c.snr_db !== undefined ? noiseFloorDbm + c.snr_db : null),
    status: c.recommendation_status,
    id: c.candidate_id,
    score: c.score,
    bandwidth_khz: c.bandwidth_khz,
    guard_band_status: c.guard_band_status,
  }));

  const getColor = (status: string) => {
    switch (status) {
      case "FREE":
        return "#10b981"; // emerald green
      case "BUSY":
        return "#ef4444"; // red
      case "UNCERTAIN":
        return "#f59e0b"; // amber
      case "RECOMMENDED":
        return "#a855f7"; // purple
      case "REJECTED":
        return "#64748b"; // gray/slate
      default:
        return "#475569";
    }
  };

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <div className="rounded-lg border border-slate-700 bg-slate-900 p-3 shadow-xl text-xs font-mono text-slate-200 space-y-1">
          <div className="font-bold text-sky-400 flex items-center justify-between gap-4">
            <span>{item.id}</span>
            <span>{item.freq.toFixed(3)} MHz</span>
          </div>
          <div>Status: <span className="font-bold" style={{ color: getColor(item.status) }}>{item.status}</span></div>
          <div>Allocation Score: <span className="font-bold text-amber-300">{item.score.toFixed(1)} / 100</span></div>
          <div>Observed Power: <span className="text-sky-300">{item.power !== null ? `${item.power.toFixed(1)} dBm` : "N/A"}</span></div>
          <div>Noise Floor Ref: <span className="text-slate-400">{noiseFloorDbm.toFixed(1)} dBm</span></div>
          <div>Bandwidth: {item.bandwidth_khz} kHz</div>
          <div>Guard Band: {item.guard_band_status || "SAFE_MARGIN"}</div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-3">
      {/* Chart Legend */}
      <div className="flex flex-wrap items-center justify-between text-xs font-mono text-slate-400 pb-1">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm bg-purple-500/60 border border-purple-400" />
            <span className="text-purple-300 font-semibold">RECOMMENDED</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm bg-emerald-500/40 border border-emerald-500" />
            <span className="text-emerald-400">FREE</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm bg-rose-500/40 border border-rose-500" />
            <span className="text-rose-400">BUSY</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm bg-amber-500/40 border border-amber-500" />
            <span className="text-amber-400">UNCERTAIN</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm bg-slate-600/40 border border-slate-500" />
            <span className="text-slate-400">REJECTED</span>
          </div>
        </div>
        <div className="flex items-center gap-2 text-[11px]">
          <span className="inline-block w-4 border-t-2 border-dashed border-rose-400" />
          <span>Noise Floor Ref ({noiseFloorDbm} dBm)</span>
        </div>
      </div>

      <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
        <ResponsiveContainer width="100%" height={320}>
          <ComposedChart
            data={data}
            onClick={(state: any) => {
              if (state && state.activeTooltipIndex !== undefined && onSelect) {
                onSelect(candidates[state.activeTooltipIndex]);
              }
            }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            {/* @ts-ignore */}
            <XAxis
              dataKey="freq"
              type="number"
              domain={[70, 160]}
              tickCount={10}
              stroke="#64748b"
              tick={{ fill: "#94a3b8", fontSize: 11, fontFamily: "monospace" }}
              label={{ value: "Frequency (MHz)", position: "insideBottom", offset: -5, fill: "#94a3b8", fontSize: 11 }}
            />
            {/* @ts-ignore */}
            <YAxis
              domain={[-120, -30]}
              stroke="#64748b"
              tick={{ fill: "#94a3b8", fontSize: 11, fontFamily: "monospace" }}
              label={{ value: "Received Power (dBm)", angle: -90, position: "insideLeft", fill: "#94a3b8", fontSize: 11 }}
            />
            <Tooltip content={<CustomTooltip />} />
            {/* @ts-ignore */}
            <ReferenceLine
              y={noiseFloorDbm}
              stroke="#f43f5e"
              strokeDasharray="4 4"
              strokeWidth={1.5}
              label={{ value: `Noise Floor (${noiseFloorDbm} dBm)`, fill: "#f43f5e", fontSize: 10, position: "insideTopLeft" }}
            />
            {candidates.map((c) => (
              /* @ts-ignore */
              <ReferenceArea
                key={c.candidate_id}
                x1={c.center_freq_mhz - c.bandwidth_khz / 2000}
                x2={c.center_freq_mhz + c.bandwidth_khz / 2000}
                y1={-120}
                y2={-30}
                fill={getColor(c.recommendation_status)}
                fillOpacity={c.candidate_id === selectedCandidate?.candidate_id ? 0.55 : 0.18}
                stroke={c.candidate_id === selectedCandidate?.candidate_id ? "#38bdf8" : undefined}
                strokeWidth={c.candidate_id === selectedCandidate?.candidate_id ? 2 : 0}
              />
            ))}
            <Line
              type="monotone"
              dataKey="power"
              stroke="#38bdf8"
              strokeWidth={1.5}
              dot={{ r: 2, fill: "#38bdf8" }}
              activeDot={{ r: 5, fill: "#06b6d4" }}
              connectNulls
              name="Observed Signal Power"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

