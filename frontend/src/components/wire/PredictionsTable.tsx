import { AvailabilityBadge } from "@/components/wire/AvailabilityBadge";
import type { PredictionRecord } from "@/types/wire-watcher";
import {
  formatFrequencyRange,
  formatLocation,
  formatNumber,
  formatProbability,
  formatTimestamp,
} from "@/utils/format";

export function PredictionsTable({
  records,
  showBandwidth = true,
}: {
  records: PredictionRecord[];
  showBandwidth?: boolean;
}) {
  return (
    <>
      {/* Desktop / tablet table */}
      <div className="-mx-4 hidden overflow-x-auto px-4 md:block">
        <table className="w-full min-w-[900px] border-separate border-spacing-0 text-xs font-mono">
          <thead>
            <tr className="label-caps text-left text-slate-500 bg-slate-50 dark:bg-slate-900">
              <th className="border-b border-border p-2.5 font-semibold">Timestamp</th>
              <th className="border-b border-border p-2.5 font-semibold">Frequency</th>
              {showBandwidth ? (
                <th className="border-b border-border p-2.5 font-semibold">Bandwidth</th>
              ) : null}
              <th className="border-b border-border p-2.5 font-semibold">Signal Power</th>
              <th className="border-b border-border p-2.5 font-semibold">RF Activity</th>
              <th className="border-b border-border p-2.5 font-semibold">Availability</th>
              <th className="border-b border-border p-2.5 font-semibold">Confidence</th>
              <th className="border-b border-border p-2.5 font-semibold">OOD Guard</th>
              <th className="border-b border-border p-2.5 font-semibold">Data Source</th>
            </tr>
          </thead>
          <tbody>
            {records.map((r) => {
              const activityLabel = r.activity || (r.signal_power_dbm !== null && r.signal_power_dbm !== undefined ? (r.signal_power_dbm >= -50 ? "DETECTED" : "NOT DETECTED") : "—");
              const availabilityLabel = r.availability || (r.ood_status ? "UNCERTAIN" : r.available ? "AVAILABLE" : "OCCUPIED");
              const oodLabel = r.ood_status ? "TRUE" : "FALSE";
              const confidenceLabel = r.confidence || (r.probability !== null && r.probability !== undefined ? (Math.abs(r.probability - 0.5) > 0.15 ? "High" : "Medium") : "—");

              return (
                <tr key={r.id} className="transition-colors hover:bg-slate-50/70 dark:hover:bg-slate-900/50">
                  <td className="border-b border-border/60 p-2.5 whitespace-nowrap text-slate-500">
                    {formatTimestamp(r.timestamp)}
                  </td>
                  <td className="border-b border-border/60 p-2.5 whitespace-nowrap font-bold text-sky-600 dark:text-sky-400">
                    {formatFrequencyRange(r.start_frequency_mhz, r.end_frequency_mhz)}
                  </td>
                  {showBandwidth ? (
                    <td className="border-b border-border/60 p-2.5 whitespace-nowrap text-slate-700 dark:text-slate-300">
                      {formatNumber(r.bandwidth_mhz, 0, " MHz")}
                    </td>
                  ) : null}
                  <td className="border-b border-border/60 p-2.5 whitespace-nowrap font-bold text-slate-900 dark:text-slate-100">
                    {formatNumber(r.signal_power_dbm, 1, " dBm")}
                  </td>
                  <td className="border-b border-border/60 p-2.5 whitespace-nowrap font-semibold">
                    <span className={activityLabel === "DETECTED" ? "text-red-600 dark:text-red-400" : "text-emerald-600 dark:text-emerald-400"}>
                      {activityLabel}
                    </span>
                  </td>
                  <td className="border-b border-border/60 p-2.5 whitespace-nowrap">
                    <AvailabilityBadge available={availabilityLabel === "AVAILABLE"} />
                  </td>
                  <td className="border-b border-border/60 p-2.5 whitespace-nowrap text-slate-600 dark:text-slate-400">
                    {confidenceLabel}
                  </td>
                  <td className="border-b border-border/60 p-2.5 whitespace-nowrap font-bold">
                    <span className={r.ood_status ? "text-amber-600 dark:text-amber-400" : "text-slate-400"}>
                      {oodLabel}
                    </span>
                  </td>
                  <td className="border-b border-border/60 p-2.5 whitespace-nowrap text-[11px] text-slate-500 uppercase">
                    {r.data_source || "RF Signal Data"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Mobile cards */}
      <ul className="space-y-3 md:hidden">
        {records.map((r) => (
          <li key={r.id} className="rounded-lg border border-border bg-surface/60 p-3 font-mono text-xs">
            <div className="flex items-start justify-between gap-3">
              <p className="font-bold text-sky-600">
                {formatFrequencyRange(r.start_frequency_mhz, r.end_frequency_mhz)}
              </p>
              <AvailabilityBadge available={r.available} />
            </div>
            <dl className="mt-3 grid grid-cols-2 gap-2 text-xs">
              <div>
                <dt className="text-slate-500 uppercase text-[10px]">Signal Power</dt>
                <dd className="font-bold">{formatNumber(r.signal_power_dbm, 1, " dBm")}</dd>
              </div>
              <div>
                <dt className="text-slate-500 uppercase text-[10px]">Activity</dt>
                <dd>{r.activity || (r.signal_power_dbm && r.signal_power_dbm >= -50 ? "DETECTED" : "NOT DETECTED")}</dd>
              </div>
              <div>
                <dt className="text-slate-500 uppercase text-[10px]">OOD Guard</dt>
                <dd className={r.ood_status ? "text-amber-600 font-bold" : "text-slate-500"}>
                  {r.ood_status ? "TRUE" : "FALSE"}
                </dd>
              </div>
              <div>
                <dt className="text-slate-500 uppercase text-[10px]">Confidence</dt>
                <dd>{r.confidence || "Medium"}</dd>
              </div>
              <div>
                <dt className="text-slate-500 uppercase text-[10px]">Source</dt>
                <dd>{r.data_source || "RF Signal Data"}</dd>
              </div>
              <div>
                <dt className="text-slate-500 uppercase text-[10px]">Timestamp</dt>
                <dd>{formatTimestamp(r.timestamp)}</dd>
              </div>
            </dl>
          </li>
        ))}
      </ul>
    </>
  );
}

