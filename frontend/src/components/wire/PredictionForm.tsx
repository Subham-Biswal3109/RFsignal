/**
 * Wire Watcher — RF Observation Ingestion Form
 *
 * Provides:
 *  - Validation via Zod with graceful handling of empty/blank optional I/Q fields
 *  - Interactive 1-click Presets (120 MHz Available, 90 MHz Occupied, 70 MHz Weak, 5000 MHz OOD)
 *  - Explicit error messages for invalid input
 *  - Clean toggle between simple RF observation and complex I/Q DSP features
 */

import React, { useState } from "react";
import { Loader2, Radar, Sparkles, AlertCircle, Radio, Check } from "lucide-react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Panel } from "@/components/wire/Panel";
import type { PredictRequest } from "@/types/wire-watcher";

// Helper to coerce empty string or null to undefined for optional numeric fields
const optionalNumber = () =>
  z.preprocess((val) => {
    if (val === "" || val === null || val === undefined) return undefined;
    const num = Number(val);
    return isNaN(num) ? undefined : num;
  }, z.number().optional());

const schema = z.object({
  frequency_mhz: z.coerce.number({ invalid_type_error: "Frequency is required" }).positive("Frequency must be > 0 MHz"),
  bandwidth_khz: z.coerce.number({ invalid_type_error: "Bandwidth is required" }).positive("Bandwidth must be > 0 kHz"),
  signal_strength_dbm: z.coerce.number({ invalid_type_error: "Signal strength is required" }),
  iq_available: z.coerce.number().int().min(0).max(1).default(0),

  iq_rms_magnitude: optionalNumber(),
  iq_magnitude_variance: optionalNumber(),
  iq_peak_magnitude: optionalNumber(),
  iq_crest_factor: optionalNumber(),
  iq_p10: optionalNumber(),
  iq_p50: optionalNumber(),
  iq_p90: optionalNumber(),
  iq_phase_concentration: optionalNumber(),
  iq_spectral_entropy: optionalNumber(),
  iq_spectral_peak_ratio: optionalNumber(),
});

type FormValues = z.infer<typeof schema>;

function Field({
  label,
  error,
  children,
}: {
  label: string;
  error?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <Label className="text-xs font-semibold uppercase tracking-wider text-slate-700 dark:text-slate-300">
        {label}
      </Label>
      {children}
      {error && <p className="text-[11px] text-red-600 dark:text-red-400 font-medium">{error}</p>}
    </div>
  );
}

export function PredictionForm({
  onSubmit,
  pending,
}: {
  onSubmit: (input: PredictRequest) => void;
  pending: boolean;
}) {
  const [showIqFields, setShowIqFields] = useState(false);

  const {
    register,
    handleSubmit,
    setValue,
    formState: { errors },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      frequency_mhz: 120,
      bandwidth_khz: 50,
      signal_strength_dbm: -60,
      iq_available: 0,
    },
  });

  const onFormSubmit = (data: FormValues) => {
    const payload: PredictRequest = {
      frequency_mhz: data.frequency_mhz,
      bandwidth_khz: data.bandwidth_khz,
      signal_strength_dbm: data.signal_strength_dbm,
      iq_available: showIqFields ? 1 : 0,
      iq_rms_magnitude: showIqFields ? data.iq_rms_magnitude : undefined,
      iq_magnitude_variance: showIqFields ? data.iq_magnitude_variance : undefined,
      iq_peak_magnitude: showIqFields ? data.iq_peak_magnitude : undefined,
      iq_crest_factor: showIqFields ? data.iq_crest_factor : undefined,
      iq_p10: showIqFields ? data.iq_p10 : undefined,
      iq_p50: showIqFields ? data.iq_p50 : undefined,
      iq_p90: showIqFields ? data.iq_p90 : undefined,
      iq_phase_concentration: showIqFields ? data.iq_phase_concentration : undefined,
      iq_spectral_entropy: showIqFields ? data.iq_spectral_entropy : undefined,
      iq_spectral_peak_ratio: showIqFields ? data.iq_spectral_peak_ratio : undefined,
    };
    onSubmit(payload);
  };

  const applyPreset = (preset: {
    freq: number;
    bw: number;
    dbm: number;
    iq?: boolean;
    iqData?: Partial<FormValues>;
  }) => {
    setValue("frequency_mhz", preset.freq, { shouldValidate: true });
    setValue("bandwidth_khz", preset.bw, { shouldValidate: true });
    setValue("signal_strength_dbm", preset.dbm, { shouldValidate: true });
    if (preset.iq && preset.iqData) {
      setShowIqFields(true);
      setValue("iq_available", 1);
      Object.entries(preset.iqData).forEach(([k, v]) => {
        setValue(k as any, v, { shouldValidate: true });
      });
    } else {
      setShowIqFields(false);
      setValue("iq_available", 0);
    }
  };

  return (
    <form onSubmit={handleSubmit(onFormSubmit)} className="space-y-6">
      {/* ── Quick Preset Selector Buttons ── */}
      <div className="p-3.5 rounded-xl border border-sky-200 dark:border-sky-900/50 bg-sky-50/60 dark:bg-sky-950/20 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold uppercase tracking-wider text-sky-900 dark:text-sky-300 flex items-center gap-1.5">
            <Sparkles className="size-3.5 text-sky-600 dark:text-sky-400" />
            1-Click Scenario Presets:
          </span>
          <span className="text-[10px] text-slate-500 font-mono">Click to auto-fill</span>
        </div>

        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => applyPreset({ freq: 120, bw: 50, dbm: -65 })}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-slate-700 hover:border-emerald-500 hover:text-emerald-600 transition-colors shadow-sm flex items-center gap-1.5"
          >
            <span className="size-2 rounded-full bg-emerald-500"></span>
            120 MHz (Clear → AVAILABLE)
          </button>

          <button
            type="button"
            onClick={() => applyPreset({ freq: 90, bw: 50, dbm: -30 })}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-slate-700 hover:border-red-500 hover:text-red-600 transition-colors shadow-sm flex items-center gap-1.5"
          >
            <span className="size-2 rounded-full bg-red-500"></span>
            90 MHz (Strong → OCCUPIED)
          </button>

          <button
            type="button"
            onClick={() => applyPreset({ freq: 70, bw: 50, dbm: -110 })}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-slate-700 hover:border-sky-500 hover:text-sky-600 transition-colors shadow-sm flex items-center gap-1.5"
          >
            <span className="size-2 rounded-full bg-sky-500"></span>
            70 MHz (Weak Noise → AVAILABLE)
          </button>

          <button
            type="button"
            onClick={() => applyPreset({ freq: 5000, bw: 20000, dbm: 20 })}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-slate-700 hover:border-amber-500 hover:text-amber-600 transition-colors shadow-sm flex items-center gap-1.5"
          >
            <span className="size-2 rounded-full bg-amber-500"></span>
            5000 MHz (OOD → UNCERTAIN)
          </button>

          <button
            type="button"
            onClick={() =>
              applyPreset({
                freq: 140,
                bw: 50,
                dbm: -55,
                iq: true,
                iqData: {
                  iq_rms_magnitude: 0.35,
                  iq_magnitude_variance: 0.02,
                  iq_peak_magnitude: 0.72,
                  iq_crest_factor: 2.06,
                  iq_p10: 0.12,
                  iq_p50: 0.31,
                  iq_p90: 0.58,
                  iq_phase_concentration: 0.45,
                  iq_spectral_entropy: 0.82,
                  iq_spectral_peak_ratio: 0.04,
                },
              })
            }
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-slate-700 hover:border-purple-500 hover:text-purple-600 transition-colors shadow-sm flex items-center gap-1.5"
          >
            <span className="size-2 rounded-full bg-purple-500"></span>
            140 MHz (With I/Q DSP)
          </button>
        </div>
      </div>

      {/* ── Primary RF Parameters ── */}
      <Panel
        title="Primary RF Signal Parameters"
        subtitle="Measured physical parameters. Signal strength drives the activity detector threshold (-50 dBm median); it is excluded from ML predictors to prevent circular leakage."
      >
        <div className="grid gap-4 sm:grid-cols-3">
          <Field label="Carrier Frequency (MHz)" error={errors.frequency_mhz?.message}>
            <div className="relative">
              <Input type="number" step="any" placeholder="e.g. 120" {...register("frequency_mhz")} />
              <span className="absolute right-3 top-2 text-xs font-mono text-slate-400 pointer-events-none">
                MHz
              </span>
            </div>
          </Field>

          <Field label="Channel Bandwidth (kHz)" error={errors.bandwidth_khz?.message}>
            <div className="relative">
              <Input type="number" step="any" placeholder="e.g. 50" {...register("bandwidth_khz")} />
              <span className="absolute right-3 top-2 text-xs font-mono text-slate-400 pointer-events-none">
                kHz
              </span>
            </div>
          </Field>

          <Field label="Received Signal Strength (dBm)" error={errors.signal_strength_dbm?.message}>
            <div className="relative">
              <Input type="number" step="any" placeholder="e.g. -60" {...register("signal_strength_dbm")} />
              <span className="absolute right-3 top-2 text-xs font-mono text-slate-400 pointer-events-none">
                dBm
              </span>
            </div>
          </Field>
        </div>

        <div className="mt-4 pt-3 border-t border-slate-200 dark:border-slate-800 flex items-center justify-between">
          <label className="flex items-center gap-2 text-xs font-medium text-slate-700 dark:text-slate-300 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showIqFields}
              onChange={(e) => {
                setShowIqFields(e.target.checked);
                setValue("iq_available", e.target.checked ? 1 : 0);
              }}
              className="rounded border-slate-300 text-sky-600 focus:ring-sky-500 size-4"
            />
            <span>Include Optional Complex I/Q Baseband DSP Features</span>
          </label>

          <span className="text-[11px] text-slate-500 font-mono">
            {showIqFields ? "I/Q Available: YES (1)" : "I/Q Available: NO (0, Median Imputed)"}
          </span>
        </div>
      </Panel>

      {/* ── Optional Derived I/Q Features ── */}
      {showIqFields && (
        <Panel
          title="Optional Derived I/Q Baseband Features"
          subtitle="Mathematical envelope and spectral entropy statistics calculated from 100 complex samples. Leave blank if unavailable; the pipeline will impute missing features without fabrication."
        >
          <div className="grid gap-4 sm:grid-cols-4">
            <Field label="RMS Magnitude" error={errors.iq_rms_magnitude?.message}>
              <Input type="number" step="any" placeholder="e.g. 0.35" {...register("iq_rms_magnitude")} />
            </Field>

            <Field label="Magnitude Variance" error={errors.iq_magnitude_variance?.message}>
              <Input type="number" step="any" placeholder="e.g. 0.02" {...register("iq_magnitude_variance")} />
            </Field>

            <Field label="Peak Magnitude" error={errors.iq_peak_magnitude?.message}>
              <Input type="number" step="any" placeholder="e.g. 0.72" {...register("iq_peak_magnitude")} />
            </Field>

            <Field label="Crest Factor" error={errors.iq_crest_factor?.message}>
              <Input type="number" step="any" placeholder="e.g. 2.06" {...register("iq_crest_factor")} />
            </Field>

            <Field label="P10 Magnitude" error={errors.iq_p10?.message}>
              <Input type="number" step="any" placeholder="e.g. 0.12" {...register("iq_p10")} />
            </Field>

            <Field label="P50 Magnitude (Median)" error={errors.iq_p50?.message}>
              <Input type="number" step="any" placeholder="e.g. 0.31" {...register("iq_p50")} />
            </Field>

            <Field label="P90 Magnitude" error={errors.iq_p90?.message}>
              <Input type="number" step="any" placeholder="e.g. 0.58" {...register("iq_p90")} />
            </Field>

            <Field label="Phase Concentration" error={errors.iq_phase_concentration?.message}>
              <Input type="number" step="any" placeholder="e.g. 0.45" {...register("iq_phase_concentration")} />
            </Field>

            <Field label="Spectral Entropy" error={errors.iq_spectral_entropy?.message}>
              <Input type="number" step="any" placeholder="e.g. 0.82" {...register("iq_spectral_entropy")} />
            </Field>

            <Field label="Spectral Peak Ratio" error={errors.iq_spectral_peak_ratio?.message}>
              <Input type="number" step="any" placeholder="e.g. 0.04" {...register("iq_spectral_peak_ratio")} />
            </Field>
          </div>
        </Panel>
      )}

      {/* ── Submit Button ── */}
      <div className="flex items-center gap-3 pt-1">
        <Button type="submit" disabled={pending} className="gap-2 px-6 py-2.5 font-bold shadow-md">
          {pending ? <Loader2 className="size-4 animate-spin" /> : <Radar className="size-4" />}
          {pending ? "Analyzing Observation…" : "Assess Spectrum Availability"}
        </Button>

        <span className="text-xs text-slate-500">
          Evaluates RF detector, ML activity probability, and multivariate OOD boundaries.
        </span>
      </div>
    </form>
  );
}
