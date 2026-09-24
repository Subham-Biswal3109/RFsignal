/**
 * Wire Watcher — ML Performance, Leakage Audit & Validation Dashboard
 *
 * Implements Phases 1 to 15 of ML Performance Improvement Investigation:
 *   - Ground-truth honesty ("Occupancy ground truth: UNVERIFIED", pseudo-label flowchart)
 *   - VALIDATED section: Frozen RF v2, ROC-AUC ≈ 0.4988, PR-AUC ≈ 0.5076, leakage audit
 *   - EXPERIMENTAL section: Feature Group Experiments (Exp A to Exp E) & Experimental Models
 *   - LIMITATION section: Phase 7 Decision (OPTION C: NO), Ground-truth improvement plan
 */

import React, { useState, useEffect } from "react";
import { Panel } from "@/components/wire/Panel";
import {
  ShieldAlert,
  ShieldCheck,
  Cpu,
  BarChart2,
  Database,
  CheckCircle2,
  AlertTriangle,
  Info,
  Layers,
  ArrowRight,
  Activity,
  Sliders,
  FlaskConical,
  Lock,
} from "lucide-react";

import { apiFetch } from "@/lib/api";

export function MLPerformanceDashboard() {
  const [auditData, setAuditData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    apiFetch<any>("/api/ml/audit")
      .then((data) => {
        setAuditData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to fetch ML audit data", err);
        setLoading(false);
      });
  }, []);

  return (
    <div className="space-y-6">
      {/* Ground-Truth Honesty Banner */}
      <div className="rounded-xl border border-amber-300 dark:border-amber-800 bg-amber-50/90 dark:bg-amber-950/40 p-5 space-y-3">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-amber-600 text-white shrink-0">
              <ShieldAlert className="size-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-amber-900 dark:text-amber-200">
                Ground-Truth Honesty & Target Disclaimer
              </h2>
              <p className="text-xs text-amber-800 dark:text-amber-300">
                This system operates on inferred activity pseudo-labels, not independently verified spectrum occupancy.
              </p>
            </div>
          </div>
          <div className="px-3 py-1.5 rounded-full text-xs font-mono font-bold bg-amber-200 dark:bg-amber-900 text-amber-900 dark:text-amber-100 border border-amber-400 dark:border-amber-700">
            Occupancy ground truth: UNVERIFIED
          </div>
        </div>

        {/* Inference Methodology Flowchart */}
        <div className="p-4 rounded-lg bg-white/80 dark:bg-slate-900/80 border border-amber-200 dark:border-amber-900">
          <div className="text-xs font-bold text-slate-700 dark:text-slate-300 mb-2">
            Activity Inference Methodology Flow:
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-2 text-center text-xs font-mono">
            <div className="p-2 rounded border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950">
              Observed RF Data
              <div className="text-[10px] text-slate-500 font-sans mt-0.5">164,160 rows (70–160 MHz)</div>
            </div>
            <div className="flex items-center justify-center text-amber-500 font-bold">
              <ArrowRight className="size-4" />
            </div>
            <div className="p-2 rounded border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950">
              Feature / DSP Analysis
              <div className="text-[10px] text-slate-500 font-sans mt-0.5">13 statistical I/Q moments</div>
            </div>
            <div className="flex items-center justify-center text-amber-500 font-bold">
              <ArrowRight className="size-4" />
            </div>
            <div className="p-2 rounded border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950">
              Inference Methodology
              <div className="text-[10px] text-slate-500 font-sans mt-0.5">Median Signal Split (Proxy)</div>
            </div>
            <div className="flex items-center justify-center text-amber-500 font-bold sm:col-span-4 lg:hidden">
              <ArrowRight className="size-4" />
            </div>
            <div className="p-2 rounded border border-amber-300 dark:border-amber-700 bg-amber-100 dark:bg-amber-950 font-bold text-amber-900 dark:text-amber-200 sm:col-span-4 lg:col-span-1">
              Inferred Activity Label
              <div className="text-[10px] font-normal text-amber-700 dark:text-amber-300 font-sans mt-0.5">
                INFERRED_RF_ACTIVITY
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Dataset Overview Panel */}
      <Panel
        title="Dataset & Telemetry Audit"
        subtitle="Full specification of observations, I/Q availability, and signal strength range."
      >
        <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-4">
          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/50">
            <div className="text-xs text-slate-500">Total Observations</div>
            <div className="text-xl font-bold font-mono text-slate-900 dark:text-slate-100 mt-1">164,160</div>
            <div className="text-[10px] text-slate-400 mt-1">Logged RF Signal Data</div>
          </div>

          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/50">
            <div className="text-xs text-slate-500">Carrier Frequencies</div>
            <div className="text-sm font-bold font-mono text-indigo-600 dark:text-indigo-400 mt-1">
              70, 90, 100, 120, 140, 160 MHz
            </div>
            <div className="text-[10px] text-slate-400 mt-1">6 VHF Spectrum Bands</div>
          </div>

          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/50">
            <div className="text-xs text-slate-500">I/Q Availability</div>
            <div className="text-xl font-bold font-mono text-emerald-600 dark:text-emerald-400 mt-1">66.6%</div>
            <div className="text-[10px] text-slate-400 mt-1">109,324 present / 54,836 missing</div>
          </div>

          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/50">
            <div className="text-xs text-slate-500">Signal Strength Span</div>
            <div className="text-sm font-bold font-mono text-slate-900 dark:text-slate-100 mt-1">
              -119.0 to -25.0 dBm
            </div>
            <div className="text-[10px] text-slate-400 mt-1">Mean: -50.0 dBm | Std: 27.1 dB</div>
          </div>
        </div>
      </Panel>

      {/* SECTION 1: VALIDATED PRODUCTION MODEL */}
      <div className="space-y-4">
        <div className="flex items-center gap-2 border-b border-purple-200 dark:border-purple-900/50 pb-2">
          <Lock className="size-5 text-purple-600 dark:text-purple-400" />
          <h3 className="text-base font-bold text-slate-900 dark:text-slate-100">
            VALIDATED PRODUCTION MODEL
          </h3>
          <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-bold bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 border border-purple-300 dark:border-purple-700">
            FROZEN v2 MODEL
          </span>
        </div>

        <div className="grid gap-4 md:grid-cols-3">
          <div className="p-4 rounded-xl border border-purple-200 dark:border-purple-900 bg-purple-50/50 dark:bg-purple-950/20">
            <div className="text-xs text-purple-700 dark:text-purple-300 font-bold">Model Architecture</div>
            <div className="text-base font-bold text-slate-900 dark:text-slate-100 mt-1">Random Forest v2</div>
            <div className="text-[11px] text-slate-500 mt-1 font-mono">60 Trees, Max Depth 10</div>
          </div>
          <div className="p-4 rounded-xl border border-purple-200 dark:border-purple-900 bg-purple-50/50 dark:bg-purple-950/20">
            <div className="text-xs text-purple-700 dark:text-purple-300 font-bold">Chronological ROC-AUC</div>
            <div className="text-xl font-bold font-mono text-purple-700 dark:text-purple-300 mt-1">0.4988</div>
            <div className="text-[11px] text-slate-500 mt-1 font-mono">Tested on 32,832 samples</div>
          </div>
          <div className="p-4 rounded-xl border border-purple-200 dark:border-purple-900 bg-purple-50/50 dark:bg-purple-950/20">
            <div className="text-xs text-purple-700 dark:text-purple-300 font-bold">Chronological PR-AUC</div>
            <div className="text-xl font-bold font-mono text-purple-700 dark:text-purple-300 mt-1">0.5076</div>
            <div className="text-[11px] text-slate-500 mt-1 font-mono">Chance level ≈ 0.5066</div>
          </div>
        </div>

        {/* Phase 1 Audit Summary Table */}
        <Panel
          title="Phase 1 — ML Implementation & Leakage Audit"
          subtitle="Systematic review of dataset structure, target generation, leakage controls, and recommendations."
        >
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 text-left font-mono text-slate-500">
                  <th className="p-3">Audit Item</th>
                  <th className="p-3">Current Implementation</th>
                  <th className="p-3">Identified Risk</th>
                  <th className="p-3">Engineering Recommendation</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800 font-sans">
                {auditData?.phase_1_ml_audit?.summary_table?.map((item: any, idx: number) => (
                  <tr key={idx} className="hover:bg-slate-50/50 dark:hover:bg-slate-900/50">
                    <td className="p-3 font-bold text-slate-900 dark:text-slate-100 font-mono">{item.Item}</td>
                    <td className="p-3 text-slate-700 dark:text-slate-300 font-mono text-[11px]">{item["Current Implementation"]}</td>
                    <td className="p-3 text-amber-600 dark:text-amber-400">{item.Risk}</td>
                    <td className="p-3 text-indigo-600 dark:text-indigo-300 font-semibold">{item.Recommendation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>

      {/* SECTION 2: EXPERIMENTAL FEATURE EXPERIMENTS & MODELS */}
      <div className="space-y-4 pt-4">
        <div className="flex items-center gap-2 border-b border-indigo-200 dark:border-indigo-900/50 pb-2">
          <FlaskConical className="size-5 text-indigo-600 dark:text-indigo-400" />
          <h3 className="text-base font-bold text-slate-900 dark:text-slate-100">
            EXPERIMENTAL FEATURE EXPERIMENTS & CANDIDATE MODELS
          </h3>
          <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-bold bg-indigo-100 dark:bg-indigo-950 text-indigo-800 dark:text-indigo-200 border border-indigo-300 dark:border-indigo-800">
            EXPERIMENTAL RESEARCH
          </span>
        </div>

        {/* Controlled Feature Group Experiments Table (Phase 4) */}
        <Panel
          title="Phase 4 — Controlled Feature Group Experiments (Exp A to Exp E)"
          subtitle="Identical train/val/test split and target across all controlled feature candidate groups."
        >
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 text-left text-slate-500">
                  <th className="p-3">Controlled Experiment Group</th>
                  <th className="p-3">Feature Count</th>
                  <th className="p-3">ROC-AUC</th>
                  <th className="p-3">PR-AUC</th>
                  <th className="p-3">Precision</th>
                  <th className="p-3">Recall</th>
                  <th className="p-3">F1 Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                {auditData?.phase_4_feature_experiments?.map((exp: any, idx: number) => (
                  <tr key={idx} className="hover:bg-slate-50/50 dark:hover:bg-slate-900/50">
                    <td className="p-3 font-bold text-slate-900 dark:text-slate-100">{exp.Experiment}</td>
                    <td className="p-3 text-slate-600 dark:text-slate-400">{exp.feature_count} features</td>
                    <td className="p-3 font-bold text-indigo-600 dark:text-indigo-400">{exp.roc_auc}</td>
                    <td className="p-3 text-slate-700 dark:text-slate-300">{exp.pr_auc}</td>
                    <td className="p-3 text-slate-700 dark:text-slate-300">{exp.precision}</td>
                    <td className="p-3 text-slate-700 dark:text-slate-300">{exp.recall}</td>
                    <td className="p-3 text-slate-700 dark:text-slate-300">{exp.f1}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        {/* Phase 5 Model Comparison Table */}
        <Panel
          title="Phase 5 — Model Architecture Comparison"
          subtitle="Comparing baseline classifiers, Frozen RF v2, Experimental RF v3, and physical DSP activity detector."
        >
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 text-left text-slate-500">
                  <th className="p-3">Model Candidate</th>
                  <th className="p-3">ROC-AUC</th>
                  <th className="p-3">PR-AUC</th>
                  <th className="p-3">Precision</th>
                  <th className="p-3">Recall</th>
                  <th className="p-3">F1 Score</th>
                  <th className="p-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                {auditData?.phase_5_model_comparison?.map((row: any, idx: number) => (
                  <tr
                    key={idx}
                    className={
                      row.model.includes("Frozen")
                        ? "bg-purple-50/60 dark:bg-purple-950/30 font-semibold"
                        : row.model.includes("Experimental")
                        ? "bg-indigo-50/60 dark:bg-indigo-950/30 font-semibold"
                        : "hover:bg-slate-50/50 dark:hover:bg-slate-900/50"
                    }
                  >
                    <td className="p-3 font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                      {row.model}
                    </td>
                    <td className="p-3 font-bold text-indigo-600 dark:text-indigo-400">{row.roc_auc}</td>
                    <td className="p-3 text-slate-700 dark:text-slate-300">{row.pr_auc}</td>
                    <td className="p-3 text-slate-700 dark:text-slate-300">{row.precision}</td>
                    <td className="p-3 text-slate-700 dark:text-slate-300">{row.recall}</td>
                    <td className="p-3 text-slate-700 dark:text-slate-300">{row.f1}</td>
                    <td className="p-3">
                      {row.status.includes("VALIDATED") ? (
                        <span className="px-2 py-0.5 rounded text-[10px] bg-purple-600 text-white font-bold">
                          PRODUCTION VALIDATED
                        </span>
                      ) : row.status.includes("EXPERIMENTAL") ? (
                        <span className="px-2 py-0.5 rounded text-[10px] bg-indigo-600 text-white font-bold">
                          EXPERIMENTAL v3
                        </span>
                      ) : row.status.includes("PRIMARY") ? (
                        <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-600 text-white font-bold">
                          PRIMARY EVIDENCE
                        </span>
                      ) : (
                        <span className="text-slate-400">Baseline</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        {/* Phase 12 Feature Importance */}
        <Panel
          title="Phase 12 — Model Feature Importance"
          subtitle="MODEL FEATURE IMPORTANCE (Gini impurity & Permutation importance). Note: Represents statistical split contribution, NOT physical RF causality."
        >
          <div className="grid gap-6 md:grid-cols-2">
            <div>
              <h4 className="text-xs font-bold text-slate-700 dark:text-slate-300 font-mono mb-3">
                Gini Impurity Importance
              </h4>
              <div className="space-y-2">
                {auditData?.phase_12_feature_importance?.gini_importance &&
                  Object.entries(auditData.phase_12_feature_importance.gini_importance)
                    .sort((a: any, b: any) => b[1] - a[1])
                    .slice(0, 8)
                    .map(([feat, val]: [string, any]) => (
                      <div key={feat} className="text-xs font-mono">
                        <div className="flex justify-between text-[11px] text-slate-600 dark:text-slate-400 mb-1">
                          <span>{feat}</span>
                          <span className="font-bold text-purple-600 dark:text-purple-400">{(val * 100).toFixed(2)}%</span>
                        </div>
                        <div className="h-1.5 w-full bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-purple-600 rounded-full"
                            style={{ width: `${Math.min(100, val * 500)}%` }}
                          />
                        </div>
                      </div>
                    ))}
              </div>
            </div>

            <div>
              <h4 className="text-xs font-bold text-slate-700 dark:text-slate-300 font-mono mb-3">
                Permutation Importance (ROC-AUC Impact)
              </h4>
              <div className="space-y-2">
                {auditData?.phase_12_feature_importance?.permutation_importance &&
                  Object.entries(auditData.phase_12_feature_importance.permutation_importance)
                    .sort((a: any, b: any) => b[1] - a[1])
                    .slice(0, 8)
                    .map(([feat, val]: [string, any]) => (
                      <div key={feat} className="text-xs font-mono">
                        <div className="flex justify-between text-[11px] text-slate-600 dark:text-slate-400 mb-1">
                          <span>{feat}</span>
                          <span className="font-bold text-indigo-600 dark:text-indigo-400">{val >= 0 ? `+${val.toFixed(4)}` : val.toFixed(4)}</span>
                        </div>
                        <div className="h-1.5 w-full bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-indigo-600 rounded-full"
                            style={{ width: `${Math.max(5, Math.abs(val) * 2000)}%` }}
                          />
                        </div>
                      </div>
                    ))}
              </div>
            </div>
          </div>
        </Panel>
      </div>

      {/* SECTION 3: GROUND-TRUTH ASSESSMENT & IMPROVEMENT PLAN */}
      <div className="space-y-4 pt-4">
        <div className="flex items-center gap-2 border-b border-amber-200 dark:border-amber-900/50 pb-2">
          <AlertTriangle className="size-5 text-amber-600 dark:text-amber-400" />
          <h3 className="text-base font-bold text-slate-900 dark:text-slate-100">
            LIMITATIONS & FUTURE DATASET IMPROVEMENT PLAN
          </h3>
        </div>

        {/* Phase 7 Scientific Decision Callout */}
        <div className="rounded-xl border border-red-300 dark:border-red-900 bg-red-50/90 dark:bg-red-950/30 p-5 space-y-2">
          <h4 className="text-sm font-bold text-red-900 dark:text-red-200 flex items-center gap-2">
            <Info className="size-4 text-red-600" />
            Phase 7 Scientific Decision: OPTION C — NO
          </h4>
          <p className="text-xs font-mono text-red-950 dark:text-red-100 leading-relaxed p-3 rounded bg-white/80 dark:bg-slate-900/80 border border-red-200 dark:border-red-900">
            {auditData?.phase_7_dataset_capability_decision?.statement ||
              "The current dataset non-power I/Q features do not contain sufficient independent predictive information to predict the inferred activity target without circular signal power leakage."}
          </p>
        </div>

        {/* Phase 10 Ground-Truth Improvement Plan */}
        <Panel
          title="Phase 10 — Ground-Truth Improvement Requirements for Future Datasets"
          subtitle="Specification of dataset requirements needed to establish independently verified spectrum ground-truth."
        >
          <div className="space-y-3 text-xs font-sans">
            <p className="text-slate-600 dark:text-slate-400">
              To solve the ground-truth limitation in future research iterations, a new RF dataset must provide:
            </p>
            <ul className="space-y-2">
              {auditData?.phase_10_ground_truth_improvement_plan?.requirements?.map((req: string, rIdx: number) => (
                <li key={rIdx} className="flex items-start gap-2 text-slate-800 dark:text-slate-200">
                  <CheckCircle2 className="size-4 text-emerald-500 shrink-0 mt-0.5" />
                  <span>{req}</span>
                </li>
              ))}
            </ul>
            <div className="p-3 rounded-lg bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 font-mono text-[11px] text-slate-500">
              Status: {auditData?.phase_10_ground_truth_improvement_plan?.current_availability || "NOT CURRENTLY AVAILABLE"}
            </div>
          </div>
        </Panel>
      </div>
    </div>
  );
}
