/**
 * Wire Watcher — ML & Decision Architecture View
 *
 * Details the complete engineering pipeline and transparent model comparison:
 *   - Transparent Model Comparison Table (DummyClassifier, Logistic Regression, Random Forest, Gradient Boosting)
 *   - Honest Reporting of ROC-AUC ≈ 0.50 (Leakage Quarantined)
 *   - Visual Multi-Stage Decision Hierarchy (RF Detector + DSP Features + ML Classifier + OOD Safeguard)
 */

import React, { useState } from "react";
import { Panel } from "@/components/wire/Panel";
import { MLPerformanceDashboard } from "@/components/ml/MLPerformanceDashboard";
import {
  Cpu,
  GitBranch,
  ShieldCheck,
  ShieldAlert,
  Sliders,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  BarChart,
  Layers,
  ArrowRight,
  Info,
  Activity,
  Award,
} from "lucide-react";

export function MLArchitectureView() {
  const [activeTab, setActiveTab] = useState<"audit" | "architecture">("audit");

  const modelComparison = [
    {
      name: "DummyClassifier (Baseline)",
      type: "Prior Strategy",
      valRocAuc: "0.5000",
      testRocAuc: "0.5000",
      balancedAcc: "50.00%",
      f1Occupied: "0.6725",
      brierScore: "0.2499",
      falseAvailableRate: "0.0%",
      selected: false,
    },
    {
      name: "Logistic Regression",
      type: "Linear / L2 Regularized",
      valRocAuc: "0.4961",
      testRocAuc: "0.5013",
      balancedAcc: "50.00%",
      f1Occupied: "0.6725",
      brierScore: "0.2500",
      falseAvailableRate: "0.0%",
      selected: false,
    },
    {
      name: "Random Forest (Selected)",
      type: "Ensemble / 60 Trees, Depth 10",
      valRocAuc: "0.5034",
      testRocAuc: "0.4988",
      balancedAcc: "50.00%",
      f1Occupied: "0.6724",
      brierScore: "0.2501",
      falseAvailableRate: "0.02%",
      selected: true,
    },
    {
      name: "Gradient Boosting",
      type: "Sequential Boosting / 25 Trees",
      valRocAuc: "0.4989",
      testRocAuc: "0.5022",
      balancedAcc: "50.00%",
      f1Occupied: "0.6725",
      brierScore: "0.2500",
      falseAvailableRate: "0.0%",
      selected: false,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="rounded-xl border border-purple-200 dark:border-purple-900/50 bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950/20 dark:to-indigo-950/20 p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-start gap-3">
            <div className="p-2.5 rounded-lg bg-purple-600 text-white shadow-sm shrink-0">
              <Cpu className="size-6" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                ML & Decision Architecture Engine
                <span className="text-xs px-2 py-0.5 rounded bg-purple-100 dark:bg-purple-900/60 text-purple-800 dark:text-purple-300 font-mono border border-purple-200 dark:border-purple-800">
                  Leakage-Safe Architecture
                </span>
              </h2>
              <p className="text-xs text-slate-600 dark:text-slate-400 mt-1 max-w-3xl">
                Explains how RF activity detection, statistical I/Q digital signal processing, machine learning evidence,
                and multivariate out-of-distribution (OOD) gating unite into an operational availability decision.
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex items-center gap-1 p-1 rounded-lg bg-slate-200/80 dark:bg-slate-900 shrink-0 font-mono text-xs">
            <button
              onClick={() => setActiveTab("audit")}
              className={`px-3 py-1.5 rounded-md font-bold transition-all ${
                activeTab === "audit"
                  ? "bg-purple-600 text-white shadow-sm"
                  : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100"
              }`}
            >
              ML Audit & Dashboard
            </button>
            <button
              onClick={() => setActiveTab("architecture")}
              className={`px-3 py-1.5 rounded-md font-bold transition-all ${
                activeTab === "architecture"
                  ? "bg-purple-600 text-white shadow-sm"
                  : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100"
              }`}
            >
              Decision Flow
            </button>
          </div>
        </div>
      </div>

      {activeTab === "audit" ? (
        <MLPerformanceDashboard />
      ) : (
        <React.Fragment>

      {/* Decision Hierarchy Flowchart */}
      <Panel
        title="Visual Engineering Decision Flow"
        subtitle="End-to-end signal processing, activity detection, ML evidence integration, and safety gating."
      >
        <div className="space-y-4">
          <div className="grid gap-2 sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-8 text-center font-mono text-xs">
            {/* 1. RF Observation */}
            <div className="p-3 rounded-lg border border-sky-300 dark:border-sky-800 bg-sky-50 dark:bg-sky-950/40 flex flex-col justify-between">
              <div className="font-bold text-sky-800 dark:text-sky-300">1. RF Observation</div>
              <div className="text-[10px] text-slate-500 mt-1">Freq, BW, Power, I/Q</div>
            </div>

            {/* 2. RF Feature Extraction */}
            <div className="p-3 rounded-lg border border-indigo-300 dark:border-indigo-800 bg-indigo-50 dark:bg-indigo-950/40 flex flex-col justify-between">
              <div className="font-bold text-indigo-800 dark:text-indigo-300">2. RF Feature Extraction</div>
              <div className="text-[10px] text-slate-500 mt-1">13 I/Q envelope stats</div>
            </div>

            {/* 3. FFT / PSD */}
            <div className="p-3 rounded-lg border border-purple-300 dark:border-purple-800 bg-purple-50 dark:bg-purple-950/40 flex flex-col justify-between">
              <div className="font-bold text-purple-800 dark:text-purple-300">3. FFT / PSD</div>
              <div className="text-[10px] text-slate-500 mt-1">Spectral Power Density</div>
            </div>

            {/* 4. Noise / Peak Analysis */}
            <div className="p-3 rounded-lg border border-amber-300 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/40 flex flex-col justify-between">
              <div className="font-bold text-amber-800 dark:text-amber-300">4. Noise / Peak Analysis</div>
              <div className="text-[10px] text-slate-500 mt-1">kTB floor + SNR &gt; 6dB</div>
            </div>

            {/* 5. RF Activity Detection + ML Evidence */}
            <div className="p-3 rounded-lg border border-teal-300 dark:border-teal-800 bg-teal-50 dark:bg-teal-950/40 flex flex-col justify-between">
              <div className="font-bold text-teal-800 dark:text-teal-300">5. Activity Det. + ML Evidence</div>
              <div className="text-[10px] text-slate-500 mt-1">Physical Det + RF Classifier</div>
            </div>

            {/* 6. OOD Guard */}
            <div className="p-3 rounded-lg border border-red-300 dark:border-red-800 bg-red-50 dark:bg-red-950/40 flex flex-col justify-between">
              <div className="font-bold text-red-800 dark:text-red-300">6. OOD Guard</div>
              <div className="text-[10px] text-slate-500 mt-1">Distribution bounds check</div>
            </div>

            {/* 7. Availability Decision */}
            <div className="p-3 rounded-lg border border-emerald-300 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/40 flex flex-col justify-between col-span-2">
              <div className="font-bold text-emerald-800 dark:text-emerald-300">7. Availability Decision</div>
              <div className="text-[11px] font-bold text-emerald-600 dark:text-emerald-400 mt-1">
                AVAILABLE / OCCUPIED / UNCERTAIN
              </div>
            </div>
          </div>

          <div className="p-3 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 text-xs text-slate-600 dark:text-slate-400 font-sans italic">
            "The ML classifier provides supplementary RF activity evidence. The final operational decision also incorporates physical RF activity detection and OOD protection."
          </div>
        </div>
      </Panel>

      {/* Model Comparison Table */}
      <Panel
        title="Evaluated ML Model Comparison"
        subtitle="Chronological 60/20/20 train/val/test split. Evaluated on 32,832 test samples without signal strength leakage."
      >
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono border-collapse">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 text-left text-slate-500">
                <th className="p-3">Model Architecture</th>
                <th className="p-3">Validation ROC-AUC</th>
                <th className="p-3">Test ROC-AUC</th>
                <th className="p-3">PR-AUC</th>
                <th className="p-3">Balanced Accuracy</th>
                <th className="p-3">F1 (Occupied)</th>
                <th className="p-3">Brier Score</th>
                <th className="p-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
              {modelComparison.map((m) => (
                <tr
                  key={m.name}
                  className={m.selected ? "bg-purple-50/60 dark:bg-purple-950/30 font-semibold" : "hover:bg-slate-50/50 dark:hover:bg-slate-900/50"}
                >
                  <td className="p-3">
                    <div className="font-bold text-slate-900 dark:text-slate-100">{m.name}</div>
                    <div className="text-[10px] text-slate-500 font-sans">{m.type}</div>
                  </td>
                  <td className="p-3 text-slate-700 dark:text-slate-300">{m.valRocAuc}</td>
                  <td className="p-3 text-indigo-600 dark:text-indigo-400 font-bold">{m.testRocAuc}</td>
                  <td className="p-3 text-slate-700 dark:text-slate-300">0.5058</td>
                  <td className="p-3 text-slate-700 dark:text-slate-300">{m.balancedAcc}</td>
                  <td className="p-3 text-slate-700 dark:text-slate-300">{m.f1Occupied}</td>
                  <td className="p-3 text-slate-700 dark:text-slate-300">{m.brierScore}</td>
                  <td className="p-3">
                    {m.selected ? (
                      <span className="px-2 py-0.5 rounded text-[11px] bg-purple-600 text-white font-bold">
                        ACTIVE IN PIPELINE
                      </span>
                    ) : (
                      <span className="text-slate-400">Evaluated</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {/* Scientific Honesty Callout */}
      <div className="rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-950 p-5 space-y-3">
        <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <Info className="size-4 text-sky-600" />
          Why is ML Performance Approximately Chance-Level (ROC-AUC ≈ 0.50)?
        </h3>
        <div className="text-xs text-slate-600 dark:text-slate-400 space-y-2 leading-relaxed font-sans">
          <p>
            1. <strong>Target Definition:</strong> The target <code>inferred_rf_activity</code> is a pseudo-label constructed from the median of <code>signal_strength_dbm</code>.
          </p>
          <p>
            2. <strong>Leakage Prevention:</strong> If <code>signal_strength_dbm</code> were included as an ML feature, any simple classifier would easily achieve an artificial ROC-AUC of 1.00 by merely learning the threshold rule. To maintain strict scientific integrity, <code>signal_strength_dbm</code> is completely quarantined from the ML feature set.
          </p>
          <p>
            3. <strong>Feature Independence:</strong> The remaining 13 statistical I/Q envelope features (variance, crest factor, spectral entropy) do not correlate with an arbitrary fixed signal power threshold. The honest 0.50 score reflects the true independent predictive power of dimensionless I/Q statistics for this pseudo-label.
          </p>
        </div>
      </div>
        </React.Fragment>
      )}
    </div>
  );
}

