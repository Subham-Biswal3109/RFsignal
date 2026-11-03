/**
 * Wire Watcher — Automatic RF Technical Report Generator (Section 19)
 *
 * Renders an end-to-end, technically defensible 10-section RF Analysis Report.
 */

import React, { useState, useEffect } from "react";
import { Panel } from "@/components/wire/Panel";
import { FileText, Download, ShieldAlert, CheckCircle2, Info, AlertTriangle, Sparkles } from "lucide-react";
import { apiFetch } from "@/lib/api";
import type { RFTechnicalReport } from "@/types/wire-watcher";

export function RFTechnicalReportView() {
  const [report, setReport] = useState<RFTechnicalReport | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchReport = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiFetch<RFTechnicalReport>("/api/report/rf-analysis?frequency_mhz=120.0&signal_power_dbm=-75.0&noise_floor_dbm=-100.0");
      setReport(data);
    } catch (err: any) {
      setError(err.message || "Failed to load report");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReport();
  }, []);

  const handlePrintDownload = () => {
    window.print();
  };

  if (loading) {
    return (
      <div className="p-8 text-center bg-slate-900 border border-slate-800 rounded-xl text-slate-400 font-mono">
        Generating Automatic RF Technical Analysis Report...
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="p-6 bg-rose-950/80 border border-rose-800 rounded-xl text-rose-300 text-xs">
        {error || "Report unavailable"}
      </div>
    );
  }

  const sec = report.sections;

  return (
    <div className="space-y-6">
      {/* Action Header */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <FileText className="w-5 h-5 text-cyan-400" />
            Automatic RF Technical Analysis Report
          </h2>
          <p className="text-xs text-slate-400 font-mono">
            Report ID: {report.report_id} | Timestamp: {new Date(report.timestamp).toLocaleString()}
          </p>
        </div>
        <button
          onClick={handlePrintDownload}
          className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-xs rounded-lg transition shadow flex items-center gap-2"
        >
          <Download className="w-4 h-4" />
          <span>Export Technical Report</span>
        </button>
      </div>

      {/* 10-Section Report Container */}
      <div className="bg-slate-950 border border-slate-800 rounded-xl p-6 space-y-6 text-slate-200 font-sans shadow-2xl">
        {/* Title Block */}
        <div className="border-b border-slate-800 pb-4">
          <div className="text-xs font-mono text-cyan-400 uppercase tracking-widest font-bold">
            WIRE WATCHER — RF SIGNAL ALLOCATION & ENGINEERING INTELLIGENCE
          </div>
          <h1 className="text-xl font-bold text-white mt-1">
            Comprehensive RF Spectrum Analysis & Availability Audit
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Defensible engineering report detailing observations, DSP spectral estimation, ML inference, OOD safety, availability decisioning, channel allocation, and system limitations.
          </p>
        </div>

        {/* Section 1: Observation */}
        <div className="space-y-2">
          <h3 className="text-sm font-bold text-cyan-300 font-mono uppercase tracking-wider">1. Observation Summary</h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-900 p-3 rounded-lg border border-slate-800 text-xs font-mono">
            <div>Source Type: <span className="text-white font-bold">{sec["1_observation"].source_type}</span></div>
            <div>Center Freq: <span className="text-white font-bold">{sec["1_observation"].center_frequency_mhz} MHz</span></div>
            <div>Bandwidth: <span className="text-white font-bold">{sec["1_observation"].bandwidth_khz} kHz</span></div>
            <div>Signal Power: <span className="text-white font-bold">{sec["1_observation"].signal_power_dbm} dBm</span></div>
          </div>
        </div>

        {/* Section 2: DSP Analysis */}
        <div className="space-y-2">
          <h3 className="text-sm font-bold text-cyan-300 font-mono uppercase tracking-wider">2. DSP & Spectral Estimation</h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-900 p-3 rounded-lg border border-slate-800 text-xs font-mono">
            <div>Sample Rate: <span className="text-white">{sec["2_dsp"].sample_rate_mhz} MS/s</span></div>
            <div>FFT Window: <span className="text-white">{sec["2_dsp"].fft_window} ({sec["2_dsp"].fft_size})</span></div>
            <div>Noise Floor: <span className="text-white">{sec["2_dsp"].noise_floor_dbm} dBm</span></div>
            <div>Detected Peaks: <span className="text-white">{sec["2_dsp"].detected_peaks_count}</span></div>
          </div>
        </div>

        {/* Section 3: RF Engineering Calculations */}
        <div className="space-y-2">
          <h3 className="text-sm font-bold text-cyan-300 font-mono uppercase tracking-wider">3. RF Engineering Parameters</h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-900 p-3 rounded-lg border border-slate-800 text-xs font-mono">
            <div>Wavelength (λ): <span className="text-amber-300 font-bold">{sec["3_rf_engineering"].wavelength_m} m</span></div>
            <div>SNR: <span className="text-emerald-300 font-bold">{sec["3_rf_engineering"].snr_db} dB</span></div>
            <div>Est. Thermal Noise: <span className="text-white">{sec["3_rf_engineering"].thermal_noise_floor_est_dbm} dBm</span></div>
            <div>Nyquist Min Rate: <span className="text-white">{sec["3_rf_engineering"].nyquist_min_sample_rate_khz} kHz</span></div>
          </div>
        </div>

        {/* Section 4: ML Evidence */}
        <div className="space-y-2">
          <h3 className="text-sm font-bold text-cyan-300 font-mono uppercase tracking-wider">4. Machine Learning Evidence</h3>
          <div className="p-3 bg-slate-900 rounded-lg border border-slate-800 text-xs font-mono space-y-1">
            <div className="flex justify-between">
              <span>Model Architecture: <strong className="text-white">{sec["4_ml"].model_architecture} ({sec["4_ml"].model_status})</strong></span>
              <span>Predicted Activity Prob: <strong className="text-amber-300">{sec["4_ml"].predicted_activity_probability}</strong></span>
            </div>
            <div className="text-slate-400 text-[11px] mt-1">
              Target Leakage Controls: <span className="text-emerald-400 font-semibold">{sec["4_ml"].target_leakage_control}</span>
            </div>
            <div className="text-slate-400 text-[11px]">
              Reported Test ROC-AUC: <span className="text-amber-300 font-semibold">{sec["4_ml"].reported_test_roc_auc}</span>
            </div>
          </div>
        </div>

        {/* Section 5: OOD Safety Guard */}
        <div className="space-y-2">
          <h3 className="text-sm font-bold text-cyan-300 font-mono uppercase tracking-wider">5. OOD Safety Guard Status</h3>
          <div className="p-3 bg-slate-900 rounded-lg border border-slate-800 text-xs font-mono flex items-center justify-between">
            <div>OOD Status: <span className="font-bold text-emerald-400">{sec["5_ood"].ood_status}</span></div>
            <div className="text-slate-400 text-[11px]">Reason: {sec["5_ood"].reason}</div>
          </div>
        </div>

        {/* Section 6: Availability Decision */}
        <div className="space-y-2">
          <h3 className="text-sm font-bold text-cyan-300 font-mono uppercase tracking-wider">6. Spectrum Availability Assessment</h3>
          <div className="p-3 bg-slate-900 rounded-lg border border-slate-800 text-xs font-mono space-y-1">
            <div className="text-sm font-bold text-emerald-400">Decision: {sec["6_availability"].decision}</div>
            <div className="text-slate-300 text-[11px]">{sec["6_availability"].explanation}</div>
          </div>
        </div>

        {/* Section 7: Channel Allocation */}
        <div className="space-y-2">
          <h3 className="text-sm font-bold text-cyan-300 font-mono uppercase tracking-wider">7. Channel Allocation Recommendation</h3>
          <div className="p-3 bg-slate-900 rounded-lg border border-slate-800 text-xs font-mono space-y-2">
            <div className="text-slate-200">{sec["7_allocation"].reason}</div>
            {sec["7_allocation"].recommended_candidate && (
              <div className="p-2 bg-purple-950/60 border border-purple-800 rounded text-purple-200">
                Recommended Candidate: <strong>{sec["7_allocation"].recommended_candidate.candidate_id}</strong> ({sec["7_allocation"].recommended_candidate.center_freq_mhz} MHz) | Score: {sec["7_allocation"].recommended_candidate.score}/100
              </div>
            )}
          </div>
        </div>

        {/* Section 8: Event Analytics */}
        <div className="space-y-2">
          <h3 className="text-sm font-bold text-cyan-300 font-mono uppercase tracking-wider">8. Event Analytics & Occupancy</h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-900 p-3 rounded-lg border border-slate-800 text-xs font-mono">
            <div>Total Logged Events: <span className="text-white">{sec["8_events"].total_events_logged}</span></div>
            <div>Avg Event Duration: <span className="text-white">{sec["8_events"].average_event_duration_s}s</span></div>
            <div>Max Event Duration: <span className="text-white">{sec["8_events"].maximum_event_duration_s}s</span></div>
            <div>Channel Utilization: <span className="text-emerald-300 font-bold">{sec["8_events"].channel_utilization_percentage}%</span></div>
          </div>
        </div>

        {/* Section 9: Data Provenance */}
        <div className="space-y-2">
          <h3 className="text-sm font-bold text-cyan-300 font-mono uppercase tracking-wider">9. Measurement Data Provenance</h3>
          <div className="p-3 bg-slate-900 rounded-lg border border-slate-800 text-xs font-mono space-y-1 text-slate-300">
            <div>REAL_MEASURED: RTL-SDR hardware capture (when connected)</div>
            <div>IQ_REPLAY: Chronological IQ binary replay</div>
            <div>DATASET: Historical RF Dataset (164,160 records)</div>
            <div>DERIVED: DSP FFT/PSD, SNR, thermal noise, channel scores, utilization %</div>
            <div>INFERRED_RF_ACTIVITY: Combined RF Activity & Availability decision</div>
            <div>ML_EVIDENCE: Random Forest inference probability</div>
            <div>APPLICATION_METADATA: User configuration, UI location, OOD thresholds</div>
          </div>
        </div>

        {/* Section 10: Limitations & Disclaimers */}
        <div className="space-y-2 pt-2 border-t border-slate-800">
          <h3 className="text-sm font-bold text-rose-400 font-mono uppercase tracking-wider flex items-center gap-1.5">
            <AlertTriangle className="w-4 h-4 text-rose-400" />
            10. System Limitations & Disclaimers
          </h3>
          <ul className="space-y-1.5 list-disc list-inside text-xs text-slate-400 bg-slate-900/90 p-4 rounded-lg border border-slate-800 font-sans">
            {sec["10_limitations"].map((item: string, idx: number) => (
              <li key={idx} className="leading-relaxed">{item}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
