/**
 * Wire Watcher — Interactive System Flow Architecture & Decision Pipeline
 *
 * ECE Final-Year Project Review & Viva Instrument Visualization:
 * Displays the complete engineering pipeline in exact processing order:
 *   1. RF Observation
 *   2. DSP / Thermal Noise Analysis (Primary Physical Detector)
 *   3. Feature Extraction (13-Moment Envelope Conversion)
 *   4. Random Forest ML (Supporting Evidence)
 *   5. OOD Safety Guard (Safety Override Gate)
 *   6. Decision Arbitration (Final Availability Decision Fusion)
 *   7. Channel Allocation (100-Point Deterministic Engineering Scoring)
 *   8. Final Recommendation (Output Beacon)
 *
 * System Hierarchy & Integrity:
 *   - DSP = Primary Physical Detection (Thermal noise floor kTB, SNR > 6 dB, energy detection)
 *   - ML = Supplementary Evidence (Random Forest 60 trees, strictly quarantined from signal power)
 *   - OOD = Safety Override (1st-99th percentile bounds, standardized Mahalanobis distance)
 *   - Decision Arbitration = Final Availability Decision (AVAILABLE / OCCUPIED / UNCERTAIN)
 *   - Channel Allocation = Deterministic Engineering Scoring (100-pt formula, guard band protection)
 */

import React, { useState, useEffect, useRef } from "react";
import {
  Radio,
  Activity,
  Sliders,
  Cpu,
  ShieldCheck,
  ShieldAlert,
  GitBranch,
  Layers,
  Award,
  Play,
  Pause,
  RotateCcw,
  ChevronRight,
  ChevronLeft,
  Info,
  CheckCircle2,
  AlertTriangle,
  Zap,
  Clock,
  RefreshCw,
  SlidersHorizontal,
  ArrowRight,
  Shield,
  FileText,
  CornerDownRight,
} from "lucide-react";
import { cn } from "@/lib/utils";

export type FlowScenario = "available" | "occupied" | "ood_override";
export type StageStatusType = "pending" | "processing" | "complete" | "uncertain";

interface StageData {
  id: number;
  slug: string;
  name: string;
  subtitle: string;
  hierarchyRole: string;
  hierarchyBadgeColor: string;
  provenance: string;
  borderActive: string;
  bgActive: string;
  textActive: string;
  glowRgba: string;
  icon: React.ComponentType<{ className?: string }>;
  metricLabel: string;
  getMetricValue: (scenario: FlowScenario) => string;
  getMetricStatus: (scenario: FlowScenario) => { label: string; color: string };
  formula: string;
  formulaSubstitution: (scenario: FlowScenario) => string;
  whatItDoes: string;
  inputDescription: string[];
  outputDescription: (scenario: FlowScenario) => string[];
  systemRole: string;
  engineeringInsight: string;
}

const STAGES: StageData[] = [
  {
    id: 1,
    slug: "rf_observation",
    name: "1. RF Observation",
    subtitle: "Incoming RF Measurement & I/Q Stream",
    hierarchyRole: "Raw Sensor Input",
    hierarchyBadgeColor: "bg-sky-500/20 text-sky-300 border-sky-500/40",
    provenance: "REAL_MEASURED",
    borderActive: "border-sky-400 shadow-sky-500/30",
    bgActive: "bg-sky-950/40",
    textActive: "text-sky-400",
    glowRgba: "rgba(56, 189, 248, 0.4)",
    icon: Radio,
    metricLabel: "Measured RF Power",
    getMetricValue: (s) => (s === "occupied" ? "-52.4 dBm" : s === "ood_override" ? "-18.0 dBm" : "-75.0 dBm"),
    getMetricStatus: (s) =>
      s === "occupied"
        ? { label: "High Power Detected", color: "text-rose-400" }
        : s === "ood_override"
        ? { label: "Receiver Saturated", color: "text-amber-400" }
        : { label: "Nominal Background", color: "text-sky-400" },
    formula: "λ = c / f_c",
    formulaSubstitution: (s) =>
      s === "ood_override"
        ? "λ = 299,792,458 / (450.0 × 10⁶) = 0.666 m (UHF out-of-band)"
        : "λ = 299,792,458 / (120.0 × 10⁶) = 2.498 m (VHF band)",
    whatItDoes:
      "Digitizes incoming RF spectrum signals from the antenna/SDR front-end. Ingests center frequency, channel bandwidth, receiver power, and raw time-domain complex I/Q samples.",
    inputDescription: [
      "Center Frequency: 120.000 MHz (VHF)",
      "Channel Bandwidth: 200.0 kHz",
      "Antenna RX Signal Power: -75.0 dBm",
      "Time-Domain I/Q Sample Stream (100 samples, 2.048 MS/s)",
    ],
    outputDescription: (s) => [
      `Raw Power Stream: ${s === "occupied" ? "-52.4 dBm" : s === "ood_override" ? "-18.0 dBm" : "-75.0 dBm"}`,
      `Electromagnetic Wavelength λ = ${s === "ood_override" ? "0.666 m" : "2.498 m"}`,
      "100 In-phase (I) and Quadrature (Q) complex data arrays",
    ],
    systemRole:
      "Captures the true physical RF environment without distortion. Preserves raw power for deterministic physical detection while quarantining it from ML.",
    engineeringInsight:
      "Physical RF measurements must enter cleanly without artificial normalization to allow the downstream DSP stage to calculate absolute thermal margins.",
  },
  {
    id: 2,
    slug: "dsp_detection",
    name: "2. DSP / Thermal Noise",
    subtitle: "Thermal Floor, SNR & Physical Detection",
    hierarchyRole: "PRIMARY PHYSICAL DETECTOR",
    hierarchyBadgeColor: "bg-indigo-500/20 text-indigo-300 border-indigo-500/40",
    provenance: "DERIVED_PHYSICS",
    borderActive: "border-indigo-400 shadow-indigo-500/30",
    bgActive: "bg-indigo-950/40",
    textActive: "text-indigo-400",
    glowRgba: "rgba(99, 102, 241, 0.4)",
    icon: Activity,
    metricLabel: "Evaluated SNR",
    getMetricValue: (s) => (s === "occupied" ? "+68.5 dB" : s === "ood_override" ? "+102.9 dB" : "+45.9 dB"),
    getMetricStatus: (s) =>
      s === "occupied"
        ? { label: "ENERGY DETECTED (>6 dB)", color: "text-rose-400" }
        : s === "ood_override"
        ? { label: "ANOMALOUS ENERGY", color: "text-amber-400" }
        : { label: "CLEAR MARGIN (>6 dB)", color: "text-emerald-400" },
    formula: "P_noise = 10·log10(k_B·T·B) ; SNR = P_rx - P_noise",
    formulaSubstitution: (s) =>
      s === "occupied"
        ? "P_noise = -120.96 dBm ; SNR = -52.4 - (-120.96) = +68.56 dB (DETECTOR ACTIVE)"
        : "P_noise = -120.96 dBm ; SNR = -75.0 - (-120.96) = +45.96 dB (DETECTOR CLEAN)",
    whatItDoes:
      "Computes the fundamental Johnson-Nyquist thermal noise floor (kTB at 290 K) and evaluates true Signal-to-Noise Ratio (SNR). Acts as the primary physical authority for declaring whether RF energy is physically present.",
    inputDescription: [
      "Measured RX Power: P_rx (dBm)",
      "Channel Bandwidth: B = 200 kHz",
      "Standard Temperature: T = 290.0 K",
      "Boltzmann Constant: k_B = 1.380649 × 10⁻²³ J/K",
    ],
    outputDescription: (s) => [
      "Thermal Noise Floor P_noise = -120.96 dBm",
      `Calculated SNR = ${s === "occupied" ? "+68.56 dB" : s === "ood_override" ? "+102.96 dB" : "+45.96 dB"}`,
      `Physical Detection Evidence: ${s === "occupied" ? "DETECTED (Energy above threshold)" : "NOT_DETECTED (Clean)"}`,
    ],
    systemRole:
      "PRIMARY PHYSICAL DETECTOR: Physical electromagnetic laws govern spectrum occupancy. If energy is detected on the channel above thermal margin, DSP flags it as active.",
    engineeringInsight:
      "DSP serves as the senior authority in the decision hierarchy. No statistical model can override the physical presence of high-energy electromagnetic transmission.",
  },
  {
    id: 3,
    slug: "feature_extraction",
    name: "3. Feature Extraction",
    subtitle: "13-Moment Statistical Envelope Transformation",
    hierarchyRole: "Dimensionless Features",
    hierarchyBadgeColor: "bg-purple-500/20 text-purple-300 border-purple-500/40",
    provenance: "DERIVED_STATISTICAL",
    borderActive: "border-purple-400 shadow-purple-500/30",
    bgActive: "bg-purple-950/40",
    textActive: "text-purple-400",
    glowRgba: "rgba(168, 85, 247, 0.4)",
    icon: Sliders,
    metricLabel: "Entropy & Crest Factor",
    getMetricValue: (s) => (s === "ood_override" ? "H=0.12, CF=12.4" : "H=0.88, CF=2.41"),
    getMetricStatus: (s) =>
      s === "ood_override"
        ? { label: "OUTLIER ENVELOPE", color: "text-amber-400" }
        : { label: "13 Features Extracted", color: "text-purple-300" },
    formula: "H = -∑ p_k · log2(p_k) ; CF = max|z| / RMS",
    formulaSubstitution: () =>
      "RMS=0.707 V, Var=0.042, CF=2.41, Phase κ=0.12, Entropy=0.884, PeakRatio=0.31",
    whatItDoes:
      "Transforms raw time-domain complex I/Q envelope samples into 13 dimensionless statistical features (RMS, variance, crest factor, phase concentration, spectral entropy, and peak ratio). Quarantines signal strength to prevent target leakage.",
    inputDescription: [
      "100 Complex I/Q Time Samples: z[n] = I[n] + j·Q[n]",
      "Center Frequency: f_c (MHz)",
      "Channel Bandwidth: B (kHz)",
      "STRICT QUARANTINE: signal_strength_dbm is excluded",
    ],
    outputDescription: (s) => [
      `Envelope RMS Magnitude: ${s === "ood_override" ? "4.12 V" : "0.707 V"}`,
      `Crest Factor: ${s === "ood_override" ? "12.40 (Spike Outlier)" : "2.41"}`,
      `Spectral Entropy: ${s === "ood_override" ? "0.12 (Concentrated)" : "0.88 (Dispersed)"}`,
      "13-feature normalized vector ready for Random Forest",
    ],
    systemRole:
      "Feature engineering layer. Extracts waveform modulation shape and spectral distribution without leaking circular threshold rules into machine learning.",
    engineeringInsight:
      "By isolating signal power from this 13-feature vector, the ML model is prevented from trivially memorizing the power threshold, maintaining rigorous scientific integrity.",
  },
  {
    id: 4,
    slug: "ml_random_forest",
    name: "4. Random Forest ML",
    subtitle: "60-Tree Ensemble Inference",
    hierarchyRole: "SUPPORTING EVIDENCE (NOT FINAL AUTHORITY)",
    hierarchyBadgeColor: "bg-fuchsia-500/20 text-fuchsia-300 border-fuchsia-500/40",
    provenance: "ML_EVIDENCE",
    borderActive: "border-fuchsia-400 shadow-fuchsia-500/30",
    bgActive: "bg-fuchsia-950/40",
    textActive: "text-fuchsia-400",
    glowRgba: "rgba(217, 70, 239, 0.4)",
    icon: Cpu,
    metricLabel: "ML Activity Probability",
    getMetricValue: (s) => (s === "occupied" ? "88.4%" : s === "ood_override" ? "94.2%" : "14.2%"),
    getMetricStatus: (s) =>
      s === "occupied"
        ? { label: "P(Occupied) = 0.884", color: "text-rose-400" }
        : s === "ood_override"
        ? { label: "High Confidence on OOD", color: "text-amber-400" }
        : { label: "P(Occupied) = 0.142", color: "text-emerald-400" },
    formula: "P(Occupied) = (1 / M) ∑ tree_m(x)",
    formulaSubstitution: (s) =>
      s === "occupied"
        ? "M=60 trees, 53 voted OCCUPIED → P = 53/60 = 88.33%"
        : s === "ood_override"
        ? "M=60 trees, 56 voted OCCUPIED → P = 56/60 = 93.33% (UNRELIABLE OOD)"
        : "M=60 trees, 9 voted OCCUPIED → P = 9/60 = 15.00%",
    whatItDoes:
      "Passes the 13-feature vector into an ensemble of 60 decision trees (max depth 10). Trees evaluate in parallel and aggregate their votes to produce an activity probability P(Occupied).",
    inputDescription: [
      "13-feature statistical I/Q vector",
      "Model: Frozen Random Forest v2 (60 trees, depth 10)",
      "Threshold for occupied declaration: 0.50",
    ],
    outputDescription: (s) => [
      `Activity Probability: P(Occupied) = ${s === "occupied" ? "88.4%" : s === "ood_override" ? "94.2%" : "14.2%"}`,
      `Tree Votes: ${s === "occupied" ? "53/60 trees voted Occupied" : s === "ood_override" ? "56/60 trees voted Occupied" : "9/60 trees voted Occupied"}`,
      "Benchmark: Test ROC-AUC = 0.7000 (Top Model in Comparison)",
    ],
    systemRole:
      "SUPPORTING EVIDENCE: ML provides statistical advisory context to assist engineering judgment. It does NOT possess unilateral authority to allocate spectrum.",
    engineeringInsight:
      "Notice how an ML model alone might show 94% confidence even when fed invalid out-of-distribution data. That is precisely why Wire Watcher places an OOD safety guard next!",
  },
  {
    id: 5,
    slug: "ood_safety_guard",
    name: "5. OOD Safety Guard",
    subtitle: "Multivariate Boundary & Distance Verification",
    hierarchyRole: "SAFETY OVERRIDE GATE",
    hierarchyBadgeColor: "bg-amber-500/20 text-amber-300 border-amber-500/40",
    provenance: "SAFETY_GATE",
    borderActive: "border-amber-400 shadow-amber-500/30",
    bgActive: "bg-amber-950/40",
    textActive: "text-amber-400",
    glowRgba: "rgba(245, 158, 11, 0.4)",
    icon: ShieldCheck,
    metricLabel: "Standardized Distance",
    getMetricValue: (s) => (s === "ood_override" ? "D = 5.84 (>3.0 LIMIT)" : "D = 1.34 (≤3.0 PASS)"),
    getMetricStatus: (s) =>
      s === "ood_override"
        ? { label: "OOD TRIGGERED: OVERRIDE", color: "text-amber-400 font-bold" }
        : { label: "IN-DISTRIBUTION: PASS", color: "text-emerald-400" },
    formula: "D = √ ∑ ((x_i - μ_i) / σ_i)²  ;  p1 ≤ x_i ≤ p99",
    formulaSubstitution: (s) =>
      s === "ood_override"
        ? "D = 5.84 > 3.0 threshold ; Crest factor 12.4 > p99(3.82) → SAFETY TRIGGER"
        : "D = 1.34 ≤ 3.0 threshold ; All features inside [p1, p99] → PASS",
    whatItDoes:
      "Performs multivariate checks against empirical training distributions: (1) 1st-99th percentile bounds per feature, (2) diagonal Mahalanobis distance check, and (3) trained frequency coverage. If any check fails, safety override forces decision to UNCERTAIN.",
    inputDescription: [
      "13-feature input vector and raw RF observation",
      "Training 1st-99th percentile bounds [p1, p99]",
      "Distance threshold D_max = 3.00 standard deviations",
      "Calibrated frequency list: [70, 90, 100, 120, 140, 160 MHz]",
    ],
    outputDescription: (s) => [
      `Distance Metric D = ${s === "ood_override" ? "5.84 (> 3.00 Threshold)" : "1.34 (≤ 3.00 Threshold)"}`,
      `OOD Status: ${s === "ood_override" ? "OOD TRIGGERED (Anomaly detected)" : "IN-DISTRIBUTION (All checks passed)"}`,
      `Safety Override State: ${s === "ood_override" ? "ACTIVE → Forcing UNCERTAIN fallback" : "INACTIVE → Passing evidence forward"}`,
    ],
    systemRole:
      "SAFETY OVERRIDE GATE: Prevents unauthorized spectrum allocation when operating outside verified operational bounds. Forces UNCERTAIN if conditions are ambiguous.",
    engineeringInsight:
      "Safety First: If the RF environment deviates from calibrated training conditions, the system rejects automated assumptions and safely forces UNCERTAIN.",
  },
  {
    id: 6,
    slug: "decision_arbitration",
    name: "6. Decision Arbitration",
    subtitle: "Evidence Fusion & Operational State",
    hierarchyRole: "FINAL AVAILABILITY DECISION",
    hierarchyBadgeColor: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
    provenance: "EVIDENCE_FUSION",
    borderActive: "border-emerald-400 shadow-emerald-500/30",
    bgActive: "bg-emerald-950/40",
    textActive: "text-emerald-400",
    glowRgba: "rgba(16, 185, 129, 0.4)",
    icon: GitBranch,
    metricLabel: "Fused Decision",
    getMetricValue: (s) => (s === "occupied" ? "OCCUPIED" : s === "ood_override" ? "UNCERTAIN" : "AVAILABLE"),
    getMetricStatus: (s) =>
      s === "occupied"
        ? { label: "OCCUPIED (Spectrum In Use)", color: "text-rose-400 font-bold" }
        : s === "ood_override"
        ? { label: "UNCERTAIN (OOD Safety Active)", color: "text-amber-400 font-bold" }
        : { label: "AVAILABLE (Clean Channel)", color: "text-emerald-400 font-bold" },
    formula: "Decision = F(DSP_primary, ML_support, OOD_gate)",
    formulaSubstitution: (s) =>
      s === "ood_override"
        ? "OOD = TRUE → Forces UNCERTAIN (prevents erroneous allocation)"
        : s === "occupied"
        ? "OOD = FALSE, DSP = DETECTED → OCCUPIED (Physical signal confirmed)"
        : "OOD = FALSE, DSP = NOT_DETECTED → AVAILABLE (Thermal headroom clear)",
    whatItDoes:
      "Fuses all three evidentiary streams: DSP physical detection (Primary Evidence), Random Forest probability (Supporting Evidence), and OOD guard status. Generates exactly one of three operational states: AVAILABLE, OCCUPIED, or UNCERTAIN.",
    inputDescription: [
      "DSP Physical Evidence: DETECTED / NOT_DETECTED",
      "ML Supporting Probability: P(Occupied) [0.0 to 1.0]",
      "OOD Safety Guard Flag: TRUE / FALSE",
    ],
    outputDescription: (s) => [
      `Operational Availability Decision: ${s === "occupied" ? "OCCUPIED" : s === "ood_override" ? "UNCERTAIN" : "AVAILABLE"}`,
      `Governing Decision Rule: ${s === "ood_override" ? "OOD == TRUE → Forced UNCERTAIN" : s === "occupied" ? "DSP == DETECTED → OCCUPIED" : "DSP == NOT_DETECTED → AVAILABLE"}`,
      `Confidence Rating: ${s === "occupied" ? "88.4%" : s === "ood_override" ? "42.0% (Capped)" : "85.8%"}`,
    ],
    systemRole:
      "FINAL AVAILABILITY DECISION: The conclusive spectrum sensing checkpoint. Only channels verified as AVAILABLE proceed to positive scoring in channel allocation.",
    engineeringInsight:
      "Decision Arbitration enforces evidence fusion. Physical evidence determines active presence; OOD enforces a fail-safe barrier.",
  },
  {
    id: 7,
    slug: "channel_allocation",
    name: "7. Channel Allocation",
    subtitle: "100-Point Deterministic Engineering Scoring",
    hierarchyRole: "DETERMINISTIC ENGINEERING SCORING",
    hierarchyBadgeColor: "bg-cyan-500/20 text-cyan-300 border-cyan-500/40",
    provenance: "ENGINEERING_ALGORITHM",
    borderActive: "border-cyan-400 shadow-cyan-500/30",
    bgActive: "bg-cyan-950/40",
    textActive: "text-cyan-400",
    glowRgba: "rgba(6, 182, 212, 0.4)",
    icon: Layers,
    metricLabel: "Candidate Channel Score",
    getMetricValue: (s) => (s === "occupied" ? "42.5 / 100" : s === "ood_override" ? "35.0 / 100" : "95.0 / 100"),
    getMetricStatus: (s) =>
      s === "occupied"
        ? { label: "BELOW THRESHOLD (<50)", color: "text-rose-400" }
        : s === "ood_override"
        ? { label: "OOD SAFETY PENALTY (-40)", color: "text-amber-400" }
        : { label: "HIGH CONFIDENCE (>50)", color: "text-emerald-400 font-bold" },
    formula: "Score = 100 - Δ_occupied - Δ_ood - Δ_uncertain - 1.5·Δ_noise - Δ_guard - Δ_interf",
    formulaSubstitution: (s) =>
      s === "occupied"
        ? "100 - 50 (Occupied) - 7.5 (Noise) = 42.5/100 (REJECTED)"
        : s === "ood_override"
        ? "100 - 40 (OOD) - 25 (Uncertain) = 35.0/100 (HELD FOR SAFETY)"
        : "100 - 0 (Available) - 5.0 (Guard Margin) = 95.0/100 (RECOMMENDED)",
    whatItDoes:
      "Divides analyzed frequency band into candidate channels, assigns 50 kHz guard band protective envelopes, and runs transparent 100-point penalty-deduction scoring. Completely deterministic engineering logic—never driven directly by ML.",
    inputDescription: [
      "Availability Decision from Stage 6",
      "Band Bounds: 70.0 to 160.0 MHz",
      "Channel Bandwidth: 200 kHz | Guard Band: 50 kHz",
      "Noise Floor Baseline: -100.0 dBm",
    ],
    outputDescription: (s) => [
      `Primary Channel Score: ${s === "occupied" ? "42.5 / 100 (Occupied penalty applied)" : s === "ood_override" ? "35.0 / 100 (OOD penalty applied)" : "95.0 / 100 (Full points passed)"}`,
      "Guard Band Analysis: Protected region margins verified",
      `Candidate Ranking: ${s === "occupied" ? "Alternate channels evaluated for rerouting" : s === "ood_override" ? "All channels held below 50 pt threshold" : "CH-001 ranked #1"}`,
    ],
    systemRole:
      "DETERMINISTIC ENGINEERING SCORING: Channel allocation is governed by rigorous engineering calculations, guard band clearance, and safety scoring.",
    engineeringInsight:
      "Notice: The Random Forest does NOT assign channels. Channel allocation is governed by transparent deterministic scoring rules.",
  },
  {
    id: 8,
    slug: "final_recommendation",
    name: "8. Final Recommendation",
    subtitle: "Recommended Operational Spectrum Channel",
    hierarchyRole: "ENGINEERING OUTPUT BEACON",
    hierarchyBadgeColor: "bg-teal-500/20 text-teal-300 border-teal-500/40",
    provenance: "FINAL_RECOMMENDATION",
    borderActive: "border-teal-400 shadow-teal-500/30",
    bgActive: "bg-teal-950/40",
    textActive: "text-teal-400",
    glowRgba: "rgba(20, 184, 166, 0.4)",
    icon: Award,
    metricLabel: "Recommended Frequency",
    getMetricValue: (s) =>
      s === "occupied" ? "CH-042 (82.400 MHz)" : s === "ood_override" ? "NO CANDIDATE (HOLD)" : "CH-001 (70.100 MHz)",
    getMetricStatus: (s) =>
      s === "occupied"
        ? { label: "REROUTED TO ALTERNATE", color: "text-sky-300" }
        : s === "ood_override"
        ? { label: "FAIL-SAFE TRANSMISSION HOLD", color: "text-amber-400 font-bold" }
        : { label: "RECOMMENDED CANDIDATE", color: "text-emerald-400 font-bold" },
    formula: "ArgMax_{c ∈ Candidates, Score_c ≥ 50} (Score_c)",
    formulaSubstitution: (s) =>
      s === "ood_override"
        ? "All candidate scores < 50.0 due to OOD penalty → SAFE HOLD"
        : s === "occupied"
        ? "CH-001 occupied (42.5); alternate CH-042 selected with Score 92.0"
        : "CH-001 selected: Center 70.100 MHz, Score 95.0, Guard SAFE_MARGIN",
    whatItDoes:
      "Emits the final recommended channel allocation to RF operators or automated cognitive radio transmitters. Displays center frequency, bandwidth, guard band clearance, and full mathematical audit trail.",
    inputDescription: [
      "Sorted Candidate Channel Scores from Stage 7",
      "Minimum Recommendation Threshold: Score ≥ 50.0",
      "Interference & Adjacent Channel Margin Check",
    ],
    outputDescription: (s) => [
      `Assigned Center Frequency: ${s === "ood_override" ? "NONE (Transmission Blocked)" : s === "occupied" ? "82.400 MHz (CH-042)" : "70.100 MHz (CH-001)"}`,
      `Recommendation Status: ${s === "ood_override" ? "NO_SUITABLE_CANDIDATE" : "RECOMMENDED_CANDIDATE"}`,
      `Final Score: ${s === "ood_override" ? "35.0 / 100" : s === "occupied" ? "92.0 / 100" : "95.0 / 100"}`,
      `Transmission Clearance: ${s === "ood_override" ? "HOLD FOR SAFETY" : "CLEARED FOR ALLOCATION"}`,
    ],
    systemRole:
      "OUTPUT BEACON: Final delivery of the recommended channel allocation with full traceability for spectrum managers and cognitive radio transceivers.",
    engineeringInsight:
      "The end-to-end flow is complete: Raw RF observation in → DSP physical check → feature extraction → ML evidence → OOD safety check → evidence fusion → engineering scoring → verified recommendation.",
  },
];

export function AnimatedSystemFlowPipeline() {
  const [currentStageIndex, setCurrentStageIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [speed, setSpeed] = useState<number>(1.0); // 0.5x, 1x, 2x
  const [scenario, setScenario] = useState<FlowScenario>("available");
  const [hoveredStageId, setHoveredStageId] = useState<number | null>(null);
  const [selectedStageId, setSelectedStageId] = useState<number>(1);
  const [reducedMotion, setReducedMotion] = useState<boolean>(false);
  const [progressPercent, setProgressPercent] = useState<number>(0);

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Check prefers-reduced-motion media query
  useEffect(() => {
    if (typeof window !== "undefined" && window.matchMedia) {
      const media = window.matchMedia("(prefers-reduced-motion: reduce)");
      setReducedMotion(media.matches);
      const listener = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
      media.addEventListener("change", listener);
      return () => media.removeEventListener("change", listener);
    }
  }, []);

  // Animation Cycle Management - strictly prevents duplicate intervals
  const STAGE_DURATION_MS = 2200 / speed;

  useEffect(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (!isPlaying || reducedMotion) return;

    const intervalStep = 40;
    const increment = (intervalStep / STAGE_DURATION_MS) * 100;

    timerRef.current = setInterval(() => {
      setProgressPercent((prev) => {
        if (prev >= 100) {
          setCurrentStageIndex((idx) => {
            const nextIdx = (idx + 1) % STAGES.length;
            setSelectedStageId(STAGES[nextIdx].id);
            return nextIdx;
          });
          return 0;
        }
        return prev + increment;
      });
    }, intervalStep);

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isPlaying, STAGE_DURATION_MS, reducedMotion]);

  // Stage Selection Handler
  const handleSelectStage = (index: number) => {
    setCurrentStageIndex(index);
    setSelectedStageId(STAGES[index].id);
    setProgressPercent(0);
  };

  // Playback Control Handlers
  const handleTogglePlay = () => {
    setIsPlaying((prev) => !prev);
  };

  const handleReplay = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setCurrentStageIndex(0);
    setSelectedStageId(STAGES[0].id);
    setProgressPercent(0);
    setIsPlaying(true);
  };

  const handleStepForward = () => {
    setIsPlaying(false);
    const nextIdx = (currentStageIndex + 1) % STAGES.length;
    setCurrentStageIndex(nextIdx);
    setSelectedStageId(STAGES[nextIdx].id);
    setProgressPercent(0);
  };

  const handleStepBack = () => {
    setIsPlaying(false);
    const prevIdx = (currentStageIndex - 1 + STAGES.length) % STAGES.length;
    setCurrentStageIndex(prevIdx);
    setSelectedStageId(STAGES[prevIdx].id);
    setProgressPercent(0);
  };

  // Stage Status Resolver (Pending, Processing, Complete, Uncertain)
  const getStageStatus = (stageIdx: number): {
    type: StageStatusType;
    label: string;
    icon: React.ComponentType<{ className?: string }>;
    color: string;
    badgeStyle: string;
  } => {
    if (stageIdx > currentStageIndex) {
      return {
        type: "pending",
        label: "PENDING",
        icon: Clock,
        color: "text-slate-500",
        badgeStyle: "bg-slate-900 border-slate-800 text-slate-500",
      };
    }
    if (stageIdx === currentStageIndex) {
      return {
        type: "processing",
        label: "PROCESSING",
        icon: RefreshCw,
        color: "text-sky-400",
        badgeStyle: "bg-sky-950/90 border-sky-600 text-sky-300 font-bold shadow-sm shadow-sky-900/40",
      };
    }
    // Completed stages
    if (scenario === "ood_override" && (stageIdx === 4 || stageIdx === 5 || stageIdx === 6 || stageIdx === 7)) {
      return {
        type: "uncertain",
        label: "UNCERTAIN",
        icon: AlertTriangle,
        color: "text-amber-400",
        badgeStyle: "bg-amber-950/80 border-amber-600 text-amber-300 font-semibold",
      };
    }
    return {
      type: "complete",
      label: "COMPLETE",
      icon: CheckCircle2,
      color: "text-emerald-400",
      badgeStyle: "bg-emerald-950/80 border-emerald-600 text-emerald-300 font-semibold",
    };
  };

  const activeStage = STAGES[currentStageIndex];
  const inspectedStage = STAGES.find((s) => s.id === selectedStageId) || activeStage;
  const inspectedStatus = getStageStatus(STAGES.findIndex((s) => s.id === inspectedStage.id));

  return (
    <div className="space-y-6 font-sans">
      {/* ── TOP CONTROL & SCENARIO BAR ────────────────────────────────────── */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/90 backdrop-blur p-4 shadow-xl flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        {/* Left: Title & Execution Mode */}
        <div className="flex items-center gap-3">
          <div className="relative p-2.5 rounded-lg bg-sky-600/20 border border-sky-500/30 text-sky-400 shrink-0">
            <Zap className="size-5 animate-pulse" />
            <span className="absolute -top-1 -right-1 size-2 rounded-full bg-emerald-400 ring-4 ring-emerald-950 animate-ping" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-black font-mono tracking-tight text-white uppercase">
                WIRE WATCHER SYSTEM FLOW PIPELINE
              </h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-sky-500/20 text-sky-300 border border-sky-500/30 font-semibold">
                8-STAGE ARCHITECTURE
              </span>
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-400 mt-0.5">
              <span className="text-emerald-400 font-mono font-semibold text-[11px]">
                ● Mode: Pipeline Playback & Audit Trace
              </span>
              <span className="text-slate-600 hidden sm:inline">|</span>
              <span className="hidden sm:inline text-slate-400 text-[11px]">
                Demonstrates chronological evaluation across RF, DSP, ML, OOD, and Allocation.
              </span>
            </div>
          </div>
        </div>

        {/* Center: Scenario Selector */}
        <div className="flex flex-wrap items-center gap-2 font-mono text-xs">
          <span className="text-[11px] text-slate-400 font-semibold uppercase mr-1 flex items-center gap-1">
            <SlidersHorizontal className="size-3 text-sky-400" /> Scenario:
          </span>

          <button
            onClick={() => {
              setScenario("available");
              handleReplay();
            }}
            className={cn(
              "px-3 py-1.5 rounded-lg border text-xs font-semibold transition-all flex items-center gap-1.5",
              scenario === "available"
                ? "bg-emerald-600/30 border-emerald-500 text-emerald-300 shadow-sm shadow-emerald-900/40"
                : "border-slate-800 bg-slate-950/60 text-slate-400 hover:text-slate-200"
            )}
            title="Clean spectrum, low noise, In-Distribution -> AVAILABLE decision"
          >
            <CheckCircle2 className="size-3.5 text-emerald-400" />
            <span>Available (Clean)</span>
          </button>

          <button
            onClick={() => {
              setScenario("occupied");
              handleReplay();
            }}
            className={cn(
              "px-3 py-1.5 rounded-lg border text-xs font-semibold transition-all flex items-center gap-1.5",
              scenario === "occupied"
                ? "bg-rose-600/30 border-rose-500 text-rose-300 shadow-sm shadow-rose-900/40"
                : "border-slate-800 bg-slate-950/60 text-slate-400 hover:text-slate-200"
            )}
            title="High power transmitter active, high SNR -> OCCUPIED decision"
          >
            <Activity className="size-3.5 text-rose-400" />
            <span>Occupied (Active RF)</span>
          </button>

          <button
            onClick={() => {
              setScenario("ood_override");
              handleReplay();
            }}
            className={cn(
              "px-3 py-1.5 rounded-lg border text-xs font-semibold transition-all flex items-center gap-1.5",
              scenario === "ood_override"
                ? "bg-amber-600/30 border-amber-500 text-amber-300 shadow-sm shadow-amber-900/40"
                : "border-slate-800 bg-slate-950/60 text-slate-400 hover:text-slate-200"
            )}
            title="Out-of-Distribution parameters -> Safety Override forces UNCERTAIN"
          >
            <ShieldAlert className="size-3.5 text-amber-400" />
            <span>OOD Safety Override</span>
          </button>
        </div>

        {/* Right: Playback Controls (Play/Pause, Replay, Step Forward/Back, Speed, Reduced Motion) */}
        <div className="flex items-center gap-2 border-t lg:border-t-0 border-slate-800 pt-3 lg:pt-0 shrink-0">
          {/* Replay */}
          <button
            onClick={handleReplay}
            className="p-1.5 rounded-lg border border-slate-800 bg-slate-950 text-slate-300 hover:text-white hover:bg-slate-800 transition flex items-center gap-1 text-xs font-mono"
            title="Replay from Stage 1"
          >
            <RotateCcw className="size-3.5" />
            <span className="hidden sm:inline">Replay</span>
          </button>

          {/* Step Back */}
          <button
            onClick={handleStepBack}
            className="p-1.5 rounded-lg border border-slate-800 bg-slate-950 text-slate-300 hover:text-white hover:bg-slate-800 transition"
            title="Step Back one stage"
          >
            <ChevronLeft className="size-4" />
          </button>

          {/* Play / Pause */}
          <button
            onClick={handleTogglePlay}
            className={cn(
              "px-3 py-1.5 rounded-lg border text-xs font-bold transition flex items-center gap-1.5",
              isPlaying
                ? "bg-sky-600 hover:bg-sky-500 text-white border-sky-400 shadow-sm shadow-sky-900/50"
                : "bg-slate-800 hover:bg-slate-700 text-slate-200 border-slate-700"
            )}
            title={isPlaying ? "Pause Flow" : "Play Flow"}
          >
            {isPlaying ? (
              <>
                <Pause className="size-3.5" />
                <span>Pause</span>
              </>
            ) : (
              <>
                <Play className="size-3.5" />
                <span>Play</span>
              </>
            )}
          </button>

          {/* Step Forward */}
          <button
            onClick={handleStepForward}
            className="p-1.5 rounded-lg border border-slate-800 bg-slate-950 text-slate-300 hover:text-white hover:bg-slate-800 transition"
            title="Step Forward one stage (Viva Demonstration)"
          >
            <ChevronRight className="size-4" />
          </button>

          {/* Speed Toggle */}
          <button
            onClick={() => {
              setSpeed((prev) => (prev === 0.5 ? 1.0 : prev === 1.0 ? 2.0 : 0.5));
            }}
            className="px-2 py-1.5 rounded-lg border border-slate-800 bg-slate-950 text-slate-300 hover:text-white text-xs font-mono font-bold transition"
            title="Playback Speed"
          >
            {speed}x
          </button>

          {/* Motion Toggle */}
          <button
            onClick={() => setReducedMotion(!reducedMotion)}
            className={cn(
              "px-2 py-1.5 rounded-lg border text-[11px] font-mono transition",
              reducedMotion
                ? "border-amber-500/50 bg-amber-950/40 text-amber-300"
                : "border-slate-800 bg-slate-950 text-slate-400 hover:text-slate-200"
            )}
            title="Toggle Reduced Motion Accessibility Setting"
          >
            {reducedMotion ? "Motion: OFF" : "Motion: ON"}
          </button>
        </div>
      </div>

      {/* ── ARCHITECTURAL HIERARCHY BANNER ─────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-center font-mono text-[11px]">
        <div className="p-2.5 rounded-lg border border-indigo-500/30 bg-indigo-950/20 text-indigo-300">
          <div className="text-[9px] uppercase tracking-wider text-indigo-400 font-bold">1. Physical Authority</div>
          <div className="font-black mt-0.5">DSP = Primary Detector</div>
        </div>

        <div className="p-2.5 rounded-lg border border-fuchsia-500/30 bg-fuchsia-950/20 text-fuchsia-300">
          <div className="text-[9px] uppercase tracking-wider text-fuchsia-400 font-bold">2. Statistical Aid</div>
          <div className="font-black mt-0.5">ML = Supplementary Evidence</div>
        </div>

        <div className="p-2.5 rounded-lg border border-amber-500/30 bg-amber-950/20 text-amber-300">
          <div className="text-[9px] uppercase tracking-wider text-amber-400 font-bold">3. Fail-Safe Barrier</div>
          <div className="font-black mt-0.5">OOD = Safety Override</div>
        </div>

        <div className="p-2.5 rounded-lg border border-emerald-500/30 bg-emerald-950/20 text-emerald-300">
          <div className="text-[9px] uppercase tracking-wider text-emerald-400 font-bold">4. Conclusive Fusion</div>
          <div className="font-black mt-0.5">Arbitration = Availability</div>
        </div>

        <div className="p-2.5 rounded-lg border border-cyan-500/30 bg-cyan-950/20 text-cyan-300 col-span-2 md:col-span-1">
          <div className="text-[9px] uppercase tracking-wider text-cyan-400 font-bold">5. Allocation Engine</div>
          <div className="font-black mt-0.5">100-Pt Deterministic Scoring</div>
        </div>
      </div>

      {/* ── MAIN ANIMATED PIPELINE CANVAS ───────────────────────────────────── */}
      <div className="rounded-2xl border border-slate-800 bg-slate-950 p-5 shadow-2xl relative overflow-hidden">
        {/* Subtle grid background */}
        <div
          className="absolute inset-0 opacity-10 pointer-events-none"
          style={{
            backgroundImage: "radial-gradient(#38bdf8 1px, transparent 1px)",
            backgroundSize: "24px 24px",
          }}
        />

        {/* Global Progress Bar along top of the pipeline */}
        <div className="mb-4 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs font-mono text-slate-400 border-b border-slate-800/80 pb-3">
          <div className="flex items-center gap-2">
            <span className="size-2 rounded-full bg-sky-400 animate-pulse" />
            <span>Active Stage:</span>
            <strong className="text-white font-bold">{activeStage.name}</strong>
            <span className={cn("px-2 py-0.5 rounded text-[10px] font-bold border", activeStage.hierarchyBadgeColor)}>
              {activeStage.hierarchyRole}
            </span>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-[11px] text-slate-500">Stage {currentStageIndex + 1} of 8</span>
            <div className="w-32 h-1.5 rounded-full bg-slate-800 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-sky-500 via-indigo-500 to-emerald-500 transition-all duration-75"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
          </div>
        </div>

        {/* ── 8-STAGE HORIZONTAL FLOW TRACK ─────────────────────────────────── */}
        <div className="relative">
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3 relative z-10">
            {STAGES.map((stage, idx) => {
              const Icon = stage.icon;
              const isActive = idx === currentStageIndex;
              const isSelected = stage.id === selectedStageId;
              const isHovered = stage.id === hoveredStageId;
              const stageStatus = getStageStatus(idx);
              const StatusIcon = stageStatus.icon;

              return (
                <div
                  key={stage.id}
                  onClick={() => handleSelectStage(idx)}
                  onMouseEnter={() => setHoveredStageId(stage.id)}
                  onMouseLeave={() => setHoveredStageId(null)}
                  className={cn(
                    "relative flex flex-col justify-between p-3.5 rounded-xl border transition-all duration-300 cursor-pointer select-none",
                    isActive
                      ? cn("border-2 scale-[1.02] shadow-lg z-20", stage.borderActive, stage.bgActive)
                      : isSelected
                      ? "border-slate-500 bg-slate-900/90 shadow-md"
                      : stageStatus.type === "complete"
                      ? "border-slate-800/90 bg-slate-900/60 opacity-90 hover:opacity-100 hover:border-slate-700"
                      : stageStatus.type === "uncertain"
                      ? "border-amber-900/50 bg-amber-950/20 opacity-90 hover:opacity-100 hover:border-amber-700"
                      : "border-slate-800/60 bg-slate-950/80 opacity-60 hover:opacity-90 hover:border-slate-700"
                  )}
                  style={{
                    boxShadow: isActive ? `0 0 24px -2px ${stage.glowRgba}` : undefined,
                  }}
                >
                  {/* Top: Header, Icon & Compact Status Indicator */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <div
                        className={cn(
                          "size-7 rounded-lg flex items-center justify-center transition-all",
                          isActive
                            ? "bg-white text-slate-900 shadow-sm"
                            : isSelected
                            ? "bg-slate-700 text-white"
                            : "bg-slate-800/80 text-slate-400"
                        )}
                      >
                        <Icon className="size-4" />
                      </div>

                      {/* Compact Multi-modal Status Indicator (Icon + Label) */}
                      <span
                        className={cn(
                          "flex items-center gap-1 text-[9px] font-mono px-1.5 py-0.5 rounded border transition-all",
                          stageStatus.badgeStyle
                        )}
                      >
                        <StatusIcon
                          className={cn(
                            "size-3",
                            stageStatus.type === "processing" ? "animate-spin" : ""
                          )}
                        />
                        <span>{stageStatus.label}</span>
                      </span>
                    </div>

                    {/* Stage Name */}
                    <div className="font-mono text-xs font-bold text-white tracking-tight leading-tight">
                      {stage.name}
                    </div>

                    {/* Hierarchy Subtitle */}
                    <div className="text-[10px] text-slate-400 mt-0.5 font-sans line-clamp-1">
                      {stage.subtitle}
                    </div>
                  </div>

                  {/* Middle: Micro Stage Visualizer (Stage-Specific) */}
                  <div className="my-2.5 py-1 border-y border-slate-800/60">
                    {stage.id === 1 && (
                      <div className="flex items-center justify-center h-7 overflow-hidden">
                        <div className="flex items-center gap-0.5 w-full">
                          {[35, 70, 45, 90, 60, 80, 40, 65, 30, 85, 50].map((h, i) => (
                            <div
                              key={i}
                              className={cn(
                                "flex-1 rounded-full transition-all duration-300",
                                isActive ? "bg-sky-400" : "bg-slate-700"
                              )}
                              style={{
                                height: `${isActive ? h : Math.max(20, h * 0.4)}%`,
                              }}
                            />
                          ))}
                        </div>
                      </div>
                    )}

                    {stage.id === 2 && (
                      <div className="space-y-1">
                        <div className="flex justify-between text-[9px] font-mono text-slate-400">
                          <span>kTB</span>
                          <span className={stage.getMetricStatus(scenario).color}>SNR +46dB</span>
                        </div>
                        <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden flex">
                          <div
                            className={cn(
                              "h-full transition-all duration-500",
                              scenario === "occupied" ? "bg-rose-500 w-[95%]" : "bg-indigo-400 w-[65%]"
                            )}
                          />
                        </div>
                      </div>
                    )}

                    {stage.id === 3 && (
                      <div className="flex flex-wrap gap-1 text-[8px] font-mono">
                        <span className="px-1 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800">RMS</span>
                        <span className="px-1 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800">Crest</span>
                        <span className="px-1 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800">Entropy</span>
                      </div>
                    )}

                    {stage.id === 4 && (
                      <div className="flex items-center justify-between gap-1 py-0.5">
                        {[1, 2, 3, 4, 5].map((t) => (
                          <div
                            key={t}
                            className={cn(
                              "flex-1 h-5 rounded flex flex-col items-center justify-center text-[7px] font-mono border transition-all",
                              isActive
                                ? "border-fuchsia-400 bg-fuchsia-950 text-fuchsia-300"
                                : "border-slate-800 bg-slate-900 text-slate-500"
                            )}
                          >
                            T{t}
                          </div>
                        ))}
                      </div>
                    )}

                    {stage.id === 5 && (
                      <div className="flex items-center justify-between text-[9px] font-mono px-1 py-0.5 rounded bg-slate-900 border border-slate-800">
                        <span className="text-slate-500">[p1</span>
                        <span className={stage.getMetricStatus(scenario).color}>
                          {scenario === "ood_override" ? "OUT OF BOUNDS" : "PASS"}
                        </span>
                        <span className="text-slate-500">p99]</span>
                      </div>
                    )}

                    {stage.id === 6 && (
                      <div className="flex items-center justify-between gap-1 text-[8px] font-mono">
                        <span
                          className={cn(
                            "px-1 py-0.5 rounded flex-1 text-center font-bold",
                            scenario === "available"
                              ? "bg-emerald-500 text-white shadow-sm"
                              : "bg-slate-900 text-slate-600"
                          )}
                        >
                          AVAIL
                        </span>
                        <span
                          className={cn(
                            "px-1 py-0.5 rounded flex-1 text-center font-bold",
                            scenario === "occupied"
                              ? "bg-rose-500 text-white shadow-sm"
                              : "bg-slate-900 text-slate-600"
                          )}
                        >
                          OCC
                        </span>
                        <span
                          className={cn(
                            "px-1 py-0.5 rounded flex-1 text-center font-bold",
                            scenario === "ood_override"
                              ? "bg-amber-500 text-white shadow-sm"
                              : "bg-slate-900 text-slate-600"
                          )}
                        >
                          UNC
                        </span>
                      </div>
                    )}

                    {stage.id === 7 && (
                      <div className="space-y-1">
                        <div className="flex justify-between text-[9px] font-mono text-slate-400">
                          <span>Score</span>
                          <span className={stage.getMetricStatus(scenario).color}>{stage.getMetricValue(scenario)}</span>
                        </div>
                        <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className={cn(
                              "h-full transition-all duration-500",
                              scenario === "available"
                                ? "bg-cyan-400 w-[95%]"
                                : scenario === "occupied"
                                ? "bg-rose-500 w-[42%]"
                                : "bg-amber-500 w-[35%]"
                            )}
                          />
                        </div>
                      </div>
                    )}

                    {stage.id === 8 && (
                      <div className="p-1 rounded bg-slate-900 border border-slate-800 text-center font-mono text-[9px]">
                        <span className={stage.getMetricStatus(scenario).color}>
                          {scenario === "ood_override" ? "SAFETY HOLD" : scenario === "occupied" ? "CH-042" : "CH-001"}
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Bottom: Telemetry readout */}
                  <div>
                    <div className="text-[9px] font-mono text-slate-500 uppercase tracking-wider">
                      {stage.metricLabel}
                    </div>
                    <div className="text-xs font-mono font-bold text-white mt-0.5">
                      {stage.getMetricValue(scenario)}
                    </div>
                  </div>

                  {/* Directional Connector Arrow / Particle to Next Stage (Desktop only) */}
                  {idx < STAGES.length - 1 && (
                    <div className="hidden lg:block absolute -right-3 top-1/2 -translate-y-1/2 z-30 pointer-events-none">
                      <div className="relative size-6 flex items-center justify-center">
                        <ChevronRight
                          className={cn(
                            "size-4 transition-all duration-300",
                            isActive ? "text-sky-400 scale-125" : "text-slate-600"
                          )}
                        />
                        {/* Animated flowing particle along the connector */}
                        {isActive && !reducedMotion && (
                          <span
                            className="absolute size-2 rounded-full bg-sky-300 shadow-md shadow-sky-400"
                            style={{
                              animation: "rf-dash-flow 1.2s infinite ease-in-out",
                            }}
                          />
                        )}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Continuous Connecting Flow Pulse Line (SVG) */}
          <div className="hidden lg:block w-full mt-4 px-2">
            <svg className="w-full h-8 overflow-visible" viewBox="0 0 800 24" preserveAspectRatio="none">
              <defs>
                <linearGradient id="flowGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.8" />
                  <stop offset="25%" stopColor="#6366f1" stopOpacity="0.8" />
                  <stop offset="50%" stopColor="#a855f7" stopOpacity="0.8" />
                  <stop offset="65%" stopColor="#f59e0b" stopOpacity="0.8" />
                  <stop offset="85%" stopColor="#10b981" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.8" />
                </linearGradient>
                <filter id="glowFilter" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="3" result="glow" />
                  <feComposite in="SourceGraphic" in2="glow" operator="over" />
                </filter>
              </defs>

              {/* Base background track */}
              <line
                x1="20"
                y1="12"
                x2="780"
                y2="12"
                stroke="#1e293b"
                strokeWidth="3"
                strokeLinecap="round"
              />

              {/* Animated active beam */}
              <line
                x1="20"
                y1="12"
                x2={20 + (currentStageIndex / 7) * 760}
                y2="12"
                stroke="url(#flowGradient)"
                strokeWidth="3"
                strokeLinecap="round"
                filter="url(#glowFilter)"
                className={!reducedMotion ? "animate-rf-dash" : undefined}
                strokeDasharray="8 6"
              />

              {/* Moving Pulse Head */}
              {!reducedMotion && (
                <circle
                  cx={20 + (currentStageIndex / 7) * 760}
                  cy="12"
                  r="5"
                  fill="#ffffff"
                  stroke="#38bdf8"
                  strokeWidth="2"
                  className="animate-ping"
                />
              )}
            </svg>
          </div>
        </div>
      </div>

      {/* ── SECTION 6: DECISION EXPLANATION & EVIDENCE FUSION BREAKDOWN ─────── */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/90 p-5 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
              <FileText className="size-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 uppercase font-bold">
                  DECISION EXPLANATION
                </span>
                <span className="text-xs text-slate-400 font-mono">
                  Why the System Produced This Final Decision
                </span>
              </div>
              <h4 className="text-base font-black font-mono text-white mt-1">
                Operational Outcome:{" "}
                <span
                  className={cn(
                    "px-2.5 py-0.5 rounded font-black",
                    scenario === "available"
                      ? "text-emerald-400 bg-emerald-950/80 border border-emerald-600"
                      : scenario === "occupied"
                      ? "text-rose-400 bg-rose-950/80 border border-rose-600"
                      : "text-amber-400 bg-amber-950/80 border border-amber-600"
                  )}
                >
                  {scenario === "available" ? "AVAILABLE" : scenario === "occupied" ? "OCCUPIED" : "UNCERTAIN"}
                </span>
              </h4>
            </div>
          </div>

          <div className="text-xs font-mono text-slate-400 flex items-center gap-2">
            <span>Scenario:</span>
            <strong className="text-white uppercase font-bold">
              {scenario === "available" ? "Clean Spectrum" : scenario === "occupied" ? "Active RF Transmitter" : "OOD Safety Trigger"}
            </strong>
          </div>
        </div>

        {/* 5-Pillar Decision Audit Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3 font-mono text-xs">
          {/* Pillar 1: DSP Physical Evidence */}
          <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between text-[10px] text-slate-400 uppercase font-bold border-b border-slate-800/80 pb-1">
              <span>1. DSP Physical Check</span>
              <span className="text-indigo-400 font-mono">PRIMARY</span>
            </div>
            <div className="space-y-1 text-[11px]">
              <div className="text-slate-400">
                Floor: <span className="text-white font-bold">-120.96 dBm</span>
              </div>
              <div className="text-slate-400">
                SNR:{" "}
                <span className={scenario === "occupied" ? "text-rose-400 font-bold" : "text-emerald-400 font-bold"}>
                  {scenario === "occupied" ? "+68.56 dB" : scenario === "ood_override" ? "+102.9 dB" : "+45.96 dB"}
                </span>
              </div>
              <div className="text-slate-400">
                Physical State:{" "}
                <strong className={scenario === "occupied" ? "text-rose-400" : "text-emerald-400"}>
                  {scenario === "occupied" ? "DETECTED" : "NOT_DETECTED"}
                </strong>
              </div>
            </div>
            <div className="text-[10px] text-slate-500 font-sans border-t border-slate-800/60 pt-1 leading-tight">
              {scenario === "occupied"
                ? "Physical detection conditions were satisfied; energy detected on channel."
                : "Signal power remains below empirical median threshold; clear noise headroom."}
            </div>
          </div>

          {/* Pillar 2: ML Supporting Evidence */}
          <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between text-[10px] text-slate-400 uppercase font-bold border-b border-slate-800/80 pb-1">
              <span>2. ML Evidence</span>
              <span className="text-fuchsia-400 font-mono">SUPPORTING</span>
            </div>
            <div className="space-y-1 text-[11px]">
              <div className="text-slate-400">
                Model: <span className="text-white">Random Forest (60 Trees)</span>
              </div>
              <div className="text-slate-400">
                P(Occupied):{" "}
                <span className={scenario === "occupied" ? "text-rose-400 font-bold" : "text-emerald-400 font-bold"}>
                  {scenario === "occupied" ? "88.4%" : scenario === "ood_override" ? "94.2% (Flagged)" : "14.2%"}
                </span>
              </div>
              <div className="text-slate-400">
                Threshold: <span className="text-slate-300">0.50</span>
              </div>
            </div>
            <div className="text-[10px] text-slate-500 font-sans border-t border-slate-800/60 pt-1 leading-tight">
              Supporting activity probability displayed for context. Strictly quarantined from signal strength.
            </div>
          </div>

          {/* Pillar 3: OOD Safety Guard */}
          <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between text-[10px] text-slate-400 uppercase font-bold border-b border-slate-800/80 pb-1">
              <span>3. OOD Safety Guard</span>
              <span className="text-amber-400 font-mono">OVERRIDE</span>
            </div>
            <div className="space-y-1 text-[11px]">
              <div className="text-slate-400">
                Distance D:{" "}
                <span className={scenario === "ood_override" ? "text-amber-400 font-bold" : "text-emerald-400 font-bold"}>
                  {scenario === "ood_override" ? "5.84 (>3.0 Limit)" : "1.34 (≤3.0 Limit)"}
                </span>
              </div>
              <div className="text-slate-400">
                Bounds:{" "}
                <strong className={scenario === "ood_override" ? "text-amber-400" : "text-emerald-400"}>
                  {scenario === "ood_override" ? "VIOLATION (CF=12.4)" : "IN-DISTRIBUTION"}
                </strong>
              </div>
              <div className="text-slate-400">
                Band: <span className="text-slate-300">{scenario === "ood_override" ? "450 MHz (Uncalibrated)" : "120 MHz (Calibrated)"}</span>
              </div>
            </div>
            <div className="text-[10px] text-slate-500 font-sans border-t border-slate-800/60 pt-1 leading-tight">
              {scenario === "ood_override"
                ? "Safety guard caused UNCERTAIN outcome. Out-of-bounds observations are never marked available."
                : "Observation is within the supported training data range."}
            </div>
          </div>

          {/* Pillar 4: Decision Arbitration Rule */}
          <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between text-[10px] text-slate-400 uppercase font-bold border-b border-slate-800/80 pb-1">
              <span>4. Governing Rule</span>
              <span className="text-emerald-400 font-mono">FUSION</span>
            </div>
            <div className="space-y-1 text-[11px]">
              <div className="text-slate-400">
                Governing Rule:
              </div>
              <div className="p-1.5 rounded bg-slate-900 border border-slate-800 text-[10px] text-sky-300 font-bold leading-tight">
                {scenario === "ood_override"
                  ? "Rule: OOD == TRUE → UNCERTAIN"
                  : scenario === "occupied"
                  ? "Rule: OOD==0 && DSP==1 → OCCUPIED"
                  : "Rule: OOD==0 && DSP==0 → AVAILABLE"}
              </div>
            </div>
            <div className="text-[10px] text-slate-500 font-sans border-t border-slate-800/60 pt-1 leading-tight">
              {scenario === "ood_override"
                ? "The safety override rule determined the uncertain state to prevent transmission risks."
                : scenario === "occupied"
                ? "The physical detection rule determined the occupied state."
                : "Clean physical evidence confirmed spectrum availability."}
            </div>
          </div>

          {/* Pillar 5: Channel Allocation Outcome */}
          <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 space-y-2 col-span-1 md:col-span-2 lg:col-span-1">
            <div className="flex items-center justify-between text-[10px] text-slate-400 uppercase font-bold border-b border-slate-800/80 pb-1">
              <span>5. Channel Allocation</span>
              <span className="text-cyan-400 font-mono">SCORING</span>
            </div>
            <div className="space-y-1 text-[11px]">
              <div className="text-slate-400">
                Candidate:{" "}
                <strong className={scenario === "ood_override" ? "text-amber-400" : "text-white"}>
                  {scenario === "ood_override" ? "TRANSMISSION HOLD" : scenario === "occupied" ? "CH-042 (82.4 MHz)" : "CH-001 (70.1 MHz)"}
                </strong>
              </div>
              <div className="text-slate-400">
                Score:{" "}
                <span className={scenario === "available" ? "text-emerald-400 font-bold" : "text-amber-400 font-bold"}>
                  {scenario === "available" ? "95.0 / 100" : scenario === "occupied" ? "92.0 / 100 (Rerouted)" : "35.0 / 100 (< 50)"}
                </span>
              </div>
              <div className="text-slate-400">
                Guard Band: <span className="text-emerald-400 font-semibold">{scenario === "ood_override" ? "UNSAFE" : "SAFE_MARGIN"}</span>
              </div>
            </div>
            <div className="text-[10px] text-slate-500 font-sans border-t border-slate-800/60 pt-1 leading-tight">
              {scenario === "ood_override"
                ? "All channels held below 50 pt threshold due to OOD penalty. No channel allocated."
                : scenario === "occupied"
                ? "CH-001 penalized (-50 pts). Engine successfully rerouted to alternate CH-042."
                : "Channel CH-001 achieved top score with verified guard-band margins."}
            </div>
          </div>
        </div>
      </div>

      {/* ── SECTION 7: INTERACTIVE STAGE DEEP-DIVE INSPECTOR ────────────────── */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/90 p-5 shadow-xl space-y-4">
        {/* Inspector Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <div className="flex items-center gap-3">
            <div className={cn("p-2 rounded-lg border", inspectedStage.bgActive, inspectedStage.borderActive)}>
              {React.createElement(inspectedStage.icon, {
                className: cn("size-5", inspectedStage.textActive),
              })}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 uppercase font-bold">
                  STAGE DEEP-DIVE INSPECTION
                </span>
                <span className={cn("text-xs font-mono font-bold px-2 py-0.5 rounded border", inspectedStage.hierarchyBadgeColor)}>
                  {inspectedStage.hierarchyRole}
                </span>
                <span className={cn("text-[10px] font-mono px-2 py-0.5 rounded border", inspectedStatus.badgeStyle)}>
                  STATUS: {inspectedStatus.label}
                </span>
              </div>
              <h4 className="text-base font-black font-mono text-white mt-1">
                {inspectedStage.name}: {inspectedStage.subtitle}
              </h4>
            </div>
          </div>

          {/* Quick jump to inspect another stage */}
          <div className="flex items-center gap-1 font-mono text-xs overflow-x-auto pb-1 sm:pb-0">
            {STAGES.map((s) => (
              <button
                key={s.id}
                onClick={() => setSelectedStageId(s.id)}
                className={cn(
                  "size-7 rounded font-bold text-xs transition flex items-center justify-center",
                  s.id === selectedStageId
                    ? "bg-sky-600 text-white shadow-sm"
                    : "bg-slate-800 text-slate-400 hover:text-white"
                )}
                title={`Inspect ${s.name}`}
              >
                {s.id}
              </button>
            ))}
          </div>
        </div>

        {/* 2-Column Inspector Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column: What It Does, Role, Inputs & Outputs */}
          <div className="space-y-4 text-xs font-sans">
            <div>
              <h5 className="font-mono uppercase font-bold text-slate-400 text-[11px] mb-1">
                1. What This Stage Does
              </h5>
              <p className="text-slate-300 leading-relaxed font-sans">{inspectedStage.whatItDoes}</p>
            </div>

            <div>
              <h5 className="font-mono uppercase font-bold text-slate-400 text-[11px] mb-1">
                2. Role in Overall System Hierarchy
              </h5>
              <p className="text-slate-300 leading-relaxed font-sans">{inspectedStage.systemRole}</p>
            </div>

            {/* Inputs and Outputs Panels */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 font-mono text-xs">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1.5">
                <div className="text-[10px] uppercase font-bold text-sky-400 border-b border-slate-800 pb-1">
                  Stage Inputs
                </div>
                {inspectedStage.inputDescription.map((inItem, idx) => (
                  <div key={idx} className="text-slate-300 text-[11px] flex items-start gap-1.5">
                    <span className="text-sky-500 shrink-0">▸</span>
                    <span>{inItem}</span>
                  </div>
                ))}
              </div>

              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1.5">
                <div className="text-[10px] uppercase font-bold text-emerald-400 border-b border-slate-800 pb-1">
                  Stage Outputs ({scenario.toUpperCase()})
                </div>
                {inspectedStage.outputDescription(scenario).map((outItem, idx) => (
                  <div key={idx} className="text-slate-300 text-[11px] flex items-start gap-1.5">
                    <span className="text-emerald-500 shrink-0">▸</span>
                    <span>{outItem}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Scientific Architecture Insight Callout */}
            <div className="p-3 rounded-lg border border-sky-500/30 bg-sky-950/20 text-sky-200 flex items-start gap-2.5">
              <Info className="size-4 text-sky-400 shrink-0 mt-0.5" />
              <div className="text-xs leading-relaxed font-sans">{inspectedStage.engineeringInsight}</div>
            </div>
          </div>

          {/* Right Column: Mathematical Formulas & Live Substitution */}
          <div className="space-y-4 font-mono text-xs">
            {/* Mathematical Formula Box */}
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                  Mathematical Formula
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-sky-300 font-bold">
                  LATEX / ECE DEFINITION
                </span>
              </div>

              <div className="p-2.5 rounded bg-slate-900 border border-slate-800 text-sky-300 font-bold text-sm">
                <code>{inspectedStage.formula}</code>
              </div>

              <div>
                <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
                  Active Substitution ({scenario.toUpperCase()} Mode)
                </div>
                <div className="p-2.5 rounded bg-slate-900/80 border border-slate-800/80 text-emerald-300 text-xs font-mono">
                  {inspectedStage.formulaSubstitution(scenario)}
                </div>
              </div>
            </div>

            {/* Stage Decision / Flow Outcome */}
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                Current Execution Status
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">Processing State:</span>
                <span className={cn("font-bold px-2 py-0.5 rounded text-[11px] border", inspectedStatus.badgeStyle)}>
                  {inspectedStatus.label}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">{inspectedStage.metricLabel}:</span>
                <span className="text-white font-bold text-sm">{inspectedStage.getMetricValue(scenario)}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">Status Verification:</span>
                <span className={cn("font-bold", inspectedStage.getMetricStatus(scenario).color)}>
                  {inspectedStage.getMetricStatus(scenario).label}
                </span>
              </div>
              <div className="pt-2 border-t border-slate-800/80 text-[11px] text-slate-500 font-sans">
                Information smoothly forwards to{" "}
                <strong className="text-slate-300">
                  {inspectedStage.id < 8 ? STAGES[inspectedStage.id].name : "Cognitive Radio Transmitter Output"}
                </strong>
                .
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
