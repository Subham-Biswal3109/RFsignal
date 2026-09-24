/**
 * Types mirroring the EXISTING Flask API contract for Wire Watcher.
 * Field names come from the running backend (POST /api/predict) and must not be renamed.
 */

export interface AnalyzeSpectrumResponse {
  data_source: string;
  frequency_range: { start_mhz: number; end_mhz: number };
  noise_floor_dbm: number;
  detected_signals: Array<{
    frequency_mhz: number;
    power_dbm: number;
    bandwidth_mhz: number;
    snr_db: number;
  }>;
  occupied_regions: Array<{ start_mhz: number; end_mhz: number }>;
  available_regions: Array<{ start_mhz: number; end_mhz: number }>;
  spectrum_data: {
    frequencies: number[];
    power_dbm: number[];
  };
  extracted_features: PredictRequest;
}

export interface PredictRequest {
  frequency_mhz: number;
  bandwidth_khz: number;
  signal_strength_dbm: number;
  iq_available: number;
  iq_rms_magnitude?: number;
  iq_magnitude_variance?: number;
  iq_peak_magnitude?: number;
  iq_crest_factor?: number;
  iq_p10?: number;
  iq_p50?: number;
  iq_p90?: number;
  iq_phase_concentration?: number;
  iq_spectral_entropy?: number;
  iq_spectral_peak_ratio?: number;
  timestamp?: string;
  location?: string;
  latitude?: number;
  longitude?: number;
}

export interface PredictResponse {
  prediction: number | string;
  available: boolean;
  probability: number;
  confidence: string;
  data_source: string;
  threshold?: number;
  important_features: string[];
  features_used: Record<string, string | number>;
  ood_warning?: boolean;
  warning?: string;
  activity?: string;
  availability?: string;
  label_type?: string;
  detector_activity?: boolean | null;
  ml_activity?: boolean;
  ml_activity_probability?: number;
}

/**
 * Normalized view of a stored prediction record coming from GET /api/predictions.
 * The backend row shape is read tolerantly: any field the backend does not
 * return stays `null` and is rendered as "—" (never faked).
 */
export interface PredictionRecord {
  id: string;
  start_frequency_mhz: number | null;
  end_frequency_mhz: number | null;
  bandwidth_mhz: number | null;
  city: string | null;
  state: string | null;
  service_type: string | null;
  available: boolean | null;
  probability: number | null;
  timestamp: string | null;
  signal_power_dbm?: number | null;
  noise_floor_dbm?: number | null;
  snr_db?: number | null;
  data_source?: string;
  ood_status?: boolean;
  activity?: string;
  availability?: string;
  confidence?: string;
  raw: Record<string, unknown>;
}


/** Shape of GET /api/health once the backend exposes it. */
export interface HealthResponse {
  status?: string;
  api?: string;
  model?: string;
  model_loaded?: boolean;
  database?: string;
  database_connected?: boolean;
  [key: string]: unknown;
}

export interface ModelInfoResponse {
  algorithm?: string;
  model_version?: string;
  training_date?: string;
  dataset_name?: string;
  dataset_type?: string;
  training_samples?: number;
  real_rf_validation?: boolean;
  best_threshold?: number;
  data_source?: string;
  features?: string[];
  limitations?: string[];
  kpis?: {
    total_predictions: number;
    available_predictions: number;
    occupied_predictions: number;
    avg_probability: number;
    ood_count: number;
  };
  [key: string]: unknown;
}

export type ApiErrorKind =
  | "network"
  | "timeout"
  | "validation"
  | "server"
  | "malformed"
  | "not_implemented";

/* ------------------------- Spectrum Simulation --------------------------
 * Types mirroring the NEW, separate /api/simulation/* endpoints.
 * This module does not replace or alter PredictRequest/PredictResponse.
 * -------------------------------------------------------------------- */

export type SimulationMode = "basic" | "ml_assisted" | "multi_user";

export type ChannelState = "OCCUPIED" | "AVAILABLE" | "UNAVAILABLE" | "ALLOCATED";

export interface SimulationChannel {
  channel_id: number;
  start_mhz: number;
  end_mhz: number;
  center_mhz: number;
  bandwidth_mhz: number;
  rf_signal_power_dbm: number;
  rf_noise_floor_dbm: number;
  rf_snr_db: number;
  rf_state: "OCCUPIED" | "AVAILABLE";
  ml_probability: number | null;
  ml_threshold: number | null;
  ml_decision: string;
  ml_ood_warning: boolean;
  ml_ood_reasons: string[];
  state: ChannelState;
}

export interface AllocationCandidate {
  channel_ids: number[];
  start_mhz: number;
  end_mhz: number;
  total_bandwidth_mhz: number;
  avg_snr_db: number;
  avg_ml_probability: number | null;
  isolation_score: number;
  score: number;
  rank: number;
}

export interface AllocationResult {
  requested_bandwidth_mhz: number;
  success: boolean;
  selected: AllocationCandidate | null;
  top_candidates: AllocationCandidate[];
  message: string;
  final_channels: SimulationChannel[];
}

export interface ResourceUtilization {
  total_mhz: number;
  occupied_mhz: number;
  available_mhz: number;
  allocated_mhz: number;
}

export interface MultiUserResult {
  user_id: string;
  requested_bandwidth_mhz: number;
  success: boolean;
  selected: AllocationCandidate | null;
  top_candidates: AllocationCandidate[];
  message: string;
}

export interface MultiUserAllocation {
  user_results: MultiUserResult[];
  final_channels: SimulationChannel[];
  utilization_timeline: ResourceUtilization[];
}

export interface SimulationRunRequest {
  start_frequency_mhz: number;
  end_frequency_mhz: number;
  channel_bandwidth_mhz: number;
  noise_floor_dbm: number;
  num_existing_users: number;
  seed?: number;
  mode: SimulationMode;
  requested_bandwidth_mhz?: number;
  users?: Array<{ user_id: string; requested_bandwidth_mhz: number }>;
  state?: string;
  city?: string;
  service_type?: string;
}

export interface SimulationRunResponse {
  mode: SimulationMode;
  channels: SimulationChannel[];
  spectrum_data: { frequencies: number[]; power_dbm: number[] };
  occupied_regions: Array<{ start_mhz: number; end_mhz: number }>;
  available_regions: Array<{ start_mhz: number; end_mhz: number }>;
  model_loaded: boolean;
  noise_floor_dbm: number;
  resource_utilization_before: ResourceUtilization;
  resource_utilization_after: ResourceUtilization;
  allocation?: AllocationResult;
  multi_user_allocation?: MultiUserAllocation;
  disclaimer: string;
}

export interface SnrSweepPoint {
  snr_db: number;
  noise_floor_dbm: number;
  probability: number | null;
  decision: string;
  threshold?: number;
  ood_warning?: boolean;
  ood_reasons?: string[];
}

export interface SnrSweepResponse {
  model_loaded: boolean;
  points: SnrSweepPoint[];
  disclaimer: string;
}

/* ------------------------- RF Interference/Jamming Detector -------------
 * A SEPARATE model/task from spectrum availability. Target: benign vs
 * malicious RF activity. Never mix with PredictRequest/PredictResponse or
 * SimulationRunResponse types above.
 * -------------------------------------------------------------------- */

export interface JammingControlledMetrics {
  description: string;
  n: number;
  class_distribution?: Record<string, number>;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusion_matrix: number[][];
  roc_auc: number;
  pr_auc: number;
}

export interface JammingSupplementaryMetrics {
  description: string;
  n: number;
  environment_composition?: Record<string, number>;
  is_environment_confounded: boolean;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusion_matrix: number[][];
  roc_auc: number;
  pr_auc: number;
}

export interface JammingModelInfoResponse {
  model_loaded: boolean;
  model_name: string;
  model_version: string;
  task: string;
  dataset_name: string;
  dataset_type: string;
  training_samples: number;
  validation_samples: number;
  test_samples: number;
  num_session_groups: { train: number; val: number; test: number };
  algorithm: string;
  best_threshold: number;
  split_methodology: string;
  threshold_tuning_methodology: string;
  primary_controlled_metrics: JammingControlledMetrics;
  supplementary_raw_test_metrics: JammingSupplementaryMetrics;
  baseline_comparison: {
    energy_baseline?: { accuracy: number; precision: number; recall: number; f1: number };
    logistic_regression?: { accuracy: number; precision: number; recall: number; f1: number; roc_auc: number };
    decision_tree?: { accuracy: number; precision: number; recall: number; f1: number; roc_auc: number };
  };
  feature_importances: Array<[string, number]>;
  limitations: string[];
  disclaimer: string;
}

export interface JammingSampleSummary {
  sample_id: string;
  file_name: string;
  true_label: "benign" | "malicious";
  band: string;
  scan_mode: string;
  waveform: string | null;
  power_dbm: number | null;
}

export interface JammingSamplesResponse {
  samples: JammingSampleSummary[];
  count: number;
}

export interface JammingPredictResponse {
  sample_id?: string;
  file_name?: string;
  true_label?: "benign" | "malicious";
  prediction: "benign" | "malicious";
  probability_malicious: number;
  threshold: number;
  correct?: boolean;
  disclaimer: string;
}

export interface ChannelCandidate {
  candidate_id: string;
  center_freq_mhz: number;
  start_freq_mhz: number;
  end_freq_mhz: number;
  bandwidth_mhz: number;
  bandwidth_khz: number;
  guard_band_mhz: number;
  protected_lower_mhz?: number;
  protected_upper_mhz?: number;
  guard_band_status?: "SAFE_MARGIN" | "MARGINAL" | "CONFLICT";
  interference_risk?: "LOW" | "MEDIUM" | "HIGH" | "UNKNOWN";
  activity: string;
  availability: string;
  ml_probability: number;
  detector_activity?: boolean | null;
  ood_warning: boolean;
  noise_floor_dbm: number;
  snr_db: number | null;
  score: number;
  scoring_breakdown: {
    base_score: number;
    activity_penalty: number;
    noise_penalty: number;
    ood_penalty: number;
    uncertainty_penalty: number;
    guard_band_penalty?: number;
    interference_penalty?: number;
  };
  recommendation_status: "RECOMMENDED" | "REVIEW_REQUIRED" | "REJECTED";
  assessment_explanation?: {
    headline: string;
    summary: string;
    recommendation: string;
    trace: {
      inputs: Record<string, string>;
      formula: string;
      substitution: string;
      result: string;
      unit: string;
      engineering_note: string;
    };
  };
}

export interface AllocationRecommendationResponse {
  request_params: {
    start_freq_mhz: number;
    end_freq_mhz: number;
    channel_bw_mhz: number;
    guard_band_mhz: number;
    noise_floor_dbm: number;
    location?: string;
  };
  scoring_weights: Record<string, number>;
  total_candidates: number;
  recommended_candidate: ChannelCandidate | null;
  has_suitable_candidate: boolean;
  reason?: string;
  supporting_evidence?: Record<string, any>;
  rejected_candidates?: ChannelCandidate[];
  candidates: ChannelCandidate[];
  disclaimer: string;
}

export interface SdrStatusResponse {
  source: string;
  provenance: string;
  connected: boolean;
  device: string | null;
  center_frequency_mhz: number;
  sample_rate_mhz: number;
  gain: string | number;
  buffer_size: number;
  message: string;
}

export interface RFEventLog {
  event_id: number;
  frequency_mhz: number;
  bandwidth_mhz: number;
  start_time: string | null;
  end_time: string | null;
  duration_seconds: number | null;
  peak_power_dbm: number;
  avg_power_dbm: number;
  activity_state: string;
  source_type: string;
  provenance: string;
}

export interface RFEventSummary {
  total_rf_events: number;
  average_event_duration_s: number;
  maximum_event_duration_s: number;
  average_peak_power_dbm: number;
  maximum_peak_power_dbm: number;
  most_active_frequency_mhz: number | null;
  most_frequent_source: string | null;
  events: RFEventLog[];
}

export interface ChannelUtilization {
  utilization_status: "AVAILABLE" | "UNAVAILABLE";
  reason: string;
  target_frequency_mhz: number | null;
  total_observation_duration_s: number;
  active_duration_s: number;
  inactive_duration_s: number;
  uncertain_duration_s: number;
  utilization_percentage: number;
  event_count: number;
  average_event_duration_s: number;
  maximum_event_duration_s: number;
  provenance: string;
}

export interface RFReplayStatus {
  mode: string;
  is_playing: boolean;
  playback_speed: number;
  current_index: number;
  total_samples: number;
  provenance: string;
  is_live: boolean;
  display_warning: string;
  last_observation?: Record<string, any> | null;
}

export interface RFTechnicalReport {
  report_id: string;
  timestamp: string;
  sections: {
    "1_observation": Record<string, any>;
    "2_dsp": Record<string, any>;
    "3_rf_engineering": Record<string, any>;
    "4_ml": Record<string, any>;
    "5_ood": Record<string, any>;
    "6_availability": Record<string, any>;
    "7_allocation": Record<string, any>;
    "8_events": Record<string, any>;
    "9_provenance": Record<string, any>;
    "10_limitations": string[];
  };
}

export interface WaveformMetrics {
  num_samples: number;
  sample_rate_mhz: number;
  center_freq_mhz: number;
  duration_us: number;
  frequency_resolution_khz: number;
  i_rms: number;
  q_rms: number;
  magnitude_rms: number;
  peak_magnitude: number;
  crest_factor_linear: number;
  crest_factor_db: number;
}

export interface WaveformSeries {
  time_us: number[];
  i: number[];
  q: number[];
  magnitude: number[];
  phase: number[];
}

export interface WaveformResponse {
  iq_available: number;
  status: "AVAILABLE" | "UNAVAILABLE";
  reason: string;
  metrics: WaveformMetrics | null;
  waveform_series: WaveformSeries | null;
  provenance?: string;
}
