# Wire Watcher — API Prediction Contract

## Provenance Taxonomy

| Label | Meaning |
|---|---|
| `REAL_MEASURED` | Directly recorded by hardware or the data source |
| `DERIVED` | Mathematically computed from `REAL_MEASURED` values |
| `INFERRED/PSEUDO-LABEL` | Proxy label generated from a training-only heuristic |
| `APPLICATION_METADATA` | Context fields not used in the ML model |

---

## POST /api/predict

**Purpose:** Classify a single RF observation as AVAILABLE, OCCUPIED, or UNCERTAIN.

### Request

```json
{
  "frequency_mhz": 120.0,
  "bandwidth_khz": 50.0,
  "signal_strength_dbm": -60.0,
  "iq_available": 0,
  "iq_rms_magnitude": null,
  "iq_magnitude_variance": null,
  "iq_peak_magnitude": null,
  "iq_crest_factor": null,
  "iq_p10": null,
  "iq_p50": null,
  "iq_p90": null,
  "iq_phase_concentration": null,
  "iq_spectral_entropy": null,
  "iq_spectral_peak_ratio": null,
  "timestamp": null,
  "location": null,
  "latitude": null,
  "longitude": null
}
```

#### Field Semantics

| Field | Required | Provenance | Description |
|---|---|---|---|
| `frequency_mhz` | ✅ | `REAL_MEASURED` | RF centre frequency in MHz (must be > 0) |
| `bandwidth_khz` | ✅ | `REAL_MEASURED` | Channel bandwidth in kHz (must be > 0) |
| `signal_strength_dbm` | ✅ | `REAL_MEASURED` | Received signal power in dBm. **Used only by the RF activity detector, NOT as an ML predictor** |
| `iq_available` | ✅ | `REAL_MEASURED` | 1 if I/Q samples are provided; 0 if absent |
| `iq_rms_magnitude` | Optional | `DERIVED` | RMS of I/Q magnitude. Null when I/Q unavailable — never fabricated |
| `iq_magnitude_variance` | Optional | `DERIVED` | Variance of I/Q magnitude |
| `iq_peak_magnitude` | Optional | `DERIVED` | Maximum I/Q magnitude |
| `iq_crest_factor` | Optional | `DERIVED` | Peak / RMS ratio |
| `iq_p10` | Optional | `DERIVED` | 10th-percentile magnitude |
| `iq_p50` | Optional | `DERIVED` | 50th-percentile magnitude |
| `iq_p90` | Optional | `DERIVED` | 90th-percentile magnitude |
| `iq_phase_concentration` | Optional | `DERIVED` | Circular mean of phase |
| `iq_spectral_entropy` | Optional | `DERIVED` | Normalised FFT spectral entropy |
| `iq_spectral_peak_ratio` | Optional | `DERIVED` | Dominant FFT bin / total power |
| `timestamp` | Optional | `APPLICATION_METADATA` | ISO-8601 observation time |
| `location` | Optional | `APPLICATION_METADATA` | Free-text location string |
| `latitude` | Optional | `APPLICATION_METADATA` | Decimal degrees |
| `longitude` | Optional | `APPLICATION_METADATA` | Decimal degrees |

#### Validation Rules

- `frequency_mhz` must be > 0
- `bandwidth_khz` must be > 0
- `signal_strength_dbm` must be a finite float
- `iq_available` must be 0 or 1
- I/Q feature fields are optional (null = no I/Q); the ML pipeline imputes missing values
- NaN and Infinity are rejected
- Missing required fields → HTTP 400

---

### Response

```json
{
  "prediction": "OCCUPIED",
  "available": false,
  "activity": "DETECTED",
  "availability": "OCCUPIED",
  "probability": 0.58,
  "occupied_probability": 0.58,
  "ml_activity_probability": 0.58,
  "confidence": "Low",
  "ml_activity": true,
  "detector_activity": true,
  "ood_warning": false,
  "warning": null,
  "data_source": "RF Signal Data — real measured observations + derived I/Q features + inferred activity pseudo-label",
  "label_type": "INFERRED_RF_ACTIVITY",
  "threshold": 0.435,
  "target_definition": "Training-only per-frequency median of recorded signal_strength_dbm defines a detectable-RF-activity proxy.",
  "model_version": "rf-signal-activity-v2",
  "live_rf_ingestion": false,
  "features_used": { ... },
  "important_features": ["num__iq_p10", "num__iq_spectral_peak_ratio", ...]
}
```

#### Response Field Semantics

| Field | Provenance | Description |
|---|---|---|
| `prediction` | OUTPUT | Same as `availability` |
| `available` | OUTPUT | Boolean shorthand: true only when `availability == "AVAILABLE"` |
| `activity` | OUTPUT | `DETECTED` / `NOT_DETECTED` / `UNCERTAIN` |
| `availability` | OUTPUT | **Primary decision**: `AVAILABLE` / `OCCUPIED` / `UNCERTAIN` |
| `probability` | OUTPUT | ML probability of `DETECTABLE_RF_ACTIVITY` (class 1) |
| `ml_activity_probability` | OUTPUT | Same as `probability` |
| `confidence` | OUTPUT | `High` / `Medium` / `Low` / `OOD / Unreliable` — based on distance from 0.5 |
| `ml_activity` | OUTPUT | True if probability ≥ threshold |
| `detector_activity` | OUTPUT | True if `signal_strength_dbm ≥ training-derived per-frequency median` |
| `ood_warning` | OUTPUT | True if observation is outside the model's training distribution |
| `warning` | OUTPUT | Human-readable OOD reason(s), or null |
| `data_source` | `APPLICATION_METADATA` | Source provenance string |
| `label_type` | `APPLICATION_METADATA` | Always `INFERRED_RF_ACTIVITY` |
| `threshold` | OUTPUT | Probability threshold used for `ml_activity` |
| `live_rf_ingestion` | `APPLICATION_METADATA` | Always false (no hardware SDR) |
| `features_used` | `DERIVED` | Feature dict sent to the ML model |
| `important_features` | `APPLICATION_METADATA` | Top-5 feature importances from model metadata |

---

## Prediction States

### AVAILABLE
- OOD: false
- detector_activity: false (signal below training median for that frequency)

### OCCUPIED
- OOD: false
- detector_activity: true (signal at or above training median)

### UNCERTAIN
- OOD: true (unknown frequency, extreme values, or multivariate distance exceeded)
- OR detector threshold not available for the requested frequency

> [!IMPORTANT]
> OOD observations are always UNCERTAIN. The system never converts uncertainty into AVAILABLE.

---

## GET /api/predictions

Returns the last 100 stored predictions (newest first).

```json
{
  "predictions": [
    {
      "id": 42,
      "start_frequency_mhz": 119.975,
      "end_frequency_mhz": 120.025,
      "bandwidth_mhz": 0.05,
      "available": false,
      "probability": 0.58,
      "timestamp": "2026-08-31T18:00:00",
      "signal_power_dbm": -60.0,
      "noise_floor_dbm": null,
      "snr_db": null,
      "ood_status": false,
      "data_source": "RF Signal Data — real measured ...",
      "activity": "DETECTED",
      "availability": "OCCUPIED",
      "confidence": "Low",
      "label_type": "INFERRED_RF_ACTIVITY",
      "model_version": "rf-signal-activity-v2"
    }
  ]
}
```

> [!NOTE]
> `noise_floor_dbm` and `snr_db` are NULL for predictions made via `/api/predict`.
> They are only populated when a future hardware RF source provides them.

---

## GET /api/health

```json
{
  "status": "ok",
  "model_loaded": true,
  "database_connected": true,
  "api": "ok",
  "model": "loaded",
  "database": "connected"
}
```

---

## GET /api/model-info

Returns model metadata + database KPIs. Key fields:

```json
{
  "model_name": "Wire Watcher RF Activity Model",
  "algorithm": "RandomForest",
  "model_version": "rf-signal-activity-v2",
  "target": "inferred_rf_activity",
  "label_type": "INFERRED/PSEUDO-LABEL",
  "target_is_ground_truth": false,
  "ground_truth_occupancy": false,
  "live_rf_ingestion": false
}
```

---

## What This System Is NOT

- ❌ Not ground-truth spectrum occupancy measurement
- ❌ Not live SDR hardware monitoring
- ❌ Not regulatory compliance certification
- ❌ Not guaranteed channel availability

## What This System IS

- ✅ ML-based RF activity detection using real measured SDR data
- ✅ Inferred/pseudo-label target derived from per-frequency signal-strength medians
- ✅ Conservative OOD detection (flags unknown inputs as UNCERTAIN)
- ✅ Honest performance reporting (ROC-AUC ≈ 0.50 — approximately chance-level)
- ✅ Leakage-safe: `signal_strength_dbm` excluded from ML predictors
