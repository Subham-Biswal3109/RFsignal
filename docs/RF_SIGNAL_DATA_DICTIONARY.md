# RF Signal Data Dictionary

The dictionary intentionally distinguishes measured source observations from derived features and application metadata. A source field is not automatically treated as an independent RF measurement.

| Column | Classification | Unit | Meaning | Used in ML? | Reason |
|---|---|---|---|---|---|
| Timestamp | REAL_MEASURED | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | Derived | Excluded unless explicitly used above |
| Frequency | REAL_MEASURED | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | Yes | Excluded unless explicitly used above |
| Signal Strength | REAL_MEASURED | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| Modulation | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| Bandwidth | REAL_MEASURED | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | Yes | Excluded unless explicitly used above |
| Location | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| Device Type | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| Antenna Type | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| Temperature | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| Humidity | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| Wind Speed | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| Precipitation | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| Weather Condition | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| Interference Type | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| Battery Level | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| Power Source | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| CPU Usage | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| Memory Usage | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| WiFi Strength | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| Disk Usage | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| System Load | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| Latitude | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| Longitude | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| Altitude(m) | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| Air Pressure | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| Device Status | APPLICATION_METADATA | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | No | Excluded unless explicitly used above |
| I/Q Data | REAL_MEASURED | source-reported / not independently calibrated | Recorded source field; exact semantics follow dataset naming only | Derived | Excluded unless explicitly used above |
| `iq_rms_magnitude` | DERIVED | dimensionless | RMS of complex-sample magnitude | Yes | Deterministic I/Q feature |
| `iq_magnitude_variance` | DERIVED | dimensionless | Variance of magnitude | Yes | Deterministic I/Q feature |
| `iq_peak_magnitude` | DERIVED | dimensionless | Maximum magnitude | Yes | Deterministic I/Q feature |
| `iq_crest_factor` | DERIVED | dimensionless | Peak magnitude / RMS magnitude | Yes | Deterministic I/Q feature |
| `iq_p10`, `iq_p50`, `iq_p90` | DERIVED | dimensionless | Magnitude percentiles | Yes | Robust I/Q summaries |
| `iq_phase_concentration` | DERIVED | [0,1] | Circular phase concentration | Yes | Phase summary |
| `iq_spectral_entropy` | DERIVED | normalized bits | Entropy of normalized FFT power distribution | Yes | Relative spectral shape only |
| `iq_spectral_peak_ratio` | DERIVED | [0,1] | Largest FFT-power bin / total power | Yes | Relative spectral concentration only |
| `inferred_rf_activity` | INFERRED/PSEUDO-LABEL | binary | Detectable RF activity proxy | TARGET | Training-only per-frequency signal-strength threshold |

## Important exclusions
`Signal Strength` is real measured source data but is excluded from the ML predictor set because it generates the pseudo-label. It remains available to the operational detector.

`Modulation`, `Interference Type`, device/system fields, location and weather fields are treated as application/source metadata rather than occupancy ground truth.

No dB-to-dBm conversion is performed. No missing I/Q is fabricated.
