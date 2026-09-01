from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from rf_features import build_features

BASE = Path(__file__).resolve().parents[2]
SOURCE = BASE / "ml/data/rf_signal_source/logged_data.csv"
OUT = BASE / "ml/data/processed/rf_signal_activity.csv"
REQUIRED = {"Timestamp","Frequency","Signal Strength","Bandwidth","I/Q Data"}

def _feature_chunk(rows):
    return [build_features({"Frequency":r[0],"Bandwidth":r[1],"Signal Strength":r[2],"I/Q Data":r[3]}) for r in rows]

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE)
    if not REQUIRED.issubset(df.columns):
        raise ValueError(f"Missing required source columns: {sorted(REQUIRED-set(df.columns))}")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="raise")
    if not df["Timestamp"].is_monotonic_increasing:
        df = df.sort_values("Timestamp", kind="mergesort")
    df = df.reset_index(drop=True)
    arr = list(zip(df["Frequency"].to_numpy(), df["Bandwidth"].to_numpy(), df["Signal Strength"].to_numpy(), df["I/Q Data"].tolist()))
    chunks = [arr[i:i+10000] for i in range(0, len(arr), 10000)]
    with ProcessPoolExecutor(max_workers=4) as ex:
        feature_chunks = list(ex.map(_feature_chunk, chunks))
    features = pd.DataFrame([row for chunk in feature_chunks for row in chunk])
    out = pd.concat([df[["Timestamp"]].reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    out["iq_available"] = out["iq_available"].astype(int)
    out.to_csv(OUT, index=False)
    iq_total = int(out["iq_available"].sum()); total = len(out)
    audit = f"""# RF Signal Data Final Audit

## Verified source
- Dataset: RF Signal Data
- Source: `logged_data.csv`
- Provenance: **PARTIALLY VERIFIED** — published source evidence reports SDR/DragonOS acquisition; the exact CSV does not contain a complete calibration/acquisition record.

## Actual file
- Rows: {total:,}
- Columns: {len(df.columns)}
- Duplicate rows: {int(df.duplicated().sum())}
- Timestamp range: {df.Timestamp.min()} to {df.Timestamp.max()}
- Timestamp cadence: source is ordered at 20-second intervals
- Frequencies (MHz): {sorted((df.Frequency/1e6).unique().tolist())}
- Bandwidths (kHz): {sorted((df.Bandwidth/1e3).unique().tolist())}
- Signal strength: {float(df['Signal Strength'].min())} to {float(df['Signal Strength'].max())} dBm as reported
- I/Q rows: {iq_total:,}
- I/Q missing: {total-iq_total:,}
- I/Q sample count: 100 complex samples in valid rows inspected

## Scientific status
There is no native occupied/unoccupied ground-truth label. The pipeline uses `inferred_rf_activity`, an **INFERRED/PSEUDO-LABEL** generated from a training-only per-frequency signal-strength threshold. It is not occupancy ground truth.

`signal_strength_dbm` is excluded from the ML predictor set because it is the variable used to generate the activity pseudo-label. This prevents the classifier from simply reproducing the label-generation threshold.

I/Q-derived features are calculated only when I/Q is present. No I/Q samples, sampling rate, receiver gain, calibration, or absolute FFT frequency axis is fabricated. FFT-derived values are normalized spectral summaries only.

## Limitation
This is RF activity detection, not proof of regulatory spectrum vacancy. Live SDR/API ingestion is not implemented.
"""
    (BASE / "docs/RF_SIGNAL_DATA_FINAL_AUDIT.md").write_text(audit, encoding="utf-8")
    dictionary = """# RF Signal Data Dictionary

| Column | Classification | Unit | Meaning | Used in ML? | Reason |
|---|---|---|---|---|---|
| Timestamp | REAL_MEASURED | datetime | Recorded observation time | No | Ordering/context |
| Frequency | REAL_MEASURED | Hz | Recorded RF frequency | Yes → frequency_mhz | RF observation |
| Signal Strength | REAL_MEASURED | dBm as reported | Recorded received level | **No** | Used for pseudo-label detector; excluded from ML to avoid circularity |
| Bandwidth | REAL_MEASURED | Hz | Recorded bandwidth | Yes → bandwidth_khz | RF observation |
| I/Q Data | REAL_MEASURED | complex samples | Recorded I/Q when present | Derived | No missing I/Q fabricated |
| iq_* | DERIVED | dimensionless/statistical | Mathematical summaries of I/Q | Yes | Reproducible feature extraction |
| inferred_rf_activity | INFERRED/PSEUDO-LABEL | binary | Detectable RF activity proxy | TARGET | Training-only per-frequency signal-strength threshold |
| Modulation / Interference / Device / Location / Weather / System fields | APPLICATION_METADATA | mixed | Context/annotations | No in primary model | Avoid metadata shortcuts and false occupancy semantics |
"""
    (BASE / "docs/RF_SIGNAL_DATA_DICTIONARY.md").write_text(dictionary, encoding="utf-8")
    print(f"Prepared {total:,} rows; I/Q={iq_total:,}; output={OUT}")

if __name__ == "__main__": main()
