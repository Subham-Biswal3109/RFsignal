import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "wire_watcher_model.pkl"


def load_model(model_path=MODEL_PATH):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    return joblib.load(model_path)


def ood_check(input_data: dict, bundle: dict):
    reasons=[]
    for col,b in bundle.get("ood_bounds",{}).items():
        v=input_data.get(col)
        if v is None or not np.isfinite(float(v)): continue
        if float(v) < b["p1"] or float(v) > b["p99"]:
            reasons.append(f"{col}={v} outside training 1st-99th percentile [{b['p1']:.4g}, {b['p99']:.4g}]")
    # Diagonal Mahalanobis distance in the trained standardized feature space.
    try:
        pre=bundle["pipeline"].named_steps["preprocessor"]
        x=pre.transform(pd.DataFrame([input_data])[bundle["feature_columns"]])
        x=np.asarray(x.toarray() if hasattr(x,"toarray") else x,dtype=float).ravel()
        mu=np.asarray(bundle.get("ood_feature_means",[]),dtype=float)
        sd=np.asarray(bundle.get("ood_feature_stds",[]),dtype=float)
        if len(mu)==len(x)==len(sd):
            sd=np.where(sd<1e-9,1.0,sd)
            distance=float(np.sqrt(np.sum(((x-mu)/sd)**2)))
            limit=float(bundle.get("ood_distance_threshold",np.inf))
            if distance > limit:
                reasons.append(f"multivariate RF distance {distance:.3f} exceeds training threshold {limit:.3f}")
    except Exception as exc:
        reasons.append(f"OOD distance unavailable: {exc}")
    return bool(reasons), reasons


import sys
from backend.rf.dsp_activity_detector import DSPActivityDetector

_dsp_detector = DSPActivityDetector()


def detector_activity(signal_strength_dbm: float, frequency_mhz: float, bundle: dict):
    thresholds = bundle.get("signal_strength_activity_thresholds_dbm", {})
    key = str(float(frequency_mhz))
    if key not in thresholds:
        # Never invent a threshold for an unseen frequency.
        return None, f"No trained activity threshold for frequency {frequency_mhz} MHz"
    threshold = float(thresholds[key])
    return bool(float(signal_strength_dbm) >= threshold), None


def predict(input_data: dict, bundle: dict):
    feature_columns = bundle["feature_columns"]
    df = pd.DataFrame([input_data])
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required features: {missing}")

    pipe = bundle["pipeline"]
    probability = float(pipe.predict_proba(df[feature_columns])[0, 1])
    ml_activity = bool(probability >= float(bundle.get("threshold_occupied", 0.5)))

    # 1. Deterministic Physical RF/DSP Detector
    dsp_eval = _dsp_detector.evaluate_observation(
        frequency_mhz=float(input_data["frequency_mhz"]),
        bandwidth_khz=float(input_data.get("bandwidth_khz", 200.0)),
        signal_strength_dbm=float(input_data["signal_strength_dbm"]),
        psd_peak_dbm=input_data.get("psd_peak_dbm"),
        iq_available=bool(input_data.get("iq_available", 0)),
        iq_rms_magnitude=input_data.get("iq_rms_magnitude"),
        iq_spectral_entropy=input_data.get("iq_spectral_entropy"),
    )

    detector, detector_reason = detector_activity(
        input_data["signal_strength_dbm"], input_data["frequency_mhz"], bundle
    )

    # 2. Out-of-Distribution (OOD) Safety Guard
    ood, reasons = ood_check(input_data, bundle)
    if detector_reason:
        reasons.append(detector_reason)
        ood = True

    # 3. Evidence Fusion & Final Availability Decision
    if ood:
        availability = "UNCERTAIN"
        activity = "UNCERTAIN"
        fusion_summary = "OOD safety override active; decision forced to UNCERTAIN."
    else:
        # Physical DSP evidence is primary
        activity = "DETECTED" if detector else "NOT_DETECTED"
        availability = "OCCUPIED" if detector else "AVAILABLE"
        fusion_summary = f"DSP detector ({dsp_eval['activity_state']}) primary evidence + ML probability ({probability:.2%}) supporting evidence."

    return {
        "activity": activity,
        "availability": availability,
        "confidence": probability if not ood else min(probability, 1.0 - probability),
        "ml_activity_probability": probability,
        "ml_activity": ml_activity,
        "detector_activity": detector,
        "dsp_evidence": dsp_eval,
        "ood": ood,
        "ood_reasons": reasons,
        "threshold": float(bundle.get("threshold_occupied", 0.5)),
        "label_type": "INFERRED_RF_ACTIVITY",
        "ground_truth": "UNVERIFIED",
        "evidence_fusion_summary": fusion_summary,
    }


def main():
    parser=argparse.ArgumentParser(description="Wire Watcher RF activity inference")
    parser.add_argument("--frequency",type=float,required=True); parser.add_argument("--bandwidth",type=float,required=True); parser.add_argument("--signal",type=float,required=True)
    parser.add_argument("--iq-available",type=int,choices=[0,1],default=0)
    args=parser.parse_args(); b=load_model()
    x={"frequency_mhz":args.frequency,"bandwidth_khz":args.bandwidth,"signal_strength_dbm":args.signal,"iq_available":args.iq_available}
    for c in b["feature_columns"]:
        x.setdefault(c,np.nan)
    print(predict(x,b))

if __name__=="__main__": main()
