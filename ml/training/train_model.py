from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.rf_features import ML_FEATURE_COLUMNS

RANDOM_STATE = 42
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data/processed/rf_signal_activity.csv"
ART = BASE / "artifacts"; RES = BASE / "results"
META = ART / "model_metadata.json"
ART.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
FEATURES = ML_FEATURE_COLUMNS
NUM = FEATURES


def chronological_split(df):
    df = df.sort_values("Timestamp").reset_index(drop=True)
    n = len(df); a = int(n * .60); b = int(n * .80)
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {
        "accuracy": float(accuracy_score(y,pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y,pred)),
        "precision_occupied": float(precision_score(y,pred,zero_division=0)),
        "recall_occupied": float(recall_score(y,pred,zero_division=0)),
        "f1_occupied": float(f1_score(y,pred,zero_division=0)),
        "roc_auc": float(roc_auc_score(y,p)),
        "pr_auc": float(average_precision_score(y,p)),
        "brier_score": float(brier_score_loss(y,p)),
        "false_available": int(fn),
        "false_occupied": int(fp),
        "false_available_rate": float(fn / max(1, (np.asarray(y)==1).sum())),
        "confusion_matrix": [[int(tn),int(fp)],[int(fn),int(tp)]],
    }


def choose_threshold(y, p):
    best = None
    for t in np.linspace(.05,.95,181):
        m = metrics(y,p,float(t))
        # Primary objective: minimize false-AVAILABLE rate. Among ties, prefer
        # balanced accuracy, then F1. This is deliberately conservative.
        key = (m["false_available_rate"], -m["balanced_accuracy"], -m["f1_occupied"])
        if best is None or key < best[0]:
            best = (key,float(t),m)
    return best[1], best[2]


def make_pipeline(model):
    pre = ColumnTransformer([("num", Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler())
    ]), NUM)])
    return Pipeline([("preprocessor",pre),("classifier",model)])


def main():
    df = pd.read_csv(DATA, parse_dates=["Timestamp"])
    train,val,test = chronological_split(df)

    # The pseudo-label is created ONLY from training observations. Signal strength
    # is intentionally excluded from the ML predictors because it is the detector
    # variable used to create the label.
    thresholds = train.groupby("frequency_mhz")["signal_strength_dbm"].median().to_dict()
    y_train = train.apply(lambda r: int(r.signal_strength_dbm >= thresholds[r.frequency_mhz]), axis=1).to_numpy()
    y_val = val.apply(lambda r: int(r.signal_strength_dbm >= thresholds[r.frequency_mhz]), axis=1).to_numpy()
    y_test = test.apply(lambda r: int(r.signal_strength_dbm >= thresholds[r.frequency_mhz]), axis=1).to_numpy()
    labeled=df.copy(); labeled["inferred_rf_activity"]=labeled.apply(lambda r: int(r.signal_strength_dbm >= thresholds[r.frequency_mhz]), axis=1); labeled.to_csv(DATA,index=False)

    models = {
        "DummyClassifier": DummyClassifier(strategy="prior"),
        "LogisticRegression": LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=60, max_depth=10, min_samples_leaf=12, class_weight="balanced", n_jobs=4, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=25, max_depth=1, learning_rate=.05, random_state=RANDOM_STATE),
    }
    xgb_available = False


    results={}; fitted={}
    for name,model in models.items():
        pipe=make_pipeline(model)
        pipe.fit(train[FEATURES], y_train)
        pv=pipe.predict_proba(val[FEATURES])[:,1]
        pt=pipe.predict_proba(test[FEATURES])[:,1]
        tv,mv=choose_threshold(y_val,pv)
        mt=metrics(y_test,pt,tv)
        results[name]={"validation_threshold":tv,"validation":mv,"test":mt}
        fitted[name]=pipe
        print(name, json.dumps(mt), flush=True)

    # Model choice: validation ROC-AUC first, then balanced accuracy. This avoids
    # selecting a model solely because it exploits the pseudo-label at a convenient threshold.
    names=[n for n in results if n!="DummyClassifier"]
    best_name=max(names,key=lambda n:(results[n]["validation"]["roc_auc"],results[n]["validation"]["balanced_accuracy"], -({"LogisticRegression":0,"GradientBoosting":1,"RandomForest":2}.get(n,9))))
    best=fitted[best_name]
    pv=best.predict_proba(val[FEATURES])[:,1]
    threshold_occupied,val_metrics=choose_threshold(y_val,pv)
    test_p=best.predict_proba(test[FEATURES])[:,1]
    test_metrics=metrics(y_test,test_p,threshold_occupied)

    # OOD parameters are fitted from training data only.
    ood_bounds={c:{"p1":float(train[c].quantile(.01)),"p99":float(train[c].quantile(.99))} for c in FEATURES}
    X_train_t=best.named_steps["preprocessor"].transform(train[FEATURES])
    mu=np.asarray(X_train_t.mean(axis=0)).ravel()
    # Diagonal robust distance is deliberately used instead of a fragile full
    # covariance inverse on this small/collinear feature space.
    sigma=np.asarray(X_train_t.std(axis=0)).ravel(); sigma[sigma<1e-9]=1.0
    train_z=np.asarray((X_train_t-mu)/sigma)
    train_dist=np.sqrt(np.sum(train_z**2,axis=1))
    ood_distance=float(np.quantile(train_dist,.9975))

    # Feature importances after preprocessing.
    clf=best.named_steps["classifier"]
    names_out=best.named_steps["preprocessor"].get_feature_names_out().tolist()
    vals=clf.feature_importances_ if hasattr(clf,"feature_importances_") else np.abs(clf.coef_[0]) if hasattr(clf,"coef_") else np.zeros(len(names_out))
    importances={k:float(v) for k,v in sorted(zip(names_out,vals),key=lambda z:z[1],reverse=True)[:20]}

    bundle={
        "pipeline":best,
        "feature_columns":FEATURES,
        "target":"inferred_rf_activity",
        "target_encoding":{"0":"NO_DETECTABLE_RF_ACTIVITY","1":"DETECTABLE_RF_ACTIVITY"},
        "threshold_occupied":float(threshold_occupied),
        "signal_strength_activity_thresholds_dbm":{str(k):float(v) for k,v in thresholds.items()},
        "ood_bounds":ood_bounds,
        "ood_distance_threshold":ood_distance,
        "ood_feature_means":mu.tolist(),
        "ood_feature_stds":sigma.tolist(),
    }
    joblib.dump(bundle,ART/"wire_watcher_model.pkl",compress=3)

    metadata={
        "model_version":"rf-signal-activity-v2",
        "training_date":datetime.now(timezone.utc).isoformat(),
        "dataset_name":"RF Signal Data / logged_data.csv",
        "dataset_type":"hybrid_real_measured_plus_derived_plus_inferred_pseudolabel",
        "provenance":"PARTIALLY VERIFIED: source/published evidence reports SDR acquisition; exact CSV lacks a complete calibration/acquisition record.",
        "target":"inferred_rf_activity",
        "target_is_ground_truth":False,
        "target_method":"Training-only per-frequency median of recorded signal_strength_dbm defines a detectable-RF-activity proxy. This is an inferred/pseudo-label, not occupancy ground truth.",
        "target_generation_feature":"signal_strength_dbm",
        "target_generation_feature_used_as_ml_predictor":False,
        "features":FEATURES,
        "excluded_features":["signal_strength_dbm","Timestamp","Modulation","Interference Type","Device Type","Device Status","Location","Latitude","Longitude","Altitude(m)","Battery Level","Power Source","CPU Usage","Memory Usage","WiFi Strength","Disk Usage","System Load","Air Pressure"],
        "split_strategy":"Chronological 60/20/20; no random shuffling. Additional frequency-holdout experiment is recommended.",
        "training_samples":len(train),"validation_samples":len(val),"test_samples":len(test),
        "class_distribution":{"train":{"0":int((y_train==0).sum()),"1":int((y_train==1).sum())},"validation":{"0":int((y_val==0).sum()),"1":int((y_val==1).sum())},"test":{"0":int((y_test==0).sum()),"1":int((y_test==1).sum())}},
        "models_evaluated":results,"final_model":best_name,"xgboost_available":xgb_available,
        "threshold_occupied":float(threshold_occupied),"threshold_available":float(1-threshold_occupied),
        "validation_metrics":val_metrics,"test_metrics":test_metrics,"feature_importances":importances,
        "activity_thresholds_dbm":{str(k):float(v) for k,v in thresholds.items()},
        "ood_method":"Per-feature 1st/99th percentile training bounds plus standardized multivariate Euclidean distance at the 99.75th training percentile. OOD is a conservative uncertainty flag, not a guarantee.",
        "data_source":"RF Signal Data — real measured observations + derived I/Q features + inferred activity pseudo-label",
        "live_rf_ingestion":False,
        "limitations":["No native occupancy ground truth exists.","The activity target is a signal-strength-derived pseudo-label.","signal_strength_dbm is excluded from ML predictors to avoid circular target reproduction.","I/Q acquisition metadata does not verify an absolute FFT frequency axis; spectral features are normalized only.","I/Q is missing in a substantial subset and is not fabricated.","The model predicts inferred RF activity, not regulatory or guaranteed channel vacancy."]
    }
    META.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    (RES/"rf_signal_model_comparison.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print("FINAL",best_name,json.dumps(test_metrics),flush=True)

if __name__=="__main__": main()
