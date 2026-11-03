"""End-to-end live API smoke test.  Run after starting app.py."""
import urllib.request
import json
import time

BASE = "http://localhost:5000"

TESTS = [
    ("Health check",  "GET",  "/api/health", None),
    ("Model info",    "GET",  "/api/model-info", None),
    ("1-Normal RF",   "POST", "/api/predict", {
        "frequency_mhz": 120.0, "bandwidth_khz": 50.0,
        "signal_strength_dbm": -60.0, "iq_available": 0}),
    ("2-Strong RF",   "POST", "/api/predict", {
        "frequency_mhz": 90.0, "bandwidth_khz": 50.0,
        "signal_strength_dbm": -30.0, "iq_available": 0}),
    ("3-Weak RF",     "POST", "/api/predict", {
        "frequency_mhz": 70.0, "bandwidth_khz": 50.0,
        "signal_strength_dbm": -110.0, "iq_available": 0}),
    ("4-OOD input",   "POST", "/api/predict", {
        "frequency_mhz": 5000.0, "bandwidth_khz": 50.0,
        "signal_strength_dbm": 30.0, "iq_available": 0}),
    ("5-Invalid",     "POST", "/api/predict", {
        "signal_strength_dbm": -60.0}),  # missing required fields
    ("6-IQ present",  "POST", "/api/predict", {
        "frequency_mhz": 140.0, "bandwidth_khz": 50.0,
        "signal_strength_dbm": -55.0, "iq_available": 1,
        "iq_rms_magnitude": 0.35, "iq_magnitude_variance": 0.02,
        "iq_peak_magnitude": 0.72, "iq_crest_factor": 2.06,
        "iq_p10": 0.12, "iq_p50": 0.31, "iq_p90": 0.58,
        "iq_phase_concentration": 0.45, "iq_spectral_entropy": 0.82,
        "iq_spectral_peak_ratio": 0.04}),
    ("7-IQ missing",  "POST", "/api/predict", {
        "frequency_mhz": 160.0, "bandwidth_khz": 50.0,
        "signal_strength_dbm": -75.0, "iq_available": 0}),
    ("History",       "GET",  "/api/predictions", None),
]

time.sleep(1)
print("=" * 72)
print("WIRE WATCHER — LIVE E2E TEST")
print("=" * 72)

all_ok = True
for name, method, path, payload in TESTS:
    try:
        if payload:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                BASE + path, data=data,
                headers={"Content-Type": "application/json"},
                method=method)
        else:
            req = urllib.request.Request(BASE + path, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                status = r.status
                body = json.loads(r.read())
        except urllib.error.HTTPError as e:
            status = e.code
            body = json.loads(e.read())

        if path == "/api/predict":
            av = body.get("availability", "?")
            ac = body.get("activity", "?")
            ood = body.get("ood_warning", "?")
            prob = body.get("probability")
            prob_s = f"{prob:.3f}" if prob is not None else "?"
            ok = status == 200 if "Invalid" not in name else status == 400
            tag = "PASS" if ok else "FAIL"
            if not ok:
                all_ok = False
            print(f"[{tag}][{status}] {name}: avail={av} act={ac} ood={ood} prob={prob_s}")
        elif path == "/api/health":
            ok = status == 200 and body.get("model_loaded")
            tag = "PASS" if ok else "FAIL"
            if not ok:
                all_ok = False
            print(f"[{tag}][{status}] {name}: {body}")
        elif path == "/api/model-info":
            ok = (status == 200 and
                  body.get("target") == "inferred_rf_activity" and
                  not body.get("ground_truth_occupancy"))
            tag = "PASS" if ok else "FAIL"
            if not ok:
                all_ok = False
            print(f"[{tag}][{status}] {name}: model={body.get('model_version')} "
                  f"ground_truth={body.get('ground_truth_occupancy')}")
        elif path == "/api/predictions":
            count = len(body.get("predictions", []))
            ok = status == 200 and count >= 1
            tag = "PASS" if ok else "FAIL"
            if not ok:
                all_ok = False
            print(f"[{tag}][{status}] {name}: count={count}")
        else:
            print(f"[----][{status}] {name}")
    except Exception as exc:
        print(f"[FAIL][???] {name}: {exc}")
        all_ok = False

print("=" * 72)
print("RESULT:", "ALL PASSED" if all_ok else "SOME FAILURES — see above")
print("=" * 72)
