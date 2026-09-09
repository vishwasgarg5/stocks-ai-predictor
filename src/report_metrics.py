"""Shared report metrics: accuracy, calibration, horizon quality, drift and reliability signals."""
import pandas as pd
import numpy as np
from .config import EVALUATIONS_DIR, DRIFT_THRESHOLD, CONFIDENCE_EVIDENCE_SAMPLES, MULTI_HORIZONS


def _accuracy_from_mape(mape):
    if mape is None or pd.isna(mape):
        return None
    return float(np.clip(100.0 - float(mape), 0, 100))


def _read_evaluations():
    frames = []
    for p in sorted(EVALUATIONS_DIR.glob("evaluation_*.csv")):
        try:
            d = pd.read_csv(p)
            if not d.empty:
                d["_EvaluationFile"] = p.name
                frames.append(d)
        except Exception:
            continue
    return frames


def _confidence_calibration(data):
    if "PredictionConfidence" not in data or "APE_Close" not in data:
        return {"ConfidenceCalibration":"UNKNOWN","ConfidenceECE":None,"ConfidenceBias":None,"ConfidenceSamples":0}
    x = data[["PredictionConfidence", "APE_Close"]].copy()
    x["Confidence"] = pd.to_numeric(x["PredictionConfidence"], errors="coerce").clip(0, 100)
    x["ObservedAccuracy"] = (1 - pd.to_numeric(x["APE_Close"], errors="coerce").abs() / 100).clip(0, 1) * 100
    x = x.dropna(subset=["Confidence", "ObservedAccuracy"])
    if x.empty:
        return {"ConfidenceCalibration":"UNKNOWN","ConfidenceECE":None,"ConfidenceBias":None,"ConfidenceSamples":0}
    # Expected calibration error: confidence should track realized accuracy.
    bins = pd.cut(x["Confidence"], bins=[-0.01, 50, 60, 70, 80, 90, 100.01], labels=False)
    grouped = x.assign(_bin=bins).groupby("_bin", observed=True)
    ece = 0.0
    for _, g in grouped:
        ece += len(g) / len(x) * abs(float(g["Confidence"].mean()) - float(g["ObservedAccuracy"].mean()))
    bias = float((x["Confidence"] - x["ObservedAccuracy"]).mean())
    if len(x) < int(CONFIDENCE_EVIDENCE_SAMPLES):
        status = "LOW EVIDENCE"
    elif ece <= 5:
        status = "CALIBRATED"
    elif ece <= 10:
        status = "WATCH"
    else:
        status = "MIS-CALIBRATED"
    return {"ConfidenceCalibration":status,"ConfidenceECE":float(ece),"ConfidenceBias":bias,"ConfidenceSamples":int(len(x))}


def _drift_metrics(data):
    if "APE_Close" not in data or len(data) < 3:
        return {"Drift":"UNKNOWN","DriftRatio":None,"RecentMAPE":None,"BaselineMAPE":None}
    # Files are sorted by market date; use the latest 5 sessions against the preceding 20.
    recent = pd.to_numeric(data["APE_Close"], errors="coerce").abs().dropna()
    if len(recent) < 3:
        return {"Drift":"UNKNOWN","DriftRatio":None,"RecentMAPE":None,"BaselineMAPE":None}
    recent = recent.tail(min(5, len(recent)))
    baseline = pd.to_numeric(data["APE_Close"], errors="coerce").abs().dropna()
    baseline = baseline.iloc[:-len(recent)].tail(20)
    if baseline.empty:
        return {"Drift":"UNKNOWN","DriftRatio":None,"RecentMAPE":float(recent.mean()),"BaselineMAPE":None}
    r = float(recent.mean()); b = float(baseline.mean()); ratio = (r - b) / max(b, 1e-9)
    if ratio >= float(DRIFT_THRESHOLD): status = "HIGH"
    elif ratio >= float(DRIFT_THRESHOLD) / 2: status = "MEDIUM"
    elif ratio <= -float(DRIFT_THRESHOLD) / 2: status = "IMPROVING"
    else: status = "LOW"
    return {"Drift":status,"DriftRatio":float(ratio),"RecentMAPE":r,"BaselineMAPE":b}


def _horizon_metrics():
    path = EVALUATIONS_DIR / "horizon_evaluations.csv"
    if not path.exists():
        return {}
    try:
        h = pd.read_csv(path)
    except Exception:
        return {}
    if h.empty or "HorizonDays" not in h or "APE" not in h:
        return {}
    out = {}
    for horizon, g in h.groupby("HorizonDays"):
        ape = pd.to_numeric(g["APE"], errors="coerce").abs().dropna()
        if ape.empty:
            continue
        out[f"{int(horizon)}D"] = _accuracy_from_mape(ape.mean())
        out[f"{int(horizon)}DSamples"] = int(len(ape))
        if "DirectionCorrect" in g:
            out[f"{int(horizon)}DDirectionAccuracy"] = float(pd.to_numeric(g["DirectionCorrect"], errors="coerce").mean() * 100)
    return out


def _bucket_metrics(data):
    if "PriceBucket" not in data or "APE_Close" not in data:
        return {}
    out = {}
    for bucket, g in data.groupby(data["PriceBucket"].astype(str)):
        ape = pd.to_numeric(g["APE_Close"], errors="coerce").abs().dropna()
        if not ape.empty:
            key = "Bucket_" + bucket.replace("-", "_").replace(">", "gt")
            out[f"{key}_Accuracy"] = _accuracy_from_mape(ape.mean())
            out[f"{key}_Samples"] = int(len(ape))
    return out


def model_report_metrics():
    frames = _read_evaluations()
    if not frames:
        return {"PreviousAccuracy":None,"CurrentAccuracy":None,"AccuracySamples":0,"DirectionAccuracy":None,"Health":"NO VALIDATION","Drift":"UNKNOWN","Trend7D":None,"Trend30D":None,"ConfidenceCalibration":"UNKNOWN","ConfidenceECE":None,"ConfidenceBias":None,"ConfidenceSamples":0}
    all_data = pd.concat(frames, ignore_index=True)
    close = pd.to_numeric(all_data.get("APE_Close", pd.Series(dtype=float)), errors="coerce").abs().dropna()
    close_acc = _accuracy_from_mape(close.mean()) if not close.empty else None
    previous = None
    if len(frames) > 1:
        prior = pd.concat(frames[:-1], ignore_index=True)
        p = pd.to_numeric(prior.get("APE_Close", pd.Series(dtype=float)), errors="coerce").abs().dropna()
        previous = _accuracy_from_mape(p.mean()) if not p.empty else None
    direction = None
    if "DirectionCorrect" in all_data:
        direction = float(pd.to_numeric(all_data["DirectionCorrect"], errors="coerce").mean() * 100)
    daily = []
    for d in frames:
        a = pd.to_numeric(d.get("APE_Close", pd.Series(dtype=float)), errors="coerce").abs().dropna()
        if not a.empty:
            daily.append(float(np.clip(100 - a.mean(), 0, 100)))
    trend7 = float(np.mean(daily[-7:])) if daily else None
    trend30 = float(np.mean(daily[-30:])) if daily else None
    drift = _drift_metrics(all_data)
    calibration = _confidence_calibration(all_data)
    if close_acc is None:
        health = "NO VALIDATION"
    elif close_acc >= 75 and drift["Drift"] not in {"HIGH", "MEDIUM"} and calibration["ConfidenceCalibration"] != "MIS-CALIBRATED":
        health = "HEALTHY"
    elif close_acc >= 65:
        health = "WATCH"
    else:
        health = "NEEDS REVIEW"
    out = {"PreviousAccuracy":previous,"CurrentAccuracy":close_acc,"AccuracySamples":int(len(all_data)),"DirectionAccuracy":direction,"Health":health,"Trend7D":trend7,"Trend30D":trend30}
    out.update(drift)
    out.update(calibration)
    out.update(_horizon_metrics())
    out.update(_bucket_metrics(all_data))
    if "Symbol" in all_data:
        stock = all_data.assign(_ape=close).groupby("Symbol")["_ape"].mean().sort_values(ascending=False)
        if not stock.empty:
            out["WorstStock"] = str(stock.index[0])
            out["WorstStockMAPE"] = float(stock.iloc[0])
    return out
