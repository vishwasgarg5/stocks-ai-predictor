"""Shared report metrics: validated accuracy, horizon accuracy, trend and scan health."""
from pathlib import Path
import pandas as pd
import numpy as np
from .config import EVALUATIONS_DIR

def _accuracy_from_mape(mape):
    if mape is None or pd.isna(mape): return None
    return float(np.clip(100.0-float(mape),0,100))

def model_report_metrics():
    files=sorted(EVALUATIONS_DIR.glob("evaluation_*.csv"))
    frames=[]
    for p in files:
        try:
            d=pd.read_csv(p)
            if not d.empty: frames.append(d)
        except Exception: pass
    if not frames: return {"PreviousAccuracy":None,"CurrentAccuracy":None,"AccuracySamples":0,"DirectionAccuracy":None,"Health":"NO VALIDATION","Drift":"UNKNOWN","Trend7D":None,"Trend30D":None}
    all_data=pd.concat(frames,ignore_index=True)
    close_acc=_accuracy_from_mape(all_data["APE_Close"].abs().mean()) if "APE_Close" in all_data else None
    previous=None
    if len(frames)>1:
        prior=pd.concat(frames[:-1],ignore_index=True)
        previous=_accuracy_from_mape(prior["APE_Close"].abs().mean()) if "APE_Close" in prior else None
    # Direction accuracy is always calculated from validated observations.
    direction=float(all_data["DirectionCorrect"].mean()*100) if "DirectionCorrect" in all_data else None
    daily=[]
    for d in frames:
        if "APE_Close" in d: daily.append(float(np.clip(100-d["APE_Close"].abs().mean(),0,100)))
    trend7=float(np.mean(daily[-7:])) if daily else None
    trend30=float(np.mean(daily[-30:])) if daily else None
    health="HEALTHY" if close_acc is not None and close_acc>=75 else ("WATCH" if close_acc is not None and close_acc>=65 else "NEEDS REVIEW")
    drift="LOW"
    if len(daily)>=3 and trend7 is not None and trend30 is not None and abs(trend7-trend30)>=10: drift="HIGH"
    elif len(daily)>=2 and abs(daily[-1]-daily[-2])>=5: drift="MEDIUM"
    out={"PreviousAccuracy":previous,"CurrentAccuracy":close_acc,"AccuracySamples":len(all_data),"DirectionAccuracy":direction,"Health":health,"Drift":drift,"Trend7D":trend7,"Trend30D":trend30}
    # Optional horizon evaluation files/columns are supported without inventing accuracy.
    for h in [1,3,5,7,20]:
        col=f"APE_Close_{h}D"
        if col in all_data: out[f"{h}D"]=_accuracy_from_mape(all_data[col].abs().mean())
    return out
