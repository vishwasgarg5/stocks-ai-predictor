"""Stage 10.6 quality controls built on the existing GitHub-persisted ledger.

This module adds decision-quality measurements without changing the core predictor:
- exact Prediction_ID tracking
- previous-close naive baseline
- horizon-specific live accuracy
- adaptive confidence from recent live evidence
- conservative long-horizon target confirmation for Portfolio Manager
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .config import EVALUATIONS_DIR,PREDICTIONS_DIR,METRICS_DIR,STATE_DIR,MULTI_HORIZONS,TRANSACTION_COST_BPS,SLIPPAGE_BPS,MODEL_VERSION
from .ledger import load_predictions

BASELINE_FILE=METRICS_DIR/"baseline_evaluations.csv"
HORIZON_FILE=METRICS_DIR/"horizon_accuracy.csv"
CONFIDENCE_FILE=METRICS_DIR/"adaptive_confidence.csv"
TARGET_POLICY_FILE=METRICS_DIR/"portfolio_target_policy.csv"


def _num(v,default=np.nan):
    try:
        x=float(v);return x if np.isfinite(x) else default
    except Exception:return default


def evaluate_baseline(evaluation: pd.DataFrame, predictions: pd.DataFrame|None=None) -> pd.DataFrame:
    """Compare AI close forecast with previous-close baseline after costs."""
    if evaluation is None or evaluation.empty:return pd.DataFrame()
    x=evaluation.copy()
    if predictions is not None and not predictions.empty and "Prediction_ID" in predictions.columns:
        keep=[c for c in ["Prediction_ID","Baseline_Close","Baseline_Cost_Pct"] if c in predictions.columns]
        if keep:x=x.merge(predictions[keep].drop_duplicates("Prediction_ID"),on="Prediction_ID",how="left")
    if "Baseline_Close" not in x.columns:
        x["Baseline_Close"]=pd.to_numeric(x.get("Previous_Close",x.get("Current_Close",np.nan)),errors="coerce")
    x["Baseline_APE"]=(pd.to_numeric(x["Actual_Close"],errors="coerce")-pd.to_numeric(x["Baseline_Close"],errors="coerce")).abs()/pd.to_numeric(x["Actual_Close"],errors="coerce").abs().clip(lower=1e-8)*100
    x["AI_APE"]=pd.to_numeric(x.get("APE_Close"),errors="coerce").abs()
    x["CostPct"]=pd.to_numeric(x.get("Baseline_Cost_Pct",(TRANSACTION_COST_BPS+SLIPPAGE_BPS)/100),errors="coerce").fillna((TRANSACTION_COST_BPS+SLIPPAGE_BPS)/100)
    x["AI_Better_Than_Baseline"]=x["AI_APE"]+x["CostPct"] < x["Baseline_APE"]
    cols=[c for c in ["Prediction_ID","MarketDate","PredictionDate","Symbol","AI_APE","Baseline_APE","CostPct","AI_Better_Than_Baseline"] if c in x.columns]
    out=x[cols].copy()
    if not out.empty:
        existing=pd.read_csv(BASELINE_FILE) if BASELINE_FILE.exists() else pd.DataFrame()
        combined=pd.concat([existing,out],ignore_index=True) if not existing.empty else out
        keys=[c for c in ["Prediction_ID","MarketDate","Symbol"] if c in combined.columns]
        if keys:combined=combined.drop_duplicates(keys,keep="last")
        combined.to_csv(BASELINE_FILE,index=False)
    return out


def horizon_accuracy() -> pd.DataFrame:
    """Return live 1/3/5/7/20D accuracy, direction and baseline edge."""
    path=EVALUATIONS_DIR/"horizon_evaluations.csv"
    if not path.exists():return pd.DataFrame()
    try:x=pd.read_csv(path)
    except Exception:return pd.DataFrame()
    if x.empty:return pd.DataFrame()
    rows=[]
    for h,g in x.groupby("HorizonDays"):
        ape=pd.to_numeric(g["APE"],errors="coerce").abs().dropna()
        direction=pd.to_numeric(g["DirectionCorrect"],errors="coerce").dropna()
        rows.append({"HorizonDays":int(h),"Samples":len(g),"MAPE":float(ape.mean()) if len(ape) else np.nan,"Accuracy":float((1-ape/100).clip(0,1).mean()*100) if len(ape) else np.nan,"DirectionAccuracy":float(direction.mean()*100) if len(direction) else np.nan})
    out=pd.DataFrame(rows).sort_values("HorizonDays")
    out.to_csv(HORIZON_FILE,index=False)
    return out


def adaptive_confidence(symbol: str|None=None, horizon: int=1) -> dict:
    """Confidence is evidence-weighted: recent live MAPE/direction/baseline edge plus uncertainty.

    Historical validation is represented by the prediction confidence itself; live evidence
    is intentionally capped so a small recent sample cannot overreact.
    """
    frames=[]
    for p in sorted(EVALUATIONS_DIR.glob("evaluation_*.csv")):
        try:
            d=pd.read_csv(p)
            if not d.empty:frames.append(d)
        except Exception:pass
    if not frames:return {"Confidence":50.0,"Samples":0,"Reason":"NO_LIVE_EVIDENCE"}
    d=pd.concat(frames,ignore_index=True)
    if symbol and "Symbol" in d.columns:d=d[d["Symbol"].astype(str)==str(symbol)]
    recent=d.tail(50)
    n=len(recent)
    mape=_num(recent.get("APE_Close",pd.Series(dtype=float)).abs().mean(),10.0)
    direction=_num(pd.to_numeric(recent.get("DirectionCorrect",pd.Series(dtype=float)),errors="coerce").mean()*100,50.0)
    baseline_edge=50.0
    if BASELINE_FILE.exists():
        try:
            b=pd.read_csv(BASELINE_FILE)
            if symbol and "Symbol" in b:b=b[b["Symbol"].astype(str)==str(symbol)]
            b=b.tail(50)
            if not b.empty:baseline_edge=float(pd.to_numeric(b["AI_Better_Than_Baseline"],errors="coerce").mean()*100)
        except Exception:pass
    # Map recent evidence to 0-100 and shrink toward 50 when sample count is small.
    error_score=max(0.0,min(100.0,100.0-mape*10.0))
    raw=0.35*error_score+0.25*direction+0.20*baseline_edge+0.20*50.0
    evidence=min(1.0,n/30.0)
    confidence=50.0+(raw-50.0)*evidence
    result={"Symbol":symbol or "ALL","HorizonDays":int(horizon),"Confidence":round(float(np.clip(confidence,0,100)),2),"RecentMAPE":round(float(mape),3),"DirectionAccuracy":round(float(direction),2),"BaselineWinRate":round(float(baseline_edge),2),"Samples":n,"ModelVersion":MODEL_VERSION}
    existing=pd.read_csv(CONFIDENCE_FILE) if CONFIDENCE_FILE.exists() else pd.DataFrame()
    out=pd.DataFrame([result]);combined=pd.concat([existing,out],ignore_index=True) if not existing.empty else out
    combined=combined.drop_duplicates(["Symbol","HorizonDays"],keep="last");combined.to_csv(CONFIDENCE_FILE,index=False)
    return result


def confirm_portfolio_target(forecasts: dict[int,float], minimum_target: float=10.0) -> dict:
    """Choose a conservative portfolio target from 60/90/180/365D forecasts.

    A >10% target is accepted only when multiple long horizons confirm it. Otherwise
    the target is capped at the strongest confirmed level, never at a single optimistic forecast.
    """
    clean={int(h):_num(v) for h,v in forecasts.items() if _num(v) is not np.nan}
    clean={h:v for h,v in clean.items() if np.isfinite(v)}
    long=[(h,v) for h,v in sorted(clean.items()) if h in (60,90,180,365)]
    if not long:return {"TargetPct":minimum_target,"Status":"NO_LONG_HORIZON_EVIDENCE","ConfirmedHorizons":[]}
    confirmed=[]
    for h,v in long:
        support=sum(1 for h2,v2 in long if h2>=h and v2>=v*0.80 and v2>=minimum_target)
        if v>=minimum_target and support>=2:confirmed.append((h,v))
    if confirmed:
        # Conservative: use the lowest return among confirming horizons, not the peak.
        target=min(v for _,v in confirmed)
        return {"TargetPct":round(float(max(minimum_target,target)),2),"Status":"MULTI_HORIZON_CONFIRMED","ConfirmedHorizons":[h for h,_ in confirmed]}
    best=max(v for _,v in long)
    return {"TargetPct":minimum_target if best>=minimum_target else max(0.0,best),"Status":"SINGLE_HORIZON_OR_INCONSISTENT","ConfirmedHorizons":[]}


def run_quality_controls(market_date: str|None=None) -> dict:
    """Run post-market quality calculations and persist GitHub state."""
    result={"ModelVersion":MODEL_VERSION,"MarketDate":market_date}
    if market_date:
        path=EVALUATIONS_DIR/f"evaluation_{market_date}.csv"
        if path.exists():
            evaluation=pd.read_csv(path);prediction_date=str(evaluation["PredictionDate"].iloc[0]) if "PredictionDate" in evaluation and not evaluation.empty else market_date
            predictions=load_predictions(pd.Timestamp(prediction_date).date())
            base=evaluate_baseline(evaluation,predictions);result["BaselineSamples"]=len(base);result["BaselineWinRate"]=float(base["AI_Better_Than_Baseline"].mean()*100) if not base.empty else None
    h=horizon_accuracy();result["HorizonMetrics"]=h.to_dict("records") if not h.empty else []
    result["AdaptiveConfidence"]=adaptive_confidence()
    STATE_DIR.mkdir(parents=True,exist_ok=True);(STATE_DIR/"quality_controls.json").write_text(json.dumps(result,indent=2,default=str))
    return result

if __name__=="__main__":
    import sys
    run_quality_controls(sys.argv[1] if len(sys.argv)>1 else None)
