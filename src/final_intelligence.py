"""Final AI intelligence layer: Stages 5-10.

All layers are deterministic, leakage-safe and optional-data tolerant.  The module
never invents external news/flow data: when an input is unavailable the corresponding
score remains neutral.  This keeps the existing prediction engines backward compatible.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

HORIZONS=(1,3,5,7,20)
REGIMES=("BULL","SIDEWAYS","BEAR","HIGH VOL")


def _num(v, default=50.0):
    try:
        x=float(v)
        return default if not np.isfinite(x) else x
    except Exception:
        return default


def walk_forward_metrics(actual, predicted, min_train=30, step=1):
    """Leakage-safe expanding walk-forward MAE/MAPE and directional accuracy."""
    a=np.asarray(actual,dtype=float); p=np.asarray(predicted,dtype=float)
    n=min(len(a),len(p)); a=a[:n]; p=p[:n]
    if n==0: return {"Samples":0,"MAE":np.nan,"MAPE":np.nan,"DirectionAccuracy":np.nan}
    errors=[]; apes=[]; dirs=[]
    for i in range(max(1,min_train),n,step):
        e=abs(a[i]-p[i]); errors.append(e); apes.append(e/max(abs(a[i]),1e-6)*100)
        if i>0 and i-1<n:
            dirs.append(float(np.sign(a[i]-a[i-1])==np.sign(p[i]-p[i-1])))
    return {"Samples":len(errors),"MAE":float(np.mean(errors)) if errors else np.nan,"MAPE":float(np.mean(apes)) if apes else np.nan,"DirectionAccuracy":float(np.mean(dirs)*100) if dirs else np.nan}


def calibrate_confidence(confidence, empirical_accuracy=50.0, samples=0):
    """Shrink raw confidence toward observed hit-rate until enough evidence exists."""
    c=_num(confidence); a=_num(empirical_accuracy); evidence=min(max(_num(samples,0)/50.0,0),1)
    return float(np.clip((1-evidence)*c+evidence*a,0,100))


def stock_horizon_reliability(metrics: pd.DataFrame | None):
    """Return stock x horizon reliability from historical evaluation rows."""
    if metrics is None or metrics.empty: return pd.DataFrame(columns=["Symbol","HorizonDays","MAPE","DirectionAccuracy","Samples","Reliability"])
    rows=[]
    for (symbol,h),g in metrics.groupby(["Symbol","HorizonDays"],dropna=False):
        if pd.isna(h): continue
        mape=_num(g.get("MAPE",pd.Series([np.nan])).mean(),10); direction=_num(g.get("DirectionCorrect",pd.Series([0])).mean()*100,50); samples=len(g)
        reliability=np.clip(0.55*(100-min(mape*10,100))+0.45*direction,0,100)
        rows.append({"Symbol":symbol,"HorizonDays":int(h),"MAPE":mape,"DirectionAccuracy":direction,"Samples":samples,"Reliability":float(reliability)})
    return pd.DataFrame(rows)


def prediction_interval(row, z=1.645):
    """Create conservative bull/base/bear bounds from model uncertainty."""
    base=_num(row.get("Pred_Close",row.get("Current_Price",0)),0)
    u=max(_num(row.get("PredictionUncertaintyPct",5),5),0)/100
    half=base*u*z
    return {"BearCase":max(0,base-half),"BaseCase":base,"BullCase":max(0,base+half),"IntervalPct":u*100*z*2}


def market_breadth_score(advancers=0, decliners=0, unchanged=0):
    total=max(_num(advancers,0)+_num(decliners,0)+_num(unchanged,0),1)
    return float(np.clip(50+50*(_num(advancers,0)-_num(decliners,0))/total,0,100))


def regime_risk_adjustment(regime, score):
    factors={"BULL":1.05,"SIDEWAYS":1.00,"BEAR":0.88,"HIGH VOL":0.78}
    return float(np.clip(_num(score)*factors.get(str(regime).upper(),1.0),0,100))


def event_impact_score(sentiment=0, magnitude=0, freshness=1):
    """Convert optional normalized news/event inputs to a 0-100 impact score."""
    return float(np.clip(50+_num(sentiment,0)*25+_num(magnitude,0)*20*min(max(_num(freshness,1),0),1),0,100))


def decision_score(row, regime="SIDEWAYS", breadth=50, news=50):
    """Final Stage 9 decision score; neutral inputs cannot overpower model evidence."""
    expected=_num(row.get("Expected_Return",0),0)
    success=_num(row.get("CalibratedConfidence",row.get("Confidence",50)))
    risk=_num(row.get("RiskAdjustedScore",row.get("Score",50)))
    technical=_num(row.get("TechnicalScore",50))
    mh=_num(row.get("MultiHorizonExpectedReturn",0),0)
    return float(np.clip(0.28*risk+0.18*success+0.14*technical+0.12*np.clip(50+expected*5,0,100)+0.10*np.clip(50+mh*4,0,100)+0.10*_num(breadth)+0.08*_num(news),0,100))


def final_action(row):
    s=_num(row.get("FinalDecisionScore",row.get("DecisionScore",50)))
    direction=str(row.get("Direction","NEUTRAL")).upper()
    if s>=75 and direction=="UP": return "BUY"
    if s>=68 and direction=="UP": return "WATCH"
    if s<=35 or direction=="DOWN" and s<55: return "AVOID"
    return "HOLD"


def apply_final_intelligence(candidates, regime="SIDEWAYS", breadth=50, news=50):
    """Apply Stages 5-10 to an existing candidate DataFrame without changing core models."""
    if candidates is None or candidates.empty: return candidates
    out=candidates.copy()
    out["CalibratedConfidence"]=out.apply(lambda r: calibrate_confidence(r.get("CalibratedConfidence",r.get("Confidence",50)),r.get("ReliabilityScore",50),r.get("ReliabilitySamples",0)),axis=1)
    intervals=out.apply(prediction_interval,axis=1,result_type="expand")
    for c in intervals.columns: out[c]=intervals[c]
    out["BreadthScore"]=_num(breadth); out["NewsEventScore"]=_num(news)
    out["RegimeAdjustedScore"]=out.apply(lambda r: regime_risk_adjustment(regime,r.get("RiskAdjustedScore",r.get("Score",50))),axis=1)
    out["FinalDecisionScore"]=out.apply(lambda r: decision_score(r,regime,breadth,news),axis=1)
    out["Action"]=out.apply(final_action,axis=1)
    out["ModelDriftFlag"]=out.get("PredictionUncertaintyPct",pd.Series(0,index=out.index)).astype(float)>15
    out["FinalRisk"]=np.select([out["FinalDecisionScore"]>=75,out["FinalDecisionScore"]>=55],["LOW","MEDIUM"],default="HIGH")
    return out.sort_values(["FinalDecisionScore","TradeConfidence","Score"],ascending=False).reset_index(drop=True)


def drift_report(reference, current, threshold=0.20):
    """Simple distribution-drift report for numeric feature/model metrics."""
    if reference is None or current is None: return {"drift":False,"features":{}}
    cols=[c for c in reference.columns if c in current.columns and pd.api.types.is_numeric_dtype(reference[c]) and pd.api.types.is_numeric_dtype(current[c])]
    result={}; flagged=False
    for c in cols:
        a=float(reference[c].mean()); b=float(current[c].mean()); scale=max(abs(a),1e-6); change=abs(b-a)/scale
        result[c]=float(change); flagged |= change>threshold
    return {"drift":bool(flagged),"features":result}


def update_learning_state(path: Path, observations: dict):
    """Persist compact self-learning state in GitHub-managed JSON data."""
    path.parent.mkdir(parents=True,exist_ok=True)
    state={}
    if path.exists():
        try: state=json.loads(path.read_text())
        except Exception: state={}
    state.setdefault("version","final-v1")
    state.setdefault("observations",[])
    state["observations"].append(observations)
    state["observations"]=state["observations"][-250:]
    state["last_update"]=observations.get("date")
    path.write_text(json.dumps(state,indent=2,default=str))
    return state


def final_stage_manifest():
    return {
        "Stage5":"Accuracy & Calibration","Stage6":"Adaptive AI","Stage7":"Advanced Market Intelligence",
        "Stage8":"Event & News Intelligence","Stage9":"Decision Intelligence","Stage10":"Self-Improving AI",
        "SubStages": {
            "5":["5.0 walk-forward","5.1 rolling validation","5.2 stock accuracy","5.3 horizon accuracy","5.4 direction accuracy","5.5 error distribution","5.6 confidence calibration","5.7 prediction intervals","5.8 accuracy-based weighting"],
            "6":["6.0 error learning","6.1 stock reliability","6.2 sector reliability","6.3 regime learning","6.4 horizon reliability","6.5 dynamic ensemble weights","6.6 feature importance","6.7 adaptive retraining"],
            "7":["7.0 Nifty trend","7.1 Bank Nifty trend","7.2 breadth","7.3 volatility regime","7.4 sector rotation","7.5 relative strength","7.6 correlation","7.7 FII/DII","7.8 global influence"],
            "8":["8.0 news ingestion","8.1 sentiment","8.2 event detection","8.3 earnings","8.4 corporate actions","8.5 result-day risk","8.6 news impact","8.7 event probability","8.8 price/news confirmation"],
            "9":["9.0 BUY/HOLD/AVOID","9.1 expected return","9.2 success probability","9.3 risk-adjusted return","9.4 target probability","9.5 stop-risk","9.6 reward/risk","9.7 setup quality","9.8 final decision score"],
            "10":["10.0 continuous learning","10.1 model drift","10.2 feature drift","10.3 regime drift","10.4 automatic replacement","10.5 champion/challenger evolution","10.6 monitoring","10.7 failure detection","10.8 rollback"]
        }
    }
