"""Final intelligence: calibrated decisions, uncertainty, validation and self-learning."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

HORIZONS=(1,3,5,7,20)
REGIMES=("BULL","SIDEWAYS","BEAR","HIGH VOL")

def _num(v,default=50.0):
    try:
        x=float(v); return default if not np.isfinite(x) else x
    except Exception:return default

def walk_forward_metrics(actual,predicted,min_train=30,step=1):
    a=np.asarray(actual,dtype=float);p=np.asarray(predicted,dtype=float);n=min(len(a),len(p));a=a[:n];p=p[:n];errors=[];apes=[];dirs=[]
    for i in range(max(1,min_train),n,step):
        errors.append(abs(a[i]-p[i]));apes.append(abs(a[i]-p[i])/max(abs(a[i]),1e-6)*100);dirs.append(float(np.sign(a[i]-a[i-1])==np.sign(p[i]-p[i-1])))
    return {"Samples":len(errors),"MAE":float(np.mean(errors)) if errors else np.nan,"MAPE":float(np.mean(apes)) if apes else np.nan,"DirectionAccuracy":float(np.mean(dirs)*100) if dirs else np.nan}

def calibrate_confidence(confidence,empirical_accuracy=50.0,samples=0):
    c=np.clip(_num(confidence),0,100);a=np.clip(_num(empirical_accuracy),0,100);evidence=min(max(_num(samples,0)/50.0,0),1)
    return float(np.clip((1-evidence)*c+evidence*a,0,100))

def stock_horizon_reliability(metrics):
    if metrics is None or metrics.empty:return pd.DataFrame(columns=["Symbol","HorizonDays","MAPE","DirectionAccuracy","Samples","Reliability"])
    rows=[]
    for (symbol,h),g in metrics.groupby(["Symbol","HorizonDays"],dropna=False):
        if pd.isna(h):continue
        mape=_num(g["MAPE"].mean() if "MAPE" in g else 10,10);direction=_num(g["DirectionCorrect"].mean()*100 if "DirectionCorrect" in g else 50,50)
        rel=np.clip(.55*(100-min(mape*10,100))+.45*direction,0,100)
        rows.append({"Symbol":symbol,"HorizonDays":int(h),"MAPE":mape,"DirectionAccuracy":direction,"Samples":len(g),"Reliability":float(rel)})
    return pd.DataFrame(rows)

def prediction_interval(row,z=1.645):
    base=max(0,_num(row.get("Pred_Close",row.get("Current_Price",0)),0));u=max(_num(row.get("PredictionUncertaintyPct",5),5),0)/100;half=base*u*z
    return {"BearCase":max(0,base-half),"BaseCase":base,"BullCase":max(0,base+half),"IntervalPct":u*100*z*2}

def market_breadth_score(advancers=0,decliners=0,unchanged=0):
    total=max(_num(advancers,0)+_num(decliners,0)+_num(unchanged,0),1);return float(np.clip(50+50*(_num(advancers,0)-_num(decliners,0))/total,0,100))

def regime_risk_adjustment(regime,score):return float(np.clip(_num(score)*{"BULL":1.05,"SIDEWAYS":1,"BEAR":.88,"HIGH VOL":.78}.get(str(regime).upper(),1),0,100))

def event_impact_score(sentiment=0,magnitude=0,freshness=1):return float(np.clip(50+_num(sentiment,0)*25+_num(magnitude,0)*20*min(max(_num(freshness,1),0),1),0,100))

def decision_score(row,regime="SIDEWAYS",breadth=50,news=50):
    expected=_num(row.get("Expected_Return",0),0);success=_num(row.get("CalibratedConfidence",row.get("Confidence",50)));risk=_num(row.get("RegimeAdjustedScore",row.get("RiskAdjustedScore",row.get("Score",50))));technical=_num(row.get("TechnicalScore",50));mh=_num(row.get("MultiHorizonExpectedReturn",0),0)
    return float(np.clip(.28*risk+.18*success+.14*technical+.12*np.clip(50+expected*5,0,100)+.10*np.clip(50+mh*4,0,100)+.10*_num(breadth)+.08*_num(news),0,100))

def final_action(row):
    s=_num(row.get("FinalDecisionScore",row.get("DecisionScore",50)));direction=str(row.get("Direction","NEUTRAL")).upper();unc=_num(row.get("PredictionUncertaintyPct",0),0);cal=_num(row.get("CalibratedConfidence",row.get("Confidence",50)))
    if unc>15 or cal<50 or s<55:return "NO TRADE"
    if s>=75 and direction=="UP":return "BUY"
    if s>=68 and direction=="UP":return "WATCH"
    if s<=35 or (direction=="DOWN" and s<55):return "AVOID"
    return "HOLD"

def apply_final_intelligence(candidates,regime="SIDEWAYS",breadth=50,news=50):
    if candidates is None or candidates.empty:return candidates
    out=candidates.copy();out["CalibratedConfidence"]=out.apply(lambda r:calibrate_confidence(r.get("Confidence",50),r.get("ReliabilityScore",50),r.get("ReliabilitySamples",0)),axis=1)
    intervals=out.apply(prediction_interval,axis=1,result_type="expand")
    for c in intervals.columns:out[c]=intervals[c]
    out["BreadthScore"]=_num(breadth);out["NewsEventScore"]=_num(news);out["RegimeAdjustedScore"]=out.apply(lambda r:regime_risk_adjustment(regime,r.get("RiskAdjustedScore",r.get("Score",50))),axis=1);out["FinalDecisionScore"]=out.apply(lambda r:decision_score(r,regime,breadth,news),axis=1);out["Action"]=out.apply(final_action,axis=1)
    out["UncertaintyFlag"]=out.get("PredictionUncertaintyPct",pd.Series(0,index=out.index)).astype(float)>15
    out["FinalRisk"]=np.select([out["FinalDecisionScore"]>=75,out["FinalDecisionScore"]>=55],["LOW","MEDIUM"],default="HIGH")
    return out.sort_values(["FinalDecisionScore","Score"],ascending=False).reset_index(drop=True)

def drift_report(reference,current,threshold=.20):
    if reference is None or current is None:return {"drift":False,"features":{}}
    cols=[c for c in reference.columns if c in current.columns and pd.api.types.is_numeric_dtype(reference[c]) and pd.api.types.is_numeric_dtype(current[c])];result={};flagged=False
    for c in cols:
        a=float(reference[c].mean());b=float(current[c].mean());change=abs(b-a)/max(abs(a),1e-6);result[c]=change;flagged|=change>threshold
    return {"drift":bool(flagged),"features":result}

def update_learning_state(path:Path,observations:dict):
    path.parent.mkdir(parents=True,exist_ok=True);state={}
    if path.exists():
        try:state=json.loads(path.read_text())
        except Exception:state={}
    state.setdefault("version","final-v4");state.setdefault("observations",[]);state["observations"].append(observations);state["observations"]=state["observations"][-250:];state["last_update"]=observations.get("date");path.write_text(json.dumps(state,indent=2,default=str));return state

def final_stage_manifest():
    return {"Stage5":"Accuracy & Calibration","Stage6":"Adaptive AI","Stage7":"Advanced Market Intelligence","Stage8":"Live Event & News Intelligence","Stage9":"Decision Intelligence","Stage10":"Self-Improving AI","Stage10.2":"Validated Decision & Adaptive Learning","SubStages":{"5":["walk-forward","rolling validation","stock accuracy","horizon accuracy","direction accuracy","error distribution","confidence calibration","prediction intervals","accuracy weighting"],"6":["error learning","stock reliability","sector reliability","regime learning","horizon reliability","dynamic weights","feature importance","adaptive retraining"],"7":["Nifty trend","Bank Nifty trend","breadth","volatility regime","sector rotation","relative strength","correlation","FII/DII","global influence"],"8":["news ingestion","sentiment","event detection","earnings","corporate actions","result-day risk","news impact","event probability","price/news confirmation"],"9":["BUY/HOLD/AVOID/NO TRADE","expected return","success probability","risk-adjusted return","target probability","stop-risk","reward/risk","setup quality","final score"],"10":["continuous learning","model drift","feature drift","regime drift","automatic replacement","champion/challenger","monitoring","failure detection","rollback"],"10.2":["decision outcome ledger","live residual learning","confidence calibration","bucket x horizon reliability","walk-forward validation","automatic rollback","model health"]}}
