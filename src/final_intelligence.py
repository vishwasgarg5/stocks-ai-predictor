"""Final intelligence: calibrated, benchmarked and cost-aware decisions."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import TRANSACTION_COST_BPS,SLIPPAGE_BPS,MIN_NET_RETURN_PCT,BENCHMARK_TOLERANCE_PCT
HORIZONS=(1,3,5,7,20)
def _num(v,default=50.0):
    try:
        x=float(v);return default if not np.isfinite(x) else x
    except Exception:return default
def walk_forward_metrics(actual,predicted,min_train=30,step=1):
    a=np.asarray(actual,float);p=np.asarray(predicted,float);n=min(len(a),len(p));a=a[:n];p=p[:n];errors=[];apes=[];dirs=[]
    for i in range(max(1,min_train),n,step):errors.append(abs(a[i]-p[i]));apes.append(abs(a[i]-p[i])/max(abs(a[i]),1e-6)*100);dirs.append(float(np.sign(a[i]-a[i-1])==np.sign(p[i]-p[i-1])))
    return {"Samples":len(errors),"MAE":float(np.mean(errors)) if errors else np.nan,"MAPE":float(np.mean(apes)) if apes else np.nan,"DirectionAccuracy":float(np.mean(dirs)*100) if dirs else np.nan}
def calibrate_confidence(confidence,empirical_accuracy=50.0,samples=0):
    c=np.clip(_num(confidence),0,100);a=np.clip(_num(empirical_accuracy),0,100);e=min(max(_num(samples,0)/50,0),1);return float(np.clip((1-e)*c+e*a,0,100))
def stock_horizon_reliability(metrics):
    if metrics is None or metrics.empty:return pd.DataFrame(columns=["Symbol","HorizonDays","MAPE","DirectionAccuracy","Samples","Reliability"])
    rows=[]
    for (symbol,h),g in metrics.groupby(["Symbol","HorizonDays"],dropna=False):
        if pd.isna(h):continue
        mape=_num(g["MAPE"].mean() if "MAPE" in g else 10,10);direction=_num(g["DirectionCorrect"].mean()*100 if "DirectionCorrect" in g else 50,50);rows.append({"Symbol":symbol,"HorizonDays":int(h),"MAPE":mape,"DirectionAccuracy":direction,"Samples":len(g),"Reliability":float(np.clip(.55*(100-min(mape*10,100))+.45*direction,0,100))})
    return pd.DataFrame(rows)
def prediction_interval(row,z=1.645):
    base=max(0,_num(row.get("Pred_Close",row.get("Current_Price",0)),0));u=max(_num(row.get("PredictionUncertaintyPct",5),5),0)/100;half=base*u*z;return {"BearCase":max(0,base-half),"BaseCase":base,"BullCase":max(0,base+half),"IntervalPct":u*100*z*2}
def market_breadth_score(advancers=0,decliners=0,unchanged=0):
    total=max(_num(advancers,0)+_num(decliners,0)+_num(unchanged,0),1);return float(np.clip(50+50*(_num(advancers,0)-_num(decliners,0))/total,0,100))
def regime_risk_adjustment(regime,score):return float(np.clip(_num(score)*{"BULL":1.05,"SIDEWAYS":1,"BEAR":.88,"HIGH VOL":.78}.get(str(regime).upper(),1),0,100))
def event_impact_score(sentiment=0,magnitude=0,freshness=1):return float(np.clip(50+_num(sentiment,0)*25+_num(magnitude,0)*20*min(max(_num(freshness,1),0),1),0,100))
def benchmark_expected_return(current_price,history=None):
    if history is None or len(history)<2:return 0.0
    close=pd.to_numeric(history["Close"],errors="coerce").dropna()
    if len(close)<2:return 0.0
    return float((float(close.iloc[-1])/float(close.iloc[-6])-1)*100) if len(close)>=6 else float((float(close.iloc[-1])/float(close.iloc[-2])-1)*100)
def net_expected_return(expected_return,cost_bps=TRANSACTION_COST_BPS,slippage_bps=SLIPPAGE_BPS):return float(_num(expected_return,0)-2*(cost_bps+slippage_bps)/100)
def decision_score(row,regime="SIDEWAYS",breadth=50,news=50):
    expected=_num(row.get("Expected_Return",0),0);success=_num(row.get("CalibratedConfidence",row.get("Confidence",50)));risk=_num(row.get("RegimeAdjustedScore",row.get("RiskAdjustedScore",row.get("Score",50))));technical=_num(row.get("TechnicalScore",50));mh=_num(row.get("MultiHorizonExpectedReturn",0),0);net=net_expected_return(expected)
    return float(np.clip(.28*risk+.18*success+.14*technical+.12*np.clip(50+net*5,0,100)+.10*np.clip(50+mh*4,0,100)+.10*_num(breadth)+.08*_num(news),0,100))
def final_action(row):
    s=_num(row.get("FinalDecisionScore",row.get("DecisionScore",50)));direction=str(row.get("Direction","NEUTRAL")).upper();unc=_num(row.get("PredictionUncertaintyPct",0),0);cal=_num(row.get("CalibratedConfidence",row.get("Confidence",50)));net=_num(row.get("NetExpectedReturn",0),0);edge=_num(row.get("BenchmarkEdgePct",0),0)
    if unc>15 or cal<50 or s<55 or net<MIN_NET_RETURN_PCT:return "NO TRADE"
    if s>=75 and direction=="UP" and edge>=-BENCHMARK_TOLERANCE_PCT:return "BUY"
    if s>=68 and direction=="UP":return "WATCH"
    if s<=35 or (direction=="DOWN" and s<55):return "AVOID"
    return "HOLD"
def apply_final_intelligence(candidates,regime="SIDEWAYS",breadth=50,news=50):
    if candidates is None or candidates.empty:return candidates
    out=candidates.copy();out["CalibratedConfidence"]=out.apply(lambda r:calibrate_confidence(r.get("Confidence",50),r.get("ReliabilityScore",50),r.get("ReliabilitySamples",0)),axis=1);intervals=out.apply(prediction_interval,axis=1,result_type="expand")
    for c in intervals.columns:out[c]=intervals[c]
    out["BreadthScore"]=_num(breadth);out["NewsEventScore"]=_num(news);out["RegimeAdjustedScore"]=out.apply(lambda r:regime_risk_adjustment(regime,r.get("RiskAdjustedScore",r.get("Score",50))),axis=1);out["NetExpectedReturn"]=out.get("Expected_Return",0).apply(net_expected_return);out["BenchmarkEdgePct"]=out.get("Expected_Return",0)-out.get("BenchmarkExpectedReturn",0);out["FinalDecisionScore"]=out.apply(lambda r:decision_score(r,regime,breadth,news),axis=1);out["Action"]=out.apply(final_action,axis=1);out["UncertaintyFlag"]=out.get("PredictionUncertaintyPct",pd.Series(0,index=out.index)).astype(float)>15;out["FinalRisk"]=np.select([out["FinalDecisionScore"]>=75,out["FinalDecisionScore"]>=55],["LOW","MEDIUM"],default="HIGH")
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
    state.setdefault("version","final-v5");state.setdefault("observations",[]);state["observations"].append(observations);state["observations"]=state["observations"][-250:];state["last_update"]=observations.get("date");path.write_text(json.dumps(state,indent=2,default=str));return state
def final_stage_manifest():return {"Stage10.3":"Walk-forward + dedicated returns + cost-aware decisions + benchmark guard","Validation":"walk-forward with embargo","Decision":"calibrated confidence, net return and benchmark edge","Learning":"decision outcomes, live residuals and rollback"}
