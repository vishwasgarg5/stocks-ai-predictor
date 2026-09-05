"""Stage 10.3 final decision intelligence."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import TRANSACTION_COST_BPS,SLIPPAGE_BPS,MIN_NET_RETURN_PCT,BENCHMARK_TOLERANCE_PCT,TARGET_HIT_LEVELS
HORIZONS=(1,3,5,7,20)
def _num(v,d=50.0):
    try:
        x=float(v); return d if not np.isfinite(x) else x
    except Exception:return d
def walk_forward_metrics(actual,predicted,min_train=30,step=1):
    a=np.asarray(actual,float);p=np.asarray(predicted,float);n=min(len(a),len(p));a=a[:n];p=p[:n];e=[];ape=[];d=[]
    for i in range(max(1,min_train),n,step):
        e.append(abs(a[i]-p[i]));ape.append(abs(a[i]-p[i])/max(abs(a[i]),1e-6)*100);d.append(float(np.sign(a[i]-a[i-1])==np.sign(p[i]-p[i-1])))
    return {"Samples":len(e),"MAE":float(np.mean(e)) if e else np.nan,"MAPE":float(np.mean(ape)) if ape else np.nan,"DirectionAccuracy":float(np.mean(d)*100) if d else np.nan}
def calibrate_confidence(confidence,empirical_accuracy=50.0,samples=0):
    c=np.clip(_num(confidence),0,100);a=np.clip(_num(empirical_accuracy),0,100);w=np.clip(_num(samples,0)/50,0,1);return float((1-w)*c+w*a)
def stock_horizon_reliability(metrics):
    if metrics is None or metrics.empty:return pd.DataFrame(columns=["Symbol","HorizonDays","MAPE","DirectionAccuracy","Samples","Reliability"])
    rows=[]
    for (symbol,h),g in metrics.groupby(["Symbol","HorizonDays"],dropna=False):
        if pd.isna(h):continue
        m=_num(g["MAPE"].mean() if "MAPE" in g else 10,10);a=_num(g["DirectionCorrect"].mean()*100 if "DirectionCorrect" in g else 50,50);rows.append({"Symbol":symbol,"HorizonDays":int(h),"MAPE":m,"DirectionAccuracy":a,"Samples":len(g),"Reliability":float(np.clip(.55*(100-min(m*10,100))+.45*a,0,100))})
    return pd.DataFrame(rows)
def prediction_interval(row,z=1.645):
    base=max(0,_num(row.get("Pred_Close",row.get("Current_Price",0)),0));u=max(_num(row.get("PredictionUncertaintyPct",5),5),0)/100;half=base*u*z;return {"BearCase":max(0,base-half),"BaseCase":base,"BullCase":base+half,"IntervalPct":u*200*z}
def market_breadth_score(advancers=0,decliners=0,unchanged=0):
    t=max(_num(advancers,0)+_num(decliners,0)+_num(unchanged,0),1);return float(np.clip(50+50*(_num(advancers,0)-_num(decliners,0))/t,0,100))
def regime_risk_adjustment(regime,score):return float(np.clip(_num(score)*{"BULL":1.05,"SIDEWAYS":1,"BEAR":.88,"HIGH VOL":.78}.get(str(regime).upper(),1),0,100))
def event_impact_score(sentiment=0,magnitude=0,freshness=1):return float(np.clip(50+_num(sentiment,0)*25+_num(magnitude,0)*20*np.clip(_num(freshness,1),0,1),0,100))
def benchmark_expected_return(current_price,history=None):
    if history is None or len(history)<2:return 0.0
    c=pd.to_numeric(history["Close"],errors="coerce").dropna()
    if len(c)<2:return 0.0
    return float((c.iloc[-1]/c.iloc[-6]-1)*100) if len(c)>=6 else float((c.iloc[-1]/c.iloc[-2]-1)*100)
def net_expected_return(expected_return,cost_bps=TRANSACTION_COST_BPS,slippage_bps=SLIPPAGE_BPS):return float(_num(expected_return,0)-2*(cost_bps+slippage_bps)/100)
def target_hit_probabilities(expected_return,uncertainty_pct=5.0,levels=TARGET_HIT_LEVELS):
    mu=_num(expected_return,0);sigma=max(abs(_num(uncertainty_pct,5)),0.5);r={}
    for level in levels:
        z=(float(level)-mu)/sigma;r[f"TargetHitProb_{str(level).replace('.','_')}Pct"]=float(100/(1+np.exp(np.clip(z,-20,20))))
    z=(-2-mu)/sigma;r["DownsideHitProb_2Pct"]=float(100/(1+np.exp(np.clip(z,-20,20))));return r
def decision_score(row,regime="SIDEWAYS",breadth=50,news=50):
    e=_num(row.get("Expected_Return",0),0);c=_num(row.get("CalibratedConfidence",row.get("Confidence",50)));risk=_num(row.get("RegimeAdjustedScore",row.get("RiskAdjustedScore",row.get("Score",50))));t=_num(row.get("TechnicalScore",50));mh=_num(row.get("MultiHorizonExpectedReturn",0),0);hit=_num(row.get("TargetHitProb_3_0Pct",50),50);net=net_expected_return(e)
    return float(np.clip(.25*risk+.17*c+.13*t+.12*np.clip(50+net*5,0,100)+.10*np.clip(50+mh*4,0,100)+.08*hit+.08*_num(breadth)+.07*_num(news),0,100))
def final_action(row):
    s=_num(row.get("FinalDecisionScore",50));d=str(row.get("Direction","NEUTRAL")).upper();u=_num(row.get("PredictionUncertaintyPct",0),0);c=_num(row.get("CalibratedConfidence",50));n=_num(row.get("NetExpectedReturn",0),0);edge=_num(row.get("BenchmarkEdgePct",0),0);hit=_num(row.get("TargetHitProb_3_0Pct",0),0)
    if u>15 or c<50 or s<55 or n<MIN_NET_RETURN_PCT:return "NO TRADE"
    if s>=75 and d=="UP" and edge>=-BENCHMARK_TOLERANCE_PCT and hit>=55:return "BUY"
    if s>=68 and d=="UP" and hit>=50:return "WATCH"
    if s<=35 or (d=="DOWN" and s<55):return "AVOID"
    return "HOLD"
def apply_final_intelligence(candidates,regime="SIDEWAYS",breadth=50,news=50):
    if candidates is None or candidates.empty:return candidates
    out=candidates.copy();out["CalibratedConfidence"]=out.apply(lambda r:calibrate_confidence(r.get("Confidence",50),r.get("ReliabilityScore",50),r.get("ReliabilitySamples",0)),axis=1)
    iv=out.apply(prediction_interval,axis=1,result_type="expand");out[iv.columns]=iv
    hp=out.apply(lambda r:target_hit_probabilities(r.get("Expected_Return",0),r.get("PredictionUncertaintyPct",5)),axis=1,result_type="expand");out[hp.columns]=hp
    out["BreadthScore"]=_num(breadth);out["NewsEventScore"]=_num(news);out["RegimeAdjustedScore"]=out.apply(lambda r:regime_risk_adjustment(regime,r.get("RiskAdjustedScore",r.get("Score",50))),axis=1);out["NetExpectedReturn"]=pd.to_numeric(out.get("Expected_Return",0),errors="coerce").fillna(0).apply(net_expected_return);out["BenchmarkEdgePct"]=pd.to_numeric(out.get("Expected_Return",0),errors="coerce").fillna(0)-pd.to_numeric(out.get("BenchmarkExpectedReturn",0),errors="coerce").fillna(0);out["FinalDecisionScore"]=out.apply(lambda r:decision_score(r,regime,breadth,news),axis=1);out["Action"]=out.apply(final_action,axis=1);out["UncertaintyFlag"]=pd.to_numeric(out.get("PredictionUncertaintyPct",0),errors="coerce").fillna(0)>15;out["FinalRisk"]=np.select([out["FinalDecisionScore"]>=75,out["FinalDecisionScore"]>=55],["LOW","MEDIUM"],default="HIGH")
    return out.sort_values(["FinalDecisionScore","Score"],ascending=False).reset_index(drop=True)
def drift_report(reference,current,threshold=.20):
    if reference is None or current is None:return {"drift":False,"features":{}}
    cols=[c for c in reference.columns if c in current.columns and pd.api.types.is_numeric_dtype(reference[c]) and pd.api.types.is_numeric_dtype(current[c])];r={};flag=False
    for c in cols:
        a=float(reference[c].mean());b=float(current[c].mean());x=abs(b-a)/max(abs(a),1e-6);r[c]=x;flag|=x>threshold
    return {"drift":bool(flag),"features":r}
def update_learning_state(path:Path,observations:dict):
    path.parent.mkdir(parents=True,exist_ok=True);state={}
    if path.exists():
        try:state=json.loads(path.read_text())
        except Exception:state={}
    state.setdefault("version","final-v6");state.setdefault("observations",[]);state["observations"].append(observations);state["observations"]=state["observations"][-250:];state["last_update"]=observations.get("date");path.write_text(json.dumps(state,indent=2,default=str));return state
def final_stage_manifest():return {"Stage10.3":"Walk-forward + dedicated returns + benchmark + cross-sectional/sector/volatility context + target-hit probabilities + cost-aware decisions","Validation":"walk-forward with embargo","Learning":"decision outcomes, drift and rollback"}
