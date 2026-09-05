"""Stage 10.4: probability, risk, abstention and adaptive decision intelligence."""
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

def calibrate_confidence(confidence,empirical_accuracy=50.0,samples=0):
    c=np.clip(_num(confidence),0,100);a=np.clip(_num(empirical_accuracy),0,100);w=np.clip(_num(samples,0)/50,0,1);return float((1-w)*c+w*a)

def benchmark_expected_return(current_price,history=None):
    if history is None or len(history)<2:return 0.0
    c=pd.to_numeric(history.get("Close"),errors="coerce").dropna()
    if len(c)<2:return 0.0
    return float((c.iloc[-1]/c.iloc[-6]-1)*100) if len(c)>=6 else float((c.iloc[-1]/c.iloc[-2]-1)*100)

def net_expected_return(expected_return,cost_bps=TRANSACTION_COST_BPS,slippage_bps=SLIPPAGE_BPS):
    return float(_num(expected_return,0)-2*(cost_bps+slippage_bps)/100)

def target_hit_probabilities(expected_return,uncertainty_pct=5.0,levels=TARGET_HIT_LEVELS):
    """Heuristic probabilities; live labels can replace these once sufficient data exists."""
    mu=_num(expected_return,0);sigma=max(abs(_num(uncertainty_pct,5)),0.5);r={}
    for level in levels:
        z=(float(level)-mu)/sigma;r[f"TargetHitProb_{str(level).replace('.','_')}Pct"]=float(100/(1+np.exp(np.clip(z,-20,20))))
    z=(-2-mu)/sigma;r["DownsideHitProb_2Pct"]=float(100/(1+np.exp(np.clip(z,-20,20))))
    return r

def trained_target_hit_probability(expected_return,uncertainty_pct=5.0,level=3.0):
    """Deterministic fallback used until a labeled classifier has enough observations."""
    return target_hit_probabilities(expected_return,uncertainty_pct,(level,))[f"TargetHitProb_{str(level).replace('.','_')}Pct"]

def volatility_adjusted_return(expected_return,volatility_pct):
    e=_num(expected_return,0);v=max(abs(_num(volatility_pct,3)),0.25);return float(e/v)

def adaptive_thresholds(regime,volatility_bucket):
    regime=str(regime).upper();vol=str(volatility_bucket).upper()
    buy=75.0;watch=68.0;min_net=MIN_NET_RETURN_PCT
    if regime in ("BEAR","HIGH VOL"):buy+=5;watch+=4;min_net+=0.25
    if vol=="HIGH":buy+=4;watch+=3;min_net+=0.25
    elif vol=="LOW":buy-=2;watch-=1
    return buy,watch,min_net

def decision_score(row,regime="SIDEWAYS",breadth=50,news=50):
    e=_num(row.get("Expected_Return",0),0);c=_num(row.get("CalibratedConfidence",row.get("Confidence",50)));risk=_num(row.get("RegimeAdjustedScore",row.get("RiskAdjustedScore",row.get("Score",50))));t=_num(row.get("TechnicalScore",50));mh=_num(row.get("MultiHorizonExpectedReturn",0),0);hit=_num(row.get("TargetHitProb_3_0Pct",50),50);edge=_num(row.get("BenchmarkEdgePct",0),0);riskret=np.clip(_num(row.get("RiskAdjustedReturn",0),0)*10+50,0,100)
    return float(np.clip(.22*risk+.16*c+.12*t+.10*np.clip(50+net_expected_return(e)*5,0,100)+.10*np.clip(50+mh*4,0,100)+.09*hit+.08*_num(breadth)+.05*_num(news)+.04*np.clip(50+edge*5,0,100)+.04*riskret,0,100))

def final_action(row,regime="SIDEWAYS"):
    s=_num(row.get("FinalDecisionScore",50));d=str(row.get("Direction","NEUTRAL")).upper();u=_num(row.get("PredictionUncertaintyPct",0),0);c=_num(row.get("CalibratedConfidence",50));n=_num(row.get("NetExpectedReturn",0),0);edge=_num(row.get("BenchmarkEdgePct",0),0);hit=_num(row.get("TargetHitProb_3_0Pct",0),0);down=_num(row.get("DownsideHitProb_2Pct",50),50);buy,watch,min_net=adaptive_thresholds(regime,row.get("VolatilityBucket","MEDIUM"))
    if u>15 or c<50 or s<55 or n<min_net:return "NO TRADE"
    if s>=buy and d=="UP" and edge>=-BENCHMARK_TOLERANCE_PCT and hit>=55 and down<60:return "BUY"
    if s>=watch and d=="UP" and hit>=50 and down<70:return "WATCH"
    if s<=35 or (d=="DOWN" and s<55):return "AVOID"
    return "HOLD"

def apply_final_intelligence(candidates,regime="SIDEWAYS",breadth=50,news=50):
    if candidates is None or candidates.empty:return candidates
    out=candidates.copy()
    out["CalibratedConfidence"]=out.apply(lambda r:calibrate_confidence(r.get("Confidence",50),r.get("ReliabilityScore",50),r.get("ReliabilitySamples",0)),axis=1)
    out["RiskAdjustedReturn"]=out.apply(lambda r:volatility_adjusted_return(r.get("Expected_Return",0),r.get("VolatilityPct",r.get("PredictionUncertaintyPct",5))),axis=1)
    iv=out.apply(lambda r:{"BearCase":max(0,_num(r.get("Pred_Close",r.get("Current_Price",0)),0)*(1-max(_num(r.get("PredictionUncertaintyPct",5),5),0)/100*1.645)),"BaseCase":max(0,_num(r.get("Pred_Close",r.get("Current_Price",0)),0)),"BullCase":max(0,_num(r.get("Pred_Close",r.get("Current_Price",0)),0)*(1+max(_num(r.get("PredictionUncertaintyPct",5),5),0)/100*1.645))},axis=1,result_type="expand");out[iv.columns]=iv
    hp=out.apply(lambda r:target_hit_probabilities(r.get("Expected_Return",0),r.get("PredictionUncertaintyPct",5)),axis=1,result_type="expand");out[hp.columns]=hp
    out["BreadthScore"]=_num(breadth);out["NewsEventScore"]=_num(news);out["RegimeAdjustedScore"]=out.apply(lambda r:np.clip(_num(r.get("RiskAdjustedScore",r.get("Score",50)))*{"BULL":1.05,"SIDEWAYS":1,"BEAR":.88,"HIGH VOL":.78}.get(str(regime).upper(),1),0,100),axis=1)
    out["NetExpectedReturn"]=pd.to_numeric(out.get("Expected_Return",0),errors="coerce").fillna(0).apply(net_expected_return);out["BenchmarkEdgePct"]=pd.to_numeric(out.get("Expected_Return",0),errors="coerce").fillna(0)-pd.to_numeric(out.get("BenchmarkExpectedReturn",0),errors="coerce").fillna(0);out["FinalDecisionScore"]=out.apply(lambda r:decision_score(r,regime,breadth,news),axis=1);out["Action"]=out.apply(lambda r:final_action(r,regime),axis=1);out["UncertaintyFlag"]=pd.to_numeric(out.get("PredictionUncertaintyPct",0),errors="coerce").fillna(0)>15;out["FinalRisk"]=np.select([out["FinalDecisionScore"]>=75,out["FinalDecisionScore"]>=55],["LOW","MEDIUM"],default="HIGH")
    return out.sort_values(["FinalDecisionScore","Score"],ascending=False).reset_index(drop=True)

def update_learning_state(path:Path,observations:dict):
    path.parent.mkdir(parents=True,exist_ok=True);state={}
    if path.exists():
        try:state=json.loads(path.read_text())
        except Exception:state={}
    state.setdefault("version","final-v7");state.setdefault("observations",[]);state["observations"].append(observations);state["observations"]=state["observations"][-250:];state["last_update"]=observations.get("date");path.write_text(json.dumps(state,indent=2,default=str));return state

def final_stage_manifest():
    return {"Stage10.4":"Probability + adaptive risk + volatility-adjusted return + benchmark edge + abstention","Validation":"walk-forward with embargo","Decision":"BUY/WATCH/HOLD/AVOID/NO TRADE","Learning":"live outcomes, drift, reliability and rollback"}
