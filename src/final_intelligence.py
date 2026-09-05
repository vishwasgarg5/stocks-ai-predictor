"""Stage 10.4: probability, reliability, risk, calibration and abstention."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import TRANSACTION_COST_BPS,SLIPPAGE_BPS,MIN_NET_RETURN_PCT,BENCHMARK_TOLERANCE_PCT,TARGET_HIT_LEVELS,TARGET_HIT_MIN_SAMPLES,EVALUATIONS_DIR,STOCK_RELIABILITY_FILE,DECISION_LEDGER_FILE
HORIZONS=(1,3,5,7,20)

def _num(v,d=50.0):
    try:
        x=float(v);return d if not np.isfinite(x) else x
    except Exception:return d

def calibrate_confidence(confidence,empirical_accuracy=50.0,samples=0):
    c=np.clip(_num(confidence),0,100);a=np.clip(_num(empirical_accuracy),0,100);w=np.clip(_num(samples,0)/50,0,1);return float((1-w)*c+w*a)

def benchmark_expected_return(current_price,history=None):
    if history is None or len(history)<2:return 0.0
    c=pd.to_numeric(history.get("Close"),errors="coerce").dropna()
    return float((c.iloc[-1]/c.iloc[-6]-1)*100) if len(c)>=6 else float((c.iloc[-1]/c.iloc[-2]-1)*100)

def net_expected_return(expected_return,cost_bps=TRANSACTION_COST_BPS,slippage_bps=SLIPPAGE_BPS):return float(_num(expected_return,0)-2*(cost_bps+slippage_bps)/100)

def _heuristic_probs(expected_return,uncertainty_pct=5.0,levels=TARGET_HIT_LEVELS):
    mu=_num(expected_return,0);sigma=max(abs(_num(uncertainty_pct,5)),0.5);r={}
    for level in levels:
        z=(float(level)-mu)/sigma;r[f"TargetHitProb_{str(level).replace('.','_')}Pct"]=float(100/(1+np.exp(np.clip(z,-20,20))))
    z=(-2-mu)/sigma;r["DownsideHitProb_2Pct"]=float(100/(1+np.exp(np.clip(z,-20,20))));return r

def _trained_probs(row):
    fallback=_heuristic_probs(row.get("Expected_Return",0),row.get("PredictionUncertaintyPct",5));
    try:
        if not DECISION_LEDGER_FILE.exists():return fallback,"FALLBACK",0
        d=pd.read_csv(DECISION_LEDGER_FILE);d=d[d.get("Outcome",pd.Series(dtype=str)).astype(str)!="OPEN"].copy()
        if len(d)<TARGET_HIT_MIN_SAMPLES or "ReturnPct" not in d:return fallback,"FALLBACK",len(d)
        cols=[c for c in ["ExpectedReturn","Confidence","PredictionUncertainty","DecisionScore","TechnicalScore","EntryPrice"] if c in d]
        if "ExpectedReturn" not in cols:return fallback,"FALLBACK",len(d)
        X=d[cols].apply(pd.to_numeric,errors="coerce").fillna(0);xr=pd.DataFrame([{c:_num(row.get(c,0),0) for c in cols}],columns=cols)
        from sklearn.linear_model import LogisticRegression
        out={}
        for level in TARGET_HIT_LEVELS:
            y=(pd.to_numeric(d["ReturnPct"],errors="coerce")>=float(level)).astype(int)
            if y.nunique()<2:continue
            m=LogisticRegression(max_iter=500,class_weight="balanced");m.fit(X,y);out[f"TargetHitProb_{str(level).replace('.','_')}Pct"]=float(m.predict_proba(xr)[0,1]*100)
        y=(pd.to_numeric(d["ReturnPct"],errors="coerce")<=-2).astype(int)
        if y.nunique()>=2:
            m=LogisticRegression(max_iter=500,class_weight="balanced");m.fit(X,y);out["DownsideHitProb_2Pct"]=float(m.predict_proba(xr)[0,1]*100)
        return {**fallback,**out},"TRAINED",len(d)
    except Exception:return fallback,"FALLBACK",0

def target_hit_probabilities(expected_return,uncertainty_pct=5.0,levels=TARGET_HIT_LEVELS):return _heuristic_probs(expected_return,uncertainty_pct,levels)

def volatility_adjusted_return(expected_return,volatility_pct):return float(_num(expected_return,0)/max(abs(_num(volatility_pct,3)),0.25))

def _stock_reliability(symbol):
    if not STOCK_RELIABILITY_FILE.exists():return 50.0,0
    try:d=pd.read_csv(STOCK_RELIABILITY_FILE);r=d[d["Symbol"].astype(str)==str(symbol)].tail(1)
    except Exception:return 50.0,0
    if r.empty:return 50.0,0
    m=_num(r.iloc[0].get("MAPE",5),5);a=_num(r.iloc[0].get("DirectionAccuracy",50),50);n=int(_num(r.iloc[0].get("Samples",0),0));return float(np.clip(.55*(100-min(m*15,100))+.45*a,0,100)),n

def _horizon_reliability(symbol,horizon):
    path=EVALUATIONS_DIR/"horizon_evaluations.csv"
    if not path.exists():return 50.0,0
    try:d=pd.read_csv(path);d=d[(d["Symbol"].astype(str)==str(symbol))&(pd.to_numeric(d["HorizonDays"],errors="coerce")==int(horizon))].tail(60)
    except Exception:return 50.0,0
    if d.empty:return 50.0,0
    m=_num(pd.to_numeric(d["APE"],errors="coerce").abs().mean(),10);a=_num(pd.to_numeric(d["DirectionCorrect"],errors="coerce").mean()*100,50);return float(np.clip(.6*(100-min(m*12.5,100))+.4*a,0,100)),len(d)

def data_quality_score(row):
    rows=_num(row.get("HistoryRows",200),200);missing=_num(row.get("MissingRatePct",0),0);stale=_num(row.get("StaleDays",0),0);anom=_num(row.get("OHLCVAnomalyRate",0),0);return float(np.clip(100-max(0,180-rows)*.25-min(missing*4,30)-min(stale*10,30)-min(anom*100,30),0,100))

def adaptive_thresholds(regime,volatility_bucket,stock_rel=50,horizon_rel=50):
    buy,watch=75.0,68.0
    if str(regime).upper() in ("BEAR","HIGH VOL"):buy+=5;watch+=4
    if str(volatility_bucket).upper()=="HIGH":buy+=4;watch+=3
    elif str(volatility_bucket).upper()=="LOW":buy-=2;watch-=1
    if stock_rel<55:buy+=4;watch+=3
    if horizon_rel<55:buy+=3;watch+=2
    return buy,watch,MIN_NET_RETURN_PCT+(0.25 if str(volatility_bucket).upper()=="HIGH" else 0)

def decision_score(row,regime="SIDEWAYS",breadth=50,news=50):
    e=_num(row.get("Expected_Return",0),0);c=_num(row.get("CalibratedConfidence",row.get("Confidence",50)));risk=_num(row.get("RegimeAdjustedScore",row.get("RiskAdjustedScore",row.get("Score",50))));t=_num(row.get("TechnicalScore",50));mh=_num(row.get("MultiHorizonExpectedReturn",0),0);hit=_num(row.get("TargetHitProb_3_0Pct",50),50);edge=_num(row.get("BenchmarkEdgePct",0),0);rr=_num(row.get("RiskAdjustedReturn",0),0);sr=_num(row.get("StockReliability",50),50);dq=_num(row.get("DataQualityScore",100),100)
    return float(np.clip(.19*risk+.14*c+.11*t+.10*np.clip(50+net_expected_return(e)*5,0,100)+.10*np.clip(50+mh*4,0,100)+.09*hit+.08*_num(breadth)+.05*_num(news)+.05*np.clip(50+edge*5,0,100)+.04*np.clip(50+rr*10,0,100)+.03*sr+.02*dq,0,100))

def final_action(row,regime="SIDEWAYS"):
    s=_num(row.get("FinalDecisionScore",50));d=str(row.get("Direction","NEUTRAL")).upper();u=_num(row.get("PredictionUncertaintyPct",0),0);c=_num(row.get("CalibratedConfidence",50));n=_num(row.get("NetExpectedReturn",0),0);edge=_num(row.get("BenchmarkEdgePct",0),0);hit=_num(row.get("TargetHitProb_3_0Pct",0),0);down=_num(row.get("DownsideHitProb_2Pct",50),50);sr=_num(row.get("StockReliability",50),50);hr=_num(row.get("HorizonReliability",50),50);dq=_num(row.get("DataQualityScore",100),100);buy,watch,min_net=adaptive_thresholds(regime,row.get("VolatilityBucket","MEDIUM"),sr,hr)
    if u>15 or c<50 or dq<60 or sr<35 or hr<35 or s<55 or n<min_net or down>=60:return "NO TRADE"
    if s>=buy and d=="UP" and edge>=-BENCHMARK_TOLERANCE_PCT and hit>=55 and down<60:return "BUY"
    if s>=watch and d=="UP" and hit>=50 and down<70:return "WATCH"
    if s<=35 or (d=="DOWN" and s<55):return "AVOID"
    return "HOLD"

def apply_final_intelligence(candidates,regime="SIDEWAYS",breadth=50,news=50):
    if candidates is None or candidates.empty:return candidates
    out=candidates.copy();out["Regime"]=regime;out["CalibratedConfidence"]=out.apply(lambda r:calibrate_confidence(r.get("Confidence",50),r.get("ReliabilityScore",50),r.get("ReliabilitySamples",0)),axis=1);out["RiskAdjustedReturn"]=out.apply(lambda r:volatility_adjusted_return(r.get("Expected_Return",0),r.get("VolatilityPct",r.get("PredictionUncertaintyPct",5))),axis=1)
    probs=[];modes=[];ps=[]
    for _,r in out.iterrows():p,m,n=_trained_probs(r);probs.append(p);modes.append(m);ps.append(n)
    hp=pd.DataFrame(probs,index=out.index);out[hp.columns]=hp;out["TargetProbabilityModel"]=modes;out["TargetProbabilitySamples"]=ps
    out["BreadthScore"]=_num(breadth);out["NewsEventScore"]=_num(news);out["RegimeAdjustedScore"]=out.apply(lambda r:float(np.clip(_num(r.get("RiskAdjustedScore",r.get("Score",50)))*{"BULL":1.05,"SIDEWAYS":1,"BEAR":.88,"HIGH VOL":.78}.get(str(regime).upper(),1),0,100)),axis=1);out["NetExpectedReturn"]=pd.to_numeric(out.get("Expected_Return",0),errors="coerce").fillna(0).apply(net_expected_return);out["BenchmarkEdgePct"]=pd.to_numeric(out.get("Expected_Return",0),errors="coerce").fillna(0)-pd.to_numeric(out.get("BenchmarkExpectedReturn",0),errors="coerce").fillna(0)
    sr=[];ss=[];hr=[];hs=[]
    for _,r in out.iterrows():
        a,n=_stock_reliability(r.get("Symbol",""));h,hn=_horizon_reliability(r.get("Symbol",""),int(r.get("PrimaryHorizon",1) or 1));sr.append(a);ss.append(n);hr.append(h);hs.append(hn)
    out["StockReliability"]=sr;out["StockReliabilitySamples"]=ss;out["HorizonReliability"]=hr;out["HorizonReliabilitySamples"]=hs;out["DataQualityScore"]=out.apply(data_quality_score,axis=1);out["FinalDecisionScore"]=out.apply(lambda r:decision_score(r,regime,breadth,news),axis=1);out["Action"]=out.apply(lambda r:final_action(r,regime),axis=1);out["UncertaintyFlag"]=pd.to_numeric(out.get("PredictionUncertaintyPct",0),errors="coerce").fillna(0)>15;out["FinalRisk"]=np.select([out["FinalDecisionScore"]>=75,out["FinalDecisionScore"]>=55],["LOW","MEDIUM"],default="HIGH");return out.sort_values(["FinalDecisionScore","Score"],ascending=False).reset_index(drop=True)

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
    state.setdefault("version","final-v8");state.setdefault("observations",[]);state["observations"].append(observations);state["observations"]=state["observations"][-250:];state["last_update"]=observations.get("date");path.write_text(json.dumps(state,indent=2,default=str));return state

def final_stage_manifest():return {"Stage10.4":"Walk-forward + trained target probabilities + stock/horizon reliability + volatility-adjusted return + benchmark edge + adaptive cost-aware decisions","Validation":"historical classifiers use only closed/evaluated outcomes; primary model remains walk-forward","Abstention":"NO TRADE for uncertainty, downside probability, weak data, poor reliability or insufficient net edge","Learning":"outcomes, horizon decay, drift, quarantine and rollback"}
