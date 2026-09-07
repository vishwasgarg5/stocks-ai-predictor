"""Stage 10.6 quality controls for live accuracy, baselines and economics."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .config import EVALUATIONS_DIR,METRICS_DIR,STATE_DIR,TRANSACTION_COST_BPS,SLIPPAGE_BPS,MODEL_VERSION
from .ledger import load_predictions
from .economic_metrics import run as run_economic_metrics

BASELINE_FILE=METRICS_DIR/"baseline_evaluations.csv"
HORIZON_FILE=METRICS_DIR/"horizon_accuracy.csv"
CONFIDENCE_FILE=METRICS_DIR/"adaptive_confidence.csv"
TARGET_POLICY_FILE=METRICS_DIR/"portfolio_target_policy.csv"

def _num(v,default=np.nan):
    try:
        x=float(v);return x if np.isfinite(x) else default
    except Exception:return default

def _series(df,column,default=np.nan):
    return pd.to_numeric(df[column],errors="coerce") if column in df.columns else pd.Series(default,index=df.index,dtype=float)

def evaluate_baseline(evaluation,predictions=None):
    if evaluation is None or evaluation.empty:return pd.DataFrame()
    x=evaluation.copy()
    if predictions is not None and not predictions.empty and "Prediction_ID" in predictions.columns:
        keep=[c for c in ["Prediction_ID","Baseline_Close","Baseline_Cost_Pct"] if c in predictions.columns]
        if keep:x=x.merge(predictions[keep].drop_duplicates("Prediction_ID"),on="Prediction_ID",how="left")
    if "Baseline_Close" not in x.columns:x["Baseline_Close"]=_series(x,"Previous_Close",np.nan)
    actual=_series(x,"Actual_Close");baseline=_series(x,"Baseline_Close");ai=_series(x,"APE_Close")
    x["Baseline_APE"]=(actual-baseline).abs()/actual.abs().clip(lower=1e-8)*100;x["AI_APE"]=ai.abs()
    default_cost=(float(TRANSACTION_COST_BPS)+float(SLIPPAGE_BPS))/100
    cost=_series(x,"Baseline_Cost_Pct",default_cost).fillna(default_cost);x["CostPct"]=cost
    x["AI_Better_Than_Baseline"]=(x["AI_APE"]+cost<x["Baseline_APE"])&x["AI_APE"].notna()&x["Baseline_APE"].notna()
    cols=[c for c in ["Prediction_ID","MarketDate","PredictionDate","Symbol","AI_APE","Baseline_APE","CostPct","AI_Better_Than_Baseline"] if c in x.columns];out=x[cols].copy()
    if not out.empty:
        existing=pd.read_csv(BASELINE_FILE) if BASELINE_FILE.exists() else pd.DataFrame();combined=pd.concat([existing,out],ignore_index=True) if not existing.empty else out
        keys=[c for c in ["Prediction_ID","MarketDate","Symbol"] if c in combined.columns]
        if keys:combined=combined.drop_duplicates(keys,keep="last")
        BASELINE_FILE.parent.mkdir(parents=True,exist_ok=True);combined.to_csv(BASELINE_FILE,index=False)
    return out

def horizon_accuracy():
    path=EVALUATIONS_DIR/"horizon_evaluations.csv"
    if not path.exists():return pd.DataFrame()
    try:x=pd.read_csv(path)
    except Exception:return pd.DataFrame()
    if x.empty or "HorizonDays" not in x.columns:return pd.DataFrame()
    ape_col="APE" if "APE" in x.columns else "APE_Close" if "APE_Close" in x.columns else None
    if ape_col is None:return pd.DataFrame()
    rows=[]
    for h,g in x.groupby("HorizonDays"):
        ape=pd.to_numeric(g[ape_col],errors="coerce").abs().dropna();direction=pd.to_numeric(g["DirectionCorrect"],errors="coerce").dropna() if "DirectionCorrect" in g else pd.Series(dtype=float)
        rows.append({"HorizonDays":int(h),"Samples":len(ape),"MAPE":_num(ape.mean()) if len(ape) else np.nan,"Accuracy":_num((1-ape/100).clip(0,1).mean()*100) if len(ape) else np.nan,"DirectionAccuracy":_num(direction.mean()*100) if len(direction) else np.nan})
    out=pd.DataFrame(rows).sort_values("HorizonDays");out.to_csv(HORIZON_FILE,index=False);return out

def adaptive_confidence(symbol=None,horizon=1):
    frames=[]
    for p in sorted(EVALUATIONS_DIR.glob("evaluation_*.csv")):
        try:
            d=pd.read_csv(p)
            if not d.empty:frames.append(d)
        except Exception:pass
    if not frames:return {"Symbol":symbol or "ALL","HorizonDays":int(horizon),"Confidence":50.0,"Samples":0,"Reason":"NO_LIVE_EVIDENCE; historical/uncertainty/regime neutral priors"}
    d=pd.concat(frames,ignore_index=True)
    if symbol and "Symbol" in d.columns:d=d[d["Symbol"].astype(str)==str(symbol)]
    recent=d.tail(50);n=len(recent);mape=_num(_series(recent,"APE_Close").abs().mean(),10.0);direction=_num(_series(recent,"DirectionCorrect").mean()*100,50.0)
    uncertainty=_num(_series(recent,"PredictionUncertaintyPct").mean(),np.nan)
    if not np.isfinite(uncertainty):uncertainty=50.0;unc_reason="uncertainty neutral prior"
    else:unc_reason="live uncertainty"
    baseline=50.0
    if BASELINE_FILE.exists():
        try:
            b=pd.read_csv(BASELINE_FILE);b=b[b["Symbol"].astype(str)==str(symbol)] if symbol and "Symbol" in b.columns else b;b=b.tail(50)
            if not b.empty:baseline=_num(pd.to_numeric(b["AI_Better_Than_Baseline"],errors="coerce").mean()*100,50.0)
        except Exception:pass
    error_score=np.clip(100-mape*10,0,100);uncertainty_score=np.clip(100-uncertainty,0,100)
    historical=50.0;hist_reason="historical validation neutral prior"
    market_regime=50.0;regime_reason="market regime neutral prior"
    raw=.25*error_score+.15*direction+.15*baseline+.15*uncertainty_score+.15*historical+.15*market_regime;evidence=min(1,n/30);confidence=50+(raw-50)*evidence
    reason=f"live MAPE + direction + baseline + {unc_reason}; {hist_reason}; {regime_reason}; sample shrinkage"
    result={"Symbol":symbol or "ALL","HorizonDays":int(horizon),"Confidence":round(float(np.clip(confidence,0,100)),2),"RecentMAPE":round(float(mape),3),"DirectionAccuracy":round(float(direction),2),"BaselineWinRate":round(float(baseline),2),"UncertaintyScore":round(float(uncertainty_score),2),"HistoricalValidationScore":historical,"MarketRegimeScore":market_regime,"Samples":n,"ModelVersion":MODEL_VERSION,"Reason":reason}
    existing=pd.read_csv(CONFIDENCE_FILE) if CONFIDENCE_FILE.exists() else pd.DataFrame();out=pd.DataFrame([result]);combined=pd.concat([existing,out],ignore_index=True) if not existing.empty else out;combined=combined.drop_duplicates(["Symbol","HorizonDays"],keep="last");combined.to_csv(CONFIDENCE_FILE,index=False);return result

def confirm_portfolio_target(forecasts,minimum_target=10.0):
    clean={int(h):_num(v) for h,v in (forecasts or {}).items() if np.isfinite(_num(v))};long=[(h,v) for h,v in sorted(clean.items()) if h in (60,90,180,365)]
    if not long:return {"TargetPct":minimum_target,"Status":"NO_LONG_HORIZON_EVIDENCE","ConfirmedHorizons":[]}
    confirmed=[]
    for h,v in long:
        support=sum(1 for h2,v2 in long if h2>=h and v2>=v*.80 and v2>=minimum_target)
        if v>=minimum_target and support>=2:confirmed.append((h,v))
    if confirmed:
        target=min(v for _,v in confirmed);result={"TargetPct":round(float(max(minimum_target,target)),2),"Status":"MULTI_HORIZON_CONFIRMED","ConfirmedHorizons":[h for h,_ in confirmed]}
    else:
        best=max(v for _,v in long);result={"TargetPct":round(float(minimum_target if best>=minimum_target else max(0,best)),2),"Status":"SINGLE_HORIZON_OR_INCONSISTENT","ConfirmedHorizons":[]}
    old=pd.read_csv(TARGET_POLICY_FILE) if TARGET_POLICY_FILE.exists() else pd.DataFrame();row={"Timestamp":pd.Timestamp.utcnow().isoformat(),**result};combined=pd.concat([old,pd.DataFrame([row])],ignore_index=True) if not old.empty else pd.DataFrame([row]);combined.to_csv(TARGET_POLICY_FILE,index=False);return result

def run_quality_controls(market_date=None):
    result={"ModelVersion":MODEL_VERSION,"MarketDate":market_date}
    evaluation=None
    if market_date:
        path=EVALUATIONS_DIR/f"evaluation_{market_date}.csv"
        if path.exists():
            evaluation=pd.read_csv(path);prediction_date=str(evaluation["PredictionDate"].iloc[0]) if "PredictionDate" in evaluation and not evaluation.empty else market_date;predictions=load_predictions(pd.Timestamp(prediction_date).date());base=evaluate_baseline(evaluation,predictions);result["BaselineSamples"]=len(base);result["BaselineWinRate"]=_num(base["AI_Better_Than_Baseline"].mean()*100) if not base.empty else None
    h=horizon_accuracy();result["HorizonMetrics"]=h.to_dict("records") if not h.empty else [];result["EconomicMetrics"]=run_economic_metrics().to_dict("records");result["AdaptiveConfidence"]=adaptive_confidence();STATE_DIR.mkdir(parents=True,exist_ok=True);(STATE_DIR/"quality_controls.json").write_text(json.dumps(result,indent=2,default=str));return result

if __name__=="__main__":
    import sys
    run_quality_controls(sys.argv[1] if len(sys.argv)>1 else None)
