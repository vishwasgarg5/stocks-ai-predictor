"""Economic evaluation for live predictions. Accuracy is not enough; measure net return."""
import numpy as np
import pandas as pd
from .config import EVALUATIONS_DIR,METRICS_DIR,TRANSACTION_COST_BPS,SLIPPAGE_BPS
OUTPUT=METRICS_DIR/"economic_performance.csv"

def _finite(v,default=0.0):
    try:
        x=float(v);return x if np.isfinite(x) else default
    except Exception:return default

def calculate(evaluations: pd.DataFrame) -> pd.DataFrame:
    if evaluations is None or evaluations.empty:return pd.DataFrame()
    x=evaluations.copy()
    if not {"Actual_Close","Previous_Close"}.issubset(x.columns):return pd.DataFrame()
    actual=pd.to_numeric(x["Actual_Close"],errors="coerce");prev=pd.to_numeric(x["Previous_Close"],errors="coerce")
    pred_col="Predicted_Close" if "Predicted_Close" in x.columns else "Prediction_Close" if "Prediction_Close" in x.columns else "Pred_Close" if "Pred_Close" in x.columns else None
    if pred_col is None:return pd.DataFrame()
    pred=pd.to_numeric(x[pred_col],errors="coerce")
    valid=actual.notna()&prev.notna()&(actual>0)&(prev>0)&pred.notna()
    x=x.loc[valid].copy();actual=actual.loc[valid];prev=prev.loc[valid];pred=pred.loc[valid]
    if x.empty:return pd.DataFrame()
    gross=(pred/prev-1)*100;realized=(actual/prev-1)*100;cost=(float(TRANSACTION_COST_BPS)+float(SLIPPAGE_BPS))/100
    x["AI_Gross_Return_Pct"]=gross;x["Realized_Return_Pct"]=realized
    x["AI_Direction"]=np.sign(gross).astype(int);x["AI_Net_Return_Pct"]=x["AI_Direction"]*realized-cost;x["Baseline_Net_Return_Pct"]=realized-cost;x["AI_Outperforms_Baseline"]=x["AI_Net_Return_Pct"]>x["Baseline_Net_Return_Pct"]
    rows=[]
    groups=x.groupby("Symbol") if "Symbol" in x.columns else [("ALL",x)]
    for symbol,g in groups:
        r=pd.to_numeric(g["AI_Net_Return_Pct"],errors="coerce").replace([np.inf,-np.inf],np.nan).dropna();wins=r[r>0];losses=r[r<0];equity=(1+r/100).cumprod();peak=equity.cummax();dd=(equity/peak-1)*100
        rows.append({"Symbol":symbol,"Samples":len(r),"NetReturnPct":_finite((equity.iloc[-1]-1)*100) if len(equity) else np.nan,"WinRatePct":_finite((r>0).mean()*100) if len(r) else np.nan,"ProfitFactor":_finite(wins.sum()/abs(losses.sum()),0.0) if len(losses) and losses.sum()!=0 else 0.0,"MaxDrawdownPct":_finite(dd.min()) if len(dd) else np.nan,"MeanNetReturnPct":_finite(r.mean()) if len(r) else np.nan,"BaselineMeanNetReturnPct":_finite(pd.to_numeric(g["Baseline_Net_Return_Pct"],errors="coerce").mean()) if len(g) else np.nan,"BaselineWinRatePct":_finite(pd.to_numeric(g["AI_Outperforms_Baseline"],errors="coerce").mean()*100) if len(g) else np.nan})
    out=pd.DataFrame(rows);OUTPUT.parent.mkdir(parents=True,exist_ok=True);out.replace([np.inf,-np.inf],np.nan).to_csv(OUTPUT,index=False);return out

def run():
    frames=[]
    for p in sorted(EVALUATIONS_DIR.glob("evaluation_*.csv")):
        try:
            d=pd.read_csv(p)
            if not d.empty:frames.append(d)
        except Exception:pass
    return calculate(pd.concat(frames,ignore_index=True) if frames else pd.DataFrame())

if __name__=="__main__":run()
