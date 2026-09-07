"""Economic evaluation for live predictions. Accuracy is not enough; measure net return."""
from pathlib import Path
import numpy as np
import pandas as pd
from .config import EVALUATIONS_DIR,METRICS_DIR,TRANSACTION_COST_BPS,SLIPPAGE_BPS

OUTPUT=METRICS_DIR/"economic_performance.csv"

def calculate(evaluations: pd.DataFrame) -> pd.DataFrame:
    if evaluations is None or evaluations.empty:return pd.DataFrame()
    x=evaluations.copy()
    required={"Actual_Close","Previous_Close"}
    if not required.issubset(x.columns):return pd.DataFrame()
    actual=pd.to_numeric(x["Actual_Close"],errors="coerce");prev=pd.to_numeric(x["Previous_Close"],errors="coerce")
    pred=pd.to_numeric(x.get("Predicted_Close",x.get("Prediction_Close",np.nan)),errors="coerce")
    gross_ai=(pred/prev-1)*100
    realized=(actual/prev-1)*100
    cost=(TRANSACTION_COST_BPS+SLIPPAGE_BPS)/100
    x["AI_Gross_Return_Pct"]=gross_ai;x["Realized_Return_Pct"]=realized
    # Only evaluate a directional trade when the model has a finite directional forecast.
    x["AI_Direction"]=np.where(gross_ai>0,1,np.where(gross_ai<0,-1,0))
    x["AI_Net_Return_Pct"]=x["AI_Direction"]*realized-cost
    x["Baseline_Net_Return_Pct"]=realized-cost
    x["AI_Outperforms_Baseline"]=x["AI_Net_Return_Pct"]>x["Baseline_Net_Return_Pct"]
    rows=[]
    for symbol,g in x.groupby("Symbol") if "Symbol" in x.columns else [("ALL",x)]:
        r=pd.to_numeric(g["AI_Net_Return_Pct"],errors="coerce").dropna();wins=r[r>0];losses=r[r<0]
        equity=(1+r.fillna(0)/100).cumprod();peak=equity.cummax();dd=(equity/peak-1)*100
        rows.append({"Symbol":symbol,"Samples":len(r),"NetReturnPct":float((equity.iloc[-1]-1)*100) if len(equity) else np.nan,"WinRatePct":float((r>0).mean()*100) if len(r) else np.nan,"ProfitFactor":float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else np.inf,"MaxDrawdownPct":float(dd.min()) if len(dd) else np.nan,"MeanNetReturnPct":float(r.mean()) if len(r) else np.nan,"BaselineMeanNetReturnPct":float(pd.to_numeric(g["Baseline_Net_Return_Pct"],errors="coerce").mean()) if len(g) else np.nan,"BaselineWinRatePct":float(pd.to_numeric(g["AI_Outperforms_Baseline"],errors="coerce").mean()*100) if len(g) else np.nan})
    out=pd.DataFrame(rows);OUTPUT.parent.mkdir(parents=True,exist_ok=True);out.to_csv(OUTPUT,index=False);return out

def run():
    frames=[]
    for p in sorted(EVALUATIONS_DIR.glob("evaluation_*.csv")):
        try:
            d=pd.read_csv(p)
            if not d.empty:frames.append(d)
        except Exception:pass
    return calculate(pd.concat(frames,ignore_index=True) if frames else pd.DataFrame())

if __name__=="__main__":run()
