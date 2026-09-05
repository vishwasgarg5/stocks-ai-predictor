"""Weekly Stage 10.2 report: accuracy, reliability, calibration, decisions and model health."""
import math
import pandas as pd
from .config import DAILY_METRICS_FILE,MODEL_VERSION,EVALUATIONS_DIR,FINAL_LEARNING_STATE_FILE,MODEL_STATE_FILE
from .telegram_report import send_telegram
from .utils import read_json
from .decision_ledger import summary as decision_summary

def _load_recent_evaluations(days=5):
    frames=[]
    for path in sorted(EVALUATIONS_DIR.glob("evaluation_*.csv")):
        try:
            d=pd.read_csv(path)
            if not d.empty:frames.append(d)
        except Exception:continue
    if not frames:return pd.DataFrame()
    x=pd.concat(frames,ignore_index=True)
    x["MarketDate"]=pd.to_datetime(x.get("MarketDate"),errors="coerce");x=x.dropna(subset=["MarketDate"]).sort_values("MarketDate");dates=x["MarketDate"].dt.date.drop_duplicates().tolist()[-days:]
    return x[x["MarketDate"].dt.date.isin(dates)]

def _bucket_summary(evals):
    if evals.empty or "PriceBucket" not in evals or "APE_Close" not in evals:return []
    rows=[]
    for b,g in evals.groupby("PriceBucket"):
        rows.append((str(b),len(g),float(g["APE_Close"].abs().mean()),float(g["DirectionCorrect"].mean()*100) if "DirectionCorrect" in g else 0))
    return sorted(rows,key=lambda x:x[2])

def _confidence_summary(evals):
    if evals.empty or "PredictionConfidence" not in evals:return []
    x=evals.copy();x["PredictionConfidence"]=pd.to_numeric(x["PredictionConfidence"],errors="coerce");x=x.dropna(subset=["PredictionConfidence"]);out=[]
    for name,lo,hi in [("High",80,101),("Medium",60,80),("Low",-1,60)]:
        g=x[(x["PredictionConfidence"]>=lo)&(x["PredictionConfidence"]<hi)]
        if not g.empty:out.append((name,len(g),float(g["APE_Close"].abs().mean()),float(g["DirectionCorrect"].mean()*100)))
    return out

def _error_distribution(evals):
    if evals.empty or "APE_Close" not in evals:return []
    a=evals["APE_Close"].abs().dropna();total=len(a)
    if not total:return []
    return [(label,int(round((a<=limit).mean()*100))) for label,limit in [("≤1%",1),("≤2%",2),("≤3%",3),("≤5%",5)]]

def _horizon_summary():
    path=EVALUATIONS_DIR/"horizon_evaluations.csv"
    if not path.exists():return [],[]
    try:x=pd.read_csv(path)
    except Exception:return [],[]
    if x.empty:return [],[]
    x["EvaluatedDate"]=pd.to_datetime(x.get("EvaluatedDate"),errors="coerce");cutoff=x["EvaluatedDate"].max()-pd.Timedelta(days=14);x=x[x["EvaluatedDate"]>=cutoff]
    out=[]
    for h,g in x.groupby("HorizonDays"):
        out.append((int(h),len(g),float(g["APE"].abs().mean()),float((1-g["APE"].abs()/100).clip(0,1).mean()*100),float(g["DirectionCorrect"].mean()*100)))
    bh=[]
    if "PriceBucket" in x:
        for (b,h),g in x.groupby(["PriceBucket","HorizonDays"]):bh.append((str(b),int(h),len(g),float(g["APE"].abs().mean())))
    return sorted(out),sorted(bh,key=lambda z:z[3])

def _wilson(successes,total,z=1.96):
    if total<=0:return None
    p=successes/total;den=1+z*z/total;centre=(p+z*z/(2*total))/den;half=z*math.sqrt((p*(1-p)+z*z/(4*total))/total)/den
    return max(0,(centre-half)*100),min(100,(centre+half)*100)

def _model_health(recent,decision):
    if recent.empty:return "NO VALIDATION"
    mape=float(recent["CloseMAPE"].mean());direction=float(recent["DirectionAccuracy"].mean());score=0.35*max(0,100-min(mape*10,100))+0.35*direction+0.15*min(100,len(recent)*20)+0.15*(decision.get("WinRate") or 50)
    return f"{score:.0f}/100"

def run():
    title=f"📊 AI NSE WEEKLY REPORT — {MODEL_VERSION}"
    if not DAILY_METRICS_FILE.exists():send_telegram(f"{title}\n\nNo evaluation data available yet.");return
    try:df=pd.read_csv(DAILY_METRICS_FILE)
    except Exception as exc:send_telegram(f"{title}\n\nUnable to read weekly metrics: {exc}");return
    required=["MarketDate","Samples","OpenMAPE","HighMAPE","LowMAPE","CloseMAPE","OverallMAPE","DirectionAccuracy"]
    missing=[c for c in required if c not in df.columns]
    if missing:send_telegram(f"{title}\n\nMissing metrics: {', '.join(missing)}");return
    df["MarketDate"]=pd.to_datetime(df["MarketDate"],errors="coerce")
    for c in required[1:]:df[c]=pd.to_numeric(df[c],errors="coerce")
    recent=df.dropna(subset=required).sort_values("MarketDate").tail(5)
    if recent.empty:send_telegram(f"{title}\n\nRecent metrics are incomplete.");return
    evals=_load_recent_evaluations(5);buckets=_bucket_summary(evals);confidence=_confidence_summary(evals);errors=_error_distribution(evals);horizons,bucket_horizons=_horizon_summary();dec=decision_summary();state=read_json(MODEL_STATE_FILE,{}) or {};health=_model_health(recent,dec)
    direction_total=int(evals["DirectionCorrect"].notna().sum()) if not evals.empty and "DirectionCorrect" in evals else 0;direction_success=int(evals["DirectionCorrect"].sum()) if direction_total else 0;ci=_wilson(direction_success,direction_total)
    lines=[title,"",f"Evaluation Days: {len(recent)}",f"Latest Day: {recent['MarketDate'].max().date()}",f"Samples: {int(recent['Samples'].sum())}","","Recent Performance",f"Open MAPE: {recent['OpenMAPE'].mean():.3f}%",f"High MAPE: {recent['HighMAPE'].mean():.3f}%",f"Low MAPE: {recent['LowMAPE'].mean():.3f}%",f"Close MAPE: {recent['CloseMAPE'].mean():.3f}%",f"Overall MAPE: {recent['OverallMAPE'].mean():.3f}%",f"Direction Accuracy: {recent['DirectionAccuracy'].mean():.1f}%"]
    if ci:lines += [f"Direction 95% CI: {ci[0]:.1f}%–{ci[1]:.1f}%"]
    if errors:lines += ["","Close Error Distribution"]+[f"Within {label}: {pct}%" for label,pct in errors]
    if buckets:lines += ["","Bucket Performance","Bucket | Samples | Close MAPE | Direction"]+[f"{b} | {n} | {m:.2f}% | {d:.1f}%" for b,n,m,d in buckets]+[f"Best bucket: {buckets[0][0]}",f"Worst bucket: {buckets[-1][0]}"]
    if horizons:lines += ["","Horizon Performance","Horizon | Samples | MAPE | Accuracy | Direction"]+[f"{h}D | {n} | {m:.2f}% | {a:.1f}% | {d:.1f}%" for h,n,m,a,d in horizons]
    if bucket_horizons:
        best=bucket_horizons[0];worst=bucket_horizons[-1];lines += ["","Bucket × Horizon","Best: %s / %sD | %d samples | %.2f%% MAPE"%(best[0],best[1],best[2],best[3]),"Worst: %s / %sD | %d samples | %.2f%% MAPE"%(worst[0],worst[1],worst[2],worst[3])]
    if confidence:lines += ["","Confidence Calibration","Band | Samples | Close MAPE | Direction"]+[f"{b} | {n} | {m:.2f}% | {d:.1f}%" for b,n,m,d in confidence]
    if dec.get("Samples",0):lines += ["","Decision Outcomes",f"Evaluated: {dec['Samples']}",f"Win Rate: {dec['WinRate']:.1f}%",f"Average Return: {dec['AvgReturn']:+.2f}%",f"BUY decisions: {dec['BUY']}",f"NO TRADE decisions: {dec['NO_TRADE']}"]
    if len(recent)>=2:
        prev=recent.iloc[-2];cur=recent.iloc[-1];lines += ["","Learning Trend",f"Close MAPE: {prev['CloseMAPE']:.3f}% → {cur['CloseMAPE']:.3f}%",f"Direction: {prev['DirectionAccuracy']:.1f}% → {cur['DirectionAccuracy']:.1f}%"]
    improvement=state.get("last_improvement");lines += ["","Model Health",f"Health Score: {health}",f"Champion: {state.get('active_variant','-')}",f"Last Decision: {state.get('last_decision','-')}",f"Validation Improvement: {'-' if improvement is None else f'{float(improvement):+.2f}%'}",f"Learning State: {'available' if FINAL_LEARNING_STATE_FILE.exists() else 'missing'}","","All validated outcomes are persisted in GitHub state for future reliability and model selection."]
    send_telegram("\n".join(lines))

if __name__=="__main__":run()
