"""Weekly Stage 10.1 performance report with bucket, horizon and confidence diagnostics."""
from pathlib import Path
import pandas as pd
import numpy as np

from .config import DAILY_METRICS_FILE, MODEL_VERSION, EVALUATIONS_DIR, PREDICTIONS_DIR, FINAL_LEARNING_STATE_FILE
from .telegram_report import send_telegram


def _load_recent_evaluations(days=5):
    frames=[]
    for path in sorted(EVALUATIONS_DIR.glob("evaluation_*.csv")):
        try:
            df=pd.read_csv(path)
            if not df.empty: frames.append(df)
        except Exception:
            continue
    if not frames:return pd.DataFrame()
    x=pd.concat(frames,ignore_index=True)
    if "MarketDate" in x.columns:
        x["MarketDate"]=pd.to_datetime(x["MarketDate"],errors="coerce")
        x=x.dropna(subset=["MarketDate"]).sort_values("MarketDate")
        dates=x["MarketDate"].dt.date.drop_duplicates().tolist()[-days:]
        x=x[x["MarketDate"].dt.date.isin(dates)]
    return x


def _bucket_summary(evals):
    if evals.empty or "PriceBucket" not in evals.columns or "APE_Close" not in evals.columns:return []
    rows=[]
    for bucket,g in evals.groupby("PriceBucket"):
        direction=float(g["DirectionCorrect"].mean()*100) if "DirectionCorrect" in g else 0.0
        rows.append((str(bucket),len(g),float(g["APE_Close"].abs().mean()),direction))
    return sorted(rows,key=lambda x:x[2])


def _confidence_summary(evals):
    if evals.empty or "PredictionConfidence" not in evals.columns:return []
    x=evals.copy();x["PredictionConfidence"]=pd.to_numeric(x["PredictionConfidence"],errors="coerce")
    x=x.dropna(subset=["PredictionConfidence"])
    if x.empty:return []
    bands=[("High",80,101),("Medium",60,80),("Low",-1,60)]
    out=[]
    for name,lo,hi in bands:
        g=x[(x["PredictionConfidence"]>=lo)&(x["PredictionConfidence"]<hi)]
        if g.empty:continue
        out.append((name,len(g),float(g["APE_Close"].abs().mean()),float(g["DirectionCorrect"].mean()*100) if "DirectionCorrect" in g else 0.0))
    return out


def _horizon_summary():
    path=EVALUATIONS_DIR/"horizon_evaluations.csv"
    if not path.exists():return []
    try:x=pd.read_csv(path)
    except Exception:return []
    if x.empty:return []
    if "EvaluatedDate" in x.columns:
        x["EvaluatedDate"]=pd.to_datetime(x["EvaluatedDate"],errors="coerce")
        cutoff=x["EvaluatedDate"].max()-pd.Timedelta(days=14)
        x=x[x["EvaluatedDate"]>=cutoff]
    if x.empty:return []
    out=[]
    for h,g in x.groupby("HorizonDays"):
        out.append((int(h),len(g),float(g["APE"].abs().mean()),float((1-g["APE"].abs()/100).clip(0,1).mean()*100)))
    return sorted(out)


def run():
    title=f"📊 AI NSE WEEKLY REPORT — {MODEL_VERSION}"
    if not DAILY_METRICS_FILE.exists():
        send_telegram(f"{title}\n\nNo evaluation data available yet.");return
    try:df=pd.read_csv(DAILY_METRICS_FILE)
    except Exception as exc:
        send_telegram(f"{title}\n\nUnable to read weekly metrics: {exc}");return
    required=["MarketDate","Samples","OpenMAPE","HighMAPE","LowMAPE","CloseMAPE","OverallMAPE","DirectionAccuracy"]
    missing=[c for c in required if c not in df.columns]
    if missing:
        send_telegram(f"{title}\n\nMissing metrics: {', '.join(missing)}");return
    df=df.copy();df["MarketDate"]=pd.to_datetime(df["MarketDate"],errors="coerce");df=df.dropna(subset=["MarketDate"])
    for col in required[1:]:df[col]=pd.to_numeric(df[col],errors="coerce")
    recent=df.sort_values("MarketDate").tail(5).dropna(subset=required[1:])
    if recent.empty:
        send_telegram(f"{title}\n\nRecent metrics are incomplete; no reliable weekly summary generated.");return

    evals=_load_recent_evaluations(5)
    buckets=_bucket_summary(evals);confidence=_confidence_summary(evals);horizons=_horizon_summary()
    lines=[title,"",f"Evaluation Days: {len(recent)}",f"Latest Day: {recent['MarketDate'].max().date()}",f"Samples: {int(recent['Samples'].fillna(0).sum())}","","Recent Performance",f"Open MAPE: {recent['OpenMAPE'].mean():.3f}%",f"High MAPE: {recent['HighMAPE'].mean():.3f}%",f"Low MAPE: {recent['LowMAPE'].mean():.3f}%",f"Close MAPE: {recent['CloseMAPE'].mean():.3f}%",f"Overall MAPE: {recent['OverallMAPE'].mean():.3f}%",f"Direction Accuracy: {recent['DirectionAccuracy'].mean():.1f}%"]

    if buckets:
        lines += ["","Bucket Performance","Bucket | Samples | Close MAPE | Direction"]
        lines += [f"{b} | {n} | {m:.2f}% | {d:.1f}%" for b,n,m,d in buckets]
        lines += [f"Best bucket: {buckets[0][0]}",f"Worst bucket: {buckets[-1][0]}"]
    if horizons:
        lines += ["","Horizon Performance","Horizon | Samples | MAPE | Accuracy"]
        lines += [f"{h}D | {n} | {m:.2f}% | {a:.1f}%" for h,n,m,a in horizons]
    if confidence:
        lines += ["","Confidence Calibration","Band | Samples | Close MAPE | Direction"]
        lines += [f"{b} | {n} | {m:.2f}% | {d:.1f}%" for b,n,m,d in confidence]

    if len(recent)>=2:
        prev=recent.iloc[-2];cur=recent.iloc[-1]
        lines += ["","Learning Trend",f"Close MAPE: {float(prev['CloseMAPE']):.3f}% → {float(cur['CloseMAPE']):.3f}%",f"Direction: {float(prev['DirectionAccuracy']):.1f}% → {float(cur['DirectionAccuracy']):.1f}%"]
    state="available" if FINAL_LEARNING_STATE_FILE.exists() else "missing"
    lines += ["",f"Learning State: {state}","Results are stored in GitHub state and used for future stock/model reliability."]
    send_telegram("\n".join(lines))


if __name__=="__main__":run()
