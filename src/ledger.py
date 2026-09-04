from pathlib import Path
import re
import pandas as pd
from .config import PREDICTIONS_DIR,EVALUATIONS_DIR,JUMP_DIR,INTRADAY_DIR,DAILY_METRICS_FILE,STOCK_RELIABILITY_FILE,JUMP_METRICS_FILE,INTRADAY_METRICS_FILE
from .utils import write_json

def prediction_path(prediction_date): return PREDICTIONS_DIR / f"predictions_{prediction_date}.csv"
def evaluation_path(market_date): return EVALUATIONS_DIR / f"evaluation_{market_date}.csv"
def jump_path(prediction_date): return JUMP_DIR / f"jump_{prediction_date}.csv"
def intraday_path(prediction_date): return INTRADAY_DIR / f"intraday_{prediction_date}.csv"
def morning_report_path(prediction_date): return PREDICTIONS_DIR / f"morning_report_{prediction_date}.json"

def prediction_exists(prediction_date):
    path=prediction_path(prediction_date)
    if not path.exists(): return False
    try:
        df=pd.read_csv(path)
        return len(df)>=5 and "Symbol" in df.columns
    except Exception: return False

def morning_report_sent(prediction_date): return morning_report_path(prediction_date).exists()
def mark_morning_report_sent(prediction_date):
    write_json(morning_report_path(prediction_date),{"PredictionDate":str(prediction_date),"ReportSent":True})

def save_predictions(df,prediction_date,metadata=None):
    path=prediction_path(prediction_date);tmp=path.with_suffix(".tmp");df.to_csv(tmp,index=False);tmp.replace(path)
    if metadata is not None: write_json(path.with_suffix(".json"),metadata)
    return path

def load_predictions(prediction_date):
    path=prediction_path(prediction_date)
    if not path.exists(): return pd.DataFrame()
    return pd.read_csv(path)

def latest_prediction_date(on_or_before=None):
    files=sorted(PREDICTIONS_DIR.glob("predictions_*.csv"));dates=[]
    for path in files:
        match=re.search(r"predictions_(\d{4}-\d{2}-\d{2})",path.name)
        if not match: continue
        value=pd.Timestamp(match.group(1)).date()
        if on_or_before is None or value<=on_or_before: dates.append(value)
    return max(dates) if dates else None

def evaluation_exists(market_date): return evaluation_path(market_date).exists()
def save_evaluation(df,market_date):
    path=evaluation_path(market_date);df.to_csv(path,index=False);return path

def append_daily_metrics(row):
    if DAILY_METRICS_FILE.exists():
        df=pd.read_csv(DAILY_METRICS_FILE);df=df[df["MarketDate"].astype(str)!=str(row["MarketDate"])]
        df=pd.concat([df,pd.DataFrame([row])],ignore_index=True)
    else: df=pd.DataFrame([row])
    df.to_csv(DAILY_METRICS_FILE,index=False)

def rebuild_stock_reliability():
    files=sorted(EVALUATIONS_DIR.glob("evaluation_*.csv"));frames=[]
    for path in files:
        try:
            df=pd.read_csv(path)
            if not df.empty: frames.append(df)
        except Exception: continue
    if not frames: return
    data=pd.concat(frames,ignore_index=True);rows=[]
    for symbol,group in data.groupby("Symbol"):
        samples=len(group);metrics={}
        for target in ["Open","High","Low","Close"]:
            column=f"APE_{target}"
            if column in group.columns: metrics[target]=float(group[column].mean())
        rows.append({"Symbol":symbol,"Samples":samples,**{f"MAPE_{k}":v for k,v in metrics.items()}})
    pd.DataFrame(rows).to_csv(STOCK_RELIABILITY_FILE,index=False)
