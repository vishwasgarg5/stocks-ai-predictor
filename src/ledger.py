from pathlib import Path
import os
import re
import hashlib
import pandas as pd
from .config import PREDICTIONS_DIR,EVALUATIONS_DIR,JUMP_DIR,INTRADAY_DIR,DAILY_METRICS_FILE,STOCK_RELIABILITY_FILE,JUMP_METRICS_FILE,INTRADAY_METRICS_FILE,MODEL_VERSION,STAGE_NAME,TRANSACTION_COST_BPS,SLIPPAGE_BPS
from .utils import write_json

def prediction_path(prediction_date): return PREDICTIONS_DIR / f"predictions_{prediction_date}.csv"
def evaluation_path(market_date): return EVALUATIONS_DIR / f"evaluation_{market_date}.csv"
def jump_path(prediction_date): return JUMP_DIR / f"jump_{prediction_date}.csv"
def intraday_path(prediction_date): return INTRADAY_DIR / f"intraday_{prediction_date}.csv"
def morning_report_path(prediction_date): return PREDICTIONS_DIR / f"morning_report_{prediction_date}.json"
def _prediction_id(run_date,symbol,cutoff_date,model_version): return "P"+hashlib.sha256(f"{run_date}|{symbol}|{cutoff_date}|{model_version}".encode()).hexdigest()[:16]
def _baseline_columns(df,metadata=None):
    x=df.copy();meta=metadata or {};run_date=meta.get("PredictionDate",x.get("PredictionDate",pd.Series([""])).iloc[0] if len(x) else "");cutoff=meta.get("DataCutoff",x.get("DataCutoff",pd.Series([run_date])).iloc[0] if len(x) else run_date);version=meta.get("ModelVersion",MODEL_VERSION)
    if "Symbol" in x.columns:x["Prediction_ID"]=[_prediction_id(run_date,str(s),cutoff,version) for s in x["Symbol"].astype(str)]
    x["Run_Date"]=str(run_date);x["Cutoff_Date"]=str(cutoff);x["Model_Version"]=str(version)
    baseline=None
    for c in ["Previous_Close","Prev_Close","Baseline_Close","Current_Close","Current_Price"]:
        if c in x.columns:baseline=pd.to_numeric(x[c],errors="coerce");break
    if baseline is not None:
        for target in ["Open","High","Low","Close"]:x[f"Baseline_{target}"]=baseline
        x["Baseline_Close"]=baseline;x["Baseline_Cost_Pct"]=(float(TRANSACTION_COST_BPS)+float(SLIPPAGE_BPS))/100.0
    return x
def prediction_exists(prediction_date):
    path=prediction_path(prediction_date)
    if not path.exists():return False
    try:df=pd.read_csv(path);return len(df)>=1 and "Symbol" in df.columns and "Prediction_ID" in df.columns
    except Exception:return False
def morning_report_sent(prediction_date):return False if os.getenv("GITHUB_EVENT_NAME")=="workflow_dispatch" else morning_report_path(prediction_date).exists()
def mark_morning_report_sent(prediction_date):write_json(morning_report_path(prediction_date),{"PredictionDate":str(prediction_date),"ReportSent":True})
def save_predictions(df,prediction_date,metadata=None):
    metadata=dict(metadata or {});metadata.setdefault("PredictionDate",str(prediction_date));metadata.setdefault("ModelVersion",MODEL_VERSION);metadata.setdefault("Stage",STAGE_NAME);path=prediction_path(prediction_date)
    if path.exists():
        try:
            existing=pd.read_csv(path)
            if not existing.empty:
                # Canonical morning forecasts are immutable. A rerun must never replace them.
                return path
        except Exception: return path
    enriched=_baseline_columns(df,metadata);tmp=path.with_suffix(".tmp");enriched.to_csv(tmp,index=False);tmp.replace(path)
    metadata["PredictionLedgerVersion"]="v4";metadata["PredictionLedgerKey"]="Prediction_ID";metadata["Baseline"]="Previous_Close";metadata["BaselineCostBps"]=float(TRANSACTION_COST_BPS)+float(SLIPPAGE_BPS);metadata["PredictionIDs"]=enriched["Prediction_ID"].astype(str).tolist() if "Prediction_ID" in enriched else []
    write_json(path.with_suffix(".json"),metadata);return path
def load_predictions(prediction_date):
    path=prediction_path(prediction_date)
    if not path.exists():return pd.DataFrame()
    try:
        df=pd.read_csv(path)
        if "Prediction_ID" not in df.columns:df=_baseline_columns(df,{"PredictionDate":str(prediction_date)})
        return df
    except Exception:return pd.DataFrame()
def load_jump_predictions(prediction_date):path=jump_path(prediction_date);return pd.read_csv(path) if path.exists() else pd.DataFrame()
def load_intraday_predictions(prediction_date):path=intraday_path(prediction_date);return pd.read_csv(path) if path.exists() else pd.DataFrame()
def save_jump_predictions(df,prediction_date):path=jump_path(prediction_date);df.to_csv(path,index=False);return path
def save_intraday_predictions(df,prediction_date):path=intraday_path(prediction_date);df.to_csv(path,index=False);return path
def latest_prediction_date(on_or_before=None):
    if on_or_before is not None:
        exact=pd.Timestamp(on_or_before).date();path=prediction_path(exact)
        if path.exists():return exact
        return None
    files=sorted(PREDICTIONS_DIR.glob("predictions_*.csv"));dates=[]
    for path in files:
        match=re.search(r"predictions_(\d{4}-\d{2}-\d{2})",path.name)
        if match:dates.append(pd.Timestamp(match.group(1)).date())
    return max(dates) if dates else None
def evaluation_exists(market_date):return evaluation_path(market_date).exists()
def save_evaluation(df,market_date):
    path=evaluation_path(market_date);x=df.copy()
    if "Prediction_ID" not in x.columns and {"PredictionDate","Symbol"}.issubset(x.columns):x["Prediction_ID"]=[_prediction_id(r.PredictionDate,r.Symbol,r.get("Cutoff_Date",r.PredictionDate),r.get("Model_Version",MODEL_VERSION)) for _,r in x.iterrows()]
    if path.exists():
        try:
            existing=pd.read_csv(path)
            if not existing.empty:return path
        except Exception:return path
    path.parent.mkdir(parents=True,exist_ok=True);x.to_csv(path,index=False);return path
def append_daily_metrics(row):
    if DAILY_METRICS_FILE.exists():df=pd.read_csv(DAILY_METRICS_FILE);df=df[df["MarketDate"].astype(str)!=str(row["MarketDate"])]
    else:df=pd.DataFrame()
    df=pd.concat([df,pd.DataFrame([row])],ignore_index=True);df.to_csv(DAILY_METRICS_FILE,index=False)
def rebuild_stock_reliability():
    frames=[]
    for path in sorted(EVALUATIONS_DIR.glob("evaluation_*.csv")):
        try:
            df=pd.read_csv(path)
            if not df.empty:frames.append(df)
        except Exception:continue
    if not frames:return
    data=pd.concat(frames,ignore_index=True);rows=[]
    for symbol,group in data.groupby("Symbol"):
        apes=[]
        for target in ["Open","High","Low","Close"]:
            col=f"APE_{target}"
            if col in group.columns:apes.append(pd.to_numeric(group[col],errors="coerce").abs().mean())
        close_ape=float(pd.to_numeric(group.get("APE_Close",pd.Series(dtype=float)),errors="coerce").abs().mean()) if "APE_Close" in group else 3.0
        direction=float(pd.to_numeric(group.get("DirectionCorrect",pd.Series(dtype=float)),errors="coerce").mean()*100) if "DirectionCorrect" in group else 50.0
        rows.append({"Symbol":symbol,"Samples":len(group),"MAPE":float(sum(apes)/len(apes)) if apes else close_ape,"MAPE_Open":float(pd.to_numeric(group.get("APE_Open",pd.Series(dtype=float)),errors="coerce").abs().mean()) if "APE_Open" in group else close_ape,"MAPE_High":float(pd.to_numeric(group.get("APE_High",pd.Series(dtype=float)),errors="coerce").abs().mean()) if "APE_High" in group else close_ape,"MAPE_Low":float(pd.to_numeric(group.get("APE_Low",pd.Series(dtype=float)),errors="coerce").abs().mean()) if "APE_Low" in group else close_ape,"MAPE_Close":close_ape,"DirectionAccuracy":direction})
    pd.DataFrame(rows).to_csv(STOCK_RELIABILITY_FILE,index=False)
