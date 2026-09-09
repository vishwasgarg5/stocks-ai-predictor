from pathlib import Path
import re
import hashlib
import pandas as pd
from .config import PREDICTIONS_DIR,EVALUATIONS_DIR,JUMP_DIR,INTRADAY_DIR,DAILY_METRICS_FILE,STOCK_RELIABILITY_FILE,MODEL_VERSION,STAGE_NAME,TRANSACTION_COST_BPS,SLIPPAGE_BPS,MAX_PER_PRICE_BUCKET,PREDICTION_TOP_N
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

def _valid_prediction_ledger(df):
    if df.empty or "Symbol" not in df.columns or "Prediction_ID" not in df.columns:return False
    if df["Symbol"].isna().any() or df["Symbol"].astype(str).str.strip().eq("").any():return False
    if df["Symbol"].astype(str).nunique()!=len(df):return False
    if len(df)>int(PREDICTION_TOP_N):return False
    ids=df["Prediction_ID"].astype(str)
    if ids.str.lower().isin(["nan","none",""]).any() or ids.str.len().lt(2).any() or ids.nunique()!=len(df):return False
    required=["Pred_Open","Pred_High","Pred_Low","Pred_Close"]
    if any(c not in df.columns for c in required) or df[required].isna().any().any():return False
    for _,r in df.iterrows():
        try:
            o,h,l,c=[float(r[k]) for k in required]
            if not (l<=min(o,c) and h>=max(o,c)):return False
        except Exception:return False
    if "PriceBucket" in df.columns:
        counts=df["PriceBucket"].astype(str).value_counts()
        if (counts>int(MAX_PER_PRICE_BUCKET)).any():return False
    return True

def prediction_exists(prediction_date):
    path=prediction_path(prediction_date)
    if not path.exists():return False
    try:return _valid_prediction_ledger(pd.read_csv(path))
    except Exception:return False

def morning_report_sent(prediction_date):
    """Return True only for a successfully versioned Telegram delivery marker."""
    path=morning_report_path(prediction_date)
    if not path.exists():return False
    try:
        data=pd.read_json(path,typ="series")
        return bool(data.get("ReportSent") is True and data.get("DeliveryVersion")=="v2")
    except Exception:return False

def mark_morning_report_sent(prediction_date):
    """Write a versioned marker so legacy markers cannot suppress retries."""
    write_json(morning_report_path(prediction_date),{"PredictionDate":str(prediction_date),"ReportSent":True,"DeliveryVersion":"v2"})

def save_predictions(df,prediction_date,metadata=None):
    metadata=dict(metadata or {});metadata.setdefault("PredictionDate",str(prediction_date));metadata.setdefault("ModelVersion",MODEL_VERSION);metadata.setdefault("Stage",STAGE_NAME);path=prediction_path(prediction_date)
    if path.exists():
        try:
            existing=pd.read_csv(path)
            if not existing.empty:
                if not _valid_prediction_ledger(existing):raise ValueError("Existing prediction ledger is corrupt and will not be overwritten")
                enriched=_baseline_columns(df,metadata);new_ids=set(enriched.get("Prediction_ID",pd.Series(dtype=str)).astype(str));old_ids=set(existing["Prediction_ID"].astype(str))
                if old_ids!=new_ids:raise ValueError("Prediction ledger is immutable: existing Prediction_ID set differs")
                return path
        except ValueError:raise
        except Exception as exc:raise ValueError(f"Existing prediction ledger unreadable: {exc}")
    enriched=_baseline_columns(df,metadata)
    if not _valid_prediction_ledger(enriched):raise ValueError("Prediction ledger failed identity/OHLC/selection integrity validation")
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(".tmp");enriched.to_csv(tmp,index=False);tmp.replace(path)
    metadata["PredictionLedgerVersion"]="v8";metadata["PredictionLedgerKey"]="Prediction_ID";metadata["Baseline"]="Previous_Close";metadata["BaselineCostBps"]=float(TRANSACTION_COST_BPS)+float(SLIPPAGE_BPS);metadata["PredictionIDs"]=enriched["Prediction_ID"].astype(str).tolist();metadata["SelectionIntegrity"]={"MaxStocks":int(PREDICTION_TOP_N),"MaxPerPriceBucket":int(MAX_PER_PRICE_BUCKET)};write_json(path.with_suffix(".json"),metadata);return path

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