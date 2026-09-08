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

def morning_report_sent(prediction_date): return morning_report_path(prediction_date).exists()
def mark_morning_report_sent(prediction_date):write_json(morning_report_path(prediction_date),{"PredictionDate":str(prediction_date),"ReportSent":True})

def save_predictions(df,prediction_date,metadata=None):
    metadata=dict(metadata or {});metadata.setdefault("PredictionDate",str(prediction_date));metadata.setdefault("ModelVersion",MODEL_VERSION);metadata.setdefault("Stage",STAGE_NAME);path=prediction_path(prediction_date)
    # Existing prediction sets are immutable. Check the persisted artifact first so a
    # harmless re-run never validates/replaces a newly supplied dataframe.
    if path.exists():
        try:
            existing=pd.read_csv(path)
            if not existing.empty:
                if not _valid_prediction_ledger(existing):raise ValueError("Existing prediction ledger is corrupt and will not be overwritten")
                enriched=_baseline_columns(df,metadata)
                new_ids=set(enriched.get("Prediction_ID",pd.Series(dtype=str)).astype(str))
                old_ids=set(existing["Prediction_ID"].astype(str))
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
def save_jump_predictions(df,prediction_date):path=jump_path(prediction_date);df.to_csv(path,index=False);return path
def save_intraday_predictions(df,prediction_date):path=intraday_path(prediction_date);df.to_csv(path,index=False);return path

def latest_prediction_date(on_or_before=None):
    if on_or_before is not None:
        exact=pd.Timestamp(on_or_before).date();return exact if prediction_path(exact).exists() else None
    files=sorted(PREDICTIONS_DIR.glob("predictions_*.csv"));dates=[]
    for path in files:
        match=re.search(r"predictions_(\d{4}-\d{2}-\d{2})",path.name)
        if match:dates.append(pd.Timestamp(match.group(1)).date())
    return max(dates) if dates else None

def evaluation_exists(market_date):return evaluation_path(market_date).exists()
def save_evaluation(df,market_date):
    path=evaluation_path(market_date);x=df.copy()
    if x.empty or "Symbol" not in x.columns or "PredictionDate" not in x.columns:raise ValueError("Evaluation requires Symbol and PredictionDate")
    if x["Symbol"].astype(str).duplicated().any():raise ValueError("Evaluation contains duplicate symbols")
    if "Prediction_ID" not in x.columns:x["Prediction_ID"]=""
    for pdate,idxs in x.groupby(x["PredictionDate"].astype(str)).groups.items():
        canonical=load_predictions(pd.Timestamp(pdate).date())
        if canonical.empty or "Prediction_ID" not in canonical.columns:raise ValueError(f"Missing canonical prediction ledger for {pdate}")
        ids=canonical[["Symbol","Prediction_ID"]].drop_duplicates("Symbol").set_index("Symbol")["Prediction_ID"].to_dict()
        for idx in idxs:
            pid=ids.get(str(x.at[idx,"Symbol"]))
            if not pid:raise ValueError(f"No canonical Prediction_ID for {x.at[idx,'Symbol']} on {pdate}")
            x.at[idx,"Prediction_ID"]=pid
    if x["Prediction_ID"].astype(str).str.lower().isin(["nan","none",""]).any() or x["Prediction_ID"].astype(str).str.len().lt(2).any():raise ValueError("Evaluation contains unbound Prediction_ID")
    if x.duplicated(["Prediction_ID","Symbol"],keep=False).any():raise ValueError("Evaluation contains duplicate prediction lineage")
    canonical=load_predictions(pd.Timestamp(str(x["PredictionDate"].iloc[0])).date())
    binding=canonical[["Symbol","Prediction_ID"]].set_index("Symbol")["Prediction_ID"].to_dict()
    for _,r in x.iterrows():
        if binding.get(str(r.Symbol))!=str(r.Prediction_ID):raise ValueError("Evaluation Prediction_ID does not bind to canonical prediction")
    if path.exists():
        try:
            existing=pd.read_csv(path)
            if not existing.empty:
                if set(existing.get("Prediction_ID",pd.Series(dtype=str)).astype(str))!=set(x["Prediction_ID"].astype(str)):raise ValueError("Evaluation is immutable: Prediction_ID set differs")
                return path
        except ValueError:raise
        except Exception as exc:raise ValueError(f"Existing evaluation unreadable: {exc}")
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
