"""Evening evaluation, accuracy and learning engine for Stage 10.1."""
from __future__ import annotations
import pandas as pd
from .config import *
from .data_store import *
from .market_data import download_many
from .learning import update_learning_state, compare_variants, model_report_metrics
from .telegram_report import evening_report
from .portfolio_report import portfolio_snapshot


def _price_bucket(value):
    try:
        p = float(value)
    except Exception:
        return "<10"
    if p >= 1000: return ">1000"
    if p >= 500: return "500-999"
    if p >= 100: return "100-499"
    if p >= 50: return "50-99"
    if p >= 10: return "10-49"
    return "<10"


def _ensure_price_bucket(df):
    x = df.copy()
    if "PriceBucket" not in x.columns:
        x["PriceBucket"] = pd.NA
    sources = [c for c in ["Current_Price", "Pred_Close", "Current_Close", "Actual_Close"] if c in x.columns]
    missing = x["PriceBucket"].isna() | x["PriceBucket"].astype(str).str.strip().isin(["", "nan", "None"])
    if sources and missing.any():
        x.loc[missing, "PriceBucket"] = x.loc[missing, sources].bfill(axis=1).iloc[:, 0].map(_price_bucket)
    x["PriceBucket"] = x["PriceBucket"].fillna("<10")
    return x


def _safe_pct(actual, predicted):
    try:
        a, p = float(actual), float(predicted)
        return abs(p-a)/abs(a)*100 if a != 0 else 0.0
    except Exception:
        return 0.0


def _evaluate(data_map, predictions):
    rows=[]
    for _, p in predictions.iterrows():
        symbol=str(p.get("Symbol", ""))
        df=data_map.get(symbol)
        if df is None or df.empty: continue
        target_date=pd.Timestamp(p.get("PredictionDate")).date() + pd.tseries.offsets.BDay(1)
        target_date=target_date.date() if hasattr(target_date,"date") else target_date
        idx=pd.to_datetime(df.index).normalize()
        hits=df.loc[idx == pd.Timestamp(target_date)]
        if hits.empty: continue
        a=hits.iloc[-1]
        r=p.to_dict(); r["Actual_Open"]=float(a.get("Open",0)); r["Actual_High"]=float(a.get("High",0)); r["Actual_Low"]=float(a.get("Low",0)); r["Actual_Close"]=float(a.get("Close",0)); r["Actual_Volume"]=float(a.get("Volume",0))
        for field in ["Open","High","Low","Close","Volume"]:
            pred=r.get(f"Pred_{field}")
            act=r.get(f"Actual_{field}")
            try: r[f"Diff_{field}"]=float(pred)-float(act); r[f"APE_{field}"]=_safe_pct(act,pred)
            except Exception: r[f"Diff_{field}"]=0.0; r[f"APE_{field}"]=0.0
        r["PriceBucket"]=_price_bucket(r.get("Current_Price",r.get("Actual_Close",0)))
        rows.append(r)
    return pd.DataFrame(rows)


def _bucket_metrics(evaluation, predictions=None):
    x=_ensure_price_bucket(evaluation)
    if x.empty or "PriceBucket" not in x.columns: return {}
    out={}
    for bucket,g in x.dropna(subset=["PriceBucket"]).groupby("PriceBucket"):
        vals=[float(g[c].mean()) for c in ["APE_Open","APE_High","APE_Low","APE_Close","APE_Volume"] if c in g.columns and pd.notna(g[c]).any()]
        out[str(bucket)]={"Samples":int(len(g)),"MAPE":round(sum(vals)/len(vals),4) if vals else 0.0,"CloseMAPE":round(float(g["APE_Close"].mean()),4) if "APE_Close" in g else 0.0}
    return out


def _horizon_evaluations(market_date, data_map, predictions=None):
    if predictions is None: return pd.DataFrame()
    x=_ensure_price_bucket(predictions)
    rows=[]
    for _,p in x.iterrows():
        symbol=str(p.get("Symbol","")); df=data_map.get(symbol)
        if df is None or df.empty: continue
        base=pd.to_datetime(p.get("PredictionDate"),errors="coerce")
        if pd.isna(base): continue
        for h in [1,3,5,7,20]:
            col=f"Horizon_{h}D"
            if col not in p or pd.isna(p.get(col)): continue
            dates=pd.to_datetime(df.index).normalize()
            future=dates[dates > base.normalize()]
            if len(future) < h: continue
            target=future[h-1]
            if target.date() != pd.Timestamp(market_date).date(): continue
            actual=float(df.loc[dates==target].iloc[-1]["Close"])
            current=float(p.get("Current_Price",p.get("Current_Close",actual)))
            expected=float(p[col])
            predicted=current*(1+expected/100)
            r=p.to_dict(); r.update({"HorizonDays":h,"TargetDate":target.date().isoformat(),"Horizon_Pred_Close":predicted,"Horizon_Actual_Close":actual,"Horizon_Diff":predicted-actual,"Horizon_APE":_safe_pct(actual,predicted)})
            rows.append(r)
    return pd.DataFrame(rows)


def _horizon_metrics(df):
    if df is None or df.empty: return {}
    return {str(int(h))+"D":{"Samples":int(len(g)),"MAPE":round(float(g["Horizon_APE"].mean()),4)} for h,g in df.groupby("HorizonDays")}


def run():
    prediction_date=latest_prediction_date()
    if not prediction_date: print("No prediction ledger found."); return
    predictions=_ensure_price_bucket(load_predictions(prediction_date))
    symbols=predictions["Symbol"].dropna().astype(str).tolist() if "Symbol" in predictions else []
    data_map=download_many(symbols,HISTORY_PERIOD)
    evaluation=_evaluate(data_map,predictions)
    if evaluation.empty: print("No actual market rows available for evaluation."); return
    market_date=pd.Timestamp(evaluation.index[-1]).date() if isinstance(evaluation.index,pd.DatetimeIndex) else pd.Timestamp.now(tz="Asia/Kolkata").date()
    save_evaluation(evaluation,market_date)
    bucket=_bucket_metrics(evaluation,predictions)
    horizon_eval=_horizon_evaluations(market_date,data_map,predictions)
    horizon_stats=_horizon_metrics(horizon_eval)
    metrics=calculate_cumulative_metrics(evaluation)
    accuracy=model_report_metrics()
    accuracy.update({"AccuracySamples":metrics.get("Samples",0)})
    learning=update_learning_state(metrics,bucket,horizon_stats)
    portfolio=_portfolio_payload()
    evening_report(evaluation=evaluation,predictions=predictions,market_date=market_date,bucket=bucket,horizon_stats=horizon_stats,accuracy=accuracy,learning=learning,portfolio=portfolio)
    print(f"Evening evaluation complete: {len(evaluation)} stocks, {len(horizon_eval)} horizon observations")


if __name__=="__main__": run()
