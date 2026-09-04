"""Evening evaluation, bucket performance, horizon learning and portfolio report."""
from pathlib import Path
import numpy as np
import pandas as pd
from .config import HISTORY_PERIOD,FINAL_LEARNING_STATE_FILE,PREDICTIONS_DIR,MULTI_HORIZONS
from .market_data import get_completed_session_date,get_previous_session_date,download_many,get_row_for_date,get_previous_row
from .ledger import load_predictions,evaluation_exists,save_evaluation,append_daily_metrics,rebuild_stock_reliability,latest_prediction_date
from .retraining import compare_variants
from .final_intelligence import update_learning_state
from .telegram_report import send_telegram,evening_report
from .portfolio_report import portfolio_snapshot
from .report_metrics import model_report_metrics
from .utils import direction_from_prices,is_weekday

def calculate_cumulative_metrics(exclude_market_date=None):
    from .config import EVALUATIONS_DIR
    frames=[]
    for path in sorted(EVALUATIONS_DIR.glob("evaluation_*.csv")):
        if exclude_market_date is not None and path.stem.endswith(str(exclude_market_date)):continue
        try:
            df=pd.read_csv(path)
            if not df.empty:frames.append(df)
        except Exception:pass
    if not frames:return {"Samples":0,"OpenMAPE":0,"HighMAPE":0,"LowMAPE":0,"CloseMAPE":0,"VolumeMAPE":0,"OverallMAPE":0,"DirectionAccuracy":0}
    data=pd.concat(frames,ignore_index=True)
    vals=[float(data[f"APE_{t}"].abs().mean()) for t in ["Open","High","Low","Close"] if f"APE_{t}" in data]
    return {"Samples":len(data),"OpenMAPE":float(data["APE_Open"].abs().mean()),"HighMAPE":float(data["APE_High"].abs().mean()),"LowMAPE":float(data["APE_Low"].abs().mean()),"CloseMAPE":float(data["APE_Close"].abs().mean()),"VolumeMAPE":float(data["APE_Volume"].abs().mean()) if "APE_Volume" in data else 0.0,"OverallMAPE":float(np.mean(vals)) if vals else 0.0,"DirectionAccuracy":float(data["DirectionCorrect"].mean()*100)}

def _model_accuracy(metrics):return max(0.0,min(100.0,100.0-float(metrics.get("CloseMAPE",100.0))))

def _price_bucket(value):
    try:p=float(value)
    except Exception:return "-"
    if not np.isfinite(p):return "-"
    if p>=1000:return ">1000"
    if p>=500:return "500-999"
    if p>=100:return "100-499"
    if p>=50:return "50-99"
    if p>=10:return "10-49"
    return "<10"

def _ensure_price_bucket(predictions):
    """Keep evening evaluation resilient to older ledgers without PriceBucket."""
    x=predictions.copy()
    if "PriceBucket" not in x.columns:
        x["PriceBucket"]=np.nan
    missing=x["PriceBucket"].isna() | x["PriceBucket"].astype(str).str.strip().isin(["","nan","None","-"])
    if missing.any():
        source=None
        for col in ["Current_Price","Pred_Close","Current_Close"]:
            if col in x.columns:
                source=col
                break
        if source is not None:
            x.loc[missing,"PriceBucket"]=x.loc[missing,source].map(_price_bucket)
    x["PriceBucket"]=x["PriceBucket"].fillna("-").astype(str)
    return x

def _bucket_metrics(evaluation,predictions):
    if evaluation is None or evaluation.empty or predictions is None or predictions.empty:return {}
    p=_ensure_price_bucket(predictions)
    cols=[c for c in ["Symbol","PriceBucket"] if c in p.columns]
    if "Symbol" not in cols or "PriceBucket" not in cols:return {}
    x=evaluation.merge(p[cols].drop_duplicates("Symbol"),on="Symbol",how="left")
    result={}
    for bucket,g in x.dropna(subset=["PriceBucket"]).groupby("PriceBucket"):
        result[str(bucket)]=float((1-g["APE_Close"].abs()/100).clip(0,1).mean()*100)
    return result

def _horizon_evaluations(market_date,data_map):
    """Evaluate stored 1/3/5/7/20-session forecasts when their target is today's market date."""
    market=pd.Timestamp(market_date).date();rows=[]
    for path in sorted(PREDICTIONS_DIR.glob("predictions_*.csv")):
        try:
            prediction_date=pd.Timestamp(path.stem.replace("predictions_","")).date();pred=_ensure_price_bucket(pd.read_csv(path))
        except Exception:continue
        if pred.empty or "Symbol" not in pred.columns:continue
        for horizon in MULTI_HORIZONS:
            col=f"Horizon_{horizon}D"
            if col not in pred.columns:continue
            for _,p in pred.iterrows():
                symbol=str(p.get("Symbol",""));df=data_map.get(symbol)
                if df is None or df.empty:continue
                idx=pd.DatetimeIndex(df.index).normalize();base=pd.Timestamp(prediction_date)
                matches=np.where(idx==base)[0]
                if len(matches)==0:continue
                target_pos=int(matches[0])+int(horizon)
                if target_pos>=len(df):continue
                target_date=idx[target_pos].date()
                if target_date!=market:continue
                actual=float(df.iloc[target_pos]["Close"]);current=float(p.get("Current_Price",0) or 0);expected=float(p.get(col,0) or 0);predicted=current*(1+expected/100) if current else np.nan
                if not np.isfinite(predicted):continue
                diff=actual-predicted;ape=diff/max(abs(actual),1e-8)*100
                rows.append({"Symbol":symbol,"PredictionDate":str(prediction_date),"HorizonDays":horizon,"Pred_Close":predicted,"Actual_Close":actual,"Diff":diff,"APE":ape,"PriceBucket":p.get("PriceBucket","-")})
    return pd.DataFrame(rows)

def _horizon_metrics(h):
    if h is None or h.empty:return {}
    out={}
    for horizon,g in h.groupby("HorizonDays"):
        out[str(int(horizon))]={"Samples":int(len(g)),"MAPE":float(g["APE"].abs().mean()),"Accuracy":float((1-g["APE"].abs()/100).clip(0,1).mean()*100)}
    return out

def _portfolio_payload():
    try:
        df,s=portfolio_snapshot();s=dict(s);s["Rows"]=[]
        if not df.empty:
            for _,r in df.sort_values("PnL").head(8).iterrows():
                price="-" if pd.isna(r.Current_Price) else f"₹{r.Current_Price:,.2f}";ret="-" if pd.isna(r.Return_Pct) else f"{r.Return_Pct:+.1f}%";avg="-" if pd.isna(r.Average_Price) else f"₹{r.Average_Price:,.2f}";target="-" if pd.isna(r.AI_Target) else f"₹{r.AI_Target:,.2f}";rec=str(r.get("Recommended_Qty",0));newavg="-" if pd.isna(r.get("New_Average_Price")) else f"₹{float(r.New_Average_Price):,.2f}";pa=str(r.get("Averaging_Action","-"))
                s["Rows"].append(f"{r.Stock}: CMP {price} | Avg {avg} | AI {target} | {ret} | {pa} {rec if pa=='AVERAGE' else ''} | NewAvg {newavg}")
        else:s["Rows"].append("Portfolio data unavailable")
        return s
    except Exception as exc:print(f"Portfolio report unavailable: {exc}");return {"Positions":0,"Value":0.0,"PnL":0.0,"Return":0.0,"Rows":["Portfolio data unavailable"]}

def run():
    if not is_weekday():print("Weekend. Evening evaluation skipped.");return
    market_date=get_completed_session_date("evening")
    if market_date is None:print("No completed market session.");return
    prediction_date=latest_prediction_date(market_date)
    if prediction_date is None:print("No morning prediction ledger found.");return
    predictions=load_predictions(prediction_date)
    if predictions.empty:print("Morning prediction file is empty.");return
    predictions=_ensure_price_bucket(predictions)
    if evaluation_exists(market_date):print(f"Evaluation already exists for {market_date}. Skipping.");return
    symbols=predictions["Symbol"].astype(str).tolist();data_map=download_many(symbols,HISTORY_PERIOD,workers=5);rows=[]
    for _,p in predictions.iterrows():
        df=data_map.get(p["Symbol"])
        if df is None:continue
        actual=get_row_for_date(df,market_date);previous=get_previous_row(df,market_date)
        if actual is None or previous is None:continue
        row={"MarketDate":str(market_date),"PredictionDate":str(prediction_date),"Symbol":p["Symbol"],"PriceBucket":p.get("PriceBucket","-"),"Pred_Open":float(p["Pred_Open"]),"Pred_High":float(p["Pred_High"]),"Pred_Low":float(p["Pred_Low"]),"Pred_Close":float(p["Pred_Close"]),"Pred_Volume":float(p.get("Pred_Volume",0)),"Actual_Open":float(actual["Open"]),"Actual_High":float(actual["High"]),"Actual_Low":float(actual["Low"]),"Actual_Close":float(actual["Close"]),"Actual_Volume":float(actual["Volume"]),"Pred_Direction":p["Direction"],"Actual_Direction":direction_from_prices(previous["Close"],actual["Close"])}
        row["DirectionCorrect"]=row["Pred_Direction"]==row["Actual_Direction"]
        for target in ["Open","High","Low","Close","Volume"]:
            pred=row[f"Pred_{target}"];act=row[f"Actual_{target}"];row[f"Diff_{target}"]=act-pred;row[f"APE_{target}"]=(act-pred)/max(abs(act),1e-8)*100
        rows.append(row)
    if not rows:print("No stocks could be evaluated.");return
    previous_metrics=calculate_cumulative_metrics(exclude_market_date=market_date);previous_accuracy=_model_accuracy(previous_metrics);evaluation=pd.DataFrame(rows);save_evaluation(evaluation,market_date);metrics=calculate_cumulative_metrics();current_accuracy=_model_accuracy(metrics);append_daily_metrics({"MarketDate":str(market_date),**metrics,"ModelAccuracy":current_accuracy});rebuild_stock_reliability()
    previous_session=get_previous_session_date(market_date);retraining={"Retrained":False,"Decision":"NO PREVIOUS SESSION"}
    if previous_session is not None:retraining=compare_variants(download_many(symbols,HISTORY_PERIOD,workers=5),symbols,previous_session)
    bucket=_bucket_metrics(evaluation,predictions);horizon_eval=_horizon_evaluations(market_date,data_map);horizon_stats=_horizon_metrics(horizon_eval);accuracy=model_report_metrics();accuracy.update({"PreviousAccuracy":previous_accuracy,"CurrentAccuracy":current_accuracy,"AccuracySamples":metrics.get("Samples",0)})
    update_learning_state(FINAL_LEARNING_STATE_FILE,{"date":str(market_date),"prediction_date":str(prediction_date),"metrics":metrics,"accuracy":accuracy,"retraining":retraining,"bucket_metrics":bucket,"horizon_metrics":horizon_stats,"horizon_samples":int(len(horizon_eval)),"symbols_evaluated":symbols})
    p=_portfolio_payload();report=evening_report(market_date,evaluation,metrics,retraining,bucket_metrics=bucket,horizon_metrics=horizon_stats,learning={"status":"UPDATED","drift":accuracy.get("Drift"),"health":accuracy.get("Health")},scan={"Universe":len(symbols),"Data":len(data_map),"Liquid":len(data_map),"AI":len(symbols),"Selected":len(evaluation)},accuracy=accuracy,portfolio=p);send_telegram(report);print(report)

if __name__=="__main__":run()
