"""Stage 4.2 evening evaluation and retraining."""
import numpy as np
import pandas as pd
from .config import HISTORY_PERIOD
from .market_data import get_completed_session_date, get_previous_session_date, download_many, get_row_for_date, get_previous_row
from .ledger import load_predictions, evaluation_exists, save_evaluation, append_daily_metrics, rebuild_stock_reliability, latest_prediction_date
from .retraining import compare_variants
from .telegram_report import send_telegram, evening_report
from .utils import direction_from_prices, is_weekday


def calculate_cumulative_metrics():
    from .config import EVALUATIONS_DIR
    frames=[]
    for path in sorted(EVALUATIONS_DIR.glob("evaluation_*.csv")):
        try:
            df=pd.read_csv(path)
            if not df.empty: frames.append(df)
        except Exception: pass
    if not frames:
        return {"Samples":0,"OpenMAPE":0,"HighMAPE":0,"LowMAPE":0,"CloseMAPE":0,"VolumeMAPE":0,"OverallMAPE":0,"DirectionAccuracy":0}
    data=pd.concat(frames,ignore_index=True)
    vals=[float(data[f"APE_{t}"].abs().mean()) for t in ["Open","High","Low","Close"] if f"APE_{t}" in data]
    return {"Samples":len(data),"OpenMAPE":float(data["APE_Open"].abs().mean()),"HighMAPE":float(data["APE_High"].abs().mean()),"LowMAPE":float(data["APE_Low"].abs().mean()),"CloseMAPE":float(data["APE_Close"].abs().mean()),"VolumeMAPE":float(data["APE_Volume"].abs().mean()) if "APE_Volume" in data else 0.0,"OverallMAPE":float(np.mean(vals)) if vals else 0.0,"DirectionAccuracy":float(data["DirectionCorrect"].mean()*100)}


def run():
    if not is_weekday():
        print("Weekend. Evening evaluation skipped."); return
    market_date=get_completed_session_date("evening")
    if market_date is None: print("No completed market session."); return
    prediction_date=latest_prediction_date(market_date)
    if prediction_date is None: print("No morning prediction ledger found."); return
    predictions=load_predictions(prediction_date)
    if predictions.empty: print("Morning prediction file is empty."); return
    if evaluation_exists(market_date): print(f"Evaluation already exists for {market_date}. Skipping."); return
    symbols=predictions["Symbol"].astype(str).tolist(); data_map=download_many(symbols,HISTORY_PERIOD,workers=5); rows=[]
    for _,p in predictions.iterrows():
        df=data_map.get(p["Symbol"])
        if df is None: continue
        actual=get_row_for_date(df,market_date); previous=get_previous_row(df,market_date)
        if actual is None or previous is None: continue
        row={"MarketDate":str(market_date),"Symbol":p["Symbol"],"Pred_Open":float(p["Pred_Open"]),"Pred_High":float(p["Pred_High"]),"Pred_Low":float(p["Pred_Low"]),"Pred_Close":float(p["Pred_Close"]),"Pred_Volume":float(p.get("Pred_Volume",0)),"Actual_Open":float(actual["Open"]),"Actual_High":float(actual["High"]),"Actual_Low":float(actual["Low"]),"Actual_Close":float(actual["Close"]),"Actual_Volume":float(actual["Volume"]),"Pred_Direction":p["Direction"],"Actual_Direction":direction_from_prices(previous["Close"],actual["Close"])}
        row["DirectionCorrect"]=row["Pred_Direction"]==row["Actual_Direction"]
        for target in ["Open","High","Low","Close","Volume"]:
            pred=row[f"Pred_{target}"]; act=row[f"Actual_{target}"]; row[f"Diff_{target}"]=act-pred; row[f"APE_{target}"]=(act-pred)/max(abs(act),1e-8)*100
        rows.append(row)
    if not rows: print("No stocks could be evaluated."); return
    evaluation=pd.DataFrame(rows); save_evaluation(evaluation,market_date); metrics=calculate_cumulative_metrics(); append_daily_metrics({"MarketDate":str(market_date),**metrics}); rebuild_stock_reliability()
    previous_session=get_previous_session_date(market_date)
    retraining={"Retrained":False,"Decision":"NO PREVIOUS SESSION"}
    if previous_session is not None:
        retraining=compare_variants(download_many(symbols,HISTORY_PERIOD,workers=5),symbols,previous_session)
    report=evening_report(market_date,evaluation,metrics,retraining); send_telegram(report); print(report)

if __name__=="__main__": run()
