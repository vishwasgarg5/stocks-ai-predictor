"""Stage 4 morning pipeline: price buckets + OHLCV + jump + intraday."""
import pandas as pd
from .config import PRESCREEN_N, HISTORY_PERIOD, JUMP_CANDIDATE_N, MODEL_VERSION, TOP_N
from .market_data import load_universe, download_many, filter_liquid_universe, get_completed_session_date, get_data_cutoff_date, get_market_regime
from .features import technical_score
from .prediction import train_stock_bundle, predict_stock
from .selection import select_top_stocks, score_candidates
from .stage4_engine import add_stage4_context, select_price_bucket_candidates
from .jump_engine import generate_jump_watchlist
from .intraday_engine import generate_intraday_watchlist
from .ledger import prediction_exists, save_predictions, save_jump_predictions, save_intraday_predictions
from .retraining import load_model_state
from .telegram_report import send_telegram, morning_report
from .utils import today_ist, is_weekday, schedule_status

def _bucket_top_stocks(candidates, max_stocks=TOP_N):
    """Return at most one strong stock per price bucket; count can be < max_stocks."""
    if candidates is None or candidates.empty: return candidates.iloc[0:0] if candidates is not None else pd.DataFrame()
    rows=[]
    for bucket, group in candidates.groupby("PriceBucket", sort=False):
        if bucket=="OUT" or group.empty: continue
        rows.append(group.sort_values(["Score","Confidence","Direction_Confidence"],ascending=False).iloc[0])
    if not rows: return candidates.iloc[0:0]
    result=pd.DataFrame(rows).sort_values(["Score","Confidence"],ascending=False).head(max_stocks).reset_index(drop=True)
    return result

def run():
    prediction_date=today_ist()
    if not is_weekday(): print("Weekend. Morning prediction skipped."); return
    if prediction_exists(prediction_date): print(f"Prediction already exists for {prediction_date}. Skipping."); return

    universe=load_universe()
    raw_data=download_many(universe,HISTORY_PERIOD,workers=8)
    data_map=filter_liquid_universe(raw_data)
    if len(data_map)<20: raise RuntimeError("Too few liquid stocks.")

    # IMPORTANT: determine the cutoff from the downloaded equity bars, not a
    # conservative hard-coded T-2 rule. Today's date is always excluded.
    nifty_fallback=get_completed_session_date("morning",prediction_date)
    cutoff_date=get_data_cutoff_date(data_map,prediction_date,fallback=nifty_fallback)
    if cutoff_date is None: raise RuntimeError("Unable to determine completed market cutoff.")
    print(f"Prediction date: {prediction_date}; completed data cutoff: {cutoff_date}")

    regime=get_market_regime(cutoff_date)["name"]; variant=load_model_state().get("active_variant","A")

    scored=[]
    for symbol,df in data_map.items():
        try:
            d=df[df.index<=pd.Timestamp(cutoff_date)]
            scored.append((symbol,technical_score(d)))
        except Exception: pass
    scored.sort(key=lambda x:x[1],reverse=True); candidate_symbols=[x[0] for x in scored[:PRESCREEN_N]]

    candidate_rows=[]
    for symbol in candidate_symbols:
        try:
            bundle=train_stock_bundle(data_map[symbol],symbol,cutoff_date,variant)
            result=predict_stock(data_map[symbol],bundle,cutoff_date)
            candidate_rows.append({"Symbol":symbol,**result,"ModelVariant":variant,"ModelVersion":MODEL_VERSION,"DataCutoff":str(cutoff_date)})
        except Exception as exc: print(f"{symbol}: prediction failed: {exc}")
    if not candidate_rows: raise RuntimeError("Unable to generate predictions.")

    candidates=add_stage4_context(pd.DataFrame(candidate_rows),data_map,regime); candidates=candidates[candidates["PriceBucket"]!="OUT"].copy()
    if candidates.empty: raise RuntimeError("No candidates inside configured price buckets.")
    candidates=score_candidates(candidates,regime)
    bucket_candidates=select_price_bucket_candidates(candidates,per_bucket=max(3,TOP_N))
    selected=_bucket_top_stocks(bucket_candidates,TOP_N)
    if selected.empty: selected=select_top_stocks(candidates,min(TOP_N,len(candidates)),regime)
    selected["PredictionDate"]=str(prediction_date)

    save_predictions(selected,prediction_date,{"Stage":"Stage 4","PredictionDate":str(prediction_date),"DataCutoff":str(cutoff_date),"ModelVariant":variant,"ModelVersion":MODEL_VERSION,"Regime":regime,"PriceBuckets":[">1000","500-999","100-499","50-99","10-49"],"MultiHorizons":[1,3,5,7,20],"SelectedStocks":selected["Symbol"].tolist()})
    jump_data={s:data_map[s] for s in candidate_symbols[:JUMP_CANDIDATE_N] if s in data_map}; jump_watchlist=generate_jump_watchlist(jump_data,cutoff_date,variant)
    if not jump_watchlist.empty: save_jump_predictions(jump_watchlist,prediction_date)
    intraday=generate_intraday_watchlist(list(data_map.keys()))
    if not intraday.empty: save_intraday_predictions(intraday,prediction_date)
    report=morning_report(prediction_date,cutoff_date,selected,jump_watchlist,intraday)
    send_telegram(report); print(report)

if __name__=="__main__": run()
