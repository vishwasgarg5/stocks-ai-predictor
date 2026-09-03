"""Stage 4.5 morning pipeline: calibrated quality stocks + jump + intraday."""
import pandas as pd
from .config import PRESCREEN_N,HISTORY_PERIOD,JUMP_CANDIDATE_N,MODEL_VERSION,TOP_N,MAX_PER_PRICE_BUCKET
from .market_data import load_universe,download_many,filter_liquid_universe,get_completed_session_date,get_data_cutoff_date,get_market_regime
from .features import technical_score
from .prediction import train_stock_bundle,predict_stock,add_multihorizon_predictions
from .multihorizon import train_horizon_models
from .selection import select_top_stocks,score_candidates
from .stage4_engine import add_stage4_context
from .stage45_engine import add_prediction_uncertainty,add_market_risk
from .jump_engine import generate_jump_watchlist
from .intraday_engine import generate_intraday_watchlist
from .ledger import prediction_exists,save_predictions,save_jump_predictions,save_intraday_predictions
from .retraining import load_model_state
from .telegram_report import send_telegram,morning_report
from .utils import today_ist,is_weekday


def _bucket_candidates(candidates,max_per_bucket=2):
    if candidates is None or candidates.empty: return candidates.iloc[0:0] if candidates is not None else pd.DataFrame()
    pieces=[]
    for _,group in candidates.groupby("PriceBucket",sort=False):
        if not group.empty: pieces.append(group.sort_values(["Score","Confidence","Direction_Confidence"],ascending=False).head(max_per_bucket))
    return pd.concat(pieces,ignore_index=True) if pieces else candidates.iloc[0:0]


def _attach_horizons(candidates,data_map,cutoff_date):
    rows=[]
    for _,row in candidates.iterrows():
        symbol=row["Symbol"]
        try:
            hb=train_horizon_models(data_map[symbol],cutoff_date)
            h=add_multihorizon_predictions(data_map[symbol],{"horizons":hb},cutoff_date)
            if h.empty: row["MultiHorizonExpectedReturn"]=0.0
            else:
                rr=h["Expected_Return"].astype(float).clip(-50,50); row["MultiHorizonExpectedReturn"]=float(rr.median())
                for horizon in [1,3,5,7,20]:
                    m=h[h["HorizonDays"]==horizon]
                    if not m.empty: row[f"Horizon_{horizon}D"]=float(m.iloc[0]["Expected_Return"])
        except Exception as exc:
            print(f"{symbol}: horizon prediction failed: {exc}"); row["MultiHorizonExpectedReturn"]=0.0
        rows.append(row)
    return pd.DataFrame(rows) if rows else candidates.iloc[0:0]


def run():
    prediction_date=today_ist()
    if not is_weekday(): print("Weekend. Morning prediction skipped."); return
    if prediction_exists(prediction_date): print(f"Prediction already exists for {prediction_date}. Skipping."); return
    universe=load_universe(); raw_data=download_many(universe,HISTORY_PERIOD,workers=8); data_map=filter_liquid_universe(raw_data)
    if len(data_map)<20: raise RuntimeError("Too few liquid stocks.")
    nifty_fallback=get_completed_session_date("morning",prediction_date); cutoff_date=get_data_cutoff_date(data_map,prediction_date,fallback=nifty_fallback)
    if cutoff_date is None: raise RuntimeError("Unable to determine completed market cutoff.")
    print(f"Prediction date: {prediction_date}; completed data cutoff: {cutoff_date}")
    regime=get_market_regime(cutoff_date)["name"]; variant=load_model_state().get("active_variant","A")
    scored=[]
    for symbol,df in data_map.items():
        try: scored.append((symbol,technical_score(df[df.index<=pd.Timestamp(cutoff_date)])))
        except Exception: pass
    scored.sort(key=lambda x:x[1],reverse=True); candidate_symbols=[x[0] for x in scored[:PRESCREEN_N]]
    candidate_rows=[]; bundles={}
    for symbol in candidate_symbols:
        try:
            bundle=train_stock_bundle(data_map[symbol],symbol,cutoff_date,variant,train_horizons=False); bundles[symbol]=bundle
            result=predict_stock(data_map[symbol],bundle,cutoff_date)
            candidate_rows.append({"Symbol":symbol,**result,"ModelVariant":variant,"ModelVersion":MODEL_VERSION,"DataCutoff":str(cutoff_date)})
        except Exception as exc: print(f"{symbol}: prediction failed: {exc}")
    if not candidate_rows: raise RuntimeError("Unable to generate predictions.")
    candidates=add_stage4_context(pd.DataFrame(candidate_rows),data_map,regime); candidates=candidates[candidates["PriceBucket"]!="OUT"].copy()
    if candidates.empty: raise RuntimeError("No candidates inside configured price buckets.")
    candidates=score_candidates(candidates,regime)
    bucket_pool=_bucket_candidates(candidates,MAX_PER_PRICE_BUCKET).reset_index(drop=True)
    bucket_pool=_attach_horizons(bucket_pool,data_map,cutoff_date)
    bucket_pool=add_prediction_uncertainty(bucket_pool,data_map,bundles)
    bucket_pool=score_candidates(bucket_pool,regime)
    bucket_pool=add_market_risk(bucket_pool,regime)
    selected=select_top_stocks(bucket_pool,TOP_N,regime,min_score=65.0,min_confidence=60.0,min_trade_confidence=60.0,max_per_bucket=MAX_PER_PRICE_BUCKET)
    selected["PredictionDate"]=str(prediction_date)
    metadata={"Stage":"Stage 4.5","PredictionDate":str(prediction_date),"DataCutoff":str(cutoff_date),"ModelVariant":variant,"ModelVersion":MODEL_VERSION,"Regime":regime,"PriceBuckets":[">1000","500-999","100-499","50-99","10-49"],"MaxPerPriceBucket":MAX_PER_PRICE_BUCKET,"MaxSelectedStocks":TOP_N,"MultiHorizons":[1,3,5,7,20],"UncertaintyCalibration":True,"SelectedStocks":selected["Symbol"].tolist()}
    save_predictions(selected,prediction_date,metadata)
    jump_data={s:data_map[s] for s in candidate_symbols[:JUMP_CANDIDATE_N] if s in data_map}; jump_watchlist=generate_jump_watchlist(jump_data,cutoff_date,variant)
    if not jump_watchlist.empty: save_jump_predictions(jump_watchlist,prediction_date)
    intraday=generate_intraday_watchlist(list(data_map.keys()))
    if not intraday.empty: save_intraday_predictions(intraday,prediction_date)
    report=morning_report(prediction_date,cutoff_date,selected,jump_watchlist,intraday); send_telegram(report); print(report)

if __name__=="__main__": run()
