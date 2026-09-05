"""Stage 10.4 morning pipeline: prediction, ranking, probability, risk and abstention."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import PRESCREEN_N,HISTORY_PERIOD,JUMP_CANDIDATE_N,MODEL_VERSION,STAGE_NAME,MAX_PER_PRICE_BUCKET,FINAL_BEST_PER_BUCKET,FINAL_LEARNING_STATE_FILE,IPO_METRICS_FILE,VOLATILITY_LOW_PCT,VOLATILITY_HIGH_PCT
from .market_data import load_universe,download_many,filter_liquid_universe,get_completed_session_date,get_data_cutoff_date,get_market_regime,get_market_snapshot
from .features import technical_score
from .prediction import train_stock_bundle,predict_stock,add_multihorizon_predictions
from .multihorizon import train_horizon_models
from .selection import select_top_stocks,score_candidates
from .stage4_engine import add_stage4_context
from .stage45_engine import add_prediction_uncertainty,add_market_risk
from .final_intelligence import apply_final_intelligence,update_learning_state,final_stage_manifest,benchmark_expected_return
from .jump_engine import generate_jump_watchlist
from .intraday_engine import generate_intraday_watchlist
from .ipo_runner import get_ipo_report
from .ledger import prediction_exists,load_predictions,save_predictions,save_jump_predictions,save_intraday_predictions,load_jump_predictions,load_intraday_predictions,morning_report_sent,mark_morning_report_sent
from .decision_ledger import save_decisions
from .retraining import load_model_state
from .telegram_report import send_telegram,morning_report
from .portfolio_report import portfolio_snapshot
from .report_metrics import model_report_metrics
from .utils import today_ist,is_weekday

def _bucket_candidates(candidates,max_per_bucket=2):
    if candidates is None or candidates.empty:return candidates.iloc[0:0] if candidates is not None else pd.DataFrame()
    pieces=[]
    for _,group in candidates.groupby("PriceBucket",sort=False):pieces.append(group.sort_values(["Score","Confidence","Direction_Confidence"],ascending=False).head(max_per_bucket))
    return pd.concat(pieces,ignore_index=True) if pieces else candidates.iloc[0:0]

def _attach_horizons(candidates,data_map,cutoff_date):
    rows=[]
    for _,row in candidates.iterrows():
        symbol=row["Symbol"]
        try:
            hb=train_horizon_models(data_map[symbol],cutoff_date);h=add_multihorizon_predictions(data_map[symbol],{"horizons":hb},cutoff_date);row["MultiHorizonExpectedReturn"]=0.0 if h.empty else float(h["Expected_Return"].astype(float).clip(-50,50).median())
            if not h.empty:
                current=float(row.get("Current_Price",0) or 0)
                if not current:
                    valid=data_map[symbol][data_map[symbol].index<=pd.Timestamp(cutoff_date)];current=float(valid.iloc[-1]["Close"]) if not valid.empty else 0.0
                for horizon in [1,3,5,7,20]:
                    m=h[h["HorizonDays"]==horizon]
                    if not m.empty:
                        expected=float(m.iloc[0]["Expected_Return"]);row[f"Horizon_{horizon}D"]=expected;row[f"Horizon_{horizon}D_Pred_Close"]=current*(1+expected/100) if current else 0.0;row[f"Horizon_{horizon}D_MAPE"]=float(m.iloc[0].get("ValidationMAPE",np.nan))
        except Exception as exc:print(f"{symbol}: horizon prediction failed: {exc}");row["MultiHorizonExpectedReturn"]=0.0
        rows.append(row)
    return pd.DataFrame(rows) if rows else candidates.iloc[0:0]

def _attach_benchmarks_and_risk(candidates,data_map,cutoff_date):
    out=candidates.copy();vol_bucket=[];vol_pct=[];bench=[]
    for _,r in out.iterrows():
        df=data_map.get(r["Symbol"])
        if df is None or df.empty:vol_bucket.append("UNKNOWN");vol_pct.append(np.nan);bench.append(0.0);continue
        valid=df[df.index<=pd.Timestamp(cutoff_date)].copy();close=pd.to_numeric(valid.get("Close"),errors="coerce").dropna();ret=close.pct_change().dropna()*100
        v=float(ret.tail(20).std()) if len(ret)>=5 else np.nan
        vol_pct.append(v);vol_bucket.append("UNKNOWN" if not np.isfinite(v) else ("LOW" if v<VOLATILITY_LOW_PCT else ("HIGH" if v>=VOLATILITY_HIGH_PCT else "MEDIUM")));bench.append(benchmark_expected_return(float(close.iloc[-1]) if len(close) else 0.0,valid))
    out["VolatilityPct"]=vol_pct;out["VolatilityBucket"]=vol_bucket;out["BenchmarkExpectedReturn"]=bench;out["BenchmarkEdgePct"]=pd.to_numeric(out.get("Expected_Return",0),errors="coerce").fillna(0)-pd.Series(bench,index=out.index)
    rank_col="FinalScore" if "FinalScore" in out.columns else "Score";out["CrossSectionRank"]=out.groupby("PriceBucket")[rank_col].rank(ascending=False,method="min");out["CrossSectionCount"]=out.groupby("PriceBucket")["Symbol"].transform("count");out["CrossSectionPercentile"]=(1-(out["CrossSectionRank"]-1)/out["CrossSectionCount"].clip(lower=1))*100
    if "SectorReturn20D" in out.columns:
        sector_median=out.groupby("Sector")["SectorReturn20D"].transform("median");out["SectorRelative20D"]=pd.to_numeric(out["SectorReturn20D"],errors="coerce")-pd.to_numeric(sector_median,errors="coerce")
    else:out["SectorRelative20D"]=0.0
    return out

def _attach_current_ohlcv(selected,data_map,cutoff_date):
    out=selected.copy()
    for c in ["Current_Open","Current_High","Current_Low","Current_Close","Current_Volume"]:out[c]=0.0
    for idx,r in out.iterrows():
        df=data_map.get(r["Symbol"])
        if df is None or df.empty:continue
        valid=df[df.index.date<=pd.Timestamp(cutoff_date).date()]
        if valid.empty:continue
        x=valid.iloc[-1];out.loc[idx,"Current_Open"]=float(x["Open"]);out.loc[idx,"Current_High"]=float(x["High"]);out.loc[idx,"Current_Low"]=float(x["Low"]);out.loc[idx,"Current_Close"]=float(x["Close"]);out.loc[idx,"Current_Volume"]=float(x["Volume"])
    return out

def _portfolio_payload():
    try:
        df,s=portfolio_snapshot();s=dict(s);s["Rows"]=[];s["Available"]=not df.empty
        if not df.empty:
            for _,r in df.sort_values("PnL").head(8).iterrows():
                price="-" if pd.isna(r.Current_Price) else f"₹{r.Current_Price:,.2f}";ret="-" if pd.isna(r.Return_Pct) else f"{r.Return_Pct:+.1f}%";avg="-" if pd.isna(r.Average_Price) else f"₹{r.Average_Price:,.2f}";target="-" if pd.isna(r.AI_Target) else f"₹{r.AI_Target:,.2f}";rec=str(r.get("Recommended_Qty",0));newavg="-" if pd.isna(r.get("New_Average_Price")) else f"₹{float(r.New_Average_Price):,.2f}";pa=str(r.get("Averaging_Action","-"));s["Rows"].append(f"{r.Stock}: CMP {price} | Avg {avg} | AI {target} | {ret} | {pa} {rec if pa=='AVERAGE' else ''} | NewAvg {newavg}")
        else:s["Rows"].append("Portfolio data unavailable")
        return s
    except Exception as exc:print(f"Portfolio report unavailable: {exc}");return {"Positions":0,"Value":0.0,"PnL":0.0,"Return":0.0,"ActionCounts":{},"Rows":["Portfolio data unavailable"],"Available":False}

def _prediction_metadata(prediction_date):
    path=Path(f"data/stage2/predictions/predictions_{prediction_date}.json")
    try:return json.loads(path.read_text()) if path.exists() else {}
    except Exception:return {}

def _send_existing_report(prediction_date):
    predictions=load_predictions(prediction_date)
    if predictions.empty:return False
    meta=_prediction_metadata(prediction_date);cutoff=meta.get("DataCutoff",prediction_date);jump_watchlist=load_jump_predictions(prediction_date);intraday=load_intraday_predictions(prediction_date);ipo=get_ipo_report();scan={"Universe":meta.get("StocksScanned",meta.get("Universe",0)),"Data":meta.get("DataStocks",meta.get("StocksWithData",0)),"Liquid":meta.get("LiquidStocks",0),"AI":meta.get("AI",meta.get("AICandidates",0)),"Selected":len(predictions)}
    report=morning_report(prediction_date,cutoff,predictions,jump_watchlist,intraday,accuracy=model_report_metrics(),scan=scan,portfolio=_portfolio_payload(),regime=meta.get("Regime","-"),market_snapshot=meta.get("MarketSnapshot",{}),ipo=ipo);sent=send_telegram(report)
    if sent:mark_morning_report_sent(prediction_date)
    return sent

def run():
    prediction_date=today_ist()
    if not is_weekday():print("Weekend. Morning prediction skipped.");return
    existing_meta=_prediction_metadata(prediction_date);current_prediction=prediction_exists(prediction_date);existing_current_version=(existing_meta.get("Stage")==STAGE_NAME and existing_meta.get("ModelVersion")==MODEL_VERSION)
    if current_prediction and existing_current_version:
        if morning_report_sent(prediction_date):print(f"Morning prediction and report already completed for {prediction_date}.")
        else:print(f"Current {STAGE_NAME} prediction exists for {prediction_date}; sending pending report.");_send_existing_report(prediction_date)
        return
    universe=load_universe();scan_count=len(universe);raw_data=download_many(universe,HISTORY_PERIOD,workers=8);data_map=filter_liquid_universe(raw_data)
    if len(data_map)<20:raise RuntimeError("Too few liquid stocks.")
    nifty_fallback=get_completed_session_date("morning",prediction_date);cutoff_date=get_data_cutoff_date(data_map,prediction_date,fallback=nifty_fallback)
    if cutoff_date is None:raise RuntimeError("Unable to determine completed market cutoff.")
    regime=get_market_regime(cutoff_date)["name"];variant=load_model_state().get("active_variant","A");snapshot=get_market_snapshot(data_map,cutoff_date);scored=[]
    for symbol,df in data_map.items():
        try:scored.append((symbol,technical_score(df[df.index<=pd.Timestamp(cutoff_date)])))
        except Exception:pass
    scored.sort(key=lambda x:x[1],reverse=True);candidate_symbols=[x[0] for x in scored[:PRESCREEN_N]];candidate_rows=[]
    for symbol in candidate_symbols:
        try:bundle=train_stock_bundle(data_map[symbol],symbol,cutoff_date,variant,train_horizons=False);result=predict_stock(data_map[symbol],bundle,cutoff_date);candidate_rows.append({"Symbol":symbol,**result,"ModelVariant":variant,"ModelVersion":MODEL_VERSION,"DataCutoff":str(cutoff_date)})
        except Exception as exc:print(f"{symbol}: prediction failed: {exc}")
    if not candidate_rows:raise RuntimeError("Unable to generate predictions.")
    candidates=add_stage4_context(pd.DataFrame(candidate_rows),data_map,regime);candidates=candidates[candidates["PriceBucket"]!="OUT"].copy()
    if candidates.empty:raise RuntimeError("No candidates inside configured price buckets.")
    candidates=score_candidates(candidates,regime);bucket_pool=_bucket_candidates(candidates,MAX_PER_PRICE_BUCKET).reset_index(drop=True);bucket_pool=_attach_horizons(bucket_pool,data_map,cutoff_date);bucket_pool=add_prediction_uncertainty(bucket_pool,data_map,{});bucket_pool=score_candidates(bucket_pool,regime);bucket_pool=add_market_risk(bucket_pool,regime);bucket_pool=_attach_benchmarks_and_risk(bucket_pool,data_map,cutoff_date)
    selected=select_top_stocks(bucket_pool,top_n=25,regime=regime,min_score=65.0,min_confidence=60.0,min_trade_confidence=60.0,max_per_bucket=FINAL_BEST_PER_BUCKET,bucket_only=False);selected=apply_final_intelligence(selected,regime=regime,breadth=float(snapshot.get("Breadth",{}).get("Score",50)),news=50);selected["PredictionDate"]=str(prediction_date);selected=_attach_current_ohlcv(selected,data_map,cutoff_date)
    metadata={"Stage":STAGE_NAME,"PredictionDate":str(prediction_date),"DataCutoff":str(cutoff_date),"ModelVariant":variant,"ModelVersion":MODEL_VERSION,"Regime":regime,"MarketSnapshot":snapshot,"PriceBuckets":[">1000","500-999","100-499","50-99","10-49"],"BestPerPriceBucket":FINAL_BEST_PER_BUCKET,"MaxSelectedStocks":25,"GlobalTopNCap":False,"MultiHorizons":[1,3,5,7,20],"FinalIntelligence":True,"TargetHitLevels":[1,2,3,5],"VolatilityBuckets":["LOW","MEDIUM","HIGH"],"CrossSectionalRanking":True,"SectorRelativeStrength":True,"AdaptiveThresholds":True,"Abstention":True,"Manifest":final_stage_manifest(),"SelectedStocks":selected["Symbol"].tolist(),"StocksScanned":scan_count,"DataStocks":len(raw_data),"AI":len(candidate_symbols),"LiquidStocks":len(data_map)}
    save_predictions(selected,prediction_date,metadata);save_decisions(selected,prediction_date);update_learning_state(FINAL_LEARNING_STATE_FILE,{"date":str(prediction_date),"regime":regime,"selected":selected[[c for c in ["Symbol","PriceBucket","VolatilityBucket","VolatilityPct","CrossSectionPercentile","SectorRelative20D","BenchmarkEdgePct","RiskAdjustedReturn","FinalDecisionScore","Action","FinalRisk","CalibratedConfidence","PredictionUncertaintyPct","TargetHitProb_3_0Pct","DownsideHitProb_2Pct"] if c in selected.columns]].to_dict("records")})
    jump_data={s:data_map[s] for s in candidate_symbols[:JUMP_CANDIDATE_N] if s in data_map};jump_watchlist=generate_jump_watchlist(jump_data,cutoff_date,variant)
    if not jump_watchlist.empty:save_jump_predictions(jump_watchlist,prediction_date)
    intraday=generate_intraday_watchlist(list(data_map.keys()),cutoff_date=cutoff_date)
    if not intraday.empty:save_intraday_predictions(intraday,prediction_date)
    try:ipo=get_ipo_report();ipo.to_csv(IPO_METRICS_FILE,index=False) if not ipo.empty else None
    except Exception as exc:print(f"IPO intelligence skipped: {exc}");ipo=pd.DataFrame()
    accuracy=model_report_metrics();scan={"Universe":len(universe),"Data":len(raw_data),"Liquid":len(data_map),"AI":len(candidate_symbols),"Selected":len(selected)};report=morning_report(prediction_date,cutoff_date,selected,jump_watchlist,intraday,market_snapshot=snapshot,regime=regime,ipo=ipo,accuracy=accuracy,scan=scan,portfolio=_portfolio_payload());sent=send_telegram(report)
    if sent:mark_morning_report_sent(prediction_date)
    print(report)

if __name__=="__main__":run()
