import numpy as np
import pandas as pd
from .config import MODEL_STATE_FILE,MIN_RELATIVE_IMPROVEMENT,DAILY_METRICS_FILE,LIVE_ROLLBACK_MAPE,LIVE_ROLLBACK_DEGRADATION,MIN_PROMOTION_SAMPLES,PROMOTION_BOOTSTRAP_ITERATIONS,PROMOTION_P_VALUE
from .ledger import rebuild_stock_reliability
from .prediction import train_stock_bundle
from .utils import read_json,write_json

def load_model_state():
    default={"active_variant":"A","previous_variant":None,"last_champion_error":None,"last_challenger_error":None,"last_improvement":None,"last_decision":"INITIAL","baseline_live_mape":None,"promotion_p_value":None}
    state=read_json(MODEL_STATE_FILE,default) or default
    for k,v in default.items():state.setdefault(k,v)
    return state

def save_model_state(state):write_json(MODEL_STATE_FILE,state)
def _live_mape():
    try:
        df=pd.read_csv(DAILY_METRICS_FILE)
        if df.empty or "CloseMAPE" not in df:return None
        x=pd.to_numeric(df["CloseMAPE"],errors="coerce").dropna();return float(x.iloc[-1]) if not x.empty else None
    except Exception:return None

def _paired_p_value(champion,challenger,iterations=PROMOTION_BOOTSTRAP_ITERATIONS):
    a=np.asarray(champion,float);b=np.asarray(challenger,float);n=min(len(a),len(b))
    if n<MIN_PROMOTION_SAMPLES:return None
    d=a[:n]-b[:n];obs=float(np.mean(d));rng=np.random.default_rng(42);means=[]
    for _ in range(iterations):means.append(float(np.mean(rng.choice(d,size=n,replace=True))))
    return float(np.mean(np.asarray(means)<=0)) if obs>0 else 1.0

def compare_variants(data_map,symbols,cutoff_date):
    state=load_model_state();active=state.get("active_variant","A");challenger="B" if active=="A" else "A";champion_errors=[];challenger_errors=[];processed=[]
    for symbol in symbols:
        df=data_map.get(symbol)
        if df is None or df.empty:continue
        try:
            champion=train_stock_bundle(df,symbol,cutoff_date,active);challenger_bundle=train_stock_bundle(df,symbol,cutoff_date,challenger);champion_errors.append(champion["validation_error"]);challenger_errors.append(challenger_bundle["validation_error"]);processed.append(symbol)
        except Exception as exc:print(f"{symbol}: retraining failed: {exc}")
    if not champion_errors:return {"Retrained":False,"Decision":"NO DATA","Champion":active,"Challenger":challenger}
    ce=float(np.mean(champion_errors));xe=float(np.mean(challenger_errors));improvement=(ce-xe)/max(ce,1e-8);live=_live_mape();baseline=state.get("baseline_live_mape");p=_paired_p_value(champion_errors,challenger_errors)
    switched=xe<ce and improvement>=MIN_RELATIVE_IMPROVEMENT and len(processed)>=MIN_PROMOTION_SAMPLES and (p is not None and p<=PROMOTION_P_VALUE)
    if switched and live is not None and live>LIVE_ROLLBACK_MAPE:switched=False
    new_active=challenger if switched else active;decision="CHALLENGER PROMOTED" if switched else "CHAMPION KEPT"
    if switched:state["previous_variant"]=active;state["baseline_live_mape"]=live
    state.update({"active_variant":new_active,"last_champion_error":ce,"last_challenger_error":xe,"last_improvement":improvement*100,"last_decision":decision,"processed_stocks":processed,"cutoff_date":str(pd.Timestamp(cutoff_date).date()),"last_live_mape":live,"promotion_p_value":p})
    save_model_state(state);rebuild_stock_reliability()
    return {"Retrained":switched,"Decision":decision,"Champion":active,"Challenger":challenger,"NewChampion":new_active,"ChampionError":ce,"ChallengerError":xe,"Improvement":improvement*100,"StocksProcessed":len(processed),"LiveMAPE":live,"PromotionPValue":p,"PromotionMinimumSamples":MIN_PROMOTION_SAMPLES}

def maybe_rollback_live():
    state=load_model_state();previous=state.get("previous_variant");active=state.get("active_variant","A");live=_live_mape();baseline=state.get("baseline_live_mape")
    if not previous or live is None or baseline is None:return {"RolledBack":False,"LiveMAPE":live,"Decision":"NO ROLLBACK DATA"}
    degraded=live>LIVE_ROLLBACK_MAPE and live>baseline*(1+LIVE_ROLLBACK_DEGRADATION)
    if degraded:
        state["active_variant"]=previous;state["previous_variant"]=active;state["last_decision"]="AUTOMATIC LIVE ROLLBACK";state["rollback_reason"]=f"Live Close MAPE {live:.3f}% vs baseline {baseline:.3f}%";state["baseline_live_mape"]=live;save_model_state(state)
        return {"RolledBack":True,"From":active,"To":previous,"LiveMAPE":live,"BaselineMAPE":baseline,"Decision":"AUTOMATIC LIVE ROLLBACK"}
    return {"RolledBack":False,"LiveMAPE":live,"BaselineMAPE":baseline,"Decision":"LIVE MODEL STABLE"}
