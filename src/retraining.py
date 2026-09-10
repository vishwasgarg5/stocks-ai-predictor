import numpy as np
import pandas as pd
from .config import MODEL_STATE_FILE,MIN_RELATIVE_IMPROVEMENT,DAILY_METRICS_FILE,LIVE_ROLLBACK_MAPE,LIVE_ROLLBACK_DEGRADATION,MIN_PROMOTION_SAMPLES,PROMOTION_BOOTSTRAP_ITERATIONS,PROMOTION_P_VALUE
from .ledger import rebuild_stock_reliability
from .prediction import train_stock_bundle
from .utils import read_json,write_json

def load_model_state():
    default={"active_variant":"A","previous_variant":None,"last_champion_error":None,"last_challenger_error":None,"last_improvement":None,"last_decision":"INITIAL","baseline_live_mape":None,"promotion_p_value":None,"models_trained":False,"state_version":2,"promotion_count":0,"last_transition":None}
    state=read_json(MODEL_STATE_FILE,default) or default
    for k,v in default.items():state.setdefault(k,v)
    return state

def save_model_state(state):state["state_version"]=2;write_json(MODEL_STATE_FILE,state)
def _live_mape():
    try:
        df=pd.read_csv(DAILY_METRICS_FILE)
        if df.empty or "CloseMAPE" not in df:return None
        x=pd.to_numeric(df["CloseMAPE"],errors="coerce").dropna();return float(x.iloc[-1]) if not x.empty else None
    except Exception:return None

def _paired_p_value(champion,challenger,iterations=PROMOTION_BOOTSTRAP_ITERATIONS):
    a=np.asarray(champion,dtype=float);b=np.asarray(challenger,dtype=float);n=min(len(a),len(b))
    if n<MIN_PROMOTION_SAMPLES:return None
    d=(a[:n]-b[:n]);d=d[np.isfinite(d)];n=len(d)
    if n<MIN_PROMOTION_SAMPLES:return None
    observed=float(np.mean(d));rng=np.random.default_rng(42);signs=rng.choice(np.array([-1.,1.]),size=(int(iterations),n));null=(signs*d).mean(axis=1);return float((np.sum(null>=observed)+1)/(len(null)+1)) if observed>0 else 1.0

def compare_variants(data_map,symbols,cutoff_date):
    state=load_model_state();active=str(state.get("active_variant","A"));challenger="B" if active=="A" else "A";ce=[];xe=[];processed=[]
    for symbol in symbols:
        df=data_map.get(symbol)
        if df is None or df.empty:continue
        try:
            champion=train_stock_bundle(df,symbol,cutoff_date,active);challenger_bundle=train_stock_bundle(df,symbol,cutoff_date,challenger);ce.append(float(champion["validation_error"]));xe.append(float(challenger_bundle["validation_error"]));processed.append(symbol)
        except Exception as exc:print(f"{symbol}: retraining failed: {exc}")
    if not ce:
        state.update({"last_decision":"NO DATA","last_transition":None,"models_trained":False});save_model_state(state);return {"ModelsTrained":False,"ChallengerPromoted":False,"Retrained":False,"Decision":"NO DATA","Champion":active,"Challenger":challenger}
    ce_mean=float(np.mean(ce));xe_mean=float(np.mean(xe));improvement=(ce_mean-xe_mean)/max(abs(ce_mean),1e-8);live=_live_mape();p=_paired_p_value(ce,xe)
    gates={"error_better":xe_mean<ce_mean,"relative_improvement":improvement>=MIN_RELATIVE_IMPROVEMENT,"sample_count":len(processed)>=MIN_PROMOTION_SAMPLES,"paired_significance":p is not None and p<=PROMOTION_P_VALUE,"live_guard":live is None or live<=LIVE_ROLLBACK_MAPE}
    switched=all(gates.values());new_active=challenger if switched else active;decision="PROMOTED" if switched else "REJECTED / CHAMPION KEPT"
    transition=None
    if switched:
        transition={"from":active,"to":challenger,"cutoff":str(pd.Timestamp(cutoff_date).date()),"improvement_pct":improvement*100,"p_value":p,"samples":len(processed)};state["previous_variant"]=active;state["promotion_count"]=int(state.get("promotion_count",0))+1;state["baseline_live_mape"]=live
    state.update({"active_variant":new_active,"last_champion_error":ce_mean,"last_challenger_error":xe_mean,"last_improvement":improvement*100,"last_decision":decision,"processed_stocks":processed,"cutoff_date":str(pd.Timestamp(cutoff_date).date()),"last_live_mape":live,"promotion_p_value":p,"promotion_gates":gates,"last_transition":transition,"models_trained":True})
    save_model_state(state);rebuild_stock_reliability()
    return {"ModelsTrained":True,"ChallengerPromoted":switched,"Retrained":True,"Decision":decision,"Champion":active,"Challenger":challenger,"NewChampion":new_active,"ChampionError":ce_mean,"ChallengerError":xe_mean,"Improvement":improvement*100,"StocksProcessed":len(processed),"LiveMAPE":live,"PromotionPValue":p,"PromotionGates":gates}

def maybe_rollback_live():
    state=load_model_state();previous=state.get("previous_variant");active=state.get("active_variant","A");live=_live_mape();baseline=state.get("baseline_live_mape")
    if not previous or live is None or baseline is None:return {"RolledBack":False,"LiveMAPE":live,"Decision":"NO ROLLBACK DATA"}
    degraded=live>LIVE_ROLLBACK_MAPE and live>baseline*(1+LIVE_ROLLBACK_DEGRADATION)
    if not degraded:return {"RolledBack":False,"LiveMAPE":live,"BaselineMAPE":baseline,"Decision":"LIVE MODEL STABLE"}
    state.update({"active_variant":previous,"previous_variant":active,"last_decision":"AUTOMATIC LIVE ROLLBACK","rollback_reason":f"Live Close MAPE {live:.3f}% vs baseline {baseline:.3f}%","baseline_live_mape":live});save_model_state(state)
    return {"RolledBack":True,"From":active,"To":previous,"LiveMAPE":live,"BaselineMAPE":baseline,"Decision":"AUTOMATIC LIVE ROLLBACK"}
