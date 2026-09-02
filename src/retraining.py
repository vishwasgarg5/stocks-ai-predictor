import pandas as pd
from .config import MODEL_STATE_FILE,MIN_RELATIVE_IMPROVEMENT
from .ledger import rebuild_stock_reliability
from .prediction import train_stock_bundle
from .utils import read_json,write_json

def load_model_state():
    default={"active_variant":"A","last_champion_error":None,"last_challenger_error":None,"last_improvement":None,"last_decision":"INITIAL"}
    return read_json(MODEL_STATE_FILE,default) or default

def save_model_state(state): write_json(MODEL_STATE_FILE,state)

def compare_variants(data_map,symbols,cutoff_date):
    state=load_model_state(); active=state.get("active_variant","A"); challenger="B" if active=="A" else "A"; champion_errors=[]; challenger_errors=[]; processed=[]
    for symbol in symbols:
        df=data_map.get(symbol)
        if df is None or df.empty:continue
        try:
            champion=train_stock_bundle(df,symbol,cutoff_date,active); challenger_bundle=train_stock_bundle(df,symbol,cutoff_date,challenger)
            champion_errors.append(champion["validation_error"]); challenger_errors.append(challenger_bundle["validation_error"]); processed.append(symbol)
        except Exception as exc: print(f"{symbol}: retraining failed: {exc}")
    if not champion_errors:return {"Retrained":False,"Decision":"NO DATA","Champion":active,"Challenger":challenger}
    ce=sum(champion_errors)/len(champion_errors); xe=sum(challenger_errors)/len(challenger_errors); improvement=(ce-xe)/max(ce,1e-8)
    switched=xe<ce and improvement>=MIN_RELATIVE_IMPROVEMENT; new_active=challenger if switched else active; decision="CHALLENGER PROMOTED" if switched else "CHAMPION KEPT"
    save_model_state({"active_variant":new_active,"last_champion_error":ce,"last_challenger_error":xe,"last_improvement":improvement*100,"last_decision":decision,"processed_stocks":processed,"cutoff_date":str(pd.Timestamp(cutoff_date).date())})
    rebuild_stock_reliability()
    return {"Retrained":switched,"Decision":decision,"Champion":active,"Challenger":challenger,"NewChampion":new_active,"ChampionError":ce,"ChallengerError":xe,"Improvement":improvement*100,"StocksProcessed":len(processed)}
