import io
from datetime import datetime, timezone
import pandas as pd

COLUMNS = ["created_at","target_date","universe_version","symbol","rank","score","base_close","pred_open","pred_high","pred_low","pred_close","actual_open","actual_high","actual_low","actual_close","open_error","high_error","low_error","close_error","direction_correct"]

def read_ledger(text: str = "") -> pd.DataFrame:
    if not text: return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(io.StringIO(text))
    for c in COLUMNS:
        if c not in df.columns: df[c] = pd.NA
    return df.reindex(columns=COLUMNS)

def ledger_text(df): return df.reindex(columns=COLUMNS).to_csv(index=False)

def save_prediction(ledger, p, rank, score, target_date=None, universe_version="NIFTY150"):
    target_date = target_date or p.get("target_date")
    row = {c: None for c in COLUMNS}
    row.update({"created_at":datetime.now(timezone.utc).isoformat(),"target_date":target_date,"universe_version":universe_version,"symbol":p["symbol"],"rank":rank,"score":score,"base_close":p["base_close"],"pred_open":p["pred_open"],"pred_high":p["pred_high"],"pred_low":p["pred_low"],"pred_close":p["pred_close"]})
    mask=(ledger.symbol.astype(str)==str(p["symbol"]))&(ledger.target_date.astype(str)==str(target_date))&(ledger.universe_version.astype(str)==str(universe_version))
    if mask.any(): return ledger
    return pd.concat([ledger,pd.DataFrame([row])],ignore_index=True)

def evaluate_pending(ledger, actuals):
    updated=0; ledger=ledger.copy()
    for idx in ledger[ledger.actual_close.isna()].index:
        symbol=str(ledger.at[idx,"symbol"]); target=pd.to_datetime(ledger.at[idx,"target_date"],errors="coerce")
        if pd.isna(target): continue
        a=actuals.get(symbol,{}).get(target.strftime("%Y-%m-%d"))
        if a is None: continue
        base=float(ledger.at[idx,"base_close"]); pc=float(ledger.at[idx,"pred_close"])
        ledger.loc[idx,["actual_open","actual_high","actual_low","actual_close"]]=[a["open"],a["high"],a["low"],a["close"]]
        ledger.loc[idx,["open_error","high_error","low_error","close_error"]]=[a["open"]-ledger.at[idx,"pred_open"],a["high"]-ledger.at[idx,"pred_high"],a["low"]-ledger.at[idx,"pred_low"],a["close"]-pc]
        ledger.at[idx,"direction_correct"]=int((pc>base)==(a["close"]>base)); updated+=1
    return ledger,updated

def performance_report(ledger):
    done=ledger.dropna(subset=["actual_close"]).copy()
    if done.empty:return pd.DataFrame()
    eps=1e-12; cols=["open","high","low","close"]
    mape={f"{c}_mape":(done[f"{c}_error"].abs()/done[f"actual_{c}"].abs().clip(lower=eps)).mean() for c in cols}
    return pd.DataFrame([{"predictions":len(done),"open_mae":done.open_error.abs().mean(),"high_mae":done.high_error.abs().mean(),"low_mae":done.low_error.abs().mean(),"close_mae":done.close_error.abs().mean(),**mape,"ohlc_mape":sum(mape.values())/4,"direction_accuracy":done.direction_correct.mean()}])
