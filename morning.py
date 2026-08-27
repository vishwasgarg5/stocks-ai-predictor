"""Morning job: predict today's session using data strictly before today's date."""
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

from config import PREDICTIONS_CSV
from src.market_data import update_universe, update_nifty, get_nifty150_universe
from src.features import add_features
from src.fundamentals import get_fundamentals
from src.ranking import rank_stocks
from src.retraining import load_champion, rolling_retrain
from src.prediction import predict_next
from src.ledger import read_ledger, ledger_text, save_prediction
from src.telegram_report import send_morning_predictions

IST = ZoneInfo("Asia/Kolkata")
UNIVERSE_VERSION = "NIFTY150"


def run():
    today = datetime.now(IST).date()
    target = str(today)
    universe = get_nifty150_universe()
    raw_all = update_universe(universe)
    nifty_all = update_nifty()
    print(f"Universe: Nifty 150 | constituents configured: {len(universe)} | usable OHLCV: {len(raw_all)}")
    if len(raw_all) < 50 or nifty_all.empty:
        raise RuntimeError(f"Insufficient data: {len(raw_all)} usable stocks")

    raw = {}
    for symbol, df in raw_all.items():
        clean = df.loc[pd.to_datetime(df.index).date < today].copy()
        if len(clean) >= 20: raw[symbol] = clean
    nifty = nifty_all.loc[pd.to_datetime(nifty_all.index).date < today].copy()
    if len(raw) < 50 or nifty.empty:
        raise RuntimeError(f"Insufficient prior-session data: {len(raw)} stocks")

    prior_session = max(df.index.max().date() for df in raw.values())
    print(f"Today (IST): {today}")
    print(f"Prediction target: {target}")
    print(f"Training/data cutoff: strictly before {today}; latest usable session: {prior_session}")

    ledger = read_ledger(PREDICTIONS_CSV.read_text(encoding="utf-8") if PREDICTIONS_CSV.exists() else "")
    existing_rows = ledger[ledger["target_date"].astype(str) == target].copy() if not ledger.empty else pd.DataFrame()

    # A prediction is reusable only when it was generated from the current universe.
    # This prevents an old NIFTY-50 prediction (e.g. Aug 27) from surviving the
    # migration to NIFTY-150. The old target rows are removed and regenerated once.
    current_rows = existing_rows[existing_rows["universe_version"].astype(str) == UNIVERSE_VERSION].copy() if not existing_rows.empty else pd.DataFrame()
    if len(current_rows) >= 5:
        current_rows = current_rows.sort_values(["rank", "symbol"]).head(5)
        telegram_rows = [{"symbol":str(r["symbol"]),"score":float(r["score"]),"open":float(r["pred_open"]),"high":float(r["pred_high"]),"low":float(r["pred_low"]),"close":float(r["pred_close"])} for _,r in current_rows.iterrows()]
        print(f"Prediction already exists for {target} under {UNIVERSE_VERSION}; reusing saved Top 5 and resending Telegram.")
        send_morning_predictions(target, telegram_rows)
        return

    if not existing_rows.empty:
        print(f"Existing prediction for {target} was created under an older/different universe; replacing it with a fresh {UNIVERSE_VERSION} prediction.")
        ledger = ledger[ledger["target_date"].astype(str) != target].copy()

    features = {s:add_features(df,nifty) for s,df in raw.items()}
    fundamentals = {s:get_fundamentals(s) for s in raw}
    top5 = rank_stocks(features,fundamentals).sort_values(["score","symbol"],ascending=[False,True]).reset_index(drop=True).head(5)
    models = load_champion()
    if models is None: models = rolling_retrain(features)

    telegram_rows=[]
    for rank,(_,row) in enumerate(top5.iterrows(),1):
        symbol=str(row["symbol"])
        p=predict_next(symbol,raw[symbol],features[symbol],models)
        ledger=save_prediction(ledger,p,rank,float(row["score"]),target_date=target,universe_version=UNIVERSE_VERSION)
        telegram_rows.append({"symbol":symbol,"score":float(row["score"]),"open":p["pred_open"],"high":p["pred_high"],"low":p["pred_low"],"close":p["pred_close"]})

    PREDICTIONS_CSV.write_text(ledger_text(ledger),encoding="utf-8")
    print(f"Prediction date: {target} | Universe: {UNIVERSE_VERSION}")
    print(top5[["symbol","score"]].to_string(index=False))
    if telegram_rows: send_morning_predictions(target,telegram_rows)


if __name__ == "__main__": run()
