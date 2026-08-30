"""Stage 1.5 morning prediction pipeline."""
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

from config import PREDICTIONS_CSV
from src.market_data import update_universe, update_nifty, get_nifty150_universe
from src.features import add_features
from src.fundamentals import get_fundamentals
from src.ranking import rank_stocks
from src.retraining import load_champion, rolling_retrain
from src.stage15 import apply_stage15_context
from src.prediction import predict_next
from src.ledger import read_ledger, ledger_text, save_prediction
from src.telegram_report import send_morning_predictions

IST = ZoneInfo("Asia/Kolkata")
UNIVERSE_VERSION = "NIFTY150-STAGE1.5"


def run():
    today = datetime.now(IST).date()
    target = str(today)
    universe = get_nifty150_universe()
    raw_all = update_universe(universe)
    nifty_all = update_nifty()
    if len(raw_all) < 50 or nifty_all.empty:
        raise RuntimeError(f"Insufficient data: {len(raw_all)} usable stocks")

    raw = {}
    for symbol, df in raw_all.items():
        clean = df.loc[pd.to_datetime(df.index).date < today].copy()
        if len(clean) >= 20:
            raw[symbol] = clean
    nifty = nifty_all.loc[pd.to_datetime(nifty_all.index).date < today].copy()
    if len(raw) < 50 or nifty.empty:
        raise RuntimeError(f"Insufficient prior-session data: {len(raw)} stocks")

    prior_session = max(df.index.max().date() for df in raw.values())
    print(f"Universe: Nifty 150 | usable OHLCV: {len(raw)}")
    print(f"Prediction target: {target} | cutoff: {prior_session}")

    ledger = read_ledger(PREDICTIONS_CSV.read_text(encoding="utf-8") if PREDICTIONS_CSV.exists() else "")
    existing = ledger[ledger.target_date.astype(str) == target].copy() if not ledger.empty else pd.DataFrame()
    current = existing[existing.universe_version.astype(str) == UNIVERSE_VERSION].copy() if not existing.empty else pd.DataFrame()
    if len(current) >= 5:
        rows = []
        for _, r in current.sort_values(["rank","symbol"]).head(5).iterrows():
            rows.append({"symbol":str(r.symbol),"score":float(r.score),"open":float(r.pred_open),"high":float(r.pred_high),"low":float(r.pred_low),"close":float(r.pred_close),"direction":str(r.predicted_direction),"confidence":float(r.confidence),"regime":str(r.regime)})
        send_morning_predictions(target, rows)
        return
    if not existing.empty:
        ledger = ledger[ledger.target_date.astype(str) != target].copy()

    features = {s: add_features(df, nifty) for s, df in raw.items()}
    features = apply_stage15_context(features, raw)
    fundamentals = {s: get_fundamentals(s) for s in raw}
    top5 = rank_stocks(features, fundamentals).sort_values(["score","symbol"], ascending=[False,True]).reset_index(drop=True).head(5)
    models = load_champion() or rolling_retrain(features)

    telegram_rows = []
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        symbol = str(row["symbol"])
        p = predict_next(symbol, raw[symbol], features[symbol], models)
        ledger = save_prediction(ledger, p, rank, float(row["score"]), target_date=target, universe_version=UNIVERSE_VERSION)
        telegram_rows.append({"symbol":symbol,"score":float(row["score"]),"open":p["pred_open"],"high":p["pred_high"],"low":p["pred_low"],"close":p["pred_close"],"direction":p["predicted_direction"],"confidence":p["confidence"],"regime":p["regime"]})

    PREDICTIONS_CSV.write_text(ledger_text(ledger), encoding="utf-8")
    send_morning_predictions(target, telegram_rows)


if __name__ == "__main__":
    run()
