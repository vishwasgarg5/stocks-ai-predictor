"""Morning job: predict today's session using only completed prior-session data."""
from datetime import date
from config import NIFTY50, PREDICTIONS_CSV
from src.market_data import read_stored, update_universe, update_nifty
from src.features import add_features
from src.fundamentals import get_fundamentals
from src.ranking import rank_stocks
from src.retraining import rolling_retrain
from src.prediction import predict_next
from src.ledger import read_ledger, ledger_text, save_prediction
from pathlib import Path


def next_weekday(d):
    from datetime import timedelta
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def run():
    raw = update_universe(NIFTY50)
    nifty = update_nifty()
    if len(raw) < 10 or nifty.empty:
        raise RuntimeError(f"Insufficient data: {len(raw)} stocks")
    ledger = read_ledger(PREDICTIONS_CSV.read_text(encoding="utf-8") if PREDICTIONS_CSV.exists() else "")
    latest = max(df.index.max().date() for df in raw.values())
    target = str(next_weekday(latest))
    # If the target already has predictions, this run is idempotent.
    existing = set(ledger.loc[ledger["target_date"].astype(str) == target, "symbol"].astype(str)) if not ledger.empty else set()
    if len(existing) >= 5:
        print(f"Prediction already exists for {target}; no new prediction generated.")
        return
    features = {s: add_features(df, nifty) for s, df in raw.items()}
    fundamentals = {s: get_fundamentals(s) for s in raw}
    top5 = rank_stocks(features, fundamentals).sort_values(["score", "symbol"], ascending=[False, True]).reset_index(drop=True).head(5)
    models = rolling_retrain(features)
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        symbol = str(row["symbol"])
        if symbol in existing:
            continue
        p = predict_next(symbol, raw[symbol], features[symbol], models)
        ledger = save_prediction(ledger, p, rank, float(row["score"]), target_date=target)
    PREDICTIONS_CSV.write_text(ledger_text(ledger), encoding="utf-8")
    print(f"Prediction date: {target}")
    print(top5[["symbol", "score"]].to_string(index=False))

if __name__ == "__main__":
    run()
