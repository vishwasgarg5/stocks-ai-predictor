"""Morning job: predict today's session using data strictly before today's date."""
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

from config import NIFTY50, PREDICTIONS_CSV
from src.market_data import update_universe, update_nifty
from src.features import add_features
from src.fundamentals import get_fundamentals
from src.ranking import rank_stocks
from src.retraining import load_champion, rolling_retrain
from src.prediction import predict_next
from src.ledger import read_ledger, ledger_text, save_prediction
from src.telegram_report import send_morning_predictions

IST = ZoneInfo("Asia/Kolkata")


def run():
    today = datetime.now(IST).date()
    target = str(today)

    raw_all = update_universe(NIFTY50)
    nifty_all = update_nifty()
    if len(raw_all) < 10 or nifty_all.empty:
        raise RuntimeError(f"Insufficient data: {len(raw_all)} stocks")

    # Hard look-ahead guard: Yahoo may expose today's row before the NSE open.
    # No row dated today or later is allowed into ranking, features, or training.
    raw = {}
    for symbol, df in raw_all.items():
        clean = df.loc[pd.to_datetime(df.index).date < today].copy()
        if len(clean) >= 20:
            raw[symbol] = clean
    nifty = nifty_all.loc[pd.to_datetime(nifty_all.index).date < today].copy()
    if len(raw) < 10 or nifty.empty:
        raise RuntimeError(f"Insufficient prior-session data: {len(raw)} stocks")

    prior_session = max(df.index.max().date() for df in raw.values())
    print(f"Today (IST): {today}")
    print(f"Prediction target: {target}")
    print(f"Training/data cutoff: strictly before {today}; latest usable session: {prior_session}")

    ledger = read_ledger(PREDICTIONS_CSV.read_text(encoding="utf-8") if PREDICTIONS_CSV.exists() else "")
    existing = set(ledger.loc[ledger["target_date"].astype(str) == target, "symbol"].astype(str)) if not ledger.empty else set()
    if len(existing) >= 5:
        print(f"Prediction already exists for {target}; no new prediction generated.")
        return

    features = {s: add_features(df, nifty) for s, df in raw.items()}
    fundamentals = {s: get_fundamentals(s) for s in raw}
    top5 = rank_stocks(features, fundamentals).sort_values(["score", "symbol"], ascending=[False, True]).reset_index(drop=True).head(5)

    models = load_champion()
    if models is None:
        models = rolling_retrain(features)

    telegram_rows = []
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        symbol = str(row["symbol"])
        if symbol in existing:
            continue
        p = predict_next(symbol, raw[symbol], features[symbol], models)
        ledger = save_prediction(ledger, p, rank, float(row["score"]), target_date=target)
        telegram_rows.append({"symbol": symbol, "score": float(row["score"]), "open": p["pred_open"], "high": p["pred_high"], "low": p["pred_low"], "close": p["pred_close"]})

    PREDICTIONS_CSV.write_text(ledger_text(ledger), encoding="utf-8")
    print(f"Prediction date: {target}")
    print(top5[["symbol", "score"]].to_string(index=False))
    if telegram_rows:
        send_morning_predictions(target, telegram_rows)


if __name__ == "__main__":
    run()
