"""GitHub-only NIFTY 50 incremental data + prediction workflow."""
from pathlib import Path
import pandas as pd
from config import NIFTY50, OHLCV_DIR, NIFTY_CSV, PREDICTIONS_CSV
from src.market_data import update_universe, update_nifty
from src.features import add_features
from src.fundamentals import get_fundamentals
from src.ranking import rank_stocks
from src.prediction import predict_next
from src.retraining import rolling_retrain
from src.ledger import read_ledger, ledger_text, save_prediction, evaluate_pending, performance_report


def main():
    print("\n=== NIFTY 50 AI STOCK PREDICTOR v2.0 ===")
    print("Persistence: GitHub files | Rolling window: latest 3 months")

    nifty = update_nifty()
    raw = update_universe(NIFTY50)
    if nifty.empty or len(raw) < 10:
        raise RuntimeError(f"Insufficient market data: NIFTY={len(nifty)}, stocks={len(raw)}")

    # Permanent prediction ledger lives in GitHub, not SQLite.
    if PREDICTIONS_CSV.exists():
        ledger = read_ledger(PREDICTIONS_CSV.read_text(encoding="utf-8"))
    else:
        ledger = read_ledger()

    # Evaluate pending predictions only when a newer completed session exists.
    actuals = {}
    for symbol, df in raw.items():
        if not df.empty:
            r = df.iloc[-1]
            actuals[symbol] = {"date": str(df.index[-1].date()), "open": float(r["Open"]), "high": float(r["High"]), "low": float(r["Low"]), "close": float(r["Close"])}
    ledger, evaluated = evaluate_pending(ledger, actuals)
    print(f"Evaluated predictions: {evaluated}")

    features = {s: add_features(df, nifty) for s, df in raw.items()}
    fundamentals = {s: get_fundamentals(s) for s in raw}
    models = rolling_retrain(features)
    top5 = rank_stocks(features, fundamentals)
    if top5.empty:
        raise RuntimeError("No stocks passed the feature quality gate")

    latest_date = max(str(df.index[-1].date()) for df in raw.values() if not df.empty)
    print("\nTOP 5")
    print(top5.to_string(index=False))
    print("\nNEXT-DAY PREDICTIONS")

    # Prevent duplicate predictions when Actions is manually run twice for the same session.
    unresolved = set(ledger.loc[ledger["actual_close"].isna(), "symbol"].astype(str))
    for _, r in top5.iterrows():
        symbol = r.symbol
        if symbol in unresolved:
            print(f"{symbol:12s} existing pending prediction - skipped")
            continue
        p = predict_next(symbol, raw[symbol], features[symbol], models)
        ledger = save_prediction(ledger, p, int(r["rank"]), float(r["score"]), target_date=latest_date)
        print(f"{symbol:12s} Open={p['pred_open']:.2f} High={p['pred_high']:.2f} Low={p['pred_low']:.2f} Close={p['pred_close']:.2f}")

    PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_CSV.write_text(ledger_text(ledger), encoding="utf-8")
    report = performance_report(ledger)
    if not report.empty:
        Path("reports").mkdir(exist_ok=True)
        report.to_csv("reports/performance.csv", index=False)
        print("\nMODEL PERFORMANCE")
        print(report.to_string(index=False))

    print(f"\nComplete. Updated {len(raw)} stocks; evaluated {evaluated}; ledger rows {len(ledger)}.")


if __name__ == "__main__":
    main()
