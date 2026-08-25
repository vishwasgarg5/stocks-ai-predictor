"""GitHub-only, deterministic NIFTY 50 incremental prediction workflow."""
from src.market_data import update_universe, update_nifty
from src.features import add_features
from src.fundamentals import get_fundamentals
from src.ranking import rank_stocks
from src.retraining import rolling_retrain
from src.prediction import predict_next
from src.ledger import read_ledger, ledger_text, save_prediction, evaluate_pending, performance_report
from config import NIFTY50, PREDICTIONS_CSV


def run():
    print("=== NIFTY 50 AI PREDICTOR ===")
    print("Persistent store: GitHub CSV | Rolling window: 3 months")
    raw = update_universe(NIFTY50)
    nifty = update_nifty()
    if len(raw) < 10 or nifty.empty:
        raise RuntimeError(f"Insufficient market data: {len(raw)} stocks")

    old_text = PREDICTIONS_CSV.read_text(encoding="utf-8") if PREDICTIONS_CSV.exists() else ""
    ledger = read_ledger(old_text)

    actuals = {}
    for symbol, df in raw.items():
        if not df.empty:
            r = df.iloc[-1]
            actuals[symbol] = {"date": df.index[-1].strftime("%Y-%m-%d"), "open": float(r["Open"]), "high": float(r["High"]), "low": float(r["Low"]), "close": float(r["Close"])}
    ledger, evaluated = evaluate_pending(ledger, actuals)

    features = {s: add_features(df, nifty) for s, df in raw.items()}
    fundamentals = {s: get_fundamentals(s) for s in raw}
    top5 = rank_stocks(features, fundamentals)
    if top5.empty:
        raise RuntimeError("No stocks passed the quality gate")

    # Deterministic ranking: identical inputs always produce the same ordering.
    top5 = top5.sort_values(["score", "symbol"], ascending=[False, True]).reset_index(drop=True)
    top5["rank"] = range(1, len(top5) + 1)

    models = rolling_retrain(features)
    latest_market_date = max(df.index.max() for df in raw.values()).date()
    target_session = str(latest_market_date)
    for _, row in top5.head(5).iterrows():
        symbol = str(row["symbol"])
        p = predict_next(symbol, raw[symbol], features[symbol], models)
        ledger = save_prediction(ledger, p, int(row["rank"]), float(row["score"]), target_date=target_session)

    PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_CSV.write_text(ledger_text(ledger), encoding="utf-8")
    report = performance_report(ledger)
    if not report.empty:
        report.to_csv("reports/performance.csv", index=False)

    print(f"Stocks updated: {len(raw)} | Predictions evaluated: {evaluated}")
    print("\nTOP 5")
    print(top5.head(5).to_string(index=False))
    print(f"\nPrediction ledger: {PREDICTIONS_CSV}")
    if not report.empty:
        print("\nPERFORMANCE")
        print(report.to_string(index=False))


if __name__ == "__main__":
    run()
