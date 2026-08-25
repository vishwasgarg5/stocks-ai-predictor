"""Evening job: fetch today's completed session, evaluate predictions, retrain."""
from config import NIFTY50, PREDICTIONS_CSV
from src.market_data import update_universe, update_nifty
from src.ledger import read_ledger, ledger_text, evaluate_pending, performance_report
from src.features import add_features
from src.retraining import rolling_retrain


def run():
    raw = update_universe(NIFTY50)
    nifty = update_nifty()
    if len(raw) < 10 or nifty.empty:
        raise RuntimeError(f"Insufficient data: {len(raw)} stocks")
    ledger = read_ledger(PREDICTIONS_CSV.read_text(encoding="utf-8") if PREDICTIONS_CSV.exists() else "")
    actuals = {}
    for symbol, df in raw.items():
        if not df.empty:
            r = df.iloc[-1]
            actuals[symbol] = {"date": df.index[-1].strftime("%Y-%m-%d"), "open": float(r["Open"]), "high": float(r["High"]), "low": float(r["Low"]), "close": float(r["Close"])}
    ledger, evaluated = evaluate_pending(ledger, actuals)
    if evaluated:
        features = {s: add_features(df, nifty) for s, df in raw.items()}
        rolling_retrain(features)
    PREDICTIONS_CSV.write_text(ledger_text(ledger), encoding="utf-8")
    report = performance_report(ledger)
    if not report.empty:
        report.to_csv("reports/performance.csv", index=False)
    print(f"Market session: {max(df.index.max() for df in raw.values()).date()}")
    print(f"Predictions evaluated: {evaluated}")
    if not report.empty:
        print(report.to_string(index=False))

if __name__ == "__main__":
    run()
