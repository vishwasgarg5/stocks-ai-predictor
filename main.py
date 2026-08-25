"""GitHub-only NIFTY 50 incremental data + prediction workflow."""
from src.market_data import update_universe, update_nifty, read_csv, csv_text
from src.features import add_features
from src.fundamentals import get_fundamentals
from src.ranking import rank_stocks
from src.prediction import predict_next
from src.retraining import rolling_retrain
from src.ledger import read_ledger, ledger_text, save_prediction, evaluate_pending, performance_report
from config import NIFTY50


def run(existing_files: dict[str, str] | None = None):
    existing_files = existing_files or {}
    existing = {s: read_csv(existing_files.get(f"data/ohlcv/{s}.csv", "")) for s in NIFTY50}
    nifty = update_nifty(read_csv(existing_files.get("data/nifty.csv", "")))
    raw = update_universe(existing)
    ledger = read_ledger(existing_files.get("data/predictions.csv", ""))

    actuals = {}
    for symbol, df in raw.items():
        if not df.empty:
            r = df.iloc[-1]
            actuals[symbol] = {"date": str(df.index[-1].date()), "open": float(r.Open), "high": float(r.High), "low": float(r.Low), "close": float(r.Close)}
    ledger, evaluated = evaluate_pending(ledger, actuals)

    features = {s: add_features(df, nifty) for s, df in raw.items()}
    fundamentals = {s: get_fundamentals(s) for s in raw}
    top5 = rank_stocks(features, fundamentals)
    if top5.empty:
        raise RuntimeError("No stocks passed the feature quality gate")

    models = rolling_retrain(features)
    existing_pred = set(ledger.loc[ledger["actual_close"].isna(), "symbol"].astype(str))
    latest_date = max(str(d.index[-1].date()) for d in raw.values() if not d.empty)
    for _, r in top5.iterrows():
        symbol = r.symbol
        # Never create duplicate predictions for the same unresolved trading session.
        if symbol in existing_pred:
            continue
        p = predict_next(symbol, raw[symbol], features[symbol], models)
        ledger = save_prediction(ledger, p, int(r["rank"]), float(r["score"]), target_date=latest_date)

    report = performance_report(ledger)
    files = {f"data/ohlcv/{s}.csv": csv_text(df) for s, df in raw.items()}
    files["data/nifty.csv"] = csv_text(nifty)
    files["data/predictions.csv"] = ledger_text(ledger)
    if not report.empty:
        files["reports/performance.csv"] = report.to_csv(index=False)
    print(f"Updated {len(raw)} stocks | evaluated {evaluated} | predictions {len(ledger)}")
    print(top5.to_string(index=False))
    return files


if __name__ == "__main__":
    print("Use the GitHub Actions workflow to run the GitHub-only pipeline and commit updated CSV files.")
