"""NIFTY 50 3-month rolling daily prediction workflow."""
from src.market_data import download_universe, download_nifty
from src.features import add_features
from src.fundamentals import get_fundamentals
from src.ranking import rank_stocks
from src.prediction import predict_next
from src.retraining import rolling_retrain
from src.ledger import init_db, save_prediction
from src.workflow import evaluate_latest_predictions
from src.evaluation import performance_report


def main():
    print("\n=== NIFTY 50 AI STOCK PREDICTOR v1.2 ===")
    print("Rolling window: latest 3 months | Daily OHLCV")
    init_db()

    # 1) Download the latest completed market session plus the rolling 3-month window.
    nifty = download_nifty()
    if nifty.empty:
        raise RuntimeError("Unable to download NIFTY index data")
    raw = download_universe()
    if len(raw) < 10:
        raise RuntimeError(f"Too few stocks downloaded: {len(raw)}")

    # 2) First settle yesterday's pending predictions using the newest completed session.
    evaluated = evaluate_latest_predictions(raw)
    print(f"Evaluated pending predictions: {evaluated}")

    # 3) Build features and retrain on the current 3-month rolling window.
    features = {s: add_features(df, nifty) for s, df in raw.items()}
    fundamentals = {s: get_fundamentals(s) for s in raw}
    models = rolling_retrain(features)

    # 4) Rank the full universe and keep only the Top 5.
    top5 = rank_stocks(features, fundamentals)
    if top5.empty:
        raise RuntimeError("No stocks passed the feature quality gate")

    print("\nTOP 5 STOCKS")
    print(top5.to_string(index=False))

    # 5) Predict the next session and write a durable ledger record.
    print("\nNEXT-DAY OHLC PREDICTIONS")
    for _, row in top5.iterrows():
        symbol = row.symbol
        p = predict_next(symbol, raw[symbol], features[symbol], models)
        save_prediction(p, int(row["rank"]), float(row["score"]))
        print(f"{symbol:12s} Open={p['pred_open']:.2f} High={p['pred_high']:.2f} Low={p['pred_low']:.2f} Close={p['pred_close']:.2f}")

    report = performance_report()
    if not report.empty:
        print("\nMODEL PERFORMANCE")
        print(report.to_string(index=False))
    print("\nDaily cycle complete: evaluate -> retrain -> rank -> predict -> ledger.")


if __name__ == "__main__":
    main()
