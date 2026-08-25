"""NIFTY 50 3-month rolling prediction pipeline."""
from src.market_data import download_universe, download_nifty
from src.features import add_features
from src.fundamentals import get_fundamentals
from src.ranking import rank_stocks
from src.prediction import train_models, predict_next
from src.ledger import init_db, save_prediction


def main():
    print("\n=== NIFTY 50 AI STOCK PREDICTOR v1 ===")
    print("Data window: latest 3 months | Frequency: daily")
    init_db()

    nifty = download_nifty()
    if nifty.empty:
        raise RuntimeError("Unable to download NIFTY index data")

    raw = download_universe()
    if len(raw) < 10:
        raise RuntimeError(f"Too few stocks downloaded: {len(raw)}")

    features = {s: add_features(df, nifty) for s, df in raw.items()}
    print("Fetching latest fundamental ranking inputs...")
    fundamentals = {s: get_fundamentals(s) for s in raw}
    top5 = rank_stocks(features, fundamentals)
    if top5.empty:
        raise RuntimeError("No stocks passed the feature quality gate")

    models = train_models(features)
    print("\nTOP 5 STOCKS")
    print(top5.to_string(index=False))

    print("\nNEXT-DAY OHLC PREDICTIONS")
    for _, row in top5.iterrows():
        symbol = row.symbol
        p = predict_next(symbol, raw[symbol], features[symbol], models)
        save_prediction(p, int(row["rank"]), float(row["score"]))
        print(f"{symbol:12s} Open={p['pred_open']:.2f} High={p['pred_high']:.2f} "
              f"Low={p['pred_low']:.2f} Close={p['pred_close']:.2f}")

    print("\nPrediction saved to SQLite ledger.")
    print("After the next market session, run the evaluation/retraining workflow.")


if __name__ == "__main__":
    main()
