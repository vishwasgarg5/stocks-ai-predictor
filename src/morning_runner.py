"""
STAGE 4 MORNING PIPELINE
========================
Easy access: this is the main morning entry point.
Stage 4 additions:
  1. Price-bucket selection: >1000, 500-999, 100-499, 50-99, 10-49.
  2. Sector intelligence: sector mapping, momentum and breadth context.
  3. Sector-aware final ranking.
Stage 2 prediction, jump and intraday engines remain intact.
"""
import pandas as pd
from .config import PRESCREEN_N, HISTORY_PERIOD, JUMP_CANDIDATE_N, MODEL_VERSION
from .market_data import load_universe, download_many, filter_liquid_universe, get_completed_session_date, get_market_regime
from .features import technical_score
from .prediction import train_stock_bundle, predict_stock
from .selection import select_top_stocks, score_candidates
from .stage4_engine import add_stage4_context, select_price_bucket_candidates
from .jump_engine import generate_jump_watchlist
from .intraday_engine import generate_intraday_watchlist
from .ledger import prediction_exists, save_predictions, save_jump_predictions, save_intraday_predictions
from .retraining import load_model_state
from .telegram_report import send_telegram, morning_report
from .utils import today_ist, is_weekday, schedule_status


def run():
    prediction_date = today_ist()
    if not is_weekday():
        print("Weekend. Morning prediction skipped.")
        return
    if prediction_exists(prediction_date):
        print(f"Prediction already exists for {prediction_date}. Skipping.")
        return

    cutoff_date = get_completed_session_date("morning")
    if cutoff_date is None:
        raise RuntimeError("Unable to determine market cutoff.")

    universe = load_universe()
    raw_data = download_many(universe, HISTORY_PERIOD, workers=8)
    data_map = filter_liquid_universe(raw_data)
    if len(data_map) < 20:
        raise RuntimeError("Too few liquid stocks.")

    regime = get_market_regime(cutoff_date)["name"]
    variant = load_model_state().get("active_variant", "A")

    # Stage 2 technical prescreen remains the first expensive-model gate.
    scored = []
    for symbol, df in data_map.items():
        try:
            hist = df[df.index <= pd.Timestamp(cutoff_date)]
            scored.append((symbol, technical_score(hist)))
        except Exception:
            pass
    scored.sort(key=lambda x: x[1], reverse=True)
    candidate_symbols = [x[0] for x in scored[:PRESCREEN_N]]

    candidate_rows = []
    for symbol in candidate_symbols:
        try:
            bundle = train_stock_bundle(data_map[symbol], symbol, cutoff_date, variant)
            result = predict_stock(data_map[symbol], bundle, cutoff_date)
            candidate_rows.append({
                "Symbol": symbol,
                **result,
                "ModelVariant": variant,
                "ModelVersion": MODEL_VERSION,
                "DataCutoff": str(cutoff_date),
            })
        except Exception as exc:
            print(f"{symbol}: prediction failed: {exc}")

    if len(candidate_rows) < 5:
        raise RuntimeError("Unable to generate five predictions.")

    # =====================================================================
    # STAGE 4 — SECTOR + PRICE-BUCKET CONTEXT
    # Easy access: see src/stage4_engine.py for all bucket definitions and
    # sector-strength logic. This step does not change the OHLC predictions.
    # =====================================================================
    candidates = add_stage4_context(pd.DataFrame(candidate_rows), data_map, regime)
    candidates = candidates[candidates["PriceBucket"] != "OUT"].copy()
    if len(candidates) < 5:
        raise RuntimeError("Too few candidates inside the configured price buckets.")

    # Score all candidates first, then keep the strongest few from each bucket.
    candidates = score_candidates(candidates, regime)
    bucket_candidates = select_price_bucket_candidates(candidates, per_bucket=3)
    selected = select_top_stocks(bucket_candidates, 5, regime)
    if len(selected) < 5:
        # Safe fallback if one bucket has too little representation.
        selected = select_top_stocks(candidates, 5, regime)

    selected["PredictionDate"] = str(prediction_date)
    save_predictions(
        selected,
        prediction_date,
        {
            "Stage": "Stage 4",
            "PredictionDate": str(prediction_date),
            "DataCutoff": str(cutoff_date),
            "ModelVariant": variant,
            "ModelVersion": MODEL_VERSION,
            "Regime": regime,
            "PriceBuckets": [">1000", "500-999", "100-499", "50-99", "10-49"],
            "SelectedStocks": selected["Symbol"].tolist(),
        },
    )

    # Stage 2 jump engine continues to operate on the strongest prescreen set.
    jump_data = {s: data_map[s] for s in candidate_symbols[:JUMP_CANDIDATE_N] if s in data_map}
    jump_watchlist = generate_jump_watchlist(jump_data, cutoff_date, variant)
    if not jump_watchlist.empty:
        save_jump_predictions(jump_watchlist, prediction_date)

    # Stage 2 intraday engine still scans the complete liquid universe.
    intraday = generate_intraday_watchlist(list(data_map.keys()))
    if not intraday.empty:
        save_intraday_predictions(intraday, prediction_date)

    report = morning_report(
        prediction_date,
        cutoff_date,
        schedule_status("morning"),
        regime,
        variant,
        selected,
        jump_watchlist,
        intraday,
        universe_count=len(universe),
        liquid_count=len(data_map),
        prescreen_count=len(candidate_symbols),
    )
    send_telegram(report)
    print(report)


if __name__ == "__main__":
    run()
