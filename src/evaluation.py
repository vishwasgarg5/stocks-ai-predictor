"""CSV-based Stage 2 evaluation utilities.

No SQLite/database dependency is used. All persistent evaluation data lives
under data/stage2/ and is committed to GitHub by the Actions workflow.
"""
from __future__ import annotations

import pandas as pd

from .config import EVALUATIONS_DIR
from .ledger import rebuild_stock_reliability


def evaluate_predictions(actuals: dict[str, dict]) -> int:
    """Evaluate all still-open next-day prediction files against actuals."""
    from .ledger import latest_prediction_date, load_predictions, save_evaluation
    from .evening import _evaluate_prediction_frame

    prediction_date = latest_prediction_date()
    if prediction_date is None:
        return 0

    predictions = load_predictions(prediction_date)
    if predictions.empty:
        return 0

    market_dates = [v.get("date") for v in actuals.values() if v.get("date")]
    market_date = max(market_dates) if market_dates else None
    if market_date is None:
        return 0

    rows = _evaluate_prediction_frame(predictions, actuals)
    if rows.empty:
        return 0

    save_evaluation(rows, market_date)
    rebuild_stock_reliability()
    return len(rows)


def performance_report() -> pd.DataFrame:
    """Return cumulative OHLC MAPE and direction accuracy from CSV evaluations."""
    frames = []
    for path in sorted(EVALUATIONS_DIR.glob("evaluation_*.csv")):
        try:
            df = pd.read_csv(path)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    result = {
        "predictions": len(df),
        "open_mape": float(df["APE_Open"].abs().mean()),
        "high_mape": float(df["APE_High"].abs().mean()),
        "low_mape": float(df["APE_Low"].abs().mean()),
        "close_mape": float(df["APE_Close"].abs().mean()),
        "overall_mape": float(df[["APE_Open", "APE_High", "APE_Low", "APE_Close"]].abs().mean().mean()),
        "direction_accuracy": float(df["DirectionCorrect"].mean() * 100),
    }
    return pd.DataFrame([result])
