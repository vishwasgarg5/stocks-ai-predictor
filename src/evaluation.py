"""CSV-based Stage 2 evaluation utilities.

No SQLite/database dependency is used. All persistent evaluation data lives
under data/stage2/ and is committed to GitHub by the Actions workflow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EVALUATIONS_DIR
from .ledger import rebuild_stock_reliability


def evaluate_predictions(actuals: dict[str, dict]) -> int:
    """Evaluate only the morning prediction for the exact actual market date.

    Fail closed: an older prediction is never substituted for today's
    evaluation. This prevents stale morning predictions from being compared
    with a later market session.
    """
    from .ledger import latest_prediction_date, load_predictions, save_evaluation
    from .evening import _evaluate_prediction_frame

    market_dates = sorted({str(v.get("date")) for v in actuals.values() if v.get("date")})
    if not market_dates:
        return 0
    market_date = market_dates[-1]

    prediction_date = latest_prediction_date(market_date)
    if prediction_date is None:
        return 0

    predictions = load_predictions(prediction_date)
    if predictions.empty:
        return 0

    rows = _evaluate_prediction_frame(predictions, actuals)
    if rows.empty:
        return 0

    # Require every evaluated row to retain immutable morning lineage.
    if "Prediction_ID" not in rows.columns or rows["Prediction_ID"].isna().any():
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
    required = ["APE_Open", "APE_High", "APE_Low", "APE_Close", "DirectionCorrect"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    ape = df[required[:4]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    direction = pd.to_numeric(df["DirectionCorrect"], errors="coerce")
    result = {
        "predictions": len(df),
        "open_mape": float(ape["APE_Open"].abs().mean()),
        "high_mape": float(ape["APE_High"].abs().mean()),
        "low_mape": float(ape["APE_Low"].abs().mean()),
        "close_mape": float(ape["APE_Close"].abs().mean()),
        "overall_mape": float(ape.abs().mean().mean()),
        "direction_accuracy": float(direction.mean() * 100) if direction.notna().any() else np.nan,
    }
    return pd.DataFrame([result])
