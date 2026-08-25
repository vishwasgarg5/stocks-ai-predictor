import sqlite3
import pandas as pd
from config import DB_PATH


def evaluate_predictions(actuals: dict[str, dict]) -> int:
    from src.ledger import evaluate_pending
    return evaluate_pending(actuals)


def performance_report() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query("SELECT open_error, high_error, low_error, close_error, direction_correct FROM predictions WHERE actual_close IS NOT NULL", con)
    if df.empty:
        return pd.DataFrame()
    return pd.DataFrame([{
        "predictions": len(df),
        "open_mae": df.open_error.abs().mean(),
        "high_mae": df.high_error.abs().mean(),
        "low_mae": df.low_error.abs().mean(),
        "close_mae": df.close_error.abs().mean(),
        "direction_accuracy": df.direction_correct.mean(),
    }])
