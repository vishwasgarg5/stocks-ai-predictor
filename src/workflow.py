import pandas as pd
from src.ledger import evaluate_pending


def actuals_from_latest(raw: dict[str, pd.DataFrame]) -> dict[str, dict]:
    actuals = {}
    for symbol, df in raw.items():
        if df.empty:
            continue
        row = df.iloc[-1]
        actuals[symbol] = {
            "date": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1]),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
        }
    return actuals


def evaluate_latest_predictions(raw: dict[str, pd.DataFrame]) -> int:
    """Evaluate pending predictions against the latest completed session."""
    return evaluate_pending(actuals_from_latest(raw))
