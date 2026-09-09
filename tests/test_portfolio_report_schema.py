import numpy as np
import pandas as pd

from src.portfolio_report import OUTPUT_COLUMNS, _decision, load_portfolio, portfolio_snapshot


def test_portfolio_schema_is_stable():
    df, summary = portfolio_snapshot()
    assert all(c in df.columns for c in OUTPUT_COLUMNS)
    assert "Decision" in df.columns
    assert "PnL" in df.columns
    assert "Return_Pct" in df.columns
    assert isinstance(summary, dict)


def test_portfolio_input_accepts_missing_average_price():
    df = load_portfolio()
    assert "Ticker" in df.columns
    assert "Quantity" in df.columns
    assert len(df) >= 1
    assert (df["Quantity"] > 0).all()


def test_decision_never_requires_missing_columns():
    decision, reason = _decision(100.0, 110.0, np.nan, 0.0, [])
    assert decision == "HOLD"
    assert reason


def test_profit_target_is_sell():
    decision, _ = _decision(121.0, 110.0, 125.0, 80.0, [])
    assert decision == "SELL"


def test_profit_target_has_precision_tolerance():
    decision, _ = _decision(121.0 - 1e-10, 110.0, 125.0, 80.0, [])
    assert decision == "SELL"
