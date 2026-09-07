import pandas as pd

from src.data_quality import validate_ohlcv


def _frame():
    idx = pd.date_range("2026-08-01", periods=35, freq="D")
    return pd.DataFrame({
        "Open": 100.0,
        "High": 102.0,
        "Low": 98.0,
        "Close": 101.0,
        "Volume": 1000.0,
    }, index=idx)


def test_data_quality_rejects_future_rows():
    ok, errors = validate_ohlcv(_frame(), cutoff_date="2026-08-20")
    assert not ok
    assert "FUTURE_DATA" in errors


def test_data_quality_rejects_duplicate_dates():
    df = _frame()
    df.index = list(df.index[:-1]) + [df.index[-2]]
    ok, errors = validate_ohlcv(df, cutoff_date="2026-09-30")
    assert not ok
    assert "DUPLICATE_DATE_INDEX" in errors


def test_data_quality_rejects_invalid_ohlc():
    df = _frame()
    df.loc[df.index[-1], "High"] = 90
    ok, errors = validate_ohlcv(df, cutoff_date="2026-09-30")
    assert not ok
    assert "HIGH_LT_OPEN_CLOSE" in errors
