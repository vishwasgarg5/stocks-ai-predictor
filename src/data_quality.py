"""Fail-closed validation for market data before model decisions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import is_nse_trading_day


def validate_ohlcv(df, cutoff_date=None, min_rows=30):
    errors = []
    if df is None or df.empty:
        return False, ["EMPTY_DATA"]

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, [f"MISSING:{c}" for c in missing]

    x = df.copy()
    for c in required:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    if len(x) < int(min_rows):
        errors.append(f"TOO_FEW_ROWS:{len(x)}")
    if x[required].isna().any().any():
        errors.append("NAN_OHLCV")
    if not np.isfinite(x[required].to_numpy(dtype=float)).all():
        errors.append("NONFINITE_OHLCV")
    if (x["Volume"] < 0).any():
        errors.append("NEGATIVE_VOLUME")
    if (x["High"] < x[["Open", "Close"]].max(axis=1)).any():
        errors.append("HIGH_LT_OPEN_CLOSE")
    if (x["Low"] > x[["Open", "Close"]].min(axis=1)).any():
        errors.append("LOW_GT_OPEN_CLOSE")
    if (x[["Open", "High", "Low", "Close"]] <= 0).any().any():
        errors.append("NONPOSITIVE_PRICE")

    idx = pd.to_datetime(x.index, errors="coerce")
    if idx.isna().any():
        errors.append("INVALID_DATE_INDEX")
    else:
        if idx.duplicated().any():
            errors.append("DUPLICATE_DATE_INDEX")
        if not idx.is_monotonic_increasing:
            errors.append("UNSORTED_DATE_INDEX")
        if cutoff_date is not None and len(idx):
            cutoff = pd.Timestamp(cutoff_date).date()
            if idx.max().date() > cutoff:
                errors.append("FUTURE_DATA")
            if not is_nse_trading_day(cutoff):
                errors.append("NON_TRADING_DAY_CUTOFF")

    return len(errors) == 0, errors


def validate_universe(data_map, cutoff_date=None, min_rows=30):
    passed, failed = {}, {}
    for symbol, df in (data_map or {}).items():
        ok, errors = validate_ohlcv(df, cutoff_date, min_rows)
        (passed if ok else failed)[symbol] = errors
    return passed, failed
