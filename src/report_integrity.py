"""Fail-closed validation for generated prediction/portfolio reports."""
from __future__ import annotations

import math
import pandas as pd


def validate_prediction_frame(df: pd.DataFrame, expected_n: int | None = None):
    errors = []
    if df is None or df.empty:
        return False, ["EMPTY_PREDICTIONS"]
    required = ["Symbol", "Pred_Open", "Pred_High", "Pred_Low", "Pred_Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.extend(f"MISSING:{c}" for c in missing)
    if "Symbol" in df.columns:
        symbols = df["Symbol"].astype(str).str.upper().str.strip()
        if symbols.duplicated().any():
            errors.append("DUPLICATE_SYMBOLS")
    for c in [x for x in required if x != "Symbol"]:
        if c in df.columns:
            values = pd.to_numeric(df[c], errors="coerce")
            if values.isna().any() or not values.map(math.isfinite).all():
                errors.append(f"INVALID:{c}")
    if all(c in df.columns for c in ["Pred_Open", "Pred_High", "Pred_Low", "Pred_Close"]):
        if (pd.to_numeric(df["Pred_High"], errors="coerce") < df[["Pred_Open", "Pred_Close"]].apply(pd.to_numeric, errors="coerce").max(axis=1)).any():
            errors.append("PRED_HIGH_INVALID")
        if (pd.to_numeric(df["Pred_Low"], errors="coerce") > df[["Pred_Open", "Pred_Close"]].apply(pd.to_numeric, errors="coerce").min(axis=1)).any():
            errors.append("PRED_LOW_INVALID")
    if expected_n is not None and len(df) > int(expected_n):
        errors.append(f"TOP_N_EXCEEDED:{len(df)}>{int(expected_n)}")
    return not errors, errors


def validate_portfolio_frame(df: pd.DataFrame):
    errors = []
    if df is None or df.empty:
        return False, ["EMPTY_PORTFOLIO"]
    required = ["Ticker", "Quantity", "Average_Price", "Current_Price", "Decision"]
    errors.extend(f"MISSING:{c}" for c in required if c not in df.columns)
    if "Ticker" in df.columns and df["Ticker"].astype(str).duplicated().any():
        errors.append("DUPLICATE_TICKER")
    if "Quantity" in df.columns and (pd.to_numeric(df["Quantity"], errors="coerce") < 0).any():
        errors.append("NEGATIVE_QUANTITY")
    return not errors, errors


def validate_telegram_messages(messages, max_length=3900):
    errors = []
    if not messages:
        return False, ["EMPTY_MESSAGES"]
    for i, message in enumerate(messages, 1):
        if not isinstance(message, str) or not message.strip():
            errors.append(f"EMPTY_MESSAGE:{i}")
        elif len(message) > max_length:
            errors.append(f"MESSAGE_TOO_LONG:{i}:{len(message)}")
    return not errors, errors
