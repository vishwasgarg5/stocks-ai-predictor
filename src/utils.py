from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from .config import (
    TIMEZONE,
    MORNING_HOUR,
    MORNING_MINUTE,
    EVENING_HOUR,
    EVENING_MINUTE,
)


IST = ZoneInfo(TIMEZONE)


def now_ist():
    return datetime.now(IST)


def today_ist():
    return now_ist().date()


def as_date(value):
    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    if isinstance(value, datetime):
        return value.date()

    return pd.Timestamp(value).date()


def flatten_yfinance_columns(df):
    if df is None or df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        new_columns = []

        for col in df.columns:
            parts = [str(x) for x in col if str(x) != ""]

            if len(parts) == 1:
                new_columns.append(parts[0])
            else:
                # Prefer OHLCV name.
                preferred = None

                for part in parts:
                    if part in [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Adj Close",
                        "Volume",
                    ]:
                        preferred = part
                        break

                new_columns.append(preferred or parts[0])

        df = df.copy()
        df.columns = new_columns

    return df


def clean_ohlcv(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = flatten_yfinance_columns(df.copy())

    required = ["Open", "High", "Low", "Close", "Volume"]

    for column in required:
        if column not in df.columns:
            return pd.DataFrame()

        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[required].copy()

    try:
        index = pd.to_datetime(df.index)
        if getattr(index, "tz", None) is not None:
            index = index.tz_localize(None)
        df.index = index.normalize()
    except Exception:
        return pd.DataFrame()

    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required)

    return df


def safe_mape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    denominator = np.maximum(np.abs(actual), 1e-8)

    return float(np.mean(np.abs((actual - predicted) / denominator)) * 100)


def normalized_mae(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    denominator = max(float(np.mean(np.abs(actual))), 1e-8)

    return float(np.mean(np.abs(actual - predicted)) / denominator)


def clamp(value, low=0.0, high=100.0):
    return max(low, min(high, float(value)))


def direction_from_return(return_value, neutral_threshold=0.002):
    if return_value > neutral_threshold:
        return "UP"

    if return_value < -neutral_threshold:
        return "DOWN"

    return "NEUTRAL"


def direction_from_prices(previous_close, future_close):
    if previous_close in [None, 0] or future_close is None:
        return "NEUTRAL"

    return direction_from_return(
        (float(future_close) / float(previous_close)) - 1.0
    )


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(f"Unsupported type: {type(value)}")


def write_json(path, data):
    path = Path(path)

    tmp = path.with_suffix(".tmp")

    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            default=json_safe,
        )

    tmp.replace(path)


def read_json(path, default=None):
    path = Path(path)

    if not path.exists():
        return default

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return default


def schedule_status(kind):
    now = now_ist()

    if kind == "morning":
        scheduled = datetime.combine(
            now.date(),
            time(MORNING_HOUR, MORNING_MINUTE),
            tzinfo=IST,
        )

    elif kind == "evening":
        scheduled = datetime.combine(
            now.date(),
            time(EVENING_HOUR, EVENING_MINUTE),
            tzinfo=IST,
        )

    else:
        return "UNKNOWN"

    minutes = (now - scheduled).total_seconds() / 60

    if minutes < -10:
        return "EARLY"

    if minutes <= 15:
        return "ON TIME"

    return "DELAYED"


def is_weekday(day=None):
    day = day or today_ist()
    return day.weekday() < 5


def format_money(value):
    if value is None or not np.isfinite(value):
        return "-"

    return f"₹{float(value):,.2f}"


def format_percent(value):
    if value is None or not np.isfinite(value):
        return "-"

    return f"{float(value):+.2f}%"


def split_messages(text, max_length=3900):
    if len(text) <= max_length:
        return [text]

    messages = []
    current = ""

    for line in text.splitlines(True):
        if len(current) + len(line) > max_length:
            if current:
                messages.append(current)
                current = ""

        current += line

    if current:
        messages.append(current)

    return messages
