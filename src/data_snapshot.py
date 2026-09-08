"""Canonical market-data snapshot layer.

All decision engines should consume a snapshot rather than independently
choosing the latest row.  The snapshot is fail-closed: stale, future, malformed
or insufficient data is never silently promoted to a decision input.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from typing import Optional

import pandas as pd

from .data_quality import validate_ohlcv
from .market_data import _read_cached_ohlcv, download_symbol
from .utils import as_date, is_nse_trading_day, today_ist


@dataclass(frozen=True)
class DataSnapshot:
    symbol: str
    requested_cutoff: str
    data_date: Optional[str]
    rows: int
    source: str
    status: str
    errors: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return self.status == "OK"

    def to_dict(self):
        return asdict(self)


def _cutoff(value=None) -> date:
    return as_date(value or today_ist())


def canonicalize(df: pd.DataFrame, cutoff_date=None) -> pd.DataFrame:
    """Normalize and hard-limit OHLCV to the requested trading-date cutoff."""
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    x.index = pd.to_datetime(x.index, errors="coerce")
    x = x[~x.index.isna()]
    x = x.sort_index()
    cutoff = _cutoff(cutoff_date)
    x = x[x.index.date <= cutoff]
    return x[~x.index.duplicated(keep="last")]


def get_snapshot(symbol: str, cutoff_date=None, min_rows: int = 60, allow_download: bool = True):
    """Return (clean OHLCV, immutable snapshot metadata).

    Cache is checked first. Download is used only when the cache cannot satisfy
    the requested cutoff. The returned frame can never contain data after the
    requested cutoff.
    """
    symbol = str(symbol).strip().upper().removesuffix(".NS")
    cutoff = _cutoff(cutoff_date)
    if not is_nse_trading_day(cutoff):
        return pd.DataFrame(), DataSnapshot(symbol, str(cutoff), None, 0, "NONE", "INVALID_CUTOFF", ("NON_TRADING_DAY_CUTOFF",))

    source = "GITHUB_OHLCV"
    df = canonicalize(_read_cached_ohlcv(symbol), cutoff)

    def validate(x):
        return validate_ohlcv(x, cutoff_date=cutoff, min_rows=min_rows)

    ok, errors = validate(df)
    if (df.empty or not ok) and allow_download:
        fresh = download_symbol(symbol, period="3y")
        fresh = canonicalize(fresh, cutoff)
        if not fresh.empty:
            df = fresh if df.empty else pd.concat([df, fresh]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
            df = canonicalize(df, cutoff)
            source = "GITHUB_OHLCV+DOWNLOAD"
        ok, errors = validate(df)

    data_date = str(df.index[-1].date()) if not df.empty else None
    status = "OK" if ok and data_date is not None else ("STALE" if data_date and data_date < str(cutoff) else "INVALID")
    return df, DataSnapshot(symbol, str(cutoff), data_date, len(df), source, status, tuple(errors))


def snapshot_map(symbols, cutoff_date=None, min_rows=60, allow_download=True):
    """Build snapshots deterministically; duplicate symbols are removed."""
    out = {}
    for symbol in dict.fromkeys(str(s).upper().removesuffix(".NS") for s in symbols):
        out[symbol] = get_snapshot(symbol, cutoff_date, min_rows, allow_download)
    return out
