"""Stocks AI Predictor package.

Install the P1 data-path guard before pipeline modules import market_data.
The market-data wrapper is intentionally cache-first: an existing GitHub
OHLCV file is reused and Yahoo is queried only for missing date ranges.
"""
from . import market_data as _market_data
from .config import MAX_UNIVERSE
from .utils import price_bucket as canonical_price_bucket
import pandas as _pd


def _period_start(today, period):
    """Return the earliest date required for a yfinance-style period."""
    p = str(period or _market_data.HISTORY_PERIOD).lower().strip()
    if p in {"5y", "5yr", "5years"}:
        return today - _pd.DateOffset(years=5)
    if p in {"1y", "1yr", "1year"}:
        return today - _pd.DateOffset(years=1)
    if p in {"6mo", "6m"}:
        return today - _pd.DateOffset(months=6)
    if p in {"3mo", "3m"}:
        return today - _pd.DateOffset(months=3)
    if p in {"1mo", "1m"}:
        return today - _pd.DateOffset(months=1)
    if p in {"5d", "1wk", "1w", "1d"}:
        days = {"5d": 7, "1wk": 14, "1w": 14, "1d": 3}[p]
        return today - _pd.Timedelta(days=days)
    return None


def _merge_and_save(symbol, frames):
    valid = [x for x in frames if x is not None and not x.empty]
    if not valid:
        return _pd.DataFrame()
    merged = _pd.concat(valid).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    _market_data._save_cached_ohlcv(symbol, merged)
    return merged


def _incremental_download_symbol(symbol, period=None, retries=2):
    """Read GitHub cache first; download only missing OHLCV ranges.

    First use for a symbol downloads the requested history and stores it.
    Subsequent runs never re-download the full history: they append only the
    missing tail (and, if necessary, the missing older head), then de-duplicate
    by trading date.  Short-period callers (for example the quality gate's
    1mo request) also reuse the long cache instead of downloading 1mo again.
    """
    symbol = _market_data.normalize_symbol(symbol)
    if not symbol:
        return None
    period = period or _market_data.HISTORY_PERIOD
    cached = _market_data._read_cached_ohlcv(symbol)
    today = _pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
    desired_start = _period_start(today, period)

    try:
        if cached.empty:
            # First observation: only here is a full-period download allowed.
            merged = _market_data._download_range(f"{symbol}.NS", period=period)
            if merged.empty and retries > 0:
                import time
                time.sleep(1.0)
                merged = _market_data._download_range(f"{symbol}.NS", period=period)
            if not merged.empty:
                _market_data._save_cached_ohlcv(symbol, merged)
        else:
            cached_start = _pd.Timestamp(cached.index.min()).normalize()
            cached_end = _pd.Timestamp(cached.index.max()).normalize()
            frames = [cached]

            # Backfill only if the repository cache does not cover the
            # requested history window.
            if desired_start is not None and cached_start > desired_start:
                older = _market_data._download_range(
                    f"{symbol}.NS",
                    start=desired_start,
                    end=cached_start + _pd.Timedelta(days=1),
                )
                if not older.empty:
                    frames.append(older)

            # Append only dates newer than the repository cache.  If the cache
            # already contains today's session (or the latest available
            # session), this path performs no network request.
            if cached_end < today:
                newer = _market_data._download_range(
                    f"{symbol}.NS",
                    start=cached_end + _pd.Timedelta(days=1),
                    end=today + _pd.Timedelta(days=1),
                )
                if not newer.empty:
                    frames.append(newer)

            merged = _merge_and_save(symbol, frames)

        if merged.empty:
            return None
        if desired_start is not None:
            merged = merged[merged.index >= desired_start]
        # Keep the original contract: callers need enough rows for indicators.
        return merged if len(merged) >= 30 else None
    except Exception as exc:
        print(f"{symbol}: incremental data update failed: {exc}")
        return None


def _bounded_download_many(symbols, period=None, workers=8):
    """Cap the universe before any stock download is submitted."""
    unique = list(dict.fromkeys(
        _market_data.normalize_symbol(s)
        for s in symbols
        if _market_data.normalize_symbol(s)
    ))
    if MAX_UNIVERSE and MAX_UNIVERSE > 0:
        unique = unique[:int(MAX_UNIVERSE)]

    result = {}
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_incremental_download_symbol, s, period): s for s in unique}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    result[symbol] = df
            except Exception as exc:
                print(f"{symbol}: {exc}")
    print(f"Repository-cache data available for {len(result)}/{len(unique)} stocks")
    return result


# Install the bounded/cache-first implementation before pipeline modules call it.
_market_data.download_symbol = _incremental_download_symbol
_market_data.download_many = _bounded_download_many
_market_data.canonical_price_bucket = canonical_price_bucket
