"""Stocks AI Predictor package.

Install the P1 data-path guard before pipeline modules import market_data.
"""
from . import market_data as _market_data
from .config import MAX_UNIVERSE
from .utils import price_bucket as canonical_price_bucket
import pandas as _pd
import numpy as _np
import time as _time

_ORIGINAL_DOWNLOAD_MANY = _market_data.download_many
_ORIGINAL_DOWNLOAD_SYMBOL = _market_data.download_symbol

def _incremental_download_symbol(symbol, period=None, retries=2):
    period = period or _market_data.HISTORY_PERIOD
    symbol = _market_data.normalize_symbol(symbol)
    ticker = f"{symbol}.NS"
    cached = _market_data._read_cached_ohlcv(symbol)
    today = _pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
    p = str(period).lower()
    years = 5 if p == "5y" else 1 if p == "1y" else None
    desired_start = today - _pd.DateOffset(years=years) if years else None
    try:
        if cached.empty:
            merged = _market_data._download_range(ticker, period=period)
        elif desired_start is not None:
            frames = [cached]
            cached_start = _pd.Timestamp(cached.index.min())
            cached_end = _pd.Timestamp(cached.index.max())
            if cached_start.date() > desired_start.date():
                frames.append(_market_data._download_range(ticker, start=desired_start, end=cached_start + _pd.Timedelta(days=1)))
            if cached_end.date() < today.date():
                frames.append(_market_data._download_range(ticker, start=cached_end + _pd.Timedelta(days=1), end=today + _pd.Timedelta(days=1)))
            merged = _pd.concat([x for x in frames if x is not None and not x.empty]).sort_index()
        else:
            fresh = _market_data._download_range(ticker, period=period)
            merged = _pd.concat([cached, fresh]).sort_index() if not cached.empty else fresh
        if not merged.empty:
            merged = merged[~merged.index.duplicated(keep="last")]
            _market_data._save_cached_ohlcv(symbol, merged)
        if desired_start is not None and not merged.empty:
            merged = merged[merged.index >= desired_start]
        if len(merged) >= 30:
            return merged
        if retries > 0 and cached.empty:
            _time.sleep(1.0)
            fresh = _market_data._download_range(ticker, period=period)
            merged = _pd.concat([merged, fresh]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            if not merged.empty:
                _market_data._save_cached_ohlcv(symbol, merged)
            if desired_start is not None:
                merged = merged[merged.index >= desired_start]
            if len(merged) >= 30:
                return merged
    except Exception as exc:
        print(f"{symbol}: incremental data update failed: {exc}")
    return None

def _bounded_download_many(symbols, period=None, workers=8):
    unique = list(dict.fromkeys(_market_data.normalize_symbol(s) for s in symbols if _market_data.normalize_symbol(s)))
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

_market_data.download_symbol = _incremental_download_symbol
_market_data.download_many = _bounded_download_many
_market_data.canonical_price_bucket = canonical_price_bucket
