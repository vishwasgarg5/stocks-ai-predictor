"""Stocks AI Predictor package bootstrap.

Installs a cache-first market-data adapter before pipeline modules import
market_data. OHLCV is stored in monthly partitions so the repository does not
accumulate one mutable CSV per stock.
"""
from . import market_data as _market_data
from .config import MAX_UNIVERSE, DATA_DIR
from .utils import price_bucket as canonical_price_bucket, clean_ohlcv
import pandas as _pd
from pathlib import Path as _Path

_PARTITION_DIR = DATA_DIR / "stage2" / "market_data"
_PARTITION_DIR.mkdir(parents=True, exist_ok=True)
_LEGACY_DIR = DATA_DIR / "stage2" / "ohlcv"


def _period_start(today, period):
    p = str(period or _market_data.HISTORY_PERIOD).lower().strip()
    return {
        "5y": today - _pd.DateOffset(years=5), "5yr": today - _pd.DateOffset(years=5), "5years": today - _pd.DateOffset(years=5),
        "1y": today - _pd.DateOffset(years=1), "1yr": today - _pd.DateOffset(years=1), "1year": today - _pd.DateOffset(years=1),
        "6mo": today - _pd.DateOffset(months=6), "6m": today - _pd.DateOffset(months=6),
        "3mo": today - _pd.DateOffset(months=3), "3m": today - _pd.DateOffset(months=3),
        "1mo": today - _pd.DateOffset(months=1), "1m": today - _pd.DateOffset(months=1),
        "5d": today - _pd.Timedelta(days=7), "1wk": today - _pd.Timedelta(days=14), "1w": today - _pd.Timedelta(days=14), "1d": today - _pd.Timedelta(days=3),
    }.get(p)


def _partition_path(date):
    d = _pd.Timestamp(date)
    return _PARTITION_DIR / f"{d.year:04d}-{d.month:02d}.csv"


def _read_partition(path):
    try:
        if not path.exists():
            return _pd.DataFrame()
        df = _pd.read_csv(path, parse_dates=["Date"])
        if "Symbol" not in df.columns or "Date" not in df.columns:
            return _pd.DataFrame()
        df["Symbol"] = df["Symbol"].astype(str).str.upper()
        df = df.set_index("Date")
        return clean_ohlcv(df)
    except Exception as exc:
        print(f"Partition read failed {path}: {exc}")
        return _pd.DataFrame()


def _read_partitioned_symbol(symbol, start=None, end=None):
    symbol = _market_data.normalize_symbol(symbol)
    if not symbol:
        return _pd.DataFrame()
    start = _pd.Timestamp(start or "2000-01-01")
    end = _pd.Timestamp(end or _pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize())
    frames = []
    cur = start.replace(day=1)
    while cur <= end:
        df = _read_partition(_partition_path(cur))
        if not df.empty:
            df = df[df["Symbol"] == symbol]
            if not df.empty:
                frames.append(df.drop(columns=["Symbol"], errors="ignore"))
        cur = cur + _pd.DateOffset(months=1)
    if not frames:
        return _pd.DataFrame()
    out = _pd.concat(frames).sort_index()
    out = out[(out.index >= start) & (out.index <= end)]
    return out[~out.index.duplicated(keep="last")]


def _read_legacy_symbol(symbol):
    path = _LEGACY_DIR / f"{_market_data.normalize_symbol(symbol)}.csv"
    if not path.exists():
        return _pd.DataFrame()
    try:
        df = _pd.read_csv(path, index_col=0, parse_dates=True)
        return clean_ohlcv(df).sort_index()
    except Exception as exc:
        print(f"{symbol}: legacy cache unreadable: {exc}")
        return _pd.DataFrame()


def _write_partitioned(frames):
    valid = [clean_ohlcv(x) for x in frames if x is not None and not x.empty]
    valid = [x for x in valid if not x.empty]
    if not valid:
        return
    combined = _pd.concat(valid).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined["Symbol"] = [getattr(x, "_cache_symbol", "") for x in []]


def _save_partitioned(symbol, df):
    if df is None or df.empty:
        return
    symbol = _market_data.normalize_symbol(symbol)
    out = clean_ohlcv(df).sort_index()
    out = out[~out.index.duplicated(keep="last")].copy()
    out["Symbol"] = symbol
    out.index.name = "Date"
    for (year, month), chunk in out.groupby([out.index.year, out.index.month]):
        path = _PARTITION_DIR / f"{int(year):04d}-{int(month):02d}.csv"
        old = _read_partition(path)
        if not old.empty:
            old = old[old["Symbol"] != symbol].copy()
            old["Symbol"] = old.get("Symbol", "")
        old2 = old.reset_index() if not old.empty else _pd.DataFrame()
        new2 = chunk.reset_index()
        if not old2.empty:
            merged = _pd.concat([old2, new2], ignore_index=True)
        else:
            merged = new2
        merged["Symbol"] = merged["Symbol"].astype(str).str.upper()
        merged["Date"] = _pd.to_datetime(merged["Date"])
        merged = merged.drop_duplicates(["Symbol", "Date"], keep="last").sort_values(["Date", "Symbol"])
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(path, index=False)


def _read_cached_ohlcv(symbol):
    symbol = _market_data.normalize_symbol(symbol)
    today = _pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
    cached = _read_partitioned_symbol(symbol, today - _pd.DateOffset(years=6), today)
    if not cached.empty:
        return cached.sort_index()
    legacy = _read_legacy_symbol(symbol)
    if not legacy.empty:
        # One-time migration. Future runs read the partitioned cache.
        _save_partitioned(symbol, legacy)
        return legacy.sort_index()
    return _pd.DataFrame()


def _save_cached_ohlcv(symbol, df):
    _save_partitioned(symbol, df)


def _incremental_download_symbol(symbol, period=None, retries=2):
    symbol = _market_data.normalize_symbol(symbol)
    if not symbol:
        return None
    period = period or _market_data.HISTORY_PERIOD
    today = _pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
    desired_start = _period_start(today, period)
    cached = _read_cached_ohlcv(symbol)
    try:
        if cached.empty:
            merged = _market_data._download_range(f"{symbol}.NS", period=period)
            if merged.empty and retries > 0:
                import time
                time.sleep(1.0)
                merged = _market_data._download_range(f"{symbol}.NS", period=period)
        else:
            cached_start = _pd.Timestamp(cached.index.min()).normalize()
            cached_end = _pd.Timestamp(cached.index.max()).normalize()
            frames = [cached]
            if desired_start is not None and cached_start > desired_start:
                older = _market_data._download_range(f"{symbol}.NS", start=desired_start, end=cached_start + _pd.Timedelta(days=1))
                if not older.empty:
                    frames.append(older)
            if cached_end < today:
                newer = _market_data._download_range(f"{symbol}.NS", start=cached_end + _pd.Timedelta(days=1), end=today + _pd.Timedelta(days=1))
                if not newer.empty:
                    frames.append(newer)
            merged = _pd.concat(frames).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
        if merged is None or merged.empty:
            return None
        _save_cached_ohlcv(symbol, merged)
        if desired_start is not None:
            merged = merged[merged.index >= desired_start]
        return merged if len(merged) >= 30 else None
    except Exception as exc:
        print(f"{symbol}: incremental data update failed: {exc}")
        return None


def _bounded_download_many(symbols, period=None, workers=8):
    unique = list(dict.fromkeys(_market_data.normalize_symbol(s) for s in symbols if _market_data.normalize_symbol(s)))
    if MAX_UNIVERSE and MAX_UNIVERSE > 0:
        unique = unique[:int(MAX_UNIVERSE)]
    result = {}
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=min(int(workers), 8)) as executor:
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


_market_data._read_cached_ohlcv = _read_cached_ohlcv
_market_data._save_cached_ohlcv = _save_cached_ohlcv
_market_data.download_symbol = _incremental_download_symbol
_market_data.download_many = _bounded_download_many
_market_data.canonical_price_bucket = canonical_price_bucket
