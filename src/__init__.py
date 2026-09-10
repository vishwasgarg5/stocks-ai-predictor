"""Package bootstrap and backwards-compatible market-data adapter."""
from pathlib import Path
from threading import RLock
import pandas as pd

from .config import DATA_DIR, HISTORY_PERIOD, MAX_UNIVERSE
from . import market_data as _market_data

_PARTITION_DIR = DATA_DIR / "stage2" / "market_data"
_LEGACY_DIR = DATA_DIR / "stage2" / "ohlcv"
_CACHE_LOCK = RLock()
_PARTITION_MEMORY = {}


def _period_start(period=HISTORY_PERIOD):
    end = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
    text = str(period or HISTORY_PERIOD).lower().strip()
    if text.endswith("y"):
        start = end - pd.DateOffset(years=max(1, int(text[:-1] or 1)))
    elif text.endswith("mo"):
        start = end - pd.DateOffset(months=max(1, int(text[:-2] or 1)))
    elif text.endswith("d"):
        start = end - pd.Timedelta(days=max(1, int(text[:-1] or 1)))
    else:
        start = end - pd.DateOffset(years=5)
    return start.normalize(), end


def _partition_path(date):
    d = pd.Timestamp(date)
    return _PARTITION_DIR / f"{d.year:04d}-{d.month:02d}.csv"


def _empty_partition():
    return pd.DataFrame(columns=["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"])


def _read_partition(path):
    path = Path(path)
    with _CACHE_LOCK:
        cached = _PARTITION_MEMORY.get(path)
        if cached is not None:
            return cached.copy()
        if not path.exists():
            return _empty_partition()
        try:
            df = pd.read_csv(path)
            if "Date" not in df.columns:
                return _empty_partition()
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
            if "Symbol" not in df.columns:
                df["Symbol"] = ""
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col not in df.columns:
                    df[col] = pd.NA
            df = df[["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Date"])
            _PARTITION_MEMORY[path] = df.copy()
            return df
        except Exception as exc:
            raise RuntimeError(f"Unable to read market-data partition {path}: {exc}") from exc


def _read_partitioned_symbol(symbol, start=None, end=None):
    start = pd.Timestamp(start) if start is not None else _period_start()[0]
    end = pd.Timestamp(end) if end is not None else _period_start()[1]
    frames = []
    month = start.replace(day=1)
    last = end.replace(day=1)
    while month <= last:
        df = _read_partition(_partition_path(month))
        if not df.empty:
            x = df[df["Symbol"].astype(str).str.upper() == str(symbol).upper()].copy()
            if not x.empty:
                frames.append(x)
        month += pd.DateOffset(months=1)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).drop_duplicates("Date", keep="last").sort_values("Date")
    out = out[(out["Date"] >= start) & (out["Date"] <= end)]
    return _market_data.clean_ohlcv(out.set_index("Date"))


def _read_legacy_symbol(symbol):
    path = _LEGACY_DIR / f"{symbol}.csv"
    if not path.exists():
        return None
    try:
        return _market_data.clean_ohlcv(pd.read_csv(path, index_col=0, parse_dates=True))
    except Exception as exc:
        raise RuntimeError(f"Unable to read legacy cache for {symbol}: {exc}") from exc


def _read_cached_ohlcv(symbol, period=HISTORY_PERIOD):
    start, end = _period_start(period)
    cached = _read_partitioned_symbol(symbol, start, end)
    if cached is not None and not cached.empty:
        return cached
    legacy = _read_legacy_symbol(symbol)
    if legacy is not None and not legacy.empty:
        try:
            _save_partitioned(symbol, legacy)
        except Exception:
            raise
        return legacy[(legacy.index >= start) & (legacy.index <= end)]
    return None


def _save_partitioned(symbol, df):
    if df is None or df.empty:
        return
    x = _market_data.clean_ohlcv(df.copy()).reset_index()
    if "Date" not in x.columns:
        x = x.rename(columns={x.columns[0]: "Date"})
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
    x["Symbol"] = str(symbol).upper()
    cols = ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
    x = x[cols].dropna(subset=["Date"])
    _PARTITION_DIR.mkdir(parents=True, exist_ok=True)
    with _CACHE_LOCK:
        for month, group in x.groupby(x["Date"].dt.to_period("M")):
            path = _PARTITION_DIR / f"{month.year:04d}-{month.month:02d}.csv"
            existing = _read_partition(path)
            if not existing.empty:
                existing = existing[existing["Symbol"].astype(str).str.upper() != str(symbol).upper()]
            merged = pd.concat([existing, group], ignore_index=True)
            merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce").dt.normalize()
            merged = merged.dropna(subset=["Date"]).drop_duplicates(["Symbol", "Date"], keep="last").sort_values(["Symbol", "Date"])
            merged.to_csv(path, index=False)
            _PARTITION_MEMORY[path] = merged.copy()


def _save_cached_ohlcv(symbol, df):
    _save_partitioned(symbol, df)


def _incremental_download_symbol(symbol, period=HISTORY_PERIOD, retries=2):
    start, end = _period_start(period)
    cached = _read_cached_ohlcv(symbol, period)
    try:
        if cached is None or cached.empty:
            fresh = _market_data._download_range(symbol, start=start, end=end, period=None, retries=retries)
        else:
            cached = _market_data.clean_ohlcv(cached)
            pieces = [cached]
            cached_start, cached_end = cached.index.min(), cached.index.max()
            if cached_start > start:
                older = _market_data._download_range(symbol, start=start, end=cached_start, period=None, retries=retries)
                if older is not None and not older.empty:
                    pieces.append(older)
            if cached_end < end:
                newer = _market_data._download_range(symbol, start=cached_end + pd.Timedelta(days=1), end=end, period=None, retries=retries)
                if newer is not None and not newer.empty:
                    pieces.append(newer)
            fresh = pd.concat(pieces)
        fresh = _market_data.clean_ohlcv(fresh)
        fresh = fresh[~fresh.index.duplicated(keep="last")]
        _save_partitioned(symbol, fresh)
        return fresh[(fresh.index >= start) & (fresh.index <= end)]
    except (TypeError, AttributeError, KeyError) as exc:
        raise RuntimeError(f"Programming error while updating {symbol}: {exc}") from exc
    except Exception as exc:
        print(f"Market data update failed for {symbol}: {exc}")
        return cached if cached is not None else None


def _bounded_download_many(symbols, period=HISTORY_PERIOD, workers=8):
    unique, seen = [], set()
    for symbol in symbols or []:
        s = str(symbol).strip().upper()
        if s and s not in seen:
            seen.add(s)
            unique.append(s)
        if len(unique) >= int(MAX_UNIVERSE):
            break
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = {}
    failures = {}
    with ThreadPoolExecutor(max_workers=min(max(1, int(workers)), 8)) as pool:
        futures = {pool.submit(_incremental_download_symbol, s, period, 2): s for s in unique}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                data = future.result()
                if data is not None and not data.empty:
                    results[symbol] = data
                else:
                    failures[symbol] = "NO_DATA"
            except RuntimeError as exc:
                failures[symbol] = str(exc)
                if "Programming error" in str(exc):
                    raise
            except Exception as exc:
                failures[symbol] = str(exc)
    print(f"Repository-cache data available for {len(results)}/{len(unique)} stocks; failures={len(failures)}")
    if failures and len(failures) <= 10:
        print("Market-data failures: " + "; ".join(f"{s}:{e}" for s, e in failures.items()))
    return results


_market_data._read_cached_ohlcv = _read_cached_ohlcv
_market_data._save_cached_ohlcv = _save_cached_ohlcv
_market_data.download_symbol = _incremental_download_symbol
_market_data.download_many = _bounded_download_many

read_cached_ohlcv = _read_cached_ohlcv
save_cached_ohlcv = _save_cached_ohlcv
download_symbol = _incremental_download_symbol
download_many = _bounded_download_many
