import io
import time
from typing import Dict
from pathlib import Path
import pandas as pd
import requests
import yfinance as yf
from config import INTERVAL, ROLLING_DAYS, MIN_ROWS, NIFTY50, OHLCV_DIR, NIFTY_CSV, UNIVERSE_CSV

NIFTY100_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"
MIDCAP50_URL = "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap50list.csv"


def ticker(symbol: str) -> str:
    return f"{symbol}.NS"


def _download(symbol_or_ticker: str, start=None, end=None, period=None) -> pd.DataFrame:
    kwargs = dict(interval=INTERVAL, auto_adjust=False, progress=False, threads=False)
    if start is not None:
        kwargs["start"] = start
        if end is not None:
            kwargs["end"] = end
    else:
        kwargs["period"] = period or "3mo"
    df = yf.download(symbol_or_ticker, **kwargs)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in cols if c in df.columns]].copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df[~df.index.duplicated(keep="last")].sort_index()


def read_stored(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return _clean(pd.read_csv(path, index_col=0, parse_dates=True))
    except Exception:
        return pd.DataFrame()


def write_stored(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index.name = "Date"
    out.to_csv(path, float_format="%.6f")


def rolling_trim(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=ROLLING_DAYS)
    return df.loc[df.index >= cutoff].sort_index()


def _read_universe_cache() -> list[str]:
    if not UNIVERSE_CSV.exists():
        return []
    try:
        df = pd.read_csv(UNIVERSE_CSV)
        return sorted(set(df["Symbol"].astype(str).str.strip().str.upper()))
    except Exception:
        return []


def _fetch_index_symbols(url: str) -> list[str]:
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    df = pd.read_csv(io.BytesIO(response.content))
    symbol_col = next((c for c in df.columns if str(c).strip().lower() in {"symbol", "symbols"}), None)
    if symbol_col is None:
        raise ValueError(f"No Symbol column in {url}")
    return sorted(set(df[symbol_col].astype(str).str.strip().str.upper()))


def get_nifty150_universe() -> list[str]:
    try:
        large = _fetch_index_symbols(NIFTY100_URL)
        mid = _fetch_index_symbols(MIDCAP50_URL)
        universe = sorted(set(large + mid))
        if len(universe) < 140:
            raise ValueError(f"Unexpected Nifty 150 universe size: {len(universe)}")
        pd.DataFrame({"Symbol": universe}).to_csv(UNIVERSE_CSV, index=False)
        print(f"Nifty 150 universe refreshed: {len(universe)} symbols")
        return universe
    except Exception as exc:
        cached = _read_universe_cache()
        if len(cached) >= 140:
            print(f"Nifty 150 refresh unavailable; using cached universe: {len(cached)} symbols ({exc})")
            return cached
        raise RuntimeError(f"Unable to load Nifty 150 universe and no valid cache exists: {exc}")


def _incremental_window(latest: pd.Timestamp) -> tuple[str, str] | None:
    """Return a valid yfinance [start, exclusive end) window.

    Never derive the end from the current clock. This avoids yfinance requests where
    start is accidentally later than end around midnight/time-zone boundaries.
    """
    start_date = latest.normalize() + pd.Timedelta(days=1)
    today = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
    # End is exclusive in yfinance. Include today and leave one extra calendar day
    # so a completed session today is returned without relying on intraday time.
    end_date = max(today + pd.Timedelta(days=1), start_date + pd.Timedelta(days=1))
    if start_date >= end_date:
        return None
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def incremental_update(symbol: str) -> pd.DataFrame:
    path = OHLCV_DIR / f"{symbol}.csv"
    existing = read_stored(path)
    if existing.empty:
        updated = _clean(_download(ticker(symbol), period="3mo"))
    else:
        window = _incremental_window(existing.index.max())
        if window is None:
            updated = existing
        else:
            start, end = window
            new = _clean(_download(ticker(symbol), start=start, end=end))
            updated = pd.concat([existing, new]) if not new.empty else existing
            updated = updated[~updated.index.duplicated(keep="last")].sort_index()
    updated = rolling_trim(updated)
    if len(updated) >= MIN_ROWS:
        write_stored(path, updated)
    return updated


def fetch_exact_session(symbol: str, target_date: str) -> dict | None:
    """Fetch only the requested completed session for evaluation.

    This deliberately bypasses the cached OHLCV ledger so an old/stale cached row cannot
    be used as the actual for a different prediction date. The one-day fetch is only for
    the pending prediction(s), not a replacement for incremental 3-month storage.
    """
    target = pd.Timestamp(target_date)
    start = target.strftime("%Y-%m-%d")
    end = (target + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        df = _clean(_download(ticker(symbol), start=start, end=end))
        if df.empty or target.normalize() not in df.index:
            return None
        r = df.loc[target.normalize()]
        return {"open": float(r["Open"]), "high": float(r["High"]), "low": float(r["Low"]), "close": float(r["Close"])}
    except Exception as exc:
        print(f"Exact-session fetch failed for {symbol} {target_date}: {exc}")
        return None


def update_universe(symbols=None) -> Dict[str, pd.DataFrame]:
    symbols = symbols or get_nifty150_universe()
    result = {}
    for i, symbol in enumerate(symbols, 1):
        try:
            df = incremental_update(symbol)
            if len(df) >= MIN_ROWS:
                result[symbol] = df
            print(f"[{i:03d}/{len(symbols)}] {symbol}: {len(df)} rows")
        except Exception as exc:
            print(f"{symbol}: ERROR {exc}")
        time.sleep(0.15)
    return result


def update_nifty() -> pd.DataFrame:
    existing = read_stored(NIFTY_CSV)
    if existing.empty:
        updated = _clean(_download("^NSEI", period="3mo"))
    else:
        window = _incremental_window(existing.index.max())
        if window is None:
            updated = existing
        else:
            start, end = window
            new = _clean(_download("^NSEI", start=start, end=end))
            updated = pd.concat([existing, new]) if not new.empty else existing
            updated = updated[~updated.index.duplicated(keep="last")].sort_index()
    updated = rolling_trim(updated)
    if not updated.empty:
        write_stored(NIFTY_CSV, updated)
    return updated


def download_universe(symbols=None):
    return update_universe(symbols)


def download_nifty():
    return update_nifty()
