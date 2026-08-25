import time
from typing import Dict
from pathlib import Path
import pandas as pd
import yfinance as yf
from config import INTERVAL, ROLLING_DAYS, MIN_ROWS, NIFTY50, OHLCV_DIR, NIFTY_CSV


def ticker(symbol: str) -> str:
    return f"{symbol}.NS"


def _download(symbol_or_ticker: str, start=None, period=None) -> pd.DataFrame:
    kwargs = dict(interval=INTERVAL, auto_adjust=False, progress=False, threads=False)
    if start is not None:
        kwargs["start"] = start
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
    df.index = idx
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


def incremental_update(symbol: str) -> pd.DataFrame:
    path = OHLCV_DIR / f"{symbol}.csv"
    existing = read_stored(path)
    if existing.empty:
        updated = _clean(_download(ticker(symbol), period="3mo"))
    else:
        start = (existing.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        new = _clean(_download(ticker(symbol), start=start))
        updated = pd.concat([existing, new])
        updated = updated[~updated.index.duplicated(keep="last")].sort_index()
    updated = rolling_trim(updated)
    if len(updated) >= MIN_ROWS:
        write_stored(path, updated)
    return updated


def update_universe(symbols=NIFTY50) -> Dict[str, pd.DataFrame]:
    result = {}
    for i, symbol in enumerate(symbols, 1):
        try:
            df = incremental_update(symbol)
            if len(df) >= MIN_ROWS:
                result[symbol] = df
            print(f"[{i:02d}/{len(symbols)}] {symbol}: {len(df)} rows")
        except Exception as exc:
            print(f"{symbol}: ERROR {exc}")
        time.sleep(0.15)
    return result


def update_nifty() -> pd.DataFrame:
    existing = read_stored(NIFTY_CSV)
    if existing.empty:
        updated = _clean(_download("^NSEI", period="3mo"))
    else:
        start = (existing.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        new = _clean(_download("^NSEI", start=start))
        updated = pd.concat([existing, new])
        updated = updated[~updated.index.duplicated(keep="last")].sort_index()
    updated = rolling_trim(updated)
    write_stored(NIFTY_CSV, updated)
    return updated


def download_universe(symbols=NIFTY50):
    return update_universe(symbols)


def download_nifty():
    return update_nifty()
