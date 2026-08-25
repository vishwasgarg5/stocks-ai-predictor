import io
import time
from typing import Dict
import pandas as pd
import yfinance as yf
from config import INTERVAL, LOOKBACK_PERIOD, MIN_ROWS, NIFTY50

DATA_DIR = "data/ohlcv"
NIFTY_PATH = "data/nifty.csv"


def ticker(symbol: str) -> str:
    return f"{symbol}.NS"


def _download(symbol_or_ticker: str, start=None, period=None) -> pd.DataFrame:
    kwargs = dict(interval=INTERVAL, auto_adjust=False, progress=False, threads=False)
    if start is not None:
        kwargs["start"] = start
    else:
        kwargs["period"] = period or LOOKBACK_PERIOD
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
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df[~df.index.duplicated(keep="last")].sort_index()


def read_csv(text: str) -> pd.DataFrame:
    if not text:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text), index_col=0, parse_dates=True)
    return _clean(df)


def csv_text(df: pd.DataFrame) -> str:
    out = df.copy()
    out.index.name = "Date"
    return out.to_csv(float_format="%.6f")


def incremental_update(symbol: str, existing: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        new = _clean(_download(ticker(symbol), period=LOOKBACK_PERIOD))
        return new.tail(70)
    last_date = existing.index.max()
    start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    new = _clean(_download(ticker(symbol), start=start))
    combined = pd.concat([existing, new]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    # Keep approximately 3 months while retaining enough rows for indicators.
    cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(months=3)
    return combined.loc[combined.index >= cutoff]


def update_universe(existing: Dict[str, pd.DataFrame] | None = None, symbols=NIFTY50):
    existing = existing or {}
    result = {}
    for i, symbol in enumerate(symbols, 1):
        try:
            df = incremental_update(symbol, existing.get(symbol, pd.DataFrame()))
            if len(df) >= MIN_ROWS:
                result[symbol] = df
            print(f"[{i:02d}/{len(symbols)}] {symbol}: {len(df)} rows")
        except Exception as exc:
            print(f"{symbol}: ERROR {exc}")
        time.sleep(0.15)
    return result


def update_nifty(existing: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        df = _clean(_download("^NSEI", period=LOOKBACK_PERIOD))
    else:
        start = (existing.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = _clean(_download("^NSEI", start=start))
        df = pd.concat([existing, df]).sort_index()
        df = df[~df.index.duplicated(keep="last")]
    cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(months=3)
    return df.loc[df.index >= cutoff]
