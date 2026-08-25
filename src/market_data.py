import time
from typing import Dict
import pandas as pd
import yfinance as yf
from config import INTERVAL, LOOKBACK_PERIOD, MIN_ROWS, NIFTY50


def ticker(symbol: str) -> str:
    return f"{symbol}.NS"


def download_symbol(symbol: str, period: str = LOOKBACK_PERIOD) -> pd.DataFrame:
    df = yf.download(ticker(symbol), period=period, interval=INTERVAL,
                     auto_adjust=False, progress=False, threads=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="last")]
    return df if len(df) >= MIN_ROWS else pd.DataFrame()


def download_universe(symbols=NIFTY50, period: str = LOOKBACK_PERIOD) -> Dict[str, pd.DataFrame]:
    result = {}
    for i, symbol in enumerate(symbols, 1):
        try:
            df = download_symbol(symbol, period)
            if not df.empty:
                result[symbol] = df
            print(f"[{i:02d}/{len(symbols)}] {symbol}: {len(df)} rows")
        except Exception as exc:
            print(f"{symbol}: ERROR {exc}")
        time.sleep(0.15)
    return result


def download_nifty(period: str = LOOKBACK_PERIOD) -> pd.DataFrame:
    df = yf.download("^NSEI", period=period, interval=INTERVAL,
                     auto_adjust=False, progress=False, threads=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])
