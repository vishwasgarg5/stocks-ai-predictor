import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .config import (
    DATA_DIR,
    UNIVERSE_FILES,
    NIFTY_SYMBOL,
    MAX_UNIVERSE,
    HISTORY_PERIOD,
    MIN_AVG_TRADED_VALUE,
    MIN_PRICE,
)

from .utils import clean_ohlcv


NSE_EQUITY_URL = (
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
)


def normalize_symbol(symbol):
    symbol = str(symbol).strip().upper()

    if symbol.endswith(".NS"):
        symbol = symbol[:-3]

    return symbol


def read_symbols_from_csv(path):
    try:
        df = pd.read_csv(path)

        possible_columns = [
            "Symbol",
            "SYMBOL",
            "symbol",
            "Ticker",
            "ticker",
        ]

        column = None

        for candidate in possible_columns:
            if candidate in df.columns:
                column = candidate
                break

        if column is None:
            return []

        symbols = [
            normalize_symbol(x)
            for x in df[column].dropna().tolist()
        ]

        return [
            x for x in symbols
            if x and x != "SYMBOL"
        ]

    except Exception:
        return []


def discover_existing_universe():
    candidates = list(UNIVERSE_FILES)

    # Also discover any CSV containing "nifty" and "150".
    try:
        for path in DATA_DIR.rglob("*.csv"):
            name = path.name.lower()

            if "nifty" in name and "150" in name:
                candidates.append(path)
    except Exception:
        pass

    seen = set()

    for path in candidates:
        path = Path(path)

        if not path.exists():
            continue

        if str(path) in seen:
            continue

        seen.add(str(path))

        symbols = read_symbols_from_csv(path)

        if len(symbols) >= 100:
            return symbols[:MAX_UNIVERSE]

    return []


def discover_existing_python_universe():
    modules = [
        "src.nifty150_symbols",
        "src.nifty150",
        "src.market_universe",
    ]

    attributes = [
        "NIFTY150_SYMBOLS",
        "NIFTY_150_SYMBOLS",
        "SYMBOLS",
        "STOCKS",
    ]

    for module_name in modules:
        try:
            module = importlib.import_module(module_name)

            for attribute in attributes:
                values = getattr(module, attribute, None)

                if values and len(values) >= 100:
                    return [
                        normalize_symbol(x)
                        for x in list(values)[:MAX_UNIVERSE]
                    ]

        except Exception:
            continue

    return []


def download_nse_equity_list():
    try:
        response = requests.get(
            NSE_EQUITY_URL,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/csv,*/*",
            },
        )

        if response.status_code != 200:
            return []

        from io import StringIO

        df = pd.read_csv(StringIO(response.text))

        column = None

        for candidate in ["SYMBOL", "Symbol", "symbol"]:
            if candidate in df.columns:
                column = candidate
                break

        if column is None:
            return []

        symbols = [
            normalize_symbol(x)
            for x in df[column].dropna().tolist()
        ]

        return list(dict.fromkeys(symbols))

    except Exception:
        return []


def load_universe():
    """
    Priority:
    1. Existing Nifty-150 CSV.
    2. Existing Python Nifty-150 list.
    3. Broad NSE equity list.

    The final broad list is later filtered by liquidity.
    """

    symbols = discover_existing_universe()

    if symbols:
        print(
            f"Using existing Nifty universe: {len(symbols)} stocks"
        )
        return symbols

    symbols = discover_existing_python_universe()

    if symbols:
        print(
            f"Using existing Python universe: {len(symbols)} stocks"
        )
        return symbols

    symbols = download_nse_equity_list()

    if symbols:
        print(
            f"Using broad NSE universe: {len(symbols)} stocks"
        )
        return symbols[:MAX_UNIVERSE]

    raise RuntimeError(
        "Unable to load stock universe. "
        "Keep your existing Nifty-150 CSV/list in the repository."
    )


def download_symbol(symbol, period=HISTORY_PERIOD):
    ticker = f"{normalize_symbol(symbol)}.NS"

    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        df = clean_ohlcv(df)

        if len(df) < 30:
            return None

        return df

    except Exception as exc:
        print(f"{symbol}: data download failed: {exc}")
        return None


def download_many(symbols, period=HISTORY_PERIOD, workers=8):
    result = {}

    symbols = list(dict.fromkeys(symbols))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_symbol, symbol, period): symbol
            for symbol in symbols
        }

        for future in as_completed(futures):
            symbol = futures[future]

            try:
                df = future.result()

                if df is not None and not df.empty:
                    result[symbol] = df

            except Exception as exc:
                print(f"{symbol}: {exc}")

    print(
        f"Downloaded usable data for {len(result)}/{len(symbols)} stocks"
    )

    return result


def liquidity_score(df):
    if df is None or df.empty:
        return 0.0

    recent = df.tail(60).copy()

    if len(recent) < 20:
        return 0.0

    traded_value = recent["Close"] * recent["Volume"]

    avg_value = float(traded_value.mean())

    if avg_value < MIN_AVG_TRADED_VALUE:
        return 0.0

    price = float(recent["Close"].iloc[-1])

    if price < MIN_PRICE:
        return 0.0

    # Score grows slowly so huge companies don't dominate.
    score = np.log10(max(avg_value, 1)) * 8

    return float(min(score, 100))


def filter_liquid_universe(data_map):
    filtered = {}

    for symbol, df in data_map.items():
        score = liquidity_score(df)

        if score > 0:
            filtered[symbol] = df

    return filtered


def get_nifty_data(period="1y"):
    try:
        df = yf.download(
            NIFTY_SYMBOL,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        return clean_ohlcv(df)

    except Exception:
        return pd.DataFrame()


def get_completed_session_date(mode="morning", reference_date=None):
    df = get_nifty_data("1mo")

    if df.empty:
        return None

    reference = reference_date

    if reference is None:
        from .utils import today_ist
        reference = today_ist()

    dates = [x.date() for x in df.index]

    if mode == "morning":
        valid = [x for x in dates if x < reference]
    else:
        valid = [x for x in dates if x <= reference]

    if not valid:
        return None

    return max(valid)


def get_previous_session_date(session_date):
    df = get_nifty_data("3mo")

    if df.empty:
        return None

    dates = sorted(
        set(x.date() for x in df.index)
    )

    previous = [
        x for x in dates
        if x < session_date
    ]

    return max(previous) if previous else None


def get_market_regime(cutoff_date=None):
    df = get_nifty_data("1y")

    if df.empty:
        return {
            "name": "UNKNOWN",
            "score": 50,
        }

    if cutoff_date is not None:
        df = df[df.index.date <= cutoff_date]

    if len(df) < 60:
        return {
            "name": "NORMAL",
            "score": 50,
        }

    close = df["Close"]

    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]

    returns = close.pct_change()
    volatility = returns.rolling(20).std().iloc[-1]

    current = close.iloc[-1]

    if current > sma20 > sma50:
        regime = "BULL"

    elif current < sma20 < sma50:
        regime = "BEAR"

    else:
        regime = "SIDEWAYS"

    if volatility > 0.018:
        regime = "HIGH VOL"

    score = {
        "BULL": 80,
        "BEAR": 40,
        "SIDEWAYS": 60,
        "HIGH VOL": 45,
    }.get(regime, 50)

    return {
        "name": regime,
        "score": score,
    }


def get_row_for_date(df, target_date):
    if df is None or df.empty:
        return None

    target_date = pd.Timestamp(target_date).date()

    for index in df.index:
        if index.date() == target_date:
            return df.loc[index]

    return None


def get_previous_row(df, target_date):
    if df is None or df.empty:
        return None

    target_date = pd.Timestamp(target_date).date()

    rows = [
        (index, row)
        for index, row in df.iterrows()
        if index.date() < target_date
    ]

    if not rows:
        return None

    return rows[-1][1]
