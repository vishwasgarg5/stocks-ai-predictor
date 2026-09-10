import importlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from .config import DATA_DIR, UNIVERSE_FILES, NIFTY_SYMBOL, MAX_UNIVERSE, HISTORY_PERIOD, MIN_AVG_TRADED_VALUE, MIN_PRICE, INDEX_SYMBOLS, INDEX_SYMBOL_FALLBACKS
from .utils import clean_ohlcv

NSE_EQUITY_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
OHLCV_CACHE_DIR = DATA_DIR / "stage2" / "ohlcv"
OHLCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_symbol(symbol):
    s = str(symbol).strip().upper()
    return s[:-3] if s.endswith(".NS") else s


def _ticker(symbol):
    s = str(symbol).strip().upper()
    if s.startswith("^") or s.endswith("=X") or s.endswith(".NS"):
        return s
    return f"{s}.NS"


def read_symbols_from_csv(path):
    try:
        df = pd.read_csv(path)
        column = next((c for c in ["SYMBOL", "Symbol", "symbol", "Ticker", "ticker"] if c in df.columns), None)
        if column is None:
            return []
        return list(dict.fromkeys(normalize_symbol(x) for x in df[column].dropna() if normalize_symbol(x) and normalize_symbol(x) != "SYMBOL"))
    except Exception:
        return []


def download_nse_equity_list():
    try:
        r = requests.get(NSE_EQUITY_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"})
        if r.status_code != 200:
            return []
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        column = next((c for c in ["SYMBOL", "Symbol", "symbol"] if c in df.columns), None)
        if column is None:
            return []
        return list(dict.fromkeys(normalize_symbol(x) for x in df[column].dropna() if normalize_symbol(x) and normalize_symbol(x) != "SYMBOL"))
    except Exception as exc:
        print(f"NSE universe download failed: {exc}")
        return []


def load_universe():
    symbols = download_nse_equity_list()
    if symbols:
        print(f"Using full NSE equity universe: {len(symbols)} stocks")
        return symbols
    for path in [Path(p) for p in UNIVERSE_FILES]:
        symbols = read_symbols_from_csv(path) if path.exists() else []
        if symbols:
            print(f"Using repository universe fallback: {len(symbols)} stocks")
            return symbols
    for module_name in ["src.nifty150_symbols", "src.nifty150", "src.market_universe"]:
        try:
            module = importlib.import_module(module_name)
            for attr in ["NIFTY150_SYMBOLS", "NIFTY_150_SYMBOLS", "SYMBOLS", "STOCKS"]:
                values = getattr(module, attr, None)
                if values:
                    return [normalize_symbol(x) for x in values]
        except Exception:
            continue
    raise RuntimeError("Unable to load NSE stock universe")


def _cache_path(symbol):
    return OHLCV_CACHE_DIR / f"{normalize_symbol(symbol)}.csv"


def _read_cached_ohlcv(symbol):
    path = _cache_path(symbol)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = clean_ohlcv(df)
        return df.sort_index() if not df.empty else pd.DataFrame()
    except Exception as exc:
        print(f"{symbol}: cached data unreadable: {exc}")
        return pd.DataFrame()


def _save_cached_ohlcv(symbol, df):
    if df is None or df.empty:
        return
    path = _cache_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = clean_ohlcv(df).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.to_csv(path)


def _download_range(ticker, start=None, end=None, period=None, retries=2):
    """Download one range with a stable public adapter contract.

    `retries` is deliberately handled here so callers cannot accidentally pass
    an unsupported keyword to yfinance. Programming errors are not swallowed.
    """
    symbol = str(ticker).strip().upper()
    yf_ticker = _ticker(symbol)
    kwargs = {"interval": "1d", "auto_adjust": False, "progress": False, "threads": False}
    if start is not None:
        kwargs["start"] = pd.Timestamp(start).strftime("%Y-%m-%d")
    if end is not None:
        kwargs["end"] = pd.Timestamp(end).strftime("%Y-%m-%d")
    if start is None and period is not None:
        kwargs["period"] = period
    last_error = None
    for attempt in range(max(0, int(retries)) + 1):
        try:
            data = yf.download(yf_ticker, **kwargs)
            cleaned = clean_ohlcv(data)
            if not cleaned.empty:
                return cleaned
            last_error = RuntimeError(f"empty response for {yf_ticker}")
        except (TypeError, AttributeError, KeyError):
            raise
        except Exception as exc:
            last_error = exc
        if attempt < max(0, int(retries)):
            time.sleep(min(2.0, 0.5 * (attempt + 1)))
    if last_error is not None:
        print(f"{symbol}: market data download failed after retries: {last_error}")
    return pd.DataFrame()


def download_symbol(symbol, period=HISTORY_PERIOD, retries=2):
    symbol = normalize_symbol(symbol)
    ticker = _ticker(symbol)
    cached = _read_cached_ohlcv(symbol)
    today = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
    desired_start = today - pd.DateOffset(years=1) if str(period).lower() == "1y" else None
    try:
        if desired_start is None:
            desired = _download_range(ticker, period=period, retries=retries)
            merged = pd.concat([cached, desired]).sort_index() if not cached.empty else desired
        elif cached.empty:
            merged = _download_range(ticker, period=period, retries=retries)
        else:
            cached_start = pd.Timestamp(cached.index.min())
            cached_end = pd.Timestamp(cached.index.max())
            frames = [cached]
            if cached_start > desired_start:
                frames.append(_download_range(ticker, start=desired_start, end=cached_start + pd.Timedelta(days=1), retries=retries))
            if cached_end < today:
                frames.append(_download_range(ticker, start=cached_end + pd.Timedelta(days=1), end=today + pd.Timedelta(days=1), retries=retries))
            merged = pd.concat([x for x in frames if x is not None and not x.empty]).sort_index()
        merged = clean_ohlcv(merged)
        merged = merged[~merged.index.duplicated(keep="last")]
        if not merged.empty:
            _save_cached_ohlcv(symbol, merged)
        if desired_start is not None and not merged.empty:
            merged = merged[merged.index >= desired_start]
        if len(merged) >= 30:
            return merged
        if retries > 0:
            fresh = _download_range(ticker, period=period, retries=retries)
            merged = pd.concat([cached, fresh]).sort_index() if not cached.empty else fresh
            merged = clean_ohlcv(merged)
            merged = merged[~merged.index.duplicated(keep="last")]
            if not merged.empty:
                _save_cached_ohlcv(symbol, merged)
            if desired_start is not None:
                merged = merged[merged.index >= desired_start]
            if len(merged) >= 30:
                return merged
    except (TypeError, AttributeError, KeyError):
        raise
    except Exception as exc:
        print(f"{symbol}: incremental data update failed: {exc}")
    return None


def download_many(symbols, period=HISTORY_PERIOD, workers=8):
    result = {}
    symbols = list(dict.fromkeys(normalize_symbol(s) for s in symbols))
    with ThreadPoolExecutor(max_workers=min(max(1, int(workers)), 8)) as executor:
        futures = {executor.submit(download_symbol, s, period): s for s in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    result[symbol] = df
            except (TypeError, AttributeError, KeyError) as exc:
                raise RuntimeError(f"Programming error while updating {symbol}: {exc}") from exc
            except Exception as exc:
                print(f"{symbol}: {exc}")
    print(f"Repository-cache data available for {len(result)}/{len(symbols)} stocks")
    return result


def liquidity_score(df):
    if df is None or len(df) < 20:
        return 0.0
    recent = df.tail(60)
    avg_value = float((recent["Close"] * recent["Volume"]).mean())
    price = float(recent["Close"].iloc[-1])
    if avg_value < MIN_AVG_TRADED_VALUE or price < MIN_PRICE:
        return 0.0
    return float(min(np.log10(max(avg_value, 1)) * 8, 100))


def filter_liquid_universe(data_map):
    return {s: df for s, df in data_map.items() if liquidity_score(df) > 0}


def apply_universe_cap(data_map):
    if not MAX_UNIVERSE or MAX_UNIVERSE <= 0:
        return data_map
    ranked = sorted(data_map.items(), key=lambda item: liquidity_score(item[1]), reverse=True)
    return dict(ranked[:int(MAX_UNIVERSE)])


def get_nifty_data(period="1y", symbol=None):
    try:
        return clean_ohlcv(yf.download(symbol or NIFTY_SYMBOL, period=period, interval="1d", auto_adjust=False, progress=False, threads=False))
    except Exception:
        return pd.DataFrame()


def _index_snapshot(symbol, period="3mo", cutoff=None):
    df = get_nifty_data(period, symbol)
    if cutoff is not None and not df.empty:
        df = df[df.index.date <= pd.Timestamp(cutoff).date()]
    if df.empty or "Close" not in df.columns:
        return {"Close": np.nan, "Change1D": np.nan, "Open": np.nan, "High": np.nan, "Low": np.nan, "High52W": np.nan, "Low52W": np.nan, "Source": symbol}
    close = float(df["Close"].iloc[-1]); prev = float(df["Close"].iloc[-2]) if len(df) > 1 else close
    return {"Close": close, "Change1D": (close / prev - 1) * 100 if prev else 0.0, "Open": float(df["Open"].iloc[-1]), "High": float(df["High"].iloc[-1]), "Low": float(df["Low"].iloc[-1]), "High52W": float(df["High"].tail(252).max()), "Low52W": float(df["Low"].tail(252).min()), "Source": symbol, "DataDate": str(pd.Timestamp(df.index[-1]).date())}


def _index_snapshot_with_fallback(key, cutoff=None):
    candidates = [INDEX_SYMBOLS.get(key)] + INDEX_SYMBOL_FALLBACKS.get(key, [])
    seen = set()
    for symbol in candidates:
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        snap = _index_snapshot(symbol, cutoff=cutoff)
        if np.isfinite(snap.get("Close", np.nan)):
            snap["IndexKey"] = key
            return snap
    return {"Close": np.nan, "Change1D": np.nan, "Open": np.nan, "High": np.nan, "Low": np.nan, "High52W": np.nan, "Low52W": np.nan, "Source": "unavailable", "IndexKey": key}


def get_market_snapshot(data_map=None, cutoff=None):
    snap = {"NIFTY": _index_snapshot_with_fallback("NIFTY", cutoff), "BANKNIFTY": _index_snapshot_with_fallback("BANK", cutoff), "FINNIFTY": _index_snapshot_with_fallback("FINN", cutoff), "MIDCPNIFTY": _index_snapshot_with_fallback("MIDCP", cutoff), "VIX": _index_snapshot_with_fallback("VIX", cutoff)}
    for key in ("NIFTYNEXT50", "NIFTY100", "NIFTY200", "NIFTY500", "MIDCAP150", "SMALLCAP250", "IT", "AUTO", "PHARMA", "METAL", "ENERGY", "REALTY", "PSUBANK"):
        snap[key] = _index_snapshot_with_fallback(key, cutoff)
    ups = downs = unchanged = 0
    cutoff_date = pd.Timestamp(cutoff).date() if cutoff is not None else None
    for df in (data_map or {}).values():
        if df is None or len(df) < 2:
            continue
        x = df[df.index.date <= cutoff_date] if cutoff_date is not None else df
        if len(x) < 2:
            continue
        a, b = float(x["Close"].iloc[-2]), float(x["Close"].iloc[-1]); ups += b > a; downs += b < a; unchanged += b == a
    total = ups + downs
    snap["Breadth"] = {"Advancers": ups, "Decliners": downs, "Unchanged": unchanged, "Ratio": ups / max(downs, 1), "Score": 100 * ups / max(total, 1) if total else 50}
    snap["RegimePrediction"] = get_market_regime(cutoff)
    return snap


def get_completed_session_date(mode="morning", reference_date=None):
    df = get_nifty_data("1mo")
    if df.empty:
        return None
    from .utils import today_ist
    reference = reference_date or today_ist()
    dates = sorted({pd.Timestamp(x).date() for x in df.index})
    valid = [x for x in dates if x < reference] if mode == "morning" else [x for x in dates if x <= reference]
    return max(valid) if valid else None


def get_data_cutoff_date(data_map, reference_date=None, fallback=None, min_fraction=0.50):
    from .utils import today_ist
    reference = pd.Timestamp(reference_date or today_ist()).date(); counts = {}; total = max(len(data_map), 1)
    for df in data_map.values():
        if df is None or df.empty:
            continue
        dates = {pd.Timestamp(x).date() for x in df.index if pd.Timestamp(x).date() < reference}
        for d in dates:
            counts[d] = counts.get(d, 0) + 1
    threshold = max(5, int(np.ceil(total * min_fraction))); valid = [d for d, c in counts.items() if c >= threshold]
    if valid:
        return max(valid)
    return fallback if fallback is not None and pd.Timestamp(fallback).date() < reference else None


def get_previous_session_date(session_date):
    df = get_nifty_data("3mo")
    if df.empty:
        return None
    dates = sorted({pd.Timestamp(x).date() for x in df.index}); previous = [x for x in dates if x < session_date]
    return max(previous) if previous else None


def get_market_regime(cutoff_date=None):
    df = get_nifty_data("1y")
    if df.empty:
        return {"name": "UNKNOWN", "score": 50, "confidence": 35}
    if cutoff_date is not None:
        df = df[df.index.date <= pd.Timestamp(cutoff_date).date()]
    if len(df) < 60:
        return {"name": "SIDEWAYS", "score": 50, "confidence": 35}
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < 60:
        return {"name": "SIDEWAYS", "score": 50, "confidence": 35}
    sma20 = close.rolling(20).mean().iloc[-1]; sma50 = close.rolling(50).mean().iloc[-1]; sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma50
    ret20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0.0; ret60 = (close.iloc[-1] / close.iloc[-61] - 1) * 100 if len(close) >= 61 else 0.0; vol = close.pct_change().rolling(20).std().iloc[-1] * 100
    current = close.iloc[-1]; score = 50.0
    score += 10 if current > sma20 else -10; score += 10 if current > sma50 else -10; score += 10 if current > sma200 else -10; score += 8 if ret20 > 2 else -8 if ret20 < -2 else 0; score += 7 if ret60 > 5 else -7 if ret60 < -5 else 0
    if vol > 2.5: score -= 8
    score = max(0, min(100, score)); name = "BULL" if score >= 65 else "BEAR" if score <= 40 else "SIDEWAYS"; confidence = max(35, min(95, 50 + abs(score - 50) * 1.2))
    return {"name": name, "score": score, "confidence": confidence, "SMA20": sma20, "SMA50": sma50, "SMA200": sma200, "Return20D": ret20, "Return60D": ret60, "Volatility20D": vol}


def get_row_for_date(df, target_date):
    if df is None or df.empty:
        return None
    target_date = pd.Timestamp(target_date).date()
    for index in df.index:
        if pd.Timestamp(index).date() == target_date:
            return df.loc[index]
    return None


def get_previous_row(df, target_date):
    if df is None or df.empty:
        return None
    target_date = pd.Timestamp(target_date).date(); rows = [(i, r) for i, r in df.iterrows() if pd.Timestamp(i).date() < target_date]
    return rows[-1][1] if rows else None
