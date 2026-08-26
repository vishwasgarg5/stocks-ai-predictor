import numpy as np
import pandas as pd
from config import FEATURES


def _rsi(close, n):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(df: pd.DataFrame, nifty: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    close = pd.to_numeric(x["Close"], errors="coerce")
    high = pd.to_numeric(x["High"], errors="coerce")
    low = pd.to_numeric(x["Low"], errors="coerce")
    volume = pd.to_numeric(x["Volume"], errors="coerce")

    for n in (1, 3, 5, 10, 20):
        x[f"ret_{n}d"] = close.pct_change(n)
    for n in (5, 10, 20, 50):
        ema = close.ewm(span=n, adjust=False).mean()
        x[f"ema_{n}_ratio"] = close / ema - 1
    x["ema_5_slope"] = close.ewm(span=5, adjust=False).mean().pct_change(3)
    x["ema_20_slope"] = close.ewm(span=20, adjust=False).mean().pct_change(5)
    x["sma_20_ratio"] = close / close.rolling(20).mean() - 1
    x["sma_50_ratio"] = close / close.rolling(50).mean() - 1

    x["rsi_7"] = _rsi(close, 7)
    x["rsi_14"] = _rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    x["macd_pct"] = macd / close
    x["macd_signal_pct"] = signal / close
    x["macd_hist_pct"] = (macd - signal) / close
    # Backward-compatible aliases for the previous Champion feature schema.
    x["macd"] = macd
    x["macd_signal"] = signal

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    x["atr_pct"] = atr / close
    x["range_pct"] = (high - low) / close
    x["gap_pct"] = x["Open"] / prev_close - 1
    daily_ret = close.pct_change()
    x["volatility_5d"] = daily_ret.rolling(5).std()
    x["volatility_10d"] = daily_ret.rolling(10).std()
    x["volatility_20d"] = daily_ret.rolling(20).std()

    # ADX retained for backward compatibility and ranking.
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    x["adx_14"] = dx.rolling(14).mean()

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper, lower = mid + 2 * std, mid - 2 * std
    x["bb_width"] = (upper - lower) / mid.replace(0, np.nan)
    x["bb_position"] = (close - lower) / (upper - lower).replace(0, np.nan)
    x["volume_ratio_5d"] = volume / volume.rolling(5).mean()
    x["volume_ratio_20d"] = volume / volume.rolling(20).mean()
    x["volume_ratio"] = x["volume_ratio_20d"]
    x["roc_5d"] = close.pct_change(5)
    x["roc_10d"] = close.pct_change(10)

    low14, high14 = low.rolling(14).min(), high.rolling(14).max()
    x["stoch_k"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    x["stoch_d"] = x["stoch_k"].rolling(3).mean()
    typical = (high + low + close) / 3
    money_flow = typical * volume
    direction = np.sign(typical.diff()).fillna(0)
    pos = money_flow.where(direction > 0, 0).rolling(14).sum()
    neg = money_flow.where(direction < 0, 0).rolling(14).sum().abs()
    x["mfi_14"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))

    nclose = pd.to_numeric(nifty["Close"], errors="coerce")
    nret = nclose.pct_change()
    for n in (1, 5, 20):
        x[f"nifty_ret_{n}d"] = ((nclose / nclose.shift(n)) - 1).reindex(x.index).ffill()
    x["nifty_volatility_10d"] = nret.rolling(10).std().reindex(x.index).ffill()
    nma20 = nclose.rolling(20).mean()
    x["nifty_trend"] = (nclose / nma20 - 1).reindex(x.index).ffill()
    x["relative_ret_5d"] = x["ret_5d"] - x["nifty_ret_5d"]
    x["relative_ret_20d"] = x["ret_20d"] - x["nifty_ret_20d"]
    x["relative_vol_ratio"] = daily_ret.rolling(10).std() / x["nifty_volatility_10d"].replace(0, np.nan)

    x["target_open"] = x["Open"].shift(-1) / close - 1
    x["target_high"] = x["High"].shift(-1) / close - 1
    x["target_low"] = x["Low"].shift(-1) / close - 1
    x["target_close"] = x["Close"].shift(-1) / close - 1
    return x.replace([np.inf, -np.inf], np.nan)


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=FEATURES + ["target_open", "target_high", "target_low", "target_close"]).copy()
