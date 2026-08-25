import numpy as np
import pandas as pd
from config import FEATURES


def add_features(df: pd.DataFrame, nifty: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    close = x["Close"]
    high, low, volume = x["High"], x["Low"], x["Volume"]

    for n in (1, 3, 5, 10, 20):
        x[f"ret_{n}d"] = close.pct_change(n)

    x["ema_20_ratio"] = close / close.ewm(span=20, adjust=False).mean() - 1
    x["ema_50_ratio"] = close / close.ewm(span=50, adjust=False).mean() - 1
    x["sma_20_ratio"] = close / close.rolling(20).mean() - 1

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    x["macd"] = ema12 - ema26
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()

    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    x["atr_pct"] = atr / close

    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    x["adx_14"] = dx.rolling(14).mean()

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    x["bb_width"] = (4 * std) / mid.replace(0, np.nan)
    x["volume_ratio"] = volume / volume.rolling(20).mean()

    nret = nifty["Close"].pct_change()
    x["nifty_ret_1d"] = nret.reindex(x.index).ffill()
    x["nifty_ret_5d"] = nret.rolling(5).sum().reindex(x.index).ffill()
    x["relative_ret_5d"] = x["ret_5d"] - x["nifty_ret_5d"]
    n20 = nret.rolling(20).sum().reindex(x.index).ffill()
    x["relative_ret_20d"] = x["ret_20d"] - n20

    # Next-day targets. Shifted before model fitting to prevent look-ahead leakage.
    x["target_open"] = x["Open"].shift(-1) / close - 1
    x["target_high"] = x["High"].shift(-1) / close - 1
    x["target_low"] = x["Low"].shift(-1) / close - 1
    x["target_close"] = x["Close"].shift(-1) / close - 1
    return x.replace([np.inf, -np.inf], np.nan)


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=FEATURES + ["target_open", "target_high", "target_low", "target_close"]).copy()
