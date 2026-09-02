import numpy as np
import pandas as pd

from .utils import clean_ohlcv


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period).mean()


def adx(df, period=14):
    high = df["High"]
    low = df["Low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0),
        0.0,
    )

    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0),
        0.0,
    )

    tr1 = high - low
    tr2 = (high - df["Close"].shift(1)).abs()
    tr3 = (low - df["Close"].shift(1)).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_value = tr.rolling(period).sum()

    plus_di = (
        100
        * plus_dm.rolling(period).sum()
        / atr_value.replace(0, np.nan)
    )

    minus_di = (
        100
        * minus_dm.rolling(period).sum()
        / atr_value.replace(0, np.nan)
    )

    denominator = (plus_di + minus_di).replace(0, np.nan)

    dx = (
        100
        * (plus_di - minus_di).abs()
        / denominator
    )

    return dx.rolling(period).mean()


def macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()

    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()

    return line, signal, line - signal


def build_features(df):
    df = clean_ohlcv(df)

    if df.empty:
        return pd.DataFrame()

    x = df.copy()

    close = x["Close"]
    volume = x["Volume"]

    # Moving averages
    x["SMA10"] = close.rolling(10).mean()
    x["SMA20"] = close.rolling(20).mean()
    x["SMA50"] = close.rolling(50).mean()
    x["EMA9"] = close.ewm(span=9, adjust=False).mean()
    x["EMA20"] = close.ewm(span=20, adjust=False).mean()

    # RSI
    x["RSI14"] = rsi(close)

    # MACD
    (
        x["MACD"],
        x["MACD_Signal"],
        x["MACD_Hist"],
    ) = macd(close)

    # Bollinger
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    x["BB_Mid"] = bb_mid
    x["BB_Upper"] = bb_mid + 2 * bb_std
    x["BB_Lower"] = bb_mid - 2 * bb_std

    x["BB_Position"] = (
        (close - x["BB_Lower"])
        / (x["BB_Upper"] - x["BB_Lower"]).replace(0, np.nan)
    )

    # ATR / ADX
    x["ATR14"] = atr(x)
    x["ATR_Pct"] = x["ATR14"] / close.replace(0, np.nan)

    x["ADX14"] = adx(x)

    # Returns
    x["Return1"] = close.pct_change(1)
    x["Return3"] = close.pct_change(3)
    x["Return5"] = close.pct_change(5)
    x["Return10"] = close.pct_change(10)
    x["Return20"] = close.pct_change(20)

    # Volatility
    x["Volatility10"] = x["Return1"].rolling(10).std()
    x["Volatility20"] = x["Return1"].rolling(20).std()

    # Volume
    volume_ma20 = volume.rolling(20).mean()

    x["Volume_Ratio"] = (
        volume / volume_ma20.replace(0, np.nan)
    )

    # OBV
    direction = np.sign(close.diff()).fillna(0)

    x["OBV"] = (
        direction * volume
    ).cumsum()

    x["OBV_Change"] = x["OBV"].pct_change(10)

    # Price structure
    x["Close_vs_SMA20"] = (
        close / x["SMA20"] - 1
    )

    x["Close_vs_SMA50"] = (
        close / x["SMA50"] - 1
    )

    x["EMA9_vs_EMA20"] = (
        x["EMA9"] / x["EMA20"] - 1
    )

    x["High_Low_Range"] = (
        (x["High"] - x["Low"])
        / close.replace(0, np.nan)
    )

    # Lag features
    for lag in [1, 2, 3, 5]:
        x[f"Close_Lag{lag}"] = close.shift(lag)
        x[f"Return_Lag{lag}"] = x["Return1"].shift(lag)

    # Candle structure
    x["Body_Pct"] = (
        (x["Close"] - x["Open"])
        / x["Open"].replace(0, np.nan)
    )

    x["Upper_Wick"] = (
        x["High"]
        - x[["Open", "Close"]].max(axis=1)
    ) / close.replace(0, np.nan)

    x["Lower_Wick"] = (
        x[["Open", "Close"]].min(axis=1)
        - x["Low"]
    ) / close.replace(0, np.nan)

    x = x.replace([np.inf, -np.inf], np.nan)

    return x


def get_feature_columns():
    return [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA10",
        "SMA20",
        "SMA50",
        "EMA9",
        "EMA20",
        "RSI14",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "BB_Mid",
        "BB_Upper",
        "BB_Lower",
        "BB_Position",
        "ATR14",
        "ATR_Pct",
        "ADX14",
        "Return1",
        "Return3",
        "Return5",
        "Return10",
        "Return20",
        "Volatility10",
        "Volatility20",
        "Volume_Ratio",
        "OBV",
        "OBV_Change",
        "Close_vs_SMA20",
        "Close_vs_SMA50",
        "EMA9_vs_EMA20",
        "High_Low_Range",
        "Close_Lag1",
        "Close_Lag2",
        "Close_Lag3",
        "Close_Lag5",
        "Return_Lag1",
        "Return_Lag2",
        "Return_Lag3",
        "Return_Lag5",
        "Body_Pct",
        "Upper_Wick",
        "Lower_Wick",
    ]


def prepare_supervised(df, cutoff_date=None):
    x = build_features(df)

    if x.empty:
        return pd.DataFrame()

    if cutoff_date is not None:
        cutoff = pd.Timestamp(cutoff_date)
        x = x[x.index <= cutoff]

    # Critical Stage 2 correction:
    # Features at T predict OHLC at T+1.
    x["Target_Open"] = x["Open"].shift(-1)
    x["Target_High"] = x["High"].shift(-1)
    x["Target_Low"] = x["Low"].shift(-1)
    x["Target_Close"] = x["Close"].shift(-1)

    future_return = (
        x["Target_Close"] / x["Close"]
    ) - 1

    x["Target_Return"] = future_return

    x["Direction"] = np.select(
        [
            future_return > 0.002,
            future_return < -0.002,
        ],
        [
            2,
            0,
        ],
        default=1,
    )

    columns = get_feature_columns() + [
        "Target_Open",
        "Target_High",
        "Target_Low",
        "Target_Close",
        "Target_Return",
        "Direction",
    ]

    x = x[columns]

    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna()

    return x


def technical_score(df):
    x = build_features(df)

    if x.empty:
        return 0.0

    row = x.dropna().iloc[-1]

    score = 50.0

    rsi_value = row.get("RSI14", 50)

    if 50 <= rsi_value <= 70:
        score += 10
    elif 45 <= rsi_value < 50:
        score += 4
    elif rsi_value > 75:
        score -= 8
    elif rsi_value < 30:
        score -= 5

    if row.get("MACD_Hist", 0) > 0:
        score += 10

    if row.get("Close_vs_SMA20", 0) > 0:
        score += 8

    if row.get("Close_vs_SMA50", 0) > 0:
        score += 8

    if row.get("EMA9_vs_EMA20", 0) > 0:
        score += 6

    volume_ratio = row.get("Volume_Ratio", 1)

    if volume_ratio > 1.2:
        score += 8
    elif volume_ratio > 1:
        score += 3

    adx_value = row.get("ADX14", 20)

    if adx_value > 25:
        score += 5

    return float(np.clip(score, 0, 100))
