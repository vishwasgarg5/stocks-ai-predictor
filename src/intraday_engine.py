import numpy as np
import pandas as pd
import yfinance as yf

from .config import (
    INTRADAY_PERIOD,
    INTRADAY_INTERVAL,
    INTRADAY_TOP_N,
    INTRADAY_MIN_ROWS,
    INTRADAY_MIN_MOVE,
)

from .utils import clean_ohlcv


def download_intraday(symbol):
    ticker = f"{symbol}.NS"

    try:
        df = yf.download(
            ticker,
            period=INTRADAY_PERIOD,
            interval=INTRADAY_INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        return clean_ohlcv(df)

    except Exception as exc:
        print(
            f"{symbol}: intraday data failed: {exc}"
        )

        return pd.DataFrame()


def add_intraday_features(df):
    x = df.copy()

    close = x["Close"]
    high = x["High"]
    low = x["Low"]
    volume = x["Volume"]

    x["EMA9"] = close.ewm(
        span=9,
        adjust=False,
    ).mean()

    x["EMA20"] = close.ewm(
        span=20,
        adjust=False,
    ).mean()

    typical_price = (
        high + low + close
    ) / 3

    cumulative_volume = volume.cumsum()

    x["VWAP"] = (
        typical_price * volume
    ).cumsum() / cumulative_volume.replace(
        0,
        np.nan,
    )

    x["VolumeMA20"] = volume.rolling(20).mean()

    x["RelativeVolume"] = (
        volume
        / x["VolumeMA20"].replace(0, np.nan)
    )

    x["Return1"] = close.pct_change()

    x["Return4"] = close.pct_change(4)

    x["Range"] = (
        high - low
    ) / close.replace(0, np.nan)

    x["Momentum"] = (
        close / close.shift(8) - 1
    )

    x["DayHigh"] = (
        x.groupby(x.index.date)["High"]
        .transform("max")
    )

    x["DayLow"] = (
        x.groupby(x.index.date)["Low"]
        .transform("min")
    )

    return x.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def calculate_intraday_setup(df):
    if df.empty:
        return None

    x = add_intraday_features(df)

    x = x.dropna()

    if len(x) < INTRADAY_MIN_ROWS:
        return None

    row = x.iloc[-1]

    current = float(row["Close"])
    vwap = float(row["VWAP"])
    ema9 = float(row["EMA9"])
    ema20 = float(row["EMA20"])

    relative_volume = float(
        row["RelativeVolume"]
    )

    momentum = float(
        row["Momentum"]
    )

    range_pct = float(
        row["Range"]
    )

    score = 50.0

    if current > vwap:
        score += 12
    else:
        score -= 12

    if ema9 > ema20:
        score += 10
    else:
        score -= 8

    if relative_volume > 1.5:
        score += 12
    elif relative_volume > 1.1:
        score += 5

    if momentum > 0.01:
        score += 10
    elif momentum < -0.01:
        score -= 10

    if range_pct > 0.015:
        score += 6

    score = float(
        np.clip(score, 0, 100)
    )

    if score >= 60:
        bias = "UP"

    elif score <= 40:
        bias = "DOWN"

    else:
        bias = "NEUTRAL"

    expected_move = max(
        abs(momentum),
        range_pct * 1.5,
        INTRADAY_MIN_MOVE,
    )

    if bias == "UP":
        target = current * (
            1 + expected_move
        )

        stop_loss = current * (
            1 - expected_move * 0.55
        )

    elif bias == "DOWN":
        target = current * (
            1 - expected_move
        )

        stop_loss = current * (
            1 + expected_move * 0.55
        )

    else:
        target = current
        stop_loss = current

    confidence = min(
        95,
        50
        + abs(score - 50)
        * 0.7
        + min(relative_volume, 3) * 5,
    )

    return {
        "Current": current,
        "Bias": bias,
        "Target": target,
        "StopLoss": stop_loss,
        "ExpectedMove": expected_move * 100,
        "Score": score,
        "Confidence": confidence,
        "RelativeVolume": relative_volume,
        "VWAP": vwap,
        "EMA9": ema9,
        "EMA20": ema20,
    }


def generate_intraday_watchlist(
    symbols,
    max_workers=6,
):
    from concurrent.futures import (
        ThreadPoolExecutor,
        as_completed,
    )

    data = {}

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = {
            executor.submit(
                download_intraday,
                symbol,
            ): symbol
            for symbol in symbols
        }

        for future in as_completed(futures):
            symbol = futures[future]

            try:
                df = future.result()

                if not df.empty:
                    data[symbol] = df

            except Exception as exc:
                print(
                    f"{symbol}: {exc}"
                )

    results = []

    for symbol, df in data.items():
        try:
            setup = calculate_intraday_setup(df)

            if setup is None:
                continue

            if setup["Bias"] == "NEUTRAL":
                continue

            results.append(
                {
                    "Symbol": symbol,
                    **setup,
                }
            )

        except Exception as exc:
            print(
                f"{symbol}: setup failed: {exc}"
            )

    if not results:
        return pd.DataFrame()

    result = pd.DataFrame(results)

    return result.sort_values(
        [
            "Score",
            "Confidence",
        ],
        ascending=False,
    ).head(
        INTRADAY_TOP_N
    ).reset_index(drop=True)
