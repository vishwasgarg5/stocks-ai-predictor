from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    STOCK_RELIABILITY_FILE,
    MIN_JUMP_CONFIDENCE,
)

from .utils import clamp


def load_reliability():
    if not STOCK_RELIABILITY_FILE.exists():
        return {}

    try:
        df = pd.read_csv(
            STOCK_RELIABILITY_FILE
        )

        result = {}

        for _, row in df.iterrows():
            symbol = str(row["Symbol"])

            direction = float(
                row.get(
                    "DirectionAccuracy",
                    50,
                )
            )

            mape = float(
                row.get(
                    "MAPE",
                    3,
                )
            )

            error_score = clamp(
                100 - (mape * 20)
            )

            score = (
                0.55 * error_score
                + 0.45 * direction
            )

            result[symbol] = score

        return result

    except Exception:
        return {}


def expected_return_score(value):
    value = float(value)

    # +5% maps approximately to 75.
    score = 50 + value * 5

    return clamp(score)


def regime_direction_score(direction, regime):
    if regime == "BULL":
        return {
            "UP": 90,
            "NEUTRAL": 55,
            "DOWN": 35,
        }.get(direction, 50)

    if regime == "BEAR":
        return {
            "UP": 35,
            "NEUTRAL": 55,
            "DOWN": 80,
        }.get(direction, 50)

    if regime == "HIGH VOL":
        return 45

    return {
        "UP": 70,
        "NEUTRAL": 55,
        "DOWN": 45,
    }.get(direction, 50)


def price_bucket(price):
    price = float(price)

    if price < 250:
        return "<₹250"

    if price < 750:
        return "₹250-750"

    if price < 1500:
        return "₹750-1500"

    if price < 3000:
        return "₹1500-3000"

    return "₹3000+"


def calculate_score(row, regime):
    technical = float(
        row.get("TechnicalScore", 50)
    )

    expected = expected_return_score(
        row.get("Expected_Return", 0)
    )

    confidence = float(
        row.get("Confidence", 50)
    )

    direction_conf = float(
        row.get(
            "Direction_Confidence",
            50,
        )
    )

    reliability = float(
        row.get("ReliabilityScore", 50)
    )

    regime_score = regime_direction_score(
        row.get("Direction", "NEUTRAL"),
        regime,
    )

    score = (
        0.25 * technical
        + 0.20 * expected
        + 0.20 * confidence
        + 0.15 * direction_conf
        + 0.10 * reliability
        + 0.10 * regime_score
    )

    return clamp(score)


def select_top_stocks(
    candidates,
    top_n=5,
    regime="SIDEWAYS",
):
    if candidates is None or candidates.empty:
        return pd.DataFrame()

    df = candidates.copy()

    df["ReliabilityScore"] = df[
        "Symbol"
    ].map(
        load_reliability()
    ).fillna(50)

    df["PriceBucket"] = df[
        "Current_Price"
    ].apply(price_bucket)

    df["Score"] = df.apply(
        lambda row: calculate_score(
            row,
            regime,
        ),
        axis=1,
    )

    df = df.sort_values(
        "Score",
        ascending=False,
    )

    selected = []

    # First attempt: price diversification.
    for bucket in [
        "<₹250",
        "₹250-750",
        "₹750-1500",
        "₹1500-3000",
        "₹3000+",
    ]:
        subset = df[
            df["PriceBucket"] == bucket
        ]

        if not subset.empty:
            selected.append(
                subset.iloc[0]
            )

        if len(selected) >= top_n:
            break

    selected_symbols = {
        row["Symbol"]
        for row in selected
    }

    # Fill remaining positions by score.
    for _, row in df.iterrows():
        if len(selected) >= top_n:
            break

        if row["Symbol"] in selected_symbols:
            continue

        selected.append(row)
        selected_symbols.add(row["Symbol"])

    if not selected:
        return pd.DataFrame()

    result = pd.DataFrame(selected)

    return result.sort_values(
        "Score",
        ascending=False,
    ).reset_index(drop=True)
