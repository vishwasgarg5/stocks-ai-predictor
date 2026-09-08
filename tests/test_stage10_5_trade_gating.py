import pandas as pd

from src.selection import select_top_stocks


def _row(symbol, expected, multi, direction="UP"):
    return {
        "Symbol": symbol,
        "PriceBucket": "100-249",
        "Expected_Return": expected,
        "MultiHorizonExpectedReturn": multi,
        "Direction": direction,
        "Confidence": 90,
        "Direction_Confidence": 90,
        "TechnicalScore": 90,
        "SectorScore": 80,
        "UncertaintyScore": 90,
    }


def test_negative_return_never_qualifies():
    df = pd.DataFrame([_row("BAD", -2, -3), _row("GOOD", 4, 3)])
    out = select_top_stocks(df, top_n=6, regime="BULL", min_score=50, min_confidence=50, min_trade_confidence=50)
    assert "BAD" not in set(out["Symbol"])
    assert "GOOD" in set(out["Symbol"])


def test_down_direction_is_watchlist_only():
    df = pd.DataFrame([_row("DOWN", 8, 5, "DOWN")])
    out = select_top_stocks(df, top_n=6, regime="SIDEWAYS", min_score=50, min_confidence=50, min_trade_confidence=50)
    assert out.empty
