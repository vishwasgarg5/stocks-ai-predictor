import math
import yfinance as yf


def _num(info, key):
    value = info.get(key)
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def get_fundamentals(symbol: str) -> dict:
    """Fetch latest point-in-time fundamentals available from Yahoo Finance.

    These are used for ranking only in v1. Missing values receive neutral scores.
    """
    try:
        info = yf.Ticker(f"{symbol}.NS").info
        return {
            "roe": _num(info, "returnOnEquity"),
            "revenue_growth": _num(info, "revenueGrowth"),
            "earnings_growth": _num(info, "earningsGrowth"),
            "debt_to_equity": _num(info, "debtToEquity"),
            "pe": _num(info, "trailingPE"),
        }
    except Exception:
        return {"roe": None, "revenue_growth": None, "earnings_growth": None,
                "debt_to_equity": None, "pe": None}


def fundamental_score(f: dict) -> float:
    score = 50.0
    if f.get("roe") is not None:
        score += max(-10, min(15, f["roe"] * 100 * 0.25))
    if f.get("revenue_growth") is not None:
        score += max(-8, min(10, f["revenue_growth"] * 100 * 0.30))
    if f.get("earnings_growth") is not None:
        score += max(-8, min(10, f["earnings_growth"] * 100 * 0.25))
    if f.get("debt_to_equity") is not None:
        score += -min(8, max(0, f["debt_to_equity"] / 100 * 2))
    if f.get("pe") is not None and f["pe"] > 0:
        score += max(-6, min(6, 25 - f["pe"])) * 0.2
    return float(max(0, min(100, score)))
