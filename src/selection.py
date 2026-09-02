"""
STAGE 4 FINAL STOCK SELECTION
==============================
Easy access: this file contains the final ranking weights.
Stage 4 adds Sector Score while retaining Stage 2 ML confidence,
reliability and market-regime logic. Price-bucket filtering is performed
before final Top-5 selection by morning_runner.py.
"""
import pandas as pd
from .config import STOCK_RELIABILITY_FILE
from .utils import clamp


def load_reliability():
    """Load historical stock reliability from GitHub-persisted CSV."""
    if not STOCK_RELIABILITY_FILE.exists():
        return {}
    try:
        df = pd.read_csv(STOCK_RELIABILITY_FILE)
        out = {}
        for _, r in df.iterrows():
            mape = float(r.get("MAPE", 3) or 3)
            direction = float(r.get("DirectionAccuracy", 50) or 50)
            out[str(r["Symbol"])] = 0.55 * clamp(100 - mape * 20) + 0.45 * direction
        return out
    except Exception:
        return {}


def expected_return_score(v):
    return clamp(50 + float(v) * 5)


def regime_direction_score(d, regime):
    if regime == "BULL":
        return {"UP": 90, "NEUTRAL": 55, "DOWN": 35}.get(d, 50)
    if regime == "BEAR":
        return {"UP": 35, "NEUTRAL": 55, "DOWN": 80}.get(d, 50)
    if regime == "HIGH VOL":
        return 45
    return {"UP": 70, "NEUTRAL": 55, "DOWN": 45}.get(d, 50)


def calculate_score(row, regime):
    """
    Stage 4 ranking formula — all components are normalized to 0-100.

    Weights (must total 100%):
      Technical Score       20%
      Expected Return       18%
      Model Confidence      18%
      Direction Confidence  14%
      Reliability           10%
      Market Regime         10%
      Sector Strength       10%
    """
    return clamp(
        0.20 * float(row.get("TechnicalScore", 50))
        + 0.18 * expected_return_score(row.get("Expected_Return", 0))
        + 0.18 * float(row.get("Confidence", 50))
        + 0.14 * float(row.get("Direction_Confidence", 50))
        + 0.10 * float(row.get("ReliabilityScore", 50))
        + 0.10 * regime_direction_score(row.get("Direction", "NEUTRAL"), regime)
        + 0.10 * float(row.get("SectorScore", 50))
    )


def score_candidates(candidates, regime="SIDEWAYS"):
    """Score every candidate without truncating; used by Stage 4 bucket selection."""
    if candidates is None or candidates.empty:
        return pd.DataFrame()
    df = candidates.copy()
    reliability = load_reliability()
    df["ReliabilityScore"] = df["Symbol"].map(reliability).fillna(50.0)
    df["Score"] = df.apply(lambda r: calculate_score(r, regime), axis=1)
    return df.sort_values(
        ["Score", "Confidence", "Direction_Confidence", "SectorScore"],
        ascending=False,
    ).reset_index(drop=True)


def select_top_stocks(candidates, top_n=5, regime="SIDEWAYS"):
    """Rank candidates and return the strongest final Top-N stocks."""
    return score_candidates(candidates, regime).head(top_n).reset_index(drop=True)
