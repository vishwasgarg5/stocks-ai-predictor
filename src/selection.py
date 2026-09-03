"""Stage 4.2 final stock selection: price bucket + sector + multi-horizon quality."""
import pandas as pd
from .config import STOCK_RELIABILITY_FILE
from .utils import clamp


def load_reliability():
    if not STOCK_RELIABILITY_FILE.exists():
        return {}
    try:
        df = pd.read_csv(STOCK_RELIABILITY_FILE)
        out = {}
        for _, r in df.iterrows():
            mape = float(r.get("MAPE", 3) or 3)
            direction = float(r.get("DirectionAccuracy", 50) or 50)
            samples = float(r.get("Samples", 0) or 0)
            # Reliability improves only when there is enough historical evidence.
            evidence = min(samples / 20.0, 1.0)
            raw = 0.55 * clamp(100 - mape * 20) + 0.45 * direction
            out[str(r["Symbol"])] = 50 + evidence * (raw - 50)
        return out
    except Exception:
        return {}


def expected_return_score(v):
    return clamp(50 + float(v) * 5)


def multi_horizon_score(v):
    """Convert average multi-horizon expected return into a stable 0-100 score."""
    try:
        return clamp(50 + float(v) * 4)
    except Exception:
        return 50.0


def regime_direction_score(d, regime):
    if regime == "BULL":
        return {"UP": 90, "NEUTRAL": 55, "DOWN": 35}.get(d, 50)
    if regime == "BEAR":
        return {"UP": 35, "NEUTRAL": 55, "DOWN": 80}.get(d, 50)
    if regime == "HIGH VOL":
        return 45
    return {"UP": 70, "NEUTRAL": 55, "DOWN": 45}.get(d, 50)


def calculate_score(row, regime):
    # Stage 4.2 weights: multi-horizon is now a real ranking component.
    return clamp(
        0.18 * float(row.get("TechnicalScore", 50))
        + 0.15 * expected_return_score(row.get("Expected_Return", 0))
        + 0.15 * float(row.get("Confidence", 50))
        + 0.12 * float(row.get("Direction_Confidence", 50))
        + 0.08 * float(row.get("ReliabilityScore", 50))
        + 0.10 * regime_direction_score(row.get("Direction", "NEUTRAL"), regime)
        + 0.10 * float(row.get("SectorScore", 50))
        + 0.12 * multi_horizon_score(row.get("MultiHorizonExpectedReturn", 0))
    )


def score_candidates(candidates, regime="SIDEWAYS"):
    if candidates is None or candidates.empty:
        return pd.DataFrame()
    df = candidates.copy()
    reliability = load_reliability()
    df["ReliabilityScore"] = df["Symbol"].map(reliability).fillna(50.0)
    df["Score"] = df.apply(lambda r: calculate_score(r, regime), axis=1)
    return df.sort_values(["Score", "Confidence", "Direction_Confidence", "SectorScore"], ascending=False).reset_index(drop=True)


def select_top_stocks(candidates, top_n=5, regime="SIDEWAYS", min_score=65.0, min_confidence=60.0):
    """Return only quality-qualified stocks; never pad the result to top_n."""
    scored = score_candidates(candidates, regime)
    if scored.empty:
        return scored
    qualified = scored[(scored["Score"] >= min_score) & (scored["Confidence"] >= min_confidence)]
    if qualified.empty:
        # In an unusually weak market, still return the single best model result
        # rather than fabricating a five-stock list.
        return scored.head(1).reset_index(drop=True)
    return qualified.head(top_n).reset_index(drop=True)
