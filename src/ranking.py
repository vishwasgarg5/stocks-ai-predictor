import numpy as np
import pandas as pd
from config import FEATURES, TOP_N


def score_latest(df: pd.DataFrame) -> float:
    r = df.iloc[-1]
    score = 50.0
    score += np.clip(r.get("rsi_14", 50) - 50, -15, 15) * 0.7
    score += np.clip(r.get("ret_5d", 0) * 100, -10, 10) * 1.0
    score += np.clip(r.get("relative_ret_5d", 0) * 100, -10, 10) * 1.2
    score += np.clip(r.get("ema_20_ratio", 0) * 100, -10, 10) * 0.8
    score += np.clip(r.get("ema_50_ratio", 0) * 100, -10, 10) * 0.6
    score += np.clip(r.get("volume_ratio", 1) - 1, -0.5, 1.0) * 8
    score += np.clip(r.get("adx_14", 20) - 20, -10, 20) * 0.2
    return float(np.clip(score, 0, 100))


def rank_stocks(feature_sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for symbol, df in feature_sets.items():
        if df.empty:
            continue
        latest = df.iloc[-1]
        if latest[FEATURES].isna().any():
            continue
        rows.append({"symbol": symbol, "score": score_latest(df)})
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out.head(TOP_N)
