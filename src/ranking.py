import numpy as np
import pandas as pd
from src.fundamentals import fundamental_score
from config import TOP_N


def technical_score(df: pd.DataFrame) -> float:
    r = df.iloc[-1]
    score = 50.0
    score += np.clip(r.get("rsi_14", 50) - 50, -15, 15) * 0.55
    score += np.clip(r.get("rsi_7", 50) - 50, -10, 10) * 0.25
    score += np.clip(r.get("ret_5d", 0) * 100, -10, 10) * 0.9
    score += np.clip(r.get("ret_20d", 0) * 100, -12, 12) * 0.35
    score += np.clip(r.get("relative_ret_5d", 0) * 100, -10, 10) * 1.1
    score += np.clip(r.get("relative_ret_20d", 0) * 100, -12, 12) * 0.5
    score += np.clip(r.get("ema_20_ratio", 0) * 100, -10, 10) * 0.65
    score += np.clip(r.get("ema_50_ratio", 0) * 100, -10, 10) * 0.45
    score += np.clip(r.get("ema_20_slope", 0) * 100, -5, 5) * 0.5
    score += np.clip(r.get("macd_hist_pct", 0) * 1000, -5, 5) * 0.5
    score += np.clip((r.get("stoch_k", 50) - 50) / 10, -5, 5) * 0.25
    score += np.clip(r.get("volume_ratio_20d", 1) - 1, -0.5, 1.0) * 7
    score += np.clip(r.get("nifty_trend", 0) * 100, -8, 8) * 0.25
    score += np.clip(r.get("relative_vol_ratio", 1) - 1, -0.5, 1.0) * 2
    return float(np.clip(score, 0, 100))


def rank_stocks(feature_sets: dict[str, pd.DataFrame], fundamentals: dict[str, dict] | None = None) -> pd.DataFrame:
    fundamentals = fundamentals or {}
    rows = []
    for symbol, df in feature_sets.items():
        if df.empty:
            continue
        required = ["ret_5d", "relative_ret_5d", "ema_20_ratio", "rsi_14", "nifty_trend"]
        if df.iloc[-1][required].isna().any():
            continue
        t = technical_score(df)
        f = fundamental_score(fundamentals.get(symbol, {}))
        final = 0.70 * t + 0.30 * f
        rows.append({"symbol": symbol, "technical_score": t, "fundamental_score": f, "score": final})
    if not rows:
        return pd.DataFrame(columns=["symbol", "technical_score", "fundamental_score", "score", "rank"])
    out = pd.DataFrame(rows).sort_values(["score", "symbol"], ascending=[False, True]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out.head(TOP_N)
