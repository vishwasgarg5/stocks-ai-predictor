import numpy as np
import pandas as pd

from src.retraining import STAGE15_FEATURES, TARGETS
from src.stage15 import direction_confidence


def predict_next(symbol: str, raw_df: pd.DataFrame, feature_df: pd.DataFrame, models: dict) -> dict:
    latest = feature_df.iloc[-1]
    x = latest[STAGE15_FEATURES].to_frame().T
    if x.isna().any().any():
        raise ValueError(f"Insufficient Stage 1.5 features for {symbol}")
    base_close = float(raw_df["Close"].iloc[-1])
    out = {"symbol": symbol, "base_close": base_close}
    for target in TARGETS:
        ret = float(models[target].predict(x)[0])
        ret = float(np.clip(ret, -0.15, 0.15))
        out[target.replace("target_", "pred_")] = base_close * (1 + ret)
    predicted_return = out["pred_close"] / base_close - 1
    volatility = float(latest.get("volatility_10d", 0.02))
    direction, confidence = direction_confidence(models.get("direction_model"), x, predicted_return, volatility)
    regime = "HIGH VOL" if float(latest.get("regime_high_vol", 0)) else ("BULL" if float(latest.get("regime_bull", 0)) else ("BEAR" if float(latest.get("regime_bear", 0)) else "SIDEWAYS"))
    out["predicted_direction"] = direction
    out["confidence"] = confidence
    out["regime"] = regime
    out["expected_return_pct"] = predicted_return * 100
    return out
