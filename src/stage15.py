"""Stage 1.5 accuracy layer: regime, breadth, walk-forward targets and confidence."""
from __future__ import annotations
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.retraining import STAGE1_FEATURES, TARGETS

STAGE15_FEATURES = STAGE1_FEATURES + [
    "market_breadth",
    "market_return_1d",
    "market_volatility_10d",
    "regime_bull",
    "regime_bear",
    "regime_high_vol",
]


def apply_stage15_context(feature_sets: dict, raw_sets: dict) -> dict:
    """Add cross-sectional market breadth and regime features using prior-session data only."""
    if not feature_sets:
        return feature_sets
    dates = sorted(set().union(*(set(df.index) for df in raw_sets.values())))
    close_panel = pd.DataFrame({s: df["Close"] for s, df in raw_sets.items()}).sort_index()
    ret_panel = close_panel.pct_change()
    breadth = (ret_panel.gt(0).sum(axis=1) / ret_panel.notna().sum(axis=1).replace(0, np.nan)).rename("market_breadth")
    market_ret = close_panel.mean(axis=1).pct_change().rename("market_return_1d")
    market_vol = market_ret.rolling(10).std().rename("market_volatility_10d")
    ma20 = market_ret.rolling(20).mean()
    trend = market_ret.rolling(5).mean()
    bull = ((breadth >= 0.55) & (trend > 0)).astype(float)
    bear = ((breadth <= 0.45) & (trend < 0)).astype(float)
    high_vol = (market_vol > market_vol.rolling(60, min_periods=20).median() * 1.25).astype(float)

    context = pd.concat([breadth, market_ret, market_vol, bull.rename("regime_bull"), bear.rename("regime_bear"), high_vol.rename("regime_high_vol")], axis=1)
    out = {}
    for symbol, df in feature_sets.items():
        x = df.copy()
        c = context.reindex(x.index).ffill()
        for col in context.columns:
            x[col] = c[col]
        out[symbol] = x.replace([np.inf, -np.inf], np.nan)
    return out


def fit_direction_model(data: pd.DataFrame):
    y = (data["target_close"] > 0).astype(int)
    if y.nunique() < 2:
        return None
    return XGBClassifier(
        n_estimators=250, max_depth=2, learning_rate=0.03,
        min_child_weight=4, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.03, reg_lambda=1.2, eval_metric="logloss",
        random_state=42, n_jobs=2
    ).fit(data[STAGE15_FEATURES], y)


def direction_confidence(model, x: pd.DataFrame, predicted_return: float, volatility: float) -> tuple[str, float]:
    if model is not None:
        p_up = float(model.predict_proba(x[STAGE15_FEATURES])[0, 1])
        direction = "UP" if p_up >= 0.5 else "DOWN"
        confidence = max(p_up, 1.0 - p_up)
    else:
        direction = "UP" if predicted_return >= 0 else "DOWN"
        confidence = 0.5
    # Penalize confidence when expected move is tiny relative to recent noise.
    noise = max(float(volatility), 1e-4)
    signal = min(abs(float(predicted_return)) / noise, 1.0)
    confidence = 0.5 + (confidence - 0.5) * signal
    return direction, float(np.clip(confidence, 0.5, 0.99))
