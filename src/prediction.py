import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from config import FEATURES, RANDOM_STATE

TARGETS = ["target_open", "target_high", "target_low", "target_close"]


def train_models(feature_sets: dict[str, pd.DataFrame]) -> dict[str, XGBRegressor]:
    frames = []
    for symbol, df in feature_sets.items():
        clean = df.dropna(subset=FEATURES + TARGETS).copy()
        if not clean.empty:
            frames.append(clean)
    if not frames:
        raise ValueError("No training rows available")
    data = pd.concat(frames, ignore_index=True)
    models = {}
    for target in TARGETS:
        model = XGBRegressor(
            n_estimators=250, max_depth=3, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85,
            objective="reg:squarederror", random_state=RANDOM_STATE,
            n_jobs=2
        )
        model.fit(data[FEATURES], data[target])
        models[target] = model
    return models


def predict_next(symbol: str, raw_df: pd.DataFrame, feature_df: pd.DataFrame, models: dict) -> dict:
    latest = feature_df.iloc[-1]
    x = latest[FEATURES].to_frame().T
    base_close = float(raw_df["Close"].iloc[-1])
    out = {"symbol": symbol, "base_close": base_close}
    for target, model in models.items():
        ret = float(model.predict(x)[0])
        out[target.replace("target_", "pred_")] = base_close * (1 + ret)
    out["predicted_direction"] = "UP" if out["pred_close"] > base_close else "DOWN"
    return out
