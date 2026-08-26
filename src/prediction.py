import numpy as np
import pandas as pd

STAGE1_FEATURES = [
    "ret_1d","ret_3d","ret_5d","ret_10d","ret_20d",
    "ema_5_ratio","ema_10_ratio","ema_20_ratio","ema_50_ratio",
    "ema_5_slope","ema_20_slope","sma_20_ratio","sma_50_ratio",
    "rsi_7","rsi_14","macd_pct","macd_signal_pct","macd_hist_pct",
    "atr_pct","range_pct","gap_pct","volatility_5d","volatility_10d","volatility_20d",
    "bb_width","bb_position","volume_ratio_5d","volume_ratio_20d",
    "roc_5d","roc_10d","stoch_k","stoch_d","mfi_14",
    "nifty_ret_1d","nifty_ret_5d","nifty_ret_20d","nifty_volatility_10d",
    "nifty_trend","relative_ret_5d","relative_ret_20d","relative_vol_ratio"
]
TARGETS = ["target_open", "target_high", "target_low", "target_close"]


def predict_next(symbol: str, raw_df: pd.DataFrame, feature_df: pd.DataFrame, models: dict) -> dict:
    latest = feature_df.iloc[-1]
    x = latest[STAGE1_FEATURES].to_frame().T
    if x.isna().any().any():
        raise ValueError(f"Insufficient Stage 1 features for {symbol}")
    base_close = float(raw_df["Close"].iloc[-1])
    out = {"symbol": symbol, "base_close": base_close}
    for target in TARGETS:
        ret = float(models[target].predict(x)[0])
        ret = float(np.clip(ret, -0.15, 0.15))
        out[target.replace("target_", "pred_")] = base_close * (1 + ret)
    out["predicted_direction"] = "UP" if out["pred_close"] > base_close else "DOWN"
    return out
