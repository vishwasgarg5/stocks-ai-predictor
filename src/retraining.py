import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from config import RANDOM_STATE, MODEL_DIR

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
MODEL_PATH = MODEL_DIR / "champion.pkl"


def build_training_frame(feature_sets, columns=STAGE1_FEATURES):
    frames = []
    for symbol, df in feature_sets.items():
        x = df.dropna(subset=columns + TARGETS).copy()
        if not x.empty:
            x["date"] = pd.to_datetime(x.index)
            x["symbol"] = symbol
            frames.append(x)
    if not frames:
        raise ValueError("No training rows available")
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def fit_model(data, columns=STAGE1_FEATURES):
    return {
        target: XGBRegressor(
            n_estimators=450, max_depth=3, learning_rate=0.025,
            min_child_weight=3, reg_alpha=0.02, reg_lambda=1.2,
            subsample=0.85, colsample_bytree=0.85,
            objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=2
        ).fit(data[columns], data[target])
        for target in TARGETS
    }


def _mae(models, data, columns=STAGE1_FEATURES):
    vals = []
    for target in TARGETS:
        pred = models[target].predict(data[columns])
        vals.append(float(np.mean(np.abs(pred - data[target].to_numpy()))))
    return float(np.mean(vals))


def load_champion():
    if not MODEL_PATH.exists():
        return None
    try:
        with MODEL_PATH.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save_champion(models):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(models, f)


def champion_challenger(feature_sets):
    # Newest 20% is a strict chronological holdout: no future rows enter training.
    data = build_training_frame(feature_sets)
    if len(data) < 30:
        raise ValueError("Not enough Stage 1 rows for validation")
    split = min(max(int(len(data) * 0.80), 1), len(data) - 1)
    train, valid = data.iloc[:split], data.iloc[split:]
    challenger = fit_model(train)
    challenger_mae = _mae(challenger, valid)

    champion = load_champion()
    champion_mae = None
    champion_valid_ok = False
    if champion is not None:
        try:
            champion_mae = _mae(champion, valid)
            champion_valid_ok = np.isfinite(champion_mae)
        except Exception:
            champion_valid_ok = False

    # Stage 1 is a deliberate model-schema upgrade. An incompatible old model
    # cannot be fairly scored with the new feature space, so bootstrap Stage 1 once.
    if champion is None or not champion_valid_ok or challenger_mae < champion_mae:
        final_model = fit_model(data)
        save_champion(final_model)
        return final_model, True, champion_mae, challenger_mae, "REPLACE CHAMPION"
    return champion, False, champion_mae, challenger_mae, "KEEP CHAMPION"


def rolling_retrain(feature_sets):
    return champion_challenger(feature_sets)[0]
