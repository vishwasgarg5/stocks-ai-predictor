import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
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
STAGE15_FEATURES = STAGE1_FEATURES + [
    "market_breadth","market_return_1d","market_volatility_10d",
    "regime_bull","regime_bear","regime_high_vol"
]
TARGETS = ["target_open", "target_high", "target_low", "target_close"]
MODEL_PATH = MODEL_DIR / "champion.pkl"


def build_training_frame(feature_sets, columns=STAGE15_FEATURES):
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


def fit_model(data, columns=STAGE15_FEATURES):
    return {
        target: XGBRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.02,
            min_child_weight=4, reg_alpha=0.03, reg_lambda=1.3,
            subsample=0.85, colsample_bytree=0.85,
            objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=2
        ).fit(data[columns], data[target])
        for target in TARGETS
    }


def fit_direction_model(data, columns=STAGE15_FEATURES):
    y = (data["target_close"] > 0).astype(int)
    if y.nunique() < 2:
        return None
    return XGBClassifier(
        n_estimators=250, max_depth=2, learning_rate=0.03,
        min_child_weight=4, reg_alpha=0.03, reg_lambda=1.2,
        subsample=0.85, colsample_bytree=0.85,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=2
    ).fit(data[columns], y)


def fit_bundle(data):
    models = fit_model(data)
    models["direction_model"] = fit_direction_model(data)
    models["stage"] = "1.5"
    models["feature_count"] = len(STAGE15_FEATURES)
    return models


def _mae(models, data, columns=STAGE15_FEATURES):
    vals = []
    for target in TARGETS:
        pred = models[target].predict(data[columns])
        vals.append(float(np.mean(np.abs(pred - data[target].to_numpy()))))
    return float(np.mean(vals))


def _walk_forward_splits(data, folds=3):
    n = len(data)
    min_train = max(int(n * 0.55), 30)
    test_size = max(int(n * 0.12), 10)
    splits = []
    for i in range(folds):
        train_end = min_train + i * test_size
        test_end = min(train_end + test_size, n)
        if test_end > train_end and train_end < n:
            splits.append((data.iloc[:train_end], data.iloc[train_end:test_end]))
    return splits


def _walk_forward_score(data):
    scores = []
    for train, valid in _walk_forward_splits(data):
        challenger = fit_bundle(train)
        scores.append(_mae(challenger, valid))
    if not scores:
        raise ValueError("Not enough rows for walk-forward validation")
    return float(np.mean(scores))


def load_champion():
    if not MODEL_PATH.exists():
        return None
    try:
        with MODEL_PATH.open("rb") as f:
            model = pickle.load(f)
        if not isinstance(model, dict) or not all(t in model for t in TARGETS):
            return None
        if model.get("stage") != "1.5" or model.get("feature_count") != len(STAGE15_FEATURES):
            return None
        return model
    except Exception:
        return None


def save_champion(models):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(models, f)


def champion_challenger(feature_sets):
    """Compare Challenger and Champion with chronological walk-forward validation."""
    data = build_training_frame(feature_sets)
    if len(data) < 60:
        raise ValueError("Not enough Stage 1.5 rows for validation")

    challenger_mae = _walk_forward_score(data)
    champion = load_champion()
    champion_mae = _walk_forward_score(data) if champion is None else _mae(champion, data.iloc[-max(20, int(len(data)*0.2)):])

    if champion is None or challenger_mae < champion_mae:
        final_model = fit_bundle(data)
        save_champion(final_model)
        decision = "REPLACE CHAMPION" if champion is not None else "BOOTSTRAP STAGE1.5 CHAMPION"
        return final_model, True, champion_mae, challenger_mae, decision
    return champion, False, champion_mae, challenger_mae, "KEEP CHAMPION"


def rolling_retrain(feature_sets):
    return champion_challenger(feature_sets)[0]
