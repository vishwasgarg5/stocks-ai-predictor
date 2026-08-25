import pandas as pd
from xgboost import XGBRegressor
from config import FEATURES, RANDOM_STATE

TARGETS = ["target_open", "target_high", "target_low", "target_close"]


def build_training_frame(feature_sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for symbol, df in feature_sets.items():
        x = df.dropna(subset=FEATURES + TARGETS).copy()
        if not x.empty:
            x["symbol"] = symbol
            frames.append(x)
    if not frames:
        raise ValueError("No training rows available")
    return pd.concat(frames, ignore_index=True)


def fit_champion(data: pd.DataFrame) -> dict:
    models = {}
    for target in TARGETS:
        model = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                             subsample=0.85, colsample_bytree=0.85,
                             objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=2)
        model.fit(data[FEATURES], data[target])
        models[target] = model
    return models


def rolling_retrain(feature_sets: dict[str, pd.DataFrame]) -> dict:
    """Retrain only on the latest 3-month rolling window."""
    return fit_champion(build_training_frame(feature_sets))
