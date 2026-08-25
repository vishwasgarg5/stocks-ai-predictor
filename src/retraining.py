import pickle
from pathlib import Path
import pandas as pd
from xgboost import XGBRegressor
from config import FEATURES, RANDOM_STATE, MODEL_DIR

TARGETS = ["target_open", "target_high", "target_low", "target_close"]
MODEL_PATH = MODEL_DIR / "champion.pkl"


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


def fit_model(data: pd.DataFrame) -> dict:
    models = {}
    for target in TARGETS:
        model = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                             subsample=0.85, colsample_bytree=0.85,
                             objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=2)
        model.fit(data[FEATURES], data[target])
        models[target] = model
    return models


def _mae(models, data):
    errors = []
    for target in TARGETS:
        pred = models[target].predict(data[FEATURES])
        errors.append(float((pred - data[target]).abs().mean()))
    return sum(errors) / len(errors)


def load_champion():
    if not MODEL_PATH.exists():
        return None
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def save_champion(models):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(models, f)


def champion_challenger(feature_sets: dict[str, pd.DataFrame]):
    """Evaluate old champion vs challenger on the newest chronological holdout.
    Replace the champion only when the challenger has strictly lower MAE."""
    data = build_training_frame(feature_sets).sort_values("date" if "date" in build_training_frame(feature_sets).columns else FEATURES[0])
    # Preserve chronological order when the feature frame has a DatetimeIndex converted to rows.
    if "date" not in data.columns:
        data = data.sort_index()
    split = max(int(len(data) * 0.8), 1)
    if split >= len(data):
        split = len(data) - 1
    train, valid = data.iloc[:split], data.iloc[split:]
    challenger = fit_model(train)
    challenger_mae = _mae(challenger, valid)
    champion = load_champion()
    if champion is None:
        champion = fit_model(data)
        save_champion(champion)
        return champion, True, None, challenger_mae
    champion_mae = _mae(champion, valid)
    if challenger_mae < champion_mae:
        champion = fit_model(data)
        save_champion(champion)
        return champion, True, champion_mae, challenger_mae
    return champion, False, champion_mae, challenger_mae


def rolling_retrain(feature_sets: dict[str, pd.DataFrame]) -> dict:
    return champion_challenger(feature_sets)[0]
