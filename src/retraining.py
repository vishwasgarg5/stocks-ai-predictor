import pickle
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
            x["date"] = pd.to_datetime(x.index)
            x["symbol"] = symbol
            frames.append(x)
    if not frames:
        raise ValueError("No training rows available")
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def fit_model(data: pd.DataFrame) -> dict:
    return {
        target: XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85,
            objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=2
        ).fit(data[FEATURES], data[target])
        for target in TARGETS
    }


def _mae(models, data):
    return sum(float((models[t].predict(data[FEATURES]) - data[t]).abs().mean()) for t in TARGETS) / len(TARGETS)


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
    """Keep the existing champion unless a newly trained challenger wins on the newest holdout."""
    data = build_training_frame(feature_sets)
    if len(data) < 10:
        raise ValueError("Not enough rows for champion/challenger validation")
    split = min(max(int(len(data) * 0.8), 1), len(data) - 1)
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
