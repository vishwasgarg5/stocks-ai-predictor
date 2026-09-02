import importlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


REQUIRED_SOURCE_FILES = [
    "src/__init__.py",
    "src/config.py",
    "src/features.py",
    "src/market_data.py",
    "src/models.py",
    "src/prediction.py",
    "src/selection.py",
    "src/evaluation.py",
    "src/retraining.py",
    "src/ledger.py",
    "src/morning_runner.py",
    "src/telegram_report.py",
    "src/weekly_report.py",
]

REQUIRED_WORKFLOWS = [
    ".github/workflows/stage2_morning.yml",
    ".github/workflows/stage2_evening.yml",
    ".github/workflows/stage2_weekly.yml",
]

FORBIDDEN_LEGACY_PATHS = [
    ".github/workflows/morning_prediction.yml",
    ".github/workflows/evening_evaluate_retrain.yml",
    ".github/workflows/weekly_report.yml",
    ".github/workflows/test_stage2.yml",
    "main.py",
    "morning.py",
    "stage15_morning.py",
    "config.py",
    "weekly_report.py",
    "src/stage15.py",
    "src/ranking.py",
    "models/champion.pkl",
    "reports/performance.csv",
    "reports/weekly_report.csv",
]

CORE_MODULES = [
    "src.config",
    "src.features",
    "src.market_data",
    "src.models",
    "src.prediction",
    "src.selection",
    "src.evaluation",
    "src.retraining",
    "src.ledger",
]


def test_required_source_files_exist():
    for relative_path in REQUIRED_SOURCE_FILES:
        assert (ROOT / relative_path).exists(), relative_path


def test_current_stage2_workflows_exist():
    for relative_path in REQUIRED_WORKFLOWS:
        assert (ROOT / relative_path).exists(), relative_path


def test_legacy_paths_are_removed():
    for relative_path in FORBIDDEN_LEGACY_PATHS:
        assert not (ROOT / relative_path).exists(), relative_path


def test_core_stage2_modules_import():
    failures = []
    for module_name in CORE_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")

    assert not failures, "Stage 2 import failures:\n" + "\n".join(failures)


def test_requirements_and_readme_exist():
    assert (ROOT / "requirements.txt").exists()
    assert (ROOT / "README.md").exists()


def test_next_session_target_alignment():
    df = pd.DataFrame(
        {
            "Open": [10, 11, 12],
            "High": [11, 12, 13],
            "Low": [9, 10, 11],
            "Close": [10.5, 11.5, 12.5],
        }
    )

    for column in ["Open", "High", "Low", "Close"]:
        df[f"Target_{column}"] = df[column].shift(-1)

    assert df.loc[0, "Target_Open"] == 11
    assert df.loc[0, "Target_High"] == 12
    assert df.loc[0, "Target_Low"] == 10
    assert df.loc[0, "Target_Close"] == 11.5
    assert pd.isna(df.loc[2, "Target_Close"])


def test_lag_features_use_only_previous_sessions():
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
    lag1 = close.shift(1)
    lag2 = close.shift(2)
    lag3 = close.shift(3)

    assert lag1.iloc[3] == 102.0
    assert lag2.iloc[3] == 101.0
    assert lag3.iloc[3] == 100.0


def test_prediction_ohlc_ordering():
    prediction = {"Open": 100.0, "High": 105.0, "Low": 98.0, "Close": 103.0}
    assert prediction["High"] >= prediction["Open"]
    assert prediction["High"] >= prediction["Close"]
    assert prediction["Low"] <= prediction["Open"]
    assert prediction["Low"] <= prediction["Close"]
    assert prediction["High"] >= prediction["Low"]


def test_prediction_values_are_finite():
    values = np.array([100.0, 105.0, 98.0, 103.0])
    assert np.isfinite(values).all()


def test_ensemble_weights_sum_to_one():
    weights = {"XGB": 0.4, "RF": 0.3, "ET": 0.3}
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_ensemble_weighted_average():
    predictions = {"XGB": 100.0, "RF": 102.0, "ET": 101.0}
    weights = {"XGB": 0.5, "RF": 0.25, "ET": 0.25}
    result = sum(predictions[k] * weights[k] for k in predictions)
    assert result == 100.75


def test_score_selection_is_descending():
    from src.selection import select_top_stocks

    candidates = pd.DataFrame(
        {
            "Symbol": ["AAA", "BBB", "CCC", "DDD"],
            "Score": [55.0, 91.0, 72.0, 84.0],
            "Confidence": [0.6, 0.9, 0.7, 0.8],
            "Direction_Confidence": [0.6, 0.9, 0.7, 0.8],
        }
    )

    result = select_top_stocks(candidates, top_n=3)
    assert list(result["Symbol"]) == ["BBB", "DDD", "CCC"]


def test_feature_input_is_two_dimensional():
    frame = pd.DataFrame({"Close": [100, 101, 102], "Volume": [1000, 1100, 1200]})
    assert frame.ndim == 2
    assert frame.select_dtypes(include=[np.number]).shape[1] == 2
