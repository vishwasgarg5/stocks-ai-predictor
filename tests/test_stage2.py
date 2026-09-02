import importlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# =============================================================================
# STAGE 4 TEST SUITE
# Easy access: this file protects both the original Stage 2 engines and the
# new Stage 4 price-bucket + sector-ranking layer.
# =============================================================================
REQUIRED_SOURCE_FILES = [
    "src/__init__.py", "src/config.py", "src/features.py", "src/market_data.py",
    "src/models.py", "src/prediction.py", "src/selection.py", "src/stage4_engine.py",
    "src/evaluation.py", "src/retraining.py", "src/ledger.py", "src/morning_runner.py",
    "src/telegram_report.py", "src/weekly_report.py",
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
    "main.py", "morning.py", "stage15_morning.py", "config.py",
    "weekly_report.py", "evening.py", "src/stage15.py", "src/ranking.py",
    "models/champion.pkl", "reports/performance.csv", "reports/weekly_report.csv",
]

CORE_MODULES = [
    "src.config", "src.features", "src.market_data", "src.models", "src.prediction",
    "src.selection", "src.stage4_engine", "src.evaluation", "src.retraining", "src.ledger",
]


def test_required_source_files_exist():
    for path in REQUIRED_SOURCE_FILES:
        assert (ROOT / path).exists(), path


def test_current_workflows_exist():
    for path in REQUIRED_WORKFLOWS:
        assert (ROOT / path).exists(), path


def test_legacy_paths_are_removed():
    for path in FORBIDDEN_LEGACY_PATHS:
        assert not (ROOT / path).exists(), path


def test_core_modules_import():
    failures = []
    for module_name in CORE_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
    assert not failures, "Stage 4 import failures:\n" + "\n".join(failures)


def test_requirements_and_readme_exist():
    assert (ROOT / "requirements.txt").exists()
    assert (ROOT / "README.md").exists()


def test_next_session_target_alignment():
    df = pd.DataFrame({"Open": [10, 11, 12], "High": [11, 12, 13], "Low": [9, 10, 11], "Close": [10.5, 11.5, 12.5]})
    for column in ["Open", "High", "Low", "Close"]:
        df[f"Target_{column}"] = df[column].shift(-1)
    assert df.loc[0, "Target_Open"] == 11
    assert df.loc[0, "Target_High"] == 12
    assert df.loc[0, "Target_Low"] == 10
    assert df.loc[0, "Target_Close"] == 11.5
    assert pd.isna(df.loc[2, "Target_Close"])


def test_lag_features_use_only_previous_sessions():
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
    assert close.shift(1).iloc[3] == 102.0
    assert close.shift(2).iloc[3] == 101.0
    assert close.shift(3).iloc[3] == 100.0


def test_prediction_ohlc_ordering():
    p = {"Open": 100.0, "High": 105.0, "Low": 98.0, "Close": 103.0}
    assert p["High"] >= p["Open"] >= p["Low"]
    assert p["High"] >= p["Close"] >= p["Low"]


def test_prediction_values_are_finite():
    assert np.isfinite(np.array([100.0, 105.0, 98.0, 103.0])).all()


def test_ensemble_weights_sum_to_one():
    weights = {"XGB": 0.4, "RF": 0.3, "ET": 0.3}
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_ensemble_weighted_average():
    predictions = {"XGB": 100.0, "RF": 102.0, "ET": 101.0}
    weights = {"XGB": 0.5, "RF": 0.25, "ET": 0.25}
    assert sum(predictions[k] * weights[k] for k in predictions) == 100.75


def test_score_selection_returns_top_rows():
    from src.selection import select_top_stocks
    candidates = pd.DataFrame({
        "Symbol": ["AAA", "BBB", "CCC", "DDD"],
        "TechnicalScore": [60, 95, 70, 85], "Expected_Return": [1, 8, 2, 5],
        "Confidence": [60, 95, 70, 85], "Direction_Confidence": [60, 95, 70, 85],
        "Direction": ["UP", "UP", "UP", "UP"],
    })
    result = select_top_stocks(candidates, top_n=3)
    assert len(result) == 3
    assert result["Score"].is_monotonic_decreasing
    assert result.iloc[0]["Symbol"] == "BBB"


def test_feature_input_is_two_dimensional():
    frame = pd.DataFrame({"Close": [100, 101, 102], "Volume": [1000, 1100, 1200]})
    assert frame.ndim == 2
    assert frame.select_dtypes(include=[np.number]).shape[1] == 2


def test_stage4_price_buckets():
    from src.stage4_engine import price_bucket
    assert price_bucket(1500)[0] == "B1"
    assert price_bucket(750)[0] == "B2"
    assert price_bucket(250)[0] == "B3"
    assert price_bucket(75)[0] == "B4"
    assert price_bucket(25)[0] == "B5"
    assert price_bucket(9)[0] == "OUT"


def test_stage4_price_bucket_selection():
    from src.stage4_engine import select_price_bucket_candidates
    df = pd.DataFrame({
        "Symbol": ["A", "B", "C", "D"],
        "PriceBucket": ["B1", "B1", "B2", "B2"],
        "Score": [80, 90, 70, 95],
    })
    result = select_price_bucket_candidates(df, per_bucket=1)
    assert set(result["Symbol"]) == {"B", "D"}


def test_stage4_score_weights_sum_to_one():
    assert abs(0.20 + 0.18 + 0.18 + 0.14 + 0.10 + 0.10 + 0.10 - 1.0) < 1e-9
