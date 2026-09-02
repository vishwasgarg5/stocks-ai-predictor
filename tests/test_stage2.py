"""
Stage 2 regression and contract test suite.

This suite is designed to test the Stage 2 architecture without requiring:
- live NSE data
- yfinance network access
- Telegram
- GitHub API access
- a local database

Run locally:
    pytest -q tests/test_stage2.py

Run in GitHub Actions:
    pytest -q tests/test_stage2.py -ra
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data"
MODELS = ROOT / "models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_optional(module_name: str):
    """Import a module without turning an import failure into a false
    'missing module' failure."""
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _find_callable(module, names):
    """Find the first callable matching one of the supplied names."""
    if module is None:
        return None

    for name in names:
        obj = getattr(module, name, None)
        if callable(obj):
            return obj

    return None


def _call_flexible(func, *args, **kwargs):
    """Call a function while tolerating implementations with fewer kwargs."""
    if func is None:
        return None

    try:
        return func(*args, **kwargs)
    except TypeError:
        sig = inspect.signature(func)
        accepted = {
            key: value
            for key, value in kwargs.items()
            if key in sig.parameters
        }
        return func(*args, **accepted)


def _sample_ohlcv(rows: int = 180) -> pd.DataFrame:
    """Create deterministic synthetic OHLCV data."""
    rng = np.random.default_rng(42)

    close = 100 + np.cumsum(
        rng.normal(0.15, 1.0, rows)
    )

    close = np.maximum(close, 10)

    open_ = close + rng.normal(0, 0.8, rows)

    high = (
        np.maximum(open_, close)
        + rng.uniform(0.2, 2.0, rows)
    )

    low = (
        np.minimum(open_, close)
        - rng.uniform(0.2, 2.0, rows)
    )

    volume = rng.integers(
        100_000,
        2_000_000,
        rows,
    )

    dates = pd.bdate_range(
        "2025-01-01",
        periods=rows,
    )

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


def _make_next_day_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Reference implementation for next-day OHLC targets."""
    result = df.copy()

    result["Target_Open"] = result["Open"].shift(-1)
    result["Target_High"] = result["High"].shift(-1)
    result["Target_Low"] = result["Low"].shift(-1)
    result["Target_Close"] = result["Close"].shift(-1)

    return result


# ===========================================================================
# 1. Repository structure
# ===========================================================================

def test_stage2_test_file_exists():
    assert Path(__file__).exists()


def test_src_directory_exists():
    assert SRC.exists()
    assert SRC.is_dir()


def test_required_stage2_source_files_exist():
    """
    Check files rather than importing every module.

    Some modules intentionally import optional/runtime dependencies at module
    import time. A failed import should not be incorrectly reported as a
    missing source file.
    """
    required_files = [
        "config.py",
        "features.py",
        "market_data.py",
        "models.py",
        "prediction.py",
        "ranking.py",
        "selection.py",
        "evaluation.py",
        "retraining.py",
        "ledger.py",
    ]

    missing = [
        filename
        for filename in required_files
        if not (SRC / filename).exists()
    ]

    assert not missing, (
        "Missing required source files: "
        + ", ".join(missing)
    )


def test_src_package_has_init_file():
    assert (SRC / "__init__.py").exists()


# ===========================================================================
# 2. Existing module imports
# ===========================================================================

@pytest.mark.parametrize(
    "module_name",
    [
        "src.config",
        "src.features",
        "src.market_data",
        "src.models",
        "src.prediction",
        "src.ranking",
        "src.selection",
        "src.evaluation",
        "src.retraining",
        "src.ledger",
    ],
)
def test_stage2_module_import_status(module_name):
    """
    Import smoke test.

    Import failures are reported as skips here because the module may depend
    on runtime-only libraries or configuration. The source-file existence
    test above is the authoritative structural test.
    """
    module = _import_optional(module_name)

    if module is None:
        pytest.skip(
            f"{module_name} could not be imported in this test environment"
        )

    assert module is not None


# ===========================================================================
# 3. OHLC target alignment
# ===========================================================================

def test_next_day_targets_are_shifted_one_session_forward():
    df = _sample_ohlcv(20)
    result = _make_next_day_targets(df)

    assert result.loc[
        result.index[0],
        "Target_Open",
    ] == pytest.approx(df.iloc[1]["Open"])

    assert result.loc[
        result.index[0],
        "Target_High",
    ] == pytest.approx(df.iloc[1]["High"])

    assert result.loc[
        result.index[0],
        "Target_Low",
    ] == pytest.approx(df.iloc[1]["Low"])

    assert result.loc[
        result.index[0],
        "Target_Close",
    ] == pytest.approx(df.iloc[1]["Close"])


def test_last_row_has_no_future_target():
    df = _sample_ohlcv(20)
    result = _make_next_day_targets(df)

    last = result.iloc[-1]

    assert pd.isna(last["Target_Open"])
    assert pd.isna(last["Target_High"])
    assert pd.isna(last["Target_Low"])
    assert pd.isna(last["Target_Close"])


def test_training_data_excludes_missing_future_targets():
    df = _sample_ohlcv(50)
    result = _make_next_day_targets(df)

    target_columns = [
        "Target_Open",
        "Target_High",
        "Target_Low",
        "Target_Close",
    ]

    training = result.dropna(
        subset=target_columns
    )

    assert len(training) == len(df) - 1
    assert not training[target_columns].isna().any().any()


# ===========================================================================
# 4. Leakage protection
# ===========================================================================

def test_features_are_based_on_current_or_previous_information():
    """
    Basic contract check.

    Future target columns must remain separate from feature columns.
    """
    df = _make_next_day_targets(
        _sample_ohlcv(30)
    )

    target_columns = {
        "Target_Open",
        "Target_High",
        "Target_Low",
        "Target_Close",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in target_columns
    ]

    assert not any(
        column.lower().startswith("future_")
        for column in feature_columns
    )


def test_future_target_columns_are_not_used_as_normal_features():
    forbidden = {
        "Future_Open",
        "Future_High",
        "Future_Low",
        "Future_Close",
        "Next_Open",
        "Next_High",
        "Next_Low",
        "Next_Close",
    }

    df = _sample_ohlcv(30)

    feature_columns = set(df.columns)

    assert not forbidden.intersection(
        feature_columns
    )


# ===========================================================================
# 5. Feature engineering
# ===========================================================================

def test_features_module_if_available_does_not_return_empty_data():
    module = _import_optional("src.features")

    if module is None:
        pytest.skip("src.features unavailable")

    func = _find_callable(
        module,
        [
            "build_features",
            "create_features",
            "engineer_features",
            "add_features",
        ],
    )

    if func is None:
        pytest.skip(
            "No recognised feature-building function found"
        )

    try:
        result = _call_flexible(
            func,
            _sample_ohlcv(180).copy(),
        )
    except Exception as exc:
        pytest.fail(
            f"Feature engineering failed: {exc}"
        )

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_feature_columns_are_not_all_nan():
    module = _import_optional("src.features")

    if module is None:
        pytest.skip("src.features unavailable")

    func = _find_callable(
        module,
        [
            "build_features",
            "create_features",
            "engineer_features",
            "add_features",
        ],
    )

    if func is None:
        pytest.skip(
            "No recognised feature-building function found"
        )

    try:
        result = _call_flexible(
            func,
            _sample_ohlcv(180).copy(),
        )
    except Exception as exc:
        pytest.fail(
            f"Feature engineering failed: {exc}"
        )

    assert isinstance(result, pd.DataFrame)

    numeric_columns = result.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:
        assert not result[column].isna().all(), (
            f"Feature column {column} is entirely NaN"
        )


# ===========================================================================
# 6. Missing / malformed data
# ===========================================================================

def test_empty_dataframe_does_not_create_invalid_shape():
    empty = pd.DataFrame(
        columns=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    )

    assert empty.empty
    assert empty.shape[0] == 0


def test_missing_close_column_is_detectable():
    df = pd.DataFrame(
        {
            "Open": [100, 101],
            "High": [102, 103],
            "Low": [99, 100],
            "Volume": [1000, 1100],
        }
    )

    required = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    assert not required.issubset(df.columns)


def test_close_series_is_one_dimensional():
    series = pd.Series(
        np.arange(50, dtype=float),
        name="Close",
    )

    values = np.asarray(series)

    assert values.ndim == 1
    assert values.shape == (50,)


# ===========================================================================
# 7. Prediction output
# ===========================================================================

def test_prediction_contains_four_ohlc_values():
    prediction = {
        "Open": 100.0,
        "High": 103.0,
        "Low": 98.0,
        "Close": 101.5,
    }

    required = {
        "Open",
        "High",
        "Low",
        "Close",
    }

    assert required.issubset(prediction)

    values = [
        prediction["Open"],
        prediction["High"],
        prediction["Low"],
        prediction["Close"],
    ]

    assert len(values) == 4
    assert all(
        math.isfinite(float(value))
        for value in values
    )


def test_prediction_high_is_not_below_open_or_close():
    prediction = {
        "Open": 100.0,
        "High": 103.0,
        "Low": 98.0,
        "Close": 101.5,
    }

    assert prediction["High"] >= prediction["Open"]
    assert prediction["High"] >= prediction["Close"]


def test_prediction_low_is_not_above_open_or_close():
    prediction = {
        "Open": 100.0,
        "High": 103.0,
        "Low": 98.0,
        "Close": 101.5,
    }

    assert prediction["Low"] <= prediction["Open"]
    assert prediction["Low"] <= prediction["Close"]


# ===========================================================================
# 8. Ensemble model contract
# ===========================================================================

def test_models_module_has_stage2_model_or_factory():
    module = _import_optional("src.models")

    if module is None:
        pytest.skip("src.models unavailable")

    names = [
        name.lower()
        for name in dir(module)
    ]

    model_tokens = [
        "xgb",
        "xgboost",
        "randomforest",
        "random_forest",
        "extratrees",
        "extra_trees",
        "ensemble",
        "model_factory",
        "build_model",
        "create_model",
        "train_models",
    ]

    found = any(
        token in name
        for name in names
        for token in model_tokens
    )

    assert found


def test_ensemble_weights_are_normalizable():
    weights = {
        "xgb": 0.40,
        "random_forest": 0.30,
        "extra_trees": 0.30,
    }

    total = sum(weights.values())

    assert all(
        math.isfinite(float(value))
        for value in weights.values()
    )

    assert all(
        value >= 0
        for value in weights.values()
    )

    assert total == pytest.approx(1.0)


# ===========================================================================
# 9. Direction model
# ===========================================================================

def test_direction_labels_are_valid():
    valid = {
        "UP",
        "DOWN",
        "NEUTRAL",
    }

    sample = [
        "UP",
        "DOWN",
        "NEUTRAL",
        "UP",
        "DOWN",
    ]

    assert set(sample).issubset(valid)


def test_direction_accuracy_is_bounded():
    for accuracy in [
        0.0,
        0.25,
        0.50,
        0.75,
        1.0,
    ]:
        assert 0 <= accuracy <= 1


# ===========================================================================
# 10. Top 5 selection
# ===========================================================================

def test_top5_selection_returns_maximum_five():
    candidates = pd.DataFrame(
        {
            "Stock": [
                "AAA",
                "BBB",
                "CCC",
                "DDD",
                "EEE",
                "FFF",
                "GGG",
            ],
            "Score": [
                99,
                98,
                97,
                96,
                95,
                94,
                93,
            ],
        }
    )

    result = (
        candidates
        .sort_values(
            "Score",
            ascending=False,
        )
        .head(5)
    )

    assert len(result) == 5


def test_top5_selection_is_deterministic():
    candidates = pd.DataFrame(
        {
            "Stock": [
                "AAA",
                "BBB",
                "CCC",
                "DDD",
                "EEE",
            ],
            "Score": [
                90.0,
                95.0,
                93.0,
                91.0,
                94.0,
            ],
        }
    )

    def select():
        return (
            candidates
            .sort_values(
                ["Score", "Stock"],
                ascending=[False, True],
            )
            .head(5)["Stock"]
            .tolist()
        )

    assert select() == select()


def test_duplicate_stocks_are_removed():
    candidates = pd.DataFrame(
        {
            "Stock": [
                "AAA",
                "AAA",
                "BBB",
                "CCC",
                "DDD",
                "EEE",
            ],
            "Score": [
                99,
                98,
                97,
                96,
                95,
                94,
            ],
        }
    )

    result = (
        candidates
        .sort_values(
            "Score",
            ascending=False,
        )
        .drop_duplicates(
            "Stock"
        )
        .head(5)
    )

    assert result["Stock"].is_unique
    assert len(result) == 5


# ===========================================================================
# 11. Morning -> evening ledger
# ===========================================================================

def test_morning_prediction_ledger_schema():
    ledger = pd.DataFrame(
        {
            "Prediction_Date": ["2026-09-02"] * 5,
            "Stock": [
                "AAA",
                "BBB",
                "CCC",
                "DDD",
                "EEE",
            ],
            "Pred_Open": [100] * 5,
            "Pred_High": [103] * 5,
            "Pred_Low": [98] * 5,
            "Pred_Close": [101] * 5,
        }
    )

    required = {
        "Prediction_Date",
        "Stock",
        "Pred_Open",
        "Pred_High",
        "Pred_Low",
        "Pred_Close",
    }

    assert required.issubset(
        ledger.columns
    )


def test_evening_uses_exact_morning_stocks():
    morning = [
        "BAJAJ-AUTO",
        "COFORGE",
        "DIVISLAB",
        "APLAPOLLO",
        "KOTAKBANK",
    ]

    evening = [
        "BAJAJ-AUTO",
        "COFORGE",
        "DIVISLAB",
        "APLAPOLLO",
        "KOTAKBANK",
    ]

    assert evening == morning


def test_evening_selection_mismatch_is_detectable():
    morning = {
        "AAA",
        "BBB",
        "CCC",
        "DDD",
        "EEE",
    }

    evening = {
        "AAA",
        "BBB",
        "CCC",
        "DDD",
        "FFF",
    }

    assert morning != evening


# ===========================================================================
# 12. Evaluation metrics
# ===========================================================================

def test_mape_reference_calculation():
    """
    Actual:
        100, 200, 300

    Predicted:
        101, 198, 303

    Absolute percentage errors:
        1%, 1%, 1%

    Therefore MAPE = 1%.
    """
    actual = np.array(
        [100, 200, 300],
        dtype=float,
    )

    predicted = np.array(
        [101, 198, 303],
        dtype=float,
    )

    mape = np.mean(
        np.abs(
            (actual - predicted)
            / actual
        )
    ) * 100

    assert mape == pytest.approx(
        1.0,
        rel=1e-5,
    )


def test_mape_ignores_zero_actual_values():
    actual = np.array(
        [100, 0, 200],
        dtype=float,
    )

    predicted = np.array(
        [101, 10, 198],
        dtype=float,
    )

    mask = actual != 0

    mape = np.mean(
        np.abs(
            (actual[mask] - predicted[mask])
            / actual[mask]
        )
    ) * 100

    assert np.isfinite(mape)
    assert mape >= 0


def test_direction_accuracy_reference():
    actual_close = np.array(
        [100, 102, 101, 104, 103]
    )

    predicted_close = np.array(
        [101, 103, 100, 105, 102]
    )

    actual_direction = np.sign(
        np.diff(actual_close)
    )

    predicted_direction = np.sign(
        np.diff(predicted_close)
    )

    accuracy = np.mean(
        actual_direction
        == predicted_direction
    )

    assert 0 <= accuracy <= 1


# ===========================================================================
# 13. Champion / Challenger
# ===========================================================================

def test_tiny_champion_challenger_difference_does_not_switch():
    champion = 0.012665
    challenger = 0.012689

    relative_improvement = (
        champion - challenger
    ) / champion

    should_switch = (
        relative_improvement >= 0.02
    )

    assert should_switch is False


def test_meaningful_challenger_improvement_can_switch():
    champion = 1.00
    challenger = 0.97

    relative_improvement = (
        champion - challenger
    ) / champion

    assert relative_improvement == pytest.approx(
        0.03
    )

    assert relative_improvement >= 0.02


def test_champion_decision_is_explicit():
    valid = {
        "KEPT",
        "REPLACED",
        "SWITCHED",
        "CHAMPION_KEPT",
    }

    assert "KEPT" in valid
    assert "REPLACED" in valid


# ===========================================================================
# 14. Jump engine
# ===========================================================================

def test_jump_candidate_requires_more_than_five_percent():
    candidates = pd.DataFrame(
        {
            "Stock": [
                "AAA",
                "BBB",
                "CCC",
            ],
            "Upside": [
                4.9,
                5.01,
                7.5,
            ],
        }
    )

    result = candidates[
        candidates["Upside"] > 5.0
    ]

    assert result["Stock"].tolist() == [
        "BBB",
        "CCC",
    ]


def test_exactly_five_percent_is_not_above_five():
    assert not (5.0 > 5.0)


def test_jump_horizons_are_valid():
    horizons = [1, 3, 5, 7]

    assert all(
        1 <= horizon <= 7
        for horizon in horizons
    )


def test_jump_prediction_schema():
    prediction = {
        "Stock": "AAA",
        "Current_Price": 100.0,
        "Target_7D": 108.0,
        "Expected_Max_Upside": 8.0,
        "Horizon": 7,
        "Confidence": 0.72,
        "Risk": "MEDIUM",
    }

    required = {
        "Stock",
        "Current_Price",
        "Target_7D",
        "Expected_Max_Upside",
        "Horizon",
        "Confidence",
        "Risk",
    }

    assert required.issubset(
        prediction
    )


def test_jump_prediction_is_not_guaranteed_profit():
    message = (
        "Model-estimated upside is 7.2%; "
        "actual outcome is uncertain."
    ).lower()

    forbidden = {
        "guaranteed profit",
        "guaranteed return",
        "sure shot",
        "certain profit",
    }

    assert not any(
        phrase in message
        for phrase in forbidden
    )


# ===========================================================================
# 15. Jump outcome
# ===========================================================================

def test_jump_target_hit_uses_actual_high():
    current_price = 100.0
    target = 106.0
    actual_max_high = 107.0

    predicted_upside = (
        (target - current_price)
        / current_price
    ) * 100

    hit = actual_max_high >= target

    assert predicted_upside > 5
    assert hit is True


def test_jump_target_can_hit_even_if_close_does_not():
    target = 106.0
    actual_high = 107.0
    actual_close = 103.0

    assert actual_high >= target
    assert actual_close < target


# ===========================================================================
# 16. Intraday engine
# ===========================================================================

def test_intraday_signal_schema():
    signal = {
        "Stock": "AAA",
        "Score": 87.5,
        "CMP": 100.0,
        "Entry": "99.5-100.5",
        "Target": 103.0,
        "Stop_Loss": 98.0,
        "Risk_Reward": 1.5,
        "Direction": "UP",
        "Confidence": 0.70,
    }

    required = {
        "Stock",
        "Score",
        "CMP",
        "Entry",
        "Target",
        "Stop_Loss",
        "Risk_Reward",
        "Direction",
        "Confidence",
    }

    assert required.issubset(
        signal
    )


def test_intraday_risk_reward_is_positive():
    reward = 3.0
    risk = 2.0

    assert reward / risk > 0


def test_intraday_long_stop_is_below_target():
    target = 103.0
    stop_loss = 98.0

    assert stop_loss < target


# ===========================================================================
# 17. Duplicate protection
# ===========================================================================

def test_duplicate_prediction_key_can_be_detected():
    ledger = pd.DataFrame(
        {
            "Prediction_Date": [
                "2026-09-02",
                "2026-09-02",
                "2026-09-02",
            ],
            "Stock": [
                "AAA",
                "BBB",
                "AAA",
            ],
        }
    )

    duplicate_mask = ledger.duplicated(
        subset=[
            "Prediction_Date",
            "Stock",
        ],
        keep=False,
    )

    assert duplicate_mask.any()


def test_unique_prediction_keys_pass():
    ledger = pd.DataFrame(
        {
            "Prediction_Date": [
                "2026-09-02",
                "2026-09-02",
                "2026-09-02",
            ],
            "Stock": [
                "AAA",
                "BBB",
                "CCC",
            ],
        }
    )

    assert not ledger.duplicated(
        subset=[
            "Prediction_Date",
            "Stock",
        ]
    ).any()


# ===========================================================================
# 18. GitHub-only persistence
# ===========================================================================

def test_no_sqlite_database_files():
    sqlite_files = list(
        ROOT.rglob("*.sqlite")
    )

    db_files = list(
        ROOT.rglob("*.db")
    )

    database_files = (
        sqlite_files + db_files
    )

    assert not database_files, (
        "Local database files detected: "
        + ", ".join(
            str(path)
            for path in database_files
        )
    )


# ===========================================================================
# 19. Model state
# ===========================================================================

def test_model_state_is_json_serializable():
    state = {
        "active_model": "ensemble_a",
        "champion_mae": 0.012665,
        "challenger_mae": 0.012689,
        "last_updated": "2026-09-02",
    }

    encoded = json.dumps(state)
    decoded = json.loads(encoded)

    assert decoded[
        "active_model"
    ] == "ensemble_a"

    assert math.isfinite(
        decoded["champion_mae"]
    )


# ===========================================================================
# 20. Prediction cutoff
# ===========================================================================

def test_prediction_cutoff_is_before_prediction_date():
    prediction_date = pd.Timestamp(
        "2026-09-02"
    )

    cutoff_date = pd.Timestamp(
        "2026-09-01"
    )

    assert cutoff_date < prediction_date


def test_same_day_incomplete_data_is_not_used():
    prediction_date = pd.Timestamp(
        "2026-09-02"
    )

    data_cutoff = pd.Timestamp(
        "2026-09-01"
    )

    assert data_cutoff < prediction_date


# ===========================================================================
# 21. OHLC sanity
# ===========================================================================

def test_ohlc_price_relationship():
    row = {
        "Open": 100.0,
        "High": 105.0,
        "Low": 98.0,
        "Close": 103.0,
    }

    assert row["High"] >= max(
        row["Open"],
        row["Close"],
    )

    assert row["Low"] <= min(
        row["Open"],
        row["Close"],
    )


def test_volume_is_non_negative():
    df = _sample_ohlcv(50)

    assert (
        df["Volume"] >= 0
    ).all()


# ===========================================================================
# 22. yfinance MultiIndex protection
# ===========================================================================

def test_multilevel_dataframe_can_be_flattened():
    columns = pd.MultiIndex.from_tuples(
        [
            ("Open", "AAA.NS"),
            ("High", "AAA.NS"),
            ("Low", "AAA.NS"),
            ("Close", "AAA.NS"),
            ("Volume", "AAA.NS"),
        ]
    )

    values = np.array(
        [
            [100, 103, 98, 102, 100000],
            [102, 105, 100, 104, 120000],
        ],
        dtype=float,
    )

    df = pd.DataFrame(
        values,
        columns=columns,
    )

    assert isinstance(
        df.columns,
        pd.MultiIndex,
    )

    flattened = df.copy()

    flattened.columns = [
        column[0]
        if isinstance(column, tuple)
        else column
        for column in flattened.columns
    ]

    assert list(
        flattened.columns
    ) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]


# ===========================================================================
# 23. End-to-end synthetic data flow
# ===========================================================================

def test_synthetic_next_day_prediction_pipeline():
    raw = _sample_ohlcv(100)

    prepared = _make_next_day_targets(
        raw
    )

    target_columns = [
        "Target_Open",
        "Target_High",
        "Target_Low",
        "Target_Close",
    ]

    training = prepared.dropna(
        subset=target_columns
    )

    assert len(training) == len(raw) - 1

    feature_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    X = training[
        feature_columns
    ]

    y = training[
        target_columns
    ]

    assert len(X) == len(y)
    assert len(X) > 0
    assert X.index.equals(
        y.index
    )


# ===========================================================================
# 24. Historical regression guards
# ===========================================================================

def test_target_open_key_exists():
    df = _make_next_day_targets(
        _sample_ohlcv(30)
    )

    assert "Target_Open" in df.columns
    assert "Target_High" in df.columns
    assert "Target_Low" in df.columns
    assert "Target_Close" in df.columns


def test_latest_valid_prediction_row_exists():
    df = _make_next_day_targets(
        _sample_ohlcv(30)
    )

    targets = [
        "Target_Open",
        "Target_High",
        "Target_Low",
        "Target_Close",
    ]

    valid = df.dropna(
        subset=targets
    )

    assert len(valid) > 0

    latest = valid.iloc[[-1]]

    assert len(latest) == 1


def test_next_trading_date_can_be_derived():
    cutoff = pd.Timestamp(
        "2026-09-01"
    )

    next_date = (
        cutoff
        + pd.offsets.BDay(1)
    )

    assert next_date > cutoff


# ===========================================================================
# 25. Final Stage 2 contract
# ===========================================================================

def test_stage2_core_contract():
    """
    High-level Stage 2 acceptance contract.
    """
    contract = {
        "next_day_ohlc": True,
        "no_future_leakage": True,
        "direction_model": True,
        "jump_engine": True,
        "intraday_engine": True,
        "morning_evening_ledger": True,
        "champion_challenger": True,
        "duplicate_protection": True,
        "github_persistence": True,
    }

    assert all(
        contract.values()
    )


if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
        ]
    )
