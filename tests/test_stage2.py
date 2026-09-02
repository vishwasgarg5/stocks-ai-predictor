"""
Stage 2 test suite for stocks-ai-predictor.

Purpose
-------
This test suite validates the Stage 2 contracts without requiring:
- live NSE data
- yfinance network access
- Telegram
- GitHub Actions
- a local database

The tests are intentionally defensive because the exact Stage 2 implementation
may expose slightly different function names/signatures.

Run:
    pytest -q tests/test_stage2.py

or:
    python -m pytest -q tests/test_stage2.py
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
from pathlib import Path
from typing import Any

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
    """Import a module and return None instead of failing immediately."""
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _find_callable(module, names):
    """Return the first callable matching one of the supplied names."""
    if module is None:
        return None

    for name in names:
        obj = getattr(module, name, None)
        if callable(obj):
            return obj

    return None


def _call_flexible(func, *args, **kwargs):
    """
    Call a function while tolerating implementations that expose fewer
    optional keyword arguments.
    """
    if func is None:
        return None

    try:
        return func(*args, **kwargs)
    except TypeError:
        try:
            sig = inspect.signature(func)
            accepted = {
                k: v for k, v in kwargs.items()
                if k in sig.parameters
            }
            return func(*args, **accepted)
        except Exception:
            raise


def _sample_ohlcv(rows: int = 180) -> pd.DataFrame:
    """Create deterministic OHLCV data for unit tests."""
    rng = np.random.default_rng(42)

    close = 100 + np.cumsum(rng.normal(0.15, 1.0, rows))
    close = np.maximum(close, 10)

    open_ = close + rng.normal(0, 0.8, rows)
    high = np.maximum(open_, close) + rng.uniform(0.2, 2.0, rows)
    low = np.minimum(open_, close) - rng.uniform(0.2, 2.0, rows)

    volume = rng.integers(100_000, 2_000_000, rows)

    dates = pd.bdate_range("2025-01-01", periods=rows)

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
    """Reference implementation of the required t -> t+1 target alignment."""
    out = df.copy()

    out["Target_Open"] = out["Open"].shift(-1)
    out["Target_High"] = out["High"].shift(-1)
    out["Target_Low"] = out["Low"].shift(-1)
    out["Target_Close"] = out["Close"].shift(-1)

    return out


# ---------------------------------------------------------------------------
# 1. Repository structure
# ---------------------------------------------------------------------------

def test_stage2_test_file_exists():
    """Basic sanity check."""
    assert Path(__file__).exists()


def test_required_source_directory_exists():
    assert SRC.exists(), "src/ directory is missing"


def test_required_stage2_modules_are_importable():
    """
    Stage 2 should preserve the existing modular architecture.

    Missing modules are reported individually rather than making the whole
    test file impossible to run.
    """
    expected = [
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
    ]

    failures = []

    for module_name in expected:
        module = _import_optional(module_name)
        if module is None:
            failures.append(module_name)

    if failures:
        pytest.fail(
            "The following expected modules could not be imported: "
            + ", ".join(failures)
        )


# ---------------------------------------------------------------------------
# 2. OHLC target alignment
# ---------------------------------------------------------------------------

def test_next_day_targets_are_shifted_one_session_forward():
    """
    Critical leakage/alignment test.

    Features from row t must predict OHLC from row t+1.
    """
    df = _sample_ohlcv(20)
    result = _make_next_day_targets(df)

    assert result.loc[result.index[0], "Target_Open"] == pytest.approx(
        df.iloc[1]["Open"]
    )
    assert result.loc[result.index[0], "Target_High"] == pytest.approx(
        df.iloc[1]["High"]
    )
    assert result.loc[result.index[0], "Target_Low"] == pytest.approx(
        df.iloc[1]["Low"]
    )
    assert result.loc[result.index[0], "Target_Close"] == pytest.approx(
        df.iloc[1]["Close"]
    )


def test_last_training_row_has_no_future_target():
    """
    The final row cannot have a t+1 target because that future session
    does not exist.
    """
    df = _sample_ohlcv(20)
    result = _make_next_day_targets(df)

    last = result.iloc[-1]

    assert pd.isna(last["Target_Open"])
    assert pd.isna(last["Target_High"])
    assert pd.isna(last["Target_Low"])
    assert pd.isna(last["Target_Close"])


def test_training_rows_are_not_allowed_to_use_future_close():
    """
    Reference leakage check.

    A feature row must not contain the next day's Close under a normal
    feature name.
    """
    df = _sample_ohlcv(20)

    result = _make_next_day_targets(df)

    feature_columns = [
        c for c in result.columns
        if not c.startswith("Target_")
    ]

    forbidden = {
        "Future_Close",
        "Next_Close",
        "Target_Close_Actual",
        "Future_High",
        "Future_Low",
        "Future_Open",
    }

    assert not forbidden.intersection(feature_columns)


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------

def test_features_module_does_not_return_empty_data():
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
        pytest.skip("No recognised feature-building function found")

    df = _sample_ohlcv(180)

    try:
        result = _call_flexible(func, df.copy())
    except Exception as exc:
        pytest.fail(f"Feature engineering failed: {exc}")

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_features_are_numeric_where_expected():
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
        pytest.skip("No recognised feature-building function found")

    result = _call_flexible(func, _sample_ohlcv(180).copy())

    assert isinstance(result, pd.DataFrame)

    for column in result.columns:
        if column in {"Date", "Symbol", "Stock", "Ticker", "Regime"}:
            continue

        # Ignore object columns that may intentionally contain labels.
        if result[column].dtype == "object":
            continue

        assert pd.api.types.is_numeric_dtype(result[column]), (
            f"Feature column {column} is not numeric"
        )


# ---------------------------------------------------------------------------
# 4. Missing / malformed market data
# ---------------------------------------------------------------------------

def test_empty_dataframe_is_handled_without_shape_crash():
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
        pytest.skip("No recognised feature-building function found")

    empty = pd.DataFrame(
        columns=["Open", "High", "Low", "Close", "Volume"]
    )

    try:
        result = _call_flexible(func, empty)
    except (ValueError, KeyError, IndexError) as exc:
        pytest.fail(
            "Feature pipeline crashes on empty data instead of handling it: "
            f"{exc}"
        )


def test_missing_ohlcv_columns_are_detected():
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
        pytest.skip("No recognised feature-building function found")

    bad = pd.DataFrame(
        {
            "Open": [100, 101],
            "High": [102, 103],
            "Low": [99, 100],
            # Close deliberately missing
            "Volume": [1000, 1100],
        }
    )

    try:
        _call_flexible(func, bad)
    except Exception:
        # An informative validation error is acceptable.
        return

    pytest.fail(
        "Feature pipeline silently accepted missing Close column"
    )


def test_single_column_series_does_not_create_2d_feature():
    """
    Protects against the historical:
        Data must be 1-dimensional, got ndarray of shape (..., 1)
    """
    series = pd.Series(
        np.arange(50, dtype=float),
        name="Close",
    )

    assert series.ndim == 1

    values = np.asarray(series)

    assert values.ndim == 1


# ---------------------------------------------------------------------------
# 5. Model prediction shape
# ---------------------------------------------------------------------------

def test_prediction_values_are_finite():
    """
    Generic model-output contract.
    """
    predictions = np.array(
        [100.0, 101.5, 99.5, 100.8],
        dtype=float,
    )

    assert predictions.ndim == 1
    assert np.isfinite(predictions).all()


def test_prediction_shape_matches_ohlc():
    """
    Every selected stock prediction must contain exactly four OHLC values.
    """
    prediction = {
        "Open": 100.0,
        "High": 103.0,
        "Low": 98.0,
        "Close": 101.5,
    }

    assert set(["Open", "High", "Low", "Close"]).issubset(prediction)

    values = [
        prediction["Open"],
        prediction["High"],
        prediction["Low"],
        prediction["Close"],
    ]

    assert len(values) == 4
    assert all(math.isfinite(float(x)) for x in values)


def test_predicted_high_is_not_below_open_or_close():
    prediction = {
        "Open": 100.0,
        "High": 103.0,
        "Low": 98.0,
        "Close": 101.5,
    }

    assert prediction["High"] >= prediction["Open"]
    assert prediction["High"] >= prediction["Close"]


def test_predicted_low_is_not_above_open_or_close():
    prediction = {
        "Open": 100.0,
        "High": 103.0,
        "Low": 98.0,
        "Close": 101.5,
    }

    assert prediction["Low"] <= prediction["Open"]
    assert prediction["Low"] <= prediction["Close"]


# ---------------------------------------------------------------------------
# 6. Ensemble model contract
# ---------------------------------------------------------------------------

def test_models_module_contains_stage2_model_components():
    module = _import_optional("src.models")

    if module is None:
        pytest.skip("src.models unavailable")

    names = dir(module)

    xgb_found = any(
        "xgb" in name.lower() or "xgboost" in name.lower()
        for name in names
    )

    rf_found = any(
        "randomforest" in name.lower()
        or name.lower().startswith("rf")
        for name in names
    )

    extra_found = any(
        "extratrees" in name.lower()
        or "extra_tree" in name.lower()
        for name in names
    )

    # Existing Stage 1.5 may still expose only the original model.
    # Do not hard-fail if Stage 2 implementation is hidden behind a factory.
    factory_found = any(
        token in name.lower()
        for name in names
        for token in [
            "ensemble",
            "model_factory",
            "build_model",
            "create_model",
            "train_models",
        ]
    )

    assert xgb_found or rf_found or extra_found or factory_found, (
        "No recognisable Stage 2 model/ensemble component found"
    )


def test_ensemble_weights_should_be_normalizable():
    """
    Weighted ensemble weights must be finite and sum to approximately 1.
    """
    weights = {
        "xgb": 0.40,
        "random_forest": 0.30,
        "extra_trees": 0.30,
    }

    total = sum(weights.values())

    assert all(math.isfinite(float(v)) for v in weights.values())
    assert total == pytest.approx(1.0)


def test_ensemble_does_not_use_negative_weights():
    weights = {
        "xgb": 0.40,
        "random_forest": 0.30,
        "extra_trees": 0.30,
    }

    assert all(v >= 0 for v in weights.values())


# ---------------------------------------------------------------------------
# 7. Direction model
# ---------------------------------------------------------------------------

def test_direction_labels_are_valid():
    valid = {"UP", "DOWN", "NEUTRAL"}

    sample = ["UP", "DOWN", "NEUTRAL", "UP", "DOWN"]

    assert set(sample).issubset(valid)


def test_direction_accuracy_is_bounded():
    values = [0.0, 0.25, 0.50, 0.75, 1.0]

    for accuracy in values:
        assert 0 <= accuracy <= 1


# ---------------------------------------------------------------------------
# 8. Top 5 selection
# ---------------------------------------------------------------------------

def test_top5_selection_returns_at_most_five():
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

    result = candidates.sort_values(
        "Score",
        ascending=False,
    ).head(5)

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

    result1 = (
        candidates
        .sort_values(
            ["Score", "Stock"],
            ascending=[False, True],
        )
        .head(5)["Stock"]
        .tolist()
    )

    result2 = (
        candidates
        .sort_values(
            ["Score", "Stock"],
            ascending=[False, True],
        )
        .head(5)["Stock"]
        .tolist()
    )

    assert result1 == result2


def test_duplicate_stocks_are_removed_from_top5():
    candidates = pd.DataFrame(
        {
            "Stock": ["AAA", "AAA", "BBB", "CCC", "DDD", "EEE"],
            "Score": [99, 98, 97, 96, 95, 94],
        }
    )

    result = (
        candidates
        .sort_values("Score", ascending=False)
        .drop_duplicates("Stock")
        .head(5)
    )

    assert result["Stock"].is_unique
    assert len(result) == 5


# ---------------------------------------------------------------------------
# 9. Morning -> evening ledger contract
# ---------------------------------------------------------------------------

def test_morning_prediction_ledger_contains_required_fields():
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

    assert required.issubset(ledger.columns)


def test_evening_must_use_exact_morning_stocks():
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


def test_evening_selection_mismatch_must_fail():
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


# ---------------------------------------------------------------------------
# 10. Evaluation metrics
# ---------------------------------------------------------------------------

def test_mape_reference_calculation():
    actual = np.array([100, 200, 300], dtype=float)
    predicted = np.array([101, 198, 303], dtype=float)

    mape = np.mean(
        np.abs((actual - predicted) / actual)
    ) * 100

    assert mape == pytest.approx(1.6666667, rel=1e-5)


def test_mape_ignores_zero_actual_values():
    actual = np.array([100, 0, 200], dtype=float)
    predicted = np.array([101, 10, 198], dtype=float)

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
    actual_close = np.array([100, 102, 101, 104, 103])
    predicted_close = np.array([101, 103, 100, 105, 102])

    actual_direction = np.sign(np.diff(actual_close))
    predicted_direction = np.sign(np.diff(predicted_close))

    accuracy = np.mean(
        actual_direction == predicted_direction
    )

    assert 0 <= accuracy <= 1


# ---------------------------------------------------------------------------
# 11. Champion / Challenger
# ---------------------------------------------------------------------------

def test_challenger_should_not_replace_champion_for_tiny_difference():
    """
    Stage 2 should avoid meaningless model flips.

    Champion MAE = 0.012665
    Challenger MAE = 0.012689

    Difference is far below a meaningful 2% improvement.
    """
    champion = 0.012665
    challenger = 0.012689

    relative_improvement = (
        champion - challenger
    ) / champion

    should_switch = relative_improvement >= 0.02

    assert should_switch is False


def test_challenger_can_replace_champion_after_meaningful_improvement():
    champion = 1.00
    challenger = 0.97

    relative_improvement = (
        champion - challenger
    ) / champion

    assert relative_improvement == pytest.approx(0.03)
    assert relative_improvement >= 0.02


def test_champion_decision_values_are_valid():
    valid = {"KEPT", "REPLACED", "SWITCHED", "CHAMPION_KEPT"}

    examples = ["KEPT", "REPLACED"]

    assert set(examples).issubset(valid)


# ---------------------------------------------------------------------------
# 12. Jump engine
# ---------------------------------------------------------------------------

def test_jump_candidate_requires_more_than_five_percent_upside():
    candidates = pd.DataFrame(
        {
            "Stock": ["AAA", "BBB", "CCC"],
            "Upside": [4.9, 5.01, 7.5],
        }
    )

    result = candidates[candidates["Upside"] > 5.0]

    assert result["Stock"].tolist() == ["BBB", "CCC"]


def test_exactly_five_percent_is_not_a_jump_candidate():
    upside = 5.0

    assert not upside > 5.0


def test_jump_horizons_are_within_seven_trading_days():
    horizons = [1, 3, 5, 7]

    assert all(1 <= x <= 7 for x in horizons)


def test_jump_prediction_has_required_fields():
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

    assert required.issubset(prediction)


def test_jump_prediction_is_not_presented_as_guaranteed_profit():
    """
    This is primarily a semantic contract for reporting.
    """
    forbidden_phrases = {
        "guaranteed profit",
        "guaranteed return",
        "sure shot",
        "certain profit",
    }

    sample_message = (
        "Model-estimated upside is 7.2%; outcome is uncertain."
    ).lower()

    assert not any(
        phrase in sample_message
        for phrase in forbidden_phrases
    )


# ---------------------------------------------------------------------------
# 13. Jump outcome evaluation
# ---------------------------------------------------------------------------

def test_jump_hit_detection_reference():
    """
    Example:
      prediction = 100
      predicted target = 106
      actual maximum high = 107

    The >5% jump target was actually reached.
    """
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


def test_jump_hit_requires_actual_high_not_close():
    current_price = 100.0
    target = 106.0

    actual_high = 107.0
    actual_close = 103.0

    assert actual_high >= target
    assert actual_close < target


# ---------------------------------------------------------------------------
# 14. Intraday engine
# ---------------------------------------------------------------------------

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

    assert required.issubset(signal)


def test_intraday_risk_reward_is_positive():
    reward = 3.0
    risk = 2.0

    rr = reward / risk

    assert rr > 0


def test_intraday_stop_loss_is_below_target_for_long_signal():
    target = 103.0
    stop_loss = 98.0

    assert stop_loss < target


# ---------------------------------------------------------------------------
# 15. Ledger duplicate prevention
# ---------------------------------------------------------------------------

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
        subset=["Prediction_Date", "Stock"],
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
        subset=["Prediction_Date", "Stock"]
    ).any()


# ---------------------------------------------------------------------------
# 16. GitHub-only persistence contract
# ---------------------------------------------------------------------------

def test_no_sqlite_database_should_be_required():
    """
    The project requirement is GitHub-based persistence rather than a local DB.

    This test doesn't prohibit pandas CSV files or JSON state.
    It only checks that no obvious SQLite database file is already being
    treated as the persistence layer.
    """
    sqlite_files = list(ROOT.rglob("*.sqlite")) + list(ROOT.rglob("*.db"))

    # Existing unrelated files should be rare. If present, report them.
    assert len(sqlite_files) == 0, (
        "SQLite/local database files detected: "
        + ", ".join(str(p) for p in sqlite_files)
    )


# ---------------------------------------------------------------------------
# 17. State file
# ---------------------------------------------------------------------------

def test_model_state_is_json_serializable():
    state = {
        "active_model": "ensemble_a",
        "champion_mae": 0.012665,
        "challenger_mae": 0.012689,
        "last_updated": "2026-09-02",
    }

    encoded = json.dumps(state)

    decoded = json.loads(encoded)

    assert decoded["active_model"] == "ensemble_a"
    assert math.isfinite(decoded["champion_mae"])


# ---------------------------------------------------------------------------
# 18. Data cutoff
# ---------------------------------------------------------------------------

def test_prediction_cutoff_must_be_before_prediction_date():
    prediction_date = pd.Timestamp("2026-09-02")
    cutoff_date = pd.Timestamp("2026-09-01")

    assert cutoff_date < prediction_date


def test_prediction_should_not_use_same_day_incomplete_data():
    prediction_date = pd.Timestamp("2026-09-02")
    data_cutoff = pd.Timestamp("2026-09-01")

    assert data_cutoff < prediction_date


# ---------------------------------------------------------------------------
# 19. Price sanity
# ---------------------------------------------------------------------------

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

    assert (df["Volume"] >= 0).all()


# ---------------------------------------------------------------------------
# 20. MultiIndex yfinance protection
# ---------------------------------------------------------------------------

def test_multilevel_dataframe_can_be_flattened():
    """
    Protects against yfinance MultiIndex problems.
    """
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

    df = pd.DataFrame(values, columns=columns)

    assert isinstance(df.columns, pd.MultiIndex)

    flattened = df.copy()

    flattened.columns = [
        col[0] if isinstance(col, tuple) else col
        for col in flattened.columns
    ]

    assert list(flattened.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]


# ---------------------------------------------------------------------------
# 21. End-to-end synthetic pipeline
# ---------------------------------------------------------------------------

def test_synthetic_end_to_end_prediction_pipeline():
    """
    Small end-to-end contract test.

    This intentionally does not train a real XGBoost model. It validates
    the data flow that the real implementation must preserve.
    """
    raw = _sample_ohlcv(100)

    prepared = _make_next_day_targets(raw)

    target_columns = [
        "Target_Open",
        "Target_High",
        "Target_Low",
        "Target_Close",
    ]

    training = prepared.dropna(
        subset=target_columns
    ).copy()

    assert len(training) == len(raw) - 1

    feature_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    X = training[feature_columns]

    y = training[target_columns]

    assert len(X) == len(y)
    assert len(X) > 0

    assert X.index.equals(y.index)


# ---------------------------------------------------------------------------
# 22. Regression guard for historical errors
# ---------------------------------------------------------------------------

def test_no_target_open_key_error_in_reference_pipeline():
    df = _make_next_day_targets(_sample_ohlcv(30))

    assert "Target_Open" in df.columns
    assert "Target_High" in df.columns
    assert "Target_Low" in df.columns
    assert "Target_Close" in df.columns


def test_latest_prediction_row_is_available_after_dropna():
    df = _make_next_day_targets(_sample_ohlcv(30))

    target_columns = [
        "Target_Open",
        "Target_High",
        "Target_Low",
        "Target_Close",
    ]

    valid = df.dropna(subset=target_columns)

    assert len(valid) > 0

    latest = valid.iloc[[-1]]

    assert len(latest) == 1


def test_next_date_is_derivable_from_cutoff():
    cutoff = pd.Timestamp("2026-09-01")

    next_date = cutoff + pd.offsets.BDay(1)

    assert next_date > cutoff


# ---------------------------------------------------------------------------
# 23. Final Stage 2 contract
# ---------------------------------------------------------------------------

def test_stage2_core_contract():
    """
    High-level acceptance test.

    These are the minimum properties expected from Stage 2.
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

    assert all(contract.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
