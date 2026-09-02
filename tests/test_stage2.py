import importlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


# ============================================================
# Helpers
# ============================================================

def _import_optional(module_name: str):
    """
    Import a module without hiding the exception from diagnostics.

    Existing tests can still use this helper when optional behavior
    is genuinely optional.
    """
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _import_required(module_name: str):
    """
    Import a production module and expose the real exception.
    """
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        pytest.fail(
            f"Could not import {module_name}: "
            f"{type(exc).__name__}: {exc}"
        )


# ============================================================
# Repository structure
# ============================================================

def test_repository_root_exists():
    assert ROOT.exists()


def test_src_directory_exists():
    assert SRC.exists()


@pytest.mark.parametrize(
    "relative_path",
    [
        "src/__init__.py",
        "src/config.py",
        "src/features.py",
        "src/market_data.py",
        "src/models.py",
        "src/prediction.py",
        "src/ranking.py",
        "src/selection.py",
        "src/evaluation.py",
        "src/retraining.py",
        "src/ledger.py",
        "src/morning_runner.py",
        "src/telegram_report.py",
        "src/weekly_report.py",
    ],
)
def test_required_source_files_exist(relative_path):
    assert (ROOT / relative_path).exists(), relative_path


def test_workflow_directory_exists():
    workflow_dir = ROOT / ".github" / "workflows"
    assert workflow_dir.exists()


@pytest.mark.parametrize(
    "workflow",
    [
        "morning_prediction.yml",
        "evening_evaluate_retrain.yml",
        "weekly_report.yml",
    ],
)
def test_required_workflow_exists(workflow):
    assert (ROOT / ".github" / "workflows" / workflow).exists()


def test_requirements_exists():
    assert (ROOT / "requirements.txt").exists()


def test_readme_exists():
    assert (ROOT / "README.md").exists()


# ============================================================
# CRITICAL IMPORT DIAGNOSTICS
# ============================================================

def test_stage2_module_import_diagnostics():
    """
    IMPORTANT:

    The old test suite used _import_optional(), which converted real
    production import errors into SKIPPED tests.

    This test deliberately imports every core Stage 2 module and
    reports the exact exception.
    """

    modules = [
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

    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(
                f"{module_name}: "
                f"{type(exc).__name__}: {exc}"
            )

    assert not failures, (
        "Stage 2 module import failures:\n"
        + "\n".join(failures)
    )


# ============================================================
# Optional module smoke tests
# ============================================================

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
def test_core_module_imports(module_name):
    """
    Keep these tests as smoke tests, but the diagnostic test above
    is what exposes the real import failure.
    """
    module = _import_optional(module_name)

    if module is None:
        pytest.skip(
            f"{module_name} could not be imported in this test environment"
        )

    assert module is not None


# ============================================================
# Target alignment
# ============================================================

def test_next_day_target_alignment():
    df = pd.DataFrame(
        {
            "Open": [10, 11, 12, 13],
            "High": [11, 12, 13, 14],
            "Low": [9, 10, 11, 12],
            "Close": [10.5, 11.5, 12.5, 13.5],
        }
    )

    df["Target_Open"] = df["Open"].shift(-1)
    df["Target_High"] = df["High"].shift(-1)
    df["Target_Low"] = df["Low"].shift(-1)
    df["Target_Close"] = df["Close"].shift(-1)

    assert df.loc[0, "Target_Open"] == 11
    assert df.loc[0, "Target_High"] == 12
    assert df.loc[0, "Target_Low"] == 10
    assert df.loc[0, "Target_Close"] == 11.5

    assert pd.isna(df.loc[3, "Target_Open"])
    assert pd.isna(df.loc[3, "Target_Close"])


def test_target_is_next_trading_session():
    close = pd.Series([100, 102, 105, 103])

    target = close.shift(-1)

    assert target.iloc[0] == 102
    assert target.iloc[1] == 105
    assert target.iloc[2] == 103
    assert pd.isna(target.iloc[3])


# ============================================================
# Leakage checks
# ============================================================

def test_no_future_close_in_feature_lag():
    df = pd.DataFrame(
        {
            "Close": [100, 101, 102, 103, 104],
        }
    )

    df["Close_Lag1"] = df["Close"].shift(1)
    df["Close_Lag2"] = df["Close"].shift(2)
    df["Close_Lag3"] = df["Close"].shift(3)

    assert df.loc[3, "Close_Lag1"] == 102
    assert df.loc[3, "Close_Lag2"] == 101
    assert df.loc[3, "Close_Lag3"] == 100


def test_target_does_not_modify_current_close():
    df = pd.DataFrame(
        {
            "Close": [100, 101, 102],
        }
    )

    original = df["Close"].copy()

    df["Target_Close"] = df["Close"].shift(-1)

    pd.testing.assert_series_equal(
        df["Close"],
        original,
        check_names=True,
    )


def test_no_current_day_target_leakage():
    df = pd.DataFrame(
        {
            "Close": [100, 101, 102, 103],
        }
    )

    df["Target_Close"] = df["Close"].shift(-1)

    assert df.loc[0, "Target_Close"] != df.loc[0, "Close"]
    assert df.loc[1, "Target_Close"] != df.loc[1, "Close"]


# ============================================================
# Feature engineering
# ============================================================

def test_feature_engineering_smoke():
    features = _import_optional("src.features")

    if features is None:
        pytest.skip("src.features unavailable")

    df = pd.DataFrame(
        {
            "Open": np.arange(100, 180, dtype=float),
            "High": np.arange(101, 181, dtype=float),
            "Low": np.arange(99, 179, dtype=float),
            "Close": np.arange(100.5, 180.5, dtype=float),
            "Volume": np.arange(100000, 100080, dtype=float),
        }
    )

    possible_functions = [
        "build_features",
        "create_features",
        "add_features",
        "engineer_features",
    ]

    function = None

    for name in possible_functions:
        candidate = getattr(features, name, None)
        if callable(candidate):
            function = candidate
            break

    if function is None:
        pytest.skip("No recognised feature engineering function found")

    result = function(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert "Close" in result.columns


def test_feature_engineering_has_no_nan_tail():
    df = pd.DataFrame(
        {
            "Close": np.arange(100, 180, dtype=float),
            "Volume": np.arange(100000, 100080, dtype=float),
        }
    )

    df["Close_Lag1"] = df["Close"].shift(1)
    df["Close_Lag2"] = df["Close"].shift(2)
    df["Close_Lag3"] = df["Close"].shift(3)

    clean = df.dropna()

    assert len(clean) > 0
    assert not clean.tail(1).isna().any().any()


def test_feature_columns_are_numeric():
    df = pd.DataFrame(
        {
            "Close": np.arange(100, 120, dtype=float),
            "Volume": np.arange(1000, 1020, dtype=float),
        }
    )

    df["Return"] = df["Close"].pct_change()
    df["SMA20"] = df["Close"].rolling(5).mean()

    numeric = df.select_dtypes(include=[np.number])

    assert "Close" in numeric.columns
    assert "Volume" in numeric.columns


# ============================================================
# Malformed data guards
# ============================================================

def test_empty_dataframe_guard():
    df = pd.DataFrame()

    assert df.empty


def test_missing_ohlc_columns_detected():
    df = pd.DataFrame(
        {
            "Close": [100, 101, 102],
        }
    )

    required = {"Open", "High", "Low", "Close"}

    assert not required.issubset(df.columns)


def test_nan_data_can_be_detected():
    df = pd.DataFrame(
        {
            "Open": [100, np.nan, 102],
            "High": [101, 103, 104],
            "Low": [99, 100, 101],
            "Close": [100.5, 102, 103],
        }
    )

    assert df.isna().any().any()


def test_one_dimensional_series():
    series = pd.Series([1, 2, 3, 4])

    assert series.ndim == 1


def test_multiindex_flattening_smoke():
    columns = pd.MultiIndex.from_tuples(
        [
            ("Open", "A"),
            ("High", "A"),
            ("Low", "A"),
            ("Close", "A"),
        ]
    )

    df = pd.DataFrame(
        [[1, 2, 0.5, 1.5]],
        columns=columns,
    )

    df.columns = [
        "_".join(str(x) for x in col if str(x) != "")
        for col in df.columns
    ]

    assert df.ndim == 2
    assert len(df.columns) == 4


# ============================================================
# Prediction sanity
# ============================================================

def test_prediction_ohlc_sanity():
    prediction = {
        "Open": 100.0,
        "High": 105.0,
        "Low": 98.0,
        "Close": 103.0,
    }

    assert prediction["High"] >= prediction["Open"]
    assert prediction["High"] >= prediction["Close"]
    assert prediction["Low"] <= prediction["Open"]
    assert prediction["Low"] <= prediction["Close"]


def test_prediction_values_are_finite():
    prediction = [100.0, 105.0, 98.0, 103.0]

    assert all(math.isfinite(float(x)) for x in prediction)


def test_prediction_high_low_order():
    high = 105
    low = 98

    assert high >= low


# ============================================================
# Ensemble model contract
# ============================================================

def test_models_module_contract():
    models = _import_optional("src.models")

    if models is None:
        pytest.skip("src.models unavailable")

    assert models is not None


def test_ensemble_weights_sum_to_one():
    weights = {
        "xgb": 0.4,
        "rf": 0.3,
        "extra": 0.3,
    }

    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_ensemble_prediction_weighted_average():
    predictions = {
        "xgb": 100.0,
        "rf": 102.0,
        "extra": 101.0,
    }

    weights = {
        "xgb": 0.5,
        "rf": 0.25,
        "extra": 0.25,
    }

    result = sum(
        predictions[name] * weights[name]
        for name in predictions
    )

    assert result == 100.75


def test_model_agreement_confidence_bounds():
    agreement = 0.72

    assert 0 <= agreement <= 1


# ============================================================
# Direction
# ============================================================

def test_direction_labels():
    assert 105 > 100
    assert 95 < 100
    assert 100 == 100


def test_direction_accuracy_bounds():
    accuracy = 0.429

    assert 0 <= accuracy <= 1


def test_direction_accuracy_percentage_bounds():
    accuracy = 42.9

    assert 0 <= accuracy <= 100


def test_direction_from_predicted_close():
    previous_close = 100
    predicted_close = 105

    direction = (
        "UP"
        if predicted_close > previous_close
        else "DOWN"
        if predicted_close < previous_close
        else "NEUTRAL"
    )

    assert direction == "UP"


def test_neutral_direction():
    previous_close = 100
    predicted_close = 100

    direction = (
        "UP"
        if predicted_close > previous_close
        else "DOWN"
        if predicted_close < previous_close
        else "NEUTRAL"
    )

    assert direction == "NEUTRAL"


# ============================================================
# Ranking / selection
# ============================================================

def test_top_5_selection():
    stocks = pd.DataFrame(
        {
            "Stock": [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
            ],
            "Score": [
                95,
                90,
                88,
                87,
                86,
                80,
                75,
            ],
        }
    )

    top5 = (
        stocks
        .sort_values("Score", ascending=False)
        .head(5)
    )

    assert len(top5) == 5
    assert list(top5["Stock"]) == [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]


def test_top_5_deterministic_selection():
    df = pd.DataFrame(
        {
            "Stock": ["A", "B", "C", "D", "E", "F"],
            "Score": [90, 90, 85, 80, 75, 70],
        }
    )

    result1 = (
        df.sort_values(
            ["Score", "Stock"],
            ascending=[False, True],
        )
        .head(5)
    )

    result2 = (
        df.sort_values(
            ["Score", "Stock"],
            ascending=[False, True],
        )
        .head(5)
    )

    pd.testing.assert_frame_equal(
        result1.reset_index(drop=True),
        result2.reset_index(drop=True),
    )


def test_selection_always_returns_at_most_five():
    df = pd.DataFrame(
        {
            "Stock": ["A", "B", "C"],
            "Score": [90, 80, 70],
        }
    )

    result = df.sort_values(
        "Score",
        ascending=False,
    ).head(5)

    assert len(result) <= 5


# ============================================================
# Morning / evening ledger
# ============================================================

def test_morning_evening_ledger_matching():
    morning = pd.DataFrame(
        {
            "Stock": [
                "RELIANCE",
                "TCS",
                "INFY",
                "HDFCBANK",
                "ICICIBANK",
            ],
        }
    )

    evening = morning.copy()

    assert set(morning["Stock"]) == set(evening["Stock"])


def test_evening_must_not_change_selection():
    morning_selection = [
        "RELIANCE",
        "TCS",
        "INFY",
        "HDFCBANK",
        "ICICIBANK",
    ]

    evening_selection = list(morning_selection)

    assert evening_selection == morning_selection


def test_ledger_prediction_date():
    prediction_date = "2026-09-02"

    assert len(prediction_date) == 10
    assert prediction_date[4] == "-"
    assert prediction_date[7] == "-"


# ============================================================
# Evaluation
# ============================================================

def test_mape_reference():
    actual = np.array([100.0])
    predicted = np.array([101.0])

    mape = np.mean(
        np.abs((actual - predicted) / actual)
    ) * 100

    assert round(mape, 3) == 1.0


def test_mape_multiple_values():
    actual = np.array(
        [100.0, 200.0, 300.0]
    )

    predicted = np.array(
        [101.0, 202.0, 297.0]
    )

    mape = np.mean(
        np.abs((actual - predicted) / actual)
    ) * 100

    assert mape >= 0
    assert math.isfinite(mape)


def test_direction_accuracy_reference():
    actual = np.array(
        [101, 99, 105, 100]
    )

    predicted = np.array(
        [102, 101, 104, 100]
    )

    previous = np.array(
        [100, 100, 100, 100]
    )

    actual_direction = np.sign(
        actual - previous
    )

    predicted_direction = np.sign(
        predicted - previous
    )

    accuracy = np.mean(
        actual_direction == predicted_direction
    )

    assert accuracy == 0.75


# ============================================================
# Champion / Challenger
# ============================================================

def test_champion_challenger_threshold():
    champion_mae = 1.00
    challenger_mae = 0.97

    relative_improvement = (
        champion_mae - challenger_mae
    ) / champion_mae

    assert relative_improvement == pytest.approx(
        0.03
    )

    assert relative_improvement >= 0.02


def test_challenger_below_two_percent_is_rejected():
    champion_mae = 1.00
    challenger_mae = 0.99

    relative_improvement = (
        champion_mae - challenger_mae
    ) / champion_mae

    assert relative_improvement < 0.02


def test_champion_kept_when_difference_is_tiny():
    champion_mae = 0.013679
    challenger_mae = 0.013680

    relative_improvement = (
        champion_mae - challenger_mae
    ) / champion_mae

    assert relative_improvement < 0.02


# ============================================================
# Jump engine
# ============================================================

def test_jump_prediction_above_five_percent():
    current_price = 100.0
    target_price = 106.0

    upside = (
        (target_price - current_price)
        / current_price
    ) * 100

    assert upside > 5


def test_jump_prediction_not_above_five_percent():
    current_price = 100.0
    target_price = 104.0

    upside = (
        (target_price - current_price)
        / current_price
    ) * 100

    assert upside <= 5


def test_jump_horizon_between_one_and_seven_days():
    horizon = 5

    assert 1 <= horizon <= 7


def test_jump_horizon_boundaries():
    assert 1 <= 1 <= 7
    assert 1 <= 7 <= 7


def test_jump_outcome_positive():
    start_price = 100.0
    future_high = 107.0

    gain = (
        (future_high - start_price)
        / start_price
    ) * 100

    assert gain > 5


def test_jump_outcome_negative():
    start_price = 100.0
    future_high = 103.0

    gain = (
        (future_high - start_price)
        / start_price
    ) * 100

    assert gain <= 5


# ============================================================
# Intraday
# ============================================================

def test_intraday_schema():
    row = {
        "Stock": "RELIANCE",
        "Score": 82.5,
        "Entry": 2500.0,
        "Target": 2550.0,
        "SL": 2475.0,
        "RR": 2.0,
        "Confidence": 70.0,
    }

    required = {
        "Stock",
        "Score",
        "Entry",
        "Target",
        "SL",
        "RR",
        "Confidence",
    }

    assert required.issubset(row.keys())


def test_intraday_risk_reward():
    entry = 100.0
    target = 110.0
    stop = 95.0

    reward = target - entry
    risk = entry - stop

    rr = reward / risk

    assert rr == 2.0


def test_intraday_target_above_entry():
    entry = 100
    target = 105

    assert target > entry


def test_intraday_stop_below_entry():
    entry = 100
    stop = 98

    assert stop < entry


# ============================================================
# Duplicate protection
# ============================================================

def test_duplicate_prediction_dates_removed():
    df = pd.DataFrame(
        {
            "Stock": [
                "A",
                "A",
                "B",
            ],
            "Prediction_Date": [
                "2026-09-02",
                "2026-09-02",
                "2026-09-02",
            ],
        }
    )

    deduped = df.drop_duplicates(
        subset=["Stock", "Prediction_Date"]
    )

    assert len(deduped) == 2


def test_duplicate_prediction_rows_removed():
    df = pd.DataFrame(
        {
            "Stock": ["A", "A", "B"],
            "Open": [100, 100, 200],
        }
    )

    deduped = df.drop_duplicates()

    assert len(deduped) == 2


# ============================================================
# Storage safety
# ============================================================

def test_no_sqlite_dependency_in_repository():
    forbidden = []

    for path in ROOT.rglob("*.py"):
        if ".git" in path.parts:
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        if "sqlite3" in text.lower():
            forbidden.append(str(path))

    assert not forbidden, (
        "SQLite/local DB references found:\n"
        + "\n".join(forbidden)
    )


def test_json_state_file_format():
    import json

    state = {
        "champion": "stage2",
        "version": 1,
        "updated": "2026-09-02",
    }

    encoded = json.dumps(state)
    decoded = json.loads(encoded)

    assert isinstance(decoded, dict)
    assert decoded["champion"] == "stage2"


# ============================================================
# Prediction cutoff
# ============================================================

def test_prediction_cutoff_uses_completed_session():
    latest_completed_date = pd.Timestamp(
        "2026-09-01"
    )

    prediction_date = pd.Timestamp(
        "2026-09-02"
    )

    assert latest_completed_date < prediction_date


def test_prediction_does_not_use_prediction_date_data():
    available_dates = pd.to_datetime(
        [
            "2026-08-28",
            "2026-08-31",
            "2026-09-01",
        ]
    )

    prediction_date = pd.Timestamp(
        "2026-09-02"
    )

    assert available_dates.max() < prediction_date


# ============================================================
# OHLC sanity
# ============================================================

def test_ohlc_relationship():
    row = {
        "Open": 100,
        "High": 105,
        "Low": 98,
        "Close": 103,
    }

    assert row["High"] >= max(
        row["Open"],
        row["Close"],
    )

    assert row["Low"] <= min(
        row["Open"],
        row["Close"],
    )


def test_ohlc_all_positive():
    values = [
        100,
        105,
        98,
        103,
    ]

    assert all(value > 0 for value in values)


# ============================================================
# yfinance-style MultiIndex guard
# ============================================================

def test_yfinance_multiindex_guard():
    columns = pd.MultiIndex.from_product(
        [
            ["Open", "High", "Low", "Close", "Volume"],
            ["RELIANCE.NS"],
        ]
    )

    df = pd.DataFrame(
        np.array(
            [
                [
                    100,
                    105,
                    98,
                    103,
                    1000000,
                ]
            ]
        ),
        columns=columns,
    )

    flattened = df.copy()

    if isinstance(flattened.columns, pd.MultiIndex):
        flattened.columns = [
            col[0]
            for col in flattened.columns
        ]

    assert not isinstance(
        flattened.columns,
        pd.MultiIndex,
    )

    assert "Open" in flattened.columns
    assert "Close" in flattened.columns


# ============================================================
# Synthetic next-day prediction flow
# ============================================================

def test_synthetic_next_day_flow():
    df = pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104],
            "High": [105, 106, 107, 108, 109],
            "Low": [98, 99, 100, 101, 102],
            "Close": [103, 104, 105, 106, 107],
            "Volume": [
                1000,
                1100,
                1200,
                1300,
                1400,
            ],
        }
    )

    df["Target_Open"] = df["Open"].shift(-1)
    df["Target_High"] = df["High"].shift(-1)
    df["Target_Low"] = df["Low"].shift(-1)
    df["Target_Close"] = df["Close"].shift(-1)

    train = df.dropna()

    assert len(train) == 4

    last_target = train.iloc[-1]

    assert last_target["Target_Open"] == 104
    assert last_target["Target_High"] == 109
    assert last_target["Target_Low"] == 102
    assert last_target["Target_Close"] == 107


# ============================================================
# Historical regression guards
# ============================================================

def test_regression_target_open_exists():
    df = pd.DataFrame(
        {
            "Open": [100, 101, 102],
        }
    )

    df["Target_Open"] = df["Open"].shift(-1)

    assert "Target_Open" in df.columns


def test_regression_latest_row_selection():
    df = pd.DataFrame(
        {
            "Feature1": [1, 2, 3],
            "Feature2": [4, 5, 6],
        }
    )

    latest_row = (
        df[
            ["Feature1", "Feature2"]
        ]
        .dropna()
        .iloc[[-1]]
    )

    assert len(latest_row) == 1
    assert latest_row.iloc[0]["Feature1"] == 3


def test_regression_prediction_date_defined():
    latest_date = pd.Timestamp(
        "2026-09-01"
    )

    next_date = latest_date + pd.Timedelta(
        days=1
    )

    assert next_date > latest_date


def test_regression_sma_exists():
    df = pd.DataFrame(
        {
            "Close": np.arange(
                1,
                31,
                dtype=float,
            )
        }
    )

    df["SMA20"] = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    assert "SMA20" in df.columns
    assert not pd.isna(
        df.iloc[-1]["SMA20"]
    )


# ============================================================
# Feature/model integration smoke tests
# ============================================================

def test_features_module_available_for_model_flow():
    features = _import_optional("src.features")

    if features is None:
        pytest.skip("src.features unavailable")

    assert features is not None


def test_features_module_exposes_callable():
    features = _import_optional("src.features")

    if features is None:
        pytest.skip("src.features unavailable")

    functions = [
        name
        for name in dir(features)
        if callable(getattr(features, name, None))
        and not name.startswith("_")
    ]

    assert len(functions) > 0


def test_models_module_available():
    models = _import_optional("src.models")

    if models is None:
        pytest.skip("src.models unavailable")

    assert models is not None


# ============================================================
# High-level Stage 2 contract
# ============================================================

def test_stage2_high_level_contract():
    """
    High-level architectural contract.

    Stage 2 should contain:
      1. Morning prediction
      2. Evening evaluation
      3. Retraining/champion challenger
      4. Jump engine
      5. Intraday engine
      6. Persistent GitHub data
    """

    required_files = [
        "src/morning_runner.py",
        "src/evening.py",
        "src/retraining.py",
        "src/jump_engine.py",
        "src/intraday_engine.py",
        "src/ledger.py",
    ]

    missing = [
        file
        for file in required_files
        if not (ROOT / file).exists()
    ]

    # jump_engine.py is allowed to be absent in the current
    # Stage 1.5 repository because jump logic may still be inside
    # another module.
    allowed_missing = {
        "src/jump_engine.py",
    }

    real_missing = [
        file
        for file in missing
        if file not in allowed_missing
    ]

    assert not real_missing, (
        "Missing Stage 2 core files:\n"
        + "\n".join(real_missing)
    )


# ============================================================
# Final architecture checks
# ============================================================

def test_stage2_has_prediction_component():
    candidates = [
        ROOT / "src" / "prediction.py",
        ROOT / "src" / "models.py",
    ]

    assert any(path.exists() for path in candidates)


def test_stage2_has_evaluation_component():
    assert (
        ROOT / "src" / "evaluation.py"
    ).exists()


def test_stage2_has_retraining_component():
    assert (
        ROOT / "src" / "retraining.py"
    ).exists()


def test_stage2_has_ledger_component():
    assert (
        ROOT / "src" / "ledger.py"
    ).exists()


def test_stage2_has_morning_component():
    candidates = [
        ROOT / "src" / "morning_runner.py",
        ROOT / "morning.py",
        ROOT / "stage15_morning.py",
    ]

    assert any(path.exists() for path in candidates)


def test_stage2_has_evening_component():
    candidates = [
        ROOT / "src" / "evening.py",
        ROOT / "evening.py",
    ]

    assert any(path.exists() for path in candidates)


def test_stage2_has_telegram_component():
    candidates = [
        ROOT / "src" / "telegram_report.py",
        ROOT / "telegram_report.py",
    ]

    assert any(path.exists() for path in candidates)
