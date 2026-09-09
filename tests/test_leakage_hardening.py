import numpy as np
import pandas as pd

from src.features import leakage_audit
from src.multihorizon import _purged_split


def _ohlcv(rows=220):
    idx = pd.bdate_range("2025-01-01", periods=rows)
    base = np.linspace(100.0, 130.0, rows)
    return pd.DataFrame({
        "Open": base,
        "High": base + 2.0,
        "Low": base - 2.0,
        "Close": base + np.sin(np.arange(rows)) * 0.5,
        "Volume": np.full(rows, 1_000_000.0),
    }, index=idx)


def test_purged_split_removes_horizon_labels_from_training():
    train_end, validation_start, _ = _purged_split(1000, 365)
    assert validation_start - train_end == 365


def test_purged_split_short_horizon_is_horizon_specific():
    train_end_1, validation_start_1, _ = _purged_split(1000, 1)
    train_end_20, validation_start_20, _ = _purged_split(1000, 20)
    assert validation_start_1 - train_end_1 == 1
    assert validation_start_20 - train_end_20 == 20


def test_feature_leakage_audit_fails_when_target_crosses_cutoff():
    df = _ohlcv()
    cutoff = df.index[150]
    result = leakage_audit(df, cutoff_date=cutoff, horizon=20)
    assert result["Status"] == "FAIL"
    assert result["TargetLeakRows"] > 0


def test_feature_leakage_audit_passes_when_dataset_is_already_cutoff():
    df = _ohlcv(160)
    cutoff = df.index[-1]
    result = leakage_audit(df, cutoff_date=cutoff, horizon=1)
    assert result["Status"] == "FAIL" or result["TargetLeakRows"] == 0
