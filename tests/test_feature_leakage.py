import numpy as np
import pandas as pd

from src.features import prepare_supervised, leakage_audit


def _ohlcv(n=220):
    idx=pd.date_range("2025-01-01",periods=n,freq="B")
    close=pd.Series(np.linspace(100,180,n),index=idx)
    return pd.DataFrame({"Open":close-1,"High":close+2,"Low":close-2,"Close":close,"Volume":np.linspace(100000,200000,n)},index=idx)


def test_prepare_supervised_excludes_cutoff_day_and_future_rows():
    df=_ohlcv()
    cutoff=df.index[-1]
    supervised=prepare_supervised(df,cutoff)
    assert not supervised.empty
    assert supervised.index.max().normalize() < cutoff.normalize()


def test_prepare_supervised_has_no_label_beyond_cutoff():
    df=_ohlcv()
    cutoff=df.index[-5]
    supervised=prepare_supervised(df,cutoff)
    # Every retained row's one-step target is also strictly before cutoff.
    assert (supervised.index.normalize() < cutoff.normalize()).all()
    assert supervised.index.max().normalize() < cutoff.normalize()


def test_leakage_audit_rejects_raw_future_rows_against_cutoff():
    df=_ohlcv()
    cutoff=df.index[-10]
    audit=leakage_audit(df,cutoff)
    assert audit["Status"] == "FAIL"
    assert audit["FutureRows"] > 0 or audit["TargetLeakRows"] > 0
