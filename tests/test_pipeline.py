import pandas as pd
import numpy as np
from src.features import add_features
from src.ranking import rank_stocks


def sample(n=70):
    idx = pd.date_range("2026-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(100, 120, n), index=idx)
    df = pd.DataFrame({
        "Open": close * .998,
        "High": close * 1.01,
        "Low": close * .99,
        "Close": close,
        "Volume": 1000000,
    }, index=idx)
    nifty = pd.DataFrame({"Close": close * 10}, index=idx)
    return df, nifty


def test_features_and_ranking():
    df, nifty = sample()
    f = add_features(df, nifty)
    assert "rsi_14" in f.columns
    assert "target_close" in f.columns
    ranked = rank_stocks({"TEST": f})
    assert len(ranked) == 1
    assert ranked.iloc[0]["symbol"] == "TEST"
