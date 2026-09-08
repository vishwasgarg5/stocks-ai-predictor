import numpy as np
import pandas as pd

from src.market_data import get_market_regime


def _frame(values):
    idx=pd.date_range("2025-01-01",periods=len(values),freq="B")
    return pd.DataFrame({"Open":values,"High":np.array(values)*1.01,"Low":np.array(values)*0.99,"Close":values,"Volume":1000000},index=idx)


def test_market_regime_returns_bull_for_strong_uptrend(monkeypatch):
    values=np.linspace(100,180,260)
    monkeypatch.setattr("src.market_data.get_nifty_data",lambda *args,**kwargs:_frame(values))
    result=get_market_regime()
    assert result["name"]=="BULL"
    assert 0 <= result["score"] <= 100
    assert 0 <= result["confidence"] <= 100


def test_market_regime_returns_bear_for_strong_downtrend(monkeypatch):
    values=np.linspace(180,100,260)
    monkeypatch.setattr("src.market_data.get_nifty_data",lambda *args,**kwargs:_frame(values))
    result=get_market_regime()
    assert result["name"]=="BEAR"
    assert 0 <= result["score"] <= 100
    assert 0 <= result["confidence"] <= 100


def test_market_regime_confidence_exists_for_short_data(monkeypatch):
    values=np.linspace(100,105,40)
    monkeypatch.setattr("src.market_data.get_nifty_data",lambda *args,**kwargs:_frame(values))
    result=get_market_regime()
    assert result["name"]=="SIDEWAYS"
    assert 0 <= result["confidence"] <= 100
