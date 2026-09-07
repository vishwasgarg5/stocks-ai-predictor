import numpy as np
import pandas as pd


def test_portfolio_report_shows_dynamic_target_and_long_horizons():
    from src.telegram_report import _portfolio
    payload={"Rows":[{"Stock":"TEST","Quantity":10,"Decision":"HOLD","Current_Price":"100","Average_Price":"90","Portfolio_Target_Pct":18,"Portfolio_Target_Status":"MULTI_HORIZON_CONFIRMED","Horizon_60D":18,"Horizon_90D":21,"Horizon_180D":19,"Horizon_365D":20,"Sell_Window":"180D"}]}
    text="\n".join(_portfolio(payload))
    assert "AI PORTFOLIO MANAGER" in text
    assert "18.0%" in text
    assert "CONFIRMED" in text
    assert "Sell Window" in text


def test_portfolio_report_does_not_force_ten_percent():
    from src.telegram_report import _portfolio
    payload={"Rows":[{"Stock":"TEST","Quantity":10,"Decision":"HOLD","Current_Price":"100","Average_Price":"90","Portfolio_Target_Pct":22,"Portfolio_Target_Status":"MULTI_HORIZON_CONFIRMED","Horizon_60D":22,"Horizon_90D":23,"Horizon_180D":24,"Horizon_365D":25,"Sell_Window":"180D"}]}
    text="\n".join(_portfolio(payload))
    assert "22.0%" in text
    assert "10% target" not in text.lower()
