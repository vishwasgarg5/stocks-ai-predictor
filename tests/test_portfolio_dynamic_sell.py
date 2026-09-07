import pandas as pd

from src.portfolio_report import _sell_plan, _next_trading_date


def _row(**values):
    base = {"AI_Confidence": 80}
    for h in (1, 3, 5, 7, 20, 60, 90, 180, 365):
        base[f"Horizon_{h}D"] = float("nan")
    base.update(values)
    return base


def test_higher_than_minimum_target_uses_reliable_long_horizon():
    row = _row(Horizon_20D=18.0, Horizon_60D=16.0)
    profit, price, window, date, status = _sell_plan(row, 450, 500, "2026-09-07")
    assert profit == 18.0
    assert price == 590.0
    assert "20D" in window
    assert date != "-"
    assert status == "TARGET_DATE"


def test_low_confidence_is_capped_at_minimum_target():
    row = _row(AI_Confidence=45, Horizon_20D=18.0, Horizon_60D=25.0)
    profit, price, window, date, _ = _sell_plan(row, 450, 500, "2026-09-07")
    assert profit == 10.0
    assert price == 550.0
    assert "20D" in window or "60D" in window


def test_no_horizon_reaches_minimum_target_waits():
    row = _row(Horizon_20D=7.0, Horizon_365D=9.0)
    profit, price, window, date, status = _sell_plan(row, 450, 500, "2026-09-07")
    assert profit == 10.0
    assert price == 550.0
    assert date == "-"
    assert status == "WAIT"


def test_nse_holiday_is_skipped():
    assert _next_trading_date("2026-09-11", 1) == "2026-09-15"
