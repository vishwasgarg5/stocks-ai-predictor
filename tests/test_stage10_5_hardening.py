import math
import pandas as pd

from src.telegram_report import _num, _pct, _accuracy
from src.portfolio_report import _is_nse_trading_day, _next_trading_date


def test_report_numeric_helpers_reject_non_finite():
    assert _num(float("nan"), 7) == 7
    assert _num(float("inf"), 7) == 7
    assert _pct(float("nan")) == "-"
    assert _accuracy(float("inf")) == "-"


def test_nse_holiday_is_not_trading_day():
    assert not _is_nse_trading_day("2026-09-14")
    assert _is_nse_trading_day("2026-09-15")


def test_sell_date_skips_nse_holiday():
    assert _next_trading_date("2026-09-11", 1) == "2026-09-15"
