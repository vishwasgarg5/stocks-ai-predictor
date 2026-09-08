"""Portfolio decision hardening for Stage 10.5."""
import numpy as np
from . import portfolio_report as _portfolio_report

def _decision(current, avg, target, confidence, forecasts):
    if current is None or not np.isfinite(current) or not np.isfinite(avg) or avg <= 0:
        return "WAIT", "NO PRICE / COST DATA"
    if not np.isfinite(target):
        return "HOLD", "AI PREDICTION UNAVAILABLE"
    profit_target = avg * (1 + _portfolio_report.TARGET_PROFIT_PCT / 100.0)
    tolerance = max(1e-9, abs(profit_target) * 1e-10)
    if current >= profit_target - tolerance:
        return "SELL", "10% profit target reached"
    if target >= profit_target and current < avg:
        strong = sum(v >= _portfolio_report.TARGET_PROFIT_PCT for _, v in forecasts)
        if strong >= 2 and confidence >= _portfolio_report.MIN_AI_CONFIDENCE:
            return "HOLD", "MULTI_HORIZON_CONFIRMED recovery"
        return "HOLD", "AI recovery target remains above cost"
    if target < current * (1 - _portfolio_report.SELL_RISK_GAP_PCT / 100):
        return "REDUCE", "AI target materially below current price"
    return "HOLD", "AI recovery not yet confirmed"

_portfolio_report._decision = _decision
