"""Runtime hardening patches for the Stage 10.5 Telegram report.

Kept separate so report fixes can be applied without replacing the large
telegram_report.py module. The package initializer imports this module once,
which patches the report functions before callers generate a report.
"""
from __future__ import annotations
from . import telegram_report as _tr

_SHORT_HORIZONS = (3, 7, 10, 20)
_LONG_HORIZONS = (60, 90, 180, 365)
_ALL_HORIZONS = _SHORT_HORIZONS + _LONG_HORIZONS


def _status(row):
    vals = []
    for horizon in _ALL_HORIZONS:
        value = _tr._num(_tr._horizon_value(row, horizon))
        if value is not None:
            vals.append(value)
    if not vals:
        return "-"
    ratio = sum(value > 0 for value in vals) / len(vals)
    if ratio >= 2 / 3:
        return "🟢 BULLISH"
    if ratio >= 1 / 3:
        return "🟡 MIXED"
    return "🔴 WEAK"


def _horizon_table(selected):
    if selected is None or selected.empty:
        return ["🔮 *MULTI-HORIZON OUTLOOK*", "No multi-horizon predictions available."]
    ordered = _tr._sort(selected).head(10)
    rows = []
    for _, row in ordered.iterrows():
        rows.append([str(row.get("Symbol", "-"))] + [_tr._pct(_tr._horizon_value(row, h)) for h in _SHORT_HORIZONS] + [_status(row)])
    short = [
        f"🔮 *MULTI-HORIZON OUTLOOK — TOP {len(rows)}*",
        "Short-term expected return by trading-day horizon.",
        *_tr._table(["Stock", "3D", "7D", "10D", "20D", "Status"], rows, max_width=13),
    ]
    long_rows = []
    for _, row in ordered.iterrows():
        long_rows.append([str(row.get("Symbol", "-"))] + [_tr._pct(_tr._horizon_value(row, h)) for h in _LONG_HORIZONS] + [_status(row)])
    long = [
        "🔭 *LONG-HORIZON OUTLOOK — 60D TO 365D*",
        "Long-term expected return by trading-day horizon.",
        *_tr._table(["Stock", "60D", "90D", "180D", "365D", "Status"], long_rows, max_width=13),
    ]
    return short + ["", *long]


def _prediction_table(selected):
    if selected is None or selected.empty:
        return []
    rows = [[str(row.get("Symbol", "-")), _tr._fmt(row.get("Pred_Open")), _tr._fmt(row.get("Pred_High")), _tr._fmt(row.get("Pred_Low")), _tr._fmt(row.get("Pred_Close"))] for _, row in _tr._sort(selected).head(10).iterrows()]
    return [f"📈 *PREDICTED OHLC — TOP {len(rows)}*", *_tr._table(["Stock", "Open", "High", "Low", "Close"], rows)]


def _portfolio_status(row):
    vals = []
    for horizon in _ALL_HORIZONS:
        value = _tr._num(row.get(f"Horizon_{horizon}D"))
        if value is not None:
            vals.append(value)
    if not vals:
        return str(row.get("Portfolio_Target_Status", row.get("Target_Status", "-")))
    ratio = sum(value > 0 for value in vals) / len(vals)
    if ratio >= 2 / 3:
        return "🟢 BULLISH"
    if ratio >= 1 / 3:
        return "🟡 MIXED"
    return "🔴 WEAK"


def _clean_market_heading(text):
    marker = "📊 *MARKET OVERVIEW*"
    if not isinstance(text, str):
        return text
    while text.count(marker) > 1:
        text = text.replace(marker, "", 1)
    return text


_tr.REPORT_HORIZONS = _ALL_HORIZONS
_tr._horizon_status = _status
_tr._horizon_table = _horizon_table
_tr._prediction_table = _prediction_table
_tr._portfolio_horizon_status = _portfolio_status
_original_morning_report = _tr.morning_report


def morning_report(*args, **kwargs):
    return _clean_market_heading(_original_morning_report(*args, **kwargs))


_tr.morning_report = morning_report
