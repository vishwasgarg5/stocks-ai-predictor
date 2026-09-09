"""Stocks AI Predictor — Stage 4 (Stage 3A + 3B + Stage 4 combined)."""

# Report hardening is loaded after package initialization so imports of
# src.telegram_report receive the corrected mobile/long-horizon renderers.
try:
    from . import telegram_report_patch as _telegram_report_patch
except Exception:
    _telegram_report_patch = None

# Runtime compatibility hardening: intraday session sizing and portfolio
# current-price fallback. This is source-controlled and never rewrites code
# from inside a workflow run.
try:
    from . import runtime_hardening as _runtime_hardening
except Exception:
    _runtime_hardening = None

# The morning runner imports these ledger functions directly, so patch both
# the ledger module and the already-imported morning_runner bindings. An old
# ReportSent=true marker without DeliveryVersion=v2 must never suppress a
# Telegram retry.
if _telegram_report_patch is not None:
    try:
        from . import ledger as _ledger
        from . import morning_runner as _morning_runner
        _ledger.morning_report_sent = _telegram_report_patch._delivery_sent
        _ledger.mark_morning_report_sent = _telegram_report_patch._mark_delivery
        _morning_runner.morning_report_sent = _telegram_report_patch._delivery_sent
        _morning_runner.mark_morning_report_sent = _telegram_report_patch._mark_delivery
    except Exception:
        pass
