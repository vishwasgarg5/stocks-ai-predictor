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
