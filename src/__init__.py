"""Stocks AI Predictor — Stage 4 (Stage 3A + 3B + Stage 4 combined)."""

# Stage 10.5 report hardening: load after package initialization so imports of
# src.telegram_report receive the corrected mobile/long-horizon renderers.
try:
    from . import telegram_report_patch as _telegram_report_patch
except Exception:
    # Keep package imports resilient for tooling that imports the package while
    # dependencies are being inspected; normal report execution will surface
    # real runtime errors from the report module itself.
    _telegram_report_patch = None
