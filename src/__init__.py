"""Stocks AI Predictor package.

Package initialization may load safe runtime hardening, but it must never import
an executable entry point such as ``src.morning_runner``. Keeping the entry
point out of ``sys.modules`` until runpy executes it prevents circular-import
RuntimeWarnings from ``python -m src.morning_runner``.
"""

# runtime_hardening imports only supporting modules and never the executable
# morning runner, so its freshness/selection guards remain active.
try:
    from . import runtime_hardening as _runtime_hardening
except Exception:
    _runtime_hardening = None
