"""Stocks AI Predictor package.

Package initialization must remain free of entry-point imports. In particular,
do not import ``morning_runner`` here: ``python -m src.morning_runner``
initializes this package before loading the runner, and importing the runner
from ``__init__`` creates a circular import warning.
"""

# These modules contain only package-level hardening and do not import the
# morning entry point, so they are safe to initialize here.
try:
    from . import telegram_report_patch as _telegram_report_patch
except Exception:
    _telegram_report_patch = None

try:
    from . import runtime_hardening as _runtime_hardening
except Exception:
    _runtime_hardening = None

if _telegram_report_patch is not None:
    try:
        _telegram_report_patch.apply_telegram_report_patches()
    except Exception:
        pass
