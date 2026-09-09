"""Stocks AI Predictor package.

Keep package initialization side-effect free. In particular, do not import
runtime entry points such as ``morning_runner`` here: ``python -m
src.morning_runner`` initializes this package before loading the runner, so
importing the runner from ``__init__`` creates a circular import.

Report compatibility patches are safe to load here because they only patch
``telegram_report``. Runner-specific patches are applied explicitly by the
runner after all of its imports have completed.
"""

try:
    from . import telegram_report_patch as _telegram_report_patch
except Exception:
    _telegram_report_patch = None

# Apply only patches that do not import an entry-point module.
if _telegram_report_patch is not None:
    try:
        _telegram_report_patch.apply_telegram_report_patches()
    except Exception:
        pass
