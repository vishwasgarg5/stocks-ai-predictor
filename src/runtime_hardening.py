"""Runtime hardening that can be applied without mutating pipeline source during runs."""
from __future__ import annotations
from . import intraday_engine as _ie
from . import portfolio_report as _pr
from .data_snapshot import get_snapshot

# A completed NSE session contains roughly 25 fifteen-minute bars.  Requiring
# 80 bars makes a morning intraday scan fail closed for every symbol.
_ie.INTRADAY_MIN_ROWS = 20


def _cached_history(ticker, cutoff_date=None):
    cutoff = _pr._effective_cutoff(cutoff_date)
    d, snap = get_snapshot(_pr._symbol(ticker), cutoff_date=cutoff, min_rows=60, allow_download=True)
    if snap.status not in {"OK", "STALE"} or d.empty:
        return _pr.pd.DataFrame()
    return d


_pr._cached_history = _cached_history
