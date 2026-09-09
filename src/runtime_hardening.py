"""Runtime hardening for data freshness and prediction quality."""
from __future__ import annotations
import numpy as np
from . import intraday_engine as _ie
from . import portfolio_report as _pr
from . import market_data as _md
from .data_snapshot import get_snapshot

# A completed NSE session contains roughly 25 fifteen-minute bars.
_ie.INTRADAY_MIN_ROWS = 20


def _cached_history(ticker, cutoff_date=None):
    cutoff = _pr._effective_cutoff(cutoff_date)
    d, snap = get_snapshot(_pr._symbol(ticker), cutoff_date=cutoff, min_rows=60, allow_download=True)
    if snap.status not in {"OK", "STALE"} or d.empty:
        return _pr.pd.DataFrame()
    return d


_pr._cached_history = _cached_history

# Never display an older index session as if it were the requested market date.
# Yahoo can legitimately return a fallback index from an earlier session when a
# symbol is unavailable; that is useful for diagnostics but unsafe for the
# daily Telegram market snapshot.
_original_index_snapshot_with_fallback = _md._index_snapshot_with_fallback


def _fresh_index_snapshot_with_fallback(key, cutoff=None):
    snap = _original_index_snapshot_with_fallback(key, cutoff)
    if cutoff is None or not snap or not np.isfinite(snap.get("Close", np.nan)):
        return snap
    expected = str(cutoff)
    actual = str(snap.get("DataDate", ""))
    if actual and actual != expected:
        return {
            "Close": np.nan, "Change1D": np.nan, "Open": np.nan,
            "High": np.nan, "Low": np.nan, "High52W": np.nan,
            "Low52W": np.nan, "Source": "stale",
            "IndexKey": key, "DataDate": actual,
            "Freshness": "STALE",
        }
    snap["Freshness"] = "CURRENT"
    return snap


_md._index_snapshot_with_fallback = _fresh_index_snapshot_with_fallback
