"""Runtime hardening for data freshness, prediction quality and Telegram-safe delivery."""
from __future__ import annotations
import numpy as np
from . import intraday_engine as _ie
from . import portfolio_report as _pr
from . import market_data as _md
from .data_snapshot import get_snapshot

_ie.INTRADAY_MIN_ROWS = 20

def _cached_history(ticker, cutoff_date=None):
    cutoff = _pr._effective_cutoff(cutoff_date)
    d, snap = get_snapshot(_pr._symbol(ticker), cutoff_date=cutoff, min_rows=60, allow_download=True)
    if snap.status not in {"OK", "STALE"} or d.empty:
        return _pr.pd.DataFrame()
    return d

_pr._cached_history = _cached_history

def _fresh_index_snapshot_with_fallback(key, cutoff=None):
    candidates=[_md.INDEX_SYMBOLS.get(key)]+_md.INDEX_SYMBOL_FALLBACKS.get(key,[])
    seen=set(); expected=str(cutoff) if cutoff is not None else None
    for symbol in candidates:
        if not symbol or symbol in seen: continue
        seen.add(symbol)
        snap=_md._index_snapshot(symbol,cutoff=cutoff)
        if not np.isfinite(snap.get("Close",np.nan)): continue
        actual=str(snap.get("DataDate",""))
        if expected is not None and actual!=expected: continue
        snap["IndexKey"]=key; snap["Freshness"]="CURRENT" if expected is None or actual==expected else "STALE"
        return snap
    return {"Close":np.nan,"Change1D":np.nan,"Open":np.nan,"High":np.nan,"Low":np.nan,"High52W":np.nan,"Low52W":np.nan,"Source":"unavailable","IndexKey":key,"DataDate":expected or "","Freshness":"UNAVAILABLE"}

_md._index_snapshot_with_fallback = _fresh_index_snapshot_with_fallback
_pr._selection_runtime_hardening = True
