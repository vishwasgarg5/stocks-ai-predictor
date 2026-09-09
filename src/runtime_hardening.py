"""Runtime hardening for data freshness, prediction quality and Telegram-safe delivery."""
from __future__ import annotations
import numpy as np
from . import intraday_engine as _ie
from . import portfolio_report as _pr
from . import market_data as _md
from . import selection as _sel
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

# Only accept index/sector data from the requested completed session. Never
# display a July value in a September report simply because Yahoo supplied it
# as a fallback.
def _fresh_index_snapshot_with_fallback(key, cutoff=None):
    candidates=[_md.INDEX_SYMBOLS.get(key)]+_md.INDEX_SYMBOL_FALLBACKS.get(key,[])
    seen=set(); expected=str(cutoff) if cutoff is not None else None
    for symbol in candidates:
        if not symbol or symbol in seen:continue
        seen.add(symbol)
        snap=_md._index_snapshot(symbol,cutoff=cutoff)
        if not np.isfinite(snap.get("Close",np.nan)):continue
        actual=str(snap.get("DataDate",""))
        if expected is not None and actual!=expected:continue
        snap["IndexKey"]=key;snap["Freshness"]="CURRENT" if expected is None or actual==expected else "STALE"
        return snap
    return {"Close":np.nan,"Change1D":np.nan,"Open":np.nan,"High":np.nan,"Low":np.nan,"High52W":np.nan,"Low52W":np.nan,"Source":"unavailable","IndexKey":key,"DataDate":expected or "","Freshness":"UNAVAILABLE"}

_md._index_snapshot_with_fallback = _fresh_index_snapshot_with_fallback

# Prediction ledger selection is deliberately independent from BUY/TRADE
# eligibility. In a BEAR market the model can abstain from trading while still
# publishing up to 10 valid, diverse forecasts for evaluation.
_original_select_top_stocks = _sel.select_top_stocks

def _prediction_first_select(candidates, top_n=None, regime="SIDEWAYS", **kwargs):
    return _sel.select_prediction_set(candidates, top_n=top_n, max_per_bucket=kwargs.get("max_per_bucket"))

_sel.select_top_stocks = _prediction_first_select
_pr._selection_runtime_hardening = True
