"""
STAGE 4 ENGINE — Market + Sector Intelligence and Price-Bucket Selection
==========================================================================
Easy-access guide:
1. Price buckets prevent the ranking from being dominated by one price range.
2. Sector mapping is cached in GitHub CSV so repeated runs do not need to
   rediscover already-known sectors.
3. Sector strength is calculated from candidate-stock momentum and breadth.
4. The engine is deliberately additive: Stage 2 OHLC/ensemble prediction is
   not replaced; Stage 4 improves the selection context around it.

No local database is used. Persistent cache lives under data/stage2/.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

from .config import STAGE4_SECTOR_MAP_FILE
from .utils import clamp

# ---------------------------------------------------------------------------
# PRICE BUCKETS — easy to find/change here
# ---------------------------------------------------------------------------
PRICE_BUCKETS = [
    ("B1", ">1000", 1000.0, float("inf")),
    ("B2", "500-999", 500.0, 1000.0),
    ("B3", "100-499", 100.0, 500.0),
    ("B4", "50-99", 50.0, 100.0),
    ("B5", "10-49", 10.0, 50.0),
]


def price_bucket(price):
    """Return the configured price bucket for the current market price."""
    p = float(price)
    for code, label, low, high in PRICE_BUCKETS:
        if low <= p < high:
            return code, label
    return "OUT", "<10 / invalid"


def _load_sector_cache():
    """Load GitHub-persisted sector mapping; return empty cache if unavailable."""
    path = Path(STAGE4_SECTOR_MAP_FILE)
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        return {
            str(r["Symbol"]): str(r["Sector"])
            for _, r in df.iterrows()
            if str(r.get("Symbol", "")) and str(r.get("Sector", "UNKNOWN"))
        }
    except Exception:
        return {}


def _save_sector_cache(cache):
    """Persist sector mapping as CSV; GitHub Actions commits it after the run."""
    path = Path(STAGE4_SECTOR_MAP_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        sorted(cache.items()), columns=["Symbol", "Sector"]
    ).to_csv(path, index=False)


def _lookup_sector(symbol):
    """Best-effort Yahoo Finance sector lookup for one NSE equity."""
    try:
        info = yf.Ticker(f"{symbol}.NS").get_info()
        sector = info.get("sector") or info.get("industry") or "UNKNOWN"
        return str(sector).strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def enrich_sectors(symbols, workers=8):
    """
    Add sector information to symbols and persist newly discovered mappings.
    Only missing symbols are queried, reducing API calls on future runs.
    """
    cache = _load_sector_cache()
    missing = [s for s in dict.fromkeys(symbols) if s not in cache]
    if missing:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_lookup_sector, s): s for s in missing}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    cache[symbol] = future.result()
                except Exception:
                    cache[symbol] = "UNKNOWN"
        _save_sector_cache(cache)
    return {s: cache.get(s, "UNKNOWN") for s in symbols}


def _stock_return(df, days=20):
    """Calculate recent return using only completed historical bars."""
    if df is None or len(df) <= days:
        return np.nan
    try:
        return float(df["Close"].iloc[-1] / df["Close"].iloc[-days - 1] - 1.0) * 100.0
    except Exception:
        return np.nan


def add_stage4_context(candidates, data_map, regime):
    """
    Add PriceBucket, Sector, SectorStrength and SectorScore to ML candidates.

    SectorStrength is a relative momentum/breadth score across the candidate
    universe, not a guarantee of future performance. It is intentionally
    capped to 0-100 before entering the final ranking.
    """
    if candidates is None or candidates.empty:
        return candidates

    df = candidates.copy()
    sector_map = enrich_sectors(df["Symbol"].tolist())
    df["Sector"] = df["Symbol"].map(sector_map).fillna("UNKNOWN")
    buckets = df["Current_Price"].apply(price_bucket)
    df["PriceBucket"] = buckets.map(lambda x: x[0])
    df["PriceBucketLabel"] = buckets.map(lambda x: x[1])

    returns = {s: _stock_return(data_map.get(s)) for s in df["Symbol"]}
    df["SectorReturn20D"] = df["Symbol"].map(returns)
    sector_stats = df.groupby("Sector", dropna=False)["SectorReturn20D"].agg(["median", "count"])
    overall_median = float(df["SectorReturn20D"].median()) if df["SectorReturn20D"].notna().any() else 0.0

    strength = []
    for _, row in df.iterrows():
        sector = row["Sector"]
        ret = row["SectorReturn20D"]
        if sector in sector_stats.index and pd.notna(ret):
            median_ret = float(sector_stats.loc[sector, "median"])
            peer_count = int(sector_stats.loc[sector, "count"])
            # Momentum relative to the full candidate universe.
            momentum = 50.0 + (median_ret - overall_median) * 8.0
            # Small sectors get less extreme scores until more peer evidence exists.
            breadth_bonus = min(max(peer_count - 1, 0), 5) * 1.5
            score = momentum + breadth_bonus
        else:
            score = 50.0
        strength.append(clamp(score))
    df["SectorStrength"] = strength

    # Stage 4 ranking component. Keep regime handling simple and transparent.
    regime_adjustment = {"BULL": 3, "BEAR": -3, "HIGH VOL": -5}.get(regime, 0)
    df["SectorScore"] = df["SectorStrength"].clip(0, 100) + regime_adjustment
    df["SectorScore"] = df["SectorScore"].clip(0, 100)
    return df


def select_price_bucket_candidates(candidates, per_bucket=3):
    """Keep the strongest few stocks from each price bucket before final ranking."""
    if candidates is None or candidates.empty:
        return candidates
    pieces = []
    for _, group in candidates.groupby("PriceBucket", sort=False):
        pieces.append(group.sort_values("Score", ascending=False).head(per_bucket))
    return pd.concat(pieces, ignore_index=True) if pieces else candidates.iloc[0:0]
