"""Stage 4 market/sector intelligence and canonical price-bucket selection."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from .config import STAGE4_SECTOR_MAP_FILE
from .utils import clamp, price_bucket

PRICE_BUCKETS = [("B1", ">2500", 2500.0, float("inf")), ("B2", "1000-2499", 1000.0, 2500.0), ("B3", "500-999", 500.0, 1000.0), ("B4", "250-499", 250.0, 500.0), ("B5", "100-249", 100.0, 250.0), ("B6", "50-99", 50.0, 100.0), ("B7", "10-49", 10.0, 50.0)]

def price_bucket_code(price):
    label=price_bucket(price)
    return next((code for code,bucket,_,_ in PRICE_BUCKETS if bucket==label), "OUT")

def _load_sector_cache():
    path=Path(STAGE4_SECTOR_MAP_FILE)
    if not path.exists(): return {}
    try:
        df=pd.read_csv(path); return {str(r["Symbol"]):str(r["Sector"]) for _,r in df.iterrows() if str(r.get("Symbol","")) and str(r.get("Sector","UNKNOWN"))}
    except Exception:return {}

def _save_sector_cache(cache):
    path=Path(STAGE4_SECTOR_MAP_FILE);path.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(sorted(cache.items()),columns=["Symbol","Sector"]).to_csv(path,index=False)

def _lookup_sector(symbol):
    try:
        info=yf.Ticker(f"{symbol}.NS").get_info();return str(info.get("sector") or info.get("industry") or "UNKNOWN").strip() or "UNKNOWN"
    except Exception:return "UNKNOWN"

def enrich_sectors(symbols,workers=8):
    cache=_load_sector_cache();missing=[s for s in dict.fromkeys(symbols) if s not in cache]
    if missing:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures={executor.submit(_lookup_sector,s):s for s in missing}
            for future in as_completed(futures):
                symbol=futures[future]
                try:cache[symbol]=future.result()
                except Exception:cache[symbol]="UNKNOWN"
        _save_sector_cache(cache)
    return {s:cache.get(s,"UNKNOWN") for s in symbols}

def _stock_return(df,days=20):
    if df is None or len(df)<=days:return np.nan
    try:return float(df["Close"].iloc[-1]/df["Close"].iloc[-days-1]-1.0)*100.0
    except Exception:return np.nan

def add_stage4_context(candidates,data_map,regime):
    if candidates is None or candidates.empty:return candidates
    df=candidates.copy();sector_map=enrich_sectors(df["Symbol"].tolist());df["Sector"]=df["Symbol"].map(sector_map).fillna("UNKNOWN")
    df["PriceBucket"]=df["Current_Price"].map(price_bucket);df["PriceBucketLabel"]=df["PriceBucket"]
    returns={s:_stock_return(data_map.get(s)) for s in df["Symbol"]};df["SectorReturn20D"]=df["Symbol"].map(returns)
    sector_stats=df.groupby("Sector",dropna=False)["SectorReturn20D"].agg(["median","count"]);overall_median=float(df["SectorReturn20D"].median()) if df["SectorReturn20D"].notna().any() else 0.0;strength=[]
    for _,row in df.iterrows():
        sector=row["Sector"];ret=row["SectorReturn20D"]
        if sector in sector_stats.index and pd.notna(ret):
            median_ret=float(sector_stats.loc[sector,"median"]);peer_count=int(sector_stats.loc[sector,"count"]);score=50.0+(median_ret-overall_median)*8.0+min(max(peer_count-1,0),5)*1.5
        else:score=50.0
        strength.append(clamp(score))
    df["SectorStrength"]=strength;regime_adjustment={"BULL":3,"BEAR":-3,"HIGH VOL":-5}.get(regime,0);df["SectorScore"]=(df["SectorStrength"].clip(0,100)+regime_adjustment).clip(0,100);return df

def select_price_bucket_candidates(candidates,per_bucket=6):
    if candidates is None or candidates.empty:return candidates
    pieces=[group.sort_values("Score",ascending=False).head(per_bucket) for _,group in candidates.groupby("PriceBucket",sort=False)]
    return pd.concat(pieces,ignore_index=True) if pieces else candidates.iloc[0:0]
