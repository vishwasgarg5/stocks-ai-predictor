"""Reliable live IPO intelligence adapter for the Stage 10.5 morning report.

Primary source: IPOMarkets (GMP + IPO calendar).
Fallback: WellWorthShares (calendar/price/date data) so the report still works
when the GMP page is JavaScript-rendered or its table layout changes.
"""
from __future__ import annotations
import io
import re
import requests
import pandas as pd
from .ipo_engine import analyze_ipos

SOURCE_URLS=(
    "https://ipomarkets.com/upcoming-ipo",
    "https://www.wellworthgroup.co/forthcoming-ipo",
)
HEADERS={"User-Agent":"Mozilla/5.0 (compatible; stocks-ai-predictor/10.5)"}

def _number_list(v):
    if pd.isna(v): return []
    s=str(v).replace(",","").replace("₹","").replace("Rs.","").strip()
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?",s)]

def _money(v):
    values=_number_list(v)
    return values[0] if values else 0.0

def _price_high(v):
    values=_number_list(v)
    return values[-1] if values else 0.0

def _normalise_columns(table):
    rename={}
    for c in table.columns:
        key=re.sub(r"[^a-z0-9 ]+"," ",str(c).lower()).strip()
        if any(x in key for x in ["company","ipo name","name"]): rename[c]="IPOName"
        elif "gmp" in key: rename[c]="GMP"
        elif "issue price" in key or "offer price" in key or "price band" in key or ("price" in key and "issue size" not in key): rename[c]="PriceHigh"
        elif "open" in key: rename[c]="OpenDate"
        elif "close" in key: rename[c]="CloseDate"
        elif "listing" in key: rename[c]="ListingDate"
        elif "status" in key: rename[c]="Status"
    return table.rename(columns=rename)

def _parse_source(url):
    r=requests.get(url,headers=HEADERS,timeout=30);r.raise_for_status()
    tables=pd.read_html(io.StringIO(r.text))
    if not tables:return pd.DataFrame()
    candidates=[]
    for raw in tables:
        t=_normalise_columns(raw.copy())
        if "IPOName" not in t.columns:continue
        t["IPOName"]=t["IPOName"].astype(str).str.replace(r"\s+"," ",regex=True).str.strip()
        t=t[t["IPOName"].ne("")&t["IPOName"].ne("nan")].copy()
        if t.empty:continue
        if "PriceHigh" in t.columns:t["PriceHigh"]=t["PriceHigh"].map(_price_high)
        else:t["PriceHigh"]=0.0
        if "GMP" in t.columns:t["GMP"]=t["GMP"].map(_money)
        else:t["GMP"]=0.0
        if "Status" not in t.columns:t["Status"]=""
        candidates.append(t)
    if not candidates:return pd.DataFrame()
    return max(candidates,key=len)

def _is_active(row):
    status=str(row.get("Status","")).upper().strip()
    if status and re.search(r"LISTED|CLOSED|COMPLETED|ENDED",status):return False
    if status and re.search(r"OPEN|UPCOMING|LIVE|ONGOING",status):return True
    today=pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)
    open_dt=pd.to_datetime(row.get("OpenDate"),errors="coerce",dayfirst=True)
    close_dt=pd.to_datetime(row.get("CloseDate"),errors="coerce",dayfirst=True)
    if pd.notna(open_dt) and pd.notna(close_dt):
        return open_dt.normalize()<=today<=close_dt.normalize()
    if pd.notna(open_dt) and open_dt.normalize()>=today-pd.Timedelta(days=1):return True
    if pd.notna(close_dt) and close_dt.normalize()>=today:return True
    return not bool(status)

def fetch_live_ipos():
    """Fetch current/upcoming IPOs with source fallback and active-date filtering."""
    combined=[]
    for url in SOURCE_URLS:
        try:
            table=_parse_source(url)
            if table.empty:continue
            active=table.apply(_is_active,axis=1)
            filtered=table.loc[active].copy()
            if not filtered.empty:combined.append(filtered)
            # IPOMarkets is preferred because it can provide GMP.
            if "ipomarkets.com" in url and not filtered.empty:return filtered.drop_duplicates(subset=["IPOName"]).reset_index(drop=True)
        except Exception as exc:
            print(f"IPO source failed {url}: {exc}")
    if not combined:return pd.DataFrame()
    out=pd.concat(combined,ignore_index=True,sort=False)
    # Prefer rows with a non-zero GMP when duplicate IPO names exist.
    out["GMP"]=pd.to_numeric(out.get("GMP",0),errors="coerce").fillna(0)
    out=out.sort_values("GMP",ascending=False).drop_duplicates(subset=["IPOName"],keep="first")
    return out.reset_index(drop=True)

def get_ipo_report():
    try:return analyze_ipos(fetch_live_ipos())
    except Exception as exc:print(f"IPO intelligence failed: {exc}");return pd.DataFrame()

if __name__=="__main__":
    df=get_ipo_report();print(df.head(15).to_string(index=False) if not df.empty else "No active/upcoming IPOs found.")
