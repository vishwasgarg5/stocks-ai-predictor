"""Reliable IPO intelligence data adapter for the Stage 10.1 morning report."""
from __future__ import annotations
import io
import re
import requests
import pandas as pd
from .ipo_engine import analyze_ipos

SOURCE_URL="https://ipomarkets.com/ipo-gmp"
HEADERS={"User-Agent":"Mozilla/5.0 (compatible; stocks-ai-predictor/10.1)"}

def _money(v):
    if pd.isna(v): return 0.0
    s=str(v).replace(",","").replace("₹","").replace("Rs.","").strip()
    m=re.search(r"[-+]?\d+(?:\.\d+)?",s)
    return float(m.group()) if m else 0.0

def _normalise_columns(table):
    rename={}
    for c in table.columns:
        key=re.sub(r"[^a-z0-9 ]+"," ",str(c).lower()).strip()
        if any(x in key for x in ["company","ipo name"]): rename[c]="IPOName"
        elif "gmp" in key: rename[c]="GMP"
        elif "issue price" in key or ("price" in key and "band" not in key): rename[c]="PriceHigh"
        elif "open" in key: rename[c]="OpenDate"
        elif "close" in key: rename[c]="CloseDate"
        elif "listing" in key: rename[c]="ListingDate"
        elif "status" in key: rename[c]="Status"
    return table.rename(columns=rename)

def _is_active(row):
    status=str(row.get("Status","")).upper().strip()
    if status and re.search(r"LISTED|CLOSED|COMPLETED|ENDED",status): return False
    if status and re.search(r"OPEN|UPCOMING|LIVE|ONGOING",status): return True
    today=pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)
    for key in ["OpenDate","CloseDate"]:
        if key in row and pd.notna(row[key]):
            dt=pd.to_datetime(row[key],errors="coerce",dayfirst=True)
            if pd.notna(dt):
                if key=="OpenDate" and dt.normalize()>=today-pd.Timedelta(days=1): return True
                if key=="CloseDate" and dt.normalize()>=today: return True
    return not bool(status)

def fetch_live_ipos():
    r=requests.get(SOURCE_URL,headers=HEADERS,timeout=30);r.raise_for_status()
    tables=pd.read_html(io.StringIO(r.text))
    if not tables:return pd.DataFrame()
    candidates=[]
    for raw in tables:
        t=_normalise_columns(raw.copy())
        if "IPOName" in t.columns:
            t["IPOName"]=t["IPOName"].astype(str).str.strip()
            t=t[t["IPOName"].ne("")&t["IPOName"].ne("nan")].copy()
            if not t.empty:candidates.append(t)
    if not candidates:return pd.DataFrame()
    table=max(candidates,key=len)
    for c in ["PriceHigh","GMP"]:
        if c not in table.columns:table[c]=0.0
        table[c]=table[c].map(_money)
    if "Status" not in table.columns:table["Status"]=""
    active=table.apply(_is_active,axis=1)
    filtered=table.loc[active].copy()
    if filtered.empty and table["Status"].astype(str).str.strip().eq("").all(): filtered=table.copy()
    return filtered.drop_duplicates(subset=["IPOName"]).reset_index(drop=True)

def get_ipo_report():
    try:return analyze_ipos(fetch_live_ipos())
    except Exception as exc:print(f"IPO intelligence failed: {exc}");return pd.DataFrame()

if __name__=="__main__":
    df=get_ipo_report();print(df.head(10).to_string(index=False) if not df.empty else "No active/upcoming IPOs found.")
