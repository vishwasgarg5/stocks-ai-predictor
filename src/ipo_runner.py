"""Reliable live IPO intelligence adapter with cached fallback."""
from __future__ import annotations
import io
import re
import requests
import pandas as pd
from .config import IPO_METRICS_FILE
from .ipo_engine import analyze_ipos
SOURCE_URLS=("https://ipomarkets.com/upcoming-ipo","https://www.wellworthgroup.co/forthcoming-ipo")
HEADERS={"User-Agent":"Mozilla/5.0 (compatible; stocks-ai-predictor/10.5)"}

def _number_list(v):
    if pd.isna(v):return []
    s=str(v).replace(",","").replace("₹","").replace("Rs.","").strip();return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?",s)]
def _money(v):
    x=_number_list(v);return x[0] if x else 0.0
def _price_high(v):
    x=_number_list(v);return x[-1] if x else 0.0
def _normalise_columns(table):
    rename={}
    for c in table.columns:
        key=re.sub(r"[^a-z0-9 ]+"," ",str(c).lower()).strip()
        if any(x in key for x in ["company","ipo name","name"]):rename[c]="IPOName"
        elif "gmp" in key:rename[c]="GMP"
        elif "issue price" in key or "offer price" in key or "price band" in key or ("price" in key and "issue size" not in key):rename[c]="PriceHigh"
        elif "open" in key:rename[c]="OpenDate"
        elif "close" in key:rename[c]="CloseDate"
        elif "listing" in key:rename[c]="ListingDate"
        elif "status" in key:rename[c]="Status"
    return table.rename(columns=rename)
def _parse_source(url):
    r=requests.get(url,headers=HEADERS,timeout=25);r.raise_for_status();tables=pd.read_html(io.StringIO(r.text));candidates=[]
    for raw in tables:
        t=_normalise_columns(raw.copy())
        if "IPOName" not in t.columns:continue
        t["IPOName"]=t["IPOName"].astype(str).str.replace(r"\s+"," ",regex=True).str.strip();t=t[t["IPOName"].ne("")&t["IPOName"].ne("nan")].copy()
        if t.empty:continue
        t["PriceHigh"]=t["PriceHigh"].map(_price_high) if "PriceHigh" in t else 0.0;t["GMP"]=t["GMP"].map(_money) if "GMP" in t else 0.0
        if "Status" not in t:t["Status"]=""
        candidates.append(t)
    return max(candidates,key=len) if candidates else pd.DataFrame()
def _is_active(row):
    status=str(row.get("Status","")).upper().strip()
    if status and re.search(r"LISTED|CLOSED|COMPLETED|ENDED",status):return False
    today=pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)
    op=pd.to_datetime(row.get("OpenDate"),errors="coerce",dayfirst=True);cl=pd.to_datetime(row.get("CloseDate"),errors="coerce",dayfirst=True)
    if pd.notna(op) and pd.notna(cl):return op.normalize()<=today<=cl.normalize() or op.normalize()>today
    if pd.notna(op):return op.normalize()>=today-pd.Timedelta(days=1)
    if pd.notna(cl):return cl.normalize()>=today
    return True

def fetch_live_ipos():
    combined=[]
    for url in SOURCE_URLS:
        try:
            table=_parse_source(url)
            if table.empty:continue
            active=table.loc[table.apply(_is_active,axis=1)].copy()
            if not active.empty:combined.append(active)
            if "ipomarkets.com" in url and not active.empty:return active.drop_duplicates("IPOName").reset_index(drop=True)
        except Exception as exc:print(f"IPO source failed {url}: {exc}")
    if not combined:return pd.DataFrame()
    out=pd.concat(combined,ignore_index=True,sort=False);out["GMP"]=pd.to_numeric(out.get("GMP",0),errors="coerce").fillna(0);return out.sort_values("GMP",ascending=False).drop_duplicates("IPOName").reset_index(drop=True)
def _cached():
    try:
        if IPO_METRICS_FILE.exists():
            df=pd.read_csv(IPO_METRICS_FILE)
            if not df.empty and "IPOName" in df.columns:return df
    except Exception as exc:print(f"IPO cache read failed: {exc}")
    return pd.DataFrame()
def get_ipo_report():
    try:
        live=fetch_live_ipos()
        if not live.empty:
            report=analyze_ipos(live);report.to_csv(IPO_METRICS_FILE,index=False);return report
        cached=_cached()
        if not cached.empty:return cached
        return pd.DataFrame()
    except Exception as exc:
        print(f"IPO intelligence failed: {exc}");return _cached()
if __name__=="__main__":
    df=get_ipo_report();print(df.head(15).to_string(index=False) if not df.empty else "No active/upcoming IPOs found.")
