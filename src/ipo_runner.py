"""Automated live IPO intelligence runner.
Fetches current Mainboard/SME IPO table, scores open/upcoming issues, and sends Telegram.
GMP is explicitly treated as unofficial sentiment, not a guaranteed listing price.
"""
from __future__ import annotations
import io
import re
import requests
import pandas as pd
from .ipo_engine import analyze_ipos
from .telegram_report import send_telegram

SOURCE_URL="https://ipomarkets.com/ipo-gmp"
HEADERS={"User-Agent":"Mozilla/5.0 (compatible; stocks-ai-predictor/10.0)"}

def _money(v):
    if pd.isna(v): return 0.0
    s=str(v).replace(",","").replace("₹","").strip()
    m=re.search(r"[-+]?\d+(?:\.\d+)?",s)
    return float(m.group()) if m else 0.0

def fetch_live_ipos():
    r=requests.get(SOURCE_URL,headers=HEADERS,timeout=30); r.raise_for_status()
    tables=pd.read_html(io.StringIO(r.text))
    if not tables: return pd.DataFrame()
    table=max(tables,key=lambda x: len(x))
    rename={}
    for c in table.columns:
        key=str(c).lower()
        if "company" in key: rename[c]="IPOName"
        elif key.strip()=="gmp": rename[c]="GMP"
        elif "issue price" in key or "price" in key and "band" not in key: rename[c]="PriceHigh"
        elif "open" in key: rename[c]="OpenDate"
        elif "close" in key: rename[c]="CloseDate"
        elif "status" in key: rename[c]="Status"
    table=table.rename(columns=rename)
    if "IPOName" not in table: return pd.DataFrame()
    for c in ["PriceHigh","GMP"]:
        if c not in table: table[c]=0
        table[c]=table[c].map(_money)
    # Keep only issues currently open or upcoming; exclude already-listed/closed rows.
    status=table.get("Status",pd.Series("",index=table.index)).astype(str).str.upper()
    mask=status.str.contains("OPEN|UPCOMING|LIVE",regex=True,na=False)
    if mask.any(): table=table.loc[mask].copy()
    return table

def run():
    try: df=fetch_live_ipos()
    except Exception as exc:
        print(f"IPO data fetch failed: {exc}"); return False
    if df.empty:
        print("No live/upcoming IPO records found."); return False
    result=analyze_ipos(df)
    lines=["🏦 *IPO INTELLIGENCE — STAGE 10*","","OPEN / UPCOMING IPOs"]
    for _,r in result.head(10).iterrows():
        lines.append(f"*{r.get('IPOName','-')}* | Price ₹{r.get('PriceHigh',0):.0f} | GMP ₹{r.get('GMPValue',0):.0f} ({r.get('GMPPct',0):.1f}%) | Score {r.get('IPOScore',0):.0f} | *{r.get('IPOAction','WATCH')}*")
    lines += ["","GMP is unofficial/unregulated and is only one input; it is not a guaranteed listing price.","Decision combines GMP, growth, profitability, ROE, leverage, valuation and subscription when available."]
    return send_telegram("\n".join(lines))

if __name__=="__main__": run()
