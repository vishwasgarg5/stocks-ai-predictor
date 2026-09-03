"""Standalone portfolio-manager adapter for Telegram reports.

This module reads portfolio_manager/data/my_portfolio.csv and uses market data only
for portfolio reporting. It never writes portfolio data into the stock predictor.
"""
from pathlib import Path
import re
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO_FILE = ROOT / "portfolio_manager" / "data" / "my_portfolio.csv"

NAME_TO_TICKER = {
    "RELIANCE INDUSTRIES": "RELIANCE.NS", "RELIANCE": "RELIANCE.NS",
    "VEDANTA IRON & STEEL": "VEDL.NS", "VEDANTA": "VEDL.NS",
    "YES BANK": "YESBANK.NS", "IRFC": "IRFC.NS", "NTPC": "NTPC.NS",
    "TATA POWER": "TATAPOWER.NS", "WIPRO": "WIPRO.NS", "PALASH SECURITIES": "PALASHSECU.NS",
    "OLA ELECTRIC MOBILITY": "OLAELEC.NS", "STAR CEMENT": "STARCEMENT.NS", "SJVN": "SJVN.NS",
    "RELIANCE POWER": "RPOWER.NS", "IRCTC": "IRCTC.NS", "SEPC": "SEPC.NS",
    "INDIAN RENEWABLE ENERGY": "IREDA.NS", "IREDA": "IREDA.NS",
}

def _ticker(value):
    raw = str(value).strip(); key = re.sub(r"\s+", " ", raw.upper())
    if key in NAME_TO_TICKER: return NAME_TO_TICKER[key]
    if raw.upper().endswith(".NS"): return raw.upper()
    return raw.upper().replace(" & ", "").replace(" ", "") + ".NS"

def load_portfolio():
    if not PORTFOLIO_FILE.exists(): return pd.DataFrame()
    df = pd.read_csv(PORTFOLIO_FILE)
    if {"Stock","Quantity","Average_Price"}.issubset(df.columns):
        out=df[["Stock","Quantity","Average_Price"]].copy()
    elif {"Stock","Quantity"}.issubset(df.columns):
        out=df[["Stock","Quantity"]].copy(); out["Average_Price"]=float("nan")
    else: return pd.DataFrame()
    out["Ticker"]=out["Stock"].map(_ticker); out["Quantity"]=pd.to_numeric(out["Quantity"],errors="coerce").fillna(0)
    out["Average_Price"]=pd.to_numeric(out["Average_Price"],errors="coerce")
    return out

def _price(ticker):
    try:
        d=yf.download(ticker,period="5d",interval="1d",auto_adjust=False,progress=False,threads=False)
        if d is None or d.empty: return None
        if isinstance(d.columns,pd.MultiIndex): d=d.xs(ticker,axis=1,level=-1) if ticker in d.columns.get_level_values(-1) else d.droplevel(-1,axis=1)
        return float(d["Close"].dropna().iloc[-1])
    except Exception: return None

def portfolio_snapshot():
    df=load_portfolio()
    if df.empty: return df, {"Positions":0,"Value":0.0,"PnL":0.0,"Return":0.0,"ActionCounts":{}}
    prices={t:_price(t) for t in df["Ticker"]}
    df["Current_Price"]=df["Ticker"].map(prices)
    df["Invested_Value"]=df["Quantity"]*df["Average_Price"]
    df["Current_Value"]=df["Quantity"]*df["Current_Price"].fillna(0)
    df["PnL"]=df["Current_Value"]-df["Invested_Value"]
    df["Return_Pct"]=df.apply(lambda r:(r.PnL/r.Invested_Value*100) if r.Invested_Value else float("nan"),axis=1)
    def action(r):
        ret=r.Return_Pct
        if pd.isna(r.Current_Price): return "DATA WAIT"
        if ret >= 8: return "PARTIAL-PROFIT"
        if ret <= -35: return "RECOVERY-WATCH"
        if ret <= -15: return "HOLD / REVIEW"
        return "HOLD"
    df["Action"]=df.apply(action,axis=1)
    total_inv=df["Invested_Value"].sum(); total_val=df["Current_Value"].sum(); pnl=df["PnL"].sum()
    counts=df["Action"].value_counts().to_dict()
    summary={"Positions":len(df),"Value":float(total_val),"PnL":float(pnl),"Return":float(pnl/total_inv*100) if total_inv else 0.0,"ActionCounts":counts}
    return df,summary

def portfolio_report_lines(limit=10):
    df,s=portfolio_snapshot()
    if df.empty: return ["💼 *PORTFOLIO MANAGER*", "No valid portfolio CSV found."]
    lines=["💼 *PORTFOLIO MANAGER*",f"Positions: {s['Positions']} | Value: ₹{s['Value']:,.0f} | P&L: ₹{s['PnL']:+,.0f} ({s['Return']:+.2f}%)"]
    rows=[]
    for _,r in df.sort_values("PnL").head(limit).iterrows():
        price="-" if pd.isna(r.Current_Price) else f"₹{r.Current_Price:,.2f}"
        avg="-" if pd.isna(r.Average_Price) else f"₹{r.Average_Price:,.2f}"
        ret="-" if pd.isna(r.Return_Pct) else f"{r.Return_Pct:+.1f}%"
        rows.append(f"• {r.Stock}: {price} | Avg {avg} | {ret} | {r.Action}")
    lines += rows
    return lines
