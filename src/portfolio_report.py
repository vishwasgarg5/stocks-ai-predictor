"""Standalone portfolio-manager adapter for Telegram reports.
Reads portfolio_manager/data/my_portfolio.csv and never feeds portfolio data into model training."""
from pathlib import Path
import re
import pandas as pd
import yfinance as yf
ROOT=Path(__file__).resolve().parents[1]; PORTFOLIO_FILE=ROOT/"portfolio_manager"/"data"/"my_portfolio.csv"
NAME_TO_TICKER={"RELIANCE INDUSTRIES":"RELIANCE.NS","RELIANCE":"RELIANCE.NS","VEDANTA IRON & STEEL":"VEDL.NS","VEDANTA":"VEDL.NS","YES BANK":"YESBANK.NS","IRFC":"IRFC.NS","NTPC":"NTPC.NS","TATA POWER":"TATAPOWER.NS","WIPRO":"WIPRO.NS","PALASH SECURITIES":"PALASHSECU.NS","OLA ELECTRIC MOBILITY":"OLAELEC.NS","STAR CEMENT":"STARCEMENT.NS","SJVN":"SJVN.NS","RELIANCE POWER":"RPOWER.NS","IRCTC":"IRCTC.NS","SEPC":"SEPC.NS","INDIAN RENEWABLE ENERGY":"IREDA.NS","IREDA":"IREDA.NS"}
def _ticker(value):
    raw=str(value).strip();key=re.sub(r"\s+"," ",raw.upper())
    if key in NAME_TO_TICKER:return NAME_TO_TICKER[key]
    if raw.upper().endswith(".NS"):return raw.upper()
    return raw.upper().replace(" & ","").replace(" ","")+".NS"
def load_portfolio():
    if not PORTFOLIO_FILE.exists():return pd.DataFrame()
    df=pd.read_csv(PORTFOLIO_FILE)
    if {"Stock","Quantity","Average_Price"}.issubset(df.columns):out=df[["Stock","Quantity","Average_Price"]].copy();out["Reported_PnL"]=float("nan");out["Reported_Return"]=float("nan")
    elif {"Stock","Quantity","Current_PnL_INR","Return_Percent"}.issubset(df.columns):out=df[["Stock","Quantity","Current_PnL_INR","Return_Percent"]].copy();out.rename(columns={"Current_PnL_INR":"Reported_PnL","Return_Percent":"Reported_Return"},inplace=True);out["Average_Price"]=float("nan")
    else:return pd.DataFrame()
    out["Ticker"]=out["Stock"].map(_ticker);out["Quantity"]=pd.to_numeric(out["Quantity"],errors="coerce").fillna(0);out["Average_Price"]=pd.to_numeric(out["Average_Price"],errors="coerce");out["Reported_PnL"]=pd.to_numeric(out["Reported_PnL"],errors="coerce");out["Reported_Return"]=pd.to_numeric(out["Reported_Return"],errors="coerce");return out
def _price(ticker):
    try:
        d=yf.download(ticker,period="5d",interval="1d",auto_adjust=False,progress=False,threads=False)
        if d is None or d.empty:return None
        if isinstance(d.columns,pd.MultiIndex):d=d.xs(ticker,axis=1,level=-1) if ticker in d.columns.get_level_values(-1) else d.droplevel(-1,axis=1)
        return float(d["Close"].dropna().iloc[-1])
    except Exception:return None
def portfolio_snapshot():
    df=load_portfolio()
    if df.empty:return df,{"Positions":0,"Value":0.0,"PnL":0.0,"Return":0.0,"ActionCounts":{}}
    df["Current_Price"]=df["Ticker"].map({t:_price(t) for t in df["Ticker"]});df["Invested_Value"]=df["Quantity"]*df["Average_Price"];df["Current_Value"]=df["Quantity"]*df["Current_Price"].fillna(0)
    # Screenshot-format CSV already contains realized/current P&L, so preserve it when average cost is absent.
    df["PnL"]=df["Current_Value"]-df["Invested_Value"];mask=df["Average_Price"].isna()&df["Reported_PnL"].notna();df.loc[mask,"PnL"]=df.loc[mask,"Reported_PnL"]
    df["Return_Pct"]=df.apply(lambda r:r.Reported_Return if pd.isna(r.Average_Price) and pd.notna(r.Reported_Return) else ((r.PnL/r.Invested_Value*100) if r.Invested_Value else float("nan")),axis=1)
    def action(r):
        ret=r.Return_Pct
        if pd.isna(r.Current_Price):return "DATA WAIT"
        if ret>=8:return "PARTIAL-PROFIT"
        if ret<=-35:return "RECOVERY-WATCH"
        if ret<=-15:return "HOLD / REVIEW"
        return "HOLD"
    df["Action"]=df.apply(action,axis=1);total_inv=df["Invested_Value"].sum();total_val=df["Current_Value"].sum();pnl=df["PnL"].sum();reported=df.loc[mask,"PnL"].sum()
    if mask.any():total_pnl=float(pnl);total_return=float(df["Return_Pct"].mean()) if total_inv==0 else float(total_pnl/total_inv*100)
    else:total_pnl=float(pnl);total_return=float(total_pnl/total_inv*100) if total_inv else 0.0
    return df,{"Positions":len(df),"Value":float(total_val),"PnL":total_pnl,"Return":total_return,"ActionCounts":df["Action"].value_counts().to_dict()}
