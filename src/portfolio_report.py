"""Standalone Portfolio Manager adapter.
Reads portfolio_manager/data/my_portfolio.csv and NEVER feeds portfolio data into model training.
Estimates missing average cost from supplied P&L/return, attaches latest AI target,
and calculates whether averaging can reasonably produce profit at that target.
"""
from pathlib import Path
import re
import pandas as pd
import yfinance as yf
from .config import PREDICTIONS_DIR

ROOT=Path(__file__).resolve().parents[1]
PORTFOLIO_FILE=ROOT/"portfolio_manager"/"data"/"my_portfolio.csv"
NAME_TO_TICKER={"RELIANCE INDUSTRIES":"RELIANCE.NS","RELIANCE":"RELIANCE.NS","VEDANTA IRON & STEEL":"VEDL.NS","VEDANTA":"VEDL.NS","YES BANK":"YESBANK.NS","IRFC":"IRFC.NS","NTPC":"NTPC.NS","TATA POWER":"TATAPOWER.NS","WIPRO":"WIPRO.NS","PALASH SECURITIES":"PALASHSECU.NS","OLA ELECTRIC MOBILITY":"OLAELEC.NS","STAR CEMENT":"STARCEMENT.NS","SJVN":"SJVN.NS","RELIANCE POWER":"RPOWER.NS","IRCTC":"IRCTC.NS","SEPC":"SEPC.NS","INDIAN RENEWABLE ENERGY":"IREDA.NS","IREDA":"IREDA.NS"}
TARGET_PROFIT_PCT=5.0
MAX_AVERAGING_CAPITAL_PCT=25.0

def _ticker(value):
    raw=str(value).strip();key=re.sub(r"\s+"," ",raw.upper())
    if key in NAME_TO_TICKER:return NAME_TO_TICKER[key]
    if raw.upper().endswith(".NS"):return raw.upper()
    return raw.upper().replace(" & ","").replace(" ","")+".NS"

def load_portfolio():
    if not PORTFOLIO_FILE.exists():return pd.DataFrame()
    df=pd.read_csv(PORTFOLIO_FILE)
    if {"Stock","Quantity","Average_Price"}.issubset(df.columns):
        out=df[["Stock","Quantity","Average_Price"]].copy();out["Reported_PnL"]=pd.NA;out["Reported_Return"]=pd.NA
    elif {"Stock","Quantity","Current_PnL_INR","Return_Percent"}.issubset(df.columns):
        out=df[["Stock","Quantity","Current_PnL_INR","Return_Percent"]].copy().rename(columns={"Current_PnL_INR":"Reported_PnL","Return_Percent":"Reported_Return"});out["Average_Price"]=pd.NA
    else:return pd.DataFrame()
    for c in ["Quantity","Average_Price","Reported_PnL","Reported_Return"]:out[c]=pd.to_numeric(out[c],errors="coerce")
    out["Quantity"]=out["Quantity"].fillna(0);out["Ticker"]=out["Stock"].map(_ticker);return out

def _price(ticker):
    try:
        d=yf.download(ticker,period="5d",interval="1d",auto_adjust=False,progress=False,threads=False)
        if d is None or d.empty:return None
        if isinstance(d.columns,pd.MultiIndex):d=d.xs(ticker,axis=1,level=-1) if ticker in d.columns.get_level_values(-1) else d.droplevel(-1,axis=1)
        s=pd.to_numeric(d["Close"],errors="coerce").dropna();return float(s.iloc[-1]) if not s.empty else None
    except Exception:return None

def _latest_predictions():
    files=sorted(PREDICTIONS_DIR.glob("predictions_*.csv"))
    for path in reversed(files):
        try:
            df=pd.read_csv(path)
            if not df.empty and "Symbol" in df.columns:return df,path.stem.replace("predictions_","")
        except Exception:continue
    return pd.DataFrame(),None

def _attach_predictions(df):
    pred,pred_date=_latest_predictions()
    if pred.empty:return df,pred_date
    keep=[c for c in ["Symbol","Pred_Close","Pred_Open","Pred_High","Pred_Low","Confidence","Direction","FinalDecisionScore","Action","Horizon_1D","Horizon_3D","Horizon_5D","Horizon_7D","Horizon_20D"] if c in pred.columns]
    p=pred[keep].copy().rename(columns={"Symbol":"Ticker","Pred_Close":"AI_Target","Pred_Open":"AI_Open","Pred_High":"AI_High","Pred_Low":"AI_Low","Action":"AI_Action"})
    return df.merge(p,on="Ticker",how="left"),pred_date

def _average_plan(row):
    qty=float(row["Quantity"] or 0);price=row["Current_Price"];avg=row["Average_Price"];target=row["AI_Target"]
    if pd.isna(avg) and price is not None and pd.notna(row["Reported_Return"]) and float(row["Reported_Return"])>-100:
        avg=price/(1+float(row["Reported_Return"])/100);row["Average_Price"]=avg;row["AveragePriceSource"]="ESTIMATED_FROM_RETURN"
    elif pd.isna(avg) and price is not None and qty>0 and pd.notna(row["Reported_PnL"]):
        avg=price-float(row["Reported_PnL"])/qty;row["Average_Price"]=avg;row["AveragePriceSource"]="ESTIMATED_FROM_PNL"
    elif pd.notna(avg):row["AveragePriceSource"]="CSV"
    else:row["AveragePriceSource"]="UNAVAILABLE"
    if price is None or pd.isna(avg) or qty<=0:
        row["Invested_Value"]=qty*avg if pd.notna(avg) else 0;row["Recovery_Gap_Pct"]=None;row["Target_Return_Pct"]=None;row["Recommended_Qty"]=0;row["New_Average_Price"]=None;row["Averaging_Action"]="DATA WAIT";return row
    row["Invested_Value"]=qty*float(avg);row["Recovery_Gap_Pct"]=(float(avg)-price)/float(avg)*100 if avg else None
    if pd.isna(target):row["Target_Return_Pct"]=None;row["Recommended_Qty"]=0;row["New_Average_Price"]=float(avg);row["Averaging_Action"]="WAIT FOR AI TARGET";return row
    target=float(target);row["Target_Return_Pct"]=(target/float(avg)-1)*100
    if target<=price or price>=float(avg):
        row["Recommended_Qty"]=0;row["New_Average_Price"]=float(avg);row["Averaging_Action"]="DO NOT AVERAGE";return row
    desired_avg=target/(1+TARGET_PROFIT_PCT/100)
    if desired_avg<=price:
        row["Recommended_Qty"]=0;row["New_Average_Price"]=float(avg);row["Averaging_Action"]="DO NOT AVERAGE";return row
    required=qty*(float(avg)-desired_avg)/(desired_avg-price);budget=qty*float(avg)*MAX_AVERAGING_CAPITAL_PCT/100;max_qty=int(budget//price);rec=min(max(0,int(required+0.9999)),max_qty);new_avg=(qty*float(avg)+rec*price)/(qty+rec) if rec>0 else float(avg);projected=(target/new_avg-1)*100
    row["Recommended_Qty"]=rec;row["New_Average_Price"]=new_avg;row["Projected_Return_At_AI_Target"]=projected;row["Max_Averaging_Capital"]=budget;row["Averaging_Action"]="AVERAGE" if rec>0 and projected>=TARGET_PROFIT_PCT else "DO NOT AVERAGE";return row

def portfolio_snapshot():
    df=load_portfolio()
    if df.empty:return df,{"Positions":0,"Value":0.0,"PnL":0.0,"Return":0.0,"ActionCounts":{}}
    prices={t:_price(t) for t in df["Ticker"].dropna().unique()};df["Current_Price"]=df["Ticker"].map(prices);df,prediction_date=_attach_predictions(df)
    # Pre-create the column because pandas Series attribute assignment is not a reliable way to add a field.
    df["AveragePriceSource"]="UNAVAILABLE"
    df=df.apply(_average_plan,axis=1);df["Current_Value"]=df["Quantity"]*df["Current_Price"].fillna(0);mask=df["Reported_PnL"].notna()&df["AveragePriceSource"].str.startswith("ESTIMATED");df["PnL"]=df["Current_Value"]-df["Invested_Value"];df.loc[mask,"PnL"]=df.loc[mask,"Reported_PnL"]
    df["Return_Pct"]=df.apply(lambda r:float(r["Reported_Return"]) if str(r["AveragePriceSource"]).startswith("ESTIMATED") and pd.notna(r["Reported_Return"]) else ((r["PnL"]/r["Invested_Value"]*100) if r["Invested_Value"] else None),axis=1)
    def action(r):
        if pd.isna(r["Current_Price"]):return "DATA WAIT"
        if r["Averaging_Action"]=="AVERAGE":return "AVERAGE"
        if pd.notna(r["AI_Target"]) and r["AI_Target"]>r["Current_Price"]:return "RECOVERY / HOLD"
        ret=r["Return_Pct"]
        if pd.notna(ret) and ret>=8:return "PARTIAL-PROFIT"
        if pd.notna(ret) and ret<=-35:return "RECOVERY-WATCH"
        if pd.notna(ret) and ret<=-15:return "HOLD / REVIEW"
        return "HOLD"
    df["Action"]=df.apply(action,axis=1);total_inv=float(df["Invested_Value"].sum());total_val=float(df["Current_Value"].sum());total_pnl=float(df["PnL"].sum());summary={"Positions":len(df),"Value":total_val,"PnL":total_pnl,"Return":total_pnl/total_inv*100 if total_inv else 0.0,"ActionCounts":df["Action"].value_counts().to_dict(),"PredictionDate":prediction_date};return df,summary
