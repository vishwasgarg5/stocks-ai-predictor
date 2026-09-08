"""Stage 10.5 portfolio manager.

Design goals:
- deterministic schema: every output row always has Decision/PnL/Return_Pct
- use repository OHLCV cache before Yahoo Finance
- never train models or make 5y downloads just to render the morning report
- consume the exact latest prediction artifact when available
- conservative averaging and 10% profit-target logic
"""
from pathlib import Path
import re
from datetime import timedelta
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO_FILE = ROOT / "portfolio_manager" / "data" / "my_portfolio.csv"
OHLCV_DIR = ROOT / "data" / "ohlcv"
PREDICTIONS_DIR = ROOT / "data" / "stage2" / "predictions"
TARGET_PROFIT_PCT = 10.0
MAX_AVERAGING_CAPITAL_PCT = 25.0
MIN_AI_CONFIDENCE = 60.0
SELL_RISK_GAP_PCT = 3.0
NSE_HOLIDAYS = {"2026-01-26","2026-03-03","2026-03-26","2026-03-31","2026-04-03","2026-04-14","2026-05-01","2026-05-28","2026-06-26","2026-08-15","2026-08-28","2026-09-14","2026-10-02","2026-10-20","2026-11-09","2026-11-24","2026-12-25"}
NAME_TO_TICKER = {"RELIANCE INDUSTRIES":"RELIANCE.NS","RELIANCE":"RELIANCE.NS","VEDANTA IRON & STEEL":"VEDL.NS","VEDANTA":"VEDL.NS","YES BANK":"YESBANK.NS","IRFC":"IRFC.NS","NTPC":"NTPC.NS","TATA POWER":"TATAPOWER.NS","WIPRO":"WIPRO.NS","PALASH SECURITIES":"PALASHSECU.NS","OLA ELECTRIC MOBILITY":"OLAELEC.NS","STAR CEMENT":"STARCEMENT.NS","SJVN":"SJVN.NS","RELIANCE POWER":"RPOWER.NS","IRCTC":"IRCTC.NS","SEPC":"SEPC.NS","INDIAN RENEWABLE ENERGY":"IREDA.NS","IREDA":"IREDA.NS"}
OUTPUT_COLUMNS=["Stock","Ticker","Quantity","Average_Price","Current_Price","Invested_Value","Current_Value","PnL","Current_PnL_INR","Return_Pct","AI_Target","AI_Confidence","AI_Direction","Decision","Sell_Window","Profit_Target_Price","Sell_Target_Price","Recommended_Qty","New_Average_Price","Projected_Return_At_AI_Target","Recovery_Gap_Pct","Sell_Reason","PredictionDate","PriceSource"]

def _is_nse_trading_day(value):
    try:d=pd.Timestamp(value).date()
    except Exception:return False
    return d.weekday()<5 and d.isoformat() not in NSE_HOLIDAYS

def _next_trading_date(value,days=1):
    try:d=pd.Timestamp(value).date()
    except Exception:raise ValueError(f"Invalid date: {value!r}")
    for _ in range(max(0,int(days))):
        d += timedelta(days=1)
        while not _is_nse_trading_day(d): d += timedelta(days=1)
    return d.isoformat()

def _ticker(value):
    raw=str(value).strip(); key=re.sub(r"\s+"," ",raw.upper())
    if key in NAME_TO_TICKER:return NAME_TO_TICKER[key]
    if raw.upper().endswith(".NS"):return raw.upper()
    return raw.upper().replace(" & ","").replace(" ","")+".NS"

def _symbol(ticker):return str(ticker).upper().removesuffix(".NS")

def load_portfolio():
    if not PORTFOLIO_FILE.exists():return pd.DataFrame(columns=["Stock","Ticker","Quantity","Average_Price","Reported_PnL","Reported_Return"])
    try:src=pd.read_csv(PORTFOLIO_FILE)
    except Exception:return pd.DataFrame(columns=["Stock","Ticker","Quantity","Average_Price","Reported_PnL","Reported_Return"])
    source_col="Symbol" if "Symbol" in src.columns else "Stock" if "Stock" in src.columns else None
    if source_col is None or "Quantity" not in src.columns:return pd.DataFrame(columns=["Stock","Ticker","Quantity","Average_Price","Reported_PnL","Reported_Return"])
    out=pd.DataFrame();out["Stock"]=src[source_col].astype(str).str.strip();out["Quantity"]=pd.to_numeric(src["Quantity"],errors="coerce").fillna(0);out["Average_Price"]=pd.to_numeric(src.get("Average_Price",np.nan),errors="coerce");out["Reported_PnL"]=pd.to_numeric(src.get("Current_PnL_INR",np.nan),errors="coerce");out["Reported_Return"]=pd.to_numeric(src.get("Return_Percent",np.nan),errors="coerce");out["Ticker"]=out["Stock"].map(_ticker);out["Stock"]=out["Ticker"].map(_symbol);return out

def _cached_price(ticker):
    path=OHLCV_DIR/f"{_symbol(ticker)}.csv"
    try:
        if not path.exists():return None,"UNAVAILABLE"
        d=pd.read_csv(path)
        if d.empty or "Close" not in d.columns:return None,"UNAVAILABLE"
        s=pd.to_numeric(d["Close"],errors="coerce").dropna()
        return (float(s.iloc[-1]),"GITHUB_OHLCV") if not s.empty else (None,"UNAVAILABLE")
    except Exception:return None,"UNAVAILABLE"

def _latest_predictions():
    files=sorted(PREDICTIONS_DIR.glob("predictions_*.csv"),key=lambda p:p.stat().st_mtime)
    for path in reversed(files):
        try:
            d=pd.read_csv(path)
            if not d.empty and "Symbol" in d.columns:
                d["Ticker"]=d["Symbol"].astype(str).map(_ticker);return d,path.stem.replace("predictions_","")
        except Exception:continue
    return pd.DataFrame(),None

def _num(row,name,default=np.nan):
    try:v=float(row.get(name,default));return v if np.isfinite(v) else default
    except (TypeError,ValueError):return default

def _forecast_return(row):
    vals=[]
    for h in (1,3,5,7,10,20):
        v=_num(row,f"Horizon_{h}D")
        if np.isfinite(v):vals.append((h,v))
    if vals:return vals
    v=_num(row,"Expected_Return");return [(1,v)] if np.isfinite(v) else []

def _decision(current,avg,target,confidence,forecasts):
    if current is None or not np.isfinite(current) or not np.isfinite(avg) or avg<=0:return "WAIT","NO PRICE / COST DATA"
    if not np.isfinite(target):return "HOLD","AI TARGET UNAVAILABLE"
    target_gap=(target/avg-1.0)*100.0; recovery_gap=(avg-current)/avg*100.0
    if current+1e-9 >= avg*(1+TARGET_PROFIT_PCT/100):return "SELL","10% profit target already reached"
    if current+1e-9 >= target and target>avg:return "SELL","AI target reached"
    if target < current*(1-SELL_RISK_GAP_PCT/100):return "SELL","AI target is materially below current price"
    positive=[v for _,v in forecasts if np.isfinite(v) and v>=TARGET_PROFIT_PCT]
    if recovery_gap>=5 and positive and confidence>=MIN_AI_CONFIDENCE:return "AVG","Multi-horizon recovery supports limited averaging"
    if target_gap>=TARGET_PROFIT_PCT and recovery_gap>0:return "HOLD","Recovery target remains above cost"
    return "HOLD","No sufficiently strong recovery confirmation"

def _plan_row(row,pred_row,prediction_date):
    ticker=row["Ticker"];current,price_source=_cached_price(ticker);qty=float(row.get("Quantity",0) or 0);avg=row.get("Average_Price",np.nan);reported_pnl=row.get("Reported_PnL",np.nan);reported_return=row.get("Reported_Return",np.nan)
    if pd.isna(avg) and current is not None and pd.notna(reported_return) and float(reported_return)>-100:avg=current/(1+float(reported_return)/100)
    if pd.isna(avg) and current is not None and qty>0 and pd.notna(reported_pnl):avg=current-float(reported_pnl)/qty
    target=_num(pred_row,"Pred_Close") if pred_row is not None else np.nan;confidence=_num(pred_row,"CalibratedConfidence",_num(pred_row,"Confidence",0)) if pred_row is not None else 0.0;direction=str(pred_row.get("Direction","-")) if pred_row is not None else "-";forecasts=_forecast_return(pred_row) if pred_row is not None else []
    invested=qty*float(avg) if pd.notna(avg) else 0.0;current_value=qty*current if current is not None else 0.0;pnl=current_value-invested;ret=pnl/invested*100 if invested>0 else np.nan;profit_target=float(avg)*(1+TARGET_PROFIT_PCT/100) if pd.notna(avg) else np.nan;recovery_gap=((float(avg)-current)/float(avg)*100) if current is not None and pd.notna(avg) and float(avg) else np.nan
    decision,reason=_decision(current,float(avg) if pd.notna(avg) else np.nan,target,confidence,forecasts);recommended_qty=0;new_avg=float(avg) if pd.notna(avg) else np.nan
    if decision=="AVG" and current is not None and pd.notna(avg) and np.isfinite(target):
        positive=max((v for _,v in forecasts if np.isfinite(v) and v>=TARGET_PROFIT_PCT),default=TARGET_PROFIT_PCT);desired_avg=target/(1+positive/100)
        if desired_avg>current:
            required=qty*(float(avg)-desired_avg)/(desired_avg-current);budget=qty*float(avg)*MAX_AVERAGING_CAPITAL_PCT/100;max_qty=int(max(0,budget//current)) if current>0 else 0;recommended_qty=min(max(0,int(np.ceil(required))),max_qty)
            if recommended_qty:new_avg=(qty*float(avg)+recommended_qty*current)/(qty+recommended_qty)
            else:decision="HOLD";reason="Averaging cap does not justify additional quantity"
    target_price=float(target) if np.isfinite(target) else np.nan;projected=((target/new_avg)-1)*100 if np.isfinite(target) and np.isfinite(new_avg) and new_avg else np.nan;sell_window="NOW" if decision=="SELL" else ("MONITOR" if np.isfinite(target_price) else "NO AI DATA")
    return {"Stock":_symbol(ticker),"Ticker":ticker,"Quantity":int(qty),"Average_Price":avg,"Current_Price":current,"Invested_Value":invested,"Current_Value":current_value,"PnL":pnl,"Current_PnL_INR":pnl,"Return_Pct":ret,"AI_Target":target,"AI_Confidence":confidence,"AI_Direction":direction,"Decision":decision,"Sell_Window":sell_window,"Profit_Target_Price":target_price if np.isfinite(target_price) else profit_target,"Sell_Target_Price":target_price,"Recommended_Qty":recommended_qty,"New_Average_Price":new_avg,"Projected_Return_At_AI_Target":projected,"Recovery_Gap_Pct":recovery_gap,"Sell_Reason":reason,"PredictionDate":prediction_date or "-","PriceSource":price_source}

def portfolio_snapshot():
    portfolio=load_portfolio();empty={"Positions":0,"Value":0.0,"PnL":0.0,"Return":0.0,"ActionCounts":{}}
    if portfolio.empty:return pd.DataFrame(columns=OUTPUT_COLUMNS),empty
    predictions,prediction_date=_latest_predictions();pred_by_ticker={}
    if not predictions.empty:
        for _,r in predictions.drop_duplicates("Ticker",keep="last").iterrows():pred_by_ticker[str(r["Ticker"])]=r
    df=pd.DataFrame([_plan_row(row,pred_by_ticker.get(str(row["Ticker"])),prediction_date) for _,row in portfolio.iterrows()])
    for c in OUTPUT_COLUMNS:
        if c not in df.columns:df[c]=np.nan
    df=df[OUTPUT_COLUMNS];df["PnL"]=pd.to_numeric(df["PnL"],errors="coerce").fillna(0.0);df["Current_PnL_INR"]=df["PnL"];df["Return_Pct"]=pd.to_numeric(df["Return_Pct"],errors="coerce");df["Decision"]=df["Decision"].fillna("WAIT").astype(str)
    total_value=float(pd.to_numeric(df["Current_Value"],errors="coerce").fillna(0).sum());total_invested=float(pd.to_numeric(df["Invested_Value"],errors="coerce").fillna(0).sum());total_pnl=total_value-total_invested
    return df,{"Positions":int(len(df)),"Value":total_value,"PnL":total_pnl,"Return":total_pnl/total_invested*100 if total_invested else 0.0,"ActionCounts":df["Decision"].value_counts().to_dict(),"Available":True}
