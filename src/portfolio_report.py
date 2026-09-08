"""Stage 28 portfolio decision engine.

Portfolio calculations are fail-closed and consume the canonical GitHub OHLCV
snapshot/cache. Direct per-position Yahoo downloads are intentionally removed.
Prediction rows are selected by prediction lineage/date, never file mtime.
"""
from __future__ import annotations
from datetime import timedelta
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .data_snapshot import get_snapshot
from .utils import today_ist

ROOT=Path(__file__).resolve().parents[1]
PORTFOLIO_FILE=ROOT/"portfolio_manager"/"data"/"my_portfolio.csv"
PREDICTIONS_DIR=ROOT/"data"/"stage2"/"predictions"
TARGET_PROFIT_PCT=10.0; MAX_AVERAGING_CAPITAL_PCT=25.0; MIN_AI_CONFIDENCE=60.0; SELL_RISK_GAP_PCT=3.0
NSE_HOLIDAYS={"2026-01-26","2026-03-03","2026-03-26","2026-03-31","2026-04-03","2026-04-14","2026-05-01","2026-05-28","2026-06-26","2026-08-15","2026-08-28","2026-09-14","2026-10-02","2026-10-20","2026-11-09","2026-11-24","2026-12-25"}
NAME_TO_TICKER={"RELIANCE INDUSTRIES":"RELIANCE.NS","RELIANCE":"RELIANCE.NS","VEDANTA":"VEDL.NS","YES BANK":"YESBANK.NS","IRFC":"IRFC.NS","NTPC":"NTPC.NS","TATA POWER":"TATAPOWER.NS","WIPRO":"WIPRO.NS","PALASH SECURITIES":"PALASHSECU.NS","OLA ELECTRIC MOBILITY":"OLAELEC.NS","STAR CEMENT":"STARCEMENT.NS","SJVN":"SJVN.NS","RELIANCE POWER":"RPOWER.NS","IRCTC":"IRCTC.NS","SEPC":"SEPC.NS","INDIAN RENEWABLE ENERGY":"IREDA.NS","IREDA":"IREDA.NS","VEDANTA IRON & STEEL":"VEDL.NS"}
OUTPUT_COLUMNS=["Stock","Ticker","Quantity","Average_Price","Current_Price","Invested_Value","Current_Value","PnL","Current_PnL_INR","Return_Pct","AI_Target","AI_Confidence","AI_Direction","Decision","Sell_Window","Profit_Target_Price","Sell_Target_Price","Recommended_Qty","New_Average_Price","Projected_Return_At_AI_Target","Recovery_Gap_Pct","Sell_Reason","PredictionDate","PriceSource"]
HORIZONS=(1,3,5,7,10,20,60,90,180,365)

def _ticker(value):
    raw=str(value).strip(); key=re.sub(r"\s+"," ",raw.upper())
    if key in NAME_TO_TICKER:return NAME_TO_TICKER[key]
    return raw.upper() if raw.upper().endswith(".NS") else raw.upper().replace(" & ","").replace(" ","")+".NS"
def _symbol(ticker):return str(ticker).upper().removesuffix(".NS")
def _is_nse_trading_day(value):
    try:d=pd.Timestamp(value).date()
    except Exception:return False
    return d.weekday()<5 and d.isoformat() not in NSE_HOLIDAYS
def _next_trading_date(value,days=1):
    d=pd.Timestamp(value).date()
    for _ in range(max(0,int(days))):
        d+=timedelta(days=1)
        while not _is_nse_trading_day(d):d+=timedelta(days=1)
    return d.isoformat()
def _num(row,name,default=np.nan):
    try:
        v=float(row.get(name,default)); return v if np.isfinite(v) else default
    except Exception:return default

def load_portfolio():
    cols=["Stock","Ticker","Quantity","Average_Price","Reported_PnL","Reported_Return"]
    if not PORTFOLIO_FILE.exists():return pd.DataFrame(columns=cols)
    try:src=pd.read_csv(PORTFOLIO_FILE)
    except Exception:return pd.DataFrame(columns=cols)
    source="Symbol" if "Symbol" in src.columns else "Stock" if "Stock" in src.columns else None
    if source is None or "Quantity" not in src.columns:return pd.DataFrame(columns=cols)
    out=pd.DataFrame(); out["Stock"]=src[source].astype(str).str.strip(); out["Quantity"]=pd.to_numeric(src["Quantity"],errors="coerce").fillna(0)
    out["Average_Price"]=pd.to_numeric(src.get("Average_Price",np.nan),errors="coerce")
    out["Reported_PnL"]=pd.to_numeric(src.get("Current_PnL_INR",np.nan),errors="coerce")
    out["Reported_Return"]=pd.to_numeric(src.get("Return_Percent",np.nan),errors="coerce")
    out["Ticker"]=out["Stock"].map(_ticker); out["Stock"]=out["Ticker"].map(_symbol)
    return out

def _effective_cutoff(requested=None):
    if requested is not None:return pd.Timestamp(requested).date()
    d=pd.Timestamp(today_ist()).date()
    if _is_nse_trading_day(d):return d
    while not _is_nse_trading_day(d):d-=timedelta(days=1)
    return d

def _cached_history(ticker,cutoff_date=None):
    """Canonical cache-only history; no direct Yahoo fallback."""
    cutoff=_effective_cutoff(cutoff_date)
    d,snap=get_snapshot(_symbol(ticker),cutoff_date=cutoff,min_rows=60,allow_download=False)
    if snap.status not in {"OK","STALE"} or d.empty:return pd.DataFrame()
    return d

def _latest_price(ticker,cutoff_date=None):
    d=_cached_history(ticker,cutoff_date)
    if d.empty or "Close" not in d:return None,"UNAVAILABLE"
    s=pd.to_numeric(d["Close"],errors="coerce").dropna()
    return (float(s.iloc[-1]),"GITHUB_OHLCV") if not s.empty else (None,"UNAVAILABLE")

def _latest_predictions():
    frames=[]
    for p in PREDICTIONS_DIR.glob("predictions_*.csv"):
        try:
            d=pd.read_csv(p)
            if d.empty or "Symbol" not in d.columns:continue
            d["_source_file"]=p.name;frames.append(d)
        except Exception:continue
    if not frames:return pd.DataFrame(),None
    allp=pd.concat(frames,ignore_index=True)
    date_col=next((c for c in ("PredictionDate","Run_Date","Cutoff_Date") if c in allp.columns),None)
    if date_col:
        allp["_lineage_date"]=pd.to_datetime(allp[date_col],errors="coerce")
        allp=allp.sort_values("_lineage_date",kind="stable")
        valid=allp["_lineage_date"].dropna()
        if not valid.empty:allp=allp[allp["_lineage_date"]==valid.max()]
    else:allp=allp.sort_values("_source_file",kind="stable")
    allp=allp.drop_duplicates("Symbol",keep="last")
    latest=None
    if "_lineage_date" in allp.columns and allp["_lineage_date"].notna().any():latest=str(allp["_lineage_date"].max().date())
    return allp,latest

def _attach_predictions(base):
    pred,date=_latest_predictions();out=base.copy()
    if pred.empty:return out,date
    pred["Ticker"]=pred["Symbol"].astype(str).map(_ticker)
    keep=[c for c in ["Ticker","Prediction_ID","Pred_Close","Pred_Open","Pred_High","Pred_Low","Confidence","CalibratedConfidence","Direction","Action"]+[f"Horizon_{h}D" for h in HORIZONS] if c in pred.columns]
    return out.merge(pred[keep].drop_duplicates("Ticker",keep="last"),on="Ticker",how="left"),date

def _average_plan(row):
    avg=float(row.get("Average_Price",np.nan));target=float(row.get("AI_Target",row.get("Pred_Close",np.nan)))
    row["Profit_Target_Price"]=round(avg*(1+TARGET_PROFIT_PCT/100),10) if np.isfinite(avg) else np.nan
    row["Projected_Return_At_AI_Target"]=round((target/avg-1)*100,10) if np.isfinite(avg) and avg else np.nan
    return row

def _forecast_return(row):return [(h,_num(row,f"Horizon_{h}D")) for h in HORIZONS if np.isfinite(_num(row,f"Horizon_{h}D"))]

def _sell_plan(row,current_price,avg_price,prediction_date):
    avg=_num({"v":avg_price},"v");confidence=_num(row,"AI_Confidence",_num(row,"Confidence",0.0));candidates=[]
    for h in (20,60,90,180,365):
        value=_num(row,f"Horizon_{h}D")
        if np.isfinite(value) and value>=TARGET_PROFIT_PCT:candidates.append((h,value))
    price=round(avg*(1+TARGET_PROFIT_PCT/100),10) if np.isfinite(avg) else np.nan
    if confidence<MIN_AI_CONFIDENCE or not candidates:return TARGET_PROFIT_PCT,price,"-","-","WAIT"
    h,profit=max(candidates,key=lambda x:(x[1],-x[0]));return float(profit),round(avg*(1+profit/100),10),f"{h}D",_next_trading_date(prediction_date,h),"TARGET_DATE"

def _decision(current,avg,target,confidence,forecasts):
    if current is None or not np.isfinite(current) or not np.isfinite(avg) or avg<=0:return "WAIT","NO PRICE / COST DATA"
    if not np.isfinite(target):return "HOLD","AI PREDICTION UNAVAILABLE"
    profit_target=avg*(1+TARGET_PROFIT_PCT/100)
    if current>=profit_target:return "SELL","10% profit target reached"
    if target>=profit_target and current<avg:
        strong=sum(v>=TARGET_PROFIT_PCT for _,v in forecasts)
        if strong>=2 and confidence>=MIN_AI_CONFIDENCE:return "HOLD","MULTI_HORIZON_CONFIRMED recovery"
        return "HOLD","AI recovery target remains above cost"
    if target<current*(1-SELL_RISK_GAP_PCT/100):return "REDUCE","AI target materially below current price"
    return "HOLD","AI recovery not yet confirmed"

def _portfolio_ai(portfolio,cutoff_date=None,variant="A"):
    if portfolio.empty:return pd.DataFrame()
    from .prediction import train_stock_bundle,predict_stock,add_multihorizon_predictions
    from .multihorizon import train_horizon_models
    rows=[];cutoff=_effective_cutoff(cutoff_date)
    for _,r in portfolio.iterrows():
        ticker=str(r["Ticker"]);symbol=_symbol(ticker);d=_cached_history(ticker,cutoff)
        if d.empty or "Close" not in d or len(d)<150:
            print(f"{symbol}: portfolio AI skipped: canonical history unavailable or <150 rows");continue
        try:
            effective=pd.Timestamp(cutoff);d=d[d.index<=effective].dropna(subset=["Open","High","Low","Close","Volume"])
            if len(d)<150:continue
            bundle=train_stock_bundle(d,symbol,effective,variant,train_horizons=False);pred=predict_stock(d,bundle,effective);item={"Symbol":symbol,**pred,"PredictionDate":str(effective.date())}
            try:
                hb=train_horizon_models(d,effective);h=add_multihorizon_predictions(d,{"horizons":hb},effective)
                for _,hr in h.iterrows():item[f"Horizon_{int(hr.HorizonDays)}D"]=float(hr.Expected_Return)
            except Exception as exc:print(f"{symbol}: portfolio horizons skipped: {exc}")
            rows.append(item)
        except Exception as exc:print(f"{symbol}: portfolio AI skipped: {exc}")
    return pd.DataFrame(rows)

def _plan_row(row,pred,prediction_date):
    ticker=row["Ticker"];current,source=_latest_price(ticker,prediction_date);qty=float(row.get("Quantity",0) or 0);avg=row.get("Average_Price",np.nan)
    reported_pnl=row.get("Reported_PnL",np.nan);reported_return=row.get("Reported_Return",np.nan)
    if pred is not None and np.isfinite(_num(pred,"Current_Price")):current=_num(pred,"Current_Price");source="AI_OHLCV"
    if pd.isna(avg) and current is not None and pd.notna(reported_return):avg=current/(1+float(reported_return)/100)
    if pd.isna(avg) and current is not None and pd.notna(reported_pnl) and qty>0:avg=current-float(reported_pnl)/qty
    target=_num(pred,"Pred_Close") if pred is not None else np.nan;confidence=_num(pred,"Confidence",0) if pred is not None else 0;direction=str(pred.get("Direction","-") if pred is not None else "-");forecasts=_forecast_return(pred) if pred is not None else []
    invested=qty*float(avg) if pd.notna(avg) else 0.;value=qty*current if current is not None else 0.;pnl=value-invested;ret=pnl/invested*100 if invested else np.nan;profit_target=float(avg)*(1+TARGET_PROFIT_PCT/100) if pd.notna(avg) else np.nan;recovery=((float(avg)-current)/float(avg)*100) if current is not None and pd.notna(avg) and float(avg) else np.nan
    decision,reason=_decision(current,float(avg) if pd.notna(avg) else np.nan,target,confidence,forecasts);projected=(target/float(avg)-1)*100 if np.isfinite(target) and pd.notna(avg) and float(avg) else np.nan
    return {"Stock":_symbol(ticker),"Ticker":ticker,"Quantity":int(qty),"Average_Price":avg,"Current_Price":current,"Invested_Value":invested,"Current_Value":value,"PnL":pnl,"Current_PnL_INR":pnl,"Return_Pct":ret,"AI_Target":target,"AI_Confidence":confidence,"AI_Direction":direction,"Decision":decision,"Sell_Window":"NOW" if decision in {"SELL","REDUCE"} else ("WATCH" if np.isfinite(target) else "NO AI DATA"),"Profit_Target_Price":profit_target,"Sell_Target_Price":target,"Recommended_Qty":0,"New_Average_Price":avg,"Projected_Return_At_AI_Target":projected,"Recovery_Gap_Pct":recovery,"Sell_Reason":reason,"PredictionDate":prediction_date or "-","PriceSource":source}

def portfolio_snapshot(cutoff_date=None,variant="A"):
    portfolio=load_portfolio();empty={"Positions":0,"Value":0.,"PnL":0.,"Return":0.,"ActionCounts":{},"Available":bool(not portfolio.empty)}
    if portfolio.empty:return pd.DataFrame(columns=OUTPUT_COLUMNS),empty
    ai=_portfolio_ai(portfolio,cutoff_date,variant);ai_by={str(r.Symbol)+".NS":r for _,r in ai.iterrows()} if not ai.empty else {}
    _,date=_latest_predictions();prediction_date=str(cutoff_date or date or "-")
    rows=[_plan_row(r,ai_by.get(str(r.Ticker)),prediction_date) for _,r in portfolio.iterrows()];df=pd.DataFrame(rows)
    for c in OUTPUT_COLUMNS:
        if c not in df.columns:df[c]=np.nan
    df=df[OUTPUT_COLUMNS];df["PnL"]=pd.to_numeric(df["PnL"],errors="coerce").fillna(0);df["Current_PnL_INR"]=df["PnL"];invested=pd.to_numeric(df["Invested_Value"],errors="coerce").fillna(0).sum();value=pd.to_numeric(df["Current_Value"],errors="coerce").fillna(0).sum();pnl=value-invested
    return df,{"Positions":len(df),"Value":float(value),"PnL":float(pnl),"Return":float(pnl/invested*100) if invested else 0.,"ActionCounts":df["Decision"].value_counts().to_dict(),"Available":True,"PredictionDate":prediction_date,"DataSource":"CANONICAL_GITHUB_OHLCV"}
