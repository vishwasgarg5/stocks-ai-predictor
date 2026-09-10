"""Stage 28 portfolio decision engine.

Portfolio calculations are fail-closed and consume canonical GitHub OHLCV.
Portfolio AI reuses the canonical morning prediction/lineage first and only
trains a holding-specific fallback when no canonical prediction exists.
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
NAME_TO_TICKER={"RELIANCE INDUSTRIES":"RELIANCE.NS","RELIANCE":"RELIANCE.NS","VEDANTA":"VEDL.NS","VEDANTA IRON & STEEL":"VEDL.NS","YES BANK":"YESBANK.NS","IRFC":"IRFC.NS","NTPC":"NTPC.NS","TATA POWER":"TATAPOWER.NS","WIPRO":"WIPRO.NS","PALASH SECURITIES":"PALASHSECU.NS","OLA ELECTRIC MOBILITY":"OLAELEC.NS","STAR CEMENT":"STARCEMENT.NS","SJVN":"SJVN.NS","RELIANCE POWER":"RPOWER.NS","IRCTC":"IRCTC.NS","SEPC":"SEPC.NS","INDIAN RENEWABLE ENERGY":"IREDA.NS","IREDA":"IREDA.NS"}
HORIZONS=(1,3,5,7,10,20,60,90,180,365)
OUTPUT_COLUMNS=["Stock","Ticker","Quantity","Average_Price","Current_Price","Invested_Value","Current_Value","PnL","Current_PnL_INR","Return_Pct","AI_Target","AI_Confidence","AI_Direction","Decision","Sell_Window","Profit_Target_Price","Sell_Target_Price","Recommended_Qty","New_Average_Price","Projected_Return_At_AI_Target","Recovery_Gap_Pct","Recovery_Probability","Recovery_Horizon_Days","Profit_Target_Probability","Sell_Reason","PredictionDate","PriceSource"]+[f"Horizon_{h}D" for h in HORIZONS]

def _ticker(value):
    raw=str(value).strip();key=re.sub(r"\s+"," ",raw.upper())
    if key in NAME_TO_TICKER:return NAME_TO_TICKER[key]
    return raw.upper() if raw.upper().endswith(".NS") else re.sub(r"[^A-Z0-9]", "", raw.upper())+".NS"

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

def _sell_plan(row,current_price,average_price,prediction_date):
    """Build a deterministic sell target from the first reliable horizon.

    A forecast must reach the configured minimum profit target before a dated
    sell plan is emitted.  Low-confidence forecasts may still establish a
    timing window, but their sell profit is capped at the minimum target.
    """
    try: avg=float(average_price)
    except Exception: avg=np.nan
    if not np.isfinite(avg) or avg<=0:
        return 10.0, np.nan, "-", "-", "WAIT"
    try: confidence=float(row.get("AI_Confidence",0) or 0)
    except Exception: confidence=0.0
    candidates=[]
    for horizon in HORIZONS:
        value=_num(row,f"Horizon_{horizon}D")
        if np.isfinite(value) and value>=TARGET_PROFIT_PCT:
            candidates.append((horizon,value))
    if not candidates:
        return TARGET_PROFIT_PCT, avg*(1+TARGET_PROFIT_PCT/100.0), "-", "-", "WAIT"
    horizon,forecast=candidates[0]
    planned_profit=float(forecast) if confidence>=MIN_AI_CONFIDENCE else TARGET_PROFIT_PCT
    planned_profit=max(TARGET_PROFIT_PCT,planned_profit)
    price=avg*(1+planned_profit/100.0)
    window=f"{horizon}D"
    date=_next_trading_date(prediction_date,horizon) if prediction_date else "-"
    return planned_profit,price,window,date,"TARGET_DATE"

def _num(row,name,default=np.nan):
    try:
        v=float(row.get(name,default));return v if np.isfinite(v) else default
    except Exception:return default

def load_portfolio():
    """Load holdings using tolerant column aliases; never discard a valid row merely because average cost is absent."""
    cols=["Stock","Ticker","Quantity","Average_Price","Reported_PnL","Reported_Return"]
    if not PORTFOLIO_FILE.exists():return pd.DataFrame(columns=cols)
    try:src=pd.read_csv(PORTFOLIO_FILE)
    except Exception:return pd.DataFrame(columns=cols)
    normalized={re.sub(r"[^a-z0-9]","",str(c).lower()):c for c in src.columns}
    def pick(*names):
        for name in names:
            c=normalized.get(re.sub(r"[^a-z0-9]","",name.lower()))
            if c:return c
        return None
    source=pick("Symbol","Stock","Ticker","Name");qty_col=pick("Quantity","Qty","Shares");avg_col=pick("Average_Price","Average Price","Avg Price","Avg_Price","Cost Price")
    pnl_col=pick("Current_PnL_INR","Current PnL INR","PnL","Profit Loss","Profit_Loss");ret_col=pick("Return_Percent","Return Percent","Return_Pct","Return")
    if source is None or qty_col is None:return pd.DataFrame(columns=cols)
    out=pd.DataFrame();out["Stock"]=src[source].astype(str).str.strip();out["Quantity"]=pd.to_numeric(src[qty_col],errors="coerce").fillna(0);out["Average_Price"]=pd.to_numeric(src[avg_col],errors="coerce") if avg_col else np.nan;out["Reported_PnL"]=pd.to_numeric(src[pnl_col],errors="coerce") if pnl_col else np.nan;out["Reported_Return"]=pd.to_numeric(src[ret_col],errors="coerce") if ret_col else np.nan;out["Ticker"]=out["Stock"].map(_ticker)
    out=out[(out["Quantity"]>0)&out["Ticker"].ne("")].copy();out["Stock"]=out["Ticker"].map(_symbol)
    return out.reset_index(drop=True)

def _effective_cutoff(requested=None):
    if requested is not None:return pd.Timestamp(requested).date()
    d=pd.Timestamp(today_ist()).date()
    while not _is_nse_trading_day(d):d-=timedelta(days=1)
    return d

def _cached_history(ticker,cutoff_date=None):
    cutoff=_effective_cutoff(cutoff_date);d,snap=get_snapshot(_symbol(ticker),cutoff_date=cutoff,min_rows=60,allow_download=False)
    if snap.status not in {"OK","STALE"} or d.empty:return pd.DataFrame()
    return d

def _latest_price(ticker,cutoff_date=None):
    d=_cached_history(ticker,cutoff_date)
    if d.empty or "Close" not in d:return None,"UNAVAILABLE"
    s=pd.to_numeric(d["Close"],errors="coerce").dropna();return (float(s.iloc[-1]),"GITHUB_OHLCV") if not s.empty else (None,"UNAVAILABLE")

def _latest_predictions():
    frames=[]
    for p in PREDICTIONS_DIR.glob("predictions_*.csv"):
        try:
            d=pd.read_csv(p)
            if d.empty or "Symbol" not in d.columns:continue
            d["_source_file"]=p.name;frames.append(d)
        except Exception:continue
    if not frames:return pd.DataFrame(),None
    allp=pd.concat(frames,ignore_index=True);date_cols=[c for c in ("PredictionDate","Run_Date","Cutoff_Date") if c in allp.columns]
    if date_cols:
        dc=date_cols[0];allp["_lineage_date"]=pd.to_datetime(allp[dc],errors="coerce");valid=allp["_lineage_date"].dropna()
        if not valid.empty:allp=allp[allp["_lineage_date"]==valid.max()]
    if len(allp):
        allp["_symbol_norm"]=allp["Symbol"].map(_symbol);allp=allp.sort_values([c for c in ("Prediction_ID","Symbol") if c in allp.columns],kind="stable").drop_duplicates("_symbol_norm",keep="last")
    latest=None
    if "_lineage_date" in allp.columns and allp["_lineage_date"].notna().any():latest=str(allp["_lineage_date"].max().date())
    return allp,latest

def _forecast_return(row):return [(h,_num(row,f"Horizon_{h}D")) for h in HORIZONS if np.isfinite(_num(row,f"Horizon_{h}D"))]

def _decision(current,avg,target,confidence,forecasts):
    try:c=float(current) if current is not None else np.nan
    except Exception:c=np.nan
    try:a=float(avg) if avg is not None else np.nan
    except Exception:a=np.nan
    try:t=float(target) if target is not None else np.nan
    except Exception:t=np.nan
    try:conf=float(confidence) if confidence is not None else 0.0
    except Exception:conf=0.0
    if not np.isfinite(c) or not np.isfinite(a) or a<=0:return "WAIT","NO PRICE / COST DATA"
    if not np.isfinite(t):return "HOLD","AI PREDICTION UNAVAILABLE"
    profit_target=a*(1+TARGET_PROFIT_PCT/100.0)
    if c>=profit_target:return "SELL","10% profit target reached"
    if t>=profit_target:
        strong=sum(1 for _,v in (forecasts or []) if np.isfinite(v) and v>=TARGET_PROFIT_PCT)
        return ("HOLD","MULTI_HORIZON_CONFIRMED recovery") if strong>=2 and conf>=MIN_AI_CONFIDENCE else ("HOLD","AI recovery target remains above cost")
    if t<c*(1-SELL_RISK_GAP_PCT/100.0):return "REDUCE","AI target materially below current price"
    return "HOLD","AI recovery not yet confirmed"

def _portfolio_ai(portfolio,cutoff_date=None,variant="A"):
    if portfolio.empty:return pd.DataFrame()
    canonical,_=_latest_predictions();canonical_by={_symbol(r.Symbol):r for _,r in canonical.iterrows()} if not canonical.empty else {};missing=[];rows=[]
    for _,r in portfolio.iterrows():
        symbol=_symbol(r["Ticker"]);cp=canonical_by.get(symbol)
        if cp is not None:
            item=cp.to_dict();item["Symbol"]=symbol;item["PredictionDate"]=str(item.get("PredictionDate") or _effective_cutoff(cutoff_date));item["PredictionSource"]="CANONICAL_MORNING";rows.append(item)
        else:missing.append(r)
    if not missing:return pd.DataFrame(rows)
    try:
        from .prediction import train_stock_bundle,predict_stock,add_multihorizon_predictions
        from .multihorizon import train_horizon_models
    except Exception:return pd.DataFrame(rows)
    cutoff=_effective_cutoff(cutoff_date)
    for _,r in pd.DataFrame(missing).iterrows():
        ticker=str(r["Ticker"]);symbol=_symbol(ticker);d=_cached_history(ticker,cutoff)
        if d.empty or "Close" not in d or len(d)<150:continue
        try:
            effective=pd.Timestamp(cutoff);d=d[d.index<=effective].dropna(subset=["Open","High","Low","Close","Volume"])
            if len(d)<150:continue
            bundle=train_stock_bundle(d,symbol,effective,variant,train_horizons=False);pred=predict_stock(d,bundle,effective);item={"Symbol":symbol,**pred,"PredictionDate":str(effective.date()),"PredictionSource":"PORTFOLIO_FALLBACK"}
            try:
                hb=train_horizon_models(d,effective);h=add_multihorizon_predictions(d,{"horizons":hb},effective)
                for _,hr in h.iterrows():item[f"Horizon_{int(hr.HorizonDays)}D"]=float(hr.Expected_Return) if pd.notna(hr.Expected_Return) else np.nan
            except Exception:pass
            rows.append(item)
        except Exception as exc:print(f"Portfolio fallback failed for {symbol}: {exc}")
    return pd.DataFrame(rows)

def _plan_row(row,pred,prediction_date):
    ticker=row["Ticker"];current,source=_latest_price(ticker,prediction_date);qty=float(row.get("Quantity",0) or 0);avg=row.get("Average_Price",np.nan);reported_pnl=row.get("Reported_PnL",np.nan);reported_return=row.get("Reported_Return",np.nan)
    if pred is not None and np.isfinite(_num(pred,"Current_Price")):current=_num(pred,"Current_Price");source="CANONICAL_PREDICTION"
    if pd.isna(avg) and current is not None and pd.notna(reported_pnl) and qty>0:avg=current-float(reported_pnl)/qty
    if pd.isna(avg) and current is not None and pd.notna(reported_return) and float(reported_return)!=-100:avg=current/(1+float(reported_return)/100)
    target=_num(pred,"Pred_Close") if pred is not None else np.nan;confidence=_num(pred,"CalibratedConfidence",_num(pred,"Confidence",0)) if pred is not None else 0;direction=str(pred.get("Direction","-") if pred is not None else "-");forecasts=_forecast_return(pred) if pred is not None else []
    invested=qty*float(avg) if pd.notna(avg) else 0.;value=qty*current if current is not None else 0.;pnl=value-invested;ret=pnl/invested*100 if invested else np.nan;profit_target=float(avg)*(1+TARGET_PROFIT_PCT/100) if pd.notna(avg) else np.nan;recovery=((float(avg)-current)/float(avg)*100) if current is not None and pd.notna(avg) and float(avg) else np.nan
    decision,reason=_decision(current,avg,target,confidence,forecasts);required_recovery=((float(avg)/float(current)-1)*100) if current is not None and np.isfinite(current) and current>0 and pd.notna(avg) else np.nan;valid_forecasts=[(h,float(v)) for h,v in forecasts if np.isfinite(v)];recovery_hits=[(h,v) for h,v in valid_forecasts if np.isfinite(required_recovery) and v>=required_recovery];recovery_horizon=min((h for h,_ in recovery_hits),default=np.nan);max_forecast=max((v for _,v in valid_forecasts),default=np.nan);recovery_probability=float(np.clip(50+50*(max_forecast/required_recovery),0,95)) if np.isfinite(required_recovery) and required_recovery>0 and np.isfinite(max_forecast) else (95.0 if current>=avg else 0.0);profit_required=((float(avg)*(1+TARGET_PROFIT_PCT/100))/float(current)-1)*100 if current is not None and np.isfinite(current) and current>0 and pd.notna(avg) else np.nan;profit_probability=float(np.clip(50+50*(max_forecast/profit_required),0,95)) if np.isfinite(profit_required) and profit_required>0 and np.isfinite(max_forecast) else (95.0 if current>=profit_target else 0.0);projected=(target/float(avg)-1)*100 if np.isfinite(target) and pd.notna(avg) and float(avg) else np.nan
    out={"Stock":_symbol(ticker),"Ticker":ticker,"Quantity":int(qty),"Average_Price":avg,"Current_Price":current,"Invested_Value":invested,"Current_Value":value,"PnL":pnl,"Current_PnL_INR":pnl,"Return_Pct":ret,"AI_Target":target,"AI_Confidence":confidence,"AI_Direction":direction,"Decision":decision,"Sell_Window":"NOW" if decision in {"SELL","REDUCE"} else ("WATCH" if np.isfinite(target) else "NO AI DATA"),"Profit_Target_Price":profit_target,"Sell_Target_Price":target,"Recommended_Qty":0,"New_Average_Price":avg,"Projected_Return_At_AI_Target":projected,"Recovery_Gap_Pct":recovery,"Recovery_Probability":recovery_probability,"Recovery_Horizon_Days":recovery_horizon,"Profit_Target_Probability":profit_probability,"Sell_Reason":reason,"PredictionDate":prediction_date or "-","PriceSource":source}
    for h in HORIZONS:out[f"Horizon_{h}D"]=_num(pred,f"Horizon_{h}D") if pred is not None else np.nan
    return out

def portfolio_snapshot(cutoff_date=None,variant="A"):
    portfolio=load_portfolio();empty={"Positions":0,"Value":0.,"PnL":0.,"Return":0.,"ActionCounts":{},"Available":bool(not portfolio.empty)}
    if portfolio.empty:return pd.DataFrame(columns=OUTPUT_COLUMNS),empty
    ai=_portfolio_ai(portfolio,cutoff_date,variant);ai_by={_symbol(r.Symbol):r for _,r in ai.iterrows()} if not ai.empty else {};_,date=_latest_predictions();prediction_date=str(cutoff_date or date or "-")
    rows=[_plan_row(r,ai_by.get(_symbol(r["Ticker"])),prediction_date) for _,r in portfolio.iterrows()];df=pd.DataFrame(rows)
    for c in OUTPUT_COLUMNS:
        if c not in df.columns:df[c]=np.nan
    df=df[OUTPUT_COLUMNS];df["PnL"]=pd.to_numeric(df["PnL"],errors="coerce").fillna(0);df["Current_PnL_INR"]=df["PnL"];invested=pd.to_numeric(df["Invested_Value"],errors="coerce").fillna(0).sum();value=pd.to_numeric(df["Current_Value"],errors="coerce").fillna(0).sum();pnl=value-invested
    return df,{"Positions":len(df),"Value":float(value),"PnL":float(pnl),"Return":float(pnl/invested*100) if invested else 0.,"ActionCounts":df["Decision"].value_counts().to_dict(),"Available":True,"PredictionDate":prediction_date,"DataSource":"CANONICAL_GITHUB_OHLCV","PredictionBinding":"CANONICAL_MORNING_FIRST"}
