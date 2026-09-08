"""Stage 10.5 portfolio manager with independent AI coverage for every holding."""
from pathlib import Path
from datetime import timedelta
import re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];PORTFOLIO_FILE=ROOT/"portfolio_manager"/"data"/"my_portfolio.csv";OHLCV_DIR=ROOT/"data"/"ohlcv";PREDICTIONS_DIR=ROOT/"data"/"stage2"/"predictions"
TARGET_PROFIT_PCT=10.0;MAX_AVERAGING_CAPITAL_PCT=25.0;MIN_AI_CONFIDENCE=60.0;SELL_RISK_GAP_PCT=3.0
NSE_HOLIDAYS={"2026-01-26","2026-03-03","2026-03-26","2026-03-31","2026-04-03","2026-04-14","2026-05-01","2026-05-28","2026-06-26","2026-08-15","2026-08-28","2026-09-14","2026-10-02","2026-10-20","2026-11-09","2026-11-24","2026-12-25"}
NAME_TO_TICKER={"RELIANCE INDUSTRIES":"RELIANCE.NS","RELIANCE":"RELIANCE.NS","VEDANTA":"VEDL.NS","YES BANK":"YESBANK.NS","IRFC":"IRFC.NS","NTPC":"NTPC.NS","TATA POWER":"TATAPOWER.NS","WIPRO":"WIPRO.NS","PALASH SECURITIES":"PALASHSECU.NS","OLA ELECTRIC MOBILITY":"OLAELEC.NS","STAR CEMENT":"STARCEMENT.NS","SJVN":"SJVN.NS","RELIANCE POWER":"RPOWER.NS","IRCTC":"IRCTC.NS","SEPC":"SEPC.NS","INDIAN RENEWABLE ENERGY":"IREDA.NS","IREDA":"IREDA.NS","VEDANTA IRON & STEEL":"VEDL.NS"}
OUTPUT_COLUMNS=["Stock","Ticker","Quantity","Average_Price","Current_Price","Invested_Value","Current_Value","PnL","Current_PnL_INR","Return_Pct","AI_Target","AI_Confidence","AI_Direction","Decision","Sell_Window","Profit_Target_Price","Sell_Target_Price","Recommended_Qty","New_Average_Price","Projected_Return_At_AI_Target","Recovery_Gap_Pct","Sell_Reason","PredictionDate","PriceSource"]

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

def _ticker(value):
    raw=str(value).strip();key=re.sub(r"\s+"," ",raw.upper())
    if key in NAME_TO_TICKER:return NAME_TO_TICKER[key]
    return raw.upper() if raw.upper().endswith(".NS") else raw.upper().replace(" & ","").replace(" ","")+".NS"

def _symbol(ticker):return str(ticker).upper().removesuffix(".NS")

def load_portfolio():
    cols=["Stock","Ticker","Quantity","Average_Price","Reported_PnL","Reported_Return"]
    if not PORTFOLIO_FILE.exists():return pd.DataFrame(columns=cols)
    try:src=pd.read_csv(PORTFOLIO_FILE)
    except Exception:return pd.DataFrame(columns=cols)
    source="Symbol" if "Symbol" in src.columns else "Stock" if "Stock" in src.columns else None
    if source is None or "Quantity" not in src.columns:return pd.DataFrame(columns=cols)
    out=pd.DataFrame();out["Stock"]=src[source].astype(str).str.strip();out["Quantity"]=pd.to_numeric(src["Quantity"],errors="coerce").fillna(0);out["Average_Price"]=pd.to_numeric(src.get("Average_Price",np.nan),errors="coerce");out["Reported_PnL"]=pd.to_numeric(src.get("Current_PnL_INR",np.nan),errors="coerce");out["Reported_Return"]=pd.to_numeric(src.get("Return_Percent",np.nan),errors="coerce");out["Ticker"]=out["Stock"].map(_ticker);out["Stock"]=out["Ticker"].map(_symbol);return out

def _cached_history(ticker):
    path=OHLCV_DIR/f"{_symbol(ticker)}.csv"
    try:
        if path.exists():
            d=pd.read_csv(path);d["Date"]=pd.to_datetime(d["Date"],errors="coerce") if "Date" in d.columns else pd.to_datetime(d.index,errors="coerce");d=d.set_index("Date") if "Date" in d.columns else d;return d.sort_index()
    except Exception:pass
    try:
        import yfinance as yf
        d=yf.download(ticker,period="2y",auto_adjust=False,progress=False,threads=False)
        if isinstance(d,pd.DataFrame) and not d.empty:
            if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
            d.index=pd.to_datetime(d.index);return d.sort_index()
    except Exception as exc:print(f"Portfolio price fallback {ticker}: {exc}")
    return pd.DataFrame()

def _latest_price(ticker):
    d=_cached_history(ticker)
    if d.empty or "Close" not in d:return None,"UNAVAILABLE"
    s=pd.to_numeric(d["Close"],errors="coerce").dropna();return (float(s.iloc[-1]),"GITHUB_OHLCV") if not s.empty else (None,"UNAVAILABLE")

def _latest_predictions():
    files=sorted(PREDICTIONS_DIR.glob("predictions_*.csv"),key=lambda p:p.stat().st_mtime)
    for p in reversed(files):
        try:
            d=pd.read_csv(p)
            if not d.empty and "Symbol" in d.columns:d["Ticker"]=d["Symbol"].astype(str).map(_ticker);return d,p.stem.replace("predictions_","")
        except Exception:pass
    return pd.DataFrame(),None

def _attach_predictions(base):
    pred,date=_latest_predictions();out=base.copy()
    if pred.empty:return out,date
    pred=pred.drop_duplicates("Ticker",keep="last")
    keep=[c for c in ["Ticker","Pred_Close","Pred_Open","Pred_High","Pred_Low","AI_High","AI_Low","Confidence","CalibratedConfidence","Direction","Action","Horizon_1D","Horizon_3D","Horizon_5D","Horizon_7D","Horizon_10D","Horizon_20D"] if c in pred.columns]
    out=out.merge(pred[keep],on="Ticker",how="left")
    if "AI_High" not in out.columns and "Pred_High" in out.columns:out["AI_High"]=out["Pred_High"]
    return out,date

def _average_plan(row):
    avg=float(row.get("Average_Price",np.nan));target=float(row.get("AI_Target",row.get("Pred_Close",np.nan)));row["Profit_Target_Price"]=avg*(1+TARGET_PROFIT_PCT/100) if np.isfinite(avg) else np.nan;row["Projected_Return_At_AI_Target"]=(target/avg-1)*100 if np.isfinite(avg) and avg else np.nan;return row

def _num(row,name,default=np.nan):
    try:v=float(row.get(name,default));return v if np.isfinite(v) else default
    except Exception:return default

def _forecast_return(row):return [(h,v) for h in (1,3,5,7,10,20) if np.isfinite(v:=_num(row,f"Horizon_{h}D"))]

def _decision(current,avg,target,confidence,forecasts):
    if current is None or not np.isfinite(current) or not np.isfinite(avg) or avg<=0:return "WAIT","NO PRICE / COST DATA"
    if not np.isfinite(target):return "HOLD","AI TARGET UNAVAILABLE"
    tolerance=max(1e-8,abs(avg)*1e-8)
    if current+tolerance>=avg*(1+TARGET_PROFIT_PCT/100):return "SELL","10% profit target already reached"
    if current+tolerance>=target and target>avg:return "SELL","AI target reached"
    if target<current*(1-SELL_RISK_GAP_PCT/100):return "SELL","AI target is materially below current price"
    positive=[v for _,v in forecasts if np.isfinite(v) and v>=TARGET_PROFIT_PCT]
    if (avg-current)/avg*100>=5 and positive and confidence>=MIN_AI_CONFIDENCE:return "AVG","Multi-horizon recovery supports limited averaging"
    if target/avg*100-100>=TARGET_PROFIT_PCT and current<avg:return "HOLD","Recovery target remains above cost"
    return "HOLD","No sufficiently strong recovery confirmation"

def _portfolio_ai(portfolio,cutoff_date=None,variant="A"):
    if portfolio.empty:return pd.DataFrame()
    from .prediction import train_stock_bundle,predict_stock,add_multihorizon_predictions
    from .multihorizon import train_horizon_models
    rows=[];pdate=str(cutoff_date or "-")
    for _,r in portfolio.iterrows():
        ticker=str(r["Ticker"]);symbol=_symbol(ticker);d=_cached_history(ticker)
        if d.empty or "Close" not in d:continue
        try:
            if cutoff_date is not None:d=d[d.index.date<=pd.Timestamp(cutoff_date).date()]
            d=d.dropna(subset=["Open","High","Low","Close","Volume"])
            if len(d)<150:continue
            bundle=train_stock_bundle(d,symbol,pdate,variant,train_horizons=False);pred=predict_stock(d,bundle,pdate);item={"Symbol":symbol,**pred,"PredictionDate":pdate}
            try:
                hb=train_horizon_models(d,pdate);h=add_multihorizon_predictions(d,{"horizons":hb},pdate)
                for _,hr in h.iterrows():item[f"Horizon_{int(hr.HorizonDays)}D"]=float(hr.Expected_Return)
            except Exception as exc:print(f"{symbol}: portfolio horizons skipped: {exc}")
            rows.append(item)
        except Exception as exc:print(f"{symbol}: portfolio AI skipped: {exc}")
    return pd.DataFrame(rows)

def _plan_row(row,pred,prediction_date):
    ticker=row["Ticker"];current,source=_latest_price(ticker);qty=float(row.get("Quantity",0) or 0);avg=row.get("Average_Price",np.nan);reported_pnl=row.get("Reported_PnL",np.nan);reported_return=row.get("Reported_Return",np.nan)
    if pred is not None and np.isfinite(_num(pred,"Current_Price")):current=_num(pred,"Current_Price");source="PORTFOLIO_AI"
    if pd.isna(avg) and current is not None and pd.notna(reported_return):avg=current/(1+float(reported_return)/100)
    if pd.isna(avg) and current is not None and pd.notna(reported_pnl) and qty>0:avg=current-float(reported_pnl)/qty
    target=_num(pred,"Pred_Close") if pred is not None else np.nan;confidence=_num(pred,"Confidence",0) if pred is not None else 0;direction=str(pred.get("Direction","-") if pred is not None else "-");forecasts=_forecast_return(pred) if pred is not None else []
    invested=qty*float(avg) if pd.notna(avg) else 0.0;value=qty*current if current is not None else 0.0;pnl=value-invested;ret=pnl/invested*100 if invested else np.nan;profit_target=float(avg)*(1+TARGET_PROFIT_PCT/100) if pd.notna(avg) else np.nan;recovery=((float(avg)-current)/float(avg)*100) if current is not None and pd.notna(avg) and float(avg) else np.nan
    decision,reason=_decision(current,float(avg) if pd.notna(avg) else np.nan,target,confidence,forecasts);newavg=float(avg) if pd.notna(avg) else np.nan;projected=(target/newavg-1)*100 if np.isfinite(target) and np.isfinite(newavg) and newavg else np.nan
    return {"Stock":_symbol(ticker),"Ticker":ticker,"Quantity":int(qty),"Average_Price":avg,"Current_Price":current,"Invested_Value":invested,"Current_Value":value,"PnL":pnl,"Current_PnL_INR":pnl,"Return_Pct":ret,"AI_Target":target,"AI_Confidence":confidence,"AI_Direction":direction,"Decision":decision,"Sell_Window":"NOW" if decision=="SELL" else ("MONITOR" if np.isfinite(target) else "NO AI DATA"),"Profit_Target_Price":target if np.isfinite(target) else profit_target,"Sell_Target_Price":target,"Recommended_Qty":0,"New_Average_Price":newavg,"Projected_Return_At_AI_Target":projected,"Recovery_Gap_Pct":recovery,"Sell_Reason":reason,"PredictionDate":prediction_date or "-","PriceSource":source}

def portfolio_snapshot(cutoff_date=None,variant="A"):
    portfolio=load_portfolio();empty={"Positions":0,"Value":0.0,"PnL":0.0,"Return":0.0,"ActionCounts":{},"Available":bool(not portfolio.empty)}
    if portfolio.empty:return pd.DataFrame(columns=OUTPUT_COLUMNS),empty
    ai=_portfolio_ai(portfolio,cutoff_date,variant);ai_by={str(r.Symbol)+".NS":r for _,r in ai.iterrows()} if not ai.empty else {};_,latest_date=_latest_predictions();prediction_date=str(cutoff_date or latest_date or "-");rows=[_plan_row(r,ai_by.get(str(r.Ticker)),prediction_date) for _,r in portfolio.iterrows()];df=pd.DataFrame(rows)
    for c in OUTPUT_COLUMNS:
        if c not in df.columns:df[c]=np.nan
    df=df[OUTPUT_COLUMNS];df["PnL"]=pd.to_numeric(df["PnL"],errors="coerce").fillna(0);df["Current_PnL_INR"]=df["PnL"];invested=pd.to_numeric(df["Invested_Value"],errors="coerce").fillna(0).sum();value=pd.to_numeric(df["Current_Value"],errors="coerce").fillna(0).sum();pnl=value-invested
    return df,{"Positions":len(df),"Value":float(value),"PnL":float(pnl),"Return":float(pnl/invested*100) if invested else 0.0,"ActionCounts":df["Decision"].value_counts().to_dict(),"Available":True}
