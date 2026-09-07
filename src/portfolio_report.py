"""Stage 10.5 Portfolio Manager: position-aware targets, averaging and dynamic exit timing."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import yfinance as yf
from .config import PREDICTIONS_DIR, HISTORY_PERIOD
from .prediction import train_stock_bundle, predict_stock, add_multihorizon_predictions
from .market_data import download_symbol
from .portfolio_long_horizon import train_portfolio_long_horizon_models, predict_portfolio_long_horizons, PORTFOLIO_LONG_HORIZONS
ROOT=Path(__file__).resolve().parents[1];PORTFOLIO_FILE=ROOT/"portfolio_manager"/"data"/"my_portfolio.csv"
NAME_TO_TICKER={"RELIANCE INDUSTRIES":"RELIANCE.NS","RELIANCE":"RELIANCE.NS","VEDANTA IRON & STEEL":"VEDL.NS","VEDANTA":"VEDL.NS","YES BANK":"YESBANK.NS","IRFC":"IRFC.NS","NTPC":"NTPC.NS","TATA POWER":"TATAPOWER.NS","WIPRO":"WIPRO.NS","PALASH SECURITIES":"PALASHSECU.NS","OLA ELECTRIC MOBILITY":"OLAELEC.NS","STAR CEMENT":"STARCEMENT.NS","SJVN":"SJVN.NS","RELIANCE POWER":"RPOWER.NS","IRCTC":"IRCTC.NS","SEPC":"SEPC.NS","INDIAN RENEWABLE ENERGY":"IREDA.NS","IREDA":"IREDA.NS"}
TARGET_PROFIT_PCT=10.0;MAX_AVERAGING_CAPITAL_PCT=25.0;MIN_AI_CONFIDENCE=60.0;SELL_RISK_GAP_PCT=3.0;SHORT_HORIZONS=(1,3,5,7,20);HORIZONS=SHORT_HORIZONS+PORTFOLIO_LONG_HORIZONS
NSE_HOLIDAYS={"2026-01-15","2026-01-26","2026-02-19","2026-03-03","2026-03-19","2026-03-26","2026-03-31","2026-04-01","2026-04-03","2026-04-14","2026-05-01","2026-05-28","2026-06-26","2026-08-26","2026-09-14","2026-10-02","2026-10-20","2026-11-08","2026-11-10","2026-11-24","2026-12-25"}
def _ticker(value):
    raw=str(value).strip();key=re.sub(r"\s+"," ",raw.upper())
    if key in NAME_TO_TICKER:return NAME_TO_TICKER[key]
    if raw.upper().endswith(".NS"):return raw.upper()
    return raw.upper().replace(" & ","").replace(" ","")+".NS"
def _canonical_ticker(value):
    raw=str(value).strip().upper()
    if not raw or raw in {"NAN","NONE","<NA>"}:return ""
    return raw if raw.endswith(".NS") else raw+".NS"
def _symbol(ticker):return str(ticker).upper().removesuffix(".NS")
def load_portfolio():
    if not PORTFOLIO_FILE.exists():return pd.DataFrame()
    df=pd.read_csv(PORTFOLIO_FILE);source_col="Symbol" if "Symbol" in df.columns else "Stock"
    if {source_col,"Quantity","Average_Price"}.issubset(df.columns):out=df[[source_col,"Quantity","Average_Price"]].copy().rename(columns={source_col:"Stock"});out["Reported_PnL"]=pd.NA;out["Reported_Return"]=pd.NA
    elif {source_col,"Quantity","Current_PnL_INR","Return_Percent"}.issubset(df.columns):out=df[[source_col,"Quantity","Current_PnL_INR","Return_Percent"]].copy().rename(columns={source_col:"Stock","Current_PnL_INR":"Reported_PnL","Return_Percent":"Reported_Return"});out["Average_Price"]=pd.NA
    else:return pd.DataFrame()
    for c in ["Quantity","Average_Price","Reported_PnL","Reported_Return"]:out[c]=pd.to_numeric(out[c],errors="coerce")
    out["Quantity"]=out["Quantity"].fillna(0);out["Ticker"]=out["Stock"].map(_ticker);out["Stock"]=out["Ticker"].map(_symbol);return out
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
def _portfolio_ai_predictions(df,pred_date):
    if df.empty or not pred_date:return df
    need=df.index[df.get("Horizon_365D",pd.Series(index=df.index,dtype=float)).isna()].tolist()
    for i in need:
        ticker=df.at[i,"Ticker"];symbol=_symbol(ticker)
        try:
            history=download_symbol(ticker,HISTORY_PERIOD)
            if history is None or history.empty:continue
            if pd.isna(df.at[i,"AI_Target"]):
                bundle=train_stock_bundle(history,symbol,pred_date,"A",train_horizons=True);result=predict_stock(history,bundle,pred_date);horizons=add_multihorizon_predictions(history,bundle,pred_date)
                df.at[i,"AI_Target"]=result.get("Pred_Close");df.at[i,"AI_Open"]=result.get("Pred_Open");df.at[i,"AI_High"]=result.get("Pred_High");df.at[i,"AI_Low"]=result.get("Pred_Low");df.at[i,"AI_Confidence"]=result.get("Confidence");df.at[i,"AI_Direction"]=result.get("Direction")
                for _,hr in horizons.iterrows():df.at[i,f"Horizon_{int(hr['HorizonDays'])}D"]=float(hr["Expected_Return"])
            long_history=download_symbol(ticker,"5y")
            if long_history is None or long_history.empty:long_history=history
            long_bundle=train_portfolio_long_horizon_models(long_history,pred_date);long_forecasts=predict_portfolio_long_horizons(long_history,long_bundle,pred_date)
            for _,hr in long_forecasts.iterrows():df.at[i,f"Horizon_{int(hr['HorizonDays'])}D"]=float(hr["Expected_Return"])
        except Exception as exc:print(f"Portfolio extended forecast warning {symbol}: {exc}")
    return df
def _attach_predictions(df):
    pred,pred_date=_latest_predictions()
    for h in HORIZONS:df[f"Horizon_{h}D"]=np.nan
    if pred.empty:
        df["AI_Target"]=np.nan;return _portfolio_ai_predictions(df,pred_date),pred_date
    keep=[c for c in ["Symbol","Pred_Close","Pred_Open","Pred_High","Pred_Low","Confidence","CalibratedConfidence","Direction","FinalDecisionScore","Action",*[f"Horizon_{h}D" for h in SHORT_HORIZONS]] if c in pred.columns]
    p=pred[keep].copy();p["Ticker"]=p["Symbol"].map(_canonical_ticker);p=p.drop(columns=["Symbol"]).rename(columns={"Pred_Close":"AI_Target","Pred_Open":"AI_Open","Pred_High":"AI_High","Pred_Low":"AI_Low","Action":"AI_Action"}).drop_duplicates(subset=["Ticker"],keep="last")
    for c in ["AI_Target","AI_Open","AI_High","AI_Low",*[f"Horizon_{h}D" for h in SHORT_HORIZONS]]:
        if c in p:p[c]=pd.to_numeric(p[c],errors="coerce")
    merged=df.merge(p,on="Ticker",how="left");return _portfolio_ai_predictions(merged,pred_date),pred_date
def _is_nse_trading_day(date):
    ts=pd.Timestamp(date);return ts.weekday()<5 and str(ts.date()) not in NSE_HOLIDAYS
def _next_trading_date(start_date,sessions):
    if not start_date:return "-"
    d=pd.Timestamp(start_date);count=0
    while count<int(sessions):
        d+=pd.Timedelta(days=1)
        if _is_nse_trading_day(d):count+=1
    return str(d.date())
def _sell_plan(row,current,avg,prediction_date):
    if current is None or pd.isna(current) or pd.isna(avg) or not prediction_date:return TARGET_PROFIT_PCT,None,"NO AI DATA","-","NO AI DATA"
    confidence=float(row.get("AI_Confidence",row.get("CalibratedConfidence",row.get("Confidence",0))) or 0);forecasts=[]
    for h in HORIZONS:
        try:
            value=float(row.get(f"Horizon_{h}D"))
            if np.isfinite(value):forecasts.append((h,value))
        except (TypeError,ValueError):pass
    if not forecasts:return TARGET_PROFIT_PCT,float(avg)*1.10,"10% target not forecast","-","WAIT"
    reliable=[(h,v) for h,v in forecasts if v>TARGET_PROFIT_PCT and confidence>=MIN_AI_CONFIDENCE]
    if reliable:target_h,target_return=max(reliable,key=lambda x:x[1])
    else:
        reached=[(h,v) for h,v in forecasts if v>=TARGET_PROFIT_PCT]
        if not reached:
            max_h=max(h for h,_ in forecasts);return TARGET_PROFIT_PCT,float(avg)*1.10,f">{max_h}D target not reached","-","WAIT"
        target_h,target_return=min(reached,key=lambda x:x[0]);target_return=TARGET_PROFIT_PCT
    target_price=float(avg)*(1+target_return/100);sell_date=_next_trading_date(prediction_date,target_h);return float(target_return),target_price,f"{target_h}D ({sell_date})",sell_date,"TARGET_DATE"
def _average_plan(row):
    qty=float(row["Quantity"] or 0);price=row["Current_Price"];avg=row["Average_Price"];target=row.get("AI_Target")
    if pd.isna(avg) and price is not None and pd.notna(row["Reported_Return"]) and float(row["Reported_Return"])>-100:avg=price/(1+float(row["Reported_Return"])/100);row["Average_Price"]=avg;row["AveragePriceSource"]="ESTIMATED_FROM_RETURN"
    elif pd.isna(avg) and price is not None and qty>0 and pd.notna(row["Reported_PnL"]):avg=price-float(row["Reported_PnL"])/qty;row["Average_Price"]=avg;row["AveragePriceSource"]="ESTIMATED_FROM_PNL"
    elif pd.notna(avg):row["AveragePriceSource"]="CSV"
    else:row["AveragePriceSource"]="UNAVAILABLE"
    row["Profit_Target_Price"]=float(avg)*1.10 if pd.notna(avg) else None
    if price is None or pd.isna(avg) or qty<=0:
        row.update({"Invested_Value":qty*avg if pd.notna(avg) else 0,"Recovery_Gap_Pct":None,"Target_Return_Pct":None,"Recommended_Qty":0,"New_Average_Price":None,"Projected_Return_At_AI_Target":None,"Averaging_Action":"DATA WAIT","Decision":"DATA WAIT","Sell_Window":"NO PRICE","Sell_Reason":"Insufficient portfolio/price data"});return row
    row["Invested_Value"]=qty*float(avg);row["Recovery_Gap_Pct"]=(float(avg)-price)/float(avg)*100 if avg else None
    if pd.isna(target):
        row.update({"Target_Return_Pct":None,"Recommended_Qty":0,"New_Average_Price":float(avg),"Projected_Return_At_AI_Target":None,"Averaging_Action":"DO NOT AVG","Decision":"HOLD","Sell_Window":"NO AI DATA","Sell_Reason":"AI forecast unavailable for this holding"});return row
    target=float(target);row["Target_Return_Pct"]=(target/float(avg)-1)*100;target_return,target_price,sell_window,sell_date,sell_status=_sell_plan(row,price,float(avg),row.get("PredictionDate"));row["Sell_Target_Profit_Pct"]=target_return;row["Sell_Target_Price"]=target_price;row["Sell_Date"]=sell_date;row["Profit_Target_Price"]=target_price;row["Sell_Window"]=f"+{target_return:.1f}% | ₹{target_price:,.2f} | {sell_date}" if sell_date!="-" else f"+{target_return:.1f}% | ₹{target_price:,.2f} | WAIT"
    if price>=target_price:
        row.update({"Recommended_Qty":0,"New_Average_Price":float(avg),"Projected_Return_At_AI_Target":(target/avg-1)*100,"Averaging_Action":"DO NOT AVG","Decision":"SELL","Sell_Window":"NOW","Sell_Date":str(row.get("PredictionDate") or "TODAY"),"Sell_Reason":f"Minimum 10% target reached; AI target {target_return:.1f}%"});return row
    desired_avg=target/1.10
    if desired_avg<=price or target<=price*(1-SELL_RISK_GAP_PCT/100):
        row.update({"Recommended_Qty":0,"New_Average_Price":float(avg),"Projected_Return_At_AI_Target":(target/avg-1)*100,"Averaging_Action":"DO NOT AVG","Decision":"SELL" if target<price*(1-SELL_RISK_GAP_PCT/100) else "HOLD","Sell_Window":"NOW" if target<price*(1-SELL_RISK_GAP_PCT/100) else row["Sell_Window"],"Sell_Reason":"AI target does not support a safe 10% recovery"});return row
    required=qty*(float(avg)-desired_avg)/(desired_avg-price);budget=qty*float(avg)*MAX_AVERAGING_CAPITAL_PCT/100;max_qty=int(budget//price);rec=min(max(0,int(required+0.9999)),max_qty);new_avg=(qty*float(avg)+rec*price)/(qty+rec) if rec>0 else float(avg);projected=(target/new_avg-1)*100
    row["Recommended_Qty"]=rec;row["New_Average_Price"]=new_avg;row["Projected_Return_At_AI_Target"]=projected;row["Max_Averaging_Capital"]=budget
    if rec>0 and projected>=TARGET_PROFIT_PCT and float(row.get("AI_Confidence",0) or 0)>=MIN_AI_CONFIDENCE and row["Recovery_Gap_Pct"]>=5:row["Averaging_Action"]="AVERAGE";row["Decision"]="AVG";row["Sell_Reason"]="AI recovery supports reduced average and profit target"
    elif row["Recovery_Gap_Pct"]<=0:row["Averaging_Action"]="DO NOT AVG";row["Decision"]="HOLD";row["Sell_Reason"]="Position is not below average cost"
    else:row["Averaging_Action"]="DO NOT AVG";row["Decision"]="HOLD";row["Sell_Reason"]="Recovery target not strong enough or confidence is low"
    if sell_status=="WAIT":row["Decision"]="HOLD";row["Sell_Reason"]="10% target is not currently supported by a reliable forecast; reassess after new data"
    return row
def portfolio_snapshot():
    df=load_portfolio()
    if df.empty:return df,{"Positions":0,"Value":0.0,"PnL":0.0,"Return":0.0,"ActionCounts":{}}
    prices={t:_price(t) for t in df["Ticker"].dropna().unique()};df["Current_Price"]=df["Ticker"].map(prices);df,prediction_date=_attach_predictions(df);df["PredictionDate"]=prediction_date;df["AveragePriceSource"]="UNAVAILABLE";df=df.apply(_average_plan,axis=1)
    df["Decision"]=df["Decision"].map(lambda x:{"SELL / PROFIT BOOK":"SELL","SELL / EXIT":"SELL","HOLD / RECOVERY":"HOLD","DATA WAIT":"WAIT"}.get(str(x),str(x).strip().split()[0] if str(x).strip() else "WAIT"))
    df["Current_Value"]=df["Quantity"]*df["Current_Price"].fillna(0);mask=df["Reported_PnL"].notna()&df["AveragePriceSource"].str.startswith("ESTIMATED");df["PnL"]=df["Current_Value"]-df["Invested_Value"];df.loc[mask,"PnL"]=df.loc[mask,"Reported_PnL"];df["Return_Pct"]=df.apply(lambda r:float(r["Reported_Return"]) if str(r["AveragePriceSource"]).startswith("ESTIMATED") and pd.notna(r["Reported_Return"]) else ((r["PnL"]/r["Invested_Value"]*100) if r["Invested_Value"] else None),axis=1)
    total_inv=float(df["Invested_Value"].sum());total_val=float(df["Current_Value"].sum());total_pnl=float(df["PnL"].sum());return df,{"Positions":len(df),"Value":total_val,"PnL":total_pnl,"Return":total_pnl/total_inv*100 if total_inv else 0.0,"ActionCounts":df["Decision"].value_counts().to_dict(),"PredictionDate":prediction_date,"LongHorizons":list(PORTFOLIO_LONG_HORIZONS)}
