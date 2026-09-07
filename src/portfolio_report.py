"""Stage 10.5 Portfolio Manager: use stored AI predictions and same AI engine for every holding."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import yfinance as yf
from .config import PREDICTIONS_DIR, HISTORY_PERIOD
from .prediction import train_stock_bundle, predict_stock, add_multihorizon_predictions
from .market_data import download_symbol
ROOT=Path(__file__).resolve().parents[1];PORTFOLIO_FILE=ROOT/"portfolio_manager"/"data"/"my_portfolio.csv"
NAME_TO_TICKER={"RELIANCE INDUSTRIES":"RELIANCE.NS","RELIANCE":"RELIANCE.NS","VEDANTA IRON & STEEL":"VEDL.NS","VEDANTA":"VEDL.NS","YES BANK":"YESBANK.NS","IRFC":"IRFC.NS","NTPC":"NTPC.NS","TATA POWER":"TATAPOWER.NS","WIPRO":"WIPRO.NS","PALASH SECURITIES":"PALASHSECU.NS","OLA ELECTRIC MOBILITY":"OLAELEC.NS","STAR CEMENT":"STARCEMENT.NS","SJVN":"SJVN.NS","RELIANCE POWER":"RPOWER.NS","IRCTC":"IRCTC.NS","SEPC":"SEPC.NS","INDIAN RENEWABLE ENERGY":"IREDA.NS","IREDA":"IREDA.NS"}
TARGET_PROFIT_PCT=10.0;MAX_AVERAGING_CAPITAL_PCT=25.0;MIN_AI_CONFIDENCE=60.0;SELL_RISK_GAP_PCT=3.0;HORIZONS=(1,3,5,7,20)
def _ticker(value):
    raw=str(value).strip();key=re.sub(r"\s+"," ",raw.upper())
    if key in NAME_TO_TICKER:return NAME_TO_TICKER[key]
    if raw.upper().endswith(".NS"):return raw.upper()
    return raw.upper().replace(" & ","").replace(" ","")+".NS"
def _canonical_ticker(value):
    raw=str(value).strip().upper()
    if not raw or raw in {"NAN","NONE","<NA>"}:return ""
    if raw.endswith(".NS"):return raw
    return raw+".NS"
def _symbol(ticker):return str(ticker).upper().removesuffix(".NS")
def load_portfolio():
    if not PORTFOLIO_FILE.exists():return pd.DataFrame()
    df=pd.read_csv(PORTFOLIO_FILE)
    source_col="Symbol" if "Symbol" in df.columns else "Stock"
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
    if df.empty:return df
    cutoff=pred_date
    if not cutoff:return df
    missing=df.loc[df["AI_Target"].isna(),"Ticker"].dropna().unique().tolist()
    for ticker in missing:
        symbol=_symbol(ticker)
        try:
            history=download_symbol(ticker,HISTORY_PERIOD)
            if history is None or history.empty:continue
            bundle=train_stock_bundle(history,symbol,cutoff,"A",train_horizons=True)
            result=predict_stock(history,bundle,cutoff);horizons=add_multihorizon_predictions(history,bundle,cutoff);idx=df.index[df["Ticker"]==ticker]
            if len(idx)==0:continue
            for i in idx:
                for key in ["Current_Price","Pred_Open","Pred_High","Pred_Low","Pred_Close","Confidence","Direction"]:
                    if key in result:df.at[i,key if key=="Current_Price" else ("AI_"+key if key.startswith("Pred_") else "AI_"+key)]=result[key]
                df.at[i,"AI_Target"]=result.get("Pred_Close");df.at[i,"AI_Open"]=result.get("Pred_Open");df.at[i,"AI_High"]=result.get("Pred_High");df.at[i,"AI_Low"]=result.get("Pred_Low");df.at[i,"AI_Confidence"]=result.get("Confidence");df.at[i,"AI_Direction"]=result.get("Direction");df.at[i,"AI_Source"]="PORTFOLIO_ON_DEMAND"
                for _,hr in horizons.iterrows():df.at[i,f"Horizon_{int(hr['HorizonDays'])}D"]=float(hr["Expected_Return"])
        except Exception as exc:print(f"Portfolio AI forecast warning {symbol}: {exc}")
    return df
def _attach_predictions(df):
    pred,pred_date=_latest_predictions()
    if pred.empty:
        df["AI_Target"]=np.nan;return _portfolio_ai_predictions(df,pred_date)
    keep=[c for c in ["Symbol","Pred_Close","Pred_Open","Pred_High","Pred_Low","Confidence","CalibratedConfidence","Direction","FinalDecisionScore","Action","Horizon_1D","Horizon_3D","Horizon_5D","Horizon_7D","Horizon_20D"] if c in pred.columns]
    p=pred[keep].copy();p["Ticker"]=p["Symbol"].map(_canonical_ticker);p=p.drop(columns=["Symbol"]).rename(columns={"Pred_Close":"AI_Target","Pred_Open":"AI_Open","Pred_High":"AI_High","Pred_Low":"AI_Low","Action":"AI_Action"});p=p.drop_duplicates(subset=["Ticker"],keep="last")
    for c in ["AI_Target","AI_Open","AI_High","AI_Low"]:
        if c in p:p[c]=pd.to_numeric(p[c],errors="coerce")
    for h in HORIZONS:
        c=f"Horizon_{h}D"
        if c in p.columns:p[c]=pd.to_numeric(p[c],errors="coerce")
    merged=df.merge(p,on="Ticker",how="left");return _portfolio_ai_predictions(merged,pred_date),pred_date
def _next_trading_date(start_date,days):
    if not start_date:return "-"
    d=pd.Timestamp(start_date);count=0
    while count<days:
        d+=pd.Timedelta(days=1)
        if d.weekday()<5:count+=1
    return str(d.date())
def _sell_plan(row,current,avg,prediction_date):
    """10% is the minimum target; use the highest reliable AI horizon above 10% when available."""
    if current is None or pd.isna(current) or pd.isna(avg) or not prediction_date:
        return TARGET_PROFIT_PCT,None,"NO AI DATA","-"
    forecasts=[]
    for h in HORIZONS:
        value=row.get(f"Horizon_{h}D")
        try:
            value=float(value)
            if np.isfinite(value):forecasts.append((h,value))
        except (TypeError,ValueError):
            continue
    if not forecasts:return TARGET_PROFIT_PCT,float(avg)*(1+TARGET_PROFIT_PCT/100),"NO AI DATA","-"
    best_h,best_return=max(forecasts,key=lambda x:x[1])
    target_return=max(TARGET_PROFIT_PCT,best_return)
    if best_return<=TARGET_PROFIT_PCT:
        target_h=None
        for h,value in forecasts:
            if value>=TARGET_PROFIT_PCT:
                target_h=h;break
    else:
        target_h=best_h
    target_price=float(avg)*(1+target_return/100)
    if target_h is None:
        return target_return,target_price,f">{max(h for h,_ in forecasts)}D","-"
    date=_next_trading_date(prediction_date,int(target_h))
    if target_h<=1:window=f"1D ({date})"
    else:window=f"{target_h}D ({_next_trading_date(prediction_date,max(1,int(target_h)-2))}→{date})"
    return target_return,target_price,window,date
def _average_plan(row):
    qty=float(row["Quantity"] or 0);price=row["Current_Price"];avg=row["Average_Price"];target=row.get("AI_Target")
    if pd.isna(avg) and price is not None and pd.notna(row["Reported_Return"]) and float(row["Reported_Return"])>-100:avg=price/(1+float(row["Reported_Return"])/100);row["Average_Price"]=avg;row["AveragePriceSource"]="ESTIMATED_FROM_RETURN"
    elif pd.isna(avg) and price is not None and qty>0 and pd.notna(row["Reported_PnL"]):avg=price-float(row["Reported_PnL"])/qty;row["Average_Price"]=avg;row["AveragePriceSource"]="ESTIMATED_FROM_PNL"
    elif pd.notna(avg):row["AveragePriceSource"]="CSV"
    else:row["AveragePriceSource"]="UNAVAILABLE"
    row["Profit_Target_Price"]=float(avg)*(1+TARGET_PROFIT_PCT/100) if pd.notna(avg) else None
    if price is None or pd.isna(avg) or qty<=0:
        row["Invested_Value"]=qty*avg if pd.notna(avg) else 0;row["Recovery_Gap_Pct"]=None;row["Target_Return_Pct"]=None;row["Recommended_Qty"]=0;row["New_Average_Price"]=None;row["Projected_Return_At_AI_Target"]=None;row["Averaging_Action"]="DATA WAIT";row["Decision"]="DATA WAIT";row["Sell_Window"]="NO PRICE";row["Sell_Reason"]="Insufficient portfolio/price data";return row
    row["Invested_Value"]=qty*float(avg);row["Recovery_Gap_Pct"]=(float(avg)-price)/float(avg)*100 if avg else None
    if pd.isna(target):
        row["Target_Return_Pct"]=None;row["Recommended_Qty"]=0;row["New_Average_Price"]=float(avg);row["Projected_Return_At_AI_Target"]=None;row["Averaging_Action"]="DO NOT AVG";row["Decision"]="HOLD";row["Sell_Window"]="NO AI DATA";row["Sell_Reason"]="AI forecast unavailable for this holding";return row
    target=float(target);row["Target_Return_Pct"]=(target/float(avg)-1)*100;conf=float(row.get("AI_Confidence",row.get("CalibratedConfidence",row.get("Confidence",0))) or 0);target_return,target_price,sell_window,sell_date=_sell_plan(row,price,float(avg),row.get("PredictionDate"));row["Sell_Target_Profit_Pct"]=target_return;row["Sell_Target_Price"]=target_price;row["Sell_Date"]=sell_date;row["Sell_Window"]=sell_window;profit_target=float(target_price or row["Profit_Target_Price"])
    if price>=profit_target:
        row["Recommended_Qty"]=0;row["New_Average_Price"]=float(avg);row["Projected_Return_At_AI_Target"]=(target/avg-1)*100;row["Averaging_Action"]="DO NOT AVG";row["Decision"]="SELL / PROFIT BOOK";row["Sell_Window"]="NOW";row["Sell_Date"]=str(row.get("PredictionDate") or "TODAY");row["Sell_Reason"]=f"Minimum {TARGET_PROFIT_PCT:.0f}% target reached; dynamic AI target {target_return:.1f}%";return row
    desired_avg=target/(1+TARGET_PROFIT_PCT/100)
    if desired_avg<=price or target<=price*(1-SELL_RISK_GAP_PCT/100):row["Recommended_Qty"]=0;row["New_Average_Price"]=float(avg);row["Projected_Return_At_AI_Target"]=(target/avg-1)*100;row["Averaging_Action"]="DO NOT AVG";row["Decision"]="SELL / EXIT" if target<price*(1-SELL_RISK_GAP_PCT/100) else "HOLD";row["Sell_Window"]="NOW" if target<price*(1-SELL_RISK_GAP_PCT/100) else row["Sell_Window"];row["Sell_Reason"]="AI target does not support a safe 10% recovery";return row
    required=qty*(float(avg)-desired_avg)/(desired_avg-price);budget=qty*float(avg)*MAX_AVERAGING_CAPITAL_PCT/100;max_qty=int(budget//price);rec=min(max(0,int(required+0.9999)),max_qty);new_avg=(qty*float(avg)+rec*price)/(qty+rec) if rec>0 else float(avg);projected=(target/new_avg-1)*100
    row["Recommended_Qty"]=rec;row["New_Average_Price"]=new_avg;row["Projected_Return_At_AI_Target"]=projected;row["Max_Averaging_Capital"]=budget
    if rec>0 and projected>=TARGET_PROFIT_PCT and conf>=MIN_AI_CONFIDENCE and row["Recovery_Gap_Pct"]>=5:row["Averaging_Action"]="AVERAGE";row["Decision"]="AVG";row["Sell_Reason"]="AI recovery supports reduced average and profit target"
    elif row["Recovery_Gap_Pct"]<=0:row["Averaging_Action"]="DO NOT AVG";row["Decision"]="HOLD";row["Sell_Reason"]="Position is not below average cost"
    else:row["Averaging_Action"]="DO NOT AVG";row["Decision"]="HOLD / RECOVERY";row["Sell_Reason"]="Recovery target not strong enough or confidence is low"
    return row
def portfolio_snapshot():
    df=load_portfolio()
    if df.empty:return df,{"Positions":0,"Value":0.0,"PnL":0.0,"Return":0.0,"ActionCounts":{}}
    prices={t:_price(t) for t in df["Ticker"].dropna().unique()};df["Current_Price"]=df["Ticker"].map(prices);df,prediction_date=_attach_predictions(df);df["PredictionDate"]=prediction_date;df["AveragePriceSource"]="UNAVAILABLE";df=df.apply(_average_plan,axis=1)
    decision_map={"SELL / PROFIT BOOK":"SELL","SELL / EXIT":"SELL","HOLD / RECOVERY":"HOLD","DATA WAIT":"WAIT"}
    df["Decision"]=df["Decision"].map(lambda x:decision_map.get(str(x),str(x).strip().split()[0] if str(x).strip() else "WAIT"))
    df["Current_Value"]=df["Quantity"]*df["Current_Price"].fillna(0);mask=df["Reported_PnL"].notna()&df["AveragePriceSource"].str.startswith("ESTIMATED");df["PnL"]=df["Current_Value"]-df["Invested_Value"];df.loc[mask,"PnL"]=df.loc[mask,"Reported_PnL"];df["Return_Pct"]=df.apply(lambda r:float(r["Reported_Return"]) if str(r["AveragePriceSource"]).startswith("ESTIMATED") and pd.notna(r["Reported_Return"]) else ((r["PnL"]/r["Invested_Value"]*100) if r["Invested_Value"] else None),axis=1)
    total_inv=float(df["Invested_Value"].sum());total_val=float(df["Current_Value"].sum());total_pnl=float(df["PnL"].sum());summary={"Positions":len(df),"Value":total_val,"PnL":total_pnl,"Return":total_pnl/total_inv*100 if total_inv else 0.0,"ActionCounts":df["Decision"].value_counts().to_dict(),"PredictionDate":prediction_date};return df,summary