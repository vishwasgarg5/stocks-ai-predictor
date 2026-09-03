import importlib
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from .config import DATA_DIR,UNIVERSE_FILES,NIFTY_SYMBOL,BANKNIFTY_SYMBOL,VIX_SYMBOL,MAX_UNIVERSE,HISTORY_PERIOD,MIN_AVG_TRADED_VALUE,MIN_PRICE
from .utils import clean_ohlcv
NSE_EQUITY_URL="https://archives.nseindia.com/content/equities/EQUITY_L.csv"

def normalize_symbol(symbol):
    symbol=str(symbol).strip().upper(); return symbol[:-3] if symbol.endswith(".NS") else symbol

def read_symbols_from_csv(path):
    try:
        df=pd.read_csv(path); column=next((c for c in ["SYMBOL","Symbol","symbol","Ticker","ticker"] if c in df.columns),None)
        if column is None:return []
        return list(dict.fromkeys(normalize_symbol(x) for x in df[column].dropna() if normalize_symbol(x) and normalize_symbol(x)!="SYMBOL"))
    except Exception:return []

def download_nse_equity_list():
    try:
        r=requests.get(NSE_EQUITY_URL,timeout=20,headers={"User-Agent":"Mozilla/5.0","Accept":"text/csv,*/*"})
        if r.status_code!=200:return []
        from io import StringIO
        df=pd.read_csv(StringIO(r.text)); column=next((c for c in ["SYMBOL","Symbol","symbol"] if c in df.columns),None)
        return list(dict.fromkeys(normalize_symbol(x) for x in df[column].dropna())) if column else []
    except Exception as exc:print(f"NSE universe download failed: {exc}");return []

def load_universe():
    symbols=download_nse_equity_list()
    if symbols:print(f"Using broad NSE universe: {len(symbols)} stocks");return symbols[:MAX_UNIVERSE]
    for path in [Path(p) for p in UNIVERSE_FILES]:
        symbols=read_symbols_from_csv(path) if path.exists() else []
        if symbols:print(f"Using repository universe fallback: {len(symbols)} stocks");return symbols[:MAX_UNIVERSE]
    for module_name in ["src.nifty150_symbols","src.nifty150","src.market_universe"]:
        try:
            module=importlib.import_module(module_name)
            for attr in ["NIFTY150_SYMBOLS","NIFTY_150_SYMBOLS","SYMBOLS","STOCKS"]:
                values=getattr(module,attr,None)
                if values:return [normalize_symbol(x) for x in list(values)[:MAX_UNIVERSE]]
        except Exception:continue
    raise RuntimeError("Unable to load NSE stock universe")

def download_symbol(symbol,period=HISTORY_PERIOD):
    try:
        df=yf.download(f"{normalize_symbol(symbol)}.NS",period=period,interval="1d",auto_adjust=False,progress=False,threads=False); df=clean_ohlcv(df)
        return df if len(df)>=30 else None
    except Exception as exc:print(f"{symbol}: data download failed: {exc}");return None

def download_many(symbols,period=HISTORY_PERIOD,workers=8):
    result={};symbols=list(dict.fromkeys(symbols))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(download_symbol,s,period):s for s in symbols}
        for future in as_completed(futures):
            symbol=futures[future]
            try:
                df=future.result()
                if df is not None and not df.empty:result[symbol]=df
            except Exception as exc:print(f"{symbol}: {exc}")
    print(f"Downloaded usable data for {len(result)}/{len(symbols)} stocks");return result

def liquidity_score(df):
    if df is None or len(df)<20:return 0.0
    recent=df.tail(60);avg_value=float((recent["Close"]*recent["Volume"]).mean());price=float(recent["Close"].iloc[-1])
    if avg_value<MIN_AVG_TRADED_VALUE or price<MIN_PRICE:return 0.0
    return float(min(np.log10(max(avg_value,1))*8,100))

def filter_liquid_universe(data_map):return {s:df for s,df in data_map.items() if liquidity_score(df)>0}

def get_nifty_data(period="1y",symbol=None):
    try:return clean_ohlcv(yf.download(symbol or NIFTY_SYMBOL,period=period,interval="1d",auto_adjust=False,progress=False,threads=False))
    except Exception:return pd.DataFrame()

def _index_snapshot(symbol,period="3mo",cutoff=None):
    df=get_nifty_data(period,symbol)
    if cutoff is not None and not df.empty: df=df[df.index.date<=pd.Timestamp(cutoff).date()]
    if df.empty:return {"Close":np.nan,"Change1D":np.nan}
    close=float(df["Close"].iloc[-1]); prev=float(df["Close"].iloc[-2]) if len(df)>1 else close
    return {"Close":close,"Change1D":(close/prev-1)*100 if prev else 0.0}

def get_market_snapshot(data_map=None,cutoff=None):
    snap={"NIFTY":_index_snapshot(NIFTY_SYMBOL,cutoff=cutoff),"BANKNIFTY":_index_snapshot(BANKNIFTY_SYMBOL,cutoff=cutoff),"VIX":_index_snapshot(VIX_SYMBOL,cutoff=cutoff)}
    if data_map:
        ups=downs=0
        cutoff_date=pd.Timestamp(cutoff).date() if cutoff is not None else None
        for df in data_map.values():
            if df is None or len(df)<2:continue
            if cutoff_date is not None:
                df=df[df.index.date<=cutoff_date]
            if len(df)<2:continue
            a,b=float(df["Close"].iloc[-2]),float(df["Close"].iloc[-1]); ups+=b>a; downs+=b<a
        total=ups+downs; snap["Breadth"]={"Advancers":ups,"Decliners":downs,"Ratio":ups/max(downs,1),"Score":100*ups/max(total,1) if total else 50}
    else:snap["Breadth"]={"Advancers":0,"Decliners":0,"Ratio":0,"Score":50}
    return snap

def get_completed_session_date(mode="morning",reference_date=None):
    df=get_nifty_data("1mo")
    if df.empty:return None
    from .utils import today_ist
    reference=reference_date or today_ist();dates=sorted({pd.Timestamp(x).date() for x in df.index})
    valid=[x for x in dates if x<reference] if mode=="morning" else [x for x in dates if x<=reference]
    return max(valid) if valid else None

def get_data_cutoff_date(data_map,reference_date=None,fallback=None,min_fraction=0.50):
    from .utils import today_ist
    reference=pd.Timestamp(reference_date or today_ist()).date();counts={};total=max(len(data_map),1)
    for df in data_map.values():
        if df is None or df.empty:continue
        dates={pd.Timestamp(x).date() for x in df.index if pd.Timestamp(x).date()<reference}
        for d in dates:counts[d]=counts.get(d,0)+1
    threshold=max(5,int(np.ceil(total*min_fraction)));valid=[d for d,c in counts.items() if c>=threshold]
    if valid:return max(valid)
    return fallback if fallback is not None and pd.Timestamp(fallback).date()<reference else None

def get_previous_session_date(session_date):
    df=get_nifty_data("3mo")
    if df.empty:return None
    dates=sorted({pd.Timestamp(x).date() for x in df.index});previous=[x for x in dates if x<session_date];return max(previous) if previous else None

def get_market_regime(cutoff_date=None):
    df=get_nifty_data("1y")
    if df.empty:return {"name":"UNKNOWN","score":50}
    if cutoff_date is not None:df=df[df.index.date<=cutoff_date]
    if len(df)<60:return {"name":"NORMAL","score":50}
    close=df["Close"];sma20=close.rolling(20).mean().iloc[-1];sma50=close.rolling(50).mean().iloc[-1];vol=close.pct_change().rolling(20).std().iloc[-1];current=close.iloc[-1]
    regime="BULL" if current>sma20>sma50 else "BEAR" if current<sma20<sma50 else "SIDEWAYS"
    if vol>0.018:regime="HIGH VOL"
    return {"name":regime,"score":{"BULL":80,"BEAR":40,"SIDEWAYS":60,"HIGH VOL":45}.get(regime,50)}

def get_row_for_date(df,target_date):
    if df is None or df.empty:return None
    target_date=pd.Timestamp(target_date).date()
    for index in df.index:
        if pd.Timestamp(index).date()==target_date:return df.loc[index]
    return None

def get_previous_row(df,target_date):
    if df is None or df.empty:return None
    target_date=pd.Timestamp(target_date).date();rows=[(i,r) for i,r in df.iterrows() if pd.Timestamp(i).date()<target_date]
    return rows[-1][1] if rows else None
