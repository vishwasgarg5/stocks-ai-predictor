import numpy as np
import pandas as pd
import yfinance as yf
from .config import INTRADAY_PERIOD, INTRADAY_INTERVAL, INTRADAY_TOP_N, INTRADAY_MIN_ROWS, INTRADAY_MIN_MOVE
from .utils import clean_ohlcv


def _yf_symbol(symbol):
    s=str(symbol).strip().upper()
    return s if s.startswith("^") or s.endswith(".NS") else f"{s}.NS"


def download_intraday(symbol, cutoff_date=None):
    try:
        df=clean_ohlcv(yf.download(_yf_symbol(symbol), period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL, auto_adjust=False, progress=False, threads=False))
        if cutoff_date is not None and not df.empty:
            cutoff=pd.Timestamp(cutoff_date)
            if getattr(df.index,"tz",None) is not None:
                cutoff=cutoff.tz_localize(df.index.tz) if cutoff.tzinfo is None else cutoff.tz_convert(df.index.tz)
            # Include the complete cutoff session, not only timestamps <= midnight.
            df=df[pd.to_datetime(df.index).date <= cutoff.date()]
        return df
    except Exception as exc:
        print(f"{symbol}: intraday data failed: {exc}")
        return pd.DataFrame()


def add_intraday_features(df):
    x=df.copy(); close,high,low,volume=x["Close"],x["High"],x["Low"],x["Volume"]
    day=pd.Series(x.index.date,index=x.index); typical=(high+low+close)/3
    x["VWAP"]=(typical*volume).groupby(day).cumsum()/volume.groupby(day).cumsum().replace(0,np.nan)
    x["EMA9"]=close.ewm(span=9,adjust=False).mean(); x["EMA20"]=close.ewm(span=20,adjust=False).mean()
    x["VolumeMA20"]=volume.rolling(20).mean(); x["RelativeVolume"]=volume/x["VolumeMA20"].replace(0,np.nan)
    x["Return1"]=close.pct_change(); x["Return4"]=close.pct_change(4); x["Range"]=(high-low)/close.replace(0,np.nan)
    x["Momentum"]=close/close.shift(8)-1; x["DayHigh"]=x.groupby(day)["High"].transform("max"); x["DayLow"]=x.groupby(day)["Low"].transform("min")
    return x.replace([np.inf,-np.inf],np.nan)


def calculate_intraday_setup(df):
    if df.empty:return None,"NO_DATA"
    x=add_intraday_features(df).dropna()
    if len(x)<INTRADAY_MIN_ROWS:return None,"INSUFFICIENT_DATA"
    row=x.iloc[-1]; current,vwap,ema9,ema20=map(float,(row["Close"],row["VWAP"],row["EMA9"],row["EMA20"]))
    relative_volume,momentum,range_pct=float(row["RelativeVolume"]),float(row["Momentum"]),float(row["Range"])
    score=50.0
    score += 12 if current>vwap else -12; score += 10 if ema9>ema20 else -8
    score += 12 if relative_volume>1.5 else 5 if relative_volume>1.1 else 0
    score += 10 if momentum>0.01 else -10 if momentum<-0.01 else 0
    score += 6 if range_pct>0.015 else 0
    score=float(np.clip(score,0,100)); bias="UP" if score>=60 else "DOWN" if score<=40 else "NEUTRAL"
    if bias=="NEUTRAL":return None,"NEUTRAL"
    # Stage 10.1: keep a useful ranked watchlist without requiring an unusually rare setup.
    if score<65:return None,"LOW_SCORE"
    if relative_volume<1.00:return None,"LOW_RELATIVE_VOLUME"
    expected_move=max(abs(momentum),range_pct*1.5,INTRADAY_MIN_MOVE)
    if bias=="UP":target,stop_loss=current*(1+expected_move),current*(1-expected_move*0.55)
    else:target,stop_loss=current*(1-expected_move),current*(1+expected_move*0.55)
    confidence=min(95.0,50+abs(score-50)*0.7+min(relative_volume,3)*5)
    if confidence<60:return None,"LOW_CONFIDENCE"
    return {"Current":current,"Bias":bias,"Target":target,"StopLoss":stop_loss,"ExpectedMove":expected_move*100,"Score":score,"Confidence":confidence,"RelativeVolume":relative_volume,"VWAP":vwap,"EMA9":ema9,"EMA20":ema20},"QUALIFIED"


def generate_intraday_watchlist(symbols, cutoff_date=None, max_workers=6):
    from concurrent.futures import ThreadPoolExecutor,as_completed
    symbols=list(dict.fromkeys(str(s).strip().upper() for s in symbols if str(s).strip()))
    data={}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures={executor.submit(download_intraday,symbol,cutoff_date):symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol=futures[future]
            try:
                df=future.result()
                if not df.empty:data[symbol]=df
            except Exception as exc:print(f"{symbol}: {exc}")
    counts={"SCANNED":len(symbols),"DATA_AVAILABLE":len(data),"NO_DATA":max(0,len(symbols)-len(data)),"INSUFFICIENT_DATA":0,"NEUTRAL":0,"LOW_SCORE":0,"LOW_RELATIVE_VOLUME":0,"LOW_CONFIDENCE":0,"QUALIFIED":0}
    results=[]
    for symbol,df in data.items():
        try:
            setup,reason=calculate_intraday_setup(df); counts[reason]=counts.get(reason,0)+1
            if setup is not None:results.append({"Symbol":symbol,**setup})
        except Exception as exc:print(f"{symbol}: setup failed: {exc}")
    counts["QUALIFIED"]=len(results)
    if not results:
        empty=pd.DataFrame();empty.attrs["scan_stats"]=counts;return empty
    result=pd.DataFrame(results).sort_values(["Score","Confidence","RelativeVolume"],ascending=False).head(INTRADAY_TOP_N).reset_index(drop=True)
    counts["RETURNED"]=len(result);result.attrs["scan_stats"]=counts
    return result
