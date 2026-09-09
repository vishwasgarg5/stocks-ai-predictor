from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from .config import TIMEZONE, MORNING_HOUR, MORNING_MINUTE, EVENING_HOUR, EVENING_MINUTE

IST = ZoneInfo(TIMEZONE)
NSE_HOLIDAYS = {"2026-01-15","2026-01-26","2026-02-19","2026-03-03","2026-03-19","2026-03-26","2026-03-31","2026-04-01","2026-04-03","2026-04-14","2026-05-01","2026-05-28","2026-06-26","2026-08-26","2026-09-14","2026-10-02","2026-10-20","2026-11-10","2026-11-24","2026-12-25"}

def now_ist(): return datetime.now(IST)
def today_ist(): return now_ist().date()
def as_date(value):
    if isinstance(value,date) and not isinstance(value,datetime): return value
    if isinstance(value,datetime): return value.date()
    return pd.Timestamp(value).date()
def is_nse_trading_day(day=None):
    d=as_date(day or today_ist()); return d.weekday()<5 and str(d) not in NSE_HOLIDAYS
def previous_nse_trading_day(day=None):
    d=as_date(day or today_ist())
    while True:
        d-=timedelta(days=1)
        if is_nse_trading_day(d): return d
def next_nse_trading_day(day=None):
    d=as_date(day or today_ist())
    while True:
        d+=timedelta(days=1)
        if is_nse_trading_day(d): return d

def flatten_yfinance_columns(df):
    if df is None or df.empty:return df
    if isinstance(df.columns,pd.MultiIndex):
        cols=[]
        for col in df.columns:
            parts=[str(x) for x in col if str(x)!=""]
            cols.append(parts[0] if len(parts)==1 else next((p for p in parts if p in ["Open","High","Low","Close","Adj Close","Volume"]),parts[0]))
        df=df.copy();df.columns=cols
    return df

def clean_ohlcv(df):
    if df is None or df.empty:return pd.DataFrame()
    df=flatten_yfinance_columns(df.copy());required=["Open","High","Low","Close","Volume"]
    for c in required:
        if c not in df.columns:return pd.DataFrame()
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df[required].copy()
    try:
        idx=pd.to_datetime(df.index)
        if getattr(idx,"tz",None) is not None:idx=idx.tz_localize(None)
        df.index=idx.normalize()
    except Exception:return pd.DataFrame()
    return df[~df.index.duplicated(keep="last")].sort_index().replace([np.inf,-np.inf],np.nan).dropna(subset=required)

def price_bucket(price):
    """Return the canonical price-bucket label used by all pipeline stages."""
    try:p=float(price)
    except (TypeError,ValueError):return "-"
    if not np.isfinite(p):return "-"
    if p>=2500:return ">2500"
    if p>=1000:return "1000-2499"
    if p>=500:return "500-999"
    if p>=250:return "250-499"
    if p>=100:return "100-249"
    if p>=50:return "50-99"
    if p>=10:return "10-49"
    return "<10"

def safe_mape(actual,predicted):
    a=np.asarray(actual,dtype=float);p=np.asarray(predicted,dtype=float);mask=np.isfinite(a)&np.isfinite(p)
    if not mask.any():return float("nan")
    a=a[mask];p=p[mask];return float(np.mean(np.abs((a-p)/np.maximum(np.abs(a),1e-8)))*100)
def normalized_mae(actual,predicted):
    a=np.asarray(actual,dtype=float);p=np.asarray(predicted,dtype=float);mask=np.isfinite(a)&np.isfinite(p)
    if not mask.any():return float("nan")
    a=a[mask];p=p[mask];return float(np.mean(np.abs(a-p))/max(float(np.mean(np.abs(a))),1e-8))
def clamp(value,low=0.0,high=100.0):return max(low,min(high,float(value)))
def direction_from_return(return_value,neutral_threshold=0.002):
    if return_value>neutral_threshold:return "UP"
    if return_value<-neutral_threshold:return "DOWN"
    return "NEUTRAL"
def direction_from_prices(previous_close,future_close):
    if previous_close in [None,0] or future_close is None:return "NEUTRAL"
    return direction_from_return(float(future_close)/float(previous_close)-1.0)
def json_safe(value):
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.floating):
        x=float(value);return x if math.isfinite(x) else None
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,(pd.Timestamp,datetime)):return value.isoformat()
    if isinstance(value,Path):return str(value)
    if isinstance(value,float):return value if math.isfinite(value) else None
    raise TypeError(f"Unsupported type: {type(value)}")

def _json_clean(value):
    if isinstance(value,dict):return {str(k): _json_clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,set)):return [_json_clean(v) for v in value]
    if isinstance(value,pd.DataFrame):return _json_clean(value.to_dict(orient="records"))
    if isinstance(value,pd.Series):return _json_clean(value.to_dict())
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.floating,float)):
        number=float(value);return number if math.isfinite(number) else None
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,(pd.Timestamp,datetime,date)):return value.isoformat()
    if isinstance(value,Path):return str(value)
    return value

def write_json(path,data):
    path=Path(path);tmp=path.with_suffix(".tmp")
    cleaned=_json_clean(data)
    with open(tmp,"w",encoding="utf-8") as f:json.dump(cleaned,f,indent=2,allow_nan=False)
    tmp.replace(path)
def read_json(path,default=None):
    path=Path(path)
    if not path.exists():return default
    try:
        with open(path,"r",encoding="utf-8") as f:return json.load(f)
    except Exception:return default
def schedule_status(kind):
    now=now_ist()
    if kind=="morning":scheduled=datetime.combine(now.date(),time(MORNING_HOUR,MORNING_MINUTE),tzinfo=IST)
    elif kind=="evening":scheduled=datetime.combine(now.date(),time(EVENING_HOUR,EVENING_MINUTE),tzinfo=IST)
    else:return "UNKNOWN"
    minutes=(now-scheduled).total_seconds()/60
    return "EARLY" if minutes<-10 else ("ON TIME" if minutes<=15 else "DELAYED")
def is_weekday(day=None):return is_nse_trading_day(day)
def format_money(value):
    try:x=float(value)
    except (TypeError,ValueError):return "-"
    return "-" if not np.isfinite(x) else f"₹{x:,.2f}"
def format_percent(value):
    try:x=float(value)
    except (TypeError,ValueError):return "-"
    return "-" if not np.isfinite(x) else f"{x:+.2f}%"
def split_messages(text,max_length=3900):
    if len(text)<=max_length:return [text]
    messages=[];current=""
    for line in text.splitlines(True):
        if len(current)+len(line)>max_length and current:messages.append(current);current=""
        current+=line
    if current:messages.append(current)
    return messages
