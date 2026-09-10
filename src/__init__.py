"""Package bootstrap and incremental monthly market-data adapter."""
from pathlib import Path
from threading import RLock
import pandas as pd
from .config import DATA_DIR,HISTORY_PERIOD,MAX_UNIVERSE
from . import market_data as _market_data
_PARTITION_DIR=DATA_DIR/'stage2'/'market_data';_LEGACY_DIR=DATA_DIR/'stage2'/'ohlcv';_LOCK=RLock();_MEM={}

def _period_start(period=HISTORY_PERIOD):
 end=pd.Timestamp.now(tz='Asia/Kolkata').tz_localize(None).normalize();t=str(period or HISTORY_PERIOD).lower().strip()
 if t.endswith('y'):start=end-pd.DateOffset(years=max(1,int(t[:-1] or 1)))
 elif t.endswith('mo'):start=end-pd.DateOffset(months=max(1,int(t[:-2] or 1)))
 elif t.endswith('d'):start=end-pd.Timedelta(days=max(1,int(t[:-1] or 1)))
 else:start=end-pd.DateOffset(years=5)
 return start.normalize(),end

def _partition_path(d):d=pd.Timestamp(d);return _PARTITION_DIR/f'{d.year:04d}-{d.month:02d}.csv'
def _empty():return pd.DataFrame(columns=['Date','Symbol','Open','High','Low','Close','Volume'])
def _read_partition(path):
 path=Path(path)
 with _LOCK:
  if path in _MEM:return _MEM[path].copy()
  if not path.exists():return _empty()
  try:
   d=pd.read_csv(path);d['Date']=pd.to_datetime(d['Date'],errors='coerce').dt.normalize();d['Symbol']=d.get('Symbol','').astype(str)
   for c in ['Open','High','Low','Close','Volume']:
    if c not in d.columns:d[c]=pd.NA
   d=d[['Date','Symbol','Open','High','Low','Close','Volume']].dropna(subset=['Date']);_MEM[path]=d.copy();return d
  except Exception as e:raise RuntimeError(f'Unable to read market-data partition {path}: {e}') from e

def _read_partitioned_symbol(symbol,start=None,end=None):
 start=pd.Timestamp(start) if start is not None else _period_start()[0];end=pd.Timestamp(end) if end is not None else _period_start()[1];out=[];m=start.replace(day=1);last=end.replace(day=1)
 while m<=last:
  d=_read_partition(_partition_path(m));x=d[d['Symbol'].str.upper()==str(symbol).upper()] if not d.empty else d
  if not x.empty:out.append(x)
  m+=pd.DateOffset(months=1)
 if not out:return None
 d=pd.concat(out,ignore_index=True).drop_duplicates(['Symbol','Date'],keep='last').sort_values('Date');d=d[(d.Date>=start)&(d.Date<=end)];return _market_data.clean_ohlcv(d.set_index('Date'))

def _read_legacy_symbol(symbol):
 p=_LEGACY_DIR/f'{symbol}.csv'
 if not p.exists():return None
 try:return _market_data.clean_ohlcv(pd.read_csv(p,index_col=0,parse_dates=True))
 except Exception as e:raise RuntimeError(f'Unable to read legacy cache for {symbol}: {e}') from e

def _read_cached_ohlcv(symbol,period=HISTORY_PERIOD):
 start,end=_period_start(period);d=_read_partitioned_symbol(symbol,start,end)
 if d is not None and not d.empty:return d
 old=_read_legacy_symbol(symbol)
 if old is not None and not old.empty:
  _save_partitioned(symbol,old);return old[(old.index>=start)&(old.index<=end)]
 return None

def _save_partitioned(symbol,df):
 if df is None or df.empty:return
 x=_market_data.clean_ohlcv(df.copy()).reset_index();x=x.rename(columns={x.columns[0]:'Date'}) if 'Date' not in x.columns else x;x['Date']=pd.to_datetime(x['Date'],errors='coerce').dt.normalize();x['Symbol']=str(symbol).upper();x=x[['Date','Symbol','Open','High','Low','Close','Volume']].dropna(subset=['Date']);_PARTITION_DIR.mkdir(parents=True,exist_ok=True)
 with _LOCK:
  for month,g in x.groupby(x.Date.dt.to_period('M')):
   p=_PARTITION_DIR/f'{month.year:04d}-{month.month:02d}.csv';g=g.sort_values('Date');old=_read_partition(p)
   if old.empty:
    g.to_csv(p,index=False);_MEM[p]=g.copy();continue
   osym=old[old.Symbol.str.upper()==str(symbol).upper()];new_dates=set(pd.to_datetime(g.Date).dt.date);old_dates=set(pd.to_datetime(osym.Date).dt.date)
   if not old.empty and (not osym.empty) and min(new_dates)>max(old_dates) and not (new_dates&old_dates):
    g.to_csv(p,mode='a',header=False,index=False);_MEM[p]=pd.concat([old,g],ignore_index=True);continue
   keep=old[old.Symbol.str.upper()!=str(symbol).upper()];merged=pd.concat([keep,g],ignore_index=True).drop_duplicates(['Symbol','Date'],keep='last').sort_values(['Symbol','Date']);merged.to_csv(p,index=False);_MEM[p]=merged.copy()

def _save_cached_ohlcv(symbol,df):_save_partitioned(symbol,df)
def _incremental_download_symbol(symbol,period=HISTORY_PERIOD,retries=2):
 start,end=_period_start(period);cached=_read_cached_ohlcv(symbol,period)
 try:
  if cached is None or cached.empty:fresh=_market_data._download_range(symbol,start=start,end=end,period=None,retries=retries)
  else:
   pieces=[cached];a,b=cached.index.min(),cached.index.max()
   if a>start:
    z=_market_data._download_range(symbol,start=start,end=a,period=None,retries=retries)
    if z is not None and not z.empty:pieces.append(z)
   if b<end:
    z=_market_data._download_range(symbol,start=b+pd.Timedelta(days=1),end=end,period=None,retries=retries)
    if z is not None and not z.empty:pieces.append(z)
   fresh=pd.concat(pieces)
  fresh=_market_data.clean_ohlcv(fresh).drop_duplicates(keep='last').sort_index();_save_partitioned(symbol,fresh);return fresh[(fresh.index>=start)&(fresh.index<=end)]
 except (TypeError,AttributeError,KeyError) as e:raise RuntimeError(f'Programming error while updating {symbol}: {e}') from e
 except Exception as e:print(f'Market data update failed for {symbol}: {e}');return cached if cached is not None else None

def _bounded_download_many(symbols,period=HISTORY_PERIOD,workers=8):
 unique=[];seen=set()
 for s in symbols or []:
  s=str(s).strip().upper()
  if s and s not in seen:seen.add(s);unique.append(s)
  if len(unique)>=int(MAX_UNIVERSE):break
 from concurrent.futures import ThreadPoolExecutor,as_completed
 results={};failures={}
 with ThreadPoolExecutor(max_workers=min(max(1,int(workers)),8)) as pool:
  fs={pool.submit(_incremental_download_symbol,s,period,2):s for s in unique}
  for f in as_completed(fs):
   s=fs[f]
   try:
    d=f.result()
    if d is not None and not d.empty:results[s]=d
    else:failures[s]='NO_DATA'
   except RuntimeError as e:
    failures[s]=str(e)
    if 'Programming error' in str(e):raise
   except Exception as e:failures[s]=str(e)
 print(f'Repository-cache data available for {len(results)}/{len(unique)} stocks; failures={len(failures)}')
 return results

_market_data._read_cached_ohlcv=_read_cached_ohlcv;_market_data._save_cached_ohlcv=_save_cached_ohlcv;_market_data.download_symbol=_incremental_download_symbol;_market_data.download_many=_bounded_download_many
read_cached_ohlcv=_read_cached_ohlcv;save_cached_ohlcv=_save_cached_ohlcv;download_symbol=_incremental_download_symbol;download_many=_bounded_download_many
