import html,math,os,re,requests
import numpy as np
import pandas as pd
from .config import TELEGRAM_MAX_LENGTH,MODEL_VERSION
_SECTION='\n§§TELEGRAM_SECTION§§\n';REPORT_HORIZONS=(3,7,10,20,60,90,180,365)
_BUCKET_LABELS={'B1':'>2500','B2':'1000-2499','B3':'500-999','B4':'250-499','B5':'100-249','B6':'50-99','B7':'10-49'};_BUCKET_ORDER=['10-49','50-99','100-249','250-499','500-999','1000-2499','>2500']
def _num(v,default=None):
 try:
  if isinstance(v,str):v=v.replace('₹','').replace(',','').replace('%','').strip()
  x=float(v);return x if math.isfinite(x) else default
 except:return default
def _fmt(v,digits=2):
 x=_num(v);return '-' if x is None else f'{x:,.{digits}f}'
def _pct(v):
 x=_num(v);return '-' if x is None else f'{x:+.1f}%'
def _accuracy(v):
 x=_num(v);return '-' if x is None else f'{x:.1f}%'
def _decision(v):
 s=str(v or '-').strip().upper()
 if s.startswith('SELL') or s in {'EXIT','PROFIT BOOK'}:return 'SELL'
 if s.startswith('BUY') or s=='ADD':return 'BUY'
 if s.startswith('AVG') or s in {'AVERAGE','AVERAGING'}:return 'AVG'
 if s.startswith('WAIT') or s.startswith('DATA WAIT'):return 'WAIT'
 if s.startswith('REDUCE'):return 'EXIT'
 if s.startswith('HOLD') or s=='RECOVERY':return 'HOLD'
 return s
def _table(headers,rows,max_width=14):
 if not rows:return []
 rows=[[str(v).replace('|','/').replace('\n',' ') for v in r] for r in rows];w=[min(max(len(str(h)),max((len(r[i]) for r in rows),default=0)),max_width) for i,h in enumerate(headers)];fit=lambda v,n:v if len(v)<=n else v[:max(1,n-1)]+'…';line=lambda r:' | '.join(fit(r[i],w[i]).ljust(w[i]) for i in range(len(w)));sep='-'*(sum(w)+3*(len(w)-1)+2);return ['```',line(headers),sep]+[line(r) for r in rows]+['```']
def _markdown_to_telegram_html(text):
 chunks=text.split('```');out=[]
 for i,c in enumerate(chunks):out.append(f'<pre>{html.escape(c.strip(chr(10)))}</pre>' if i%2 else re.sub(r'\*([^*\n]+)\*',r'<b>\1</b>',html.escape(c)))
 return ''.join(out)
def _plain_text_fallback(text):return text.replace('```','').replace('*','').replace('_','')
def _post_telegram(message,token,chat_id):
 try:
  r=requests.post(f'https://api.telegram.org/bot{token}/sendMessage',json={'chat_id':chat_id,'text':_markdown_to_telegram_html(message),'parse_mode':'HTML'},timeout=20)
  if r.status_code==200:return True
 except:pass
 try:return requests.post(f'https://api.telegram.org/bot{token}/sendMessage',json={'chat_id':chat_id,'text':_plain_text_fallback(message)},timeout=20).status_code==200
 except:return False
def _split_safe(text,max_length):
 if len(text)<=max_length:return [text]
 out=[];cur=''
 for line in text.splitlines(True):
  if cur and len(cur)+len(line)>max_length:out.append(cur.rstrip());cur=''
  cur+=line
 if cur:out.append(cur.rstrip())
 return out
def _report_messages(text):return [p for g in [x.strip() for x in text.split(_SECTION) if x.strip()] for p in _split_safe(g,TELEGRAM_MAX_LENGTH)]
def send_telegram(text):
 token,chat_id=os.getenv('TELEGRAM_BOT_TOKEN'),os.getenv('TELEGRAM_CHAT_ID')
 if not token or not chat_id:return False
 m=_report_messages(text);return bool(m) and all(_post_telegram(x,token,chat_id) for x in m)
def _bucket_label(b,g=None):
 if g is not None and 'PriceBucketLabel' in g.columns and not g['PriceBucketLabel'].dropna().empty:return str(g['PriceBucketLabel'].dropna().iloc[0])
 return _BUCKET_LABELS.get(str(b),str(b))
def _ordered_price_buckets(v):
 p=[str(x) for x in pd.Series(v).dropna().unique()];return [x for x in _BUCKET_ORDER if x in p]+[x for x in p if x not in _BUCKET_ORDER]
def _sort(g):
 for c in ('FinalDecisionScore','TradeConfidence','Score'):
  if c in g.columns:return g.sort_values(c,ascending=False,kind='mergesort')
 return g.sort_values('Symbol',kind='mergesort')
def _scan(s):
 s=s or {};return 'Scanned {:,} | Data {:,} | Liquid {:,} | AI {:,} | Prediction set {:,}'.format(*[int(_num(s.get(k),0) or 0) for k in ('Universe','Data','Liquid','AI','Selected')])
def _market(s,regime):
 rows=[]
 for n,k in (('NIFTY','NIFTY'),('BANK','BANKNIFTY'),('FINN','FINNIFTY'),('MIDCP','MIDCPNIFTY'),('VIX','VIX')):
  x=(s or {}).get(k,{}) or {};rows.append([n,_fmt(x.get('Close')),_pct(x.get('Change1D'))])
 b=(s or {}).get('Breadth',{}) or {};return _table(['Index','Value','1D%'],rows)+[f"Breadth: {int(_num(b.get('Advancers'),0) or 0)}↑ / {int(_num(b.get('Decliners'),0) or 0)}↓ | Regime: {regime or '-'}"]
def _bucket_sections(s):
 if s is None or s.empty or 'PriceBucket' not in s.columns:return ['🎯 *PREDICTION SET BY PRICE BUCKET*','No valid prediction candidates.']
 out=['🎯 *PREDICTION SET BY PRICE BUCKET*']
 for b in _ordered_price_buckets(s.PriceBucket):
  g=_sort(s[s.PriceBucket.astype(str)==b]).head(5);rows=[]
  for _,r in g.iterrows():
   cp,p=_num(r.get('Current_Price',r.get('Current_Close'))),_num(r.get('Pred_Close'));rows.append([r.get('Symbol','-'),f'₹{_fmt(cp)}',_pct((p/cp-1)*100 if cp and p else None),_pct(r.get('Horizon_1D')),_pct(r.get('Horizon_5D')),_decision(r.get('Action'))])
  if rows:out += [f'💎 *₹ {_bucket_label(b,g)}* | 5 STOCKS',*_table(['Stock','CMP','Exp','1D','5D','Action'],rows)]
 return out
def _best_pick_table(s):
 if s is None or s.empty or 'PriceBucket' not in s.columns:return ['🏆 *BEST PICK — 1 PER PRICE BUCKET*','No valid candidates.']
 rows=[]
 for b in _ordered_price_buckets(s.PriceBucket):
  g=s[s.PriceBucket.astype(str)==b]
  if not g.empty:
   r=_sort(g).iloc[0];rows.append([_bucket_label(b,g),r.get('Symbol','-'),f"₹{_fmt(r.get('Current_Price',r.get('Current_Close')))}",_pct(r.get('Expected_Return')),_decision(r.get('Action'))])
 return ['🏆 *BEST PICK — 1 PER PRICE BUCKET*',*_table(['Bucket','Stock','CMP','Exp','Action'],rows)]
def _prediction_table(s):
 n=min(10,0 if s is None else len(s));return [f'📈 *PREDICTED OHLC — TOP {n}*','No predictions available.'] if not n else [f'📈 *PREDICTED OHLC — TOP {n}*',*_table(['Stock','Open','High','Low','Close'],[[r.get('Symbol','-'),_fmt(r.get('Pred_Open')),_fmt(r.get('Pred_High')),_fmt(r.get('Pred_Low')),_fmt(r.get('Pred_Close'))] for _,r in _sort(s).head(n).iterrows()])]
def _horizon_value(r,h):
 for c in (f'Horizon_{h}D',f'Expected_Return_{h}D',f'Return_{h}D',f'Expected_{h}D'):
  if c in r.index:return r.get(c)
def _status_from_values(v):
 v=[x for x in v if x is not None and np.isfinite(x)]
 if len(v)<3:return 'N/A'
 q=sum(x>0 for x in v)/len(v);return '🟢 BULLISH' if q>=2/3 else '🟡 MIXED' if q>=1/3 else '🔴 WEAK'
def _horizon_table(s):
 n=min(10,0 if s is None else len(s))
 if not n:return ['🔮 *MULTI-HORIZON OUTLOOK — TOP 0*','No multi-horizon predictions available.']
 rows=[[r.get('Symbol','-')]+[_pct(_horizon_value(r,h)) for h in REPORT_HORIZONS]+[_status_from_values([_num(_horizon_value(r,h)) for h in REPORT_HORIZONS])] for _,r in _sort(s).head(n).iterrows()]
 return [f'🔮 *MULTI-HORIZON OUTLOOK — TOP {n}*',*_table(['Stock','3D','7D','10D','20D','60D','90D','180D','365D','Status'],rows,11)]
def _jump(x):
 if x is None or x.empty:return ['🔥 *JUMP WATCH — TOP 5*','No valid jump candidates.']
 rows=[]
 for _,r in x.head(5).iterrows():
  c,t=_num(r.get('Current_Price')), _num(r.get('Target_Level'));p=_num(r.get('Jump_Probability'))
  if c is not None and t is not None:rows.append([r.get('Symbol','-'),f'₹{_fmt(c)}',f'₹{_fmt(t)}',_pct((t/c-1)*100),f'{p:.0f}%' if p is not None and p>0 else 'N/A'])
 return ['🔥 *JUMP WATCH — TOP 5*',*_table(['Stock','CMP','Target','Upside','Prob'],rows)] if rows else ['🔥 *JUMP WATCH — TOP 5*','No valid jump candidates.']
def _intraday(x):
 if x is None or x.empty:return ['⚡ *INTRADAY TOP 5*','No qualifying intraday setup.']
 return ['⚡ *INTRADAY TOP 5*',*_table(['Stock','Status','Bias','CMP','Target','SL','Conf'],[[r.get('Symbol','-'),r.get('Status','-'),_decision(r.get('Bias')),f"₹{_fmt(r.get('Current'))}",f"₹{_fmt(r.get('Target'))}",f"₹{_fmt(r.get('StopLoss'))}",f"{_num(r.get('Confidence'),0):.0f}%"] for _,r in x.head(5).iterrows()])]
def _ipo(x):
 if x is None or getattr(x,'empty',True):return ['🏦 *IPO INTELLIGENCE*','No active/upcoming IPOs.']
 return ['🏦 *IPO INTELLIGENCE — TOP 5*',*_table(['IPO','Status','Price','GMP','GMP%','AI View'],[[r.get('IPOName','-'),r.get('Status',r.get('IPOStatus','-')),f"₹{_fmt(r.get('PriceHigh'),0)}",f"₹{_fmt(r.get('GMPValue'),0)}",_pct(r.get('GMPPct')),_decision(r.get('IPOAction','WATCH'))] for _,r in x.head(5).iterrows()])]
def _portfolio(p):
 if not p:return ['💼 *AI PORTFOLIO MANAGER*','No portfolio positions available today.']
 rows=[]
 for x in p.get('Rows',[]):
  if not isinstance(x,dict):continue
  raw=_decision(x.get('Decision'));ret=_num(x.get('Return_Pct'));cur=_num(x.get('Current_Price'));avg=_num(x.get('Average_Price'));target=_num(x.get('Profit_Target_Price')) or _num(x.get('AI_Target'));plan=x.get('New_Average_Price','-')
  if cur is not None and avg is not None and cur>=avg*1.10:act='PROFIT BOOK'
  elif raw in {'BUY','AVG','EXIT','WAIT'}:act=raw
  elif ret is not None and ret<=-20:act='AVERAGE' if any((_num(x.get(f'Horizon_{h}D')) or -999)>10 for h in REPORT_HORIZONS) else 'EXIT'
  elif target is not None and cur is not None and target<cur:act='EXIT'
  else:act='HOLD'
  rows.append([x.get('Stock','-'),x.get('Quantity','-'),act,f"₹{_fmt(cur)}",f"₹{_fmt(avg)}",f"₹{_fmt(target)}" if target is not None else 'N/A',_pct(ret),str(plan)])
 return ['💼 *AI PORTFOLIO MANAGER*',*_table(['Stock','Qty','Action','CMP','Avg','Target','Return','Avg Plan'],rows,13)] if rows else ['💼 *AI PORTFOLIO MANAGER*','No renderable portfolio rows.']
def _evaluation_sections(e):
 if e is None or e.empty:return ['📊 *PREDICTION vs ACTUAL*','No completed predictions available for evaluation.']
 out=['📊 *PREDICTION vs ACTUAL*']
 for _,r in e.iterrows():out += [f"💎 *{r.get('Symbol',r.get('Stock','-'))}*",*_table(['Type','Open','High','Low','Close'],[[k]+[_fmt(r.get(f'{p}{f}')) for f in ('Open','High','Low','Close')] for k,p in (('Predicted','Pred_'),('Actual','Actual_'))])]
 return out
def evening_report(market_date,evaluation,metrics,retraining,**kwargs):
 metrics,retraining=metrics or {},retraining or {};return '\n'.join([f'🌙 *AI NSE EVENING REPORT*\n📅 {market_date}\n⚙️ {MODEL_VERSION}',_SECTION,*_evaluation_sections(evaluation),_SECTION,'📈 *MODEL ACCURACY*',f"Samples: {int(_num(metrics.get('Samples'),0) or 0)} | Overall MAPE: {_accuracy(metrics.get('OverallMAPE'))} | Close MAPE: {_accuracy(metrics.get('CloseMAPE'))}",f"Direction accuracy: {_accuracy(metrics.get('DirectionAccuracy'))}",_SECTION,'🧠 *MODEL LEARNING*',f"Retrained: {'YES' if retraining.get('Retrained',False) else 'NO'} | Decision: {retraining.get('Decision','-')}",_SECTION,*_portfolio(kwargs.get('portfolio',{}))])
def morning_report(prediction_date,cutoff_date,selected,jump_watchlist,intraday,**kwargs):
 scan,acc=kwargs.get('scan',{}),kwargs.get('accuracy',{});return '\n'.join([f'📈 *AI NSE MORNING REPORT*\n📅 {prediction_date}\n⚙️ {MODEL_VERSION}\nData cutoff: {cutoff_date}',_SECTION,'📊 *MARKET OVERVIEW*',*_market(kwargs.get('market_snapshot',{}),kwargs.get('regime','-')),_scan(scan),_SECTION,*_bucket_sections(selected),_SECTION,*_best_pick_table(selected),_SECTION,*_prediction_table(selected),_SECTION,*_horizon_table(selected),_SECTION,*_jump(jump_watchlist),_SECTION,*_intraday(intraday),_SECTION,*_ipo(kwargs.get('ipo',pd.DataFrame())),_SECTION,*_portfolio(kwargs.get('portfolio',{}))])
