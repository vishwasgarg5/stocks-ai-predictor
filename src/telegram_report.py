"""Compact, defensive Telegram renderer for Stage 28."""
import html,math,os,re,requests
import numpy as np
import pandas as pd
from .config import TELEGRAM_MAX_LENGTH,MODEL_VERSION
_SECTION="\n§§TELEGRAM_SECTION§§\n"
REPORT_HORIZONS=(3,7,10,20,60,180,365)
_BUCKET_LABELS={"B1":">2500","B2":"1000-2499","B3":"500-999","B4":"250-499","B5":"100-249","B6":"50-99","B7":"10-49"}
_BUCKET_ORDER=["10-49","50-99","100-249","250-499","500-999","1000-2499",">2500"]
def _num(v,default=None):
 try:
  if isinstance(v,str):v=v.replace("₹","").replace(",","").replace("%","").strip()
  x=float(v);return x if math.isfinite(x) else default
 except (TypeError,ValueError):return default
def _fmt(v,digits=2):
 x=_num(v);return "-" if x is None else f"{x:,.{digits}f}"
def _pct(v):
 x=_num(v);return "-" if x is None else f"{x:+.1f}%"
def _accuracy(v):
 x=_num(v);return "-" if x is None else f"{x:.1f}%"
def _decision(v):
 s=str(v or "-").strip().upper()
 if s.startswith("SELL") or s in {"EXIT","PROFIT BOOK"}:return "SELL"
 if s.startswith("BUY") or s=="ADD":return "BUY"
 if s.startswith("AVG") or s in {"AVERAGE","AVERAGING"}:return "AVG"
 if s.startswith("WAIT") or s.startswith("DATA WAIT"):return "WAIT"
 if s.startswith("HOLD") or s=="RECOVERY":return "HOLD"
 return s
def _table(headers,rows,max_width=14):
 if not rows:return []
 rows=[[str(v).replace("|","/").replace("\n"," ") for v in r] for r in rows];w=[min(max(len(str(h)),max((len(r[i]) for r in rows),default=0)),max_width) for i,h in enumerate(headers)]
 fit=lambda v,n:v if len(v)<=n else v[:max(1,n-1)]+"…";line=lambda r:" | ".join(fit(r[i],w[i]).ljust(w[i]) for i in range(len(w)));sep="-"*(sum(w)+3*(len(w)-1)+2)
 return ["```",line(headers),sep]+[line(r) for r in rows]+["```"]
def _markdown_to_telegram_html(text):
 chunks=text.split("```");out=[]
 for i,c in enumerate(chunks):
  if i%2:out.append(f"<pre>{html.escape(c.strip(chr(10)))}</pre>")
  else:out.append(re.sub(r"\*([^*\n]+)\*",r"<b>\1</b>",html.escape(c)))
 return "".join(out)
def _plain_text_fallback(text):return text.replace("```","").replace("*","").replace("_","")
def _post_telegram(message,token,chat_id):
 url=f"https://api.telegram.org/bot{token}/sendMessage"
 try:
  r=requests.post(url,json={"chat_id":chat_id,"text":_markdown_to_telegram_html(message),"parse_mode":"HTML"},timeout=20)
  if r.status_code==200:return True
 except Exception:pass
 try:return requests.post(url,json={"chat_id":chat_id,"text":_plain_text_fallback(message)},timeout=20).status_code==200
 except Exception:return False
def _split_safe(text,max_length):
 if len(text)<=max_length:return [text]
 out=[];cur=""
 for line in text.splitlines(True):
  if cur and len(cur)+len(line)>max_length:out.append(cur.rstrip());cur=""
  cur+=line
 if cur:out.append(cur.rstrip())
 return out
def _report_messages(text):return [p for g in [x.strip() for x in text.split(_SECTION) if x.strip()] for p in _split_safe(g,TELEGRAM_MAX_LENGTH)]
def send_telegram(text):
 token,chat_id=os.getenv("TELEGRAM_BOT_TOKEN"),os.getenv("TELEGRAM_CHAT_ID")
 if not token or not chat_id:return False
 msgs=_report_messages(text);return bool(msgs) and all(_post_telegram(x,token,chat_id) for x in msgs)
def _bucket_label(bucket,group=None):
 if group is not None and "PriceBucketLabel" in group.columns and not group["PriceBucketLabel"].dropna().empty:return str(group["PriceBucketLabel"].dropna().iloc[0])
 return _BUCKET_LABELS.get(str(bucket),str(bucket))
def _ordered_price_buckets(values):
 p=[str(x) for x in pd.Series(values).dropna().unique()]
 if any(x.startswith("B") for x in p):return [x for x in ["B7","B6","B5","B4","B3","B2","B1"] if x in p]+[x for x in p if x not in _BUCKET_LABELS]
 return [x for x in _BUCKET_ORDER if x in p]+[x for x in p if x not in _BUCKET_ORDER]
def _score_col(g):
 for c in ("FinalDecisionScore","TradeConfidence","Score"):
  if c in g.columns:return c
def _sort(g):
 c=_score_col(g);return g.sort_values(c,ascending=False,kind="mergesort") if c else g.sort_values("Symbol",kind="mergesort")
def _scan(scan):
 s=scan or {};return "Scanned {:,} | Data {:,} | Liquid {:,} | AI {:,} | Prediction set {:,}".format(*[int(_num(s.get(k),0) or 0) for k in ("Universe","Data","Liquid","AI","Selected")])
def _market(snapshot,regime):
 snapshot=snapshot or {};rows=[]
 for name,key in (("NIFTY","NIFTY"),("BANK","BANKNIFTY"),("FINN","FINNIFTY"),("MIDCP","MIDCPNIFTY"),("VIX","VIX")):
  x=snapshot.get(key,{}) or {};rows.append([name,_fmt(x.get("Close")),_pct(x.get("Change1D"))])
 b=snapshot.get("Breadth",{}) or {};return _table(["Index","Value","1D%"],rows)+[f"Breadth: {int(_num(b.get('Advancers'),0) or 0)}↑ / {int(_num(b.get('Decliners'),0) or 0)}↓ | Regime: {regime or '-'}"]
def _bucket_sections(s):
 if s is None or s.empty or "PriceBucket" not in s.columns:return ["🎯 *PREDICTION SET BY PRICE BUCKET*","No prediction candidates available."]
 out=["🎯 *PREDICTION SET BY PRICE BUCKET*"]
 for b in _ordered_price_buckets(s.PriceBucket):
  g=_sort(s[s.PriceBucket.astype(str)==b]).head(6);rows=[]
  for _,r in g.iterrows():
   cp,p=_num(r.get("Current_Price",r.get("Current_Close"))),_num(r.get("Pred_Close"));rows.append([r.get("Symbol","-"),f"₹{_fmt(cp)}",_pct((p/cp-1)*100 if cp and p else None),_pct(r.get("Horizon_1D")),_pct(r.get("Horizon_5D")),_decision(r.get("Action"))])
  if rows:out += [f"💎 *₹ {_bucket_label(b,g)}* | MAX 6",*_table(["Stock","CMP","Exp","1D","5D","Action"],rows)]
 return out
def _best_pick_table(s):
 if s is None or s.empty or "PriceBucket" not in s.columns:return []
 rows=[]
 for b in _ordered_price_buckets(s.PriceBucket):
  g=s[s.PriceBucket.astype(str)==b]
  if not g.empty:
   r=_sort(g).iloc[0];rows.append([_bucket_label(b,g),r.get("Symbol","-"),f"₹{_fmt(r.get('Current_Price',r.get('Current_Close')))}",_pct(r.get("Expected_Return")),_decision(r.get("Action"))])
 return ["🏆 *BEST PICK — 1 PER PRICE BUCKET*",*_table(["Bucket","Stock","CMP","Exp","Action"],rows)] if rows else []
def _prediction_table(s):
 n=min(10,0 if s is None else len(s));
 if n==0:return ["📈 *PREDICTED OHLC — TOP 0*","No predictions available."]
 return [f"📈 *PREDICTED OHLC — TOP {n}*",*_table(["Stock","Open","High","Low","Close"],[[r.get("Symbol","-"),_fmt(r.get("Pred_Open")),_fmt(r.get("Pred_High")),_fmt(r.get("Pred_Low")),_fmt(r.get("Pred_Close"))] for _,r in _sort(s).head(n).iterrows()])]
def _horizon_value(r,h):
 for c in (f"Horizon_{h}D",f"Expected_Return_{h}D",f"Return_{h}D",f"Expected_{h}D"):
  if c in r.index:return r.get(c)
def _status_from_values(v):
 v=[x for x in v if x is not None and np.isfinite(x)]
 if len(v)<3:return "N/A"
 ratio=sum(x>0 for x in v)/len(v);return "🟢 BULLISH" if ratio>=2/3 else "🟡 MIXED" if ratio>=1/3 else "🔴 WEAK"
def _horizon_status(r):return _status_from_values([_num(_horizon_value(r,h)) for h in REPORT_HORIZONS])
def _horizon_table(s):
 n=min(10,0 if s is None else len(s))
 if not n:return ["🔮 *MULTI-HORIZON OUTLOOK — TOP 0*","No multi-horizon predictions available."]
 rows=[[r.get("Symbol","-")]+[_pct(_horizon_value(r,h)) for h in REPORT_HORIZONS]+[_horizon_status(r)] for _,r in _sort(s).head(n).iterrows()]
 return [f"🔮 *MULTI-HORIZON OUTLOOK — TOP {n}*",*_table(["Stock","3D","7D","10D","20D","60D","180D","365D","Status"],rows,11)]
def _jump(x):
 if x is None or x.empty:return ["🔥 *JUMP WATCH — TOP 5*","No valid jump candidates."]
 rows=[]
 for _,r in x.head(5).iterrows():rows.append([r.get("Symbol","-"),f"₹{_fmt(r.get('Current_Price'))}",f"₹{_fmt(r.get('Target_Level'))}",_pct((_num(r.get('Target_Level'))/_num(r.get('Current_Price'))-1)*100 if _num(r.get('Target_Level')) and _num(r.get('Current_Price')) else None),f"{_num(r.get('Jump_Probability'),0):.0f}%"])
 return ["🔥 *JUMP WATCH — TOP 5*",*_table(["Stock","CMP","Target","Upside","Prob"],rows)]
def _intraday(x):
 if x is None or x.empty:return ["⚡ *INTRADAY TOP 5*","No qualifying intraday setup."]
 return ["⚡ *INTRADAY TOP 5*",*_table(["Stock","Status","Bias","CMP","Target","SL","Conf"],[[r.get("Symbol","-"),r.get("Status","-"),_decision(r.get("Bias")),f"₹{_fmt(r.get('Current'))}",f"₹{_fmt(r.get('Target'))}",f"₹{_fmt(r.get('StopLoss'))}",f"{_num(r.get('Confidence'),0):.0f}%"] for _,r in x.head(5).iterrows()])]
def _ipo(x):
 if x is None or getattr(x,"empty",True):return ["🏦 *IPO INTELLIGENCE*","No active/upcoming IPOs."]
 return ["🏦 *IPO INTELLIGENCE — TOP 5*",*_table(["IPO","Status","Price","GMP","GMP%","AI View"],[[r.get("IPOName","-"),r.get("Status",r.get("IPOStatus","-")),f"₹{_fmt(r.get('PriceHigh'),0)}",f"₹{_fmt(r.get('GMPValue'),0)}",_pct(r.get('GMPPct')),_decision(r.get('IPOAction','WATCH'))] for _,r in x.head(5).iterrows()])]
def _portfolio_horizon_status(x):return _status_from_values([_num(x.get(f"Horizon_{h}D")) for h in REPORT_HORIZONS])
def _portfolio(p):
 if not p:return ["💼 *AI PORTFOLIO MANAGER*","No portfolio positions available today."]
 rows=[]
 for x in p.get("Rows",[]):
  if isinstance(x,dict):rows.append([x.get("Stock","-"),x.get("Quantity","-"),_decision(x.get("Decision")),x.get("Current_Price","-"),x.get("Average_Price","-"),x.get("AI_Target","-"),x.get("Return_Pct","-"),x.get("Profit_Target","-"),_portfolio_horizon_status(x),x.get("Sell_Window","-")])
 return ["💼 *AI PORTFOLIO MANAGER*",*_table(["Stock","Qty","Decision","CMP","Avg","AI Target","Return","10% Target","Horizon","Sell Window"],rows)] if rows else ["💼 *AI PORTFOLIO MANAGER*","Portfolio positions exist, but no renderable rows were produced."]
def _evaluation_sections(e):
 if e is None or e.empty:return ["📊 *PREDICTION vs ACTUAL*","No completed predictions available for evaluation."]
 out=["📊 *PREDICTION vs ACTUAL*"]
 for _,r in e.iterrows():
  rows=[[k,*[_fmt(r.get(f"{p}{f}")) for f in ("Open","High","Low","Close")]] for k,p in (("Predicted","Pred_"),("Actual","Actual_"))];out += [f"💎 *{r.get('Symbol',r.get('Stock','-'))}*",*_table(["Type","Open","High","Low","Close"],rows)]
 return out
def evening_report(market_date,evaluation,metrics,retraining,**kwargs):
 metrics,retraining=metrics or {},retraining or {};return "\n".join([f"🌙 *AI NSE EVENING REPORT*\n📅 {market_date}\n⚙️ {MODEL_VERSION}",_SECTION,*_evaluation_sections(evaluation),_SECTION,"📈 *MODEL ACCURACY*",f"Samples: {int(_num(metrics.get('Samples'),0) or 0)} | Overall MAPE: {_accuracy(metrics.get('OverallMAPE'))} | Close MAPE: {_accuracy(metrics.get('CloseMAPE'))}",f"Direction accuracy: {_accuracy(metrics.get('DirectionAccuracy'))}",_SECTION,"🧠 *MODEL LEARNING*",f"Retrained: {'YES' if retraining.get('Retrained',False) else 'NO'} | Decision: {retraining.get('Decision','-')}",_SECTION,*_portfolio(kwargs.get('portfolio',{}))])
def morning_report(prediction_date,cutoff_date,selected,jump_watchlist,intraday,**kwargs):
 acc,scan=kwargs.get('accuracy',{}),kwargs.get('scan',{});return "\n".join([f"📈 *AI NSE MORNING REPORT*\n📅 {prediction_date}\n⚙️ {MODEL_VERSION}\nData cutoff: {cutoff_date}",_SECTION,"📊 *MARKET OVERVIEW*",*_market(kwargs.get('market_snapshot',{}),kwargs.get('regime','-')),_scan(scan),_SECTION,*_bucket_sections(selected),_SECTION,*_best_pick_table(selected),_SECTION,*_prediction_table(selected),_SECTION,*_horizon_table(selected),_SECTION,*_jump(jump_watchlist),_SECTION,*_intraday(intraday),_SECTION,*_ipo(kwargs.get('ipo',pd.DataFrame())),_SECTION,*_portfolio(kwargs.get('portfolio',{}))])
