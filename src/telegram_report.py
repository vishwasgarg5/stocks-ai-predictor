"""Telegram report builders for Stage 28."""
import html,math,os,re,requests
import pandas as pd
from .config import TELEGRAM_MAX_LENGTH,MODEL_VERSION
_SECTION="\n§§TELEGRAM_SECTION§§\n"
_BUCKET_ORDER=["10-49","50-99","100-249","250-499","500-999","1000-2499",">2500"]
_BUCKET_LABELS={"B1":">2500","B2":"1000-2499","B3":"500-999","B4":"250-499","B5":"100-249","B6":"50-99","B7":"10-49"}
_CODE_RE=re.compile(r"^```$");REPORT_HORIZONS=(3,7,10,20,60,180,365)
def _markdown_to_telegram_html(text):
    chunks=text.split("```");out=[]
    for i,chunk in enumerate(chunks):
        if i%2:out.append(f"<pre>{html.escape(chunk.strip(chr(10)))}</pre>")
        else:
            safe=html.escape(chunk);safe=re.sub(r"\*([^*\n]+)\*",r"<b>\1</b>",safe);out.append(safe)
    return "".join(out)
def _plain_text_fallback(text):return text.replace("```","").replace("*","").replace("_","")
def _post_telegram(message,token,chat_id):
    url=f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r=requests.post(url,json={"chat_id":chat_id,"text":_markdown_to_telegram_html(message),"parse_mode":"HTML"},timeout=20)
        if r.status_code==200:return True
    except Exception as exc:print("Telegram HTML exception:",exc)
    try:
        r=requests.post(url,json={"chat_id":chat_id,"text":_plain_text_fallback(message)},timeout=20)
        if r.status_code==200:return True
    except Exception as exc:print("Telegram plain-text exception:",exc)
    return False
def _split_safe(text,max_length):
    if len(text)<=max_length:return [text]
    result=[];current="";in_code=False
    for line in text.splitlines(True):
        toggle=bool(_CODE_RE.match(line.strip()))
        if current and len(current)+len(line)+(4 if in_code else 0)>max_length:
            if in_code:current+="```\n"
            result.append(current.rstrip("\n"));current="```\n" if in_code else ""
        current+=line
        if toggle:in_code=not in_code
    if current:
        if in_code:current+="```\n"
        result.append(current.rstrip("\n"))
    return [x for x in result if x]
def _report_messages(text):
    parts=[p.strip() for p in text.split(_SECTION) if p.strip()]
    groups=[];current=[];boundaries={"BEST PICK","JUMP WATCH","MODEL LEARNING","AI PORTFOLIO MANAGER","IPO INTELLIGENCE","MULTI-HORIZON"}
    for part in parts:
        current.append(part)
        if any(x in part.upper() for x in boundaries):groups.append("\n\n".join(current));current=[]
    if current:groups.append("\n\n".join(current))
    out=[]
    for group in groups:out.extend(_split_safe(group,TELEGRAM_MAX_LENGTH))
    return out
def send_telegram(text):
    token,chat_id=os.getenv("TELEGRAM_BOT_TOKEN"),os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:return False
    messages=_report_messages(text);print(f"Telegram report: sending {len(messages)} message(s).")
    return bool(messages) and all(_post_telegram(m,token,chat_id) for m in messages)
def _num(v,default=None):
    try:
        if isinstance(v,str):v=v.strip().replace("₹","").replace(",","").replace("%","")
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
    if s.startswith("SELL") or s in {"PROFIT BOOK","EXIT"}:return "SELL"
    if s.startswith("HOLD") or s=="RECOVERY":return "HOLD"
    if s.startswith("DATA WAIT") or s=="WAIT":return "WAIT"
    if s.startswith("AVG") or s in {"AVERAGE","AVERAGING"}:return "AVG"
    if s.startswith("BUY") or s=="ADD":return "BUY"
    return s or "-"
def _table(headers,rows,max_width=12):
    if not rows:return []
    rows=[[str(x).replace("|","/").replace("\n"," ") for x in row] for row in rows];widths=[len(str(h)) for h in headers]
    for row in rows:
        for i,v in enumerate(row):
            if i<len(widths):widths[i]=min(max(widths[i],len(v)),max_width)
    def fit(v,w):return v if len(v)<=w else v[:max(1,w-1)]+"…"
    line=lambda row:" | ".join(fit(v,widths[i]).ljust(widths[i]) for i,v in enumerate(row));sep="-"*(sum(widths)+3*(len(widths)-1)+2)
    return ["```",line(headers),sep,*[line(r) for r in rows],"```"]
def _bucket_label(bucket,group=None):
    if group is not None and "PriceBucketLabel" in group.columns:
        v=group["PriceBucketLabel"].dropna().astype(str)
        if not v.empty:return v.iloc[0]
    return _BUCKET_LABELS.get(str(bucket),str(bucket))
def _ordered_price_buckets(values):
    present=[str(x) for x in pd.Series(values).dropna().unique()] if values is not None else []
    if any(x.startswith("B") for x in present):
        order=["B7","B6","B5","B4","B3","B2","B1"];return [x for x in order if x in present]+[x for x in present if x not in order]
    return [x for x in _BUCKET_ORDER if x in present]+[x for x in present if x not in _BUCKET_ORDER]
def _market(snapshot,regime):
    snapshot=snapshot or {};rows=[]
    for name,key in [("NIFTY","NIFTY"),("BANK","BANKNIFTY"),("FINN","FINNIFTY"),("MIDCP","MIDCPNIFTY")]:
        x=snapshot.get(key,{}) or {};rows.append([name,_fmt(x.get("Close")),_pct(x.get("Change1D"))])
    x=snapshot.get("VIX",{}) or {};rows.append(["VIX",_fmt(x.get("Close")),"-"]);b=snapshot.get("Breadth",{}) or {}
    return _table(["Index","Value","1D%"],rows)+[f"Breadth: {int(_num(b.get('Advancers'),0) or 0)}↑ / {int(_num(b.get('Decliners'),0) or 0)}↓ | Regime: {regime or '-'}"]
def _scan(scan):
    scan=scan or {};vals={k:int(_num(scan.get(k),0) or 0) for k in ("Universe","Data","Liquid","AI","Selected")}
    return f"Scanned {vals['Universe']:,} | Data {vals['Data']:,} | Liquid {vals['Liquid']:,} | AI {vals['AI']:,} | Prediction set {vals['Selected']:,}"
def _score_col(g):
    for c in ("FinalDecisionScore","TradeConfidence","Score"):
        if c in g.columns:return c
    return None
def _sort(g):
    c=_score_col(g);return g.sort_values(c,ascending=False,kind="mergesort") if c else g.sort_values("Symbol",kind="mergesort")
def _bucket_sections(selected):
    if selected is None or selected.empty or "PriceBucket" not in selected.columns:return ["🎯 *PREDICTION SET BY PRICE BUCKET*","No prediction candidates available."]
    lines=["🎯 *PREDICTION SET BY PRICE BUCKET*"];shown=0
    for bucket in _ordered_price_buckets(selected["PriceBucket"]):
        g=selected[selected["PriceBucket"].astype(str)==bucket].copy();rows=[]
        for _,r in _sort(g).head(6).iterrows():
            cp=_num(r.get("Current_Price",r.get("Current_Close")));pred=_num(r.get("Pred_Close"));exp=(pred/cp-1)*100 if cp and cp>0 and pred is not None else None
            rows.append([str(r.get("Symbol","-")),f"₹{_fmt(cp)}",_pct(exp),_pct(r.get("Horizon_1D")),_pct(r.get("Horizon_5D")),_pct(r.get("Horizon_20D")),_decision(r.get("Action"))]);shown+=1
        if rows:lines += [f"💎 *₹ {_bucket_label(bucket,g)}* | MAX 6",*_table(["Stock","CMP","Exp","1D","5D","20D","Action"],rows)]
    return lines if shown else ["🎯 *PREDICTION SET BY PRICE BUCKET*","No prediction candidates available."]
def _best_pick_table(selected):
    if selected is None or selected.empty or "PriceBucket" not in selected.columns:return []
    rows=[]
    for bucket in _ordered_price_buckets(selected["PriceBucket"]):
        g=selected[selected["PriceBucket"].astype(str)==bucket]
        if g.empty:continue
        r=_sort(g).iloc[0];score=_num(r.get("FinalDecisionScore",r.get("Score")))
        rows.append([f"₹{_bucket_label(bucket,g)}",str(r.get("Symbol","-")),f"₹{_fmt(r.get('Current_Price',r.get('Current_Close')))}",_pct(r.get("Expected_Return")),"-" if score is None else f"{score:.0f}",_decision(r.get("Action"))])
    return ["🏆 *BEST PICK — 1 PER PRICE BUCKET*",*_table(["Bucket","Stock","CMP","Exp","Score","Action"],rows)] if rows else []
def _prediction_table(selected):
    if selected is None or selected.empty:return ["📈 *PREDICTED OHLC — TOP 10*","No predictions available."]
    rows=[[str(r.get("Symbol","-")),_fmt(r.get("Pred_Open")),_fmt(r.get("Pred_High")),_fmt(r.get("Pred_Low")),_fmt(r.get("Pred_Close"))] for _,r in _sort(selected).head(10).iterrows()]
    return ["📈 *PREDICTED OHLC — TOP 10*",*_table(["Stock","Open","High","Low","Close"],rows)]
def _horizon_value(r,h):
    for c in (f"Horizon_{h}D",f"Expected_Return_{h}D",f"Return_{h}D",f"Expected_{h}D"):
        if c in r.index:return r.get(c)
    return None
def _horizon_status(r):
    vals=[_num(_horizon_value(r,h)) for h in REPORT_HORIZONS];vals=[v for v in vals if v is not None]
    if len(vals)<3:return "N/A"
    positive=sum(v>0 for v in vals);return "🟢 BULLISH" if positive==3 else "🟡 MIXED" if positive==2 else "🔴 WEAK"
def _horizon_table(selected):
    if selected is None or selected.empty:return ["🔮 *MULTI-HORIZON OUTLOOK — TOP 10*","No multi-horizon predictions available."]
    rows=[]
    for _,r in _sort(selected).head(10).iterrows():rows.append([str(r.get("Symbol","-"))]+[_pct(_horizon_value(r,h)) for h in REPORT_HORIZONS]+[_horizon_status(r)])
    return ["🔮 *MULTI-HORIZON OUTLOOK — TOP 10*","3D / 7D / 10D / 20D / 60D / 180D / 365D trading-day expected return.",*_table(["Stock","3D","7D","10D","20D","60D","180D","365D","Status"],rows,max_width=11)]
def _jump(j):
    if j is None or j.empty:return ["🔥 *JUMP WATCH — TOP 5*","No valid jump candidates."]
    x=j.copy()
    if "Jump_Probability" in x.columns:x["_jump_sort"]=pd.to_numeric(x["Jump_Probability"],errors="coerce");x=x.sort_values("_jump_sort",ascending=False,kind="mergesort")
    rows=[]
    for _,r in x.iterrows():
        cp=_num(r.get("Current_Price"));target=_num(r.get("Target_Level"));prob=_num(r.get("Jump_Probability"))
        if cp is None or cp<=0 or target is None or target<=0:continue
        rows.append([str(r.get("Symbol","-")),f"₹{_fmt(cp)}",f"₹{_fmt(target)}",_pct((target/cp-1)*100),f"{max(0,min(100,prob or 0)):.0f}%"])
        if len(rows)==5:break
    return ["🔥 *JUMP WATCH — TOP 5*",*_table(["Stock","CMP","Target","Upside","Prob"],rows)] if rows else ["🔥 *JUMP WATCH — TOP 5*","No valid jump candidates."]
def _intraday(x):
    if x is None or x.empty:return ["⚡ *INTRADAY TOP 5*","No qualifying intraday setup. Live intraday feed may be unavailable, stale, or all candidates failed quality/score gates."]
    rows=[]
    for _,r in x.head(5).iterrows():
        conf=max(0,min(100,_num(r.get("Confidence"),0) or 0));rows.append([str(r.get("Symbol","-")),str(r.get("Status",_decision(r.get("Bias")))),_decision(r.get("Bias")),f"₹{_fmt(r.get('Current'))}",f"₹{_fmt(r.get('Target'))}",f"₹{_fmt(r.get('StopLoss'))}",f"{conf:.0f}%"])
    return ["⚡ *INTRADAY TOP 5*",*_table(["Stock","Status","Bias","CMP","Target","SL","Conf"],rows)]
def _ipo(x):
    if x is None or (hasattr(x,"empty") and x.empty):return ["🏦 *IPO INTELLIGENCE*","No active/upcoming IPOs."]
    y=x.copy();rows=[]
    for _,r in y.head(5).iterrows():rows.append([str(r.get("IPOName","-")),str(r.get("Status",r.get("IPOStatus","-"))),f"₹{_fmt(r.get('PriceHigh',0),0)}",f"₹{_fmt(r.get('GMPValue',0),0)}",_pct(r.get("GMPPct",0)),_decision(r.get("IPOAction","WATCH"))])
    return ["🏦 *IPO INTELLIGENCE — TOP 5*","Subscription/GMP/valuation inputs are shown when supplied by the IPO feed.",*_table(["IPO","Status","Price","GMP","GMP%","AI View"],rows,max_width=14)]
def _portfolio_horizon_status(x):
    vals=[_num(x.get(f"Horizon_{h}D")) for h in REPORT_HORIZONS];vals=[v for v in vals if v is not None]
    if len(vals)>=3:
        positive=sum(v>0 for v in vals);return "🟢 BULLISH" if positive==3 else "🟡 MIXED" if positive==2 else "🔴 WEAK"
    ai=_num(x.get("AI_Target"));cp=_num(x.get("Current_Price"));
    if ai is not None and cp not in (None,0):return "TARGET +" if ai>cp else "TARGET -"
    return "N/A"
def _portfolio(p):
    lines=["💼 *AI PORTFOLIO MANAGER*","All holdings are shown; AI target/horizon fields are N/A only when no prediction is available."]
    if not p:return lines+["No portfolio positions available today."]
    rows=[]
    for x in p.get("Rows",[]):
        if not isinstance(x,dict):continue
        target=x.get("Profit_Target",x.get("Sell_Target_Profit_Pct",x.get("Portfolio_Target_Pct")));rows.append([x.get("Stock","-"),x.get("Quantity","-"),_decision(x.get("Decision")),x.get("Current_Price","-"),x.get("Average_Price","-"),x.get("AI_Target","-"),x.get("Return_Pct","-"),target if _num(target) is not None else "-",_portfolio_horizon_status(x),str(x.get("Sell_Window",x.get("Sell_Date","-")))])
    if rows:lines += _table(["Stock","Qty","Decision","CMP","Avg","AI Target","Return","10% Target","Horizon","Sell Window"],rows,max_width=14)
    return lines
def _evaluation_sections(evaluation):
    if evaluation is None or evaluation.empty:return ["📊 *PREDICTION vs ACTUAL*","No completed predictions available for evaluation."]
    lines=["📊 *PREDICTION vs ACTUAL*"]
    for _,r in evaluation.iterrows():
        stock=str(r.get("Symbol",r.get("Stock","-")));bucket=_BUCKET_LABELS.get(str(r.get("PriceBucket","")),str(r.get("PriceBucket","-")));lines += [f"💎 *{stock}* | ₹ {bucket}"];rows=[]
        for kind,prefix in (("Predicted","Pred_"),("Actual","Actual_"),("Difference%","Diff_")):
            vals=[_fmt(r.get(f"{prefix}{field}")) for field in ("Open","High","Low","Close")]
            if kind=="Difference%":
                vals=[]
                for field in ("Open","High","Low","Close"):
                    d=_num(r.get(f"Diff_{field}"));a=_num(r.get(f"Actual_{field}"));vals.append(_pct((d/a*100) if d is not None and a not in (None,0) else d))
            rows.append([kind,*vals])
        lines += _table(["Type","Open","High","Low","Close"],rows)
    return lines
def evening_report(market_date,evaluation,metrics,retraining,**kwargs):
    metrics,retraining=metrics or {},retraining or {};bucket_metrics=kwargs.get("bucket_metrics",{}) or {};horizon_metrics=kwargs.get("horizon_metrics",{}) or {};learning=kwargs.get("learning",{}) or {};accuracy=kwargs.get("accuracy",{}) or {};scan=kwargs.get("scan",{}) or {};portfolio=kwargs.get("portfolio",{})
    lines=[f"🌙 *AI NSE EVENING REPORT*\n📅 {market_date}\n⚙️ {MODEL_VERSION}",_SECTION,*_evaluation_sections(evaluation),_SECTION,"📈 *MODEL ACCURACY*",f"Samples: {int(_num(metrics.get('Samples'),0) or 0)} | Overall MAPE: {_accuracy(metrics.get('OverallMAPE'))} | Close MAPE: {_accuracy(metrics.get('CloseMAPE'))}",f"Direction accuracy: {_accuracy(metrics.get('DirectionAccuracy'))}",_scan(scan)]
    if bucket_metrics:lines += [_SECTION,"📦 *BUCKET ACCURACY*",*_table(["Bucket","Accuracy"],[[str(k),_accuracy(v)] for k,v in bucket_metrics.items()])]
    if horizon_metrics:lines += [_SECTION,"🔮 *HORIZON ACCURACY*",*_table(["Horizon","N","Accuracy","Direction"],[[str(h),int(_num((m or {}).get('Samples'),0) or 0),_accuracy((m or {}).get('Accuracy')),_accuracy((m or {}).get('DirectionAccuracy'))] for h,m in horizon_metrics.items()])]
    lines += [_SECTION,"🧠 *MODEL LEARNING*",f"Retrained: {'YES' if retraining.get('Retrained',False) else 'NO'} | Decision: {str(retraining.get('Decision','-'))}",f"Improvement: {_pct(retraining.get('Improvement'))}",f"Previous accuracy: {_accuracy(accuracy.get('PreviousAccuracy'))} → Current accuracy: {_accuracy(accuracy.get('CurrentAccuracy'))}",f"Learning state: {learning.get('status','-')} | Health: {learning.get('health','-')} | Drift: {_pct(learning.get('drift'))}",f"Rollback: {'YES' if learning.get('rollback') else 'NO'}",_SECTION,*_portfolio(portfolio)]
    return "\n".join(lines)
def morning_report(prediction_date,cutoff_date,selected,jump_watchlist,intraday,**kwargs):
    accuracy,scan=kwargs.get("accuracy",{}),kwargs.get("scan",{});snapshot,regime=kwargs.get("market_snapshot",{}),kwargs.get("regime","-");portfolio,ipo=kwargs.get("portfolio",{}),kwargs.get("ipo",pd.DataFrame())
    lines=[f"📈 *AI NSE MORNING REPORT*\n📅 {prediction_date}\n⚙️ {MODEL_VERSION}\nData cutoff: {cutoff_date}",_SECTION,"📊 *MARKET OVERVIEW*",*_market(snapshot,regime),_scan(scan),(f"Accuracy: PENDING EVENING EVALUATION | Samples 0" if int(_num(accuracy.get("AccuracySamples",accuracy.get("Samples",0)),0) or 0)==0 else f"Accuracy: {_accuracy(accuracy.get('PreviousAccuracy'))} → {_accuracy(accuracy.get('CurrentAccuracy'))} | Samples {int(_num(accuracy.get('AccuracySamples',accuracy.get('Samples',0)),0) or 0)}"),_SECTION,*_bucket_sections(selected),_SECTION,*_best_pick_table(selected),_SECTION,*_prediction_table(selected),_SECTION,*_horizon_table(selected),_SECTION,*_jump(jump_watchlist),_SECTION,*_intraday(intraday),_SECTION,*_ipo(ipo),_SECTION,*_portfolio(portfolio)]
    return "\n".join(lines)
