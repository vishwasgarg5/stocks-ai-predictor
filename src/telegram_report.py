"""Mobile-first Telegram reports with deterministic section/message ordering."""
import os
import requests
import pandas as pd
from .config import TELEGRAM_MAX_LENGTH, MODEL_VERSION, PRICE_BUCKET_NAMES
from .utils import split_messages

_SECTION="\n§§TELEGRAM_SECTION§§\n"
_BUCKET_MESSAGE=_SECTION

def _send_part(part, token, chat_id):
    ok=True
    for message in split_messages(part.strip(), TELEGRAM_MAX_LENGTH):
        try:
            r=requests.post(f"https://api.telegram.org/bot{token}/sendMessage",json={"chat_id":chat_id,"text":message,"parse_mode":"Markdown"},timeout=20)
            if r.status_code!=200: print("Telegram error:",r.text); ok=False
        except Exception as exc: print("Telegram exception:",exc); ok=False
    return ok

def send_telegram(text):
    token=os.getenv("TELEGRAM_BOT_TOKEN"); chat_id=os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id: print("Telegram secrets not configured."); return False
    # Every section is deliberately separated before sending. This preserves order on Telegram.
    parts=[p for p in text.split(_SECTION) if p.strip()]
    return all(_send_part(p,token,chat_id) for p in parts)

def _fmt(v,digits=2):
    try:return f"{float(v):,.{digits}f}"
    except (TypeError,ValueError):return "-"

def _pct(v):
    try:return f"{float(v):+.1f}%"
    except (TypeError,ValueError):return "-"

def _accuracy(v):
    try:return f"{float(v):.1f}%"
    except (TypeError,ValueError):return "-"

def _table(headers,rows):
    if not rows:return []
    out=["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]
    out += ["| "+" | ".join(str(x).replace("|","/") for x in row)+" |" for row in rows]
    return out

def _ordered_price_buckets(values):
    configured=list(PRICE_BUCKET_NAMES); present=[] if values is None else [str(x) for x in pd.Series(values).dropna().unique()]
    return configured+[x for x in present if x not in configured]

def _market(snapshot,regime):
    snapshot=snapshot or {}; rows=[]
    for name,key in [("NIFTY","NIFTY"),("BANKNIFTY","BANKNIFTY"),("FINNIFTY","FINNIFTY"),("MIDCPNIFTY","MIDCPNIFTY")]:
        x=snapshot.get(key,{}) ; rows.append([name,_fmt(x.get("Close")),_pct(x.get("Change1D"))])
    x=snapshot.get("VIX",{}); rows.append(["VIX",_fmt(x.get("Close")),"-"])
    br=snapshot.get("Breadth",{}); return _table(["Index","Value","1D%"],rows)+[f"Breadth: {int(br.get('Advancers',0))}↑ / {int(br.get('Decliners',0))}↓  •  Regime: {regime}"]

def _scan(scan):return f"🔎 {int(scan.get('Universe',0)):,} scanned • {int(scan.get('Data',0)):,} data • {int(scan.get('Liquid',0)):,} liquid • {int(scan.get('AI',0)):,} AI • {int(scan.get('Selected',0)):,} qualified"

def _score_col(g):
    for c in ["FinalDecisionScore","TradeConfidence","Score"]:
        if c in g.columns:return c
    return None

def _stock_rows(g):
    rows=[]
    for _,r in g.iterrows():
        cmp=float(r.get("Current_Price",r.get("Current_Close",0)) or 0); pred=float(r.get("Pred_Close",0) or 0); exp=(pred/cmp-1)*100 if cmp else 0
        score=r.get("FinalDecisionScore",r.get("FinalScore",r.get("Score",r.get("TradeConfidence",0))))
        rows.append([f"*{r.get('Symbol','-')}*",f"₹{_fmt(cmp)}",_pct(exp),_pct(r.get("Horizon_1D")),_pct(r.get("Horizon_5D")),_pct(r.get("Horizon_20D")),_fmt(score,0),str(r.get("Action","-"))])
    return rows

def _sort(g):
    c=_score_col(g)
    return g.sort_values(c,ascending=False,kind="mergesort") if c else g.sort_values("Symbol",kind="mergesort")

def _bucket_sections(selected):
    if selected is None or selected.empty:return ["No qualifying stock today."]
    if "PriceBucket" not in selected.columns:return ["No price bucket data."]
    lines=["🎯 *QUALIFIED STOCKS BY PRICE BUCKET*"]
    for bucket in _ordered_price_buckets(selected["PriceBucket"]):
        g=selected[selected["PriceBucket"].astype(str)==bucket].copy()
        if g.empty:continue
        g=_sort(g).head(6)
        lines += [_SECTION,f"💎 *₹ {bucket}*",*_table(["Stock","CMP","Exp%","1D","5D","20D","Score","Action"],_stock_rows(g))]
    return lines

def _best_pick_table(selected):
    if selected is None or selected.empty or "PriceBucket" not in selected.columns:return []
    rows=[]
    for bucket in _ordered_price_buckets(selected["PriceBucket"]):
        g=selected[selected["PriceBucket"].astype(str)==bucket].copy()
        if g.empty:continue
        r=_sort(g).iloc[0]; rows.append([f"*₹ {bucket}*",f"*{r.get('Symbol','-')}*",f"₹{_fmt(r.get('Current_Price',r.get('Current_Close')))}",_pct(r.get('Expected_Return')),f"{float(r.get('FinalDecisionScore',r.get('Score',0))):.0f}",str(r.get('Action','-')),f"{float(r.get('CalibratedConfidence',r.get('Confidence',0))):.0f}%"])
    return ["🏆 *BEST PICK — ONE PER PRICE BUCKET*"]+_table(["Bucket","Stock","CMP","Exp%","Score","Action","Conf."],rows)

def _prediction_table(selected):
    if selected is None or selected.empty:return []
    rows=[[f"*{r.get('Symbol','-')}*",_fmt(r.get("Pred_Open")),_fmt(r.get("Pred_High")),_fmt(r.get("Pred_Low")),_fmt(r.get("Pred_Close")),_fmt(r.get("Pred_Volume"),0),_pct(r.get("Expected_Return"))] for _,r in selected.iterrows()]
    return ["📈 *PREDICTED OHLCV*"]+_table(["Stock","Open","High","Low","Close","Volume","Exp%"],rows)

def _horizon_table(selected):
    if selected is None or selected.empty:return []
    rows=[[f"*{r.get('Symbol','-')}*"]+[_pct(r.get(f"Horizon_{h}D")) for h in (1,3,5,7,20)] for _,r in selected.iterrows()]
    return ["🔮 *MULTI-HORIZON OUTLOOK*"]+_table(["Stock","1D","3D","5D","7D","20D"],rows)

def _jump(j):
    if j is None or j.empty:return []
    rows=[]
    for _,r in j.iterrows():
        cp=float(r.get("Current_Price",0) or 0);t=float(r.get("Target_Level",0) or 0);rows.append([f"*{r.get('Symbol','-')}*",f"₹{_fmt(cp)}",f"₹{_fmt(t)}",_pct((t/cp-1)*100 if cp else 0),f"{float(r.get('Jump_Probability',0) or 0):.0f}%"])
    return ["🔥 *JUMP WATCH*"]+_table(["Stock","CMP","Target","Upside","Prob."],rows)

def _intraday(x):
    if x is None or x.empty:return []
    rows=[[f"*{r.get('Symbol','-')}*",r.get('Bias','-'),f"₹{_fmt(r.get('Current'))}",f"₹{_fmt(r.get('Target'))}",f"₹{_fmt(r.get('StopLoss'))}",f"{float(r.get('Confidence',0) or 0):.0f}%"] for _,r in x.iterrows()]
    return ["⚡ *INTRADAY*"]+_table(["Stock","Bias","CMP","Target","SL","Conf."],rows)

def _ipo(x):
    if x is None or (hasattr(x,"empty") and x.empty):return ["🏦 *IPO*  |  No active/upcoming IPOs found"]
    rows=[[f"*{r.get('IPOName','-')}*",f"₹{_fmt(r.get('PriceHigh',0),0)}",f"₹{_fmt(r.get('GMPValue',0),0)}",_pct(r.get('GMPPct',0)),_fmt(r.get('IPOScore',0),0),r.get('IPOAction','WATCH')] for _,r in x.iterrows()]
    return ["🏦 *IPO INTELLIGENCE*"]+_table(["IPO","Price","GMP","GMP%","Score","Action"],rows)

def _portfolio(p):
    if not p:return []
    lines=["💼 *AI PORTFOLIO MANAGER*",f"Positions: {p.get('Positions',0)} • Value: ₹{p.get('Value',0):,.0f} • P&L: ₹{p.get('PnL',0):+,.0f} ({p.get('Return',0):+.2f}%)"]
    rows=[]
    for x in p.get("Rows",[]):
        if isinstance(x,dict):rows.append([x.get("Stock","-"),x.get("Quantity","-"),x.get("Decision","-"),x.get("Current_Price","-"),x.get("Average_Price","-"),x.get("Profit_Target","-"),x.get("Sell_Window","-")])
        else:lines.append(f"• {x}")
    if rows:lines += ["📌 *DAILY PORTFOLIO ACTION*",*_table(["Stock","Qty","Decision","CMP","Avg","10% Target","Sell Window"],rows)]
    if p.get("AveragePlans"):
        lines += ["➕ *AVERAGING PLANS*",*_table(["Stock","Add Qty","CMP","New Avg","Target","Sell Window"],[[x.get("Stock","-"),x.get("Recommended_Qty",0),x.get("Current_Price","-"),x.get("New_Average_Price","-"),x.get("Profit_Target","-"),x.get("Sell_Window","-")] for x in p["AveragePlans"]])]
    if p.get("SellAlerts"):
        lines += ["🚨 *SELL / PROFIT-BOOK ALERTS*",*_table(["Stock","CMP","Target","When","Reason"],[[x.get("Stock","-"),x.get("Current_Price","-"),x.get("Profit_Target","-"),x.get("Sell_Window","NOW"),x.get("Reason","-")] for x in p["SellAlerts"]])]
    return lines

def morning_report(prediction_date,cutoff_date,selected,jump_watchlist,intraday,**kwargs):
    accuracy=kwargs.get("accuracy",{});scan=kwargs.get("scan",{});portfolio=kwargs.get("portfolio",{});snapshot=kwargs.get("market_snapshot",{});regime=kwargs.get("regime","-");ipo=kwargs.get("ipo",pd.DataFrame())
    lines=[f"📈 *AI NSE MORNING REPORT*\n📅 Prediction: *{prediction_date}*\n⚙️ {MODEL_VERSION}",_SECTION,_scan(scan),f"🤖 Accuracy {_accuracy(accuracy.get('PreviousAccuracy'))} → {_accuracy(accuracy.get('CurrentAccuracy'))} • {int(accuracy.get('AccuracySamples',accuracy.get('Samples',0)) or 0)} validated","📊 *MARKET OVERVIEW*",*_market(snapshot,regime),_SECTION,*_bucket_sections(selected),_SECTION,*_best_pick_table(selected),_SECTION,*_prediction_table(selected),_SECTION,*_horizon_table(selected),_SECTION,*_jump(jump_watchlist),_SECTION,*_intraday(intraday),_SECTION,*_ipo(ipo),_SECTION,*_portfolio(portfolio)]
    return "\n".join(lines)

def _evening_bucket_sections(evaluation):
    if evaluation is None or evaluation.empty:return ["No predictions available for evaluation."]
    if "PriceBucket" not in evaluation.columns:return ["No price bucket data."]
    lines=["📋 *PREDICTION vs ACTUAL — BY PRICE BUCKET*"]
    for bucket in _ordered_price_buckets(evaluation["PriceBucket"]):
        g=evaluation[evaluation["PriceBucket"].astype(str)==bucket].copy()
        if g.empty:continue
        rows=[]
        for _,r in g.sort_values("APE_Close",key=lambda s:s.abs(),kind="mergesort").head(6).iterrows():
            ok="✅" if bool(r.get("DirectionCorrect",False)) else "❌";rows.append([f"*{r.get('Symbol','-')}* {ok}",f"O {_fmt(r.get('Pred_Open'))}/ {_fmt(r.get('Actual_Open'))}",f"H {_fmt(r.get('Pred_High'))}/ {_fmt(r.get('Actual_High'))}",f"L {_fmt(r.get('Pred_Low'))}/ {_fmt(r.get('Actual_Low'))}",f"C {_fmt(r.get('Pred_Close'))}/ {_fmt(r.get('Actual_Close'))}",f"{abs(float(r.get('APE_Close',0) or 0)):.2f}%"])
        lines += [_SECTION,f"💎 *₹ {bucket}*",*_table(["Stock","Open P/A","High P/A","Low P/A","Close P/A","Close APE"],rows)]
    return lines

def _evening_accuracy_table(bucket):
    return ["No sufficient bucket sample"] if not bucket else _table(["Price Bucket","Accuracy"],[[k,_accuracy(v)] for k,v in bucket.items()])

def _evening_horizon_table(h):
    return ["No horizon target matured yet"] if not h else _table(["Horizon","Accuracy","Samples"],[[f"{k}D",f"{v.get('Accuracy',0):.1f}%",v.get('Samples',0)] for k,v in sorted(h.items(),key=lambda x:int(x[0]))])

def evening_report(market_date,evaluation,metrics,retraining,**kwargs):
    accuracy=kwargs.get("accuracy",{});scan=kwargs.get("scan",{});bucket=kwargs.get("bucket_metrics",{});portfolio=kwargs.get("portfolio",{});learning=kwargs.get("learning",{});horizon=kwargs.get("horizon_metrics",{})
    lines=[f"🌙 *AI NSE EVENING REPORT*\n📅 Market: *{market_date}*\n⚙️ {MODEL_VERSION}",_SECTION,_scan(scan),f"🤖 Accuracy {_accuracy(accuracy.get('PreviousAccuracy'))} → {_accuracy(accuracy.get('CurrentAccuracy'))}",*_evening_bucket_sections(evaluation),_SECTION,"📊 *BUCKET ACCURACY*",*_evening_accuracy_table(bucket),_SECTION,"🎯 *HORIZON ACCURACY*",*_evening_horizon_table(horizon),_SECTION,"🧠 *MODEL LEARNING*",*_table(["Metric","Value"],[["Samples",metrics.get('Samples',0)],["Overall MAPE",f"{metrics.get('OverallMAPE',0):.3f}%"],["Close MAPE",f"{metrics.get('CloseMAPE',0):.3f}%"],["Direction Accuracy",f"{metrics.get('DirectionAccuracy',0):.1f}%"],["Champion/Challenger",retraining.get('Decision','-')],["Model Replaced","YES" if retraining.get('Retrained') else "NO"],["Improvement",f"{retraining.get('Improvement',0):+.2f}%"],["Learning State",learning.get('status','UPDATED')]]),_SECTION,*_portfolio(portfolio)]
    return "\n".join(lines)
