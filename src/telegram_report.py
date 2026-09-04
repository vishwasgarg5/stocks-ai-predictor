"""Professional mobile-first multi-table Telegram reports for Stage 10.1."""
import os
import requests
import pandas as pd
from .config import TELEGRAM_MAX_LENGTH, MODEL_VERSION, PRICE_BUCKET_NAMES
from .utils import format_money, split_messages

_BUCKET_MESSAGE="\n§§BUCKET_MESSAGE§§\n"

def send_telegram(text):
    token=os.getenv("TELEGRAM_BOT_TOKEN");chat_id=os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram secrets not configured.");return False
    url=f"https://api.telegram.org/bot{token}/sendMessage"
    parts=text.split(_BUCKET_MESSAGE) if _BUCKET_MESSAGE in text else [text]
    success=True
    for part in parts:
        if not part.strip():continue
        for message in split_messages(part.strip(),TELEGRAM_MAX_LENGTH):
            try:
                r=requests.post(url,json={"chat_id":chat_id,"text":message,"parse_mode":"Markdown"},timeout=20)
                if r.status_code!=200:print("Telegram error:",r.text);success=False
            except Exception as exc:print("Telegram exception:",exc);success=False
    return success

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
    lines=["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]
    for row in rows:lines.append("| "+" | ".join(str(x).replace("|","/") for x in row)+" |")
    return lines

def _market(snapshot,regime):
    snapshot=snapshot or {};n=snapshot.get("NIFTY",{});b=snapshot.get("BANKNIFTY",{});f=snapshot.get("FINNIFTY",{});m=snapshot.get("MIDCPNIFTY",{});v=snapshot.get("VIX",{});br=snapshot.get("Breadth",{})
    rows=[
        ["NIFTY",_fmt(n.get("Close")),_pct(n.get("Change1D"))],
        ["BANKNIFTY",_fmt(b.get("Close")),_pct(b.get("Change1D"))],
        ["FINNIFTY",_fmt(f.get("Close")),_pct(f.get("Change1D"))],
        ["MIDCPNIFTY",_fmt(m.get("Close")),_pct(m.get("Change1D"))],
        ["VIX",_fmt(v.get("Close")),"-"]]
    return _table(["Index","Value","1D%"],rows)+[f"Breadth: {int(br.get('Advancers',0))}↑ / {int(br.get('Decliners',0))}↓  •  Regime: {regime}"]

def _scan(scan):
    return f"🔎 {int(scan.get('Universe',0)):,} scanned • {int(scan.get('Data',0)):,} data • {int(scan.get('Liquid',0)):,} liquid • {int(scan.get('AI',0)):,} AI • {int(scan.get('Selected',0)):,} selected"

def _stock_rows(group):
    rows=[]
    for _,r in group.iterrows():
        cmp=float(r.get("Current_Price",r.get("Current_Close",0)) or 0);pred=float(r.get("Pred_Close",0) or 0)
        exp=((pred/cmp)-1)*100 if cmp else 0
        score=r.get("FinalScore",r.get("Score",r.get("TradeConfidence",0)))
        rows.append([f"*{r.get('Symbol','-')}*",f"₹{_fmt(cmp)}",_pct(exp),_pct(r.get("Horizon_1D")),_pct(r.get("Horizon_5D")),_pct(r.get("Horizon_20D")),_fmt(score,0)])
    return rows

def _bucket_sections(selected,warning=False,title="TOP STOCKS"):
    if selected is None or selected.empty:return ["No qualifying stock today."]
    lines=[f"🎯 *{title}*  |  MULTIPLE PRICE-BUCKET TABLES"]
    buckets=list(PRICE_BUCKET_NAMES)+[b for b in selected.get("PriceBucket",pd.Series(dtype=str)).astype(str).unique() if b not in PRICE_BUCKET_NAMES]
    for bucket in buckets:
        if "PriceBucket" not in selected.columns:continue
        group=selected[selected["PriceBucket"].astype(str)==bucket].copy()
        if group.empty:continue
        sort_cols=[c for c in ["TradeConfidence","Score","Confidence"] if c in group.columns]
        if sort_cols:group=group.sort_values(sort_cols,ascending=False)
        label=str(group.iloc[0].get("PriceBucketLabel",bucket));label=bucket if label in {"nan","None","-"} else label
        lines += [_BUCKET_MESSAGE,f"💎 *{label}*"]
        lines += _table(["Stock","CMP","Exp%","1D","5D","20D","Score"],_stock_rows(group.head(5)))
        lines.append("")
    return lines

def _prediction_table(selected):
    if selected is None or selected.empty:return []
    rows=[]
    for _,r in selected.head(25).iterrows():
        rows.append([f"*{r.get('Symbol','-')}*",_fmt(r.get("Pred_Open")),_fmt(r.get("Pred_High")),_fmt(r.get("Pred_Low")),_fmt(r.get("Pred_Close")),_fmt(r.get("Pred_Volume"),0),_pct(r.get("Expected_Return"))])
    return ["📈 *PREDICTED OHLCV — SELECTED STOCKS*"]+_table(["Stock","Open","High","Low","Close","Volume","Exp%"],rows)

def _horizon_table(selected):
    if selected is None or selected.empty:return []
    rows=[]
    for _,r in selected.head(25).iterrows():
        rows.append([f"*{r.get('Symbol','-')}*"]+[_pct(r.get(f"Horizon_{h}D")) for h in (1,3,5,7,20)])
    return ["🔮 *MULTI-HORIZON OUTLOOK*"]+_table(["Stock","1D","3D","5D","7D","20D"],rows)

def _portfolio(p):
    if not p:return []
    lines=["💼 *PORTFOLIO*",f"Positions: {p.get('Positions',0)}  •  Value: ₹{p.get('Value',0):,.0f}  •  P&L: ₹{p.get('PnL',0):+,.0f} ({p.get('Return',0):+.2f}%)"]
    rows=[]
    for x in p.get("Rows",[]):
        if isinstance(x,dict):rows.append([x.get("Stock","-"),x.get("Quantity","-"),_fmt(x.get("Average_Price")),_fmt(x.get("Current_Price")),_pct(x.get("Return_Pct")),x.get("AI_Action",x.get("Action","-"))])
        else:lines.append(f"• {x}")
    if rows:lines += _table(["Stock","Qty","Avg","CMP","P/L%","Action"],rows)
    return lines

def _ipo(ipo):
    if ipo is None or (hasattr(ipo,"empty") and ipo.empty):return ["🏦 *IPO*  |  No active/upcoming IPOs found"]
    rows=[]
    for _,r in ipo.head(8).iterrows():rows.append([f"*{r.get('IPOName','-')}*",f"₹{_fmt(r.get('PriceHigh',0),0)}",f"₹{_fmt(r.get('GMPValue',0),0)}",_pct(r.get('GMPPct',0)),_fmt(r.get('IPOScore',0),0),r.get('IPOAction','WATCH')])
    return ["🏦 *IPO INTELLIGENCE*"]+_table(["IPO","Price","GMP","GMP%","Score","Action"],rows)

def _jump(jump_watchlist):
    if jump_watchlist is None or jump_watchlist.empty:return []
    rows=[]
    for _,r in jump_watchlist.head(5).iterrows():
        cp=float(r.get("Current_Price",0) or 0);target=float(r.get("Target_Level",0) or 0);rows.append([f"*{r.get('Symbol','-')}*",f"₹{_fmt(cp)}",f"₹{_fmt(target)}",_pct((target/cp-1)*100 if cp else 0),f"{float(r.get('Jump_Probability',0) or 0):.0f}%"])
    return ["🔥 *JUMP WATCH — TOP 5*"]+_table(["Stock","CMP","Target","Upside","Prob."],rows)

def _intraday(intraday):
    if intraday is None or intraday.empty:return []
    rows=[]
    for _,r in intraday.head(5).iterrows():rows.append([f"*{r.get('Symbol','-')}*",r.get('Bias','-'),f"₹{_fmt(r.get('Current'))}",f"₹{_fmt(r.get('Target'))}",f"₹{_fmt(r.get('StopLoss'))}",f"{float(r.get('Confidence',0) or 0):.0f}%"])
    return ["⚡ *INTRADAY — TOP 5*"]+_table(["Stock","Bias","CMP","Target","SL","Conf."],rows)

def morning_report(prediction_date,cutoff_date,selected,jump_watchlist,intraday,**kwargs):
    accuracy=kwargs.get("accuracy",{});scan=kwargs.get("scan",{});portfolio=kwargs.get("portfolio",{});snapshot=kwargs.get("market_snapshot",{});regime=kwargs.get("regime","-");ipo=kwargs.get("ipo",pd.DataFrame())
    lines=["📈 *AI NSE MORNING REPORT*",f"📅 Prediction: *{prediction_date}*",f"⚙️ {MODEL_VERSION}",_scan(scan),f"🤖 Accuracy {_accuracy(accuracy.get('PreviousAccuracy'))} → {_accuracy(accuracy.get('CurrentAccuracy'))}  •  {int(accuracy.get('AccuracySamples',accuracy.get('Samples',0)) or 0)} validated","","📊 *MARKET OVERVIEW*"]+_market(snapshot,regime)
    lines += _bucket_sections(selected,title="TOP STOCKS BY PRICE BUCKET")
    lines += _prediction_table(selected)+[""]+_horizon_table(selected)+[""]+_jump(jump_watchlist)+[""]+_intraday(intraday)+[""]+_ipo(ipo)+[""]+_portfolio(portfolio)
    return "\n".join(lines)

def _evening_bucket_sections(evaluation):
    if evaluation is None or evaluation.empty:return ["No predictions available for evaluation."]
    lines=["📋 *PREDICTION vs ACTUAL — BY PRICE BUCKET*"]
    buckets=list(PRICE_BUCKET_NAMES)+[b for b in evaluation.get("PriceBucket",pd.Series(dtype=str)).astype(str).unique() if b not in PRICE_BUCKET_NAMES]
    for bucket in buckets:
        if "PriceBucket" not in evaluation.columns:continue
        group=evaluation[evaluation["PriceBucket"].astype(str)==bucket].copy()
        if group.empty:continue
        rows=[]
        for _,r in group.sort_values("APE_Close",key=lambda s:s.abs()).head(5).iterrows():
            ok="✅" if bool(r.get("DirectionCorrect",False)) else "❌"
            rows.append([f"*{r.get('Symbol','-')}* {ok}",f"O {_fmt(r.get('Pred_Open'))}/ {_fmt(r.get('Actual_Open'))}",f"H {_fmt(r.get('Pred_High'))}/ {_fmt(r.get('Actual_High'))}",f"L {_fmt(r.get('Pred_Low'))}/ {_fmt(r.get('Actual_Low'))}",f"C {_fmt(r.get('Pred_Close'))}/ {_fmt(r.get('Actual_Close'))}",f"{abs(float(r.get('APE_Close',0) or 0)):.2f}%"])
        lines += [_BUCKET_MESSAGE,f"💎 *{bucket}*",*_table(["Stock","Open P/A","High P/A","Low P/A","Close P/A","Close APE"],rows),""]
    return lines

def _evening_accuracy_table(bucket):
    if not bucket:return ["No sufficient bucket sample"]
    rows=[[k,_accuracy(v)] for k,v in bucket.items()]
    return _table(["Price Bucket","Accuracy"],rows)

def _evening_horizon_table(horizon):
    if not horizon:return ["No horizon target matured yet"]
    rows=[[f"{k}D",f"{v.get('Accuracy',0):.1f}%",v.get('Samples',0)] for k,v in sorted(horizon.items(),key=lambda x:int(x[0]))]
    return _table(["Horizon","Accuracy","Samples"],rows)

def evening_report(market_date,evaluation,metrics,retraining,**kwargs):
    accuracy=kwargs.get("accuracy",{});scan=kwargs.get("scan",{});bucket=kwargs.get("bucket_metrics",{});portfolio=kwargs.get("portfolio",{});learning=kwargs.get("learning",{});horizon=kwargs.get("horizon_metrics",{})
    lines=["🌙 *AI NSE EVENING REPORT*",f"📅 Market: *{market_date}*",f"⚙️ {MODEL_VERSION}",_scan(scan),f"🤖 Accuracy {_accuracy(accuracy.get('PreviousAccuracy'))} → {_accuracy(accuracy.get('CurrentAccuracy'))}",""]+_evening_bucket_sections(evaluation)+["📊 *BUCKET ACCURACY*"]+_evening_accuracy_table(bucket)+["","🎯 *HORIZON ACCURACY*"]+_evening_horizon_table(horizon)+["","🧠 *MODEL LEARNING*"]+_table(["Metric","Value"],[["Samples",metrics.get('Samples',0)],["Overall MAPE",f"{metrics.get('OverallMAPE',0):.3f}%"],["Close MAPE",f"{metrics.get('CloseMAPE',0):.3f}%"],["Direction Accuracy",f"{metrics.get('DirectionAccuracy',0):.1f}%"],["Champion/Challenger",retraining.get('Decision','-')],["Model Replaced","YES" if retraining.get('Retrained') else "NO"],["Improvement",f"{retraining.get('Improvement',0):+.2f}%"],["Learning State",learning.get('status','UPDATED')]])
    if portfolio:lines += ["",*_portfolio(portfolio)]
    return "\n".join(lines)
