"""Mobile-first Telegram reports for the final NSE AI pipeline.
Morning has exactly three numbered sections so the report stays easy to remember:
1) market + price-bucket stocks + IPO/new listings,
2) 7-session jump watch,
3) intraday screener.
"""
import os
import requests
import pandas as pd
from .config import TELEGRAM_MAX_LENGTH, MODEL_VERSION
from .utils import format_money, format_percent, split_messages

def send_telegram(text):
    token=os.getenv("TELEGRAM_BOT_TOKEN"); chat_id=os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id: print("Telegram secrets not configured."); return False
    url=f"https://api.telegram.org/bot{token}/sendMessage"; success=True
    for message in split_messages(text,TELEGRAM_MAX_LENGTH):
        try:
            r=requests.post(url,json={"chat_id":chat_id,"text":message,"parse_mode":"Markdown"},timeout=20)
            if r.status_code!=200: print("Telegram error:",r.text); success=False
        except Exception as exc: print("Telegram exception:",exc); success=False
    return success

def _table(headers,rows):
    if not rows:return ""
    all_rows=[headers]+[[str(x) for x in row] for row in rows]
    widths=[max(len(row[i]) for row in all_rows) for i in range(len(headers))]
    sep="  ".join("-"*w for w in widths)
    body=["  ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) for row in all_rows]
    return "\n".join([body[0],sep]+body[1:])

def _transpose_table(headers,rows):
    if not rows:return ""
    output=[[headers[1]]+[str(r[1]) for r in rows]]
    for i in range(2,len(headers)):output.append([headers[i]]+[str(r[i]) for r in rows])
    return _table(output[0],output[1:])

def _market_lines(snapshot,regime):
    n=snapshot.get("NIFTY",{}); b=snapshot.get("BANKNIFTY",{}); v=snapshot.get("VIX",{}); br=snapshot.get("Breadth",{})
    def q(x): return "-" if pd.isna(x) else f"{x:.2f}"
    return [f"NIFTY {q(n.get('Close',float('nan')))} ({q(n.get('Change1D',float('nan')))}%) | BANK NIFTY {q(b.get('Close',float('nan')))} ({q(b.get('Change1D',float('nan')))}%)",f"VIX {q(v.get('Close',float('nan')))} | Breadth {int(br.get('Advancers',0))}↑/{int(br.get('Decliners',0))}↓ | Regime {regime}"]

def _stock_rows(selected):
    rows=[]
    for i,(_,r) in enumerate(selected.iterrows(),1):
        rows.append([i,r.get("Symbol","-"),r.get("PriceBucket","-"),format_money(r.get("Current_Price",0)),format_money(r.get("Current_Open",0)),format_money(r.get("Current_High",0)),format_money(r.get("Current_Low",0)),format_money(r.get("Current_Close",r.get("Current_Price",0))),f"{float(r.get('Current_Volume',0)):,.0f}",f"{float(r.get('Pred_Close',0)):.2f}",f"{float(r.get('Confidence',0)):.0f}%",str(r.get("Action",r.get("Direction","-")))])
    return rows

def morning_report(prediction_date,cutoff_date,selected,jump_watchlist,intraday,**kwargs):
    snapshot=kwargs.get("market_snapshot",{}); regime=kwargs.get("regime","-"); ipo=kwargs.get("ipo")
    lines=["📈 *AI NSE MORNING REPORT*",f"_{prediction_date} | Data: {cutoff_date} | {MODEL_VERSION}_","","🎯 *1. MARKET + PRICE-BUCKET STOCKS + IPO*"]
    lines += _market_lines(snapshot,regime)+["","*Stocks — current OHLCV + AI forecast*"]
    if selected is not None and not selected.empty:
        lines += ["```",_transpose_table(["#","Stock","Bucket","CMP","Open","High","Low","Close","Volume","AI Close","Conf","Action"],_stock_rows(selected)),"```"]
    else: lines.append("No stock passed the quality gate today.")
    lines += ["","*Bucket coverage:* >₹1000 | ₹500–999 | ₹100–499 | ₹50–99 | ₹10–49"]
    if ipo is not None and not ipo.empty:
        lines.append("*IPO / NEW LISTINGS*")
        for _,r in ipo.head(8).iterrows():
            lines.append(f"• {r.get('IPOName','-')} | ₹{float(r.get('PriceHigh',0)):.0f} | GMP ₹{float(r.get('GMPValue',0)):.0f} ({float(r.get('GMPPct',0)):.1f}%) | Score {float(r.get('IPOScore',0)):.0f} | {r.get('IPOAction','WATCH')}")
        lines.append("GMP is unofficial; never treated as a guaranteed listing price.")
    else: lines.append("*IPO / NEW LISTINGS:* No verified live records available.")

    lines += ["","🔥 *2. +5% JUMP WATCH — NEXT 7 SESSIONS*"]
    if jump_watchlist is not None and not jump_watchlist.empty:
        rows=[]
        for i,(_,r) in enumerate(jump_watchlist.iterrows(),1):
            cp=float(r["Current_Price"]); target=float(r["Target_Level"]); rows.append([i,r["Symbol"],format_money(cp),format_money(target),format_percent((target/cp-1)*100 if cp else 0),format_percent(float(r.get("Estimated_7D_Upside",0))),f"{float(r.get('Jump_Probability',0)):.0f}%"])
        lines += ["```",_transpose_table(["#","Stock","CMP","+5% Target","Target%","7D Exp%","Prob"],rows),"```"]
    else: lines.append("No strong +5% candidate passed the gate.")

    lines += ["","⚡ *3. INTRADAY AI SCREENER*"]
    if intraday is not None and not intraday.empty:
        rows=[]
        for i,(_,r) in enumerate(intraday.iterrows(),1): rows.append([i,r.get("Symbol","-"),r.get("Bias","-"),format_money(r.get("Current",0)),format_money(r.get("Target",0)),format_money(r.get("StopLoss",0)),f"{float(r.get('Confidence',0)):.0f}%"])
        lines += ["```",_transpose_table(["#","Stock","Bias","CMP","Target","SL","Conf"],rows),"```"]
    else: lines.append("No confirmed intraday setup today; live conditions did not pass the gate.")
    return "\n".join(lines)

def evening_report(market_date,evaluation,metrics,retraining,**kwargs):
    bucket=kwargs.get("bucket_metrics",{}); intraday=kwargs.get("intraday_metrics",{}); ipo=kwargs.get("ipo_results"); learning=kwargs.get("learning",{})
    lines=["🌙 *AI NSE EVENING REPORT*",f"_{market_date} | {MODEL_VERSION}_","","📊 *1. ACTUAL MARKET + PREDICTION CHECK*"]
    if evaluation is not None and not evaluation.empty:
        rows=[]; dr=[]
        for _,r in evaluation.iterrows():
            rows += [[r["Symbol"],"PRED",f"{r['Pred_Open']:.2f}",f"{r['Pred_High']:.2f}",f"{r['Pred_Low']:.2f}",f"{r['Pred_Close']:.2f}"],["","ACT",f"{r['Actual_Open']:.2f}",f"{r['Actual_High']:.2f}",f"{r['Actual_Low']:.2f}",f"{r['Actual_Close']:.2f}"],["","DIFF",f"{r['Diff_Open']:+.2f}",f"{r['Diff_High']:+.2f}",f"{r['Diff_Low']:+.2f}",f"{r['Diff_Close']:+.2f}"]]
            dr.append([r["Symbol"],f"{r['Pred_Direction']} → {r['Actual_Direction']}","YES" if r["DirectionCorrect"] else "NO"])
        lines += ["```",_table(["Stock","Type","Open","High","Low","Close"],rows),"```","","Direction","```",_table(["Stock","Pred → Actual","Correct"],dr),"```"]
    else: lines.append("No predictions available for evaluation.")
    lines += ["","📦 *2. BUCKET + INTRADAY + IPO RESULTS*"]
    if bucket: lines.append("Buckets: "+" | ".join(f"{k} {v:.1f}%" for k,v in bucket.items()))
    else: lines.append("Buckets: not enough evaluated samples.")
    if intraday: lines.append("Intraday: "+" | ".join(f"{k} {v}" for k,v in intraday.items()))
    if ipo is not None and not ipo.empty: lines.append(f"IPO/new listings evaluated: {len(ipo)}")
    lines += ["","🧠 *3. STAGE 10 LEARNING*"]
    lines += ["```",_table(["Metric","Value"],[["Samples",metrics.get("Samples",0)],["Overall MAPE",f"{metrics.get('OverallMAPE',0):.3f}%"],["Close MAPE",f"{metrics.get('CloseMAPE',0):.3f}%"],["Direction Accuracy",f"{metrics.get('DirectionAccuracy',0):.1f}%"],["Champion/Challenger",retraining.get("Decision","-")],["Model Replaced","YES" if retraining.get("Retrained") else "NO"],["Improvement",f"{retraining.get('Improvement',0):+.2f}%"],["Learning State",learning.get("status","UPDATED")]]) ,"```"]
    return "\n".join(lines)
