"""Mobile-first Telegram reports for the final NSE AI pipeline."""
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
    return _table([headers[0]]+[str(r[0]) for r in rows],[[headers[i]]+[str(r[i]) for r in rows] for i in range(1,len(headers))])

def _fmt_accuracy(v): return "-" if v is None or pd.isna(v) else f"{float(v):.1f}%"

def _accuracy_lines(accuracy):
    return [f"🤖 Accuracy: Previous {_fmt_accuracy(accuracy.get('PreviousAccuracy'))} → Current {_fmt_accuracy(accuracy.get('CurrentAccuracy'))} | Validated {int(accuracy.get('AccuracySamples',accuracy.get('Samples',0)) or 0)} samples",
            f"📏 Horizons: 1D {_fmt_accuracy(accuracy.get('1D'))} | 3D {_fmt_accuracy(accuracy.get('3D'))} | 5D {_fmt_accuracy(accuracy.get('5D'))} | 7D {_fmt_accuracy(accuracy.get('7D'))} | 20D {_fmt_accuracy(accuracy.get('20D'))} | Direction {_fmt_accuracy(accuracy.get('DirectionAccuracy'))}",
            f"🧠 Health: {accuracy.get('Health','-')} | Drift {accuracy.get('Drift','-')} | 7D trend {_fmt_accuracy(accuracy.get('Trend7D'))} | 30D trend {_fmt_accuracy(accuracy.get('Trend30D'))}"]

def _market_lines(snapshot,regime):
    n=snapshot.get("NIFTY",{}); b=snapshot.get("BANKNIFTY",{}); v=snapshot.get("VIX",{}); br=snapshot.get("Breadth",{})
    def q(x): return "-" if x is None or pd.isna(x) else f"{float(x):.2f}"
    return [f"NIFTY {q(n.get('Close'))} ({q(n.get('Change1D'))}%) | BANK NIFTY {q(b.get('Close'))} ({q(b.get('Change1D'))}%)",f"VIX {q(v.get('Close'))} | Breadth {int(br.get('Advancers',0))}↑/{int(br.get('Decliners',0))}↓ | Regime {regime}"]

def _stock_rows(selected):
    rows=[]
    for i,(_,r) in enumerate(selected.iterrows(),1):
        rows.append([i,r.get("Symbol","-"),r.get("PriceBucket","-"),format_money(r.get("Current_Price",0)),format_money(r.get("Current_Open",0)),format_money(r.get("Current_High",0)),format_money(r.get("Current_Low",0)),format_money(r.get("Current_Close",r.get("Current_Price",0))),f"{float(r.get('Current_Volume',0)):,.0f}",f"{float(r.get('Pred_Close',0)):.2f}",f"{float(r.get('Confidence',0)):.0f}%",str(r.get("Action",r.get("Direction","-")))])
    return rows

def _scan_lines(scan):
    return [f"🔎 Scan funnel: Universe {int(scan.get('Universe',0)):,} → Data {int(scan.get('Data',0)):,} → Liquid {int(scan.get('Liquid',0)):,} → AI {int(scan.get('AI',0)):,} → Selected {int(scan.get('Selected',0)):,}"]

def _portfolio_lines(portfolio):
    if not portfolio: return ["💼 Portfolio Manager: no portfolio CSV / no verified prices."]
    lines=[f"💼 *PORTFOLIO MANAGER* | Positions {portfolio.get('Positions',0)} | Value ₹{portfolio.get('Value',0):,.0f} | P&L ₹{portfolio.get('PnL',0):+,.0f} ({portfolio.get('Return',0):+.2f}%)"]
    for item in portfolio.get("Rows",[]): lines.append(f"• {item}")
    return lines

def morning_report(prediction_date,cutoff_date,selected,jump_watchlist,intraday,**kwargs):
    snapshot=kwargs.get("market_snapshot",{}); regime=kwargs.get("regime","-"); ipo=kwargs.get("ipo"); accuracy=kwargs.get("accuracy",{}); scan=kwargs.get("scan",{}); portfolio=kwargs.get("portfolio",{})
    lines=["📈 *AI NSE MORNING REPORT*",f"_{prediction_date} | Data: {cutoff_date} | {MODEL_VERSION}_"]
    lines += _scan_lines(scan)+_accuracy_lines(accuracy)
    lines += ["","🎯 *1. TOP STOCKS*"]+_market_lines(snapshot,regime)+["","*Price buckets — current OHLCV + AI forecast*"]
    if selected is not None and not selected.empty: lines += ["```",_transpose_table(["#","Stock","Bucket","CMP","Open","High","Low","Close","Volume","AI Close","Conf","Action"],_stock_rows(selected)),"```"]
    else: lines.append("No stock passed the quality gate today.")
    lines += ["","*Bucket coverage:* >₹1000 | ₹500–999 | ₹100–499 | ₹50–99 | ₹10–49"]
    if ipo is not None and not ipo.empty:
        lines.append("*IPO / NEW LISTINGS*")
        for _,r in ipo.head(8).iterrows(): lines.append(f"• {r.get('IPOName','-')} | ₹{float(r.get('PriceHigh',0)):.0f} | GMP ₹{float(r.get('GMPValue',0)):.0f} ({float(r.get('GMPPct',0)):.1f}%) | Score {float(r.get('IPOScore',0)):.0f} | {r.get('IPOAction','WATCH')}")
        lines.append("GMP is unofficial; never treated as a guaranteed listing price.")
    else: lines.append("*IPO / NEW LISTINGS:* No verified live records available.")
    lines += _portfolio_lines(portfolio)
    lines += ["","🔥 *2. +5% JUMP WATCH*"]
    if jump_watchlist is not None and not jump_watchlist.empty:
        rows=[]
        for i,(_,r) in enumerate(jump_watchlist.iterrows(),1):
            cp=float(r["Current_Price"]); target=float(r["Target_Level"]); rows.append([i,r["Symbol"],format_money(cp),format_money(target),format_percent((target/cp-1)*100 if cp else 0),format_percent(float(r.get("Estimated_7D_Upside",0))),f"{float(r.get('Jump_Probability',0)):.0f}%"])
        lines += ["```",_transpose_table(["#","Stock","CMP","+5% Target","Target%","7D Exp%","Prob"],rows),"```"]
    else: lines.append("No strong +5% candidate passed the gate.")
    lines += ["","⚡ *3. INTRADAY STOCKS*"]
    if intraday is not None and not intraday.empty:
        rows=[]
        for i,(_,r) in enumerate(intraday.iterrows(),1): rows.append([i,r.get("Symbol","-"),r.get("Bias","-"),format_money(r.get("Current",0)),format_money(r.get("Target",0)),format_money(r.get("StopLoss",0)),f"{float(r.get('Confidence',0)):.0f}%"])
        lines += ["```",_transpose_table(["#","Stock","Bias","CMP","Target","SL","Conf"],rows),"```"]
    else: lines.append("No confirmed intraday setup today; live conditions did not pass the gate.")
    return "\n".join(lines)

def evening_report(market_date,evaluation,metrics,retraining,**kwargs):
    bucket=kwargs.get("bucket_metrics",{}); intraday=kwargs.get("intraday_metrics",{}); ipo=kwargs.get("ipo_results"); learning=kwargs.get("learning",{}); accuracy=kwargs.get("accuracy",{}); scan=kwargs.get("scan",{}); portfolio=kwargs.get("portfolio",{})
    lines=["🌙 *AI NSE EVENING REPORT*",f"_{market_date} | {MODEL_VERSION}_"]+_scan_lines(scan)+_accuracy_lines(accuracy)
    lines += ["","🎯 *1. TOP STOCKS — PREDICTION CHECK*"]
    if evaluation is not None and not evaluation.empty:
        rows=[]; dr=[]
        for _,r in evaluation.iterrows():
            rows += [[r["Symbol"],"PRED",f"{r['Pred_Open']:.2f}",f"{r['Pred_High']:.2f}",f"{r['Pred_Low']:.2f}",f"{r['Pred_Close']:.2f}"],["","ACT",f"{r['Actual_Open']:.2f}",f"{r['Actual_High']:.2f}",f"{r['Actual_Low']:.2f}",f"{r['Actual_Close']:.2f}"],["","DIFF",f"{r['Diff_Open']:+.2f}",f"{r['Diff_High']:+.2f}",f"{r['Diff_Low']:+.2f}",f"{r['Diff_Close']:+.2f}"]]
            dr.append([r["Symbol"],f"{r['Pred_Direction']} → {r['Actual_Direction']}","YES" if r["DirectionCorrect"] else "NO"])
        lines += ["```",_table(["Stock","Type","Open","High","Low","Close"],rows),"```","","Direction","```",_table(["Stock","Pred → Actual","Correct"],dr),"```"]
    else: lines.append("No predictions available for evaluation.")
    lines += ["","🔥 *2. BUCKET + INTRADAY + IPO RESULTS*"]
    lines.append("Buckets: "+(" | ".join(f"{k} {v:.1f}%" for k,v in bucket.items()) if bucket else "not enough evaluated samples."))
    if intraday: lines.append("Intraday: "+" | ".join(f"{k} {v}" for k,v in intraday.items()))
    if ipo is not None and not ipo.empty: lines.append(f"IPO/new listings evaluated: {len(ipo)}")
    lines += _portfolio_lines(portfolio)
    lines += ["","🧠 *3. STAGE 10 LEARNING*"]
    lines += ["```",_table(["Metric","Value"],[["Samples",metrics.get("Samples",0)],["Overall MAPE",f"{metrics.get('OverallMAPE',0):.3f}%"],["Close MAPE",f"{metrics.get('CloseMAPE',0):.3f}%"],["Direction Accuracy",f"{metrics.get('DirectionAccuracy',0):.1f}%"],["Previous Accuracy",_fmt_accuracy(accuracy.get('PreviousAccuracy'))],["Current Accuracy",_fmt_accuracy(accuracy.get('CurrentAccuracy'))],["Champion/Challenger",retraining.get("Decision","-")],["Model Replaced","YES" if retraining.get("Retrained") else "NO"],["Improvement",f"{retraining.get('Improvement',0):+.2f}%"],["Learning State",learning.get("status","UPDATED")]]),"```"]
    return "\n".join(lines)
