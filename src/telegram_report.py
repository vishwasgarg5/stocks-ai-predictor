"""Stage 4 Telegram reporting.
Morning report intentionally contains only three actionable sections:
1) bucket-based top stocks with predicted OHLCV,
2) 7-session +5% jump watchlist,
3) intraday setups.
Evening reporting remains unchanged for evaluation/retraining.
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
    all_rows=[headers]+[[str(x) for x in row] for row in rows]
    widths=[max(len(row[i]) for row in all_rows) for i in range(len(headers))]
    sep="  ".join("-"*w for w in widths)
    body=["  ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) for row in all_rows]
    return "\n".join([body[0],sep]+body[1:])

def morning_report(prediction_date,cutoff_date,selected,jump_watchlist,intraday,**kwargs):
    """Compact actionable morning report; no scan diagnostics or model internals."""
    lines=["📈 *AI NSE MORNING REPORT*",f"_{prediction_date} | Data: {cutoff_date}_","","🎯 *1. TOP STOCKS — PRICE BUCKET + PREDICTED OHLCV*"]
    if selected is not None and not selected.empty:
        rows=[]
        for i,(_,r) in enumerate(selected.iterrows(),1):
            rows.append([i,r["Symbol"],r.get("PriceBucket","-"),f"{r.get('Score',0):.1f}",r.get("Direction","-"),f"{r.get('Confidence',0):.0f}%",format_money(r["Pred_Open"]),format_money(r["Pred_High"]),format_money(r["Pred_Low"]),format_money(r["Pred_Close"]),f"{float(r.get('Pred_Volume',0)):,.0f}"])
        lines += ["```",_table(["#","Stock","Bucket","Score","Dir","Conf","Open","High","Low","Close","Volume"],rows),"```"]
    else: lines.append("No qualifying bucket-based stock today.")

    lines += ["","🔥 *2. +5% JUMP WATCH — NEXT 7 SESSIONS*"]
    if jump_watchlist is not None and not jump_watchlist.empty:
        rows=[]
        for i,(_,r) in enumerate(jump_watchlist.iterrows(),1):
            cp=float(r["Current_Price"]); target=float(r["Target_Level"]); target_pct=((target/cp)-1)*100 if cp else 0
            rows.append([i,r["Symbol"],format_money(cp),format_money(target),format_percent(target_pct),format_percent(float(r.get("Estimated_7D_Upside",0))),f"{float(r.get('Jump_Probability',0)):.0f}%"])
        lines += ["```",_table(["#","Stock","CMP","+5% Target","Target%","7D Exp%","Prob"],rows),"```"]
    else: lines.append("No strong +5% candidates today.")

    lines += ["","⚡ *3. INTRADAY STOCKS*"]
    if intraday is not None and not intraday.empty:
        rows=[]
        for i,(_,r) in enumerate(intraday.iterrows(),1):
            rows.append([i,r["Symbol"],r["Bias"],format_money(r["Current"]),format_money(r["Target"]),format_money(r["StopLoss"]),f"{float(r.get('Confidence',0)):.0f}%"])
        lines += ["```",_table(["#","Stock","Bias","CMP","Target","SL","Conf"],rows),"```"]
    else: lines.append("No strong intraday setup today.")
    return "\n".join(lines)

def evening_report(market_date,evaluation,metrics,retraining):
    lines=["🌙 AI NSE EVENING REPORT — STAGE 4","```",_table(["Report","Value"],[["Market Date",market_date],["Evaluated Stocks",len(evaluation) if evaluation is not None else 0],["Model Version",MODEL_VERSION]]),"```","","📊 PREDICTED vs ACTUAL OHLC"]
    if evaluation is not None and not evaluation.empty:
        rows=[]; dr=[]
        for _,r in evaluation.iterrows():
            rows += [[r["Symbol"],"PRED",f"{r['Pred_Open']:.2f}",f"{r['Pred_High']:.2f}",f"{r['Pred_Low']:.2f}",f"{r['Pred_Close']:.2f}"],["","ACT",f"{r['Actual_Open']:.2f}",f"{r['Actual_High']:.2f}",f"{r['Actual_Low']:.2f}",f"{r['Actual_Close']:.2f}"],["","DIFF",f"{r['Diff_Open']:+.2f}",f"{r['Diff_High']:+.2f}",f"{r['Diff_Low']:+.2f}",f"{r['Diff_Close']:+.2f}"]]; dr.append([r["Symbol"],f"{r['Pred_Direction']} → {r['Actual_Direction']}","YES" if r["DirectionCorrect"] else "NO"])
        lines += ["```",_table(["Stock","Type","Open","High","Low","Close"],rows),"```","","🧭 DIRECTION CHECK","```",_table(["Stock","Pred → Actual","Correct"],dr),"```"]
    else: lines.append("No predictions available for evaluation.")
    lines += ["","📈 MODEL ACCURACY","```",_table(["Metric","Value"],[["Samples",metrics.get("Samples",0)],["Open MAPE",f"{metrics.get('OpenMAPE',0):.3f}%"],["High MAPE",f"{metrics.get('HighMAPE',0):.3f}%"],["Low MAPE",f"{metrics.get('LowMAPE',0):.3f}%"],["Close MAPE",f"{metrics.get('CloseMAPE',0):.3f}%"],["Overall MAPE",f"{metrics.get('OverallMAPE',0):.3f}%"],["Direction Accuracy",f"{metrics.get('DirectionAccuracy',0):.1f}%"]]),"```","","🏆 CHAMPION / CHALLENGER","```",_table(["Metric","Value"],[["Decision",retraining.get("Decision","-")],["Champion Error",f"{retraining.get('ChampionError',0):.6f}"],["Challenger Error",f"{retraining.get('ChallengerError',0):.6f}"],["Improvement",f"{retraining.get('Improvement',0):+.2f}%"],["Model Replaced","YES" if retraining.get("Retrained") else "NO"]]),"```"]
    return "\n".join(lines)
