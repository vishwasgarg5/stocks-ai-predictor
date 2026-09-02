"""
STAGE 4 TELEGRAM REPORT
=======================
Easy access: this file controls the Telegram layout.
Includes Stage 3A price buckets, Stage 3B multi-horizon forecasts,
and Stage 4 sector intelligence while retaining Stage 2 reports.
"""
import os
import requests
import pandas as pd
from .config import TELEGRAM_MAX_LENGTH, MODEL_VERSION
from .utils import format_money, format_percent, split_messages
from .selection import expected_return_score, regime_direction_score

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

def _evaluation_components(row,regime):
    return {"Technical":float(row.get("TechnicalScore",50)),"Expected Return":expected_return_score(row.get("Expected_Return",0)),"Model Confidence":float(row.get("Confidence",50)),"Direction":float(row.get("Direction_Confidence",50)),"Reliability":float(row.get("ReliabilityScore",50)),"Regime":regime_direction_score(row.get("Direction","NEUTRAL"),regime),"Sector":float(row.get("SectorScore",50))}

def morning_report(prediction_date,cutoff_date,schedule_status,regime,model_variant,selected,jump_watchlist,intraday,universe_count=None,liquid_count=None,prescreen_count=None,horizon_tables=None):
    horizon_tables=horizon_tables or {}
    lines=["📈 AI NSE STOCK PREDICTION — STAGE 4","```",_table(["Market Info","Value"],[["Prediction Date",prediction_date],["Data Cutoff",cutoff_date],["Schedule",schedule_status],["Market Regime",regime],["Model",model_variant],["Model Version",MODEL_VERSION]]),"```","","🔎 SCAN SUMMARY","```",_table(["Stage","Stocks"],[["Universe Loaded",universe_count if universe_count is not None else "-"],["Liquid Universe",liquid_count if liquid_count is not None else "-"],["Technical Prescreen",prescreen_count if prescreen_count is not None else "-"],["ML Final Top 5",5 if selected is not None and not selected.empty else 0],["Jump Watch Candidates",30 if jump_watchlist is not None and not jump_watchlist.empty else 0],["Intraday Scan",liquid_count if liquid_count is not None else "-"]]),"```","","🎯 NEXT-DAY TOP 5"]
    if selected is not None and not selected.empty:
        rows=[]
        for i,(_,r) in enumerate(selected.iterrows(),1): rows.append([i,r["Symbol"],r.get("PriceBucket","-"),r.get("Sector","-"),f"{r['Score']:.1f}",r["Direction"],f"{r['Confidence']:.0f}%",f"{r['Expected_Return']:+.2f}%",f"{r['Pred_Close']:.2f}"])
        lines += ["```",_table(["#","Stock","Bucket","Sector","Score","Dir","Conf","Exp%","Close"],rows),"```","","📊 PREDICTED OHLC","```"]
        lines += [_table(["Stock","Open","High","Low","Close"],[[r["Symbol"],f"{r['Pred_Open']:.2f}",f"{r['Pred_High']:.2f}",f"{r['Pred_Low']:.2f}",f"{r['Pred_Close']:.2f}"] for _,r in selected.iterrows()]),"```","","🔭 MULTI-HORIZON FORECAST — STAGE 3B","```"]
        mh=[]
        for _,r in selected.iterrows():
            h=horizon_tables.get(r["Symbol"],pd.DataFrame())
            vals=[]
            for d in [1,3,5,7,20]:
                z=h[h["HorizonDays"]==d] if not h.empty else pd.DataFrame()
                vals.append(f"{float(z['Expected_Return'].iloc[0]):+.1f}%" if not z.empty else "-")
            mh.append([r["Symbol"],*vals])
        lines += [_table(["Stock","1D","3D","5D","7D","20D"],mh),"```","","🧠 TOP 5 EVALUATION","```"]
        cr=[]
        for _,r in selected.iterrows():
            c=_evaluation_components(r,regime); cr.append([r["Symbol"],f"{c['Technical']:.0f}",f"{c['Expected Return']:.0f}",f"{c['Model Confidence']:.0f}",f"{c['Direction']:.0f}",f"{c['Reliability']:.0f}",f"{c['Regime']:.0f}",f"{c['Sector']:.0f}",f"{r['Score']:.1f}"])
        lines += [_table(["Stock","Tech","ExpRet","Model","Dir","Reliab","Regime","Sector","Final"],cr),"```","","⚖️ SELECTION WEIGHTS","```",_table(["Component","Weight"],[["Technical Score","20%"],["Expected Return","18%"],["Model Confidence","18%"],["Direction Confidence","14%"],["Reliability","10%"],["Market Regime","10%"],["Sector Strength","10%"]]),"```","","🏷️ PRICE BUCKETS","```",_table(["Bucket","Price"],[["B1","> ₹1,000"],["B2","₹500–₹999"],["B3","₹100–₹499"],["B4","₹50–₹99"],["B5","₹10–₹49"]]),"```"]
    else: lines.append("No next-day predictions generated.")
    lines += ["","🔥 7-DAY +5% JUMP WATCHLIST"]
    if jump_watchlist is not None and not jump_watchlist.empty:
        rows=[]
        for i,(_,r) in enumerate(jump_watchlist.iterrows(),1):
            cp=float(r["Current_Price"]); target=float(r["Target_Level"]); tp=((target/cp)-1)*100 if cp else 0
            rows.append([i,r["Symbol"],format_money(cp),format_money(target),format_percent(tp),format_percent(float(r.get("Estimated_7D_Upside",0))),f"{r['Jump_Probability']:.0f}%"])
        lines += ["```",_table(["#","Stock","CMP","Target","Target%","7D Exp%","Prob"],rows),"```","```",_table(["Meaning","Definition"],[["Target%","Distance from CMP to +5% target"],["7D Exp%","Model-estimated 7-trading-day upside"],["Prob","Probability of reaching +5% within 7 sessions"]]),"```"]
    else: lines.append("No strong +5% candidates today.")
    lines += ["","⚡ INTRADAY TOP 5"]
    if intraday is not None and not intraday.empty:
        lines += ["```",_table(["#","Stock","Bias","CMP","Target","SL","Conf"],[[i,r["Symbol"],r["Bias"],format_money(r["Current"]),format_money(r["Target"]),format_money(r["StopLoss"]),f"{r['Confidence']:.0f}%"] for i,(_,r) in enumerate(intraday.iterrows(),1)]),"```"]
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
