"""Persistent decision-outcome ledger used for live learning and weekly diagnostics."""
from pathlib import Path
import pandas as pd
import numpy as np
from .config import DECISION_LEDGER_FILE

COLUMNS=["PredictionDate","EvaluationDate","Symbol","PriceBucket","Action","DecisionScore","Confidence","EntryPrice","PredictedClose","ActualOpen","ActualHigh","ActualLow","ActualClose","ReturnPct","MFEPct","MAEPct","Outcome","DirectionCorrect"]

def _read():
    if not DECISION_LEDGER_FILE.exists():return pd.DataFrame(columns=COLUMNS)
    try:return pd.read_csv(DECISION_LEDGER_FILE)
    except Exception:return pd.DataFrame(columns=COLUMNS)

def save_decisions(predictions,prediction_date):
    if predictions is None or predictions.empty:return
    rows=[]
    for _,r in predictions.iterrows():
        rows.append({"PredictionDate":str(prediction_date),"EvaluationDate":"","Symbol":str(r.get("Symbol","")),"PriceBucket":str(r.get("PriceBucket","-")),"Action":str(r.get("Action","HOLD")),"DecisionScore":float(r.get("FinalDecisionScore",0) or 0),"Confidence":float(r.get("CalibratedConfidence",r.get("Confidence",0)) or 0),"EntryPrice":float(r.get("Current_Close",r.get("Current_Price",0)) or 0),"PredictedClose":float(r.get("Pred_Close",0) or 0),"ActualOpen":np.nan,"ActualHigh":np.nan,"ActualLow":np.nan,"ActualClose":np.nan,"ReturnPct":np.nan,"MFEPct":np.nan,"MAEPct":np.nan,"Outcome":"OPEN","DirectionCorrect":np.nan})
    old=_read();x=pd.concat([old,pd.DataFrame(rows)],ignore_index=True);x=x.drop_duplicates(["PredictionDate","Symbol"],keep="last");DECISION_LEDGER_FILE.parent.mkdir(parents=True,exist_ok=True);x.to_csv(DECISION_LEDGER_FILE,index=False)

def evaluate_decisions(prediction_date,evaluation_date,data_map):
    x=_read()
    if x.empty:return pd.DataFrame()
    mask=(x["PredictionDate"].astype(str)==str(prediction_date))&(x["Outcome"].astype(str)=="OPEN")
    rows=[]
    for idx,r in x[mask].iterrows():
        df=data_map.get(str(r["Symbol"]))
        if df is None or df.empty:continue
        dates=pd.DatetimeIndex(df.index).normalize();target=pd.Timestamp(evaluation_date);hits=np.where(dates==target)[0]
        if len(hits)==0:continue
        a=df.iloc[int(hits[0])];entry=float(r["EntryPrice"]);close=float(a["Close"]);high=float(a["High"]);low=float(a["Low"])
        if entry<=0:continue
        ret=(close/entry-1)*100;mfe=(high/entry-1)*100;mae=(low/entry-1)*100;action=str(r["Action"]).upper()
        outcome="WIN" if (action=="BUY" and ret>0) or (action=="AVOID" and ret<0) else ("LOSS" if (action=="BUY" and ret<0) or (action=="AVOID" and ret>0) else "NEUTRAL")
        x.loc[idx,["EvaluationDate","ActualOpen","ActualHigh","ActualLow","ActualClose","ReturnPct","MFEPct","MAEPct","Outcome","DirectionCorrect"]]=[str(evaluation_date),float(a["Open"]),high,low,close,ret,mfe,mae,outcome,float((ret>0 and r["PredictedClose"]>entry) or (ret<0 and r["PredictedClose"]<entry) or (ret==0 and r["PredictedClose"]==entry))]
        rows.append(x.loc[idx].to_dict())
    x.to_csv(DECISION_LEDGER_FILE,index=False)
    return pd.DataFrame(rows)

def summary(days=30):
    x=_read()
    if x.empty:return {"Samples":0,"WinRate":None,"AvgReturn":None,"BUY":0,"NO_TRADE":0}
    x=x[x["Outcome"].astype(str)!="OPEN"].copy()
    if x.empty:return {"Samples":0,"WinRate":None,"AvgReturn":None,"BUY":0,"NO_TRADE":0}
    x["EvaluationDate"]=pd.to_datetime(x["EvaluationDate"],errors="coerce");x=x.sort_values("EvaluationDate").tail(max(days*25,days))
    return {"Samples":len(x),"WinRate":float((x["Outcome"]=="WIN").mean()*100),"AvgReturn":float(pd.to_numeric(x["ReturnPct"],errors="coerce").mean()),"BUY":int((x["Action"]=="BUY").sum()),"NO_TRADE":int((x["Action"]=="NO TRADE").sum())}
