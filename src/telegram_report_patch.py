"""Runtime hardening for Stage 28 reports and the morning horizon pipeline."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import telegram_report as _tr
from . import morning_runner as _mr
from .multihorizon import train_horizon_models
from .prediction import add_multihorizon_predictions
from .selection import select_prediction_set

_SHORT_HORIZONS=(3,7,10,20)
_LONG_HORIZONS=(60,90,180,365)
_ALL_HORIZONS=(1,)+_SHORT_HORIZONS+_LONG_HORIZONS


def _status(row):
    vals=[]
    for horizon in _ALL_HORIZONS:
        value=_tr._num(_tr._horizon_value(row,horizon))
        if value is not None:vals.append(value)
    if not vals:return "N/A"
    ratio=sum(value>0 for value in vals)/len(vals)
    if ratio>=2/3:return "🟢 BULLISH"
    if ratio>=1/3:return "🟡 MIXED"
    return "🔴 WEAK"


def _horizon_table(selected):
    if selected is None or selected.empty:return ["🔮 *MULTI-HORIZON OUTLOOK*","No multi-horizon predictions available."]
    ordered=_tr._sort(selected).head(10);rows=[]
    for _,row in ordered.iterrows():rows.append([str(row.get("Symbol","-"))]+[_tr._pct(_tr._horizon_value(row,h)) for h in _SHORT_HORIZONS]+[_status(row)])
    short=[f"🔮 *MULTI-HORIZON OUTLOOK — TOP {len(rows)}*","Short-term expected return by trading-day horizon.",*_tr._table(["Stock","3D","7D","10D","20D","Status"],rows,max_width=13)]
    long_rows=[]
    for _,row in ordered.iterrows():long_rows.append([str(row.get("Symbol","-"))]+[_tr._pct(_tr._horizon_value(row,h)) for h in _LONG_HORIZONS]+[_status(row)])
    long=["🔭 *LONG-HORIZON OUTLOOK — 60D TO 365D*","Long-term expected return by trading-day horizon.",*_tr._table(["Stock","60D","90D","180D","365D","Status"],long_rows,max_width=13)]
    return short+["",*long]


def _prediction_table(selected):
    if selected is None or selected.empty:return ["📈 *PREDICTED OHLC — TOP 10*","No predictions available."]
    ordered=_tr._sort(selected).head(10);rows=[[str(row.get("Symbol","-")),_tr._fmt(row.get("Pred_Open")),_tr._fmt(row.get("Pred_High")),_tr._fmt(row.get("Pred_Low")),_tr._fmt(row.get("Pred_Close")),_tr._pct(row.get("Expected_Return"))] for _,row in ordered.iterrows()]
    return [f"📈 *PREDICTED OHLC — TOP {len(rows)}*",*_tr._table(["Stock","Open","High","Low","Close","Exp%"],rows)]


def _portfolio_status(row):
    vals=[]
    for horizon in _ALL_HORIZONS:
        value=_tr._num(row.get(f"Horizon_{horizon}D"))
        if value is not None:vals.append(value)
    if not vals:return str(row.get("Portfolio_Target_Status",row.get("Target_Status","N/A")))
    ratio=sum(value>0 for value in vals)/len(vals)
    if ratio>=2/3:return "🟢 BULLISH"
    if ratio>=1/3:return "🟡 MIXED"
    return "🔴 WEAK"


def _attach_horizons_all(candidates,data_map,cutoff_date):
    """Attach every configured horizon once; never silently drop 60/90/180/365D."""
    rows=[]
    for _,row in candidates.iterrows():
        symbol=row["Symbol"]
        try:
            hb=train_horizon_models(data_map[symbol],cutoff_date);h=add_multihorizon_predictions(data_map[symbol],{"horizons":hb},cutoff_date)
            row["MultiHorizonExpectedReturn"]=0.0 if h.empty else float(pd.to_numeric(h["Expected_Return"],errors="coerce").clip(-50,50).median())
            if not h.empty:
                for horizon in (1,3,5,7,10,20,60,90,180,365):
                    m=h[h["HorizonDays"]==horizon]
                    if m.empty:continue
                    row[f"Horizon_{horizon}D"]=float(m.iloc[0]["Expected_Return"]);row[f"Horizon_{horizon}D_Pred_Close"]=float(m.iloc[0]["Pred_Close"]) if np.isfinite(float(m.iloc[0]["Pred_Close"])) else np.nan;row[f"Horizon_{horizon}D_MAPE"]=float(m.iloc[0].get("ValidationMAPE",np.nan));row[f"Horizon_{horizon}D_Samples"]=int(m.iloc[0].get("Samples",0) or 0)
        except Exception as exc:print(f"{symbol}: horizon prediction failed: {exc}");row["MultiHorizonExpectedReturn"]=0.0
        rows.append(row)
    return pd.DataFrame(rows) if rows else candidates.iloc[0:0]

_tr.REPORT_HORIZONS=_ALL_HORIZONS
_tr._horizon_status=_status
_tr._horizon_table=_horizon_table
_tr._prediction_table=_prediction_table
_tr._portfolio_horizon_status=_portfolio_status
_mr._attach_horizons=_attach_horizons_all

# The ledger is prediction-first: strict BUY/TRADE rules are applied later via
# the Action field, never by shrinking the canonical prediction set.
def _prediction_first_select(candidates,top_n=None,regime="SIDEWAYS",**kwargs):
    return select_prediction_set(candidates,top_n=top_n,max_per_bucket=kwargs.get("max_per_bucket"))
_mr.select_top_stocks=_prediction_first_select

_original_morning_report=_tr.morning_report

def _clean_market_heading(text):
    marker="📊 *MARKET OVERVIEW*"
    if not isinstance(text,str):return text
    while text.count(marker)>1:text=text.replace(marker,"",1)
    return text

def _action_summary(selected):
    if selected is None or selected.empty:return "🎯 *ACTION SUMMARY*\nBUY 0 | HOLD 0 | AVG 0 | SELL 0 | WAIT 0 | NO TRADE 0"
    counts={"BUY":0,"HOLD":0,"AVG":0,"SELL":0,"WAIT":0,"NO TRADE":0}
    for value in selected.get("Action",pd.Series(dtype=object)):
        d=_tr._decision(value)
        if d in counts:counts[d]+=1
        elif d in {"NO_TRADE","NOTRADE"}:counts["NO TRADE"]+=1
    return ("🎯 *ACTION SUMMARY*\n"+f"BUY {counts['BUY']} | HOLD {counts['HOLD']} | AVG {counts['AVG']} | SELL {counts['SELL']} | WAIT {counts['WAIT']} | NO TRADE {counts['NO TRADE']}")

def morning_report(*args,**kwargs):
    text=_clean_market_heading(_original_morning_report(*args,**kwargs));selected=kwargs.get("selected")
    if selected is None and len(args)>=3:selected=args[2]
    text=text.replace("QUALIFIED STOCKS BY PRICE BUCKET","PREDICTION SET BY PRICE BUCKET").replace("Qualified Stocks:","Prediction Set:")
    marker="🎯 *PREDICTION SET BY PRICE BUCKET*"
    return text.replace(marker,_action_summary(selected)+"\n\n"+marker,1) if marker in text else _action_summary(selected)+"\n\n"+text

_tr.morning_report=morning_report
