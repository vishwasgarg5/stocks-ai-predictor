"""Stage 4.5 — uncertainty, data quality and market risk intelligence."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .features import build_features
from .models import TARGETS

def add_prediction_uncertainty(candidates,data_map,bundles):
    if candidates is None or candidates.empty:return candidates
    out=candidates.copy();spreads=[];vols=[];hist=[];missing=[];stale=[];anomaly=[]
    for _,row in out.iterrows():
        symbol=row["Symbol"];bundle=bundles.get(symbol);spread=[]
        try:
            raw=data_map[symbol].copy();raw.index=pd.DatetimeIndex(raw.index);valid=raw[raw.index<=pd.Timestamp(row["DataCutoff"])].copy();hist.append(len(valid));
            expected=pd.date_range(valid.index.min().date(),valid.index.max().date(),freq="B") if len(valid)>1 else []
            missing.append(float(max(0,len(expected)-len(valid))/max(len(expected),1)*100) if len(valid)>1 else 0.0);stale.append(float(max((pd.Timestamp(row["DataCutoff"])-valid.index[-1]).days,0)) if not valid.empty else 30.0)
            anomaly_mask=(pd.to_numeric(valid.get("High"),errors="coerce")<pd.to_numeric(valid.get("Low"),errors="coerce"))|(pd.to_numeric(valid.get("Volume"),errors="coerce")<0);anomaly.append(float(anomaly_mask.mean()) if len(valid) else 1.0)
            ret=pd.to_numeric(valid["Close"],errors="coerce").pct_change().dropna()*100;vols.append(float(ret.tail(20).std()) if len(ret)>=5 else 3.0)
            df=build_features(valid);latest=df.dropna().iloc[[-1]]
            for target in TARGETS:
                info=bundle["targets"][target];vals=[float(model.predict(latest[bundle["features"]])[0]) for model in info["models"]];base=max(abs(float(row.get(f"Pred_{target}",latest[target].iloc[0]))),1e-6);spread.append(float(np.std(vals)/base*100))
        except Exception:
            spreads.append(5.0);vols.append(3.0);hist.append(0);missing.append(100.0);stale.append(30.0);anomaly.append(1.0);continue
        spreads.append(float(np.mean(spread)) if spread else 5.0)
    out["PredictionUncertaintyPct"]=spreads;out["UncertaintyScore"]=(100-out["PredictionUncertaintyPct"].clip(0,25)*4).clip(0,100);out["CalibratedConfidence"]=(0.70*out.get("Confidence",50)+0.30*out["UncertaintyScore"]).clip(0,100);out["RiskFlag"]=np.select([out["PredictionUncertaintyPct"]<=4,out["PredictionUncertaintyPct"]<=8],["LOW","MEDIUM"],default="HIGH");out["VolatilityPct"]=vols;out["HistoryRows"]=hist;out["MissingRatePct"]=missing;out["StaleDays"]=stale;out["OHLCVAnomalyRate"]=anomaly;out["DataQualityScore"]=(100-np.maximum(0,180-out["HistoryRows"])*.25-np.minimum(out["MissingRatePct"]*4,30)-np.minimum(out["StaleDays"]*10,30)-np.minimum(out["OHLCVAnomalyRate"]*100,30)).clip(0,100);return out

def add_market_risk(candidates,regime):
    out=candidates.copy();base={"BULL":85,"SIDEWAYS":65,"BEAR":40,"HIGH VOL":30}.get(str(regime).upper(),60);out["MarketRiskScore"]=base;out["RiskAdjustedScore"]=(out["Score"]*.75+out["MarketRiskScore"]*.10+out["UncertaintyScore"]*.15).clip(0,100);return out
