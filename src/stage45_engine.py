"""Stage 4.5 — accuracy calibration, uncertainty and risk intelligence."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .features import build_features
from .models import TARGETS, predict_ensemble


def add_prediction_uncertainty(candidates, data_map, bundles):
    """Add ensemble dispersion and a conservative confidence adjustment."""
    if candidates is None or candidates.empty:
        return candidates
    out=candidates.copy()
    spreads=[]
    for _,row in out.iterrows():
        symbol=row["Symbol"]
        bundle=bundles.get(symbol)
        spread=[]
        try:
            df=build_features(data_map[symbol]); latest=df[df.index<=pd.Timestamp(row["DataCutoff"])].dropna().iloc[[-1]]
            for target in TARGETS:
                info=bundle["targets"][target]
                vals=[]
                for model in info["models"]:
                    vals.append(float(model.predict(latest[bundle["features"]])[0]))
                base=max(abs(float(row.get(f"Pred_{target}",latest[target].iloc[0]))),1e-6)
                spread.append(float(np.std(vals)/base*100))
        except Exception:
            pass
        spreads.append(float(np.mean(spread)) if spread else 5.0)
    out["PredictionUncertaintyPct"]=spreads
    out["UncertaintyScore"]=(100-out["PredictionUncertaintyPct"].clip(0,25)*4).clip(0,100)
    out["CalibratedConfidence"]=(0.70*out.get("Confidence",50)+0.30*out["UncertaintyScore"]).clip(0,100)
    out["RiskFlag"]=np.select([out["PredictionUncertaintyPct"]<=4,out["PredictionUncertaintyPct"]<=8],["LOW","MEDIUM"],default="HIGH")
    return out


def add_market_risk(candidates, regime):
    """Apply a simple regime-aware risk score without leaking future data."""
    out=candidates.copy()
    base={"BULL":85,"SIDEWAYS":65,"BEAR":40,"HIGH VOL":30}.get(str(regime).upper(),60)
    out["MarketRiskScore"]=base
    out["RiskAdjustedScore"]=(out["Score"]*0.75+out["MarketRiskScore"]*0.10+out["UncertaintyScore"]*0.15).clip(0,100)
    return out
