"""Stage 4.5 — uncertainty, data quality and market risk intelligence."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .features import build_features
from .models import TARGETS

def add_prediction_uncertainty(candidates, data_map, bundles):
    if candidates is None or candidates.empty:
        return candidates
    out = candidates.loc[:, ~candidates.columns.duplicated()].copy()
    spreads, vols, hist, missing, stale, anomaly = [], [], [], [], [], []
    for _, row in out.iterrows():
        symbol = row["Symbol"]
        bundle = (bundles or {}).get(symbol)
        spread_values = []
        row_vol, row_hist, row_missing, row_stale, row_anomaly = 3.0, 0, 100.0, 30.0, 1.0
        try:
            raw = data_map[symbol].copy(); raw.index = pd.DatetimeIndex(raw.index)
            valid = raw[raw.index <= pd.Timestamp(row["DataCutoff"])].copy(); row_hist = len(valid)
            if len(valid) > 1:
                expected = pd.date_range(valid.index.min().date(), valid.index.max().date(), freq="B")
                row_missing = float(max(0, len(expected) - len(valid)) / max(len(expected), 1) * 100)
            if not valid.empty:
                row_stale = float(max((pd.Timestamp(row["DataCutoff"]) - valid.index[-1]).days, 0))
            high = pd.to_numeric(valid.get("High"), errors="coerce"); low = pd.to_numeric(valid.get("Low"), errors="coerce"); volume = pd.to_numeric(valid.get("Volume"), errors="coerce")
            anomaly_mask = (high < low) | (volume < 0); row_anomaly = float(anomaly_mask.mean()) if len(valid) else 1.0
            close = pd.to_numeric(valid["Close"], errors="coerce"); ret = close.pct_change().dropna() * 100
            row_vol = float(ret.tail(20).std()) if len(ret) >= 5 else 3.0
            if bundle and isinstance(bundle, dict) and "targets" in bundle:
                df = build_features(valid); latest = df.dropna().iloc[[-1]]
                for target in TARGETS:
                    info = bundle["targets"].get(target, {})
                    models = info.get("models", [])
                    if not models: continue
                    values = [float(model.predict(latest[bundle["features"]])[0]) for model in models]
                    predicted = row.get(f"Pred_{target}", latest[target].iloc[0]); base = max(abs(float(predicted)), 1e-6)
                    spread_values.append(float(np.std(values) / base * 100))
            if spread_values:
                row_spread = float(np.mean(spread_values))
            else:
                # Safe fallback when the caller cannot retain trained bundles.
                # This keeps uncertainty deterministic rather than silently failing.
                preds = [pd.to_numeric(row.get(f"Pred_{t}"), errors="coerce") for t in TARGETS]
                p = [float(x) for x in preds if pd.notna(x)]
                center = max(abs(float(np.mean(p))), 1e-6) if p else 1.0
                row_spread = float((np.std(p) / center) * 100) if len(p) > 1 else min(max(row_vol, 1.0), 12.0)
        except Exception:
            row_spread = min(max(row_vol, 3.0), 12.0)
        spreads.append(row_spread); vols.append(row_vol); hist.append(row_hist); missing.append(row_missing); stale.append(row_stale); anomaly.append(row_anomaly)
    out["PredictionUncertaintyPct"] = pd.Series(spreads, index=out.index, dtype=float)
    out["UncertaintyScore"] = (100 - out["PredictionUncertaintyPct"].clip(0, 25) * 4).clip(0, 100)
    confidence = pd.to_numeric(out["Confidence"], errors="coerce").fillna(50.0) if "Confidence" in out.columns else pd.Series(50.0, index=out.index, dtype=float)
    out["CalibratedConfidence"] = (0.70 * confidence + 0.30 * out["UncertaintyScore"]).clip(0, 100)
    out["RiskFlag"] = np.select([out["PredictionUncertaintyPct"] <= 4, out["PredictionUncertaintyPct"] <= 8], ["LOW", "MEDIUM"], default="HIGH")
    out["VolatilityPct"] = pd.Series(vols, index=out.index, dtype=float); out["HistoryRows"] = pd.Series(hist, index=out.index, dtype=int); out["MissingRatePct"] = pd.Series(missing, index=out.index, dtype=float); out["StaleDays"] = pd.Series(stale, index=out.index, dtype=float); out["OHLCVAnomalyRate"] = pd.Series(anomaly, index=out.index, dtype=float)
    out["DataQualityScore"] = (100 - np.maximum(0, 180 - out["HistoryRows"]) * 0.25 - np.minimum(out["MissingRatePct"] * 4, 30) - np.minimum(out["StaleDays"] * 10, 30) - np.minimum(out["OHLCVAnomalyRate"] * 100, 30)).clip(0, 100)
    return out

def add_market_risk(candidates, regime):
    out = candidates.loc[:, ~candidates.columns.duplicated()].copy()
    base = {"BULL": 85, "SIDEWAYS": 65, "BEAR": 40, "HIGH VOL": 30}.get(str(regime).upper(), 60)
    out["MarketRiskScore"] = base
    out["RiskAdjustedScore"] = (out["Score"] * 0.75 + out["MarketRiskScore"] * 0.10 + out["UncertaintyScore"] * 0.15).clip(0, 100)
    return out
