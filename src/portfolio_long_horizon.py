"""Portfolio-only extended horizon return models (60/90/180/365 sessions).

This module deliberately stays outside the core morning/evening prediction horizon set.
It reuses the project's feature pipeline and ensemble family so Portfolio Manager can
look up to roughly one year without making the daily stock-selection workflow heavier.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from .features import build_features, get_feature_columns

PORTFOLIO_LONG_HORIZONS = (60, 90, 180, 365)


def _models(seed=42):
    return [
        XGBRegressor(n_estimators=140, max_depth=4, learning_rate=0.04, subsample=0.85,
                     colsample_bytree=0.85, objective="reg:squarederror", random_state=seed, n_jobs=2),
        RandomForestRegressor(n_estimators=140, max_depth=10, min_samples_leaf=2, random_state=seed, n_jobs=2),
        ExtraTreesRegressor(n_estimators=140, max_depth=12, min_samples_leaf=2, random_state=seed, n_jobs=2),
    ]


def _mape(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean(np.abs((p - y) / np.maximum(np.abs(y), 1e-6))) * 100)


def _weights(errors):
    e = np.asarray(errors, dtype=float)
    e[~np.isfinite(e)] = 1.0
    inv = 1.0 / np.maximum(e, 1e-6)
    return inv / inv.sum()


def _train(work, features, target):
    split = max(60, int(len(work) * 0.8))
    split = min(split, len(work) - 1)
    Xtr, Xv = work[features].iloc[:split], work[features].iloc[split:]
    ytr, yv = work[target].iloc[:split], work[target].iloc[split:]
    validation = _models()
    preds, errors = [], []
    for model in validation:
        model.fit(Xtr, ytr)
        pred = model.predict(Xv)
        preds.append(pred)
        errors.append(_mape(yv, pred))
    weights = _weights(errors)
    final = _models()
    for model in final:
        model.fit(work[features], work[target])
    ensemble = np.average(np.vstack(preds), axis=0, weights=weights)
    return {"models": final, "weights": weights.tolist(), "validation_mape": _mape(yv, ensemble), "samples": len(work)}


def train_portfolio_long_horizon_models(df, cutoff_date):
    x = build_features(df)
    x = x[x.index <= pd.Timestamp(cutoff_date)].copy()
    features = get_feature_columns()
    result = {"features": features, "horizons": {}}
    for horizon in PORTFOLIO_LONG_HORIZONS:
        work = x[features].copy()
        work["target_return"] = (x["Close"].shift(-horizon) / x["Close"] - 1) * 100
        work = work.replace([np.inf, -np.inf], np.nan).dropna()
        if len(work) < 150:
            continue
        result["horizons"][horizon] = _train(work, features, "target_return")
    return result


def predict_portfolio_long_horizons(df, bundle, cutoff_date):
    x = build_features(df)
    x = x[x.index <= pd.Timestamp(cutoff_date)]
    usable = x[bundle["features"]].dropna()
    if usable.empty:
        return pd.DataFrame()
    latest = usable.iloc[[-1]]
    rows = []
    for horizon, info in bundle.get("horizons", {}).items():
        preds = np.array([m.predict(latest)[0] for m in info["models"]], dtype=float)
        value = float(preds @ np.asarray(info["weights"], dtype=float))
        if np.isfinite(value):
            rows.append({"HorizonDays": int(horizon), "Expected_Return": value,
                         "ValidationMAPE": info["validation_mape"], "Samples": info["samples"]})
    return pd.DataFrame(rows).sort_values("HorizonDays") if rows else pd.DataFrame()
