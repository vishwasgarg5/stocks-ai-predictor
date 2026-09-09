"""Robust multi-horizon forecasting through 365 trading days with explicit per-horizon status."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from xgboost import XGBRegressor

from .features import build_features, get_feature_columns
from .config import MULTI_HORIZONS

HORIZONS = tuple(MULTI_HORIZONS)


def _models(seed=42):
    return [
        XGBRegressor(n_estimators=180, max_depth=4, learning_rate=0.04, subsample=0.85,
                     colsample_bytree=0.85, objective="reg:squarederror", random_state=seed, n_jobs=2),
        RandomForestRegressor(n_estimators=180, max_depth=10, min_samples_leaf=2,
                              random_state=seed, n_jobs=2),
        ExtraTreesRegressor(n_estimators=180, max_depth=12, min_samples_leaf=2,
                            random_state=seed, n_jobs=2),
    ]


def _weights(errors):
    e = np.asarray(errors, float)
    e[~np.isfinite(e)] = 1.0
    inv = 1.0 / np.maximum(e, 1e-6)
    return inv / inv.sum()


def _mape(y, p):
    return float(np.mean(np.abs((np.asarray(p) - np.asarray(y)) /
                               np.maximum(np.abs(np.asarray(y)), 1e-6))) * 100)


def _train_target(work, features, target, horizon):
    n = len(work)
    split = max(60, int(n * 0.80))
    split = min(split, n - 1)

    # A 365-day forecast must not discard 365 observations from training.
    # Keep a small fixed embargo to preserve chronological separation while
    # retaining enough training data for long horizons.
    embargo = min(10, max(0, split - 30))
    train_end = split - embargo
    if train_end < 30:
        train_end = 30

    Xtr = work[features].iloc[:train_end]
    Xv = work[features].iloc[split:]
    ytr = work[target].iloc[:train_end]
    yv = work[target].iloc[split:]
    if len(Xv) < 10 or len(Xtr) < 30:
        raise ValueError("Insufficient chronological validation data")

    vp, errors = [], []
    for model in _models():
        model.fit(Xtr, ytr)
        pred = model.predict(Xv)
        vp.append(pred)
        errors.append(_mape(yv, pred))

    weights = _weights(errors)
    ensemble = np.average(np.vstack(vp), axis=0, weights=weights)
    final = _models()
    for model in final:
        model.fit(work[features], work[target])

    return {
        "models": final,
        "weights": weights.tolist(),
        "validation_mape": _mape(yv, ensemble),
        "validation_samples": len(yv),
        "samples": len(work),
        "horizon_days": horizon,
        "validation_embargo": embargo,
    }


def train_horizon_models(df, cutoff_date):
    x = build_features(df)
    x = x[x.index <= pd.Timestamp(cutoff_date)].copy()
    features = get_feature_columns()
    result = {"features": features, "horizons": {}, "status": {}}

    for h in HORIZONS:
        work = x[features].copy()
        work["target_close"] = x["Close"].shift(-h)
        work["target_return"] = (x["Close"].shift(-h) / x["Close"] - 1) * 100
        work = work.replace([np.inf, -np.inf], np.nan).dropna()

        # Need enough rows for training, validation and the horizon target.
        # Do not impose an artificial 100+h threshold that blocks long horizons.
        minimum = max(150, 120 + min(h, 60))
        if len(work) < minimum:
            result["status"][h] = {
                "Status": "INSUFFICIENT_DATA",
                "Samples": len(work),
                "Minimum": minimum,
            }
            continue
        try:
            result["horizons"][h] = {
                "close": _train_target(work, features, "target_close", h),
                "return": _train_target(work, features, "target_return", h),
            }
            result["status"][h] = {
                "Status": "VALID",
                "Samples": len(work),
                "Minimum": minimum,
            }
        except Exception as exc:
            result["status"][h] = {
                "Status": "MODEL_FAILED",
                "Samples": len(work),
                "Minimum": minimum,
                "Error": str(exc),
            }

    if not result["horizons"]:
        raise ValueError("No multi-horizon models could be trained")
    return result


def predict_horizons(df, bundle, cutoff_date):
    x = build_features(df)
    x = x[x.index <= pd.Timestamp(cutoff_date)]
    usable = x[bundle["features"]].dropna()
    if usable.empty:
        raise ValueError("No usable multi-horizon feature row")

    latest = usable.iloc[[-1]]
    current = float(latest["Close"].iloc[0])
    rows = []
    for h in HORIZONS:
        info = bundle["horizons"].get(h)
        status = bundle.get("status", {}).get(h, {})
        if info is None:
            rows.append({
                "HorizonDays": h,
                "Status": status.get("Status", "UNAVAILABLE"),
                "Pred_Close": np.nan,
                "Expected_Return": np.nan,
                "CloseDerivedReturn": np.nan,
                "ValidationMAPE": np.nan,
                "ReturnValidationMAPE": np.nan,
                "Samples": status.get("Samples", 0),
            })
            continue

        close = np.array([m.predict(latest)[0] for m in info["close"]["models"]])
        ret = np.array([m.predict(latest)[0] for m in info["return"]["models"]])
        cw = np.asarray(info["close"]["weights"])
        rw = np.asarray(info["return"]["weights"])
        close_pred = float(close @ cw)
        return_pred = float(ret @ rw)
        rows.append({
            "HorizonDays": h,
            "Status": "VALID",
            "Pred_Close": close_pred,
            "Expected_Return": return_pred,
            "CloseDerivedReturn": (close_pred / current - 1) * 100,
            "ValidationMAPE": info["close"]["validation_mape"],
            "ReturnValidationMAPE": info["return"]["validation_mape"],
            "Samples": info["close"]["samples"],
        })

    return pd.DataFrame(rows)
