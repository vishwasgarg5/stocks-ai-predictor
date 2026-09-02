"""
STAGE 3B / STAGE 4 — MULTI-HORIZON PRICE FORECASTING
=====================================================
Easy access: this file contains the complete Stage 3B engine.
HORIZONS = 1, 3, 5, 7 and 20 trading sessions.
Stage 2 next-day OHLC remains the primary OHLC engine; this module adds
future-close/return forecasts without using future input data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from .features import build_features, get_feature_columns

HORIZONS = (1, 3, 5, 7, 20)

def _models(seed=42):
    return [
        XGBRegressor(n_estimators=180,max_depth=4,learning_rate=0.04,subsample=0.85,colsample_bytree=0.85,objective="reg:squarederror",random_state=seed,n_jobs=2),
        RandomForestRegressor(n_estimators=180,max_depth=10,min_samples_leaf=2,random_state=seed,n_jobs=2),
        ExtraTreesRegressor(n_estimators=180,max_depth=12,min_samples_leaf=2,random_state=seed,n_jobs=2),
    ]

def _weights(errors):
    e=np.asarray(errors,dtype=float); e[~np.isfinite(e)]=1.0
    inv=1.0/np.maximum(e,1e-6)
    return inv/inv.sum()

def train_horizon_models(df, cutoff_date):
    """Train one chronological-validation ensemble for every horizon."""
    x=build_features(df); x=x[x.index<=pd.Timestamp(cutoff_date)].copy()
    features=get_feature_columns(); result={"features":features,"horizons":{}}
    for horizon in HORIZONS:
        work=x[features].copy(); work["target"]=x["Close"].shift(-horizon)
        work=work.replace([np.inf,-np.inf],np.nan).dropna()
        if len(work)<150: continue
        split=max(50,int(len(work)*0.80)); split=min(split,len(work)-1)
        Xtr,Xv=work[features].iloc[:split],work[features].iloc[split:]
        ytr,yv=work.target.iloc[:split],work.target.iloc[split:]
        validation=[]
        for model in _models():
            model.fit(Xtr,ytr); pred=model.predict(Xv)
            validation.append(float(np.mean(np.abs((pred-yv.to_numpy())/np.maximum(np.abs(yv.to_numpy()),1e-6)))))
        weights=_weights(validation); final=_models()
        for model in final: model.fit(work[features],work.target)
        result["horizons"][horizon]={"models":final,"weights":weights.tolist(),"validation_mape":float(np.mean(validation)*100),"samples":len(work)}
    if not result["horizons"]: raise ValueError("No multi-horizon models could be trained")
    return result

def predict_horizons(df,bundle,cutoff_date):
    """Return predicted close, expected return and validation MAPE per horizon."""
    x=build_features(df); x=x[x.index<=pd.Timestamp(cutoff_date)]
    usable=x[bundle["features"]].dropna()
    if usable.empty: raise ValueError("No usable multi-horizon feature row")
    latest=usable.iloc[[-1]]; current=float(latest["Close"].iloc[0]); rows=[]
    for horizon in HORIZONS:
        info=bundle["horizons"].get(horizon)
        if info is None: continue
        preds=np.array([m.predict(latest)[0] for m in info["models"]],dtype=float)
        close_pred=float(np.dot(preds,np.asarray(info["weights"])))
        rows.append({"HorizonDays":horizon,"Pred_Close":close_pred,"Expected_Return":(close_pred/current-1)*100,"ValidationMAPE":info["validation_mape"],"Samples":info["samples"]})
    return pd.DataFrame(rows)
