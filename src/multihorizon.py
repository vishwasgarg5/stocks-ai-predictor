"""Multi-horizon close/return forecasting for 1/3/5/7/20/60/90/180/365 sessions."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from xgboost import XGBRegressor
from .features import build_features,get_feature_columns
HORIZONS=(1,3,5,7,20,60,90,180,365)

def _models(seed=42):
    return [XGBRegressor(n_estimators=180,max_depth=4,learning_rate=0.04,subsample=0.85,colsample_bytree=0.85,objective="reg:squarederror",random_state=seed,n_jobs=2),RandomForestRegressor(n_estimators=180,max_depth=10,min_samples_leaf=2,random_state=seed,n_jobs=2),ExtraTreesRegressor(n_estimators=180,max_depth=12,min_samples_leaf=2,random_state=seed,n_jobs=2)]
def _weights(errors):
    e=np.asarray(errors,float);e[~np.isfinite(e)]=1.;inv=1/np.maximum(e,1e-6);return inv/inv.sum()
def _mape(y,p):return float(np.mean(np.abs((np.asarray(p)-np.asarray(y))/np.maximum(np.abs(np.asarray(y)),1e-6)))*100)
def _train_target(work,features,target,horizon):
    # Purge the validation boundary by the forecast horizon so targets cannot overlap the training window.
    n=len(work);split=max(60,int(n*.8));split=min(split,n-1);purge=min(horizon,max(0,split-30));train_end=max(30,split-purge)
    Xtr,Xv=work[features].iloc[:train_end],work[features].iloc[split:];ytr,yv=work[target].iloc[:train_end],work[target].iloc[split:]
    if len(Xv)<10 or len(Xtr)<30:raise ValueError("Insufficient chronological validation data")
    validation_models=_models();vp=[];errors=[]
    for m in validation_models:
        m.fit(Xtr,ytr);pred=m.predict(Xv);vp.append(pred);errors.append(_mape(yv,pred))
    weights=_weights(errors);ensemble=np.average(np.vstack(vp),axis=0,weights=weights)
    final=_models()
    for m in final:m.fit(work[features],work[target])
    return {"models":final,"weights":weights.tolist(),"validation_mape":_mape(yv,ensemble),"validation_samples":len(yv),"samples":len(work)}
def train_horizon_models(df,cutoff_date):
    x=build_features(df);x=x[x.index<=pd.Timestamp(cutoff_date)].copy();features=get_feature_columns();result={"features":features,"horizons":{}}
    for h in HORIZONS:
        work=x[features].copy();work["target_close"]=x["Close"].shift(-h);work["target_return"]=(x["Close"].shift(-h)/x["Close"]-1)*100;work=work.replace([np.inf,-np.inf],np.nan).dropna()
        # Long horizons need substantially more history. Do not fabricate a forecast when evidence is too thin.
        minimum=max(150,100+h)
        if len(work)<minimum:continue
        try:result["horizons"][h]={"close":_train_target(work,features,"target_close",h),"return":_train_target(work,features,"target_return",h)}
        except Exception as exc:print(f"Horizon {h}D training skipped: {exc}")
    if not result["horizons"]:raise ValueError("No multi-horizon models could be trained")
    return result
def predict_horizons(df,bundle,cutoff_date):
    x=build_features(df);x=x[x.index<=pd.Timestamp(cutoff_date)];usable=x[bundle["features"]].dropna()
    if usable.empty:raise ValueError("No usable multi-horizon feature row")
    latest=usable.iloc[[-1]];current=float(latest["Close"].iloc[0]);rows=[]
    for h in HORIZONS:
        info=bundle["horizons"].get(h)
        if info is None:continue
        close=np.array([m.predict(latest)[0] for m in info["close"]["models"]]);ret=np.array([m.predict(latest)[0] for m in info["return"]["models"]]);cw=np.asarray(info["close"]["weights"]);rw=np.asarray(info["return"]["weights"]);close_pred=float(close@cw);return_pred=float(ret@rw);close_return=(close_pred/current-1)*100
        rows.append({"HorizonDays":h,"Pred_Close":close_pred,"Expected_Return":return_pred,"CloseDerivedReturn":close_return,"ValidationMAPE":info["close"]["validation_mape"],"ReturnValidationMAPE":info["return"]["validation_mape"],"Samples":info["close"]["samples"]})
    return pd.DataFrame(rows)
