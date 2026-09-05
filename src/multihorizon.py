"""Multi-horizon close and dedicated return forecasting for 1/3/5/7/20 sessions."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from xgboost import XGBRegressor
from .features import build_features,get_feature_columns
HORIZONS=(1,3,5,7,20)

def _models(seed=42):
    return [XGBRegressor(n_estimators=180,max_depth=4,learning_rate=0.04,subsample=0.85,colsample_bytree=0.85,objective="reg:squarederror",random_state=seed,n_jobs=2),RandomForestRegressor(n_estimators=180,max_depth=10,min_samples_leaf=2,random_state=seed,n_jobs=2),ExtraTreesRegressor(n_estimators=180,max_depth=12,min_samples_leaf=2,random_state=seed,n_jobs=2)]
def _weights(errors):
    e=np.asarray(errors,float);e[~np.isfinite(e)]=1.;inv=1/np.maximum(e,1e-6);return inv/inv.sum()
def _mape(y,p):return float(np.mean(np.abs((np.asarray(p)-np.asarray(y))/np.maximum(np.abs(np.asarray(y)),1e-6)))*100)
def _train_target(work,features,target):
    split=max(60,int(len(work)*.8));split=min(split,len(work)-1);Xtr,Xv=work[features].iloc[:split],work[features].iloc[split:];ytr,yv=work[target].iloc[:split],work[target].iloc[split:];errors=[]
    for m in _models():m.fit(Xtr,ytr);errors.append(_mape(yv,m.predict(Xv)))
    weights=_weights(errors);final=_models()
    for m in final:m.fit(work[features],work[target])
    return {"models":final,"weights":weights.tolist(),"validation_mape":_mape(yv,sum(w*m.predict(Xv) for w,m in zip(weights,_models()))),"samples":len(work)}
def train_horizon_models(df,cutoff_date):
    x=build_features(df);x=x[x.index<=pd.Timestamp(cutoff_date)].copy();features=get_feature_columns();result={"features":features,"horizons":{}}
    for h in HORIZONS:
        work=x[features].copy();work["target_close"]=x["Close"].shift(-h);work["target_return"]=(x["Close"].shift(-h)/x["Close"]-1)*100;work=work.replace([np.inf,-np.inf],np.nan).dropna()
        if len(work)<150:continue
        result["horizons"][h]={"close":_train_target(work,features,"target_close"),"return":_train_target(work,features,"target_return")}
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
