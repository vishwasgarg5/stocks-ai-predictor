"""Multi-horizon forecasting with horizon-aware purged validation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from .features import build_features, get_feature_columns
from .config import MULTI_HORIZONS
HORIZONS=tuple(MULTI_HORIZONS)

def _models(seed=42):
    return [XGBRegressor(n_estimators=180,max_depth=4,learning_rate=0.04,subsample=0.85,colsample_bytree=0.85,objective="reg:squarederror",random_state=seed,n_jobs=2),RandomForestRegressor(n_estimators=180,max_depth=10,min_samples_leaf=2,random_state=seed,n_jobs=2),ExtraTreesRegressor(n_estimators=180,max_depth=12,min_samples_leaf=2,random_state=seed,n_jobs=2)]
def _weights(errors):
    e=np.asarray(errors,float);e[~np.isfinite(e)]=1.;inv=1./np.maximum(e,1e-6);return inv/inv.sum()
def _mape(y,p):return float(np.mean(np.abs((np.asarray(p)-np.asarray(y))/np.maximum(np.abs(np.asarray(y)),1e-6)))*100)
def _purged_split(n,horizon,validation_fraction=0.20):
    if n<100:return None
    split=max(60,int(n*(1.-validation_fraction)));split=min(split,n-10);train_end=split-int(horizon)
    if train_end<30 or n-split<10:return None
    return train_end,split,n
def _train_target(work,features,target,horizon):
    split_info=_purged_split(len(work),horizon)
    if split_info is None:raise ValueError(f"Insufficient samples for horizon-aware validation: h={horizon}, n={len(work)}")
    train_end,split,val_end=split_info;Xtr=work[features].iloc[:train_end];Xv=work[features].iloc[split:val_end];ytr=work[target].iloc[:train_end];yv=work[target].iloc[split:val_end]
    if len(Xv)<10 or len(Xtr)<30:raise ValueError("Insufficient chronological validation data")
    vp=[];errors=[]
    for model in _models():model.fit(Xtr,ytr);pred=model.predict(Xv);vp.append(pred);errors.append(_mape(yv,pred))
    weights=_weights(errors);ensemble=np.average(np.vstack(vp),axis=0,weights=weights);final=_models()
    for model in final:model.fit(work[features],work[target])
    return {"models":final,"weights":weights.tolist(),"validation_mape":_mape(yv,ensemble),"validation_samples":len(yv),"samples":len(work),"horizon_days":horizon,"validation_embargo":int(horizon),"validation_method":"chronological_purged_holdout","validation_train_end":str(work.index[train_end-1].date()),"validation_start":str(work.index[split].date()),"validation_end":str(work.index[val_end-1].date())}
def train_horizon_models(df,cutoff_date):
    # Use the long history already expanded by the canonical prediction model
    # when the candidate scan itself was intentionally short-window.
    source_df=df.attrs.get("expanded_history",df) if hasattr(df,"attrs") else df
    x=build_features(source_df);x=x[x.index<=pd.Timestamp(cutoff_date)].copy();features=get_feature_columns();result={"features":features,"horizons":{},"status":{}}
    for h in HORIZONS:
        work=x[features].copy();work["target_close"]=x["Close"].shift(-h);work["target_return"]=(x["Close"].shift(-h)/x["Close"]-1)*100;work=work.replace([np.inf,-np.inf],np.nan).dropna();minimum=max(180,80+int(h)*2)
        if len(work)<minimum:result["status"][h]={"Status":"INSUFFICIENT_DATA","Samples":len(work),"Minimum":minimum};continue
        try:
            result["horizons"][h]={"close":_train_target(work,features,"target_close",h),"return":_train_target(work,features,"target_return",h)};result["status"][h]={"Status":"VALID","Samples":len(work),"Minimum":minimum,"ValidationMethod":"chronological_purged_holdout","EmbargoSessions":int(h)}
        except Exception as exc:result["status"][h]={"Status":"MODEL_FAILED","Samples":len(work),"Minimum":minimum,"Error":str(exc)}
    if not result["horizons"]:raise ValueError("No multi-horizon models could be trained")
    return result
def predict_horizons(df,bundle,cutoff_date):
    source_df=df.attrs.get("expanded_history",df) if hasattr(df,"attrs") else df;x=build_features(source_df);x=x[x.index<=pd.Timestamp(cutoff_date)];usable=x[bundle["features"]].dropna()
    if usable.empty:raise ValueError("No usable multi-horizon feature row")
    latest=usable.iloc[[-1]];current=float(latest["Close"].iloc[0]);rows=[]
    for h in HORIZONS:
        info=bundle["horizons"].get(h);status=bundle.get("status",{}).get(h,{})
        if info is None:rows.append({"HorizonDays":h,"Status":status.get("Status","UNAVAILABLE"),"Pred_Close":np.nan,"Expected_Return":np.nan,"CloseDerivedReturn":np.nan,"ValidationMAPE":np.nan,"ReturnValidationMAPE":np.nan,"Samples":status.get("Samples",0)});continue
        close=np.array([m.predict(latest)[0] for m in info["close"]["models"]]);ret=np.array([m.predict(latest)[0] for m in info["return"]["models"]]);cw=np.asarray(info["close"]["weights"]);rw=np.asarray(info["return"]["weights"]);close_pred=float(close@cw);return_pred=float(ret@rw);rows.append({"HorizonDays":h,"Status":"VALID","Pred_Close":close_pred,"Expected_Return":return_pred,"CloseDerivedReturn":(close_pred/current-1)*100,"ValidationMAPE":info["close"]["validation_mape"],"ReturnValidationMAPE":info["return"]["validation_mape"],"Samples":info["close"]["samples"]})
    return pd.DataFrame(rows)