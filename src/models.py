import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from .config import VALIDATION_FRACTION, WALK_FORWARD_MIN_TRAIN, WALK_FORWARD_FOLDS, WALK_FORWARD_EMBARGO
from .utils import safe_mape, normalized_mae

TARGETS=["Open","High","Low","Close"]

def create_regressors(variant="A"):
    if variant=="A":
        xgb_params={"n_estimators":250,"learning_rate":0.035,"max_depth":5,"min_child_weight":3,"subsample":0.85,"colsample_bytree":0.85}; rf_params={"n_estimators":220,"max_depth":12,"min_samples_leaf":2}; et_params={"n_estimators":220,"max_depth":14,"min_samples_leaf":2}
    else:
        xgb_params={"n_estimators":350,"learning_rate":0.025,"max_depth":4,"min_child_weight":2,"subsample":0.90,"colsample_bytree":0.90}; rf_params={"n_estimators":300,"max_depth":15,"min_samples_leaf":1}; et_params={"n_estimators":300,"max_depth":18,"min_samples_leaf":1}
    return {"XGB":XGBRegressor(**xgb_params,objective="reg:squarederror",random_state=42,n_jobs=2,verbosity=0),"RF":RandomForestRegressor(**rf_params,random_state=42,n_jobs=2),"ET":ExtraTreesRegressor(**et_params,random_state=42,n_jobs=2)}

def create_direction_model(variant="A"):
    if variant=="A": return XGBClassifier(n_estimators=180,learning_rate=0.04,max_depth=4,min_child_weight=2,subsample=0.85,colsample_bytree=0.85,objective="multi:softprob",eval_metric="mlogloss",random_state=42,n_jobs=2,verbosity=0)
    return XGBClassifier(n_estimators=280,learning_rate=0.03,max_depth=3,min_child_weight=2,subsample=0.9,colsample_bytree=0.9,objective="multi:softprob",eval_metric="mlogloss",random_state=43,n_jobs=2,verbosity=0)

def _walk_forward_splits(n,min_train=WALK_FORWARD_MIN_TRAIN,folds=WALK_FORWARD_FOLDS,embargo=WALK_FORWARD_EMBARGO):
    if n<=min_train+embargo+1:return []
    remaining=n-min_train-embargo; step=max(1,remaining//max(folds,1)); out=[]
    for k in range(folds):
        val_start=min_train+k*step; val_end=min(n,val_start+step); train_end=val_start-embargo
        if val_end>val_start and train_end>=min_train:out.append((train_end,val_start,val_end))
    if not out:
        val_start=max(min_train,n-max(20,int(n*VALIDATION_FRACTION)));out=[(max(min_train,val_start-embargo),val_start,n)]
    return out

def fit_target_ensemble(X,y,variant="A"):
    if len(X)<100:raise ValueError("Not enough samples")
    splits=_walk_forward_splits(len(X))
    if not splits:raise ValueError("Not enough samples for walk-forward validation")
    preds={name:[] for name in create_regressors(variant)};actual=[]
    for train_end,val_start,val_end in splits:
        actual.extend(y.iloc[val_start:val_end].to_numpy())
        for name,model in create_regressors(variant).items():
            model.fit(X.iloc[:train_end],y.iloc[:train_end]);preds[name].extend(model.predict(X.iloc[val_start:val_end]))
    actual=np.asarray(actual,float);errors={n:safe_mape(actual,np.asarray(p,float)) for n,p in preds.items()};inv={n:1/max(e,0.0001) for n,e in errors.items()};total=sum(inv.values());weights={n:v/total for n,v in inv.items()}
    ensemble=sum((weights[n]*np.asarray(p) for n,p in preds.items()),np.zeros(len(actual)));mape=safe_mape(actual,ensemble);err=normalized_mae(actual,ensemble)
    final_models=create_regressors(variant)
    for model in final_models.values():model.fit(X,y)
    return {"models":final_models,"weights":weights,"validation_mape":mape,"validation_error":err,"component_errors":errors,"validation_start":str(X.index[splits[0][1]].date()),"validation_end":str(X.index[splits[-1][2]-1].date()),"validation_samples":len(actual),"validation_method":"walk-forward","validation_folds":len(splits),"embargo_samples":WALK_FORWARD_EMBARGO}

def predict_ensemble(bundle,X):
    predictions={name:model.predict(X) for name,model in bundle["models"].items()};output=np.zeros(len(X))
    for name,prediction in predictions.items():output+=bundle["weights"][name]*prediction
    return output,predictions

def model_agreement(predictions,final_prediction):
    values=np.column_stack(list(predictions.values()));spread=np.std(values,axis=1);denominator=np.maximum(np.abs(final_prediction),1e-8);return np.clip(100*(1-np.clip(spread/denominator,0,1)),0,100)

def fit_direction_model(X,y,variant="A"):
    splits=_walk_forward_splits(len(X));correct=[]
    if not splits:raise ValueError("Not enough samples for walk-forward direction validation")
    for train_end,val_start,val_end in splits:
        model=create_direction_model(variant) if len(np.unique(y.iloc[:train_end]))>1 else RandomForestClassifier(n_estimators=150,random_state=42,n_jobs=2)
        model.fit(X.iloc[:train_end],y.iloc[:train_end]);correct.extend((model.predict(X.iloc[val_start:val_end])==y.iloc[val_start:val_end]).astype(float))
    final_model=create_direction_model(variant) if len(np.unique(y))>1 else RandomForestClassifier(n_estimators=150,random_state=42,n_jobs=2);final_model.fit(X,y)
    return {"model":final_model,"validation_accuracy":float(np.mean(correct)*100) if correct else 50.0,"validation_method":"walk-forward","validation_folds":len(splits),"embargo_samples":WALK_FORWARD_EMBARGO}
