import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from .config import VALIDATION_FRACTION
from .utils import safe_mape, normalized_mae

# OHLC remain direct price targets. Volume is handled separately in log1p space.
TARGETS=["Open","High","Low","Close"]

def create_regressors(variant="A"):
    if variant=="A":
        xgb_params={"n_estimators":250,"learning_rate":0.035,"max_depth":5,"min_child_weight":3,"subsample":0.85,"colsample_bytree":0.85}
        rf_params={"n_estimators":220,"max_depth":12,"min_samples_leaf":2}; et_params={"n_estimators":220,"max_depth":14,"min_samples_leaf":2}
    else:
        xgb_params={"n_estimators":350,"learning_rate":0.025,"max_depth":4,"min_child_weight":2,"subsample":0.90,"colsample_bytree":0.90}
        rf_params={"n_estimators":300,"max_depth":15,"min_samples_leaf":1}; et_params={"n_estimators":300,"max_depth":18,"min_samples_leaf":1}
    return {"XGB":XGBRegressor(**xgb_params,objective="reg:squarederror",random_state=42,n_jobs=2,verbosity=0),"RF":RandomForestRegressor(**rf_params,random_state=42,n_jobs=2),"ET":ExtraTreesRegressor(**et_params,random_state=42,n_jobs=2)}

def create_direction_model(variant="A"):
    if variant=="A": return XGBClassifier(n_estimators=180,learning_rate=0.04,max_depth=4,min_child_weight=2,subsample=0.85,colsample_bytree=0.85,objective="multi:softprob",eval_metric="mlogloss",random_state=42,n_jobs=2,verbosity=0)
    return XGBClassifier(n_estimators=280,learning_rate=0.03,max_depth=3,min_child_weight=2,subsample=0.9,colsample_bytree=0.9,objective="multi:softprob",eval_metric="mlogloss",random_state=43,n_jobs=2,verbosity=0)

def fit_target_ensemble(X,y,variant="A"):
    if len(X)<100: raise ValueError("Not enough samples")
    split=max(50,min(int(len(X)*(1-VALIDATION_FRACTION)),len(X)-20)); X_train=X.iloc[:split]; X_val=X.iloc[split:]; y_train=y.iloc[:split]; y_val=y.iloc[split:]
    validation_models=create_regressors(variant); validation_predictions={}; errors={}
    for name,model in validation_models.items():
        model.fit(X_train,y_train); prediction=model.predict(X_val); validation_predictions[name]=prediction; errors[name]=safe_mape(y_val,prediction)
    inverse_errors={name:1/max(error,0.0001) for name,error in errors.items()}; total=sum(inverse_errors.values()); weights={name:value/total for name,value in inverse_errors.items()}
    ensemble_validation=sum((weights[name]*prediction for name,prediction in validation_predictions.items()),np.zeros(len(X_val)))
    ensemble_mape=safe_mape(y_val,ensemble_validation); ensemble_error=normalized_mae(y_val,ensemble_validation)
    final_models=create_regressors(variant)
    for model in final_models.values(): model.fit(X,y)
    return {"models":final_models,"weights":weights,"validation_mape":ensemble_mape,"validation_error":ensemble_error,"component_errors":errors,"validation_start":str(X_val.index[0].date()),"validation_end":str(X_val.index[-1].date()),"validation_samples":len(X_val)}

def predict_ensemble(bundle,X):
    predictions={name:model.predict(X) for name,model in bundle["models"].items()}; output=np.zeros(len(X))
    for name,prediction in predictions.items(): output += bundle["weights"][name]*prediction
    return output,predictions

def model_agreement(predictions,final_prediction):
    values=np.column_stack(list(predictions.values())); spread=np.std(values,axis=1); denominator=np.maximum(np.abs(final_prediction),1e-8); return np.clip(100*(1-np.clip(spread/denominator,0,1)),0,100)

def fit_direction_model(X,y,variant="A"):
    split=max(50,min(int(len(X)*(1-VALIDATION_FRACTION)),len(X)-20)); X_train=X.iloc[:split]; X_val=X.iloc[split:]; y_train=y.iloc[:split]; y_val=y.iloc[split:]
    unique_classes=len(np.unique(y_train)); model=RandomForestClassifier(n_estimators=150,random_state=42,n_jobs=2) if unique_classes<2 else create_direction_model(variant)
    model.fit(X_train,y_train); validation_accuracy=(model.predict(X_val)==y_val).mean()*100; params={k:v for k,v in model.get_params().items() if k!="n_jobs"}; final_model=model.__class__(**params,n_jobs=2); final_model.fit(X,y)
    return {"model":final_model,"validation_accuracy":float(validation_accuracy)}
