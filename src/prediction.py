"""
STAGE 4 PREDICTION API
======================
Easy access: Stage 2 next-day OHLC + Stage 3B multi-horizon predictions.
Stage 3B is additive; existing OHLC prediction behavior is retained.
"""
import numpy as np
import pandas as pd
from .features import get_feature_columns, prepare_supervised, build_features, technical_score
from .models import TARGETS, fit_target_ensemble, fit_direction_model, predict_ensemble, model_agreement
from .multihorizon import train_horizon_models, predict_horizons

def train_stock_bundle(df,symbol,cutoff_date,variant="A"):
    supervised=prepare_supervised(df,cutoff_date)
    if len(supervised)<150: raise ValueError(f"{symbol}: only {len(supervised)} supervised rows")
    features=get_feature_columns(); X=supervised[features]; target_bundles={}; validation_mape=[]; validation_error=[]
    for target in TARGETS:
        bundle=fit_target_ensemble(X,supervised[f"Target_{target}"],variant); target_bundles[target]=bundle
        validation_mape.append(bundle["validation_mape"]); validation_error.append(bundle["validation_error"])
    direction_bundle=fit_direction_model(X,supervised["Direction"],variant)
    # STAGE 3B: train independent 1/3/5/7/20-session close forecasts.
    horizon_bundle=train_horizon_models(df,cutoff_date)
    return {"symbol":symbol,"variant":variant,"cutoff_date":str(pd.Timestamp(cutoff_date).date()),"features":features,"targets":target_bundles,"direction":direction_bundle,"horizons":horizon_bundle,"validation_mape":float(np.mean(validation_mape)),"validation_error":float(np.mean(validation_error)),"direction_validation_accuracy":direction_bundle["validation_accuracy"],"training_samples":len(supervised)}

def predict_stock(df,bundle,cutoff_date):
    features_df=build_features(df)
    if features_df.empty: raise ValueError("No features")
    cutoff=pd.Timestamp(cutoff_date); features_df=features_df[features_df.index<=cutoff]; usable=features_df[bundle["features"]].dropna()
    if usable.empty: raise ValueError("No usable latest feature row")
    latest=usable.iloc[[-1]]; predictions={}; agreements=[]
    for target in TARGETS:
        final_prediction,component_predictions=predict_ensemble(bundle["targets"][target],latest); value=float(final_prediction[0]); predictions[target]=value
        agreements.append(float(model_agreement(component_predictions,final_prediction)[0]))
    direction_model=bundle["direction"]["model"]; direction_label=int(direction_model.predict(latest)[0]); direction_probability=50.0
    try: direction_probability=float(np.max(direction_model.predict_proba(latest)[0])*100)
    except Exception: pass
    current_close=float(latest["Close"].iloc[0]); expected_return=predictions["Close"]/current_close-1; confidence=float(0.65*np.mean(agreements)+0.35*direction_probability)
    direction_map={0:"DOWN",1:"NEUTRAL",2:"UP"}
    return {"Current_Price":current_close,"Pred_Open":predictions["Open"],"Pred_High":predictions["High"],"Pred_Low":predictions["Low"],"Pred_Close":predictions["Close"],"Expected_Return":expected_return*100,"Direction":direction_map.get(direction_label,"NEUTRAL"),"Direction_Confidence":direction_probability,"Confidence":confidence,"TechnicalScore":technical_score(df[df.index<=cutoff]),"ValidationMAPE":bundle["validation_mape"],"ValidationError":bundle["validation_error"],"DirectionValidationAccuracy":bundle["direction_validation_accuracy"]}

def add_multihorizon_predictions(df,bundle,cutoff_date):
    """STAGE 3B: return 1/3/5/7/20-session close forecasts."""
    return predict_horizons(df,bundle["horizons"],cutoff_date)
