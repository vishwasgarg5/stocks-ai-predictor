"""Stage 4.2 prediction API: next-session OHLCV + optional multi-horizon forecasts."""
import numpy as np
import pandas as pd
from .features import get_feature_columns, prepare_supervised, build_features, technical_score
from .models import TARGETS, fit_target_ensemble, fit_direction_model, predict_ensemble, model_agreement
from .multihorizon import train_horizon_models, predict_horizons


def train_stock_bundle(df, symbol, cutoff_date, variant="A", train_horizons=False):
    supervised = prepare_supervised(df, cutoff_date)
    if len(supervised) < 150:
        raise ValueError(f"{symbol}: only {len(supervised)} supervised rows")
    features = get_feature_columns()
    X = supervised[features]
    target_bundles, validation_mape, validation_error = {}, [], []
    for target in TARGETS:
        bundle = fit_target_ensemble(X, supervised[f"Target_{target}"], variant)
        target_bundles[target] = bundle
        validation_mape.append(bundle["validation_mape"])
        validation_error.append(bundle["validation_error"])
    direction_bundle = fit_direction_model(X, supervised["Direction"], variant)
    horizon_bundle = train_horizon_models(df, cutoff_date) if train_horizons else None
    return {"symbol": symbol, "variant": variant, "cutoff_date": str(pd.Timestamp(cutoff_date).date()),
            "features": features, "targets": target_bundles, "direction": direction_bundle,
            "horizons": horizon_bundle, "validation_mape": float(np.mean(validation_mape)),
            "validation_error": float(np.mean(validation_error)),
            "direction_validation_accuracy": direction_bundle["validation_accuracy"],
            "training_samples": len(supervised)}


def _enforce_ohlc_consistency(predictions):
    o, h, l, c = map(float, (predictions["Open"], predictions["High"], predictions["Low"], predictions["Close"]))
    predictions["High"] = max(h, o, c)
    predictions["Low"] = min(l, o, c)
    return predictions


def predict_stock(df, bundle, cutoff_date):
    features_df = build_features(df)
    if features_df.empty:
        raise ValueError("No features")
    cutoff = pd.Timestamp(cutoff_date)
    features_df = features_df[features_df.index <= cutoff]
    usable = features_df[bundle["features"]].dropna()
    if usable.empty:
        raise ValueError("No usable latest feature row")
    latest = usable.iloc[[-1]]
    predictions, agreements = {}, []
    for target in TARGETS:
        final_prediction, component_predictions = predict_ensemble(bundle["targets"][target], latest)
        value = float(final_prediction[0])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite {target} prediction")
        predictions[target] = max(value, 0.0)
        agreements.append(float(model_agreement(component_predictions, final_prediction)[0]))
    predictions = _enforce_ohlc_consistency(predictions)
    direction_model = bundle["direction"]["model"]
    direction_label = int(direction_model.predict(latest)[0])
    try:
        direction_probability = float(np.max(direction_model.predict_proba(latest)[0]) * 100)
    except Exception:
        direction_probability = 50.0
    current_close = float(latest["Close"].iloc[0])
    expected_return = predictions["Close"] / current_close - 1
    confidence = float(0.65 * np.mean(agreements) + 0.35 * direction_probability)
    direction_map = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
    return {"Current_Price": current_close, "Current_Volume": float(latest["Volume"].iloc[0]),
            "Pred_Open": predictions["Open"], "Pred_High": predictions["High"],
            "Pred_Low": predictions["Low"], "Pred_Close": predictions["Close"],
            "Pred_Volume": predictions["Volume"], "Expected_Return": expected_return * 100,
            "Direction": direction_map.get(direction_label, "NEUTRAL"),
            "Direction_Confidence": direction_probability, "Confidence": confidence,
            "TechnicalScore": technical_score(df[df.index <= cutoff]),
            "ValidationMAPE": bundle["validation_mape"], "ValidationError": bundle["validation_error"],
            "DirectionValidationAccuracy": bundle["direction_validation_accuracy"]}


def add_multihorizon_predictions(df, bundle, cutoff_date):
    if bundle.get("horizons") is None:
        return pd.DataFrame()
    return predict_horizons(df, bundle["horizons"], cutoff_date)
