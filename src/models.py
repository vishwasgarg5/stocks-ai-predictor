import numpy as np

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    RandomForestClassifier,
)

from xgboost import (
    XGBRegressor,
    XGBClassifier,
)

from .config import VALIDATION_FRACTION
from .utils import safe_mape, normalized_mae


TARGETS = [
    "Open",
    "High",
    "Low",
    "Close",
]


def create_regressors(variant="A"):
    if variant == "A":
        xgb_params = {
            "n_estimators": 250,
            "learning_rate": 0.035,
            "max_depth": 5,
            "min_child_weight": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        }

        rf_params = {
            "n_estimators": 220,
            "max_depth": 12,
            "min_samples_leaf": 2,
        }

        et_params = {
            "n_estimators": 220,
            "max_depth": 14,
            "min_samples_leaf": 2,
        }

    else:
        xgb_params = {
            "n_estimators": 350,
            "learning_rate": 0.025,
            "max_depth": 4,
            "min_child_weight": 2,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
        }

        rf_params = {
            "n_estimators": 300,
            "max_depth": 15,
            "min_samples_leaf": 1,
        }

        et_params = {
            "n_estimators": 300,
            "max_depth": 18,
            "min_samples_leaf": 1,
        }

    models = {
        "XGB": XGBRegressor(
            **xgb_params,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=2,
            verbosity=0,
        ),

        "RF": RandomForestRegressor(
            **rf_params,
            random_state=42,
            n_jobs=2,
        ),

        "ET": ExtraTreesRegressor(
            **et_params,
            random_state=42,
            n_jobs=2,
        ),
    }

    return models


def create_direction_model(variant="A"):
    if variant == "A":
        return XGBClassifier(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=4,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=2,
            verbosity=0,
        )

    return XGBClassifier(
        n_estimators=280,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=43,
        n_jobs=2,
        verbosity=0,
    )


def fit_target_ensemble(X, y, variant="A"):
    if len(X) < 100:
        raise ValueError("Not enough samples")

    split = int(len(X) * (1 - VALIDATION_FRACTION))

    split = max(50, min(split, len(X) - 20))

    X_train = X.iloc[:split]
    X_val = X.iloc[split:]

    y_train = y.iloc[:split]
    y_val = y.iloc[split:]

    validation_models = create_regressors(variant)

    validation_predictions = {}
    errors = {}

    for name, model in validation_models.items():
        model.fit(X_train, y_train)

        prediction = model.predict(X_val)

        validation_predictions[name] = prediction

        errors[name] = safe_mape(
            y_val,
            prediction,
        )

    inverse_errors = {
        name: 1 / max(error, 0.0001)
        for name, error in errors.items()
    }

    total = sum(inverse_errors.values())

    weights = {
        name: value / total
        for name, value in inverse_errors.items()
    }

    ensemble_validation = np.zeros(len(X_val))

    for name, prediction in validation_predictions.items():
        ensemble_validation += (
            weights[name] * prediction
        )

    ensemble_mape = safe_mape(
        y_val,
        ensemble_validation,
    )

    ensemble_error = normalized_mae(
        y_val,
        ensemble_validation,
    )

    # Refit all component models on the full historical set.
    final_models = create_regressors(variant)

    for model in final_models.values():
        model.fit(X, y)

    return {
        "models": final_models,
        "weights": weights,
        "validation_mape": ensemble_mape,
        "validation_error": ensemble_error,
        "component_errors": errors,
        "validation_start": str(X_val.index[0].date()),
        "validation_end": str(X_val.index[-1].date()),
        "validation_samples": len(X_val),
    }


def predict_ensemble(bundle, X):
    predictions = {}

    for name, model in bundle["models"].items():
        predictions[name] = model.predict(X)

    output = np.zeros(len(X))

    for name, prediction in predictions.items():
        output += (
            bundle["weights"][name] * prediction
        )

    return output, predictions


def model_agreement(predictions, final_prediction):
    values = np.column_stack(
        list(predictions.values())
    )

    spread = np.std(values, axis=1)

    denominator = np.maximum(
        np.abs(final_prediction),
        1e-8,
    )

    relative_spread = spread / denominator

    agreement = 100 * (
        1 - np.clip(relative_spread, 0, 1)
    )

    return np.clip(agreement, 0, 100)


def fit_direction_model(X, y, variant="A"):
    split = int(len(X) * (1 - VALIDATION_FRACTION))

    split = max(50, min(split, len(X) - 20))

    X_train = X.iloc[:split]
    X_val = X.iloc[split:]

    y_train = y.iloc[:split]
    y_val = y.iloc[split:]

    unique_classes = len(np.unique(y_train))

    if unique_classes < 2:
        model = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=2,
        )

    else:
        model = create_direction_model(variant)

    model.fit(X_train, y_train)

    validation_accuracy = (
        model.predict(X_val) == y_val
    ).mean() * 100

    final_model = model.__class__(
        **{
            key: value
            for key, value in model.get_params().items()
            if key != "n_jobs"
        },
        n_jobs=2,
    )

    final_model.fit(X, y)

    return {
        "model": final_model,
        "validation_accuracy": float(validation_accuracy),
    }
