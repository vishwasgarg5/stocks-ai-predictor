import numpy as np
import pandas as pd

from .features import (
    get_feature_columns,
    prepare_supervised,
    build_features,
    technical_score,
)

from .models import (
    TARGETS,
    fit_target_ensemble,
    fit_direction_model,
    predict_ensemble,
    model_agreement,
)


def train_stock_bundle(
    df,
    symbol,
    cutoff_date,
    variant="A",
):
    supervised = prepare_supervised(
        df,
        cutoff_date,
    )

    if len(supervised) < 150:
        raise ValueError(
            f"{symbol}: only {len(supervised)} supervised rows"
        )

    features = get_feature_columns()

    X = supervised[features]

    target_bundles = {}

    validation_mape = []
    validation_error = []

    for target in TARGETS:
        y = supervised[f"Target_{target}"]

        bundle = fit_target_ensemble(
            X,
            y,
            variant,
        )

        target_bundles[target] = bundle

        validation_mape.append(
            bundle["validation_mape"]
        )

        validation_error.append(
            bundle["validation_error"]
        )

    direction_bundle = fit_direction_model(
        X,
        supervised["Direction"],
        variant,
    )

    return {
        "symbol": symbol,
        "variant": variant,
        "cutoff_date": str(pd.Timestamp(cutoff_date).date()),
        "features": features,
        "targets": target_bundles,
        "direction": direction_bundle,
        "validation_mape": float(np.mean(validation_mape)),
        "validation_error": float(np.mean(validation_error)),
        "direction_validation_accuracy": direction_bundle[
            "validation_accuracy"
        ],
        "training_samples": len(supervised),
    }


def predict_stock(
    df,
    bundle,
    cutoff_date,
):
    features_df = build_features(df)

    if features_df.empty:
        raise ValueError("No features")

    cutoff = pd.Timestamp(cutoff_date)

    features_df = features_df[
        features_df.index <= cutoff
    ]

    usable = features_df[
        bundle["features"]
    ].dropna()

    if usable.empty:
        raise ValueError("No usable latest feature row")

    latest = usable.iloc[[-1]]

    predictions = {}

    agreements = []

    for target in TARGETS:
        final_prediction, component_predictions = (
            predict_ensemble(
                bundle["targets"][target],
                latest,
            )
        )

        value = float(final_prediction[0])

        predictions[target] = value

        agreement = model_agreement(
            component_predictions,
            final_prediction,
        )[0]

        agreements.append(float(agreement))

    direction_model = bundle["direction"]["model"]

    direction_label = int(
        direction_model.predict(latest)[0]
    )

    direction_probability = 50.0

    try:
        probabilities = direction_model.predict_proba(
            latest
        )[0]

        direction_probability = float(
            np.max(probabilities) * 100
        )

    except Exception:
        pass

    current_close = float(
        latest["Close"].iloc[0]
    )

    expected_return = (
        predictions["Close"] / current_close
    ) - 1

    confidence = float(
        0.65 * np.mean(agreements)
        + 0.35 * direction_probability
    )

    direction_map = {
        0: "DOWN",
        1: "NEUTRAL",
        2: "UP",
    }

    return {
        "Current_Price": current_close,
        "Pred_Open": predictions["Open"],
        "Pred_High": predictions["High"],
        "Pred_Low": predictions["Low"],
        "Pred_Close": predictions["Close"],
        "Expected_Return": expected_return * 100,
        "Direction": direction_map.get(
            direction_label,
            "NEUTRAL",
        ),
        "Direction_Confidence": direction_probability,
        "Confidence": confidence,
        "TechnicalScore": technical_score(
            df[df.index <= cutoff]
        ),
        "ValidationMAPE": bundle["validation_mape"],
        "ValidationError": bundle["validation_error"],
        "DirectionValidationAccuracy": bundle[
            "direction_validation_accuracy"
        ],
    }
