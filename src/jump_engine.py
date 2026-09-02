import numpy as np
import pandas as pd

from .config import (
    JUMP_THRESHOLD,
    JUMP_HORIZON_DAYS,
    JUMP_TOP_N,
    JUMP_CANDIDATE_N,
)

from .features import (
    build_features,
    technical_score,
)

from .prediction import train_stock_bundle


def calculate_jump_score(
    current_price,
    predicted_close,
    predicted_high,
    confidence,
    technical,
):
    close_return = (
        predicted_close / current_price - 1
    )

    high_return = (
        predicted_high / current_price - 1
    )

    target_score = np.clip(
        high_return / JUMP_THRESHOLD * 100,
        0,
        100,
    )

    close_score = np.clip(
        close_return / JUMP_THRESHOLD * 100,
        0,
        100,
    )

    score = (
        0.30 * target_score
        + 0.20 * close_score
        + 0.25 * confidence
        + 0.25 * technical
    )

    return float(np.clip(score, 0, 100))


def generate_jump_watchlist(
    data_map,
    cutoff_date,
    variant="A",
):
    candidates = []

    for symbol, df in data_map.items():
        try:
            df = df[
                df.index
                <= pd.Timestamp(cutoff_date)
            ]

            if len(df) < 150:
                continue

            tech = technical_score(df)

            # Fast momentum gate.
            if tech < 55:
                continue

            bundle = train_stock_bundle(
                df,
                symbol,
                cutoff_date,
                variant,
            )

            features = build_features(df)

            latest = features.dropna().iloc[-1]

            current = float(
                latest["Close"]
            )

            # Predict 1 day first.
            from .models import (
                TARGETS,
                predict_ensemble,
            )

            predictions = {}

            for target in TARGETS:
                value, _ = predict_ensemble(
                    bundle["targets"][target],
                    latest.to_frame().T[
                        bundle["features"]
                    ],
                )

                predictions[target] = float(
                    value[0]
                )

            confidence = float(
                bundle["direction_validation_accuracy"]
            )

            expected_high = (
                predictions["High"]
                / current
                - 1
            )

            expected_close = (
                predictions["Close"]
                / current
                - 1
            )

            # Approximate multi-day potential by
            # compounding the predicted daily return.
            daily_return = max(
                expected_close,
                0,
            )

            seven_day_potential = (
                (1 + daily_return)
                ** JUMP_HORIZON_DAYS
                - 1
            )

            max_potential = max(
                expected_high,
                seven_day_potential,
            )

            probability = np.clip(
                50
                + max_potential * 250
                + (confidence - 50) * 0.35,
                0,
                95,
            )

            score = calculate_jump_score(
                current,
                predictions["Close"],
                predictions["High"],
                probability,
                tech,
            )

            candidates.append(
                {
                    "Symbol": symbol,
                    "Current_Price": current,
                    "Predicted_Close_1D": predictions[
                        "Close"
                    ],
                    "Predicted_High_1D": predictions[
                        "High"
                    ],
                    "Expected_1D_Return": expected_close
                    * 100,
                    "Estimated_7D_Upside": seven_day_potential
                    * 100,
                    "Jump_Probability": probability,
                    "Confidence": confidence,
                    "TechnicalScore": tech,
                    "JumpScore": score,
                    "Target_Level": current
                    * (1 + JUMP_THRESHOLD),
                    "Status": "OPEN",
                    "Remaining_Days": JUMP_HORIZON_DAYS,
                }
            )

        except Exception as exc:
            print(
                f"{symbol}: jump prediction failed: {exc}"
            )

    if not candidates:
        return pd.DataFrame()

    result = pd.DataFrame(candidates)

    result = result.sort_values(
        [
            "JumpScore",
            "Jump_Probability",
        ],
        ascending=False,
    )

    return result.head(
        JUMP_TOP_N
    ).reset_index(drop=True)


def evaluate_jump_prediction(
    prediction_row,
    actual_history,
):
    symbol = prediction_row["Symbol"]

    current_price = float(
        prediction_row["Current_Price"]
    )

    target = float(
        prediction_row["Target_Level"]
    )

    start_date = pd.Timestamp(
        prediction_row["Prediction_Date"]
    )

    history = actual_history.copy()

    history = history[
        history.index > start_date
    ].head(JUMP_HORIZON_DAYS)

    if history.empty:
        return {
            "Symbol": symbol,
            "Hit": False,
            "DaysToHit": None,
            "MaxUpside": None,
            "ObservationDays": 0,
        }

    max_high = float(
        history["High"].max()
    )

    hit_rows = history[
        history["High"] >= target
    ]

    if not hit_rows.empty:
        first_hit = hit_rows.index[0]

        days = (
            list(history.index).index(first_hit)
            + 1
        )

        return {
            "Symbol": symbol,
            "Hit": True,
            "DaysToHit": days,
            "MaxUpside": (
                max_high / current_price - 1
            )
            * 100,
            "ObservationDays": len(history),
        }

    return {
        "Symbol": symbol,
        "Hit": False,
        "DaysToHit": None,
        "MaxUpside": (
            max_high / current_price - 1
        )
        * 100,
        "ObservationDays": len(history),
    }
