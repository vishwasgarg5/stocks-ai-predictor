import numpy as np
import pandas as pd

from .config import (
    HISTORY_PERIOD,
)

from .market_data import (
    get_completed_session_date,
    get_previous_session_date,
    download_many,
    get_row_for_date,
    get_previous_row,
)

from .ledger import (
    load_predictions,
    prediction_exists,
    evaluation_exists,
    save_evaluation,
    append_daily_metrics,
    rebuild_stock_reliability,
    latest_prediction_date,
)

from .retraining import (
    compare_variants,
)

from .telegram_report import (
    send_telegram,
    evening_report,
)

from .utils import (
    today_ist,
    direction_from_prices,
    is_weekday,
    safe_mape,
)


def calculate_cumulative_metrics():
    from .config import EVALUATIONS_DIR

    files = sorted(
        EVALUATIONS_DIR.glob(
            "evaluation_*.csv"
        )
    )

    frames = []

    for path in files:
        try:
            df = pd.read_csv(path)

            if not df.empty:
                frames.append(df)

        except Exception:
            continue

    if not frames:
        return {
            "Samples": 0,
            "OpenMAPE": 0,
            "HighMAPE": 0,
            "LowMAPE": 0,
            "CloseMAPE": 0,
            "OverallMAPE": 0,
            "DirectionAccuracy": 0,
        }

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    open_mape = data[
        "APE_Open"
    ].abs().mean()

    high_mape = data[
        "APE_High"
    ].abs().mean()

    low_mape = data[
        "APE_Low"
    ].abs().mean()

    close_mape = data[
        "APE_Close"
    ].abs().mean()

    overall = np.mean(
        [
            open_mape,
            high_mape,
            low_mape,
            close_mape,
        ]
    )

    direction_accuracy = (
        data["DirectionCorrect"].mean()
        * 100
    )

    return {
        "Samples": len(data),
        "OpenMAPE": open_mape,
        "HighMAPE": high_mape,
        "LowMAPE": low_mape,
        "CloseMAPE": close_mape,
        "OverallMAPE": overall,
        "DirectionAccuracy": direction_accuracy,
    }


def run():
    today = today_ist()

    if not is_weekday():
        print(
            "Weekend. Evening evaluation skipped."
        )
        return

    market_date = get_completed_session_date(
        "evening"
    )

    if market_date is None:
        print(
            "No completed market session."
        )
        return

    # Exact prediction date first.
    prediction_date = latest_prediction_date(
        market_date
    )

    if prediction_date is None:
        print(
            "No morning prediction ledger found."
        )
        return

    predictions = load_predictions(
        prediction_date
    )

    if predictions.empty:
        print(
            "Morning prediction file is empty."
        )
        return

    # Do NOT rerun stock selection.
    symbols = predictions[
        "Symbol"
    ].astype(str).tolist()

    # Prevent duplicate evaluation.
    if evaluation_exists(
        market_date
    ):
        print(
            f"Evaluation already exists for "
            f"{market_date}. Skipping."
        )
        return

    data_map = download_many(
        symbols,
        HISTORY_PERIOD,
        workers=5,
    )

    evaluation_rows = []

    for _, prediction in predictions.iterrows():
        symbol = prediction["Symbol"]

        df = data_map.get(symbol)

        if df is None:
            print(
                f"{symbol}: no actual data"
            )
            continue

        actual = get_row_for_date(
            df,
            market_date,
        )

        previous = get_previous_row(
            df,
            market_date,
        )

        if actual is None:
            print(
                f"{symbol}: actual session missing"
            )
            continue

        if previous is None:
            print(
                f"{symbol}: previous close missing"
            )
            continue

        pred_direction = prediction[
            "Direction"
        ]

        actual_direction = direction_from_prices(
            previous["Close"],
            actual["Close"],
        )

        row = {
            "MarketDate": str(market_date),
            "Symbol": symbol,

            "Pred_Open": prediction[
                "Pred_Open"
            ],
            "Pred_High": prediction[
                "Pred_High"
            ],
            "Pred_Low": prediction[
                "Pred_Low"
            ],
            "Pred_Close": prediction[
                "Pred_Close"
            ],

            "Actual_Open": float(
                actual["Open"]
            ),
            "Actual_High": float(
                actual["High"]
            ),
            "Actual_Low": float(
                actual["Low"]
            ),
            "Actual_Close": float(
                actual["Close"]
            ),

            "Pred_Direction": pred_direction,
            "Actual_Direction": actual_direction,

            "DirectionCorrect": (
                pred_direction
                == actual_direction
            ),
        }

        for target in [
            "Open",
            "High",
            "Low",
            "Close",
        ]:
            predicted = row[
                f"Pred_{target}"
            ]

            actual_value = row[
                f"Actual_{target}"
            ]

            row[
                f"Diff_{target}"
            ] = (
                actual_value
                - predicted
            )

            row[
                f"APE_{target}"
            ] = (
                (actual_value - predicted)
                / max(
                    abs(actual_value),
                    1e-8,
                )
                * 100
            )

        evaluation_rows.append(row)

    if not evaluation_rows:
        print(
            "No stocks could be evaluated."
        )
        return

    evaluation = pd.DataFrame(
        evaluation_rows
    )

    save_evaluation(
        evaluation,
        market_date,
    )

    metrics = calculate_cumulative_metrics()

    append_daily_metrics(
        {
            "MarketDate": str(
                market_date
            ),
            **metrics,
        }
    )

    rebuild_stock_reliability()

    # ---------------------------------------------------------
    # Retraining
    # ---------------------------------------------------------

    previous_session = (
        get_previous_session_date(
            market_date
        )
    )

    if previous_session is None:
        retraining = {
            "Retrained": False,
            "Decision": "NO PREVIOUS SESSION",
        }

    else:
        retraining_data = download_many(
            symbols,
            HISTORY_PERIOD,
            workers=5,
        )

        retraining = compare_variants(
            retraining_data,
            symbols,
            previous_session,
        )

    report = evening_report(
        market_date,
        evaluation,
        metrics,
        retraining,
    )

    send_telegram(report)

    print(report)


if __name__ == "__main__":
    run()
