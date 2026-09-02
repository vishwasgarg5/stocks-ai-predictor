import pandas as pd

from .config import (
    PRESCREEN_N,
    HISTORY_PERIOD,
    JUMP_CANDIDATE_N,
    INTRADAY_TOP_N,
    MODEL_VERSION,
)

from .market_data import (
    load_universe,
    download_many,
    filter_liquid_universe,
    get_completed_session_date,
    get_market_regime,
)

from .features import technical_score

from .prediction import (
    train_stock_bundle,
    predict_stock,
)

from .selection import select_top_stocks

from .jump_engine import (
    generate_jump_watchlist,
)

from .intraday_engine import (
    generate_intraday_watchlist,
)

from .ledger import (
    prediction_exists,
    save_predictions,
    save_jump_predictions,
    save_intraday_predictions,
)

from .retraining import load_model_state

from .telegram_report import send_telegram, morning_report

from .utils import (
    today_ist,
    is_weekday,
    schedule_status,
)


def run():
    prediction_date = today_ist()

    if not is_weekday():
        print(
            "Weekend. Morning prediction skipped."
        )
        return

    # Prevent duplicate scheduled/manual runs.
    if prediction_exists(
        prediction_date
    ):
        print(
            f"Prediction already exists for "
            f"{prediction_date}. Skipping."
        )
        return

    cutoff_date = get_completed_session_date(
        "morning"
    )

    if cutoff_date is None:
        raise RuntimeError(
            "Unable to determine market cutoff."
        )

    universe = load_universe()

    raw_data = download_many(
        universe,
        HISTORY_PERIOD,
        workers=8,
    )

    data_map = filter_liquid_universe(
        raw_data
    )

    if len(data_map) < 20:
        raise RuntimeError(
            "Too few liquid stocks."
        )

    regime_info = get_market_regime(
        cutoff_date
    )

    regime = regime_info["name"]

    state = load_model_state()

    variant = state.get(
        "active_variant",
        "A",
    )

    # ---------------------------------------------------------
    # Fast prescreen
    # ---------------------------------------------------------

    scored = []

    for symbol, df in data_map.items():
        try:
            score = technical_score(
                df[df.index <= pd.Timestamp(cutoff_date)]
            )

            scored.append(
                (
                    symbol,
                    score,
                )
            )

        except Exception:
            continue

    scored.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    candidate_symbols = [
        x[0]
        for x in scored[:PRESCREEN_N]
    ]

    candidate_rows = []

    # ---------------------------------------------------------
    # Next-day prediction
    # ---------------------------------------------------------

    for symbol in candidate_symbols:
        df = data_map.get(symbol)

        if df is None:
            continue

        try:
            bundle = train_stock_bundle(
                df,
                symbol,
                cutoff_date,
                variant,
            )

            result = predict_stock(
                df,
                bundle,
                cutoff_date,
            )

            candidate_rows.append(
                {
                    "Symbol": symbol,
                    **result,
                    "ModelVariant": variant,
                    "ModelVersion": MODEL_VERSION,
                    "DataCutoff": str(
                        cutoff_date
                    ),
                }
            )

        except Exception as exc:
            print(
                f"{symbol}: prediction failed: {exc}"
            )

    if len(candidate_rows) < 5:
        raise RuntimeError(
            "Unable to generate five predictions."
        )

    candidates = pd.DataFrame(
        candidate_rows
    )

    selected = select_top_stocks(
        candidates,
        top_n=5,
        regime=regime,
    )

    selected["PredictionDate"] = str(
        prediction_date
    )

    save_predictions(
        selected,
        prediction_date,
        {
            "Stage": "Stage 2",
            "PredictionDate": str(
                prediction_date
            ),
            "DataCutoff": str(cutoff_date),
            "ModelVariant": variant,
            "Regime": regime,
            "SelectedStocks": selected[
                "Symbol"
            ].tolist(),
        },
    )

    # ---------------------------------------------------------
    # 7-day jump engine
    # ---------------------------------------------------------

    jump_data = {}

    for symbol in candidate_symbols[
        :JUMP_CANDIDATE_N
    ]:
        if symbol in data_map:
            jump_data[symbol] = data_map[
                symbol
            ]

    jump_watchlist = generate_jump_watchlist(
        jump_data,
        cutoff_date,
        variant,
    )

    if not jump_watchlist.empty:
        save_jump_predictions(
            jump_watchlist,
            prediction_date,
        )

    # ---------------------------------------------------------
    # Intraday engine
    # ---------------------------------------------------------

    intraday = generate_intraday_watchlist(
        list(data_map.keys())[
            : min(
                len(data_map),
                300,
            )
        ]
    )

    if not intraday.empty:
        save_intraday_predictions(
            intraday,
            prediction_date,
        )

    # ---------------------------------------------------------
    # Telegram
    # ---------------------------------------------------------

    report = morning_report(
        prediction_date=prediction_date,
        cutoff_date=cutoff_date,
        schedule_status=schedule_status(
            "morning"
        ),
        regime=regime,
        model_variant=variant,
        selected=selected,
        jump_watchlist=jump_watchlist,
        intraday=intraday,
    )

    send_telegram(report)

    print(report)


if __name__ == "__main__":
    run()
