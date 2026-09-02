import os
import requests
import pandas as pd

from .config import TELEGRAM_MAX_LENGTH
from .utils import (
    format_money,
    format_percent,
    split_messages,
)


def send_telegram(text):
    token = os.getenv(
        "TELEGRAM_BOT_TOKEN"
    )

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID"
    )

    if not token or not chat_id:
        print(
            "Telegram secrets not configured."
        )
        return False

    url = (
        f"https://api.telegram.org/bot"
        f"{token}/sendMessage"
    )

    success = True

    for message in split_messages(
        text,
        TELEGRAM_MAX_LENGTH,
    ):
        try:
            response = requests.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": message,
                },
                timeout=20,
            )

            if response.status_code != 200:
                print(
                    "Telegram error:",
                    response.text,
                )
                success = False

        except Exception as exc:
            print(
                "Telegram exception:",
                exc,
            )
            success = False

    return success


def morning_report(
    prediction_date,
    cutoff_date,
    schedule_status,
    regime,
    model_variant,
    selected,
    jump_watchlist,
    intraday,
):
    lines = [
        "📈 AI NSE STOCK PREDICTION — STAGE 2",
        f"Prediction Date: {prediction_date}",
        f"Data Cutoff: {cutoff_date}",
        f"Schedule: {schedule_status}",
        f"Market Regime: {regime}",
        f"Champion Model: {model_variant}",
        "",
        "🎯 NEXT-DAY TOP 5",
        "",
    ]

    for i, (_, row) in enumerate(
        selected.iterrows(),
        1,
    ):
        lines.extend(
            [
                (
                    f"{i}. {row['Symbol']} | "
                    f"Score {row['Score']:.1f} | "
                    f"{row['Direction']} "
                    f"{row['Direction_Confidence']:.0f}%"
                ),
                (
                    f"   O {row['Pred_Open']:.2f} "
                    f"H {row['Pred_High']:.2f} "
                    f"L {row['Pred_Low']:.2f} "
                    f"C {row['Pred_Close']:.2f}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "🔥 7-DAY +5% JUMP WATCHLIST",
            "",
        ]
    )

    if jump_watchlist is not None and not jump_watchlist.empty:
        for i, (_, row) in enumerate(
            jump_watchlist.iterrows(),
            1,
        ):
            lines.append(
                (
                    f"{i}. {row['Symbol']} | "
                    f"CMP {format_money(row['Current_Price'])} | "
                    f"Target {format_money(row['Target_Level'])} | "
                    f"Potential {format_percent(row['Estimated_7D_Upside'])} | "
                    f"Prob {row['Jump_Probability']:.0f}%"
                )
            )
    else:
        lines.append(
            "No strong +5% candidates today."
        )

    lines.extend(
        [
            "",
            "⚡ INTRADAY TOP 5",
            "",
        ]
    )

    if intraday is not None and not intraday.empty:
        for i, (_, row) in enumerate(
            intraday.iterrows(),
            1,
        ):
            lines.append(
                (
                    f"{i}. {row['Symbol']} | "
                    f"{row['Bias']} | "
                    f"CMP {format_money(row['Current'])} | "
                    f"Target {format_money(row['Target'])} | "
                    f"SL {format_money(row['StopLoss'])} | "
                    f"Conf {row['Confidence']:.0f}%"
                )
            )
    else:
        lines.append(
            "No strong intraday setup today."
        )

    lines.extend(
        [
            "",
            "Stage 2 stores all prediction data "
            "for future learning."
        ]
    )

    return "\n".join(lines)


def evening_report(
    market_date,
    evaluation,
    metrics,
    retraining,
):
    lines = [
        "🌙 AI NSE EVENING REPORT — STAGE 2",
        f"Market Date: {market_date}",
        "",
        "NEXT-DAY PREDICTION EVALUATION",
        "",
    ]

    for _, row in evaluation.iterrows():
        lines.extend(
            [
                (
                    f"{row['Symbol']}"
                ),
                (
                    f"PRED O {row['Pred_Open']:.2f} "
                    f"H {row['Pred_High']:.2f} "
                    f"L {row['Pred_Low']:.2f} "
                    f"C {row['Pred_Close']:.2f}"
                ),
                (
                    f"ACT  O {row['Actual_Open']:.2f} "
                    f"H {row['Actual_High']:.2f} "
                    f"L {row['Actual_Low']:.2f} "
                    f"C {row['Actual_Close']:.2f}"
                ),
                (
                    f"DIFF O {row['Diff_Open']:+.2f} "
                    f"H {row['Diff_High']:+.2f} "
                    f"L {row['Diff_Low']:+.2f} "
                    f"C {row['Diff_Close']:+.2f}"
                ),
                (
                    f"Direction: "
                    f"{row['Pred_Direction']} → "
                    f"{row['Actual_Direction']} "
                    f"{'✅' if row['DirectionCorrect'] else '❌'}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "📊 MODEL ACCURACY",
            f"Samples: {metrics['Samples']}",
            f"Open MAPE: {metrics['OpenMAPE']:.3f}%",
            f"High MAPE: {metrics['HighMAPE']:.3f}%",
            f"Low MAPE: {metrics['LowMAPE']:.3f}%",
            f"Close MAPE: {metrics['CloseMAPE']:.3f}%",
            f"Overall MAPE: {metrics['OverallMAPE']:.3f}%",
            f"Direction Accuracy: {metrics['DirectionAccuracy']:.1f}%",
            "",
            "🏆 CHAMPION / CHALLENGER",
            f"Decision: {retraining.get('Decision', '-')}",
            f"Champion Error: {retraining.get('ChampionError', 0):.6f}",
            f"Challenger Error: {retraining.get('ChallengerError', 0):.6f}",
            f"Improvement: {retraining.get('Improvement', 0):+.2f}%",
            f"Model Replaced: {'YES' if retraining.get('Retrained') else 'NO'}",
        ]
    )

    return "\n".join(lines)
