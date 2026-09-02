import os
import requests
import pandas as pd

from .config import TELEGRAM_MAX_LENGTH
from .utils import format_money, format_percent, split_messages


def send_telegram(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram secrets not configured.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    success = True
    for message in split_messages(text, TELEGRAM_MAX_LENGTH):
        try:
            response = requests.post(
                url,
                json={"chat_id": chat_id, "text": message},
                timeout=20,
            )
            if response.status_code != 200:
                print("Telegram error:", response.text)
                success = False
        except Exception as exc:
            print("Telegram exception:", exc)
            success = False
    return success


def _table(headers, rows):
    """Create a Telegram-safe monospace table without requiring Markdown parsing."""
    all_rows = [headers] + [[str(x) for x in row] for row in rows]
    widths = [max(len(row[i]) for row in all_rows) for i in range(len(headers))]
    separator = "  ".join("-" * width for width in widths)
    body = ["  ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) for row in all_rows]
    return "\n".join([body[0], separator] + body[1:])


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
    ]

    if selected is not None and not selected.empty:
        rows = []
        for i, (_, row) in enumerate(selected.iterrows(), 1):
            rows.append([
                i,
                row["Symbol"],
                f"{row['Score']:.1f}",
                row["Direction"],
                f"{row['Direction_Confidence']:.0f}%",
                f"{row['Pred_Open']:.2f}",
                f"{row['Pred_High']:.2f}",
                f"{row['Pred_Low']:.2f}",
                f"{row['Pred_Close']:.2f}",
            ])
        lines += [
            "```",
            _table(["#", "Stock", "Score", "Dir", "Conf", "Open", "High", "Low", "Close"], rows),
            "```",
        ]
    else:
        lines.append("No next-day predictions generated.")

    lines += ["", "🔥 7-DAY +5% JUMP WATCHLIST"]
    if jump_watchlist is not None and not jump_watchlist.empty:
        rows = []
        for i, (_, row) in enumerate(jump_watchlist.iterrows(), 1):
            rows.append([
                i,
                row["Symbol"],
                format_money(row["Current_Price"]),
                format_money(row["Target_Level"]),
                format_percent(row["Estimated_7D_Upside"]),
                f"{row['Jump_Probability']:.0f}%",
            ])
        lines += [
            "```",
            _table(["#", "Stock", "CMP", "Target", "Upside", "Prob"], rows),
            "```",
        ]
    else:
        lines.append("No strong +5% candidates today.")

    lines += ["", "⚡ INTRADAY TOP 5"]
    if intraday is not None and not intraday.empty:
        rows = []
        for i, (_, row) in enumerate(intraday.iterrows(), 1):
            rows.append([
                i,
                row["Symbol"],
                row["Bias"],
                format_money(row["Current"]),
                format_money(row["Target"]),
                format_money(row["StopLoss"]),
                f"{row['Confidence']:.0f}%",
            ])
        lines += [
            "```",
            _table(["#", "Stock", "Bias", "CMP", "Target", "SL", "Conf"], rows),
            "```",
        ]
    else:
        lines.append("No strong intraday setup today.")

    lines += ["", "Stage 2 stores all prediction data for future learning."]
    return "\n".join(lines)


def evening_report(market_date, evaluation, metrics, retraining):
    lines = [
        "🌙 AI NSE EVENING REPORT — STAGE 2",
        f"Market Date: {market_date}",
        "",
        "📊 PREDICTED vs ACTUAL OHLC",
    ]

    if evaluation is not None and not evaluation.empty:
        rows = []
        for _, row in evaluation.iterrows():
            rows.extend([
                [row["Symbol"], "PRED", f"{row['Pred_Open']:.2f}", f"{row['Pred_High']:.2f}", f"{row['Pred_Low']:.2f}", f"{row['Pred_Close']:.2f}"],
                ["", "ACT", f"{row['Actual_Open']:.2f}", f"{row['Actual_High']:.2f}", f"{row['Actual_Low']:.2f}", f"{row['Actual_Close']:.2f}"],
                ["", "DIFF", f"{row['Diff_Open']:+.2f}", f"{row['Diff_High']:+.2f}", f"{row['Diff_Low']:+.2f}", f"{row['Diff_Close']:+.2f}"],
            ])
        lines += [
            "```",
            _table(["Stock", "Type", "Open", "High", "Low", "Close"], rows),
            "```",
            "",
            "Direction: ✅ = correct, ❌ = incorrect",
        ]
        for _, row in evaluation.iterrows():
            lines.append(f"{row['Symbol']}: {row['Pred_Direction']} → {row['Actual_Direction']} {'✅' if row['DirectionCorrect'] else '❌'}")
    else:
        lines.append("No predictions available for evaluation.")

    lines += [
        "",
        "📈 MODEL ACCURACY",
        "```",
        _table(
            ["Metric", "Value"],
            [
                ["Samples", metrics.get("Samples", 0)],
                ["Open MAPE", f"{metrics.get('OpenMAPE', 0):.3f}%"],
                ["High MAPE", f"{metrics.get('HighMAPE', 0):.3f}%"],
                ["Low MAPE", f"{metrics.get('LowMAPE', 0):.3f}%"],
                ["Close MAPE", f"{metrics.get('CloseMAPE', 0):.3f}%"],
                ["Overall MAPE", f"{metrics.get('OverallMAPE', 0):.3f}%"],
                ["Direction Accuracy", f"{metrics.get('DirectionAccuracy', 0):.1f}%"],
            ],
        ),
        "```",
        "",
        "🏆 CHAMPION / CHALLENGER",
        "```",
        _table(
            ["Metric", "Value"],
            [
                ["Decision", retraining.get("Decision", "-")],
                ["Champion Error", f"{retraining.get('ChampionError', 0):.6f}"],
                ["Challenger Error", f"{retraining.get('ChallengerError', 0):.6f}"],
                ["Improvement", f"{retraining.get('Improvement', 0):+.2f}%"],
                ["Model Replaced", "YES" if retraining.get("Retrained") else "NO"],
            ],
        ),
        "```",
    ]
    return "\n".join(lines)
