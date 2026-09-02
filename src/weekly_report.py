import pandas as pd

from .config import DAILY_METRICS_FILE
from .telegram_report import send_telegram


def run():
    if not DAILY_METRICS_FILE.exists():
        send_telegram(
            "📊 STAGE 2 WEEKLY REPORT\n\n"
            "No evaluation data available yet."
        )
        return

    df = pd.read_csv(
        DAILY_METRICS_FILE
    )

    if df.empty:
        return

    df = df.sort_values(
        "MarketDate"
    )

    recent = df.tail(5)

    lines = [
        "📊 AI NSE WEEKLY REPORT — STAGE 2",
        "",
        f"Evaluation Days: {len(recent)}",
        f"Samples: {int(recent['Samples'].sum())}",
        "",
        "Recent Performance",
        f"Open MAPE: {recent['OpenMAPE'].mean():.3f}%",
        f"High MAPE: {recent['HighMAPE'].mean():.3f}%",
        f"Low MAPE: {recent['LowMAPE'].mean():.3f}%",
        f"Close MAPE: {recent['CloseMAPE'].mean():.3f}%",
        f"Overall MAPE: {recent['OverallMAPE'].mean():.3f}%",
        f"Direction Accuracy: {recent['DirectionAccuracy'].mean():.1f}%",
        "",
        "The system stores these results and uses "
        "them for future stock/model reliability.",
    ]

    send_telegram(
        "\n".join(lines)
    )


if __name__ == "__main__":
    run()
