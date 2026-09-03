"""Weekly Stage 10 performance report."""
import pandas as pd

from .config import DAILY_METRICS_FILE, MODEL_VERSION
from .telegram_report import send_telegram


def run():
    if not DAILY_METRICS_FILE.exists():
        send_telegram(
            f"📊 AI NSE WEEKLY REPORT — {MODEL_VERSION}\n\n"
            "No evaluation data available yet."
        )
        return

    df = pd.read_csv(DAILY_METRICS_FILE)
    if df.empty:
        return

    df = df.sort_values("MarketDate")
    recent = df.tail(5)
    required = ["Samples", "OpenMAPE", "HighMAPE", "LowMAPE", "CloseMAPE", "OverallMAPE", "DirectionAccuracy"]
    missing = [c for c in required if c not in recent.columns]
    if missing:
        send_telegram(f"📊 AI NSE WEEKLY REPORT — {MODEL_VERSION}\n\nMissing metrics: {', '.join(missing)}")
        return

    lines = [
        f"📊 AI NSE WEEKLY REPORT — {MODEL_VERSION}",
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
        "Results are stored in GitHub state and used for future stock/model reliability.",
    ]
    send_telegram("\n".join(lines))


if __name__ == "__main__":
    run()
