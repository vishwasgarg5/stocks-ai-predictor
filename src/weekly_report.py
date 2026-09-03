"""Weekly Stage 10 performance report."""
import pandas as pd

from .config import DAILY_METRICS_FILE, MODEL_VERSION
from .telegram_report import send_telegram


def run():
    title=f"📊 AI NSE WEEKLY REPORT — {MODEL_VERSION}"
    if not DAILY_METRICS_FILE.exists():
        send_telegram(f"{title}\n\nNo evaluation data available yet.")
        return

    try:
        df=pd.read_csv(DAILY_METRICS_FILE)
    except Exception as exc:
        send_telegram(f"{title}\n\nUnable to read weekly metrics: {exc}")
        return
    if df.empty:
        send_telegram(f"{title}\n\nNo evaluation data available yet.")
        return

    required=["MarketDate","Samples","OpenMAPE","HighMAPE","LowMAPE","CloseMAPE","OverallMAPE","DirectionAccuracy"]
    missing=[c for c in required if c not in df.columns]
    if missing:
        send_telegram(f"{title}\n\nMissing metrics: {', '.join(missing)}")
        return

    df=df.copy(); df["MarketDate"]=pd.to_datetime(df["MarketDate"],errors="coerce"); df=df.dropna(subset=["MarketDate"])
    if df.empty:
        send_telegram(f"{title}\n\nNo valid evaluation dates available.")
        return
    for col in required[1:]: df[col]=pd.to_numeric(df[col],errors="coerce")
    recent=df.sort_values("MarketDate").tail(5)
    if recent.empty:
        send_telegram(f"{title}\n\nNo valid recent metrics available.")
        return
    recent=recent.dropna(subset=["OpenMAPE","HighMAPE","LowMAPE","CloseMAPE","OverallMAPE","DirectionAccuracy"])
    if recent.empty:
        send_telegram(f"{title}\n\nRecent metrics are incomplete; no reliable weekly summary generated.")
        return

    lines=[title,"",f"Evaluation Days: {len(recent)}",f"Latest Day: {recent['MarketDate'].max().date()}",f"Samples: {int(recent['Samples'].fillna(0).sum())}","","Recent Performance",f"Open MAPE: {recent['OpenMAPE'].mean():.3f}%",f"High MAPE: {recent['HighMAPE'].mean():.3f}%",f"Low MAPE: {recent['LowMAPE'].mean():.3f}%",f"Close MAPE: {recent['CloseMAPE'].mean():.3f}%",f"Overall MAPE: {recent['OverallMAPE'].mean():.3f}%",f"Direction Accuracy: {recent['DirectionAccuracy'].mean():.1f}%","", "Results are stored in GitHub state and used for future stock/model reliability."]
    send_telegram("\n".join(lines))


if __name__=="__main__":run()
