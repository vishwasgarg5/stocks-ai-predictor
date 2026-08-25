"""Telegram notifications. Credentials are read only from environment variables."""
import os
import requests


def _send(text: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram credentials not configured; skipping Telegram notification.")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=20)
    r.raise_for_status()
    return True


def send_morning_predictions(prediction_date: str, rows):
    lines = [
        "<b>📈 AI NSE STOCK PREDICTION</b>",
        f"Prediction Date: <b>{prediction_date}</b>",
        "",
        "<pre>Rank Stock        Score   Open       High       Low        Close\n"
        "--------------------------------------------------------------"
    ]
    for rank, row in enumerate(rows, 1):
        lines.append(f"{rank:<4} {row['symbol']:<11} {row['score']:>6.2f}  {row['open']:>9.2f} {row['high']:>9.2f} {row['low']:>9.2f} {row['close']:>9.2f}")
    lines.append("</pre>")
    lines.append("Data: previous completed trading session")
    _send("\n".join(lines))


def send_evening(session_date: str, evaluated: int, ledger, report):
    lines = [
        "<b>🌙 AI NSE EVENING REPORT</b>",
        f"Market Date: <b>{session_date}</b>",
        f"Predictions Evaluated: <b>{evaluated}</b>",
        ""
    ]
    if ledger is not None and evaluated:
        import pandas as pd
        done = ledger[(ledger["actual_close"].notna()) & (ledger["target_date"].astype(str) == session_date)].tail(5)
        for _, r in done.iterrows():
            lines += [
                f"<b>{r['symbol']}</b>",
                "<pre>Type       Open       High       Low        Close\n"
                "Predicted  " + f"{r['pred_open']:>9.2f} {r['pred_high']:>9.2f} {r['pred_low']:>9.2f} {r['pred_close']:>9.2f}\n"
                "Actual     " + f"{r['actual_open']:>9.2f} {r['actual_high']:>9.2f} {r['actual_low']:>9.2f} {r['actual_close']:>9.2f}\n"
                "Difference " + f"{r['actual_open']-r['pred_open']:>+9.2f} {r['actual_high']-r['pred_high']:>+9.2f} {r['actual_low']-r['pred_low']:>+9.2f} {r['actual_close']-r['pred_close']:>+9.2f}</pre>"
            ]
    if report is not None and not report.empty:
        r = report.iloc[0]
        lines += [
            "<pre>OVERALL PERFORMANCE\n"
            "-------------------\n"
            f"Predictions         {int(r['predictions'])}\n"
            f"Open MAE            {r['open_mae']:.2f}\n"
            f"High MAE            {r['high_mae']:.2f}\n"
            f"Low MAE             {r['low_mae']:.2f}\n"
            f"Close MAE           {r['close_mae']:.2f}\n"
            f"Direction Accuracy  {r['direction_accuracy']*100:.1f}%</pre>"
        ]
    lines.append("Model retraining: completed when new actuals were available")
    _send("\n".join(lines))
