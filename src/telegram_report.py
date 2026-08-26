"""Telegram notifications. Credentials are read only from environment variables."""
import os
import requests


def _send(text: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram credentials not configured; skipping Telegram notification.")
        return False
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=20)
    r.raise_for_status()
    return True


def send_morning_predictions(prediction_date: str, rows):
    lines = ["<b>📈 AI NSE STOCK PREDICTION</b>", f"Prediction Date: <b>{prediction_date}</b>", "", "<pre>Rank Stock        Score   Open       High       Low        Close\n--------------------------------------------------------------"]
    for rank, row in enumerate(rows, 1):
        lines.append(f"{rank:<4} {row['symbol']:<11} {row['score']:>6.2f}  {row['open']:>9.2f} {row['high']:>9.2f} {row['low']:>9.2f} {row['close']:>9.2f}")
    lines += ["</pre>", "Data: prior completed trading session only"]
    _send("\n".join(lines))


def send_evening(session_date, evaluated, ledger, report, retrained=False, champion_mae=None, challenger_mae=None):
    lines = ["<b>🌙 AI NSE EVENING REPORT</b>", f"Market Date: <b>{session_date}</b>", f"Predictions Evaluated: <b>{evaluated}</b>", ""]
    if ledger is not None and evaluated:
        done = ledger[(ledger["actual_close"].notna()) & (ledger["target_date"].astype(str) == session_date)].tail(5)
        for _, r in done.iterrows():
            lines += [f"<b>{r['symbol']}</b>", "<pre>Type       Open       High       Low        Close\n" +
                "Predicted  " + f"{r['pred_open']:>9.2f} {r['pred_high']:>9.2f} {r['pred_low']:>9.2f} {r['pred_close']:>9.2f}\n" +
                "Actual     " + f"{r['actual_open']:>9.2f} {r['actual_high']:>9.2f} {r['actual_low']:>9.2f} {r['actual_close']:>9.2f}\n" +
                "Difference " + f"{r['actual_open']-r['pred_open']:>+9.2f} {r['actual_high']-r['pred_high']:>+9.2f} {r['actual_low']-r['pred_low']:>+9.2f} {r['actual_close']-r['pred_close']:>+9.2f}</pre>"]
    if report is not None and not report.empty:
        r = report.iloc[0]
        lines.append("<pre>ACCURACY\n--------\n" + f"Predictions         {int(r['predictions'])}\nOpen MAE            {r['open_mae']:.2f}\nHigh MAE            {r['high_mae']:.2f}\nLow MAE             {r['low_mae']:.2f}\nClose MAE           {r['close_mae']:.2f}\nOpen MAPE           {r['open_mape']*100:.3f}%\nHigh MAPE           {r['high_mape']*100:.3f}%\nLow MAPE            {r['low_mape']*100:.3f}%\nClose MAPE          {r['close_mape']*100:.3f}%\nOverall OHLC MAPE   {r['ohlc_mape']*100:.3f}%\nDirection Accuracy  {r['direction_accuracy']*100:.1f}%</pre>")
    if evaluated:
        if champion_mae is None:
            lines.append("<b>Model decision: INITIAL CHAMPION CREATED</b>")
        else:
            lines.append("<b>Model retrained: YES — challenger was better</b>" if retrained else "<b>Model retrained: NO — old champion was better</b>")
            lines.append(f"Validation MAE: Champion <b>{champion_mae:.6f}</b> | Challenger <b>{challenger_mae:.6f}</b>")
    else:
        lines.append("<b>Model retrained: NO — no new target-session actuals available</b>")
    _send("\n".join(lines))
