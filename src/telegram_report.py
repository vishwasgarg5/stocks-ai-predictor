"""Telegram notifications with compact table-first reports."""
import os
import requests


def _send(text: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram credentials not configured; skipping Telegram notification.")
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=20)
        r.raise_for_status()
        if not r.json().get("ok"):
            print(f"Telegram API rejected message: {r.json()}")
            return False
        print("Telegram sent successfully")
        return True
    except Exception as exc:
        print(f"Telegram send failed: {exc}")
        return False


def send_morning_predictions(prediction_date: str, rows):
    lines = ["<b>📈 AI NSE STOCK PREDICTION — STAGE 1.5</b>", f"Prediction Date: <b>{prediction_date}</b>", "", "<pre>RK STOCK        SCORE   OPEN     HIGH      LOW    CLOSE  DIR CONF  REGIME\n--------------------------------------------------------------------------"]
    for rank, row in enumerate(rows, 1):
        conf = float(row.get("confidence", 0.5)) * 100
        lines.append(f"{rank:>2} {row['symbol']:<11} {row['score']:>6.1f} {row['open']:>8.2f} {row['high']:>8.2f} {row['low']:>8.2f} {row['close']:>8.2f} {row.get('direction','NA')[:4]:>4} {conf:>5.1f}% {row.get('regime','NA')[:8]:>8}")
    lines += ["</pre>", "Model: Stage 1.5 | Data cutoff: prior completed session"]
    return _send("\n".join(lines))


def _err_pct(pred, actual):
    if actual is None or pred is None or actual == 0:
        return None
    return (actual - pred) / actual * 100.0


def send_evening(session_date, evaluated, ledger, report, retrained=False, champion_mae=None, challenger_mae=None):
    lines = ["<b>🌙 AI NSE EVENING REPORT — STAGE 1.5</b>", f"Market Date: <b>{session_date}</b>", f"Predictions Evaluated: <b>{evaluated}</b>", ""]
    if ledger is not None and evaluated:
        done = ledger[(ledger["actual_close"].notna()) & (ledger["target_date"].astype(str) == session_date)].sort_values(["rank", "symbol"]).head(5)
        lines.append("<pre>STOCK       TYPE       OPEN      HIGH       LOW     CLOSE")
        for _, r in done.iterrows():
            lines.append(f"{r['symbol']:<10} PRED   {r['pred_open']:>9.2f} {r['pred_high']:>9.2f} {r['pred_low']:>9.2f} {r['pred_close']:>9.2f}")
            lines.append(f"{'':<10} ACT    {r['actual_open']:>9.2f} {r['actual_high']:>9.2f} {r['actual_low']:>9.2f} {r['actual_close']:>9.2f}")
            lines.append(f"{'':<10} DIFF   {r['actual_open']-r['pred_open']:>+9.2f} {r['actual_high']-r['pred_high']:>+9.2f} {r['actual_low']-r['pred_low']:>+9.2f} {r['actual_close']-r['pred_close']:>+9.2f}")
        lines.append("</pre>")
    if report is not None and not report.empty:
        r = report.iloc[0]
        lines.append("<pre>MODEL ACCURACY\n-----------------------------")
        lines.append(f"Samples             {int(r['predictions'])}")
        lines.append(f"Open MAPE           {r['open_mape']*100:.3f}%")
        lines.append(f"High MAPE           {r['high_mape']*100:.3f}%")
        lines.append(f"Low MAPE            {r['low_mape']*100:.3f}%")
        lines.append(f"Close MAPE          {r['close_mape']*100:.3f}%")
        lines.append(f"Overall OHLC MAPE   {r['ohlc_mape']*100:.3f}%")
        lines.append(f"Direction Accuracy  {r['direction_accuracy']*100:.1f}%")
        lines.append("</pre>")
    if evaluated:
        if champion_mae is None:
            lines.append("<b>Champion: Stage 1.5 bootstrap</b>")
        else:
            lines.append(f"<b>Champion Decision: {'REPLACED' if retrained else 'KEPT'}</b>")
            if challenger_mae is not None:
                lines.append(f"Validation MAE — Champion: {champion_mae:.6f} | Challenger: {challenger_mae:.6f}")
    else:
        lines.append("<b>No new target-session actuals — no retraining</b>")
    return _send("\n".join(lines))
