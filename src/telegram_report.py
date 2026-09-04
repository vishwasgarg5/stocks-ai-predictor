"""Professional mobile-first Telegram reports for Stage 10.1."""
import os
import requests
import pandas as pd
from .config import TELEGRAM_MAX_LENGTH, MODEL_VERSION, PRICE_BUCKET_NAMES
from .utils import format_money, split_messages


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
            r = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=20)
            if r.status_code != 200:
                print("Telegram error:", r.text)
                success = False
        except Exception as exc:
            print("Telegram exception:", exc)
            success = False
    return success


def _fmt(v, digits=2):
    try:
        return f"{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _pct(v):
    try:
        return f"{float(v):+.1f}%"
    except (TypeError, ValueError):
        return "-"


def _accuracy(v):
    try:
        return f"{float(v):.1f}%"
    except (TypeError, ValueError):
        return "-"


def _decision(r, expected, rr, warning=False):
    action = str(r.get("Action", r.get("Direction", "WATCH"))).upper()
    confidence = float(r.get("Confidence", 0) or 0)
    if warning and action in {"BUY", "STRONG BUY"}:
        return "WATCH"
    if expected <= 0 or confidence < 50:
        return "WATCH" if action not in {"SELL", "AVOID"} else action
    try:
        if rr != "-" and float(rr.rstrip("x")) >= 2 and confidence >= 65:
            return "BUY"
    except ValueError:
        pass
    return "HOLD" if action in {"BUY", "HOLD"} else "WATCH"


def _stock_card(r, warning=False):
    cmp = float(r.get("Current_Price", r.get("Current_Close", 0)) or 0)
    pred = float(r.get("Pred_Close", 0) or 0)
    expected = ((pred / cmp) - 1) * 100 if cmp else 0
    sl = r.get("StopLoss", r.get("SL"))
    sl_text = format_money(sl) if sl is not None and not pd.isna(sl) else "-"
    rr = "-"
    try:
        reward, risk = pred - cmp, cmp - float(sl)
        if reward > 0 and risk > 0:
            rr = f"{reward / risk:.1f}x"
    except (TypeError, ValueError):
        pass
    decision = _decision(r, expected, rr, warning)
    return [
        f"*{r.get('Symbol', '-')}*   |   *{decision}*",
        f"₹{_fmt(cmp)}  →  *₹{_fmt(pred)}*   {_pct(expected)}",
        f"O {_fmt(r.get('Current_Open'))}  H {_fmt(r.get('Current_High'))}  L {_fmt(r.get('Current_Low'))}  C {_fmt(r.get('Current_Close', cmp))}",
        f"Vol {_fmt(r.get('Current_Volume'), 0)}   •   SL {sl_text}   •   R/R {rr}",
        f"Confidence  *{float(r.get('Confidence', 0) or 0):.0f}%*",
    ]


def _bucket_sections(selected, warning=False):
    if selected is None or selected.empty:
        return ["No qualifying stock today."]
    lines = []
    for bucket in list(PRICE_BUCKET_NAMES) + [b for b in selected.get("PriceBucket", pd.Series(dtype=str)).astype(str).unique() if b not in PRICE_BUCKET_NAMES]:
        group = selected[selected["PriceBucket"].astype(str) == bucket].copy() if "PriceBucket" in selected.columns else pd.DataFrame()
        if group.empty:
            continue
        sort_cols = [c for c in ["TradeConfidence", "Score", "Confidence"] if c in group.columns]
        if sort_cols:
            group = group.sort_values(sort_cols, ascending=False)
        # Show the human-readable bucket label, never the internal B1/B2/B3 code.
        label = str(group.iloc[0].get("PriceBucketLabel", bucket))
        if label in {"nan", "None", "-"}: label = bucket
        lines += [f"\n💎 *{label}*", *_stock_card(group.iloc[0], warning)]
    return lines


def _scan(scan):
    return f"🔎 {int(scan.get('Universe', 0)):,} scanned  •  {int(scan.get('Data', 0)):,} data  •  {int(scan.get('Liquid', 0)):,} liquid  •  {int(scan.get('AI', 0)):,} AI  •  {int(scan.get('Selected', 0)):,} selected"


def _market(snapshot, regime):
    snapshot = snapshot or {}
    n, b, v = snapshot.get("NIFTY", {}), snapshot.get("BANKNIFTY", {}), snapshot.get("VIX", {})
    br = snapshot.get("Breadth", {})
    return [
        f"📊 NIFTY {_fmt(n.get('Close'))} ({_pct(n.get('Change1D'))})  •  BANK {_fmt(b.get('Close'))} ({_pct(b.get('Change1D'))})",
        f"VIX {_fmt(v.get('Close'))}  •  Breadth {int(br.get('Advancers', 0))}↑ / {int(br.get('Decliners', 0))}↓  •  {regime}",
    ]


def _portfolio(p):
    if not p:
        return []
    return [f"💼 *PORTFOLIO*  {p.get('Positions', 0)} positions  •  ₹{p.get('Value', 0):,.0f}  •  P&L ₹{p.get('PnL', 0):+,.0f} ({p.get('Return', 0):+.2f}%)"] + [f"• {x}" for x in p.get("Rows", [])]


def morning_report(prediction_date, cutoff_date, selected, jump_watchlist, intraday, **kwargs):
    accuracy = kwargs.get("accuracy", {})
    scan = kwargs.get("scan", {})
    portfolio = kwargs.get("portfolio", {})
    snapshot = kwargs.get("market_snapshot", {})
    regime = kwargs.get("regime", "-")
    ipo = kwargs.get("ipo")
    warning = str(accuracy.get("Health", "")).upper() in {"WARNING", "DEGRADED", "POOR"} or str(accuracy.get("Drift", "")).upper() in {"WARNING", "HIGH", "DEGRADED"}
    lines = [
        "📈 *AI NSE MORNING REPORT*",
        f"📅 *Prediction: {prediction_date}*",
        f"⚙️ {MODEL_VERSION}",
        _scan(scan),
        f"🤖 Accuracy {_accuracy(accuracy.get('PreviousAccuracy'))} → {_accuracy(accuracy.get('CurrentAccuracy'))}  •  {int(accuracy.get('AccuracySamples', accuracy.get('Samples', 0)) or 0)} validated",
        "",
        "🎯 *TOP STOCKS*  |  BEST PICK BY PRICE BUCKET",
    ]
    lines += _market(snapshot, regime)
    lines += _bucket_sections(selected, warning)
    if portfolio:
        lines += ["\n" + x for x in _portfolio(portfolio)]
    lines.append("\n🔥 *JUMP WATCH*  |  TOP 5")
    if jump_watchlist is not None and not jump_watchlist.empty:
        for i, (_, r) in enumerate(jump_watchlist.head(5).iterrows(), 1):
            cp, target = float(r.get("Current_Price", 0) or 0), float(r.get("Target_Level", 0) or 0)
            lines.append(f"{i}. *{r.get('Symbol', '-')}*  ₹{cp:,.2f} → ₹{target:,.2f}  {_pct((target / cp - 1) * 100 if cp else 0)}  •  Prob {float(r.get('Jump_Probability', 0) or 0):.0f}%")
    else:
        lines.append("No strong jump setup.")
    lines.append("\n⚡ *INTRADAY*  |  TOP 5")
    if intraday is not None and not intraday.empty:
        for i, (_, r) in enumerate(intraday.head(5).iterrows(), 1):
            lines.append(f"{i}. *{r.get('Symbol', '-')}*  {r.get('Bias', '-')}  ₹{_fmt(r.get('Current'))} → ₹{_fmt(r.get('Target'))}  •  SL ₹{_fmt(r.get('StopLoss'))}  •  {float(r.get('Confidence', 0) or 0):.0f}%")
    else:
        lines.append("No confirmed intraday setup.")
    return "\n".join(lines)


def evening_report(market_date, evaluation, metrics, retraining, **kwargs):
    accuracy = kwargs.get("accuracy", {})
    scan = kwargs.get("scan", {})
    bucket = kwargs.get("bucket_metrics", {})
    portfolio = kwargs.get("portfolio", {})
    learning = kwargs.get("learning", {})
    lines = [
        "🌙 *AI NSE EVENING REPORT*",
        f"📅 *Market: {market_date}*",
        f"⚙️ {MODEL_VERSION}",
        _scan(scan),
        f"🤖 Accuracy {_accuracy(accuracy.get('PreviousAccuracy'))} → {_accuracy(accuracy.get('CurrentAccuracy'))}",
        "",
        "🎯 *PREDICTION vs ACTUAL*",
    ]
    if evaluation is not None and not evaluation.empty:
        for _, r in evaluation.iterrows():
            diff = float(r.get("Diff_Close", 0) or 0)
            actual = float(r.get("Actual_Close", 1) or 1)
            ok = "✅" if bool(r.get("DirectionCorrect", False)) else "❌"
            lines += [
                f"\n*{r.get('Symbol', '-')}*  {ok}",
                f"PRED  O {_fmt(r.get('Pred_Open'))}  H {_fmt(r.get('Pred_High'))}  L {_fmt(r.get('Pred_Low'))}  C {_fmt(r.get('Pred_Close'))}",
                f"ACT   O {_fmt(r.get('Actual_Open'))}  H {_fmt(r.get('Actual_High'))}  L {_fmt(r.get('Actual_Low'))}  C {_fmt(r.get('Actual_Close'))}",
                f"DIFF  O {float(r.get('Diff_Open', 0) or 0):+.2f}  H {float(r.get('Diff_High', 0) or 0):+.2f}  L {float(r.get('Diff_Low', 0) or 0):+.2f}  C {diff:+.2f} ({diff / actual * 100:+.2f}%)",
                f"Direction  {r.get('Pred_Direction', '-')} → {r.get('Actual_Direction', '-')}",
            ]
    else:
        lines.append("No predictions available for evaluation.")
    lines += ["\n📊 *RESULTS*", "Buckets  " + (" • ".join(f"{k} {v:.1f}%" for k, v in bucket.items()) if bucket else "No sufficient sample"), "\n🧠 *MODEL LEARNING*", f"Samples  {metrics.get('Samples', 0)}", f"Overall MAPE  {metrics.get('OverallMAPE', 0):.3f}%", f"Close MAPE  {metrics.get('CloseMAPE', 0):.3f}%", f"Direction Accuracy  {metrics.get('DirectionAccuracy', 0):.1f}%", f"Champion/Challenger  {retraining.get('Decision', '-')}", f"Model Replaced  {'YES' if retraining.get('Retrained') else 'NO'}", f"Improvement  {retraining.get('Improvement', 0):+.2f}%", f"Learning State  {learning.get('status', 'UPDATED')}" ]
    if portfolio:
        lines += ["\n" + x for x in _portfolio(portfolio)]
    return "\n".join(lines)
