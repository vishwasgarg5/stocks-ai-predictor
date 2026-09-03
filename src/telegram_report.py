"""Mobile-first Telegram reports for the final NSE AI pipeline."""
import os
import requests
import pandas as pd
from .config import TELEGRAM_MAX_LENGTH, MODEL_VERSION
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
            r = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=20)
            if r.status_code != 200:
                print("Telegram error:", r.text)
                success = False
        except Exception as exc:
            print("Telegram exception:", exc)
            success = False
    return success


def _table(headers, rows):
    if not rows:
        return ""
    all_rows = [headers] + [[str(x) for x in row] for row in rows]
    widths = [max(len(row[i]) for row in all_rows) for i in range(len(headers))]
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) for row in all_rows]
    return "\n".join([body[0], sep] + body[1:])


def _transpose_table(headers, rows):
    if not rows:
        return ""
    return _table([headers[0]] + [str(r[0]) for r in rows], [[headers[i]] + [str(r[i]) for r in rows] for i in range(1, len(headers))])


def _fmt_accuracy(v):
    return "-" if v is None or pd.isna(v) else f"{float(v):.1f}%"


def _accuracy_lines(a, compact=False):
    main = f"🤖 Accuracy: {_fmt_accuracy(a.get('PreviousAccuracy'))} → {_fmt_accuracy(a.get('CurrentAccuracy'))} | {int(a.get('AccuracySamples', a.get('Samples', 0)) or 0)} validated"
    if compact:
        return [main]
    return [main, f"📏 Horizons: 1D {_fmt_accuracy(a.get('1D'))} | 3D {_fmt_accuracy(a.get('3D'))} | 5D {_fmt_accuracy(a.get('5D'))} | 7D {_fmt_accuracy(a.get('7D'))} | 20D {_fmt_accuracy(a.get('20D'))} | Direction {_fmt_accuracy(a.get('DirectionAccuracy'))}", f"🧠 Health: {a.get('Health', '-')} | Drift {a.get('Drift', '-')} | 7D {_fmt_accuracy(a.get('Trend7D'))} | 30D {_fmt_accuracy(a.get('Trend30D'))}"]


def _market_lines(snapshot, regime):
    n = snapshot.get("NIFTY", {})
    b = snapshot.get("BANKNIFTY", {})
    v = snapshot.get("VIX", {})
    br = snapshot.get("Breadth", {})
    def q(x):
        return "-" if x is None or pd.isna(x) else f"{float(x):.2f}"
    return [f"NIFTY {q(n.get('Close'))} ({q(n.get('Change1D'))}%) | BANK NIFTY {q(b.get('Close'))} ({q(b.get('Change1D'))}%)", f"VIX {q(v.get('Close'))} | Breadth {int(br.get('Advancers', 0))}↑/{int(br.get('Decliners', 0))}↓ | Regime {regime}"]


def _sector_lines(selected):
    if selected is None or selected.empty or "Sector" not in selected.columns:
        return []
    x = selected.copy()
    x["_score"] = pd.to_numeric(x.get("SectorStrength", 50), errors="coerce").fillna(50)
    g = x.groupby("Sector")["_score"].mean().sort_values(ascending=False).head(5)
    return ["🏭 Sector AI: " + " | ".join(f"{k} {v:.0f}" for k, v in g.items())]


def _decision(r, expected, rr, model_warning=False):
    action = str(r.get("Action", r.get("Direction", "WATCH"))).upper()
    confidence = float(r.get("Confidence", 0) or 0)
    if model_warning and action in {"BUY", "STRONG BUY"}:
        return "WATCH"
    if expected <= 0 or confidence < 50:
        return "WATCH" if action not in {"SELL", "AVOID"} else action
    if rr != "-" and float(rr[:-1]) >= 2 and confidence >= 65:
        return "BUY"
    return "HOLD" if action in {"BUY", "HOLD"} else "WATCH"


def _stock_rows(selected, model_warning=False):
    rows = []
    for i, (_, r) in enumerate(selected.head(5).iterrows(), 1):
        cmp = float(r.get("Current_Price", r.get("Current_Close", 0)) or 0)
        pred = float(r.get("Pred_Close", 0) or 0)
        expected = ((pred / cmp) - 1) * 100 if cmp else 0
        sl = r.get("StopLoss", r.get("SL"))
        sl_text = format_money(sl) if sl is not None and not pd.isna(sl) else "-"
        rr = "-"
        try:
            sl_value = float(sl)
            reward = pred - cmp
            risk = cmp - sl_value
            if reward > 0 and risk > 0:
                rr = f"{reward / risk:.1f}x"
        except (TypeError, ValueError):
            pass
        rows.append([i, r.get("Symbol", "-"), r.get("PriceBucket", "-"), format_money(cmp), format_money(r.get("Current_Open", 0)), format_money(r.get("Current_High", 0)), format_money(r.get("Current_Low", 0)), format_money(r.get("Current_Close", cmp)), f"{float(r.get('Current_Volume', 0) or 0):,.0f}", f"{pred:.2f}", f"{expected:+.1f}%", sl_text, rr, f"{float(r.get('Confidence', 0) or 0):.0f}%", _decision(r, expected, rr, model_warning)])
    return rows


def _scan_lines(scan):
    return [f"🔎 Scan: {int(scan.get('Universe', 0)):,} → {int(scan.get('Data', 0)):,} data → {int(scan.get('Liquid', 0)):,} liquid → {int(scan.get('AI', 0)):,} AI → {int(scan.get('Selected', 0)):,} selected"]


def _portfolio_lines(p):
    if not p:
        return ["💼 Portfolio Manager: no portfolio CSV / no verified prices."]
    lines = [f"💼 *PORTFOLIO MANAGER* | Positions {p.get('Positions', 0)} | Value ₹{p.get('Value', 0):,.0f} | P&L ₹{p.get('PnL', 0):+,.0f} ({p.get('Return', 0):+.2f}%)"]
    lines += [f"• {x}" for x in p.get("Rows", [])]
    return lines


def morning_report(prediction_date, cutoff_date, selected, jump_watchlist, intraday, **kwargs):
    snapshot = kwargs.get("market_snapshot", {})
    regime = kwargs.get("regime", "-")
    ipo = kwargs.get("ipo")
    accuracy = kwargs.get("accuracy", {})
    scan = kwargs.get("scan", {})
    portfolio = kwargs.get("portfolio", {})
    warning = str(accuracy.get("Health", "")).upper() in {"WARNING", "DEGRADED", "POOR"} or str(accuracy.get("Drift", "")).upper() in {"WARNING", "HIGH", "DEGRADED"}
    lines = ["📈 *AI NSE MORNING REPORT*", f"_{prediction_date} | Data: {cutoff_date} | {MODEL_VERSION}_", *(_scan_lines(scan)), *(_accuracy_lines(accuracy, compact=True)), "", "🎯 *1. TOP STOCKS*"]
    if warning:
        lines.append("⚠️ *MODEL WARNING* — confidence reduced because model health/drift is weak.")
    lines += _market_lines(snapshot, regime) + _sector_lines(selected) + ["", "*Price Bucket + Current OHLCV + AI Forecast*"]
    if selected is not None and not selected.empty:
        lines += ["```", _transpose_table(["#", "Stock", "Price Bucket", "CMP", "Open", "High", "Low", "Close", "Volume", "AI Target", "Exp%", "SL", "R/R", "Conf", "AI Decision"], _stock_rows(selected, warning)), "```"]
    else:
        lines.append("No stock passed the quality gate today.")
    if ipo is not None and not ipo.empty:
        lines.append("*IPO / NEW LISTINGS*")
        for _, r in ipo.head(5).iterrows():
            lines.append(f"• {r.get('IPOName', '-')} | ₹{float(r.get('PriceHigh', 0)):.0f} | GMP ₹{float(r.get('GMPValue', 0)):.0f} ({float(r.get('GMPPct', 0)):.1f}%) | Score {float(r.get('IPOScore', 0)):.0f} | {r.get('IPOAction', 'WATCH')}")
        lines.append("GMP is unofficial; not a guaranteed listing price.")
    else:
        lines.append("*IPO / NEW LISTINGS:* No verified live records available.")
    lines += _portfolio_lines(portfolio) + ["", "🔥 *2. +5% JUMP WATCH*"]
    if jump_watchlist is not None and not jump_watchlist.empty:
        rows = []
        for i, (_, r) in enumerate(jump_watchlist.head(5).iterrows(), 1):
            cp = float(r["Current_Price"])
            target = float(r["Target_Level"])
            prob = float(r.get("Jump_Probability", 0) or 0)
            rows.append([i, r["Symbol"], format_money(cp), format_money(target), format_percent((target / cp - 1) * 100 if cp else 0), format_percent(float(r.get("Estimated_7D_Upside", 0))), f"{prob:.0f}%"])
        lines += ["```", _transpose_table(["#", "Stock", "CMP", "+5% Target", "Target%", "7D Exp%", "Prob"], rows), "```"]
    else:
        lines.append("No strong +5% candidate passed the gate.")
    lines += ["", "⚡ *3. INTRADAY STOCKS*"]
    if intraday is not None and not intraday.empty:
        rows = [[i, r.get("Symbol", "-"), r.get("Bias", "-"), format_money(r.get("Current", 0)), format_money(r.get("Target", 0)), format_money(r.get("StopLoss", 0)), f"{float(r.get('Confidence', 0) or 0):.0f}%"] for i, (_, r) in enumerate(intraday.head(5).iterrows(), 1)]
        lines += ["```", _transpose_table(["#", "Stock", "Bias", "CMP", "Target", "SL", "Conf"], rows), "```"]
    else:
        lines.append("No confirmed intraday setup today; live conditions did not pass the gate.")
    return "\n".join(lines)


def evening_report(market_date, evaluation, metrics, retraining, **kwargs):
    bucket = kwargs.get("bucket_metrics", {})
    intraday = kwargs.get("intraday_metrics", {})
    ipo = kwargs.get("ipo_results")
    learning = kwargs.get("learning", {})
    accuracy = kwargs.get("accuracy", {})
    scan = kwargs.get("scan", {})
    portfolio = kwargs.get("portfolio", {})
    lines = ["🌙 *AI NSE EVENING REPORT*", f"_{market_date} | {MODEL_VERSION}_"] + _scan_lines(scan) + _accuracy_lines(accuracy) + ["", "🎯 *1. TOP 5 — PREDICTION vs ACTUAL*"]
    if evaluation is not None and not evaluation.empty:
        rows, dr = [], []
        for _, r in evaluation.head(5).iterrows():
            rows += [[r["Symbol"], "PRED", f"{r['Pred_Open']:.2f}", f"{r['Pred_High']:.2f}", f"{r['Pred_Low']:.2f}", f"{r['Pred_Close']:.2f}"], ["", "ACT", f"{r['Actual_Open']:.2f}", f"{r['Actual_High']:.2f}", f"{r['Actual_Low']:.2f}", f"{r['Actual_Close']:.2f}"], ["", "DIFF", f"{r['Diff_Open']:+.2f}", f"{r['Diff_High']:+.2f}", f"{r['Diff_Low']:+.2f}", f"{r['Diff_Close']:+.2f}"]]
            dr.append([r["Symbol"], f"{r['Pred_Direction']} → {r['Actual_Direction']}", "YES" if r["DirectionCorrect"] else "NO"])
        lines += ["```", _table(["Stock", "Type", "Open", "High", "Low", "Close"], rows), "```", "", "*Direction*", "```", _table(["Stock", "Pred → Actual", "Correct"], dr), "```"]
    else:
        lines.append("No predictions available for evaluation.")
    lines += ["", "🔥 *2. RESULTS"]
    lines.append("Price buckets: " + (" | ".join(f"{k} {v:.1f}%" for k, v in bucket.items()) if bucket else "not enough evaluated samples."))
    if intraday:
        lines.append("Intraday: " + " | ".join(f"{k} {v}" for k, v in intraday.items()))
    if ipo is not None and not ipo.empty:
        lines.append(f"IPO/new listings evaluated: {len(ipo)}")
    lines += _portfolio_lines(portfolio) + ["", "🧠 *3. MODEL LEARNING*"]
    lines += ["```", _table(["Metric", "Value"], [["Samples", metrics.get("Samples", 0)], ["Overall MAPE", f"{metrics.get('OverallMAPE', 0):.3f}%"], ["Close MAPE", f"{metrics.get('CloseMAPE', 0):.3f}%"], ["Direction Accuracy", f"{metrics.get('DirectionAccuracy', 0):.1f}%"], ["Previous Accuracy", _fmt_accuracy(accuracy.get("PreviousAccuracy"))], ["Current Accuracy", _fmt_accuracy(accuracy.get("CurrentAccuracy"))], ["Champion/Challenger", retraining.get("Decision", "-")], ["Model Replaced", "YES" if retraining.get("Retrained") else "NO"], ["Improvement", f"{retraining.get('Improvement', 0):+.2f}%"], ["Learning State", learning.get("status", "UPDATED")]]), "```"]
    return "\n".join(lines)
