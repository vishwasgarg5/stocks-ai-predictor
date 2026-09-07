"""Compact Telegram reports optimized for mobile screens."""
import html
import os
import re
import requests
import pandas as pd
from .config import TELEGRAM_MAX_LENGTH, MODEL_VERSION, PRICE_BUCKET_NAMES
from .utils import split_messages

_SECTION = "\n§§TELEGRAM_SECTION§§\n"


def _markdown_to_telegram_html(text):
    """Convert the small Markdown subset used by reports into safe Telegram HTML."""
    chunks = text.split("```")
    rendered = []
    for i, chunk in enumerate(chunks):
        if i % 2 == 1:
            rendered.append(f"<pre>{html.escape(chunk.strip(chr(10)))}</pre>")
            continue
        safe = html.escape(chunk)
        # Report headings are generated as *text*. Dynamic values are escaped above.
        safe = re.sub(r"\*([^*\n]+)\*", r"<b>\1</b>", safe)
        rendered.append(safe)
    return "".join(rendered)


def _plain_text_fallback(text):
    return text.replace("```", "").replace("*", "").replace("_", "")


def _post_telegram(message, token, chat_id):
    """Send formatted HTML, then retry as plain text if Telegram rejects entities."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": chat_id, "text": _markdown_to_telegram_html(message), "parse_mode": "HTML"},
            timeout=20,
        )
        if r.status_code == 200:
            return True
        print("Telegram HTML error:", r.text)
    except Exception as exc:
        print("Telegram HTML exception:", exc)

    try:
        fallback = requests.post(
            url,
            json={"chat_id": chat_id, "text": _plain_text_fallback(message)},
            timeout=20,
        )
        if fallback.status_code == 200:
            print("Telegram: delivered using plain-text fallback.")
            return True
        print("Telegram plain-text fallback error:", fallback.text)
    except Exception as exc:
        print("Telegram plain-text fallback exception:", exc)
    return False


def _send_part(part, token, chat_id):
    ok = True
    for message in split_messages(part.strip(), TELEGRAM_MAX_LENGTH):
        if not _post_telegram(message, token, chat_id):
            ok = False
    return ok


def send_telegram(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram secrets not configured.")
        return False
    parts = [p for p in text.split(_SECTION) if p.strip()]
    # Try every section instead of stopping at the first failed section.
    results = [_send_part(p, token, chat_id) for p in parts]
    return bool(results) and all(results)


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


def _table(headers, rows):
    """Return a compact monospaced table. Short headers keep it readable on phones."""
    if not rows:
        return []
    clean = []
    for row in rows:
        clean.append([str(x).replace("|", "/").replace("\n", " ") for x in row])
    widths = [len(str(h)) for h in headers]
    for row in clean:
        for i, value in enumerate(row):
            if i < len(widths):
                widths[i] = min(max(widths[i], len(value)), 12)
    def fit(value, width):
        value = str(value)
        return value if len(value) <= width else value[: max(1, width - 1)] + "…"
    def line(row):
        return "  ".join(fit(v, widths[i]).ljust(widths[i]) for i, v in enumerate(row))
    return ["```", line(headers), "  ".join("-" * w for w in widths), *[line(r) for r in clean], "```"]


def _ordered_price_buckets(values):
    configured = list(PRICE_BUCKET_NAMES)
    present = [] if values is None else [str(x) for x in pd.Series(values).dropna().unique()]
    return configured + [x for x in present if x not in configured]


def _market(snapshot, regime):
    snapshot = snapshot or {}
    rows = []
    for name, key in [("NIFTY", "NIFTY"), ("BANK", "BANKNIFTY"), ("FINN", "FINNIFTY"), ("MIDCP", "MIDCPNIFTY")]:
        x = snapshot.get(key, {})
        rows.append([name, _fmt(x.get("Close")), _pct(x.get("Change1D"))])
    x = snapshot.get("VIX", {})
    rows.append(["VIX", _fmt(x.get("Close")), "-"])
    br = snapshot.get("Breadth", {})
    return _table(["Index", "Value", "1D%"], rows) + [
        f"Breadth: {int(br.get('Advancers', 0))}↑ / {int(br.get('Decliners', 0))}↓  |  Regime: {regime}"
    ]


def _scan(scan):
    return (
        f"Scanned {int(scan.get('Universe', 0)):,} | Data {int(scan.get('Data', 0)):,} | "
        f"Liquid {int(scan.get('Liquid', 0)):,} | AI {int(scan.get('AI', 0)):,} | "
        f"Qualified {int(scan.get('Selected', 0)):,}"
    )


def _score_col(g):
    for c in ["FinalDecisionScore", "TradeConfidence", "Score"]:
        if c in g.columns:
            return c
    return None


def _sort(g):
    c = _score_col(g)
    return g.sort_values(c, ascending=False, kind="mergesort") if c else g.sort_values("Symbol", kind="mergesort")


def _stock_rows(g):
    rows = []
    for _, r in g.iterrows():
        cmp = float(r.get("Current_Price", r.get("Current_Close", 0)) or 0)
        pred = float(r.get("Pred_Close", 0) or 0)
        exp = (pred / cmp - 1) * 100 if cmp else 0
        rows.append([
            str(r.get("Symbol", "-")),
            f"₹{_fmt(cmp)}",
            _pct(exp),
            _pct(r.get("Horizon_1D")),
            _pct(r.get("Horizon_5D")),
            _pct(r.get("Horizon_20D")),
            str(r.get("Action", "-")),
        ])
    return rows


def _bucket_sections(selected):
    if selected is None or selected.empty:
        return ["No qualifying stock today."]
    if "PriceBucket" not in selected.columns:
        return ["No price bucket data."]
    lines = ["🎯 *QUALIFIED STOCKS BY PRICE BUCKET*"]
    for bucket in _ordered_price_buckets(selected["PriceBucket"]):
        g = selected[selected["PriceBucket"].astype(str) == bucket].copy()
        if g.empty:
            continue
        g = _sort(g).head(6)
        lines += [
            _SECTION,
            f"💎 *₹ {bucket}*  |  MAX 6",
            *_table(["Stock", "CMP", "Exp", "1D", "5D", "20D", "Action"], _stock_rows(g)),
        ]
    return lines


def _best_pick_table(selected):
    if selected is None or selected.empty or "PriceBucket" not in selected.columns:
        return []
    rows = []
    for bucket in _ordered_price_buckets(selected["PriceBucket"]):
        g = selected[selected["PriceBucket"].astype(str) == bucket].copy()
        if g.empty:
            continue
        r = _sort(g).iloc[0]
        rows.append([
            f"₹{bucket}",
            str(r.get("Symbol", "-")),
            f"₹{_fmt(r.get('Current_Price', r.get('Current_Close')))}",
            _pct(r.get("Expected_Return")),
            f"{float(r.get('FinalDecisionScore', r.get('Score', 0))):.0f}",
            str(r.get("Action", "-")),
        ])
    return ["🏆 *BEST PICK — 1 PER PRICE BUCKET*"] + _table(["Bucket", "Stock", "CMP", "Exp", "Score", "Action"], rows)


def _prediction_table(selected):
    if selected is None or selected.empty:
        return []
    rows = [[
        str(r.get("Symbol", "-")),
        _fmt(r.get("Pred_Open")),
        _fmt(r.get("Pred_High")),
        _fmt(r.get("Pred_Low")),
        _fmt(r.get("Pred_Close")),
        _pct(r.get("Expected_Return")),
    ] for _, r in selected.iterrows()]
    return ["📈 *PREDICTED OHLCV*"] + _table(["Stock", "Open", "High", "Low", "Close", "Exp"], rows)


def _horizon_table(selected):
    if selected is None or selected.empty:
        return []
    rows = [[str(r.get("Symbol", "-"))] + [_pct(r.get(f"Horizon_{h}D")) for h in (1, 3, 5, 7, 20)] for _, r in selected.iterrows()]
    return ["🔮 *MULTI-HORIZON OUTLOOK*"] + _table(["Stock", "1D", "3D", "5D", "7D", "20D"], rows)


def _jump(j):
    if j is None or j.empty:
        return []
    rows = []
    for _, r in j.iterrows():
        cp = float(r.get("Current_Price", 0) or 0)
        target = float(r.get("Target_Level", 0) or 0)
        rows.append([str(r.get("Symbol", "-")), f"₹{_fmt(cp)}", f"₹{_fmt(target)}", _pct((target / cp - 1) * 100 if cp else 0), f"{float(r.get('Jump_Probability', 0) or 0):.0f}%"])
    return ["🔥 *JUMP WATCH*"] + _table(["Stock", "CMP", "Target", "Upside", "Prob"], rows)


def _intraday(x):
    if x is None or x.empty:
        return []
    rows = [[str(r.get("Symbol", "-")), str(r.get("Bias", "-")), f"₹{_fmt(r.get('Current'))}", f"₹{_fmt(r.get('Target'))}", f"₹{_fmt(r.get('StopLoss'))}", f"{float(r.get('Confidence', 0) or 0):.0f}%"] for _, r in x.iterrows()]
    return ["⚡ *INTRADAY*"] + _table(["Stock", "Bias", "CMP", "Target", "SL", "Conf"], rows)


def _ipo(x):
    if x is None or (hasattr(x, "empty") and x.empty):
        return ["🏦 *IPO*", *_table(["Status"], [["No active/upcoming IPOs"]])]
    rows = [[str(r.get("IPOName", "-")), f"₹{_fmt(r.get('PriceHigh', 0), 0)}", f"₹{_fmt(r.get('GMPValue', 0), 0)}", _pct(r.get("GMPPct", 0)), str(r.get("IPOAction", "WATCH"))] for _, r in x.iterrows()]
    return ["🏦 *IPO INTELLIGENCE*"] + _table(["IPO", "Price", "GMP", "GMP%", "Action"], rows)


def _portfolio(p):
    if not p:
        return []
    rows = []
    for x in p.get("Rows", []):
        if isinstance(x, dict):
            rows.append([
                x.get("Stock", "-"), x.get("Quantity", "-"), x.get("Decision", "-"),
                x.get("Current_Price", "-"), x.get("Average_Price", "-"),
                x.get("Profit_Target", "-"), x.get("Sell_Window", "-"),
            ])
    lines = [
        "💼 *AI PORTFOLIO MANAGER*",
        *_table(["Stock", "Qty", "Decision", "CMP", "Avg", "Target", "Window"], rows),
    ]
    if p.get("AveragePlans"):
        lines += ["➕ *AVERAGING PLANS*", *_table(["Stock", "Add", "CMP", "NewAvg", "Target", "Window"], [[
            x.get("Stock", "-"), x.get("Recommended_Qty", 0), x.get("Current_Price", "-"),
            x.get("New_Average_Price", "-"), x.get("Profit_Target", "-"), x.get("Sell_Window", "-"),
        ] for x in p["AveragePlans"]])]
    if p.get("SellAlerts"):
        lines += ["🚨 *SELL / PROFIT-BOOK ALERTS*", *_table(["Stock", "CMP", "Target", "When", "Reason"], [[
            x.get("Stock", "-"), x.get("Current_Price", "-"), x.get("Profit_Target", "-"),
            x.get("Sell_Window", "NOW"), x.get("Reason", "-"),
        ] for x in p["SellAlerts"]])]
    return lines


def morning_report(prediction_date, cutoff_date, selected, jump_watchlist, intraday, **kwargs):
    accuracy = kwargs.get("accuracy", {})
    scan = kwargs.get("scan", {})
    portfolio = kwargs.get("portfolio", {})
    snapshot = kwargs.get("market_snapshot", {})
    regime = kwargs.get("regime", "-")
    ipo = kwargs.get("ipo", pd.DataFrame())
    lines = [
        f"📈 *AI NSE MORNING REPORT*\n📅 {prediction_date}\n⚙️ {MODEL_VERSION}",
        _SECTION,
        "📊 *MARKET OVERVIEW*",
        *_market(snapshot, regime),
        _scan(scan),
        f"Accuracy: {_accuracy(accuracy.get('PreviousAccuracy'))} → {_accuracy(accuracy.get('CurrentAccuracy'))} | Samples {int(accuracy.get('AccuracySamples', accuracy.get('Samples', 0)) or 0)}",
        _SECTION,
        *_bucket_sections(selected),
        _SECTION,
        *_best_pick_table(selected),
        _SECTION,
        *_prediction_table(selected),
        _SECTION,
        *_horizon_table(selected),
        _SECTION,
        *_jump(jump_watchlist),
        _SECTION,
        *_intraday(intraday),
        _SECTION,
        *_ipo(ipo),
        _SECTION,
        *_portfolio(portfolio),
    ]
    return "\n".join(lines)


def _evening_bucket_sections(evaluation):
    if evaluation is None or evaluation.empty:
        return ["No predictions available for evaluation."]
    if "PriceBucket" not in evaluation.columns:
        return ["No price bucket data."]
    lines = ["📋 *PREDICTION vs ACTUAL — BY PRICE BUCKET*"]
    for bucket in _ordered_price_buckets(evaluation["PriceBucket"]):
        g = evaluation[evaluation["PriceBucket"].astype(str) == bucket].copy()
        if g.empty:
            continue
        rows = []
        for _, r in g.sort_values("APE_Close", key=lambda s: s.abs(), kind="mergesort").head(6).iterrows():
            ok = "OK" if bool(r.get("DirectionCorrect", False)) else "NO"
            rows.append([
                f"{r.get('Symbol', '-')}/{ok}",
                f"{_fmt(r.get('Pred_Open'))}/{_fmt(r.get('Actual_Open'))}",
                f"{_fmt(r.get('Pred_High'))}/{_fmt(r.get('Actual_High'))}",
                f"{_fmt(r.get('Pred_Low'))}/{_fmt(r.get('Actual_Low'))}",
                f"{_fmt(r.get('Pred_Close'))}/{_fmt(r.get('Actual_Close'))}",
                f"{abs(float(r.get('APE_Close', 0) or 0)):.2f}%",
            ])
        lines += [_SECTION, f"💎 *₹ {bucket}*", *_table(["Stock", "Open P/A", "High P/A", "Low P/A", "Close P/A", "APE"], rows)]
    return lines


def _evening_accuracy_table(bucket):
    return ["No sufficient bucket sample"] if not bucket else _table(["Bucket", "Accuracy"], [[k, _accuracy(v)] for k, v in bucket.items()])


def _evening_horizon_table(h):
    return ["No horizon target matured yet"] if not h else _table(["Horizon", "Accuracy", "Samples"], [[f"{k}D", f"{v.get('Accuracy', 0):.1f}%", v.get("Samples", 0)] for k, v in sorted(h.items(), key=lambda x: int(x[0]))])


def evening_report(market_date, evaluation, metrics, retraining, **kwargs):
    accuracy = kwargs.get("accuracy", {})
    scan = kwargs.get("scan", {})
    bucket = kwargs.get("bucket_metrics", {})
    portfolio = kwargs.get("portfolio", {})
    learning = kwargs.get("learning", {})
    horizon = kwargs.get("horizon_metrics", {})
    lines = [
        f"🌙 *AI NSE EVENING REPORT*\n📅 {market_date}\n⚙️ {MODEL_VERSION}",
        _SECTION,
        "📊 *SCAN & ACCURACY*",
        _scan(scan),
        f"Accuracy: {_accuracy(accuracy.get('PreviousAccuracy'))} → {_accuracy(accuracy.get('CurrentAccuracy'))}",
        _SECTION,
        *_evening_bucket_sections(evaluation),
        _SECTION,
        "📊 *BUCKET ACCURACY*",
        *_evening_accuracy_table(bucket),
        _SECTION,
        "🎯 *HORIZON ACCURACY*",
        *_evening_horizon_table(horizon),
        _SECTION,
        "🧠 *MODEL LEARNING*",
        *_table(["Metric", "Value"], [
            ["Samples", metrics.get("Samples", 0)],
            ["Overall MAPE", f"{metrics.get('OverallMAPE', 0):.3f}%"],
            ["Close MAPE", f"{metrics.get('CloseMAPE', 0):.3f}%"],
            ["Direction Acc", f"{metrics.get('DirectionAccuracy', 0):.1f}%"],
            ["Champion", retraining.get("Decision", "-")],
            ["Replaced", "YES" if retraining.get("Retrained") else "NO"],
            ["Improvement", f"{retraining.get('Improvement', 0):+.2f}%"],
            ["Learning", learning.get("status", "UPDATED")],
        ]),
        _SECTION,
        *_portfolio(portfolio),
    ]
    return "\n".join(lines)
