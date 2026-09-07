"""Telegram report builders for Stage 10.5 morning and evening reports."""
import html
import os
import re
import requests
import pandas as pd
from .config import TELEGRAM_MAX_LENGTH, MODEL_VERSION

_SECTION = "\n§§TELEGRAM_SECTION§§\n"
_BUCKET_ORDER = ["10-49", "50-99", "100-249", "250-499", "500-999", "1000-2499", ">2500"]
_BUCKET_LABELS = {"B1": ">2500", "B2": "1000-2499", "B3": "500-999", "B4": "250-499", "B5": "100-249", "B6": "50-99", "B7": "10-49"}
_CODE_RE = re.compile(r"^```$")


def _markdown_to_telegram_html(text):
    chunks = text.split("```")
    out = []
    for i, chunk in enumerate(chunks):
        if i % 2:
            out.append(f"<pre>{html.escape(chunk.strip(chr(10)))}</pre>")
        else:
            safe = html.escape(chunk)
            safe = re.sub(r"\*([^*\n]+)\*", r"<b>\1</b>", safe)
            out.append(safe)
    return "".join(out)


def _plain_text_fallback(text):
    return text.replace("```", "").replace("*", "").replace("_", "")


def _post_telegram(message, token, chat_id):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": _markdown_to_telegram_html(message), "parse_mode": "HTML"}, timeout=20)
        if r.status_code == 200:
            return True
        print("Telegram HTML error:", r.text)
    except Exception as exc:
        print("Telegram HTML exception:", exc)
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": _plain_text_fallback(message)}, timeout=20)
        if r.status_code == 200:
            return True
        print("Telegram plain-text fallback error:", r.text)
    except Exception as exc:
        print("Telegram plain-text fallback exception:", exc)
    return False


def _split_safe(text, max_length):
    if len(text) <= max_length:
        return [text]
    result, current, in_code = [], "", False
    for line in text.splitlines(True):
        toggle = bool(_CODE_RE.match(line.strip()))
        if current and len(current) + len(line) + (4 if in_code else 0) > max_length:
            if in_code:
                current += "```\n"
            result.append(current.rstrip("\n"))
            current = "```\n" if in_code else ""
        current += line
        if toggle:
            in_code = not in_code
    if current:
        if in_code:
            current += "```\n"
        result.append(current.rstrip("\n"))
    return [x for x in result if x]


def _report_messages(text):
    parts = [p.strip() for p in text.split(_SECTION) if p.strip()]
    if not parts:
        return []
    groups, current = [], []
    boundaries = {"BEST PICK", "JUMP WATCH", "MODEL LEARNING", "AI PORTFOLIO MANAGER"}
    for part in parts:
        current.append(part)
        if any(x in part.upper() for x in boundaries):
            groups.append("\n\n".join(current))
            current = []
    if current:
        groups.append("\n\n".join(current))
    out = []
    for group in groups:
        out.extend(_split_safe(group, TELEGRAM_MAX_LENGTH))
    return out


def send_telegram(text):
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram secrets not configured.")
        return False
    messages = _report_messages(text)
    print(f"Telegram report: sending {len(messages)} message(s).")
    return bool(messages) and all(_post_telegram(m, token, chat_id) for m in messages)


def _fmt(v, digits=2):
    try:
        x = float(v)
        return "-" if not pd.notna(x) else f"{x:,.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _pct(v):
    try:
        x = float(v)
        return "-" if not pd.notna(x) else f"{x:+.1f}%"
    except (TypeError, ValueError):
        return "-"


def _accuracy(v):
    try:
        x = float(v)
        return "-" if not pd.notna(x) else f"{x:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _table(headers, rows, max_width=12):
    if not rows:
        return []
    rows = [[str(x).replace("|", "/").replace("\n", " ") for x in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            if i < len(widths):
                widths[i] = min(max(widths[i], len(value)), max_width)
    def fit(v, w):
        return v if len(v) <= w else v[:max(1, w - 1)] + "…"
    def line(row):
        return " | ".join(fit(v, widths[i]).ljust(widths[i]) for i, v in enumerate(row))
    sep = "-" * (sum(widths) + 3 * (len(widths) - 1) + 2)
    return ["```", line(headers), sep, *[line(r) for r in rows], "```"]


def _bucket_label(bucket, group=None):
    if group is not None and "PriceBucketLabel" in group.columns:
        values = group["PriceBucketLabel"].dropna().astype(str)
        if not values.empty:
            return values.iloc[0]
    return _BUCKET_LABELS.get(str(bucket), str(bucket))


def _ordered_price_buckets(values):
    present = [str(x) for x in pd.Series(values).dropna().unique()] if values is not None else []
    codes = [x for x in present if x.startswith("B")]
    if codes:
        order = ["B7", "B6", "B5", "B4", "B3", "B2", "B1"]
        return [x for x in order if x in codes] + [x for x in present if x not in order]
    return [x for x in _BUCKET_ORDER if x in present] + [x for x in present if x not in _BUCKET_ORDER]


def _market(snapshot, regime):
    snapshot = snapshot or {}
    rows = []
    for name, key in [("NIFTY", "NIFTY"), ("BANK", "BANKNIFTY"), ("FINN", "FINNIFTY"), ("MIDCP", "MIDCPNIFTY")]:
        x = snapshot.get(key, {}) or {}
        rows.append([name, _fmt(x.get("Close")), _pct(x.get("Change1D"))])
    x = snapshot.get("VIX", {}) or {}
    rows.append(["VIX", _fmt(x.get("Close")), "-"])
    b = snapshot.get("Breadth", {}) or {}
    return _table(["Index", "Value", "1D%"], rows) + [f"Breadth: {int(b.get('Advancers',0) or 0)}↑ / {int(b.get('Decliners',0) or 0)}↓ | Regime: {regime or '-'}"]


def _scan(scan):
    scan = scan or {}
    return f"Scanned {int(scan.get('Universe',0) or 0):,} | Data {int(scan.get('Data',0) or 0):,} | Liquid {int(scan.get('Liquid',0) or 0):,} | AI {int(scan.get('AI',0) or 0):,} | Qualified {int(scan.get('Selected',0) or 0):,}"


def _score_col(g):
    for c in ("FinalDecisionScore", "TradeConfidence", "Score"):
        if c in g.columns:
            return c
    return None


def _sort(g):
    c = _score_col(g)
    return g.sort_values(c, ascending=False, kind="mergesort") if c else g.sort_values("Symbol", kind="mergesort")


def _bucket_sections(selected):
    if selected is None or selected.empty or "PriceBucket" not in selected.columns:
        return ["🎯 *QUALIFIED STOCKS BY PRICE BUCKET*", "No qualifying stock today."]
    lines = ["🎯 *QUALIFIED STOCKS BY PRICE BUCKET*"]
    for bucket in _ordered_price_buckets(selected["PriceBucket"]):
        g = selected[selected["PriceBucket"].astype(str) == bucket].copy()
        if g.empty:
            continue
        rows = []
        for _, r in _sort(g).head(6).iterrows():
            try: cmp = float(r.get("Current_Price", r.get("Current_Close", 0)) or 0)
            except (TypeError, ValueError): cmp = 0
            try: pred = float(r.get("Pred_Close", 0) or 0)
            except (TypeError, ValueError): pred = 0
            exp = (pred / cmp - 1) * 100 if cmp else None
            rows.append([str(r.get("Symbol","-")), f"₹{_fmt(cmp)}", _pct(exp), _pct(r.get("Horizon_1D")), _pct(r.get("Horizon_5D")), _pct(r.get("Horizon_20D")), str(r.get("Action","-"))])
        lines += [f"💎 *₹ {_bucket_label(bucket,g)}* | MAX 6", *_table(["Stock","CMP","Exp","1D","5D","20D","Action"], rows)]
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
        score = _num(r.get("FinalDecisionScore", r.get("Score")), None)
        rows.append([f"₹{_bucket_label(bucket,g)}", str(r.get("Symbol","-")), f"₹{_fmt(r.get('Current_Price',r.get('Current_Close')))}", _pct(r.get("Expected_Return")), "-" if score is None else f"{score:.0f}", _decision(r.get("Action"))])
    return ["🏆 *BEST PICK — 1 PER PRICE BUCKET*", *_table(["Bucket","Stock","CMP","Exp","Score","Action"], rows)]


def _prediction_table(selected):
    if selected is None or selected.empty:
        return []
    rows = [[str(r.get("Symbol","-")), _fmt(r.get("Pred_Open")), _fmt(r.get("Pred_High")), _fmt(r.get("Pred_Low")), _fmt(r.get("Pred_Close"))] for _, r in _sort(selected).head(10).iterrows()]
    return ["📈 *PREDICTED OHLC — TOP 10*"] + _table(["Stock","Open","High","Low","Close"], rows)


def _horizon_table(selected):
    if selected is None or selected.empty: return []
    rows = [[str(r.get("Symbol","-"))] + [_pct(r.get(f"Horizon_{h}D")) for h in (1,3,5,7,20)] for _, r in _sort(selected).head(10).iterrows()]
    return ["🔮 *MULTI-HORIZON OUTLOOK — TOP 10*"] + _table(["Stock","1D","3D","5D","7D","20D"], rows)


def _jump(j):
    if j is None or j.empty:
        return []
    x = j.copy()
    if "Jump_Probability" in x.columns:
        x["_jump_sort"] = pd.to_numeric(x["Jump_Probability"], errors="coerce")
        x = x.sort_values("_jump_sort", ascending=False, na_position="last", kind="mergesort")
    rows = []
    for _, r in x.iterrows():
        symbol = str(r.get("Symbol","-")).strip()
        cp = _num(r.get("Current_Price")); target = _num(r.get("Target_Level")); prob = _num(r.get("Jump_Probability"))
        if not symbol or symbol.lower() in {"nan", "none"} or cp is None or cp <= 0 or target is None or target <= 0:
            continue
        prob = max(0.0, min(100.0, prob if prob is not None else 0.0))
        rows.append([symbol, f"₹{_fmt(cp)}", f"₹{_fmt(target)}", _pct((target/cp-1)*100), f"{prob:.0f}%"])
        if len(rows) == 5:
            break
    return ["🔥 *JUMP WATCH — TOP 5*"] + (_table(["Stock","CMP","Target","Upside","Prob"], rows) if rows else ["No valid jump candidates."])


def _intraday(x):
    if x is None or x.empty: return []
    rows = [[str(r.get("Symbol","-")), str(r.get("Bias","-")), f"₹{_fmt(r.get('Current'))}", f"₹{_fmt(r.get('Target'))}", f"₹{_fmt(r.get('StopLoss'))}", f"{float(r.get('Confidence',0) or 0):.0f}%"] for _, r in x.iterrows()]
    return ["⚡ *INTRADAY*"] + _table(["Stock","Bias","CMP","Target","SL","Conf"], rows)


def _ipo(x):
    if x is None or (hasattr(x,"empty") and x.empty): return ["🏦 *IPO*", "No active/upcoming IPOs."]
    rows = [[str(r.get("IPOName","-")), f"₹{_fmt(r.get('PriceHigh',0),0)}", f"₹{_fmt(r.get('GMPValue',0),0)}", _pct(r.get("GMPPct",0)), str(r.get("IPOAction","WATCH"))] for _, r in x.iterrows()]
    return ["🏦 *IPO INTELLIGENCE*"] + _table(["IPO","Price","GMP","GMP%","Action"], rows)


def _portfolio(p):
    if not p:
        return []
    rows = []
    for x in p.get("Rows", []):
        if isinstance(x, dict):
            rows.append([x.get("Stock","-"), x.get("Quantity","-"), _decision(x.get("Decision")), x.get("Current_Price","-"), x.get("Average_Price","-"), x.get("Profit_Target","-"), x.get("Sell_Window","-")])
    lines = ["💼 *AI PORTFOLIO MANAGER*", "10% target is based on portfolio average; Window is based on the Top-10 AI multi-horizon forecast."]
    if rows:
        lines += _table(["Stock","Qty","Decision","CMP","Avg","10% Target","AI Window"], rows, max_width=18)
    if p.get("AveragePlans"):
        lines += ["➕ *AVERAGING PLANS*"] + _table(["Stock","Add","CMP","NewAvg","10% Target","AI Window"], [[x.get("Stock","-"),x.get("Recommended_Qty",0),x.get("Current_Price","-"),x.get("New_Average_Price","-"),x.get("Profit_Target","-"),x.get("Sell_Window","-")] for x in p["AveragePlans"]], max_width=18)
    if p.get("SellAlerts"):
        lines += ["🚨 *SELL / PROFIT-BOOK ALERTS*"] + _table(["Stock","CMP","Target","When","Reason"], [[x.get("Stock","-"),x.get("Current_Price","-"),x.get("Profit_Target","-"),x.get("Sell_Window","NOW"),x.get("Reason","-")] for x in p["SellAlerts"]], max_width=18)
    return lines


def morning_report(prediction_date, cutoff_date, selected, jump_watchlist, intraday, **kwargs):
    accuracy, scan = kwargs.get("accuracy",{}), kwargs.get("scan",{})
    snapshot, regime = kwargs.get("market_snapshot",{}), kwargs.get("regime","-")
    portfolio, ipo = kwargs.get("portfolio",{}), kwargs.get("ipo",pd.DataFrame())
    lines = [f"📈 *AI NSE MORNING REPORT*\n📅 {prediction_date}\n⚙️ {MODEL_VERSION}", _SECTION, "📊 *MARKET OVERVIEW*", *_market(snapshot,regime), _scan(scan), f"Accuracy: {_accuracy(accuracy.get('PreviousAccuracy'))} → {_accuracy(accuracy.get('CurrentAccuracy'))} | Samples {int(accuracy.get('AccuracySamples',accuracy.get('Samples',0)) or 0)}", _SECTION, *_bucket_sections(selected), _SECTION, *_best_pick_table(selected), _SECTION, *_prediction_table(selected), _SECTION, *_horizon_table(selected), _SECTION, *_jump(jump_watchlist), _SECTION, *_intraday(intraday), _SECTION, *_ipo(ipo), _SECTION, *_portfolio(portfolio)]
    return "\n".join(lines)


def _diff_pct(predicted, actual):
    try:
        p, a = float(predicted), float(actual)
        return None if abs(a) < 1e-12 else (a - p) / abs(a) * 100.0
    except (TypeError, ValueError):
        return None


def _evaluation_rows(evaluation, market_date=None, limit=10):
    if evaluation is None or evaluation.empty:
        return []
    e = evaluation.copy()
    if market_date is not None and "EvaluationDate" in e.columns:
        d = e["EvaluationDate"].astype(str).str[:10]
        target = str(market_date)[:10]
        latest = e[d == target]
        if not latest.empty:
            e = latest
    if "EvaluationDate" in e.columns:
        e = e.sort_values("EvaluationDate", kind="mergesort")
    e = e.drop_duplicates(subset=["Symbol"], keep="last").head(limit)
    rows = []
    for _, r in e.iterrows():
        symbol = str(r.get("Symbol", "-")).strip()
        if not symbol or symbol.lower() in {"nan", "none"}:
            continue
        bucket = _bucket_label(r.get("PriceBucket", "-"))
        pred = [r.get("Pred_Open"), r.get("Pred_High"), r.get("Pred_Low"), r.get("Pred_Close")]
        actual = [r.get("Actual_Open"), r.get("Actual_High"), r.get("Actual_Low"), r.get("Actual_Close")]
        diff = [_diff_pct(p, a) for p, a in zip(pred, actual)]
        rows.extend([
            [symbol, "Predicted", *[_fmt(v) for v in pred]],
            [symbol, "Actual", *[_fmt(v) for v in actual]],
            [symbol, "Difference%", *[_pct(v) for v in diff]],
        ])
    return rows


def evening_report(market_date, evaluation, metrics, retraining, **kwargs):
    """Build the complete Stage 10.5 evening report without touching model/data code."""
    metrics, retraining = metrics or {}, retraining or {}
    bucket_metrics = kwargs.get("bucket_metrics",{}) or {}
    horizon_metrics = kwargs.get("horizon_metrics",{}) or {}
    learning = kwargs.get("learning",{}) or {}
    accuracy = kwargs.get("accuracy",{}) or {}
    scan = kwargs.get("scan",{}) or {}
    portfolio = kwargs.get("portfolio",{}) or {}
    overall_acc = max(0.0, 100.0 - float(metrics.get("OverallMAPE",100.0) or 100.0))
    close_acc = max(0.0, 100.0 - float(metrics.get("CloseMAPE",100.0) or 100.0))
    lines = [
        f"🌙 *AI NSE EVENING EVALUATION*\n📅 {market_date}\n⚙️ {MODEL_VERSION}",
        _SECTION, "📊 *MODEL PERFORMANCE*",
        f"Samples: {int(metrics.get('Samples',0) or 0)} | Overall MAPE: {_pct(metrics.get('OverallMAPE'))}",
        f"Open MAPE: {_pct(metrics.get('OpenMAPE'))} | High MAPE: {_pct(metrics.get('HighMAPE'))}",
        f"Low MAPE: {_pct(metrics.get('LowMAPE'))} | Close MAPE: {_pct(metrics.get('CloseMAPE'))}",
        f"Accuracy: Overall {_accuracy(overall_acc)} | Close {_accuracy(close_acc)} | Direction {_accuracy(metrics.get('DirectionAccuracy'))}",
        _scan(scan), _SECTION, "📋 *PREDICTION vs ACTUAL*",
        "Predicted → Actual → signed Difference% for every OHLC line item.",
    ]
    rows = _evaluation_rows(evaluation, market_date=market_date, limit=10)
    lines += _table(["Stock","Type","Open","High","Low","Close"], rows, max_width=12) if rows else ["No completed stock evaluations."]
    lines += [_SECTION, "🎯 *PRICE-BUCKET ACCURACY*"]
    bucket_rows = [[_bucket_label(k),_accuracy(v)] for k,v in bucket_metrics.items()]
    lines += _table(["Price Bucket","Accuracy"], bucket_rows) if bucket_rows else ["No bucket metrics."]
    if horizon_metrics:
        lines += [_SECTION, "🔮 *HORIZON LEARNING*"]
        hrows = []
        for h in sorted(horizon_metrics, key=lambda x:int(x)):
            m = horizon_metrics[h] or {}
            hrows.append([f"{h}D",int(m.get("Samples",0) or 0),_accuracy(m.get("Accuracy")),_accuracy(m.get("DirectionAccuracy"))])
        lines += _table(["Horizon","N","Accuracy","Direction"], hrows)
    improvement = retraining.get("Improvement")
    lines += [
        _SECTION, "🧠 *MODEL LEARNING*",
        f"Retrained: {'YES' if retraining.get('Retrained',False) else 'NO'} | Decision: {str(retraining.get('Decision','-'))}",
        f"Improvement: {_pct(improvement)}" if improvement is not None else "Improvement: -",
        f"Previous accuracy: {_accuracy(accuracy.get('PreviousAccuracy'))} → Current accuracy: {_accuracy(accuracy.get('CurrentAccuracy'))}",
        f"Learning state: {str(learning.get('status','-'))} | Health: {str(learning.get('health','-'))} | Drift: {_pct(learning.get('drift'))}",
        f"Rollback: {'YES' if learning.get('rollback') else 'NO'}",
    ]
    if portfolio:
        lines += [_SECTION, *_portfolio(portfolio)]
    return "\n".join(lines)
