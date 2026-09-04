"""Mobile-first Telegram reports for Stage 10.1 NSE AI pipeline."""
import os
import requests
import pandas as pd
from .config import TELEGRAM_MAX_LENGTH, MODEL_VERSION, PRICE_BUCKET_NAMES
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


def _fmt_accuracy(v):
    return "-" if v is None or pd.isna(v) else f"{float(v):.1f}%"


def _num(v, digits=2):
    try:
        return f"{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return "-"


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


def _stock_card(r, model_warning=False):
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
    decision = _decision(r, expected, rr, model_warning)
    volume = _num(r.get("Current_Volume", 0), 0)
    return [
        f"*{r.get('Symbol', '-')}*  |  *{decision}*",
        f"CMP  ₹{_num(cmp)}   →   AI Target  ₹{_num(pred)}   (*{expected:+.1f}%*)",
        f"OHLC  O ₹{_num(r.get('Current_Open', 0))}  H ₹{_num(r.get('Current_High', 0))}  L ₹{_num(r.get('Current_Low', 0))}  C ₹{_num(r.get('Current_Close', cmp))}",
        f"Volume  {volume}   |   SL  {sl_text}   |   R/R  {rr}",
        f"Confidence  {float(r.get('Confidence', 0) or 0):.0f}%",
    ]


def _bucket_sections(selected, model_warning=False):
    """Render compact one-card-per-bucket output for Telegram mobile screens."""
    if selected is None or selected.empty:
        return ["No stock passed the quality gate today."]
    lines = []
    if "PriceBucket" not in selected.columns:
        return ["No price bucket was assigned."]
    for bucket in list(PRICE_BUCKET_NAMES) + [b for b in selected["PriceBucket"].astype(str).unique() if b not in PRICE_BUCKET_NAMES]:
        group = selected[selected["PriceBucket"].astype(str) == bucket].copy()
        if group.empty:
            continue
        sort_cols = [c for c in ["TradeConfidence", "Score", "Confidence"] if c in group.columns]
        if sort_cols:
            group = group.sort_values(sort_cols, ascending=False)
        row = group.iloc[0]
        lines += [f"\n💎 *BUCKET {bucket}*", *_stock_card(row, model_warning)]
    return lines


def _accuracy_lines(a, compact=False):
    main = f"🤖 Accuracy  {_fmt_accuracy(a.get('PreviousAccuracy'))} → {_fmt_accuracy(a.get('CurrentAccuracy'))}  |  {int(a.get('AccuracySamples', a.get('Samples', 0)) or 0)} validated"
    if compact:
        return [main]
    return [
        main,
        f"Horizons  1D {_fmt_accuracy(a.get('1D'))} • 3D {_fmt_accuracy(a.get('3D'))} • 5D {_fmt_accuracy(a.get('5D'))}",
        f"          7D {_fmt_accuracy(a.get('7D'))} • 20D {_fmt_accuracy(a.get('20D'))} • Dir {_fmt_accuracy(a.get('DirectionAccuracy'))}",
        f"🧠 Health  {a.get('Health', '-')} • Drift {a.get('Drift', '-')} • 7D {_fmt_accuracy(a.get('Trend7D'))} • 30D {_fmt_accuracy(a.get('Trend30D'))}",
    ]


def _market_lines(snapshot, regime):
    n = snapshot.get("NIFTY", {})
    b = snapshot.get("BANKNIFTY", {})
    v = snapshot.get("VIX", {})
    br = snapshot.get("Breadth", {})
    def q(x):
        return "-" if x is None or pd.isna(x) else f"{float(x):.2f}"
    return [
        f"📊 NIFTY  {q(n.get('Close'))} ({q(n.get('Change1D'))}%)  |  BANK NIFTY  {q(b.get('Close'))} ({q(b.get('Change1D'))}%)",
        f"VIX {q(v.get('Close'))}  |  Breadth {int(br.get('Advancers', 0))}↑ / {int(br.get('Decliners', 0))}↓  |  Regime {regime}",
    ]


def _sector_lines(selected):
    if selected is None or selected.empty or "Sector" not in selected.columns:
        return []
    x = selected.copy()
    x["_score"] = pd.to_numeric(x.get("SectorStrength", 50), errors="coerce").fillna(50)
    g = x.groupby("Sector")["_score"].mean().sort_values(ascending=False).head(5)
    return ["🏭 Sector AI  " + " • ".join(f"{k} {v:.0f}" for k, v in g.items())]


def _scan_lines(scan):
    return [f"🔎 Scan  {int(scan.get('Universe', 0)):,} → {int(scan.get('Data', 0)):,} data → {int(scan.get('Liquid', 0)):,} liquid → {int(scan.get('AI', 0)):,} AI → {int(scan.get('Selected', 0)):,} picks"]


def _portfolio_lines(p):
    if not p:
        return ["💼 Portfolio  No portfolio CSV / no verified prices."]
    lines = [f"💼 *PORTFOLIO*  {p.get('Positions', 0)} positions  |  Value ₹{p.get('Value', 0):,.0f}  |  P&L ₹{p.get('PnL', 0):+,.0f} ({p.get('Return', 0):+.2f}%)"]
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
    lines = [
        "📈 *AI NSE MORNING REPORT*",
        f"📅 {prediction_date}  |  Data till {cutoff_date}",
        f"⚙️ {MODEL_VERSION}",
        *_scan_lines(scan),
        *_accuracy_lines(accuracy, compact=True),
        "",
        "🎯 *TOP STOCKS — BEST PICK BY PRICE BUCKET*",
        "_Each bucket is independent; no global Top-5 cap._",
    ]
    if warning:
        lines.append("⚠️ *MODEL WARNING*  Weak health/drift → BUY signals downgraded to WATCH.")
    lines += _market_lines(snapshot, regime) + _sector_lines(selected) + _bucket_sections(selected, warning)

    lines.append("\n🆕 *IPO / NEW LISTINGS*")
    if ipo is not None and not ipo.empty:
        for _, r in ipo.head(5).iterrows():
            lines.append(f"• *{r.get('IPOName', '-')}*  ₹{float(r.get('PriceHigh', 0)):.0f}  |  GMP ₹{float(r.get('GMPValue', 0)):.0f} ({float(r.get('GMPPct', 0)):.1f}%)  |  {r.get('IPOAction', 'WATCH')}")
        lines.append("_GMP is unofficial; not a guaranteed listing price._")
    else:
        lines.append("No verified live IPO records available.")

    lines += _portfolio_lines(portfolio)
    lines.append("\n🔥 *+5% JUMP WATCH — TOP 5*")
    if jump_watchlist is not None and not jump_watchlist.empty:
        for i, (_, r) in enumerate(jump_watchlist.head(5).iterrows(), 1):
            cp = float(r.get("Current_Price", 0) or 0)
            target = float(r.get("Target_Level", 0) or 0)
            prob = float(r.get("Jump_Probability", 0) or 0)
            target_pct = (target / cp - 1) * 100 if cp else 0
            lines += [f"{i}. *{r.get('Symbol', '-')}*  ₹{cp:,.2f} → ₹{target:,.2f}  |  {target_pct:+.1f}%  |  7D {float(r.get('Estimated_7D_Upside', 0) or 0):+.1f}%  |  Prob {prob:.0f}%"]
    else:
        lines.append("No strong +5% candidate passed the gate.")

    lines.append("\n⚡ *INTRADAY — TOP 5*")
    if intraday is not None and not intraday.empty:
        for i, (_, r) in enumerate(intraday.head(5).iterrows(), 1):
            lines.append(f"{i}. *{r.get('Symbol', '-')}*  {r.get('Bias', '-')}  |  ₹{_num(r.get('Current', 0))} → ₹{_num(r.get('Target', 0))}  |  SL ₹{_num(r.get('StopLoss', 0))}  |  {float(r.get('Confidence', 0) or 0):.0f}%")
    else:
        lines.append("No confirmed intraday setup today.")
    lines.append("\n⚠️ *Decision support only — not financial advice.*")
    return "\n".join(lines)


def evening_report(market_date, evaluation, metrics, retraining, **kwargs):
    bucket = kwargs.get("bucket_metrics", {})
    intraday = kwargs.get("intraday_metrics", {})
    ipo = kwargs.get("ipo_results")
    learning = kwargs.get("learning", {})
    accuracy = kwargs.get("accuracy", {})
    scan = kwargs.get("scan", {})
    portfolio = kwargs.get("portfolio", {})
    lines = [
        "🌙 *AI NSE EVENING REPORT*",
        f"📅 {market_date}  |  {MODEL_VERSION}",
        *_scan_lines(scan),
        *_accuracy_lines(accuracy),
        "",
        "🎯 *PREDICTION vs ACTUAL — BUCKET PICKS*",
    ]
    if evaluation is not None and not evaluation.empty:
        for _, r in evaluation.iterrows():
            close_diff = float(r.get("Diff_Close", 0) or 0)
            close_pct = (close_diff / float(r.get("Actual_Close", 1) or 1)) * 100
            correct = "✅" if bool(r.get("DirectionCorrect", False)) else "❌"
            lines += [
                f"\n*{r.get('Symbol', '-')}*  {correct}",
                f"PRED  O {_num(r.get('Pred_Open'))}  H {_num(r.get('Pred_High'))}  L {_num(r.get('Pred_Low'))}  C {_num(r.get('Pred_Close'))}",
                f"ACT   O {_num(r.get('Actual_Open'))}  H {_num(r.get('Actual_High'))}  L {_num(r.get('Actual_Low'))}  C {_num(r.get('Actual_Close'))}",
                f"DIFF  O {float(r.get('Diff_Open', 0) or 0):+.2f}  H {float(r.get('Diff_High', 0) or 0):+.2f}  L {float(r.get('Diff_Low', 0) or 0):+.2f}  C {close_diff:+.2f} ({close_pct:+.2f}%)",
                f"Direction  {r.get('Pred_Direction', '-')} → {r.get('Actual_Direction', '-')}",
            ]
    else:
        lines.append("No predictions available for evaluation.")

    lines.append("\n🔥 *RESULTS*")
    lines.append("Buckets  " + (" • ".join(f"{k} {v:.1f}%" for k, v in bucket.items()) if bucket else "not enough samples"))
    if intraday:
        lines.append("Intraday  " + " • ".join(f"{k} {v}" for k, v in intraday.items()))
    if ipo is not None and not ipo.empty:
        lines.append(f"IPO/new listings evaluated  {len(ipo)}")
    lines += _portfolio_lines(portfolio)

    lines += [
        "\n🧠 *MODEL LEARNING*",
        f"Samples  {metrics.get('Samples', 0)}",
        f"Overall MAPE  {metrics.get('OverallMAPE', 0):.3f}%",
        f"Close MAPE  {metrics.get('CloseMAPE', 0):.3f}%",
        f"Direction Accuracy  {metrics.get('DirectionAccuracy', 0):.1f}%",
        f"Accuracy  {_fmt_accuracy(accuracy.get('PreviousAccuracy'))} → {_fmt_accuracy(accuracy.get('CurrentAccuracy'))}",
        f"Champion/Challenger  {retraining.get('Decision', '-')}",
        f"Model Replaced  {'YES' if retraining.get('Retrained') else 'NO'}",
        f"Improvement  {retraining.get('Improvement', 0):+.2f}%",
        f"Learning State  {learning.get('status', 'UPDATED')}",
        "\n⚠️ *Decision support only — not financial advice.*",
    ]
    return "\n".join(lines)
