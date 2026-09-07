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
    """Convert the small Markdown subset used by our reports to safe Telegram HTML.

    Report values are treated as data and HTML-escaped before being sent. This avoids
    Telegram entity-parser failures caused by stock symbols, punctuation, IPO names,
    negative numbers, or other dynamic values containing Markdown characters.
    """
    chunks = text.split("```")
    rendered = []
    for i, chunk in enumerate(chunks):
        if i % 2 == 1:
            rendered.append(f"<pre>{html.escape(chunk.strip(chr(10)))}</pre>")
            continue

        safe = html.escape(chunk)
        # Our generated report only uses *text* for section headings. Convert that
        # controlled subset to HTML bold; all other characters remain literal text.
        safe = re.sub(r"\*([^*\n]+)\*", r"<b>\1</b>", safe)
        rendered.append(safe)
    return "".join(rendered)


def _plain_text_fallback(text):
    """Return a readable unformatted message for the last-resort Telegram retry."""
    return text.replace("```", "").replace("*", "").replace("_", "")


def _post_telegram(message, token, chat_id):
    """Send one message, retrying without formatting if Telegram rejects HTML."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": _markdown_to_telegram_html(message), "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code == 200:
            return True
        print("Telegram HTML error:", r.text)
    except Exception as exc:
        print("Telegram HTML exception:", exc)

    # Formatting must never prevent delivery of a generated report. Telegram's
    # plain-text mode has no entity parser, so dynamic values cannot break it.
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
    # Do not short-circuit on the first failed section: attempt every section so a
    # transient Telegram error cannot silently suppress the rest of the report.
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

