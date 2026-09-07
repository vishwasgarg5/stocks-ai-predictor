"""Production data-quality gate used by scheduled Stage 10.5 workflows."""
from __future__ import annotations

import sys

from .data_quality import validate_universe
from .ledger import load_predictions, latest_prediction_date
from .market_data import (
    download_many,
    get_completed_session_date,
    get_data_cutoff_date,
    load_universe,
)
from .utils import is_weekday


def _fail(failed: dict[str, list[str]], label: str) -> None:
    if not failed:
        return
    sample = ", ".join(f"{s}:{'|'.join(e)}" for s, e in list(failed.items())[:10])
    raise RuntimeError(f"LIVE_DATA_QUALITY_FAILED {label} count={len(failed)} sample={sample}")


def morning_gate() -> dict:
    """Validate the same broad market universe before morning model execution."""
    if not is_weekday():
        return {"Status": "SKIP_WEEKEND"}
    universe = load_universe()
    raw = download_many(universe, "1mo", workers=8)
    fallback = get_completed_session_date("morning")
    cutoff = get_data_cutoff_date(raw, None, fallback=fallback)
    if cutoff is None:
        raise RuntimeError("LIVE_DATA_QUALITY_FAILED: no completed market cutoff")
    passed, failed = validate_universe(raw, cutoff_date=cutoff, min_rows=30)
    _fail(failed, "MORNING")
    if len(passed) < 20:
        raise RuntimeError(f"LIVE_DATA_QUALITY_FAILED MORNING valid_stocks={len(passed)} < 20")
    return {"Status": "PASS", "Cutoff": str(cutoff), "Validated": len(passed), "Failed": 0}


def evening_gate() -> dict:
    """Validate the exact immutable morning prediction symbols against today's session."""
    if not is_weekday():
        return {"Status": "SKIP_WEEKEND"}
    market_date = get_completed_session_date("evening")
    if market_date is None:
        return {"Status": "SKIP_NO_SESSION"}
    prediction_date = latest_prediction_date(market_date)
    if prediction_date is None:
        raise RuntimeError(f"LIVE_DATA_QUALITY_FAILED EVENING: no exact prediction for {market_date}")
    predictions = load_predictions(prediction_date)
    symbols = predictions.get("Symbol", []).astype(str).dropna().unique().tolist()
    if not symbols:
        raise RuntimeError("LIVE_DATA_QUALITY_FAILED EVENING: prediction ledger has no symbols")
    raw = download_many(symbols, "1mo", workers=5)
    passed, failed = validate_universe(raw, cutoff_date=market_date, min_rows=30)
    _fail(failed, "EVENING")
    missing = sorted(set(symbols) - set(passed))
    if missing:
        raise RuntimeError(f"LIVE_DATA_QUALITY_FAILED EVENING missing_symbols={','.join(missing[:10])}")
    return {"Status": "PASS", "MarketDate": str(market_date), "PredictionDate": str(prediction_date), "Validated": len(passed), "Failed": 0}


def main() -> None:
    mode = (sys.argv[1] if len(sys.argv) > 1 else "morning").lower()
    result = morning_gate() if mode == "morning" else evening_gate() if mode == "evening" else None
    if result is None:
        raise SystemExit("Usage: python -m src.live_quality_gate [morning|evening]")
    print(f"LIVE_DATA_QUALITY {result}")


if __name__ == "__main__":
    main()
