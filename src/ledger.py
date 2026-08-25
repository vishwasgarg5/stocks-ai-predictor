import io
from datetime import datetime, timezone
import pandas as pd

PREDICTION_PATH = "data/predictions.csv"

COLUMNS = [
    "created_at","target_date","symbol","rank","score","base_close",
    "pred_open","pred_high","pred_low","pred_close",
    "actual_open","actual_high","actual_low","actual_close",
    "open_error","high_error","low_error","close_error","direction_correct"
]


def read_ledger(text: str = "") -> pd.DataFrame:
    if not text:
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(io.StringIO(text))


def ledger_text(df: pd.DataFrame) -> str:
    return df.reindex(columns=COLUMNS).to_csv(index=False)


def save_prediction(ledger: pd.DataFrame, p: dict, rank: int, score: float, target_date=None) -> pd.DataFrame:
    row = {c: None for c in COLUMNS}
    row.update({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_date": target_date,
        "symbol": p["symbol"], "rank": rank, "score": score,
        "base_close": p["base_close"], "pred_open": p["pred_open"],
        "pred_high": p["pred_high"], "pred_low": p["pred_low"], "pred_close": p["pred_close"]
    })
    return pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)


def evaluate_pending(ledger: pd.DataFrame, actuals: dict[str, dict]) -> tuple[pd.DataFrame, int]:
    updated = 0
    ledger = ledger.copy()
    pending = ledger[ledger["actual_close"].isna()].index
    for idx in pending:
        symbol = ledger.at[idx, "symbol"]
        a = actuals.get(symbol)
        if not a:
            continue
        base = float(ledger.at[idx, "base_close"])
        pc = float(ledger.at[idx, "pred_close"])
        ledger.loc[idx, ["target_date","actual_open","actual_high","actual_low","actual_close"]] = [a.get("date"),a["open"],a["high"],a["low"],a["close"]]
        ledger.loc[idx, ["open_error","high_error","low_error","close_error"]] = [a["open"]-ledger.at[idx,"pred_open"],a["high"]-ledger.at[idx,"pred_high"],a["low"]-ledger.at[idx,"pred_low"],a["close"]-pc]
        ledger.at[idx, "direction_correct"] = int((pc > base) == (a["close"] > base))
        updated += 1
    return ledger, updated


def performance_report(ledger: pd.DataFrame) -> pd.DataFrame:
    done = ledger.dropna(subset=["actual_close"])
    if done.empty:
        return pd.DataFrame()
    return pd.DataFrame([{
        "predictions": len(done),
        "open_mae": done.open_error.abs().mean(),
        "high_mae": done.high_error.abs().mean(),
        "low_mae": done.low_error.abs().mean(),
        "close_mae": done.close_error.abs().mean(),
        "direction_accuracy": done.direction_correct.mean()
    }])
