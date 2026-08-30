"""Evening job: evaluate today's predictions against exact target-date OHLC, then retrain only if challenger wins."""
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

from config import PREDICTIONS_CSV
from src.market_data import update_universe, update_nifty, get_nifty150_universe, fetch_exact_session
from src.ledger import read_ledger, ledger_text, evaluate_pending, performance_report
from src.features import add_features
from src.stage15 import apply_stage15_context
from src.retraining import champion_challenger
from src.telegram_report import send_evening

IST = ZoneInfo("Asia/Kolkata")


def run():
    universe = get_nifty150_universe()
    raw = update_universe(universe)
    nifty = update_nifty()
    print(f"Universe: Nifty 150 | constituents configured: {len(universe)} | usable OHLCV: {len(raw)}")
    if len(raw) < 50 or nifty.empty:
        raise RuntimeError(f"Insufficient data: {len(raw)} usable stocks")

    ledger = read_ledger(PREDICTIONS_CSV.read_text(encoding="utf-8") if PREDICTIONS_CSV.exists() else "")
    actuals = {}
    pending = ledger[ledger["actual_close"].isna()].copy() if not ledger.empty else pd.DataFrame()
    for _, r in pending.iterrows():
        symbol = str(r["symbol"])
        target = pd.to_datetime(r["target_date"], errors="coerce")
        if pd.isna(target):
            continue
        actual = fetch_exact_session(symbol, target.strftime("%Y-%m-%d"))
        if actual is not None:
            actuals.setdefault(symbol, {})[target.strftime("%Y-%m-%d")] = actual

    ledger, evaluated = evaluate_pending(ledger, actuals)
    retrained = False
    champion_mae = challenger_mae = None
    decision = "NO EVALUATION"
    if evaluated:
        features = {s: add_features(df, nifty) for s, df in raw.items()}
        features = apply_stage15_context(features, raw)
        _, retrained, champion_mae, challenger_mae, decision = champion_challenger(features)

    PREDICTIONS_CSV.write_text(ledger_text(ledger), encoding="utf-8")
    report = performance_report(ledger)
    if not report.empty:
        report.to_csv("reports/performance.csv", index=False)

    evaluated_dates = ledger.loc[ledger["actual_close"].notna(), "target_date"].astype(str) if not ledger.empty else pd.Series(dtype=str)
    session_date = max(evaluated_dates) if evaluated_dates.size else datetime.now(IST).date().isoformat()
    print(f"Market/evaluation session: {session_date}")
    print(f"Predictions evaluated this run: {evaluated}")
    print(f"Model retrained: {'YES' if retrained else 'NO'}")
    print(f"Model decision: {decision}")
    if champion_mae is not None:
        print(f"Champion validation MAE: {champion_mae:.6f}")
        print(f"Challenger validation MAE: {challenger_mae:.6f}")
    send_evening(session_date, evaluated, ledger, report, retrained, champion_mae, challenger_mae)


if __name__ == "__main__":
    run()
