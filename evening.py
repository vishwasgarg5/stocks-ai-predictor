"""Evening job: evaluate today's predictions, then retrain only if challenger wins."""
from config import NIFTY50, PREDICTIONS_CSV
from src.market_data import update_universe, update_nifty
from src.ledger import read_ledger, ledger_text, evaluate_pending, performance_report
from src.features import add_features
from src.retraining import champion_challenger
from src.telegram_report import send_evening


def run():
    raw = update_universe(NIFTY50)
    nifty = update_nifty()
    if len(raw) < 10 or nifty.empty:
        raise RuntimeError(f"Insufficient data: {len(raw)} stocks")
    ledger = read_ledger(PREDICTIONS_CSV.read_text(encoding="utf-8") if PREDICTIONS_CSV.exists() else "")
    actuals = {}
    for symbol, df in raw.items():
        if df.empty:
            continue
        actuals[symbol] = {}
        for date, r in df.iterrows():
            actuals[symbol][date.strftime("%Y-%m-%d")] = {"open": float(r["Open"]), "high": float(r["High"]), "low": float(r["Low"]), "close": float(r["Close"])}
    ledger, evaluated = evaluate_pending(ledger, actuals)
    retrained = False
    champion_mae = challenger_mae = None
    if evaluated:
        features = {s: add_features(df, nifty) for s, df in raw.items()}
        _, retrained, champion_mae, challenger_mae = champion_challenger(features)
    PREDICTIONS_CSV.write_text(ledger_text(ledger), encoding="utf-8")
    report = performance_report(ledger)
    if not report.empty:
        report.to_csv("reports/performance.csv", index=False)
    session_date = max(df.index.max() for df in raw.values()).strftime("%Y-%m-%d")
    print(f"Market session: {session_date}")
    print(f"Predictions evaluated: {evaluated}")
    print(f"Model retrained: {'YES' if retrained else 'NO'}")
    if champion_mae is not None:
        print(f"Champion validation MAE: {champion_mae:.6f}")
        print(f"Challenger validation MAE: {challenger_mae:.6f}")
        print(f"Model decision: {'REPLACE CHAMPION' if retrained else 'KEEP CHAMPION'}")
    if not report.empty:
        print(report.to_string(index=False))
    send_evening(session_date, evaluated, ledger, report, retrained, champion_mae, challenger_mae)

if __name__ == "__main__":
    run()
