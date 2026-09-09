import pandas as pd


def test_portfolio_output_schema_includes_all_horizons():
    from src.portfolio_report import OUTPUT_COLUMNS, HORIZONS
    for horizon in HORIZONS:
        assert f"Horizon_{horizon}D" in OUTPUT_COLUMNS


def test_plan_row_preserves_horizon_predictions():
    from src.portfolio_report import _plan_row
    pred = {"Current_Price": 100.0, "Pred_Close": 105.0, "Confidence": 80.0}
    pred.update({f"Horizon_{h}D": float(h) for h in (1, 3, 5, 7, 10, 20, 60, 90, 180, 365)})
    row = {"Ticker": "TEST.NS", "Quantity": 1, "Average_Price": 100.0, "Reported_PnL": 0.0, "Reported_Return": 0.0}
    # Avoid market-data access in the unit test.
    import src.portfolio_report as module
    original = module._latest_price
    module._latest_price = lambda ticker, cutoff=None: (100.0, "TEST")
    try:
        out = _plan_row(row, pred, "2026-09-09")
    finally:
        module._latest_price = original
    for h in (1, 3, 5, 7, 10, 20, 60, 90, 180, 365):
        assert out[f"Horizon_{h}D"] == float(h)
