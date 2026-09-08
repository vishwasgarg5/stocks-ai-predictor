import inspect

from src import portfolio_report


def test_portfolio_uses_canonical_snapshot_and_no_direct_yfinance():
    source = inspect.getsource(portfolio_report)
    assert "get_snapshot" in source
    assert "yfinance" not in source
    assert "allow_download=False" in source


def test_portfolio_prediction_selection_uses_lineage_date_not_mtime():
    source = inspect.getsource(portfolio_report._latest_predictions)
    assert "PredictionDate" in source
    assert "Run_Date" in source
    assert "Cutoff_Date" in source
    assert "st_mtime" not in source


def test_decision_never_returns_blank():
    cases = [
        (None, None, None, 0, []),
        (100, 90, float("nan"), 0, []),
        (100, 90, 80, 50, []),
    ]
    for current, avg, target, confidence, forecasts in cases:
        decision, reason = portfolio_report._decision(current, avg, target, confidence, forecasts)
        assert decision
        assert reason
