import pandas as pd

from src.final_intelligence import apply_final_intelligence, update_learning_state
from src.telegram_report import morning_report, evening_report


def _candidate_frame():
    return pd.DataFrame([
        {
            "Symbol": "TEST",
            "PriceBucket": "B3",
            "Current_Price": 250.0,
            "Pred_Close": 265.0,
            "Pred_Open": 252.0,
            "Pred_High": 268.0,
            "Pred_Low": 248.0,
            "Confidence": 85.0,
            "ReliabilityScore": 85.0,
            "ReliabilitySamples": 60,
            "Direction": "UP",
            "Expected_Return": 6.0,
            "MultiHorizonExpectedReturn": 5.0,
            "TechnicalScore": 85.0,
            "RiskAdjustedScore": 85.0,
            "Score": 85.0,
            "PredictionUncertaintyPct": 4.0,
        }
    ])


def test_stage10_candidate_to_morning_report_integration():
    candidates = apply_final_intelligence(_candidate_frame(), regime="BULL", breadth=70, news=65)
    assert len(candidates) == 1
    row = candidates.iloc[0]
    assert row["Action"] == "BUY"
    assert 0 <= row["FinalDecisionScore"] <= 100
    assert row["BearCase"] <= row["BaseCase"] <= row["BullCase"]

    report = morning_report(
        "2026-09-03",
        "2026-09-02",
        candidates,
        pd.DataFrame(),
        pd.DataFrame(),
        market_snapshot={"NIFTY": {}, "BANKNIFTY": {}, "VIX": {}, "Breadth": {}},
        regime="BULL",
        accuracy={"PreviousAccuracy": 70, "CurrentAccuracy": 72, "AccuracySamples": 20},
        scan={"Universe": 100, "Data": 95, "Liquid": 90, "AI": 20, "Selected": 1},
    )
    assert "TOP 5 AI STOCKS" in report
    assert "Price Bucket" in report
    assert "TEST" in report
    assert "2. +5% JUMP WATCH" in report
    assert "3. INTRADAY STOCKS" in report


def test_stage10_learning_state_persists(tmp_path):
    path = tmp_path / "learning.json"
    state = update_learning_state(path, {"date": "2026-09-03", "selected": [{"Symbol": "TEST"}]})
    assert path.exists()
    assert state["observations"][-1]["date"] == "2026-09-03"


def test_stage10_evening_report_integration():
    evaluation = pd.DataFrame([
        {
            "Symbol": "TEST",
            "Pred_Open": 252.0,
            "Pred_High": 268.0,
            "Pred_Low": 248.0,
            "Pred_Close": 265.0,
            "Actual_Open": 253.0,
            "Actual_High": 267.0,
            "Actual_Low": 249.0,
            "Actual_Close": 264.0,
            "Diff_Open": 1.0,
            "Diff_High": -1.0,
            "Diff_Low": 1.0,
            "Diff_Close": -1.0,
            "Pred_Direction": "UP",
            "Actual_Direction": "UP",
            "DirectionCorrect": True,
        }
    ])
    report = evening_report(
        "2026-09-03",
        evaluation,
        {"Samples": 1, "OverallMAPE": 0.5, "CloseMAPE": 0.4, "DirectionAccuracy": 100},
        {"Retrained": False, "Decision": "KEEP CHAMPION", "Improvement": 0},
        bucket_metrics={"B3": 99.6},
        learning={"status": "UPDATED"},
        accuracy={"PreviousAccuracy": 70, "CurrentAccuracy": 72, "AccuracySamples": 1},
        scan={"Universe": 1, "Data": 1, "Liquid": 1, "AI": 1, "Selected": 1},
    )
    assert "PREDICTION vs ACTUAL" in report
    assert "TEST" in report
    assert "MODEL LEARNING" in report
    assert "Current Accuracy" in report
