import pandas as pd
from src.quality_controls import confirm_portfolio_target, evaluate_baseline


def test_multi_horizon_target_requires_confirmation():
    strong=confirm_portfolio_target({60:18,90:21,180:19,365:20})
    assert strong["Status"] == "MULTI_HORIZON_CONFIRMED"
    assert strong["TargetPct"] >= 18

    weak=confirm_portfolio_target({60:21,90:7,180:4,365:3})
    assert weak["Status"] == "SINGLE_HORIZON_OR_INCONSISTENT"
    assert weak["TargetPct"] == 10


def test_baseline_beats_only_when_ai_has_lower_error_after_cost():
    evaluation=pd.DataFrame([{
        "Prediction_ID":"P1","MarketDate":"2026-09-07","PredictionDate":"2026-09-04",
        "Symbol":"TEST","Actual_Close":110,"APE_Close":1.0
    }])
    predictions=pd.DataFrame([{
        "Prediction_ID":"P1","Baseline_Close":100,"Baseline_Cost_Pct":0.30
    }])
    out=evaluate_baseline(evaluation,predictions)
    assert bool(out.iloc[0]["AI_Better_Than_Baseline"])
