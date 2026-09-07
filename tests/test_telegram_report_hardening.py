import pandas as pd
from src.telegram_report import morning_report


def test_morning_report_has_long_horizons_ipo_and_no_signal():
    selected = pd.DataFrame([{
        "Symbol": "TEST", "PriceBucket": "B5", "PriceBucketLabel": "100-249",
        "Current_Price": 150, "Pred_Open": 151, "Pred_High": 156,
        "Pred_Low": 149, "Pred_Close": 155, "Expected_Return": 3.33,
        "Horizon_1D": 1, "Horizon_5D": 4, "Horizon_20D": 8,
        "Horizon_60D": 18, "Horizon_90D": 21, "Horizon_180D": 19,
        "Horizon_365D": 20, "FinalDecisionScore": 90, "Action": "BUY",
    }])
    ipo = pd.DataFrame([{
        "IPOName": "TEST IPO", "Status": "OPEN", "PriceHigh": 100,
        "GMPValue": 20, "GMPPct": 20, "IPOAction": "BUY",
    }])
    report = morning_report(
        "2026-09-07", "2026-09-04", selected, pd.DataFrame(), pd.DataFrame(),
        accuracy={"PreviousAccuracy": 95, "CurrentAccuracy": 96, "Samples": 10},
        scan={"Universe": 100, "Data": 90, "Liquid": 80, "AI": 20, "Selected": 1},
        ipo=ipo,
    )
    assert "60D" in report and "90D" in report and "180D" in report and "365D" in report
    assert "IPO INTELLIGENCE" in report and "TEST IPO" in report
    assert "CONFIRMED" in report


def test_empty_morning_report_is_not_filled_with_fake_picks():
    report = morning_report("2026-09-07", "2026-09-04", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ipo=pd.DataFrame())
    assert "No high-confidence setup today." in report
    assert "No active/upcoming IPOs." in report
