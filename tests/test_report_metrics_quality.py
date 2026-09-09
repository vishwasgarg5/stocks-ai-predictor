import pandas as pd

import src.report_metrics as rm


def test_confidence_calibration_detects_overconfidence():
    df = pd.DataFrame({
        "PredictionConfidence": [90, 90, 90, 90],
        "APE_Close": [20, 20, 20, 20],
    })
    out = rm._confidence_calibration(df)
    assert out["ConfidenceCalibration"] == "LOW EVIDENCE"
    assert out["ConfidenceECE"] > 10


def test_drift_detects_recent_error_degradation():
    df = pd.DataFrame({"APE_Close": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]})
    out = rm._drift_metrics(df)
    assert out["RecentMAPE"] == 2.0
    assert out["BaselineMAPE"] == 1.0
    assert out["Drift"] == "HIGH"


def test_horizon_metrics_includes_365d(tmp_path, monkeypatch):
    monkeypatch.setattr(rm, "EVALUATIONS_DIR", tmp_path)
    pd.DataFrame([
        {"HorizonDays": 365, "APE": 5.0, "DirectionCorrect": True},
        {"HorizonDays": 365, "APE": 10.0, "DirectionCorrect": False},
    ]).to_csv(tmp_path / "horizon_evaluations.csv", index=False)
    out = rm._horizon_metrics()
    assert out["365D"] == 92.5
    assert out["365DSamples"] == 2
    assert out["365DDirectionAccuracy"] == 50.0


def test_model_report_exposes_horizon_and_calibration(monkeypatch, tmp_path):
    monkeypatch.setattr(rm, "EVALUATIONS_DIR", tmp_path)
    pd.DataFrame([
        {"Symbol": "AAA", "APE_Close": 2.0, "PredictionConfidence": 80, "DirectionCorrect": True, "PriceBucket": "100-249"},
        {"Symbol": "BBB", "APE_Close": 4.0, "PredictionConfidence": 60, "DirectionCorrect": False, "PriceBucket": "50-99"},
        {"Symbol": "CCC", "APE_Close": 3.0, "PredictionConfidence": 70, "DirectionCorrect": True, "PriceBucket": "100-249"},
    ]).to_csv(tmp_path / "evaluation_2026-09-08.csv", index=False)
    pd.DataFrame([
        {"HorizonDays": 365, "APE": 5.0, "DirectionCorrect": True},
    ]).to_csv(tmp_path / "horizon_evaluations.csv", index=False)
    out = rm.model_report_metrics()
    assert out["CurrentAccuracy"] == 97.0
    assert out["AccuracySamples"] == 3
    assert out["365D"] == 95.0
    assert "ConfidenceECE" in out
    assert "Drift" in out
    assert out["WorstStock"] in {"AAA", "BBB", "CCC"}
