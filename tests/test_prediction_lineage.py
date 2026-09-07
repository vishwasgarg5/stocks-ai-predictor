import pandas as pd
from src.ledger import _prediction_id, prediction_path, latest_prediction_date, save_predictions


def test_prediction_id_is_deterministic():
    assert _prediction_id("2026-09-07", "RELIANCE", "2026-09-04", "v1") == _prediction_id("2026-09-07", "RELIANCE", "2026-09-04", "v1")
    assert _prediction_id("2026-09-07", "RELIANCE", "2026-09-04", "v1") != _prediction_id("2026-09-07", "TCS", "2026-09-04", "v1")


def test_latest_prediction_date_with_bound_is_exact(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ledger.PREDICTIONS_DIR", tmp_path)
    pd.DataFrame([{"Symbol":"TEST","Prediction_ID":"P1"}]).to_csv(tmp_path / "predictions_2026-09-04.csv", index=False)
    assert latest_prediction_date("2026-09-07") is None
    assert latest_prediction_date("2026-09-04") == pd.Timestamp("2026-09-04").date()


def test_save_predictions_does_not_overwrite_existing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ledger.PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr("src.ledger.morning_report_path", lambda d: tmp_path / f"morning_report_{d}.json")
    date="2026-09-07"
    path=prediction_path(date)
    original=pd.DataFrame([{"Symbol":"TEST","Current_Close":100,"Pred_Close":105}])
    save_predictions(original,date,{"PredictionDate":date,"DataCutoff":"2026-09-04","ModelVersion":"v1"})
    before=path.read_text()
    replacement=pd.DataFrame([{"Symbol":"TEST","Current_Close":100,"Pred_Close":999}])
    save_predictions(replacement,date,{"PredictionDate":date,"DataCutoff":"2026-09-04","ModelVersion":"v1"})
    assert path.read_text() == before
