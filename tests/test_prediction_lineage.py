import pandas as pd
import pytest
from src.ledger import _prediction_id, prediction_path, latest_prediction_date, save_predictions, save_evaluation


def _predictions():
    return pd.DataFrame([
        {"Symbol":"AAA","Pred_Open":101,"Pred_High":105,"Pred_Low":99,"Pred_Close":103,"Previous_Close":100},
        {"Symbol":"BBB","Pred_Open":201,"Pred_High":205,"Pred_Low":199,"Pred_Close":203,"Previous_Close":200},
    ])


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
    date="2026-09-07"; meta={"PredictionDate":date,"DataCutoff":"2026-09-04","ModelVersion":"v1"}
    path=prediction_path(date)
    original=_predictions().iloc[[0]].copy();save_predictions(original,date,meta);before=path.read_text()
    replacement=original.copy();replacement.loc[replacement.index[0],"Pred_Close"]=999
    save_predictions(replacement,date,meta)
    assert path.read_text() == before


def test_evaluation_binds_to_canonical_prediction_ids(tmp_path, monkeypatch):
    pred_dir=tmp_path/"predictions";eval_dir=tmp_path/"evaluations"
    monkeypatch.setattr("src.ledger.PREDICTIONS_DIR", pred_dir);monkeypatch.setattr("src.ledger.EVALUATIONS_DIR", eval_dir)
    date="2026-09-08";save_predictions(_predictions(),date,{"PredictionDate":date,"DataCutoff":"2026-09-07","ModelVersion":"v1"})
    evaluation=pd.DataFrame([
        {"MarketDate":date,"PredictionDate":date,"Symbol":"AAA","Pred_Open":101,"Actual_Open":102},
        {"MarketDate":date,"PredictionDate":date,"Symbol":"BBB","Pred_Open":201,"Actual_Open":202},
    ])
    out=pd.read_csv(save_evaluation(evaluation,date));canonical=pd.read_csv(pred_dir/f"predictions_{date}.csv")
    assert set(out["Prediction_ID"]) == set(canonical["Prediction_ID"])
    assert out["Prediction_ID"].is_unique


def test_evaluation_rejects_unknown_symbol(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ledger.PREDICTIONS_DIR", tmp_path/"predictions");monkeypatch.setattr("src.ledger.EVALUATIONS_DIR", tmp_path/"evaluations")
    date="2026-09-08";save_predictions(_predictions(),date,{"PredictionDate":date,"DataCutoff":"2026-09-07","ModelVersion":"v1"})
    bad=pd.DataFrame([{"MarketDate":date,"PredictionDate":date,"Symbol":"ZZZ"}])
    with pytest.raises(ValueError):save_evaluation(bad,date)


def test_prediction_ledger_rejects_invalid_ohlc(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ledger.PREDICTIONS_DIR", tmp_path)
    bad=_predictions().iloc[[0]].copy();bad.loc[bad.index[0],"Pred_Low"]=200
    with pytest.raises(ValueError):save_predictions(bad,"2026-09-08",{"PredictionDate":"2026-09-08","DataCutoff":"2026-09-07","ModelVersion":"v1"})
