import pandas as pd

from src.report_integrity import validate_prediction_frame, validate_portfolio_frame, validate_telegram_messages


def test_prediction_schema_and_ohlc_invariants():
    df = pd.DataFrame([{"Symbol":"TEST","Pred_Open":100,"Pred_High":110,"Pred_Low":95,"Pred_Close":105}])
    assert validate_prediction_frame(df, expected_n=10)[0]


def test_prediction_duplicate_is_rejected():
    df = pd.DataFrame([
        {"Symbol":"TEST","Pred_Open":100,"Pred_High":110,"Pred_Low":95,"Pred_Close":105},
        {"Symbol":"TEST","Pred_Open":100,"Pred_High":110,"Pred_Low":95,"Pred_Close":105},
    ])
    ok, errors = validate_prediction_frame(df, expected_n=10)
    assert not ok and "DUPLICATE_SYMBOLS" in errors


def test_portfolio_duplicate_is_rejected():
    df = pd.DataFrame([{"Ticker":"TEST.NS","Quantity":1,"Average_Price":100,"Current_Price":105,"Decision":"HOLD"},{"Ticker":"TEST.NS","Quantity":2,"Average_Price":100,"Current_Price":105,"Decision":"HOLD"}])
    ok, errors = validate_portfolio_frame(df)
    assert not ok and "DUPLICATE_TICKER" in errors


def test_telegram_length_gate():
    assert validate_telegram_messages(["ok"], 10)[0]
    assert not validate_telegram_messages(["12345678901"], 10)[0]
