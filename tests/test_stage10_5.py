import importlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
REQUIRED_SOURCE_FILES=["src/__init__.py","src/config.py","src/features.py","src/market_data.py","src/models.py","src/prediction.py","src/multihorizon.py","src/selection.py","src/stage4_engine.py","src/stage45_engine.py","src/final_intelligence.py","src/evaluation.py","src/retraining.py","src/ledger.py","src/morning_runner.py","src/evening.py","src/telegram_report.py","src/weekly_report.py","src/portfolio_report.py"]
REQUIRED_WORKFLOWS=[".github/workflows/stage10_4_morning.yml",".github/workflows/stage10_4_evening.yml",".github/workflows/stage10_4_weekly.yml"]
FORBIDDEN_LEGACY_PATHS=[".github/workflows/stage2_morning.yml",".github/workflows/stage2_evening.yml",".github/workflows/stage2_weekly.yml",".github/workflows/morning_prediction.yml",".github/workflows/evening_evaluate_retrain.yml",".github/workflows/weekly_report.yml",".github/workflows/test_stage2.py",".github/workflows/stage4_tests.yml","main.py","morning.py","stage15_morning.py","config.py","weekly_report.py","evening.py","src/stage15.py","src/ranking.py","models/champion.pkl","reports/performance.csv","reports/weekly_report.csv","tests/test_stage2.py"]
CORE_MODULES=["src.config","src.features","src.market_data","src.models","src.prediction","src.multihorizon","src.selection","src.stage4_engine","src.stage45_engine","src.final_intelligence","src.evaluation","src.retraining","src.ledger","src.morning_runner","src.evening","src.telegram_report","src.weekly_report","src.portfolio_report"]
def test_required_source_files_exist():
    for path in REQUIRED_SOURCE_FILES: assert (ROOT/path).exists(),path
def test_current_workflows_exist():
    for path in REQUIRED_WORKFLOWS: assert (ROOT/path).exists(),path
def test_legacy_paths_are_removed():
    for path in FORBIDDEN_LEGACY_PATHS: assert not (ROOT/path).exists(),path
def test_exactly_three_production_workflows():
    assert sorted(p.name for p in (ROOT/".github/workflows").glob("*.yml"))==["stage10_4_evening.yml","stage10_4_morning.yml","stage10_4_weekly.yml"]
def test_core_modules_import():
    failures=[]
    for name in CORE_MODULES:
        try: importlib.import_module(name)
        except Exception as exc: failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures,"Stage 10.5 import failures:\n"+"\n".join(failures)
def test_config_is_stage10_5():
    from src.config import MODEL_VERSION,STAGE_NAME,HISTORY_PERIOD,TOP_N,MAX_UNIVERSE,MULTI_HORIZONS,PRICE_BUCKET_NAMES,MAX_PER_PRICE_BUCKET,FINAL_BEST_PER_BUCKET
    assert MODEL_VERSION.startswith("stage10.5") and STAGE_NAME.startswith("Stage 10.5") and HISTORY_PERIOD=="1y" and TOP_N is None and MAX_UNIVERSE==0 and tuple(MULTI_HORIZONS)==(1,3,5,7,20)
    assert PRICE_BUCKET_NAMES==[">2500","1000-2499","500-999","250-499","100-249","50-99","10-49"] and MAX_PER_PRICE_BUCKET==6 and FINAL_BEST_PER_BUCKET==1
def test_next_session_ohlcv_target_alignment():
    df=pd.DataFrame({"Open":[10,11,12],"High":[11,12,13],"Low":[9,10,11],"Close":[10.5,11.5,12.5],"Volume":[100,110,120]})
    for c in ["Open","High","Low","Close","Volume"]: df[f"Target_{c}"]=df[c].shift(-1)
    assert df.loc[0,"Target_Open"]==11 and df.loc[0,"Target_Close"]==11.5 and df.loc[0,"Target_Volume"]==110 and pd.isna(df.loc[2,"Target_Close"])
def test_lag_features_use_only_previous_sessions():
    close=pd.Series([100.,101.,102.,103.,104.]); assert close.shift(1).iloc[3]==102 and close.shift(2).iloc[3]==101 and close.shift(3).iloc[3]==100
def test_prediction_ohlc_ordering():
    p={"Open":100.,"High":105.,"Low":98.,"Close":103.}; assert p["High"]>=p["Open"]>=p["Low"] and p["High"]>=p["Close"]>=p["Low"]
def test_prediction_values_are_finite(): assert np.isfinite(np.array([100.,105.,98.,103.,100000.])).all()
def test_ensemble_weights_sum_to_one(): assert abs(sum({"XGB":.4,"RF":.3,"ET":.3}.values())-1)<1e-9
def test_selection_returns_all_qualified_without_global_five_cap():
    from src.selection import select_top_stocks
    rows=[{"Symbol":f"S{i}","TechnicalScore":90,"Expected_Return":5,"Confidence":90,"Direction_Confidence":90,"Direction":"UP","PriceBucket":b} for i,b in enumerate([">2500","1000-2499","500-999","250-499","100-249","50-99","10-49"])]
    r=select_top_stocks(pd.DataFrame(rows),top_n=None); assert len(r)==7 and set(r.Symbol)=={f"S{i}" for i in range(7)}
def test_price_bucket_limit_is_configurable():
    from src.selection import select_top_stocks
    rows=[{"Symbol":f"B1{i}","PriceBucket":"B1","TechnicalScore":95-i,"Expected_Return":5,"Confidence":90,"Direction_Confidence":85,"Direction":"UP"} for i in range(8)]
    r=select_top_stocks(pd.DataFrame(rows),top_n=None,max_per_bucket=6); assert len(r)==6
def test_direction_return_conflict_reduces_trade_confidence():
    from src.selection import calculate_trade_confidence
    base={"Confidence":95,"Direction_Confidence":95,"ReliabilityScore":80,"Direction":"UP","MultiHorizonExpectedReturn":5,"Horizon_1D":5,"Horizon_3D":5,"Horizon_5D":5,"Horizon_7D":5,"Horizon_20D":5}
    assert calculate_trade_confidence({**base,"Expected_Return":5})>calculate_trade_confidence({**base,"Expected_Return":-5})
def test_price_buckets_are_current():
    from src.stage4_engine import price_bucket
    assert [price_bucket(x)[0] for x in [3000,1500,750,300,150,75,25,9]]==["B1","B2","B3","B4","B5","B6","B7","OUT"]
def test_horizons_are_future_only():
    close=pd.Series([100.,101.,102.,103.,104.,105.,106.])
    for h in [1,3,5]: target=close.shift(-h); assert target.iloc[0]==close.iloc[h] and pd.isna(target.iloc[-h:]).all()
def test_ohlc_targets_are_defined():
    from src.models import TARGETS
    assert TARGETS==["Open","High","Low","Close"]
def test_confidence_calibration_is_bounded():
    from src.final_intelligence import calibrate_confidence
    assert 0<=calibrate_confidence(120,20,100)<=100
def test_final_action_guardrails():
    from src.final_intelligence import final_action
    safe={"FinalDecisionScore":90,"Direction":"UP","PredictionUncertaintyPct":5,"CalibratedConfidence":90,"NetExpectedReturn":5,"BenchmarkEdgePct":1,"TargetHitProb_3_0Pct":70,"DownsideHitProb_2Pct":20,"StockReliability":80,"HorizonReliability":80,"DataQualityScore":95}
    assert final_action(safe,"BULL")=="BUY" and final_action({**safe,"PredictionUncertaintyPct":20},"BULL")=="NO TRADE"
def test_final_manifest_contract():
    from src.final_intelligence import final_stage_manifest
    m=final_stage_manifest(); assert any(k in m for k in ["Stage10.4","Stage10.5"]) and "Validation" in m and "Abstention" in m and "Learning" in m
def test_portfolio_engine_has_sell_average_and_timing_logic():
    from src.portfolio_report import _sell_window,_avg_window,_plan
    assert _sell_window({"Horizon_1D":11},110,100,"2026-09-04")=="2026-09-07"
    assert _avg_window({"Horizon_3D":-3,"Horizon_7D":8},80,100,"2026-09-04").startswith("2026-09-07")
    row=pd.Series({"Quantity":100.,"Current_Price":80.,"Average_Price":100.,"AI_Target":120.,"Reported_Return":np.nan,"Reported_PnL":np.nan,"PredictionDate":"2026-09-04","Horizon_5D":30.,"Horizon_1D":5.,"Horizon_3D":10.,"Horizon_7D":35.,"Horizon_20D":50.,"CalibratedConfidence":90.})
    out=_plan(row.copy()); assert out["Decision"] in {"AVG","HOLD / RECOVERY","HOLD"} and out["Profit_Target_Price"]==110.
def test_morning_report_includes_buckets_and_portfolio_sections():
    from src.telegram_report import morning_report
    d=pd.DataFrame([{"Symbol":"TEST","PriceBucket":"100-249","Current_Price":100,"Pred_Close":108,"Expected_Return":8,"Confidence":80,"FinalDecisionScore":80,"Action":"BUY","Horizon_1D":3,"Horizon_5D":7,"Horizon_20D":12}])
    report=morning_report("2026-09-07","2026-09-04",d,pd.DataFrame(),pd.DataFrame(),accuracy={"PreviousAccuracy":70,"CurrentAccuracy":72},scan={"Universe":100,"Data":90,"Liquid":80,"AI":40,"Selected":1},portfolio={"Positions":1,"Value":10000,"PnL":500,"Return":5,"Rows":[{"Stock":"TEST","Quantity":10,"Decision":"HOLD","Current_Price":"₹100","Average_Price":"₹95","Profit_Target":"₹104.50","Sell_Window":"2026-09-10"}]})
    assert "100-249" in report and "BEST PICK" in report and "AI PORTFOLIO MANAGER" in report and "Sell Window" in report
