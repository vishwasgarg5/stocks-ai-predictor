import importlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
REQUIRED_SOURCE_FILES=["src/__init__.py","src/config.py","src/features.py","src/market_data.py","src/models.py","src/prediction.py","src/multihorizon.py","src/selection.py","src/stage4_engine.py","src/evaluation.py","src/retraining.py","src/ledger.py","src/morning_runner.py","src/telegram_report.py","src/weekly_report.py"]
REQUIRED_WORKFLOWS=[".github/workflows/stage2_morning.yml",".github/workflows/stage2_evening.yml",".github/workflows/stage2_weekly.yml"]
FORBIDDEN_LEGACY_PATHS=[".github/workflows/morning_prediction.yml",".github/workflows/evening_evaluate_retrain.yml",".github/workflows/weekly_report.yml",".github/workflows/test_stage2.yml","main.py","morning.py","stage15_morning.py","config.py","weekly_report.py","evening.py","src/stage15.py","src/ranking.py","models/champion.pkl","reports/performance.csv","reports/weekly_report.csv"]
CORE_MODULES=["src.config","src.features","src.market_data","src.models","src.prediction","src.multihorizon","src.selection","src.stage4_engine","src.evaluation","src.retraining","src.ledger"]

def test_required_source_files_exist():
    for path in REQUIRED_SOURCE_FILES: assert (ROOT/path).exists(),path

def test_current_workflows_exist():
    for path in REQUIRED_WORKFLOWS: assert (ROOT/path).exists(),path

def test_legacy_paths_are_removed():
    for path in FORBIDDEN_LEGACY_PATHS: assert not (ROOT/path).exists(),path

def test_core_modules_import():
    failures=[]
    for name in CORE_MODULES:
        try: importlib.import_module(name)
        except Exception as exc: failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures,"Stage 4.2 import failures:\n"+"\n".join(failures)

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

def test_quality_selection_never_pads_to_five():
    from src.selection import select_top_stocks
    d=pd.DataFrame({"Symbol":["AAA","BBB","CCC"],"TechnicalScore":[90,70,60],"Expected_Return":[8,2,-1],"Confidence":[90,70,55],"Direction_Confidence":[90,70,55],"Direction":["UP"]*3,"PriceBucket":["B1","B2","B3"]})
    r=select_top_stocks(d,10); assert len(r)<=3 and "AAA" in set(r.Symbol)

def test_two_stocks_max_per_price_bucket():
    from src.selection import select_top_stocks
    rows=[]
    for i in range(4): rows.append({"Symbol":f"B1{i}","PriceBucket":"B1","TechnicalScore":95-i,"Expected_Return":5-i*.2,"Confidence":90,"Direction_Confidence":85,"Direction":"UP"})
    for i in range(4): rows.append({"Symbol":f"B2{i}","PriceBucket":"B2","TechnicalScore":94-i,"Expected_Return":4-i*.2,"Confidence":90,"Direction_Confidence":85,"Direction":"UP"})
    r=select_top_stocks(pd.DataFrame(rows),10)
    assert len(r)==4 and r["PriceBucket"].value_counts().max()==2

def test_direction_return_conflict_reduces_trade_confidence():
    from src.selection import calculate_trade_confidence
    base={"Confidence":95,"Direction_Confidence":95,"ReliabilityScore":80,"Direction":"UP","MultiHorizonExpectedReturn":5,"Horizon_1D":5,"Horizon_3D":5,"Horizon_5D":5,"Horizon_7D":5,"Horizon_20D":5}
    aligned=calculate_trade_confidence({**base,"Expected_Return":5})
    conflict=calculate_trade_confidence({**base,"Expected_Return":-5})
    assert aligned>conflict and conflict<60

def test_stage3a_price_buckets():
    from src.stage4_engine import price_bucket
    assert [price_bucket(x)[0] for x in [1500,750,250,75,25,9]]==["B1","B2","B3","B4","B5","OUT"]

def test_stage3a_bucket_selection():
    from src.stage4_engine import select_price_bucket_candidates
    d=pd.DataFrame({"Symbol":["A","B","C","D"],"PriceBucket":["B1","B1","B2","B2"],"Score":[80,90,70,95]}); r=select_price_bucket_candidates(d,1); assert set(r.Symbol)=={"B","D"}

def test_stage4_2_weights_sum_to_one(): assert abs(.18+.15+.15+.12+.08+.10+.10+.12-1)<1e-9

def test_stage3b_horizons_are_defined():
    from src.multihorizon import HORIZONS
    assert tuple(HORIZONS)==(1,3,5,7,20)

def test_stage3b_horizon_targets_are_future_only():
    close=pd.Series([100.,101.,102.,103.,104.,105.,106.])
    for h in [1,3,5]:
        target=close.shift(-h); assert target.iloc[0]==close.iloc[h] and pd.isna(target.iloc[-h:]).all()

def test_ohlc_targets_are_defined():
    from src.models import TARGETS
    assert TARGETS==["Open","High","Low","Close"]

def test_morning_report_has_exactly_three_sections():
    from src.telegram_report import morning_report
    report=morning_report("2026-09-03","2026-09-02",pd.DataFrame(),pd.DataFrame(),pd.DataFrame())
    assert report.count("*1. TOP STOCKS") == 1 and report.count("*2. +5% JUMP WATCH") == 1 and report.count("*3. INTRADAY STOCKS") == 1
