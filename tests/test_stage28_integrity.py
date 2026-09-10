import numpy as np
import pandas as pd

def test_selection_never_exceeds_bucket_cap():
    from src.selection import select_top_stocks
    rows=[]
    for i in range(12):rows.append({"Symbol":f"AAA{i}","PriceBucket":"100-249","TechnicalScore":90-i,"Expected_Return":4.0,"Confidence":90,"Direction_Confidence":90,"Direction":"UP","MultiHorizonExpectedReturn":4.0,"UncertaintyScore":90})
    for i in range(6):rows.append({"Symbol":f"BBB{i}","PriceBucket":"500-999","TechnicalScore":85-i,"Expected_Return":3.0,"Confidence":90,"Direction_Confidence":90,"Direction":"UP","MultiHorizonExpectedReturn":3.0,"UncertaintyScore":90})
    out=select_top_stocks(pd.DataFrame(rows),top_n=10,max_per_bucket=6,min_score=0,min_confidence=0,min_trade_confidence=0);assert len(out)<=10;assert out.groupby("PriceBucket").size().max()<=6

def test_selection_is_deterministic_for_equal_inputs():
    from src.selection import select_top_stocks
    rows=[{"Symbol":f"S{i}","PriceBucket":"100-249","TechnicalScore":80-i,"Expected_Return":2,"Confidence":80,"Direction_Confidence":80,"Direction":"UP","MultiHorizonExpectedReturn":2,"UncertaintyScore":80} for i in range(8)]
    assert select_top_stocks(pd.DataFrame(rows),top_n=6,max_per_bucket=6,min_score=0,min_confidence=0,min_trade_confidence=0)["Symbol"].tolist()==select_top_stocks(pd.DataFrame(rows),top_n=6,max_per_bucket=6,min_score=0,min_confidence=0,min_trade_confidence=0)["Symbol"].tolist()

def test_prediction_id_is_deterministic():
    from src.ledger import _prediction_id
    assert _prediction_id("2026-09-08","RELIANCE","2026-09-07","stage28-v1.0")==_prediction_id("2026-09-08","RELIANCE","2026-09-07","stage28-v1.0")
    assert _prediction_id("2026-09-08","RELIANCE","2026-09-07","stage28-v1.0")!=_prediction_id("2026-09-08","RELIANCE","2026-09-06","stage28-v1.0")

def test_horizon_configuration_contains_full_year():
    from src.config import MULTI_HORIZONS
    assert tuple(MULTI_HORIZONS)==(1,3,5,7,10,20,60,90,180,365)

def test_cutoff_supervised_rows_are_strictly_before_cutoff():
    from src.features import prepare_supervised
    idx=pd.date_range('2026-01-01',periods=220,freq='D');df=pd.DataFrame({'Open':100+np.arange(220)*.1,'High':101+np.arange(220)*.1,'Low':99+np.arange(220)*.1,'Close':100+np.arange(220)*.1,'Volume':100000},index=idx);cutoff=idx[-20];sup=prepare_supervised(df,cutoff);assert not sup.empty and sup.index.max()<cutoff

def test_report_never_renders_nan():
    from src.telegram_report import morning_report
    out=morning_report('2026-09-10','2026-09-09',pd.DataFrame([{'Symbol':'AAA','PriceBucket':'B0','Current_Price':5,'Pred_Close':5.2,'Horizon_1D':np.nan,'Horizon_5D':np.nan}]),pd.DataFrame(),pd.DataFrame(),accuracy={},scan={},portfolio={});assert 'nan' not in out.lower()
