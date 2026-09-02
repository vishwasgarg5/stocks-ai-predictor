import pandas as pd
from .config import STOCK_RELIABILITY_FILE
from .utils import clamp

def load_reliability():
    if not STOCK_RELIABILITY_FILE.exists(): return {}
    try:
        df=pd.read_csv(STOCK_RELIABILITY_FILE); out={}
        for _,r in df.iterrows():
            mape=float(r.get("MAPE",3) or 3); direction=float(r.get("DirectionAccuracy",50) or 50)
            out[str(r["Symbol"])] = .55*clamp(100-mape*20)+.45*direction
        return out
    except Exception:return {}

def expected_return_score(v): return clamp(50+float(v)*5)
def regime_direction_score(d,regime):
    if regime=="BULL": return {"UP":90,"NEUTRAL":55,"DOWN":35}.get(d,50)
    if regime=="BEAR": return {"UP":35,"NEUTRAL":55,"DOWN":80}.get(d,50)
    if regime=="HIGH VOL": return 45
    return {"UP":70,"NEUTRAL":55,"DOWN":45}.get(d,50)
def calculate_score(row,regime):
    return clamp(.25*float(row.get("TechnicalScore",50))+.20*expected_return_score(row.get("Expected_Return",0))+.20*float(row.get("Confidence",50))+.15*float(row.get("Direction_Confidence",50))+.10*float(row.get("ReliabilityScore",50))+.10*regime_direction_score(row.get("Direction","NEUTRAL"),regime))
def select_top_stocks(candidates,top_n=5,regime="SIDEWAYS"):
    if candidates is None or candidates.empty:return pd.DataFrame()
    df=candidates.copy(); df["ReliabilityScore"]=df["Symbol"].map(load_reliability()).fillna(50.0); df["Score"]=df.apply(lambda r:calculate_score(r,regime),axis=1)
    return df.sort_values(["Score","Confidence","Direction_Confidence"],ascending=False).head(top_n).reset_index(drop=True)
