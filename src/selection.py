"""Stage 28 precision selection with strict eligibility, bucket caps and deterministic fallback."""
import pandas as pd
from .config import STOCK_RELIABILITY_FILE,MAX_PREDICTION_UNCERTAINTY,TOP_N,MAX_PER_PRICE_BUCKET,MIN_NET_RETURN_PCT
from .utils import clamp

def load_reliability():
    if not STOCK_RELIABILITY_FILE.exists(): return {}
    try:
        df=pd.read_csv(STOCK_RELIABILITY_FILE);out={}
        for _,r in df.iterrows():
            mape=float(r.get("MAPE",3) or 3);direction=float(r.get("DirectionAccuracy",50) or 50);samples=float(r.get("Samples",0) or 0);evidence=min(samples/20.0,1.0);raw=0.55*clamp(100-mape*20)+0.45*direction;out[str(r["Symbol"])]=50+evidence*(raw-50)
        return out
    except Exception:return {}
def expected_return_score(v):return clamp(50+float(v)*5)
def multi_horizon_score(v):
    try:return clamp(50+float(v)*4)
    except Exception:return 50.0
def regime_direction_score(d,regime):
    if regime=="BULL":return {"UP":90,"NEUTRAL":55,"DOWN":35}.get(d,50)
    if regime=="BEAR":return {"UP":35,"NEUTRAL":55,"DOWN":80}.get(d,50)
    if regime=="HIGH VOL":return 45
    return {"UP":70,"NEUTRAL":55,"DOWN":45}.get(d,50)
def direction_return_alignment(direction,expected_return):
    try:r=float(expected_return)
    except Exception:return 50.0
    d=str(direction).upper()
    if d=="UP":return clamp(50+r*25)
    if d=="DOWN":return clamp(50-r*25)
    return clamp(100-abs(r)*20)
def horizon_alignment(row):
    values=[]
    for h in (1,3,5,7,10,20,60,90,180,365):
        for key in (f"H{h}D_Return",f"Horizon_{h}D",f"Expected_Return_{h}D"):
            if key in row and pd.notna(row[key]):
                try:values.append(float(row[key]));break
                except (TypeError,ValueError):pass
    if not values:return 50.0
    direction=str(row.get("Direction","NEUTRAL")).upper();agreeing=sum(v>0 for v in values) if direction=="UP" else sum(v<0 for v in values) if direction=="DOWN" else sum(abs(v)<=1.5 for v in values)
    return 100.0*agreeing/len(values)
def calculate_trade_confidence(row):return clamp(0.27*float(row.get("Confidence",50))+0.13*float(row.get("Direction_Confidence",50))+0.35*direction_return_alignment(row.get("Direction","NEUTRAL"),row.get("Expected_Return",0))+0.05*float(row.get("ReliabilityScore",50))+0.10*horizon_alignment(row)+0.10*float(row.get("UncertaintyScore",50)))
def calculate_score(row,regime):return clamp(0.16*float(row.get("TechnicalScore",50))+0.14*expected_return_score(row.get("Expected_Return",0))+0.14*float(row.get("Confidence",50))+0.11*float(row.get("Direction_Confidence",50))+0.07*float(row.get("ReliabilityScore",50))+0.09*regime_direction_score(row.get("Direction","NEUTRAL"),regime)+0.09*float(row.get("SectorScore",50))+0.10*multi_horizon_score(row.get("MultiHorizonExpectedReturn",0))+0.10*float(row.get("UncertaintyScore",50)))
def _strict_trade_eligible(df):
    if df.empty:return df
    out=df.copy();exp=pd.to_numeric(out.get("Expected_Return",0),errors="coerce").fillna(-999);mh=pd.to_numeric(out.get("MultiHorizonExpectedReturn",exp),errors="coerce").fillna(-999);align=pd.to_numeric(out.get("DirectionReturnAlignment",0),errors="coerce").fillna(0);direction=out.get("Direction",pd.Series("NEUTRAL",index=out.index)).astype(str).str.upper()
    return out[(direction=="UP")&(exp>=float(MIN_NET_RETURN_PCT))&(mh>0)&(align>=60)].copy()
def _apply_uncertainty_cap(df):
    if "PredictionUncertaintyPct" not in df.columns:return df
    return df[pd.to_numeric(df["PredictionUncertaintyPct"],errors="coerce")<=MAX_PREDICTION_UNCERTAINTY].copy()
def _bucket_cap(df,max_per_bucket):
    if df.empty:return df
    pieces=[]
    for bucket,group in df.groupby("PriceBucket",sort=False):
        g=group.sort_values(["TradeConfidence","Score","Confidence","Direction_Confidence"],ascending=False)
        limit=len(g) if max_per_bucket is None or max_per_bucket<=0 else min(len(g),int(max_per_bucket));pieces.append(g.head(limit))
    return pd.concat(pieces,ignore_index=True) if pieces else df.iloc[0:0]
def score_candidates(candidates,regime="SIDEWAYS"):
    if candidates is None or candidates.empty:return pd.DataFrame()
    df=candidates.copy()
    if "MultiHorizonExpectedReturn" not in df.columns:df["MultiHorizonExpectedReturn"]=pd.to_numeric(df.get("Expected_Return",0),errors="coerce").fillna(0.0)
    for c,v in [("SectorScore",50.0),("UncertaintyScore",50.0)]:
        if c not in df.columns:df[c]=v
    reliability=load_reliability();df["ReliabilityScore"]=df["Symbol"].map(reliability).fillna(50.0);df["TradeConfidence"]=df.apply(calculate_trade_confidence,axis=1);df["TradeQuality"]=df["TradeConfidence"].map(lambda x:"HIGH" if x>=75 else "MEDIUM" if x>=60 else "LOW");df["DirectionReturnAlignment"]=df.apply(lambda r:direction_return_alignment(r.get("Direction","NEUTRAL"),r.get("Expected_Return",0)),axis=1);df["Score"]=df.apply(lambda r:calculate_score(r,regime),axis=1)
    return df.sort_values(["TradeConfidence","Score","Confidence","Direction_Confidence","SectorScore"],ascending=False).reset_index(drop=True)
def select_top_stocks(candidates,top_n=TOP_N,regime="SIDEWAYS",min_score=65.0,min_confidence=60.0,min_trade_confidence=60.0,max_per_bucket=MAX_PER_PRICE_BUCKET,bucket_only=False):
    scored=score_candidates(candidates,regime)
    if scored.empty:return scored
    base=scored[(scored["Score"]>=min_score)&(scored["Confidence"]>=min_confidence)&(scored["TradeConfidence"]>=min_trade_confidence)&(scored["DirectionReturnAlignment"]>=60.0)].copy()
    strict=_apply_uncertainty_cap(_strict_trade_eligible(base))
    if strict.empty:strict=_apply_uncertainty_cap(base[base["Direction"].astype(str).str.upper().eq("UP")])
    if strict.empty:strict=base[base["Direction"].astype(str).str.upper().eq("UP")].copy()
    capped=_bucket_cap(strict,max_per_bucket)
    if bucket_only:return capped.sort_values(["PriceBucket","TradeConfidence","Score"],ascending=[True,False,False]).reset_index(drop=True)
    n=None if top_n is None else int(top_n)
    if n is None or n<=0:return capped.sort_values(["PriceBucket","TradeConfidence","Score"],ascending=[True,False,False]).reset_index(drop=True)
    seeded=[]
    for _,group in capped.groupby("PriceBucket",sort=False):
        if not group.empty:seeded.append(group.iloc[0])
    chosen=pd.DataFrame(seeded).drop_duplicates("Symbol") if seeded else capped.iloc[0:0]
    if len(chosen)<n:
        remaining=capped[~capped["Symbol"].isin(chosen["Symbol"])].sort_values(["TradeConfidence","Score","Confidence","Direction_Confidence"],ascending=False)
        chosen=pd.concat([chosen,remaining.head(n-len(chosen))],ignore_index=True)
    return chosen.drop_duplicates("Symbol").head(n).sort_values(["PriceBucket","TradeConfidence","Score"],ascending=[True,False,False]).reset_index(drop=True)
