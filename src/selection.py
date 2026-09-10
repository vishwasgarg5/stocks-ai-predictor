from .config import STOCK_RELIABILITY_FILE,MAX_PREDICTION_UNCERTAINTY,TOP_N,MAX_PER_PRICE_BUCKET,MIN_NET_RETURN_PCT
import pandas as pd
from .utils import clamp
BUCKET_SELECTION_SIZE=5

def load_reliability():
    if not STOCK_RELIABILITY_FILE.exists(): return {}
    try:
        df=pd.read_csv(STOCK_RELIABILITY_FILE); out={}
        for _,r in df.iterrows():
            mape=float(r.get('RecentMAPE',r.get('MAPE',3)) or 3); direction=float(r.get('DirectionAccuracy',50) or 50); samples=float(r.get('Samples',0) or 0); evidence=min(samples/20.0,1.0); raw=.55*clamp(100-mape*20)+.45*direction; out[str(r['Symbol'])]=50+evidence*(raw-50)
        return out
    except Exception:return {}

def expected_return_score(v): return clamp(50+float(v)*5)
def multi_horizon_score(v):
    try:return clamp(50+float(v)*4)
    except Exception:return 50.
def regime_direction_score(d,regime):
    if regime=='BULL': return {'UP':90,'NEUTRAL':55,'DOWN':35}.get(d,50)
    if regime=='BEAR': return {'UP':35,'NEUTRAL':55,'DOWN':80}.get(d,50)
    if regime=='HIGH VOL': return 45
    return {'UP':70,'NEUTRAL':55,'DOWN':45}.get(d,50)
def direction_return_alignment(direction,expected_return):
    try:r=float(expected_return)
    except Exception:return 50.
    d=str(direction).upper()
    if d=='UP':return clamp(50+r*25)
    if d=='DOWN':return clamp(50-r*25)
    return clamp(100-abs(r)*20)
def horizon_alignment(row):
    values=[]
    for h in (1,3,5,7,10,20,60,90,180,365):
        for key in (f'H{h}D_Return',f'Horizon_{h}D',f'Expected_Return_{h}D'):
            if key in row and pd.notna(row[key]):
                try:values.append(float(row[key]));break
                except (TypeError,ValueError):pass
    if not values:return 50.
    d=str(row.get('Direction','NEUTRAL')).upper(); agree=sum(v>0 for v in values) if d=='UP' else sum(v<0 for v in values) if d=='DOWN' else sum(abs(v)<=1.5 for v in values)
    return 100*agree/len(values)
def calculate_trade_confidence(row):
    return clamp(.27*float(row.get('Confidence',50))+.13*float(row.get('Direction_Confidence',50))+.35*direction_return_alignment(row.get('Direction','NEUTRAL'),row.get('Expected_Return',0))+.05*float(row.get('ReliabilityScore',50))+.10*horizon_alignment(row)+.10*float(row.get('UncertaintyScore',50)))
def calculate_score(row,regime):
    return clamp(.16*float(row.get('TechnicalScore',50))+.14*expected_return_score(row.get('Expected_Return',0))+.14*float(row.get('Confidence',50))+.11*float(row.get('Direction_Confidence',50))+.07*float(row.get('ReliabilityScore',50))+.09*regime_direction_score(row.get('Direction','NEUTRAL'),regime)+.09*float(row.get('SectorScore',50))+.10*multi_horizon_score(row.get('MultiHorizonExpectedReturn',0))+.10*float(row.get('UncertaintyScore',50)))
def _strict_trade_eligible(df):
    if df.empty:return df
    out=df.copy(); exp=pd.to_numeric(out.get('Expected_Return',0),errors='coerce').fillna(-999); mh=pd.to_numeric(out.get('MultiHorizonExpectedReturn',exp),errors='coerce').fillna(-999); align=pd.to_numeric(out.get('DirectionReturnAlignment',0),errors='coerce').fillna(0); direction=out.get('Direction',pd.Series('NEUTRAL',index=out.index)).astype(str).str.upper()
    return out[(direction=='UP')&(exp>=float(MIN_NET_RETURN_PCT))&(mh>0)&(align>=60)].copy()
def _apply_uncertainty_cap(df):
    if 'PredictionUncertaintyPct' not in df.columns:return df
    return df[pd.to_numeric(df['PredictionUncertaintyPct'],errors='coerce')<=MAX_PREDICTION_UNCERTAINTY].copy()
def _rank(g):
    return g.sort_values(['TradeConfidence','Score','Confidence','Direction_Confidence','Symbol'],ascending=[False,False,False,False,True],kind='mergesort')
def _bucket_cap(df,limit=BUCKET_SELECTION_SIZE):
    if df.empty or 'PriceBucket' not in df.columns:return df
    return pd.concat([_rank(g).head(limit) for _,g in df.groupby('PriceBucket',sort=True)],ignore_index=True) if len(df) else df

def score_candidates(candidates,regime='SIDEWAYS'):
    if candidates is None or candidates.empty:return pd.DataFrame()
    df=candidates.copy()
    if 'MultiHorizonExpectedReturn' not in df.columns:df['MultiHorizonExpectedReturn']=pd.to_numeric(df.get('Expected_Return',0),errors='coerce').fillna(0.)
    for c,v in [('SectorScore',50.),('UncertaintyScore',50.)]:
        if c not in df.columns:df[c]=v
    rel=load_reliability(); df['ReliabilityScore']=df['Symbol'].map(rel).fillna(50.); df['TradeConfidence']=df.apply(calculate_trade_confidence,axis=1); df['TradeQuality']=df['TradeConfidence'].map(lambda x:'HIGH' if x>=75 else 'MEDIUM' if x>=60 else 'LOW'); df['DirectionReturnAlignment']=df.apply(lambda r:direction_return_alignment(r.get('Direction','NEUTRAL'),r.get('Expected_Return',0)),axis=1); df['Score']=df.apply(lambda r:calculate_score(r,regime),axis=1)
    return _rank(df).reset_index(drop=True)

def select_prediction_set(candidates,top_n=TOP_N,max_per_bucket=MAX_PER_PRICE_BUCKET,regime='SIDEWAYS'):
    scored=score_candidates(candidates,regime)
    if scored.empty:return scored
    pool=scored.copy()
    for c in ('Pred_Open','Pred_High','Pred_Low','Pred_Close','Current_Price'):
        if c in pool.columns:pool=pool[pd.to_numeric(pool[c],errors='coerce').notna()]
    pool=pool[pool['PriceBucket'].astype(str).ne('OUT')]
    if pool.empty:return pool
    preferred=_apply_uncertainty_cap(pool)
    if preferred.empty:preferred=pool
    bucketed=_bucket_cap(preferred,BUCKET_SELECTION_SIZE)
    fallback=_bucket_cap(pool,BUCKET_SELECTION_SIZE)
    bucketed=pd.concat([bucketed,fallback],ignore_index=True).drop_duplicates('Symbol')
    bucket_count=pool['PriceBucket'].nunique(); coverage_target=BUCKET_SELECTION_SIZE*bucket_count
    requested=int(top_n) if top_n is not None and int(top_n)>0 else 0
    target=max(requested,coverage_target)
    if len(bucketed)<target:
        extra=_rank(pool[~pool['Symbol'].isin(bucketed['Symbol'])]); bucketed=pd.concat([bucketed,extra],ignore_index=True)
    return bucketed.drop_duplicates('Symbol').head(min(target,len(bucketed))).reset_index(drop=True)

def select_top_stocks(candidates,top_n=TOP_N,regime='SIDEWAYS',min_score=65.,min_confidence=60.,min_trade_confidence=60.,max_per_bucket=MAX_PER_PRICE_BUCKET,bucket_only=False):
    scored=score_candidates(candidates,regime)
    if scored.empty:return scored
    base=scored[(scored['Score']>=min_score)&(scored['Confidence']>=min_confidence)&(scored['TradeConfidence']>=min_trade_confidence)&(scored['DirectionReturnAlignment']>=60)].copy(); strict=_apply_uncertainty_cap(_strict_trade_eligible(base)); fallback=False
    if strict.empty:strict=_apply_uncertainty_cap(base[base['Direction'].astype(str).str.upper().eq('UP')])
    if strict.empty:strict=base[base['Direction'].astype(str).str.upper().eq('UP')].copy()
    if strict.empty:strict=_apply_uncertainty_cap(scored.copy());fallback=True
    if strict.empty:strict=scored.copy();fallback=True
    strict['SelectionTier']='PREDICTION_ONLY' if fallback else 'RECOMMENDED'; capped=_bucket_cap(strict,BUCKET_SELECTION_SIZE); n=int(top_n) if top_n is not None else 0; target=max(n,BUCKET_SELECTION_SIZE*capped['PriceBucket'].nunique()) if 'PriceBucket' in capped.columns else n
    if len(capped)<target:
        pool=_bucket_cap(_apply_uncertainty_cap(scored.copy()),BUCKET_SELECTION_SIZE)
        if len(pool)<target:pool=_bucket_cap(scored.copy(),BUCKET_SELECTION_SIZE)
        capped=pd.concat([capped,pool[~pool['Symbol'].isin(capped['Symbol'])]],ignore_index=True).drop_duplicates('Symbol')
    return capped.head(min(target,len(capped))).reset_index(drop=True)
