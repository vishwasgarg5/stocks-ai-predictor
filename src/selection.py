import pandas as pd
from .config import STOCK_RELIABILITY_FILE,MAX_PREDICTION_UNCERTAINTY,TOP_N,MAX_PER_PRICE_BUCKET,MIN_NET_RETURN_PCT
from .utils import clamp
BUCKET_SELECTION_SIZE=5

def load_reliability():
 if not STOCK_RELIABILITY_FILE.exists():return {}
 try:
  d=pd.read_csv(STOCK_RELIABILITY_FILE);o={}
  for _,r in d.iterrows():
   m=float(r.get('RecentMAPE',r.get('MAPE',3)) or 3);a=float(r.get('DirectionAccuracy',50) or 50);n=float(r.get('Samples',0) or 0);o[str(r['Symbol'])]=50+min(n/20.,1.)*(.55*clamp(100-m*20)+.45*a-50)
  return o
 except Exception:return {}
def expected_return_score(v):return clamp(50+float(v)*5)
def multi_horizon_score(v):
 try:return clamp(50+float(v)*4)
 except Exception:return 50.
def regime_direction_score(d,regime):
 if regime=='BULL':return {'UP':90,'NEUTRAL':55,'DOWN':35}.get(d,50)
 if regime=='BEAR':return {'UP':35,'NEUTRAL':55,'DOWN':80}.get(d,50)
 if regime=='HIGH VOL':return 45
 return {'UP':70,'NEUTRAL':55,'DOWN':45}.get(d,50)
def direction_return_alignment(direction,expected_return):
 try:r=float(expected_return)
 except Exception:return 50.
 d=str(direction).upper();return clamp(50+r*25) if d=='UP' else clamp(50-r*25) if d=='DOWN' else clamp(100-abs(r)*20)
def horizon_alignment(row):
 v=[]
 for h in (1,3,5,7,10,20,60,90,180,365):
  for k in (f'H{h}D_Return',f'Horizon_{h}D',f'Expected_Return_{h}D'):
   if k in row and pd.notna(row[k]):
    try:v.append(float(row[k]));break
    except:pass
 if not v:return 50.
 d=str(row.get('Direction','NEUTRAL')).upper();a=sum(x>0 for x in v) if d=='UP' else sum(x<0 for x in v) if d=='DOWN' else sum(abs(x)<=1.5 for x in v);return 100*a/len(v)
def calculate_trade_confidence(r):
 alignment=direction_return_alignment(r.get('Direction','NEUTRAL'),r.get('Expected_Return',0))
 return clamp(.25*float(r.get('Confidence',50))+.15*float(r.get('Direction_Confidence',50))+.20*float(r.get('ReliabilityScore',50))+.20*horizon_alignment(r)+.10*float(r.get('UncertaintyScore',50))+.10*alignment)
def calculate_score(r,regime):return clamp(.20*float(r.get('TechnicalScore',50))+.10*expected_return_score(r.get('Expected_Return',0))+.15*float(r.get('Confidence',50))+.10*float(r.get('Direction_Confidence',50))+.10*float(r.get('ReliabilityScore',50))+.10*regime_direction_score(r.get('Direction','NEUTRAL'),regime)+.10*float(r.get('SectorScore',50))+.10*multi_horizon_score(r.get('MultiHorizonExpectedReturn',0))+.05*float(r.get('UncertaintyScore',50)))
def _strict_trade_eligible(d):
 if d.empty:return d
 e=pd.to_numeric(d.get('Expected_Return',0),errors='coerce').fillna(-999);m=pd.to_numeric(d.get('MultiHorizonExpectedReturn',e),errors='coerce').fillna(-999);a=pd.to_numeric(d.get('DirectionReturnAlignment',0),errors='coerce').fillna(0);x=d.get('Direction',pd.Series('NEUTRAL',index=d.index)).astype(str).str.upper();return d[(x=='UP')&(e>=float(MIN_NET_RETURN_PCT))&(m>0)&(a>=60)].copy()
def _apply_uncertainty_cap(d):
 if 'PredictionUncertaintyPct' not in d.columns:return d
 return d[pd.to_numeric(d['PredictionUncertaintyPct'],errors='coerce')<=MAX_PREDICTION_UNCERTAINTY].copy()
def _rank(d):return d.sort_values(['TradeConfidence','Score','Confidence','Direction_Confidence','Symbol'],ascending=[False,False,False,False,True],kind='mergesort')
def _bucket_cap(d,n):
 if d.empty or 'PriceBucket' not in d.columns:return d
 return pd.concat([_rank(g).head(int(n)) for _,g in d.groupby('PriceBucket',sort=True)],ignore_index=True)
def score_candidates(candidates,regime='SIDEWAYS'):
 if candidates is None or candidates.empty:return pd.DataFrame()
 d=candidates.copy()
 if 'MultiHorizonExpectedReturn' not in d.columns:d['MultiHorizonExpectedReturn']=pd.to_numeric(d.get('Expected_Return',0),errors='coerce').fillna(0.)
 for c,v in [('SectorScore',50.),('UncertaintyScore',50.)]:
  if c not in d.columns:d[c]=v
 rel=load_reliability();d['ReliabilityScore']=d['Symbol'].map(rel).fillna(50.);d['DirectionReturnAlignment']=d.apply(lambda r:direction_return_alignment(r.get('Direction','NEUTRAL'),r.get('Expected_Return',0)),axis=1);d['TradeConfidence']=d.apply(calculate_trade_confidence,axis=1);d['TradeQuality']=d['TradeConfidence'].map(lambda x:'HIGH' if x>=75 else 'MEDIUM' if x>=60 else 'LOW');d['Score']=d.apply(lambda r:calculate_score(r,regime),axis=1);return _rank(d).reset_index(drop=True)
def select_prediction_set(candidates,top_n=TOP_N,max_per_bucket=MAX_PER_PRICE_BUCKET,regime='SIDEWAYS'):
 s=score_candidates(candidates,regime)
 if s.empty:return s
 p=s[s['PriceBucket'].astype(str).ne('OUT')].copy()
 for c in ('Pred_Open','Pred_High','Pred_Low','Pred_Close','Current_Price'):
  if c in p.columns:p=p[pd.to_numeric(p[c],errors='coerce').notna()]
 if p.empty:return p
 q=_apply_uncertainty_cap(p);q=q if not q.empty else p
 n=max(1,int(top_n) if top_n and int(top_n)>0 else TOP_N);bucket_count=p['PriceBucket'].nunique();minimum=min(n,bucket_count);b=_bucket_cap(q,1).head(minimum);remaining=_rank(q[~q['Symbol'].isin(b['Symbol'])]);out=pd.concat([b,remaining],ignore_index=True).drop_duplicates('Symbol').head(n)
 return out.reset_index(drop=True)
def select_top_stocks(candidates,top_n=TOP_N,regime='SIDEWAYS',min_score=65.,min_confidence=60.,min_trade_confidence=60.,max_per_bucket=MAX_PER_PRICE_BUCKET,bucket_only=False):
 s=score_candidates(candidates,regime)
 if s.empty:return s
 b=s[(s['Score']>=min_score)&(s['Confidence']>=min_confidence)&(s['TradeConfidence']>=min_trade_confidence)&(s['DirectionReturnAlignment']>=60)].copy();x=_apply_uncertainty_cap(_strict_trade_eligible(b))
 if x.empty:x=_apply_uncertainty_cap(b[b['Direction'].astype(str).str.upper().eq('UP')])
 if x.empty:x=b[b['Direction'].astype(str).str.upper().eq('UP')].copy()
 if x.empty:return s.iloc[0:0].copy()
 x['SelectionTier']='RECOMMENDED';x=_bucket_cap(x,max_per_bucket);n=None if top_n is None else int(top_n)
 if n is None or n<=0:return x.reset_index(drop=True)
 return x.head(n).reset_index(drop=True)
