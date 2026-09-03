"""IPO intelligence: issue/GMP/fundamentals/subscription/risk based decision support.
GMP is treated as an unregulated sentiment input, never as a guaranteed listing price.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

IPO_FIELDS=["IPOName","Symbol","OpenDate","CloseDate","ListingDate","PriceLow","PriceHigh","LotSize","GMP","GMPUpdatedAt","QIBSubscription","NIISubscription","RetailSubscription","EmployeeSubscription","RevenueGrowth","ProfitGrowth","ROE","DebtToEquity","PE","IndustryPE","PromoterHolding","FreshIssuePct","OFS_Pct"]

def gmp_metrics(price_high,gmp):
    price=max(float(price_high),1); premium=float(gmp or 0); pct=premium/price*100
    return premium,pct,price+premium

def ipo_score(row):
    price=float(row.get("PriceHigh",0) or 0); gmp=float(row.get("GMP",0) or 0)
    _,gmp_pct,_=gmp_metrics(price,gmp)
    score=50+np.clip(gmp_pct, -30, 50)*0.45
    score+=np.clip(float(row.get("RevenueGrowth",0) or 0),-30,50)*0.20
    score+=np.clip(float(row.get("ProfitGrowth",0) or 0),-30,50)*0.20
    roe=float(row.get("ROE",0) or 0); score+=np.clip(roe,0,30)*0.25
    debt=float(row.get("DebtToEquity",1) or 1); score-=np.clip(debt-1,0,4)*5
    pe=float(row.get("PE",0) or 0); ipe=float(row.get("IndustryPE",0) or 0)
    if pe>0 and ipe>0: score+=np.clip((ipe-pe)/ipe*20,-20,20)
    # Subscription is useful confirmation, but not allowed to dominate.
    q=float(row.get("QIBSubscription",0) or 0); n=float(row.get("NIISubscription",0) or 0); r=float(row.get("RetailSubscription",0) or 0)
    score+=np.clip(np.log1p(q)*2+np.log1p(n)*1.5+np.log1p(r)*1, -10,15)
    return float(np.clip(score,0,100))

def ipo_decision(row):
    s=ipo_score(row); gmp=float(row.get("GMP",0) or 0); price=float(row.get("PriceHigh",0) or 0)
    gmp_pct=(gmp/max(price,1))*100
    # Never issue BUY solely from GMP.
    if s>=75 and gmp_pct>=8: return "BUY"
    if s>=65: return "CONSIDER"
    if s>=50: return "WATCH"
    return "AVOID"

def analyze_ipos(ipos):
    df=ipos.copy() if isinstance(ipos,pd.DataFrame) else pd.DataFrame(ipos)
    if df.empty: return df
    for c in ["PriceHigh","GMP","RevenueGrowth","ProfitGrowth","ROE","DebtToEquity","PE","IndustryPE","QIBSubscription","NIISubscription","RetailSubscription"]:
        if c not in df: df[c]=0
    gm=df.apply(lambda r:gmp_metrics(r["PriceHigh"],r["GMP"]),axis=1,result_type="expand"); gm.columns=["GMPValue","GMPPct","ImpliedListingPrice"]
    df=pd.concat([df,gm],axis=1); df["IPOScore"]=df.apply(ipo_score,axis=1); df["IPOAction"]=df.apply(ipo_decision,axis=1)
    df["GMPWarning"]=np.where(df["GMPPct"]<0,"NEGATIVE GMP",np.where(df["GMPPct"]<5,"LOW GMP","POSITIVE GMP"))
    return df.sort_values(["IPOScore","GMPPct"],ascending=False).reset_index(drop=True)

def ipo_report(ipos):
    df=analyze_ipos(ipos)
    cols=[c for c in ["IPOName","OpenDate","CloseDate","PriceHigh","GMPValue","GMPPct","ImpliedListingPrice","QIBSubscription","NIISubscription","RetailSubscription","IPOScore","IPOAction","GMPWarning"] if c in df]
    return df[cols] if not df.empty else df
