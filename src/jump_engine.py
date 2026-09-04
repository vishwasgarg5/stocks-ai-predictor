import numpy as np
import pandas as pd

from .config import JUMP_THRESHOLD,JUMP_HORIZON_DAYS,JUMP_TOP_N,JUMP_CANDIDATE_N,MIN_JUMP_PROBABILITY,MIN_JUMP_7D_UPSIDE
from .features import technical_score
from .prediction import train_stock_bundle,predict_stock
from .multihorizon import train_horizon_models,predict_horizons


def calculate_jump_score(current_price,predicted_close,predicted_high,confidence,technical,upside_7d=0.0):
    close_return=predicted_close/current_price-1; high_return=predicted_high/current_price-1
    target_score=np.clip(high_return/max(JUMP_THRESHOLD,0.01)*100,0,100)
    close_score=np.clip(close_return/max(JUMP_THRESHOLD,0.01)*100,0,100)
    horizon_score=np.clip(upside_7d/max(JUMP_THRESHOLD*100,1.0)*100,0,100)
    return float(np.clip(0.25*target_score+0.15*close_score+0.20*horizon_score+0.20*confidence+0.20*technical,0,100))


def generate_jump_watchlist(data_map,cutoff_date,variant="A"):
    candidates=[]
    for symbol,df in data_map.items():
        try:
            df=df[df.index<=pd.Timestamp(cutoff_date)]
            if len(df)<150:continue
            tech=technical_score(df)
            if tech<50:continue
            bundle=train_stock_bundle(df,symbol,cutoff_date,variant,train_horizons=False)
            result=predict_stock(df,bundle,cutoff_date)
            current=float(result.get("Current_Price",df["Close"].iloc[-1]))
            predictions={t:float(result[f"Pred_{t}"]) for t in ["Open","High","Low","Close","Volume"]}
            hb=train_horizon_models(df,cutoff_date);horizons=predict_horizons(df,hb,cutoff_date)
            h7=horizons[horizons["HorizonDays"]==JUMP_HORIZON_DAYS]
            true_7d_return=float(h7.iloc[0]["Expected_Return"]) if not h7.empty else 0.0
            confidence=float(bundle.get("direction_validation_accuracy",50))
            expected_high=predictions["High"]/current-1;expected_close=predictions["Close"]/current-1
            max_potential=max(expected_high,true_7d_return/100)
            probability=float(np.clip(50+max_potential*250+(confidence-50)*0.35,0,95))
            score=calculate_jump_score(current,predictions["Close"],predictions["High"],probability,tech,true_7d_return)
            # A jump candidate needs either a meaningful 1D high move or a meaningful 7D upside.
            if probability<MIN_JUMP_PROBABILITY:continue
            if max(expected_high*100,true_7d_return)<3.0:continue
            candidates.append({"Symbol":symbol,"Current_Price":current,"Predicted_Close_1D":predictions["Close"],"Predicted_High_1D":predictions["High"],"Expected_1D_Return":expected_close*100,"Estimated_7D_Upside":true_7d_return,"Jump_Probability":probability,"Confidence":confidence,"TechnicalScore":tech,"JumpScore":score,"Target_Level":current*(1+JUMP_THRESHOLD),"Status":"OPEN","Remaining_Days":JUMP_HORIZON_DAYS})
        except Exception as exc:print(f"{symbol}: jump prediction failed: {exc}")
    if not candidates:return pd.DataFrame()
    return pd.DataFrame(candidates).sort_values(["JumpScore","Jump_Probability","Estimated_7D_Upside"],ascending=False).head(JUMP_TOP_N).reset_index(drop=True)


def evaluate_jump_prediction(prediction_row,actual_history):
    symbol=prediction_row["Symbol"];current_price=float(prediction_row["Current_Price"]);target=float(prediction_row["Target_Level"]);start_date=pd.Timestamp(prediction_row["Prediction_Date"])
    history=actual_history.copy();history=history[history.index>start_date].head(JUMP_HORIZON_DAYS)
    if history.empty:return {"Symbol":symbol,"Hit":False,"DaysToHit":None,"MaxUpside":None,"ObservationDays":0}
    max_high=float(history["High"].max());hit_rows=history[history["High"]>=target]
    if not hit_rows.empty:
        first_hit=hit_rows.index[0];days=list(history.index).index(first_hit)+1
        return {"Symbol":symbol,"Hit":True,"DaysToHit":days,"MaxUpside":(max_high/current_price-1)*100,"ObservationDays":len(history)}
    return {"Symbol":symbol,"Hit":False,"DaysToHit":None,"MaxUpside":(max_high/current_price-1)*100,"ObservationDays":len(history)}
