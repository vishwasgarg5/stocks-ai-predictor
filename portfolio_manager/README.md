# Portfolio Manager — standalone

This folder is a **separate decision-support module**. It does not modify or feed portfolio data into the core stock-prediction model.

## Input

`portfolio_manager/data/my_portfolio.csv` accepts the current screenshot format:

```text
index,Stock,Quantity,Current_PnL_INR,Return_Percent
```

If `Average_Price` is supplied, it is used directly. If it is missing, the manager estimates the purchase average from the current market price plus the supplied P&L/return and marks the source as estimated.

## Decision engine

```text
Portfolio CSV
     ↓
Current Market Price
     ↓
Purchase Average
     ↓
Latest AI Prediction
     ↓
AI Target Return
     ↓
Recovery Gap
     ↓
Can averaging reach ≥5% profit at AI target?
     ↓
AVERAGE / HOLD / DO NOT AVERAGE
```

For an averaging candidate it calculates recommended additional quantity, new average price, projected return at the AI target and maximum averaging capital. Averaging is capped at **25% of the existing position cost** so the manager does not recommend unlimited averaging.

It also reports multi-horizon outlook when prediction data provides it, recovery status and profit-booking/recovery actions.

No profit is guaranteed. Estimated purchase averages must be treated as estimates until verified against the broker statement. fileciteturn100file0L2-L5
