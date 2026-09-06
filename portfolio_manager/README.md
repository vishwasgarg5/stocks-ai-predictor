# Portfolio Manager — standalone

This folder is a **separate decision-support module**. It does not modify or feed portfolio data into the core stock-prediction model.

## Input

`portfolio_manager/data/my_portfolio.csv` accepts:

```text
index,Stock,Quantity,Current_PnL_INR,Return_Percent
```

If `Average_Price` is supplied, it is used directly. If it is missing, the manager estimates purchase average from the current market price plus supplied P&L/return and marks it as estimated.

## Daily decision engine

```text
Portfolio CSV
     ↓
Current Market Price
     ↓
Purchase Average
     ↓
Latest validated AI prediction + 1/3/5/7/20D horizons
     ↓
Recovery + downside analysis
     ↓
SELL / HOLD / AVG / DO NOT AVG
```

### SELL
- Profit-book when the configured **10% target** is reached.
- Detects when the AI target no longer supports a safe recovery.
- Provides a sell window based on the first forecast horizon expected to reach the 10% target.

### HOLD
- Used when recovery remains supported but a sell/average trigger is not justified.
- Deep-loss positions are kept in recovery watch rather than automatically sold.

### AVG
- Only when the stock is sufficiently below the current average, AI recovery supports the target, and calibrated confidence is at least 60%.
- Calculates additional quantity, maximum averaging capital and the resulting new average.
- Averaging capital is capped at **25% of existing position cost**.

### DO NOT AVG
- Used when the AI target cannot support a safe recovery, confidence is too low, or the position is not meaningfully below average.

## Sell timing

The manager does not invent an exact sell date. It evaluates the available 1D/3D/5D/7D/20D forecasts and reports either:

- `NOW`
- a forecast sell window
- `NOT REACHED IN 20D`
- `NO AI TARGET`

This makes the timing probability-based rather than pretending the model can know the exact future market date.

## Telegram report

The morning and evening reports now include:

- Daily action for every portfolio holding
- CMP and average price
- 10% profit target
- Sell window
- SELL / profit-book alerts
- AVG recommendations with additional quantity
- New average after averaging
- Reason for the decision

Portfolio data remains completely separate from model training. No profit is guaranteed, and estimated purchase averages should be verified against the broker statement.
