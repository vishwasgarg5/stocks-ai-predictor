# AI NSE Stock Predictor — Stage 2

Stage 2 expands the existing stock prediction system into a
multi-horizon NSE intelligence engine.

## Stage 2 engines

### 1. Next-Day Prediction

Scans the expanded NSE universe and selects the best five stocks.

Predicts:

- Open
- High
- Low
- Close
- Direction
- Confidence
- Technical score
- Market regime

### 2. 7-Day +5% Jump Engine

Every morning the system searches for five stocks that have a
reasonable probability of reaching at least +5% within the next
seven trading sessions.

It stores:

- Current price
- +5% target
- Estimated upside
- Jump probability
- Confidence
- Technical setup

The prediction remains open for seven sessions.

The system records:

- Target hit
- Days to target
- Maximum upside
- Failed prediction

This creates a dedicated learning dataset.

### 3. Intraday Engine

The system scans liquid NSE stocks using intraday data.

Signals include:

- VWAP
- EMA 9
- EMA 20
- Relative volume
- Momentum
- Intraday range

The output contains:

- Bias
- Current price
- Target
- Stop-loss
- Confidence
- Score

## Model architecture

Stage 2 uses an ensemble:

- XGBoost
- Random Forest
- Extra Trees

The ensemble weights are learned from chronological
validation performance.

## Champion / Challenger

Two model configurations are maintained:

- Variant A
- Variant B

The active configuration is Champion.

The other configuration is Challenger.

The evening engine compares them using the same historical
training/validation process.

The Challenger must improve validation error by at least 2%
before becoming Champion.

This prevents tiny meaningless improvements from replacing
the current model.

## Important prediction rule

Features from trading session T predict the OHLC of session T+1.

This prevents same-day target leakage.

## Data storage

No local database is used.

Stage 2 stores persistent data in:

data/stage2/

including:

- predictions
- evaluations
- jump predictions
- intraday predictions
- metrics
- model state

All of this can be committed back to GitHub.

## Morning

The morning workflow:

1. Loads broad NSE universe.
2. Applies liquidity filtering.
3. Calculates technical indicators.
4. Prescreens candidates.
5. Runs Stage 2 ensemble.
6. Selects next-day Top 5.
7. Runs 7-day +5% engine.
8. Runs intraday engine.
9. Saves all results.
10. Sends Telegram report.

## Evening

The evening workflow:

1. Loads the exact morning prediction ledger.
2. Does NOT select stocks again.
3. Fetches actual OHLC.
4. Calculates prediction errors.
5. Calculates direction accuracy.
6. Updates cumulative metrics.
7. Updates stock reliability.
8. Runs Champion/Challenger comparison.
9. Promotes Challenger only when meaningfully better.
10. Saves all learning data.

## Telegram secrets

Create:

TELEGRAM_BOT_TOKEN

TELEGRAM_CHAT_ID

under:

Repository → Settings → Secrets and variables → Actions

## Running manually

Morning:

python -m src.morning_runner

Evening:

python -m src.evening

Weekly:

python -m src.weekly_report

## Stage 2 philosophy

The system does not automatically assume that a more complicated
model is better.

It keeps measuring:

- MAPE
- normalized validation error
- direction accuracy
- 7-day +5% hit rate
- intraday success rate

and only promotes a challenger when the evidence shows a
meaningful improvement.
