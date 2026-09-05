# AI NSE Stock Predictor — Stage 10.2

Production-oriented NSE stock prediction and decision-support system covering the cumulative **Stage 1 → Stage 10.2** pipeline.

> Predictions are model outputs, not guaranteed returns or investment advice.

## Production flow

```text
NSE UNIVERSE
 ↓ DATA QUALITY + LIQUIDITY
 ↓ CAUSAL TECHNICAL FEATURES + LEAKAGE AUDIT
 ↓ AI OHLCV PREDICTION
 ↓ PRICE BUCKETS
 ↓ 1D / 3D / 5D / 7D / 20D
 ↓ MARKET / SECTOR / RISK / UNCERTAINTY
 ↓ CONFIDENCE CALIBRATION
 ↓ BUY / WATCH / HOLD / AVOID / NO TRADE
 ↓ GITHUB PREDICTION + DECISION LEDGERS
 ↓ MARKET CLOSE
 ↓ OHLCV + HORIZON + DECISION OUTCOME EVALUATION
 ↓ BUCKET × HORIZON × CONFIDENCE RELIABILITY
 ↓ CHAMPION / CHALLENGER + LIVE HEALTH GUARD
 ↓ AUTOMATIC ROLLBACK WHEN LIVE QUALITY DEGRADES
 ↓ GITHUB STATE UPDATE
```

## Stage 10.2 improvements

- Explicit stored predicted close for every 1D/3D/5D/7D/20D forecast.
- Historical horizon forecasts are evaluated when they mature, even if the stock is no longer selected.
- Persistent horizon evaluation ledger with predicted/actual/error/direction/confidence fields.
- Persistent decision-outcome ledger measuring decision return, maximum favourable/adverse move and win/loss outcome.
- Confidence calibration and uncertainty-aware **NO TRADE** filtering.
- Bucket-wise, horizon-wise and confidence-band reliability.
- Statistical confidence interval for direction accuracy in weekly reporting.
- Model-health score and live-quality monitoring.
- Champion/challenger promotion guarded by live performance.
- Automatic live rollback to the previous model when quality materially deteriorates.
- Explicit feature causality/leakage audit.
- All persistent state remains GitHub-only; no local database is required.

## Price buckets

```text
>1000
500-999
100-499
50-99
10-49
```

Up to five qualified stocks can be selected per configured bucket, subject to score, confidence, uncertainty and risk gates.

## Forecasts

Every selected stock stores:

- Current OHLCV
- Next-session predicted OHLCV
- Expected return and direction
- Calibrated confidence
- Prediction interval
- 1D / 3D / 5D / 7D / 20D expected return
- 1D / 3D / 5D / 7D / 20D predicted close
- Price bucket
- Final decision and risk

## Validation and learning

The evening pipeline evaluates:

1. Predicted vs actual OHLCV.
2. Direction accuracy.
3. Every matured multi-horizon forecast.
4. Price-bucket performance.
5. Bucket × horizon performance.
6. Confidence-band performance.
7. Decision outcomes.
8. Live model quality and drift.

Validated observations are persisted in GitHub state and used for reliability weighting, model selection and rollback protection.

## Weekly report

The weekly report includes:

- OHLC MAPE and direction accuracy
- 95% direction-accuracy confidence interval
- Close-error distribution (≤1%, ≤2%, ≤3%, ≤5%)
- Price-bucket performance
- Horizon performance
- Bucket × horizon best/worst combinations
- Confidence calibration
- Decision win rate and average return
- Learning trend
- Champion/challenger status
- Model health score

## Automation

Only **3 production workflows** are used:

```text
🌅 Morning     08:15 IST, Monday–Friday
🌙 Evening     22:00 IST, Monday–Friday
📊 Weekly      10:00 IST, Saturday
```

No separate regression-test workflow is part of production automation.

## Manual runs

```bash
python -m src.morning_runner
python -m src.evening
python -m src.weekly_report
```

## Repository structure

```text
src/                    Core Stage 10.2 application
.github/workflows/      Morning / Evening / Weekly only
data/stage2/            GitHub-persisted predictions, evaluations and state
portfolio_manager/      Separate portfolio-manager project components
requirements.txt        Python dependencies
tests/                  Development tests
README.md               Project documentation
```
