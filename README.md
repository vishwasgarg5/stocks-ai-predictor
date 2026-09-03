# AI NSE Stock Predictor — Stage 10

Production-oriented NSE stock prediction and decision-support system built as a cumulative **Stage 1 → Stage 10** pipeline. It combines OHLCV prediction, price-bucket coverage, market/sector intelligence, jump-watch and intraday screening, decision scoring, portfolio reporting, accuracy validation and self-improving model controls.

> **Purpose:** generate compact, evidence-based NSE watchlists and evaluate the model against actual market results. Predictions are not guaranteed returns or investment advice.

## Architecture

```text
STAGE 1  FOUNDATION
OHLCV → Features → Indicators → XGBoost → Time-series validation → Accuracy
   ↓
STAGE 2  PRODUCTION PREDICTION
NSE universe → Data quality → OHLCV → Direction → +5% Jump → Intraday → Evaluation → Champion/Challenger → Telegram
   ↓
STAGE 3  ADVANCED FORECASTING
Price Buckets → 1D / 3D / 5D / 7D / 20D forecasts
   ↓
STAGE 4  MARKET INTELLIGENCE
Market regime → Sector strength → Rotation → Breadth → Ranking → Risk intelligence
   ↓
STAGE 5  ACCURACY & CALIBRATION
Walk-forward → Rolling accuracy → Stock/horizon accuracy → Direction → Error analysis → Calibration
   ↓
STAGE 6  ADAPTIVE AI
Error learning → Reliability → Regime learning → Dynamic weighting → Feature importance → Adaptive retraining
   ↓
STAGE 7  ADVANCED MARKET INTELLIGENCE
NIFTY → BANK NIFTY → Breadth → Volatility → Relative strength → Correlation → FII/DII → Global influence
   ↓
STAGE 8  EVENT & NEWS INTELLIGENCE
News → Sentiment → Events → Earnings → Corporate actions → Event risk → Price/news confirmation
   ↓
STAGE 9  DECISION INTELLIGENCE
BUY/HOLD/AVOID → Expected return → Probability → Risk-adjusted return → Target → Stop risk → Reward/Risk → Final score
   ↓
STAGE 10  SELF-IMPROVING AI
Continuous learning → Model/feature/regime drift → Failure detection → Champion/Challenger → Replacement → Monitoring → Rollback
```

## Daily workflow

```text
                    ┌───────────────┐
                    │  NSE UNIVERSE │
                    └───────┬───────┘
                            ↓
                    DATA QUALITY GATE
                            ↓
             ┌──────────────┴──────────────┐
             ↓                             ↓
       AI PREDICTION                 INTRADAY SCAN
             ↓                             ↓
       PRICE BUCKETS                  TOP SETUPS
             ↓                             ↓
       DECISION ENGINE ← MARKET / SECTOR / RISK
             ↓
       MORNING TELEGRAM
             ↓
       MARKET CLOSE
             ↓
       ACTUAL vs PREDICTED
             ↓
       ACCURACY + ERROR ANALYSIS
             ↓
       CHALLENGER / RETRAINING
             ↓
       EVENING TELEGRAM
             ↓
       GITHUB STATE UPDATE
```

## Morning Telegram report

The report is intentionally **decision-focused and mobile-friendly**. It does not add unnecessary tables or duplicate information.

### 1. Top 5 AI Stocks

Each selected stock includes:

- **Price Bucket** — price-band classification used for balanced coverage.
- Current **CMP + Open + High + Low + Close + Volume**.
- **AI Target** / predicted close.
- **Expected %** move from CMP to AI target.
- **Stop Loss** when available.
- **Risk/Reward** when calculable.
- **Confidence**.
- **AI Decision** — BUY / HOLD / WATCH / AVOID.

Selection is subject to data, liquidity, model-confidence and quality controls; weak stocks are not added merely to fill the list.

### 2. +5% Jump Watch

Up to 5 candidates with:

- CMP
- +5% target
- target percentage
- estimated 7-session upside
- probability

Candidates are subject to the jump-quality gates rather than being presented as guaranteed movers.

### 3. Intraday

Up to 5 qualified setups with:

- Bias
- CMP
- Target
- Stop Loss
- Confidence

If no setup passes the live quality gates, the report explicitly says so.

### Other morning intelligence

- Market snapshot: NIFTY, BANK NIFTY, VIX and breadth.
- Market regime.
- Sector AI strength.
- IPO / new-listing information when verified data is available.
- Portfolio Manager summary and position alerts.
- Scan funnel showing universe → data → liquid → AI → selected counts.
- Compact previous → current model accuracy.
- Model-health warning when health/drift is weak; recommendation confidence is reduced rather than hiding the warning.

## Price buckets

Price buckets are part of the **core selection and reporting layer**, not a separate afterthought.

The report identifies the price band for every selected stock so the final list can be inspected across different price levels instead of unintentionally concentrating in one range.

The bucket shown in Telegram is derived from the stock-selection data and remains visible beside the stock name.

## Evening Telegram report

```text
PREDICTION vs ACTUAL
        ↓
PRED OHLC
ACT  OHLC
DIFF OHLC
        ↓
Direction: Predicted → Actual
        ↓
Price-bucket results
Intraday results
IPO/new-listing results
        ↓
Model Learning
Samples → MAPE → Direction Accuracy
Previous → Current Accuracy
Champion/Challenger
Retrained? → Improvement → Learning State
```

The evening report is the validation layer. It measures what actually happened instead of treating a prediction as success merely because it was generated.

## Self-improving learning loop

```text
Prediction
    ↓
Actual market result
    ↓
Error measurement
    ↓
Accuracy / reliability
    ↓
Calibration + drift detection
    ↓
Challenger model
    ↓
Compare with champion
    ↓
Promote only when the challenger is better
    ↓
Continue learning
```

Model replacement is therefore performance-driven rather than automatic retraining for its own sake.

## Data integrity

- Missing optional external information is not fabricated.
- FII/DII, global and news inputs remain neutral when no verified source is available.
- Future information must not leak into historical prediction/evaluation.
- Actual market data is used for post-market validation.
- Reports distinguish predictions from verified actual results.

## Persistence

Generated state and model/report data are persisted **GitHub-only** under the repository data/state paths. The project does not depend on a local database for its persistent learning state.

GitHub Actions can run the scheduled morning/evening/weekly jobs and commit generated state back to the repository.

## Current release

- **Stage:** Stage 10 — Final Self-Improving AI
- **Model version:** `stage10-v1.0`
- **Forecast horizons:** 1D, 3D, 5D, 7D and 20D
- **Selection:** quality-gated Top 5 reporting with price-bucket visibility
- **Intraday:** quality-gated Top 5
- **Jump Watch:** quality-gated Top 5
- **Validation:** prediction vs actual + direction accuracy + error metrics
- **Learning:** champion/challenger + drift/failure controls
- **Reporting:** mobile-first Telegram morning and evening reports

## Manual runs

```bash
python -m src.morning_runner
python -m src.evening
python -m src.weekly_report
```

## Repository structure

```text
src/                    Core prediction, screening, intelligence and reporting code
.github/workflows/      Scheduled GitHub Actions workflows
data/                   GitHub-persisted state and generated data
models/                 Saved model artifacts
reports/                Generated reports
main.py                 Main entry point
morning.py              Morning entry point
evening.py              Evening entry point
weekly_report.py        Weekly reporting entry point
requirements.txt        Python dependencies
tests/                  Regression and pipeline tests
```
