# AI NSE Stock Predictor — Stage 10

Production-oriented NSE stock prediction and decision-support system covering the cumulative **Stage 1 → Stage 10** pipeline.

> Predictions are model outputs, not guaranteed returns or investment advice.

## Production flow

```text
NSE UNIVERSE
    ↓
DATA QUALITY + LIQUIDITY
    ↓
TECHNICAL FEATURES
    ↓
AI OHLC PREDICTION
    ↓
PRICE BUCKETS
    ↓
1D / 3D / 5D / 7D / 20D
    ↓
MARKET + SECTOR + RISK
    ↓
CONFIDENCE + DECISION ENGINE
    ↓
TOP 5 AI STOCKS
    ↓
+5% JUMP WATCH TOP 5
    ↓
INTRADAY TOP 5
    ↓
TELEGRAM
    ↓
MARKET CLOSE
    ↓
PREDICTION vs ACTUAL
    ↓
ACCURACY + RELIABILITY
    ↓
CHAMPION / CHALLENGER
    ↓
GITHUB STATE UPDATE
```

## Stage 1 → 10

```text
1 Foundation
   ↓
2 Production prediction
   ↓
3 Advanced multi-horizon forecasting
   ↓
4 Market intelligence
   ↓
5 Accuracy & calibration
   ↓
6 Adaptive AI
   ↓
7 Advanced market intelligence
   ↓
8 Event/news intelligence
   ↓
9 Decision intelligence
   ↓
10 Self-improving AI
```

## Morning report

Exactly **3 actionable sections**:

1. **Top Stocks** — maximum 5 quality-gated stocks.
2. **+5% Jump Watch** — maximum 5 qualified candidates.
3. **Intraday Stocks** — maximum 5 qualified setups.

Top stocks show Price Bucket, CMP, current OHLCV, AI Target, Expected %, Stop Loss when available, R/R when calculable, Confidence and AI Decision.

Supporting information includes market snapshot, sector strength, IPO/new-listing information when verified, portfolio snapshot, scan counts, model accuracy and model-health warnings.

The system never fills the list with weak stocks just to reach five.

## Evening report

```text
PREDICTED OHLC
      ↓
ACTUAL OHLC
      ↓
DIFFERENCE
      ↓
DIRECTION ACCURACY
      ↓
PRICE-BUCKET RESULTS
      ↓
MODEL LEARNING
      ↓
CHAMPION / CHALLENGER
```

The evening job validates predictions against the completed market session and updates reliability/learning state.

## Self-improving AI

The model tracks prediction error, direction accuracy, stock reliability, confidence, uncertainty and model drift. A challenger is promoted only when it clears the configured improvement threshold.

## Data integrity

- Historical predictions use only information available at the prediction cutoff.
- Actual market data is used only for post-market validation.
- Missing optional external information is not fabricated.
- Price-bucket and quality gates are applied before final selection.
- Persistent state is GitHub-only; no local database is required.

## Current release

- **Stage:** Stage 10 — Final Self-Improving AI
- **Model:** `stage10-v1.1`
- **Forecasts:** 1D, 3D, 5D, 7D, 20D
- **Universe cap:** 1,000 NSE stocks
- **Final AI selection:** Top 5
- **Price-bucket limit:** maximum 2 per bucket
- **Jump Watch:** Top 5
- **Intraday:** Top 5
- **Validation:** OHLCV error + direction accuracy
- **Learning:** champion/challenger + drift controls

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
src/                    Core Stage 10 application
.github/workflows/      Morning / Evening / Weekly only
data/stage2/            GitHub-persisted predictions, evaluations and state
portfolio_manager/      Separate portfolio-manager project components
requirements.txt        Python dependencies
tests/                  Development regression tests
README.md               Project documentation
```
