# AI NSE Stock Predictor — Stage 10.4

Production-oriented NSE stock prediction and decision-support system. The live repository implements the cumulative Stage 1 → **Stage 10.4** pipeline.

> Predictions are model outputs, not guaranteed returns or investment advice.

## Production flow

```text
NSE UNIVERSE
 ↓ DATA QUALITY + LIQUIDITY
 ↓ CAUSAL TECHNICAL / FUNDAMENTAL FEATURES + LEAKAGE CONTROLS
 ↓ AI OHLCV PREDICTION
 ↓ PRICE-BUCKET RANKING
 ↓ 1D / 3D / 5D / 7D / 20D FORECASTS
 ↓ MARKET / SECTOR / VOLATILITY / BENCHMARK CONTEXT
 ↓ CONFIDENCE + TARGET-HIT PROBABILITIES
 ↓ STOCK / HORIZON RELIABILITY
 ↓ COST-AWARE RISK + ADAPTIVE THRESHOLDS
 ↓ BUY / WATCH / HOLD / AVOID / NO TRADE
 ↓ GITHUB PREDICTION + DECISION LEDGERS
 ↓ MARKET CLOSE EVALUATION
 ↓ BUCKET × HORIZON × CONFIDENCE LEARNING
 ↓ CHAMPION / CHALLENGER + LIVE HEALTH GUARD
 ↓ ROLLBACK PROTECTION
 ↓ GITHUB STATE UPDATE
```

## Stage 10.4

- Walk-forward validation and leakage-aware features.
- AI next-session OHLCV prediction.
- Multi-horizon 1D / 3D / 5D / 7D / 20D forecasts.
- Nifty 150 universe with liquidity/data-quality filtering.
- Price-bucket selection with **no global five-stock cap**; every qualified stock can be retained within the scanned candidate universe.
- Separate **best pick per price bucket** in Telegram reporting.
- Confidence calibration and target-hit probability models.
- Downside probability and prediction-uncertainty abstention.
- Stock-specific and horizon-specific reliability.
- Volatility-adjusted return, transaction cost and slippage awareness.
- Benchmark-relative expected-return edge.
- Adaptive BUY/WATCH/NO-TRADE thresholds by regime and reliability.
- Intraday, jump and IPO intelligence modules.
- Decision-outcome and multi-horizon evaluation ledgers.
- Champion/challenger model selection with statistical promotion gates.
- Live model-health monitoring and rollback protection.
- GitHub-only persistent state; no local database.
- Portfolio reporting remains a separate component and does not change the core prediction model.

## Price buckets

```text
>1000
500-999
100-499
50-99
10-49
```

The morning report shows **all qualified stocks by bucket**, followed by a separate table containing the best pick from each bucket. The number of stocks is not artificially capped at five.

## Forecast output

Selected/qualified stocks can include:

- Current OHLCV
- Predicted next-session OHLCV
- Expected return and direction
- Calibrated confidence
- Prediction uncertainty
- Target-hit probabilities
- 1D / 3D / 5D / 7D / 20D expected returns
- Predicted closes by horizon
- Price bucket
- Reliability and risk scores
- Final decision and risk classification

## Validation and learning

The evening pipeline evaluates:

1. Predicted vs actual OHLCV.
2. Direction accuracy.
3. Matured multi-horizon forecasts, including stocks no longer selected.
4. Price-bucket performance.
5. Bucket × horizon performance.
6. Confidence calibration.
7. Decision outcomes and realised returns.
8. Model health, drift and live degradation.

Validated observations remain in GitHub state and support reliability weighting, model comparison and rollback protection.

## Weekly report

The weekly report covers OHLC MAPE, direction accuracy, confidence intervals, close-error bands, bucket performance, horizon performance, bucket × horizon results, confidence calibration, decision win rate/return, learning trend, champion/challenger status and model health.

## Automation

Exactly **3 production workflows** are used:

```text
🌅 Stage 10.4 Morning     08:15 IST, Monday–Friday
🌙 Stage 10.4 Evening     16:30 IST, Monday–Friday
📊 Stage 10.4 Weekly      10:00 IST, Saturday
```

Morning is one hour before the 09:15 IST NSE cash-market open. No separate regression-test workflow is used in production automation.

## Manual runs

```bash
python -m src.morning_runner
python -m src.evening
python -m src.weekly_report
```

## Repository structure

```text
src/                    Core Stage 10.4 application
.github/workflows/      Stage 10.4 Morning / Evening / Weekly only
data/stage2/            GitHub-persisted predictions, evaluations and state
portfolio_manager/      Separate portfolio-manager components
requirements.txt        Python dependencies
tests/                  Development tests
README.md               Project documentation
```
