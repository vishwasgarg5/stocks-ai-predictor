# stocks-ai-predictor

AI-assisted NSE stock prediction pipeline with GitHub Actions, incremental GitHub-persisted market data, prediction lineage, evening evaluation/retraining, portfolio intelligence, and Telegram reporting.

## Production flow

```text
NSE universe
   ↓
GitHub monthly OHLCV cache
   ↓
incremental update (only missing dates)
   ↓
quality / liquidity screening
   ↓
cheap technical ranking
   ↓
bounded ML candidate pool
   ↓
next-session OHLC prediction
   ↓
immutable prediction ledger
   ↓
16:30 IST evaluation
   ↓
accuracy + decision learning + conditional retraining
   ↓
Telegram report
```

## Scheduled workflows

- Morning prediction: **05:00 IST, Monday-Friday**
- Evening evaluation/retraining: **16:30 IST, Monday-Friday**
- Weekly report: scheduled separately
- P1 validation: validation-only workflow

## Market-data storage

New market data is stored in monthly partitions under `data/stage2/market_data/YYYY-MM.csv` with `Date,Symbol,Open,High,Low,Close,Volume` columns. Existing legacy `data/stage2/ohlcv/` data is read for migration compatibility.

The cache is incremental: after initial history exists, the pipeline requests only missing older/newer ranges, merges and deduplicates by `Symbol + Date`, and keeps the partition write protected for concurrent workers.

## Model safety

Training features are cutoff-aware: prediction-day information is not included in the training sample. Evening evaluation uses the immutable morning prediction ledger and validates `Prediction_ID` lineage before accepting results.

## Repository state

Generated market state is persisted in GitHub so scheduled runs can reuse prior history without a separate database.
