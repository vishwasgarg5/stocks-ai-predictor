# Stage 2 generated market data

Market history is persisted in monthly partitions under `data/stage2/market_data/`.

Format:

`YYYY-MM.csv` with columns `Date,Symbol,Open,High,Low,Close,Volume`.

The application reads the GitHub cache first and downloads only missing date ranges. Legacy per-symbol files under `data/ohlcv/` and `data/stage2/ohlcv/` are supported during migration but are not the target storage format.
