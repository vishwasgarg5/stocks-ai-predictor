# GitHub data store

This project intentionally uses GitHub as the persistent data store. No SQLite or local database is required for the production workflow.

- `data/ohlcv/` — one rolling CSV per NIFTY 50 stock, keeping only the latest 3 months.
- `data/nifty.csv` — rolling NIFTY index data.
- `data/predictions.csv` — permanent prediction/actual ledger.
- `reports/performance.csv` — permanent model performance history.

The updater reads the latest date already committed to GitHub and downloads only data after that date. It then appends new rows and trims the OHLCV files back to the configured rolling window.