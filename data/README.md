# GitHub data store

GitHub is the persistent data store for the production workflow. No SQLite or local database is required.

- `data/stage2/ohlcv/` — one CSV per NSE equity. Existing history is reused first; only missing historical/new session rows are fetched, merged, de-duplicated and persisted.
- `data/stage2/predictions/` — prediction and actual ledgers used by morning/evening reports.
- `data/stage2/jump/` — saved Jump Watch predictions.
- `data/stage2/intraday/` — saved intraday predictions.
- `data/stage2/ipo/` — saved IPO intelligence.
- `data/stage2/metrics/` — model and evaluation metrics.
- `data/stage2/state/` — learning/model state.

### Data reuse policy

1. Read the stock's existing OHLCV CSV from GitHub first.
2. If the cache already covers the configured history, do not re-download that history.
3. If history is incomplete, fetch only the missing historical range.
4. If the latest stored session is older than the current session, fetch only the missing tail.
5. Merge and de-duplicate by trading date, then save the updated CSV.
6. The morning/evening workflows commit `data/stage2/`, so the next run can reuse the data.
7. Telegram reports surface the available stored prediction/OHLCV, Jump Watch, intraday, IPO and portfolio datasets in their respective compact tables.

This keeps GitHub as the source of truth while avoiding repeated full-history downloads.