# AI NSE Stock Predictor — Stage 4 (3A + 3B + 4)

Stage 4 is the combined production release containing **Stage 3A price-bucket selection + Stage 3B multi-horizon prediction + Stage 4 market/sector intelligence**, while retaining the Stage 2 OHLC, jump and intraday engines.

## Roadmap

| Stage | Status | Purpose |
|---|---|---|
| Stage 1 | Complete | Basic OHLC prediction |
| Stage 1.5 | Complete | Accuracy/validation improvements |
| Stage 2 | Complete | Production NSE prediction, jump and intraday engines |
| Stage 3A | **Included** | Price-bucket selection |
| Stage 3B | **Included** | 1D/3D/5D/7D/20D multi-horizon forecasts |
| Stage 4 | **Active** | Market + sector intelligence and sector-aware ranking |

## Stage 3A — Price buckets

| Bucket | Current Price |
|---|---:|
| B1 | > ₹1,000 |
| B2 | ₹500–₹999 |
| B3 | ₹100–₹499 |
| B4 | ₹50–₹99 |
| B5 | ₹10–₹49 |

Stocks are scored first, then the strongest candidates from each bucket are retained before final Top 5 ranking. The model does **not** force one stock from every bucket.

## Stage 3B — Multi-horizon prediction

The model now independently forecasts future closing price and expected return for:

**1, 3, 5, 7 and 20 trading sessions.**

Each horizon uses chronological validation and an XGBoost + Random Forest + Extra Trees ensemble with validation-derived weights.

Morning Telegram reports the expected return for every horizon for the final Top 5.

## Stage 4 — Market + Sector Intelligence

Stage 4 adds:

- Nifty market regime
- Sector mapping
- 20-session sector/candidate momentum
- Relative sector strength
- Sector-aware ranking
- Cached sector mapping in GitHub

### Final ranking

| Component | Weight |
|---|---:|
| Technical Score | 20% |
| Expected Return | 18% |
| Model Confidence | 18% |
| Direction Confidence | 14% |
| Reliability | 10% |
| Market Regime | 10% |
| Sector Strength | 10% |
| **Total** | **100%** |

## Existing engines retained

- Next-day Open/High/Low/Close prediction
- 7-day +5% jump watchlist
- Intraday Top 5
- Champion/Challenger model control
- Evening prediction-vs-actual evaluation
- GitHub-only learning/evaluation data

## Morning flow

`NSE universe → liquidity → technical prescreen → Stage 2 ML → Stage 3A price bucket → Stage 4 sector context → final ranking → Top 5 → Stage 3B horizons → jump → intraday → Telegram`

Morning predictions use only completed sessions before the prediction date, preventing same-day target leakage.

## Code navigation — easy access

Prominent comments identify each stage directly in the code:

- `src/multihorizon.py` — **STAGE 3B: 1/3/5/7/20-day engine**
- `src/stage4_engine.py` — **STAGE 3A: price buckets + STAGE 4: sector intelligence**
- `src/selection.py` — **STAGE 4 final ranking**
- `src/prediction.py` — **Stage 2 OHLC + Stage 3B integration**
- `src/morning_runner.py` — **complete Stage 3A + 3B + 4 pipeline**
- `src/telegram_report.py` — **Stage 3B/Stage 4 tables**
- `src/config.py` — **active stage/version and centralized settings**

## Data

No local database is used. Persistent data is stored under `data/stage2/` and committed to GitHub by Actions.

## Model learning

Evening evaluation measures MAPE, direction accuracy and stock reliability. Champion/Challenger replacement requires the configured meaningful improvement threshold rather than reacting to tiny changes.

## Manual runs

```bash
python -m src.morning_runner
python -m src.evening
python -m src.weekly_report
```
