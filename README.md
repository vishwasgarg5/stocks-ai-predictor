# AI NSE Stock Predictor — Stage 4 (3A + 3B + 4)

Stage 4 is the combined production release containing **Stage 3A price-bucket selection + Stage 3B multi-horizon prediction + Stage 4 market/sector intelligence**, while retaining the Stage 2 OHLC, jump and intraday engines.

## Roadmap

| Stage | Status | Purpose |
|---|---|---|
| Stage 1 | Complete | Basic OHLC prediction |
| Stage 1.5 | Complete | Accuracy/validation improvements |
| Stage 2 | Complete | Production NSE prediction, jump and intraday engines |
| Stage 3A | Included | Price-bucket selection |
| Stage 3B | Included | 1D/3D/5D/7D/20D multi-horizon forecasts |
| Stage 4 | Active | Market + sector intelligence and sector-aware ranking |

## Stage 3A — Price buckets

| Bucket | Current Price |
|---|---:|
| B1 | > ₹1,000 |
| B2 | ₹500–₹999 |
| B3 | ₹100–₹499 |
| B4 | ₹50–₹99 |
| B5 | ₹10–₹49 |

Stocks are scored first and the strongest stock from each available price bucket is eligible for the final list. The morning list can therefore contain **fewer than five stocks** when fewer buckets have strong candidates; it never forces a weak stock merely to reach five.

## Stage 3B — Multi-horizon prediction

The model independently forecasts future closing price and expected return for **1, 3, 5, 7 and 20 trading sessions**, using chronological validation and an XGBoost + Random Forest + Extra Trees ensemble with validation-derived weights.

These forecasts remain part of the model pipeline but are not printed in the compact morning Telegram message.

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

- Next-day **Open/High/Low/Close/Volume (OHLCV)** prediction
- Valid OHLC ordering enforcement (High cannot be below Open/Close; Low cannot be above them)
- 7-session +5% jump watchlist
- Intraday Top 5
- Champion/Challenger model control
- Evening prediction-vs-actual evaluation and retraining
- GitHub-only learning/evaluation data

## Morning Telegram — intentionally compact

Only three actionable sections are sent:

1. **Top stocks by price bucket** — predicted Open, High, Low, Close and Volume
2. **+5% jump watchlist** — next 7 sessions
3. **Intraday stocks** — bias, CMP, target, stop-loss and confidence

No scan summary, multi-horizon table, selection-weight table or model diagnostics are included in the morning Telegram message.

## Morning flow

`NSE universe → liquidity → completed-session cutoff → technical prescreen → Stage 2 OHLCV ML → Stage 3A price bucket → Stage 4 sector context → bucket-based final list → jump → intraday → Telegram`

Morning predictions exclude today's date and determine the cutoff from the latest completed equity-session bars actually available across the downloaded universe. A Nifty-derived cutoff is used only as a fallback. This avoids the previous unnecessary T-2 cutoff when yesterday's completed data is already available.

## Code navigation

- `src/multihorizon.py` — **STAGE 3B: 1/3/5/7/20-session engine**
- `src/stage4_engine.py` — **STAGE 3A: price buckets + STAGE 4: sector intelligence**
- `src/selection.py` — **STAGE 4 final ranking**
- `src/prediction.py` — **Stage 2 OHLCV + Stage 3B integration**
- `src/morning_runner.py` — **complete morning pipeline**
- `src/telegram_report.py` — **compact 3-section morning report + evening report**
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
