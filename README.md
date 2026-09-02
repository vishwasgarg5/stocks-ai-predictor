# AI NSE Stock Predictor — Stage 4

Stage 4 upgrades the production stock-prediction system with **price-bucket selection** and **market/sector intelligence**, while keeping the proven Stage 2 OHLC, jump and intraday engines intact.

## Stage roadmap

| Stage | Purpose |
|---|---|
| Stage 1 | Basic OHLC prediction |
| Stage 1.5 | Accuracy and validation improvements |
| Stage 2 | Production NSE prediction, jump and intraday engines |
| Stage 3A | Price-bucket selection |
| Stage 3B | Multi-horizon prediction roadmap |
| **Stage 4** | **Market + sector intelligence and sector-aware ranking** |
| Stage 5+ | Advanced ranking, events/news, risk, backtesting and continuous learning |

## Stage 4 additions

### 1. Price-bucket selection

Stocks are classified by current price into five buckets:

| Bucket | Price |
|---|---:|
| B1 | > ₹1,000 |
| B2 | ₹500–₹999 |
| B3 | ₹100–₹499 |
| B4 | ₹50–₹99 |
| B5 | ₹10–₹49 |

The model does **not** force one stock from every bucket. It keeps the strongest candidates from each bucket and then performs the final ML ranking.

### 2. Sector intelligence

The Stage 4 engine adds:

- Sector mapping
- 20-session sector/candidate momentum
- Relative sector strength
- Candidate breadth information
- Sector score used in final ranking

Sector mappings are cached in `data/stage2/metrics/sector_map.csv` and committed to GitHub. No local database is used.

### 3. Sector-aware final ranking

The final score now uses:

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

## Existing Stage 2 engines

### Next-Day Prediction

Scans the broad NSE universe, applies liquidity and technical screening, then predicts:

- Open
- High
- Low
- Close
- Direction
- Confidence
- Technical score
- Market regime
- Sector strength
- Price bucket

The final output is the strongest five stocks after Stage 4 ranking.

### 7-Day +5% Jump Engine

Every morning the system searches for stocks with a reasonable probability of reaching at least +5% within seven trading sessions.

### Intraday Engine

The system scans the complete liquid universe using intraday data and reports strong setups with bias, target, stop-loss and confidence.

## Model architecture

The prediction engine uses an ensemble of:

- XGBoost
- Random Forest
- Extra Trees

Ensemble weights are learned from chronological validation performance.

## Champion / Challenger

Two model configurations are maintained:

- Variant A — Champion
- Variant B — Challenger

The Challenger must improve validation error by at least 2% before replacing the Champion.

## Prediction safety

Features from trading session T predict the OHLC of session T+1. Morning predictions use only completed market data before the prediction date, preventing same-day target leakage.

## Data storage

No local database is used. Persistent Stage 4 data remains under:

`data/stage2/`

including predictions, evaluations, jump data, intraday data, metrics, model state and the Stage 4 sector cache.

## Morning workflow

1. Load broad NSE universe.
2. Apply liquidity filtering.
3. Calculate technical indicators.
4. Prescreen candidates.
5. Run Stage 2 ensemble prediction.
6. Add Stage 4 sector intelligence.
7. Assign price buckets.
8. Score candidates using the Stage 4 ranking formula.
9. Apply price-bucket candidate balancing.
10. Select final Top 5.
11. Run 7-day jump engine.
12. Run intraday engine.
13. Save results to GitHub.
14. Send precise tabular Telegram report.

## Evening workflow

1. Load the exact morning prediction ledger.
2. Do not select stocks again.
3. Fetch actual OHLC.
4. Calculate prediction errors.
5. Calculate direction accuracy.
6. Update cumulative metrics.
7. Update stock reliability.
8. Run Champion/Challenger comparison.
9. Promote Challenger only when meaningfully better.
10. Save learning data.

## Code navigation

Stage 4 code is deliberately documented with prominent `STAGE 4` comments so future changes can be located quickly:

- `src/stage4_engine.py` — **price buckets + sector intelligence**
- `src/selection.py` — **Stage 4 final ranking weights**
- `src/morning_runner.py` — **Stage 4 integration point**
- `src/telegram_report.py` — **Stage 4 Telegram tables**
- `src/config.py` — **Stage/version and Stage 4 paths**
- `tests/test_stage2.py` — **Stage 4 regression tests**

## Telegram secrets

Create these GitHub Actions secrets:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

under Repository → Settings → Secrets and variables → Actions.

## Manual runs

```bash
python -m src.morning_runner
python -m src.evening
python -m src.weekly_report
```

## Stage 4 philosophy

Stage 4 is an **additive selection layer**, not a replacement for the existing prediction model. The system continues to measure prediction error, direction accuracy, jump performance, intraday performance and stock reliability before allowing future model changes to become the active Champion.
