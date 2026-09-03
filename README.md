# AI NSE Stock Predictor — Stage 4.2

Production release combining **price-bucket selection + multi-horizon forecasting + market/sector intelligence**, while retaining next-day OHLCV, jump and intraday engines.

## Morning Telegram — only 3 sections

1. **Top stocks by price bucket** — only quality-qualified stocks with predicted **Open / High / Low / Close / Volume**.
2. **+5% jump watch** — candidates with a modeled path toward +5% within 7 trading sessions.
3. **Intraday stocks** — only actionable UP/DOWN setups passing stricter quality and volume filters.

The model does **not** force five stocks. If only two qualify, only two are reported.

## Price buckets

| Bucket | Current Price |
|---|---:|
| B1 | > ₹1,000 |
| B2 | ₹500–₹999 |
| B3 | ₹100–₹499 |
| B4 | ₹50–₹99 |
| B5 | ₹10–₹49 |

Buckets provide diversification but never force a weak stock into the final list.

## Prediction engines

- Next-session Open/High/Low/Close/Volume ensemble.
- Valid OHLC geometry enforcement.
- Independent 1/3/5/7/20 trading-session close forecasts.
- Multi-horizon expected return contributes to final ranking.
- Expensive multi-horizon models run only for the small post-bucket candidate pool.

## Stage 4.2 intelligence

- Broad NSE universe capped at 1,000 stocks.
- Liquidity and technical prescreening.
- Nifty market regime.
- Cached Yahoo Finance sector mapping.
- Sector momentum and relative strength.
- Evidence-weighted historical stock reliability.
- Quality gates prevent weak predictions from filling the Top list.

### Ranking weights

| Component | Weight |
|---|---:|
| Technical | 18% |
| Next-day Expected Return | 15% |
| Model Confidence | 15% |
| Direction Confidence | 12% |
| Historical Reliability | 8% |
| Market Regime | 10% |
| Sector Strength | 10% |
| Multi-Horizon Return | 12% |
| **Total** | **100%** |

## Data safety and learning

Morning prediction uses the latest completed equity session available before the prediction date and never today's partial bar. Persistent state is GitHub-only; there is no local database.

Evening evaluation compares predicted vs actual OHLCV, maintains cumulative accuracy metrics, updates reliability and retains champion/challenger retraining.

## Schedule

- Morning: **5:00 AM IST**, Monday–Friday
- Evening: **4:30 PM IST**, Monday–Friday
- Weekly: **6:00 PM IST Friday**

## Manual runs

```bash
python -m src.morning_runner
python -m src.evening
python -m src.weekly_report
```
