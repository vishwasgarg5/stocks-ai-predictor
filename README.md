# AI NSE Stock Predictor — Stage 4.2

Production release combining **price-bucket selection + multi-horizon forecasting + market/sector intelligence**, while retaining next-day OHLCV, jump and intraday engines.

## Morning Telegram — only 3 sections

1. **Top stocks by price bucket** — up to **2 qualified stocks in each of 5 price buckets** (maximum 10), with predicted **Open / High / Low / Close / Volume**.
2. **+5% jump watch** — only candidates passing both a modeled probability gate and minimum 7-session upside estimate.
3. **Intraday stocks** — only actionable UP/DOWN setups passing stricter score, volume and confidence filters; when none qualify, the report shows the scan/rejection summary.

The model never pads the list with weak stocks. A bucket can have 0, 1 or 2 stocks.

## Price buckets

| Bucket | Current Price | Maximum |
|---|---:|---:|
| B1 | > ₹1,000 | 2 |
| B2 | ₹500–₹999 | 2 |
| B3 | ₹100–₹499 | 2 |
| B4 | ₹50–₹99 | 2 |
| B5 | ₹10–₹49 | 2 |

Buckets provide diversification without forcing a weak stock into the final list.

## Prediction engines

- Next-session Open/High/Low/Close/Volume ensemble.
- Valid OHLC geometry enforcement.
- Independent 1/3/5/7/20 trading-session close forecasts.
- Multi-horizon expected return contributes to final ranking.
- Expensive multi-horizon models run only for the small post-bucket candidate pool.

## Stage 4.2 intelligence and quality controls

- Broad NSE universe capped at 1,000 stocks.
- Liquidity and technical prescreening.
- Nifty market regime.
- Cached Yahoo Finance sector mapping.
- Sector momentum and relative strength.
- Evidence-weighted historical stock reliability.
- **Trade Confidence** is separate from raw model confidence.
- Direction-vs-return alignment detects conflicts such as `UP` direction with a strongly negative predicted return.
- Multi-horizon directional agreement contributes to Trade Confidence.
- Final quality gate uses score, model confidence, trade confidence and signal alignment.
- Jump Watch rejects weak probability/upside combinations instead of filling five rows.
- Intraday engine records rejection reasons and reports the funnel when no setup qualifies.

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

Evening evaluation compares every saved prediction against actual OHLCV, maintains cumulative accuracy metrics, updates stock reliability and retains champion/challenger retraining.

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
