# AI NSE Stock Predictor — Final Stage 10

A production-oriented NSE stock prediction system with a cumulative **Stage 1 → Stage 10** architecture. Existing next-day OHLCV, +5% jump and intraday engines are retained; the final intelligence layer adds accuracy calibration, adaptive learning, market intelligence, event/news hooks, decision scoring and drift monitoring.

## Final architecture

```text
STAGE 1  Foundation
  1.0 Basic OHLC → 1.1 Features → 1.2 Indicators → 1.3 XGBoost → 1.4 Time-series validation → 1.5 Accuracy
      ↓
STAGE 2  Production Prediction
  2.0 NSE universe → 2.1 Data → 2.2 OHLCV → 2.3 Direction → 2.4 +5% Jump → 2.5 Intraday → 2.6 Evaluation → 2.7 Champion/Challenger → 2.8 Telegram
      ↓
STAGE 3  Advanced Forecasting
  3A Price Buckets → 3B 1D/3D/5D/7D/20D horizons
      ↓
STAGE 4  Market Intelligence
  4.0 Regime → 4.1 Sector Strength → 4.2 Rotation → 4.3 Breadth → 4.4 Ranking → 4.5 Risk Intelligence
      ↓
STAGE 5  Accuracy & Calibration
  5.0 Walk-forward → 5.1 Rolling → 5.2 Stock accuracy → 5.3 Horizon accuracy → 5.4 Direction → 5.5 Errors → 5.6 Calibration → 5.7 Intervals → 5.8 Accuracy weighting
      ↓
STAGE 6  Adaptive AI
  6.0 Error learning → 6.1 Stock reliability → 6.2 Sector reliability → 6.3 Regime learning → 6.4 Horizon reliability → 6.5 Dynamic ensemble → 6.6 Feature importance → 6.7 Adaptive retraining
      ↓
STAGE 7  Advanced Market Intelligence
  7.0 Nifty → 7.1 Bank Nifty → 7.2 Breadth → 7.3 Volatility → 7.4 Rotation → 7.5 Relative strength → 7.6 Correlation → 7.7 FII/DII → 7.8 Global influence
      ↓
STAGE 8  Event & News Intelligence
  8.0 Ingestion → 8.1 Sentiment → 8.2 Events → 8.3 Earnings → 8.4 Corporate actions → 8.5 Result risk → 8.6 Impact → 8.7 Event probability → 8.8 Price/news confirmation
      ↓
STAGE 9  Decision Intelligence
  9.0 BUY/HOLD/AVOID → 9.1 Return → 9.2 Success probability → 9.3 Risk-adjusted return → 9.4 Target probability → 9.5 Stop-risk → 9.6 Reward/Risk → 9.7 Setup quality → 9.8 Final score
      ↓
STAGE 10  Self-Improving AI
  10.0 Continuous learning → 10.1 Model drift → 10.2 Feature drift → 10.3 Regime drift → 10.4 Model replacement → 10.5 Champion/Challenger evolution → 10.6 Monitoring → 10.7 Failure detection → 10.8 Rollback
```

## Morning output — exactly 3 actionable sections

1. **Top stocks** — up to 2 qualified stocks per price bucket, maximum 10; no weak-stock padding.
2. **+5% jump watch** — true 7-session forecast with probability/upside gates.
3. **Intraday stocks** — strict volume/confidence/direction gates and rejection funnel when empty.

The selected stock records now also carry calibrated confidence, prediction interval, market/breadth/news hooks, regime-adjusted score, final decision score, action and risk fields.

### Important data rule

Stages 7–8 never fabricate unavailable FII/DII, global or news information. Missing optional external inputs stay neutral until a real data source is supplied. This prevents false signals and future-data leakage.

## Learning loop

```text
Prediction → Actual result → Error → Reliability → Calibration → Challenger → Compare → Promote only if better
                         ↘ drift detection / failure detection / rollback ↗
```

Persistent state remains **GitHub-only** under `data/stage2/`; there is no local database. GitHub Actions commits generated state/data back to the repository.

## Current release

- `STAGE_NAME = Stage 10 (Final Self-Improving AI)`
- `MODEL_VERSION = stage10-v1.0`
- Universe cap: 1,000 liquid NSE stocks
- Forecast horizons: 1, 3, 5, 7 and 20 sessions
- Existing champion/challenger retraining preserved
- Final regression suite covers Stages 1–10 architecture

## Manual runs

```bash
python -m src.morning_runner
python -m src.evening
python -m src.weekly_report
```
