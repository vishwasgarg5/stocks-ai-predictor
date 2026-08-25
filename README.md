# NIFTY 50 AI Stock Predictor

Version 1 uses a **rolling 3-month daily data window**.

## Flow

1. Download the current/fallback NIFTY 50 universe.
2. Download the latest 3 months of daily OHLCV for each stock and NIFTY.
3. Build technical, volume, volatility and relative-strength features.
4. Rank the NIFTY 50 and select Top 5.
5. Train four XGBoost models for next-day Open/High/Low/Close returns.
6. Convert predicted returns to prices.
7. Store predictions in SQLite.
8. On subsequent runs, actual next-day OHLC can be attached to pending predictions and errors measured.
9. The rolling 3-month window naturally moves forward as new data arrives.

## Setup

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Important

This is a research/testing system, not a guarantee of future returns. The first version deliberately keeps the feature set compact because only 3 months of data are requested. Fundamental data integration, walk-forward backtesting, confidence calibration, champion/challenger retraining and automated post-market evaluation are planned for the next version.
