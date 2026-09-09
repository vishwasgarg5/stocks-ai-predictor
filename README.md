# AI NSE Stock Predictor — Stage 28

Production-oriented NSE market prediction, ranking, portfolio decision-support and learning platform. **Stage 28 is cumulative:** it keeps the earlier Stage 1 → 10.5 engines and adds the Stage 11 → 28 architecture as a unified command-center contract.

> Predictions, rankings and portfolio decisions are research/decision-support outputs, not guaranteed returns or investment advice.

## Stage 28 stack

```text
Stage 1  → Base prediction
Stage 2  → Data / persistence / evaluation foundation
Stage 3  → Feature engineering / model validation
Stage 4  → Universe / liquidity / price-bucket selection
Stage 4.5→ Multi-horizon forecasting
Stage 5  → Actual-vs-predicted evaluation
Stage 6  → Stock / horizon reliability
Stage 7  → Market / sector / benchmark context
Stage 8  → Risk / uncertainty / cost-aware decisions
Stage 9  → Calibration / drift / champion-challenger governance
Stage 10 → Integrated decision intelligence
Stage 10.4→ Probability / adaptive risk / abstention
Stage 10.5→ Portfolio timing / averaging / exit intelligence
Stage 11 → Data intelligence and freshness
Stage 12 → Advanced ensemble prediction
Stage 13 → 1D–365D horizon intelligence
Stage 14 → Market-regime intelligence
Stage 15 → Cross-sectional stock ranking
Stage 16 → Portfolio intelligence
Stage 17 → Trade intelligence
Stage 18 → Champion-challenger learning
Stage 19 → Walk-forward backtesting / simulation
Stage 20 → Prediction calibration / uncertainty
Stage 21 → Events / corporate-action intelligence
Stage 22 → Intraday AI
Stage 23 → Explainable AI
Stage 24 → Adaptive risk engine
Stage 25 → Autonomous learning
Stage 26 → Production reliability / recovery
Stage 27 → Unified decision intelligence
Stage 28 → AI Investment Command Center
```

## Architecture

```text
NSE UNIVERSE
   ↓
CANONICAL DATA + FRESHNESS
   ↓
DATA QUALITY / QUARANTINE
   ↓
FEATURES + MARKET REGIME + SECTOR CONTEXT
   ↓
ENSEMBLE / HORIZON MODELS
   ↓
PREDICTION + UNCERTAINTY + CALIBRATION
   ↓
CROSS-SECTIONAL RANKING
   ↓
RISK + COST + BENCHMARK ANALYSIS
   ↓
PORTFOLIO POSITION ENGINE
   ↓
TRADE DECISION / ABSTENTION
   ↓
TELEGRAM COMMAND CENTER
   ↓
ACTUAL OUTCOME + BACKTEST EVALUATION
   ↓
RELIABILITY / DRIFT / CHALLENGER
   ↓
PROMOTION OR ROLLBACK
```

## Production invariants

- Nifty-150 screening is preserved.
- Morning selection remains capped at **10 final stocks** with price-bucket diversification.
- Multi-horizon forecasts extend to **365 days**: `1, 3, 5, 7, 10, 20, 60, 90, 180, 365`.
- Prediction lineage remains immutable: prediction → exact actual session → OHLC error.
- Portfolio logic remains position-aware and separate from the core prediction model.
- Averaging requires evidence; being down alone is not a buy signal.
- Transaction cost and slippage are included in decision evaluation.
- Champion/challenger and rollback gates remain mandatory.
- GitHub remains the persistent state store; no local database is introduced.
- Telegram is presentation-only; calculations happen before formatting.
- Report integrity prevents duplicate symbols, malformed OHLC and invalid TOP-N output.

## Stage 28 contract

`src/stage28.py` provides the versioned machine-readable capability manifest and CI-safe contract validation. It does not fabricate data or bypass existing prediction/risk gates.

```python
from src.stage28 import manifest, validate_contract

ok, errors = validate_contract()
print(manifest())
```

## Core modules

```text
src/config.py
src/data_snapshot.py
src/market_data.py
src/features.py
src/prediction.py
src/multihorizon.py
src/selection.py
src/stage4_engine.py
src/stage45_engine.py
src/final_intelligence.py
src/ledger.py
src/decision_ledger.py
src/portfolio_report.py
src/jump_engine.py
src/intraday_engine.py
src/ipo_runner.py
src/report_integrity.py
src/stage28.py
src/morning_runner.py
src/evening.py
src/weekly_report.py
```

## State

```text
data/stage2/predictions/
data/stage2/evaluations/
data/stage2/metrics/
data/stage2/state/
data/stage2/jump/
data/stage2/intraday/
data/stage2/ipo/
```

State is committed to GitHub rather than stored in a local database.

## Validation

Stage 28 adds a contract test alongside the existing regression, hardening, portfolio, lineage, market and report-integrity tests:

```bash
python -m pytest -q tests/
```

## Important limitation

Calling the project **Stage 28** establishes the cumulative architecture and contract; it does not mean every advanced Stage 11–28 research component has already demonstrated production superiority. New models and decision engines must earn promotion through out-of-sample validation, calibration, risk checks and rollback safeguards.

**Current version: `stage28-v1.0`**
