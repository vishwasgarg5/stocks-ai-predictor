# AI NSE Stock Predictor — Stage 10.5

Production-oriented NSE stock prediction, market intelligence and portfolio decision-support system. The repository represents a cumulative **Stage 1 → Stage 10.5** evolution, with Stage 10.5 as the current production intelligence layer.

> **Important:** Predictions and portfolio decisions are model outputs for research/decision support. They are not guaranteed returns and are not investment advice.

## 1. What the model does

The system continuously turns NSE market data into:

- Next-session **Open / High / Low / Close** predictions.
- Multi-horizon forecasts for **1 / 3 / 5 / 7 / 20 trading days**.
- Price-bucket-wise stock selection rather than a global five-stock limit.
- A separate **best pick from each price bucket**.
- Confidence, uncertainty, probability and reliability assessment.
- Market, sector, volatility and benchmark context.
- BUY / WATCH / HOLD / AVOID / NO TRADE decisions.
- Jump-watch, intraday and IPO intelligence.
- Portfolio-aware sell, hold, averaging and recovery intelligence.
- Evening outcome evaluation and learning from realised results.
- Champion/challenger and rollback protection.
- GitHub-persisted learning state without a local database.

## 2. Complete stage-wise evolution

| Stage | Purpose | Key capability | Status |
|---|---|---|---|
| **Stage 1** | Base prediction | Initial NSE stock prediction pipeline using market history and ML | ✅ Complete |
| **Stage 2** | Data & pipeline foundation | Persistent prediction/evaluation flow, OHLCV targets, scheduled morning/evening processing | ✅ Complete |
| **Stage 3** | Feature & model improvement | Technical indicators, lag/return/volatility features and stronger ML validation | ✅ Complete |
| **Stage 4** | Selection intelligence | Stock scoring, price buckets, liquidity/data-quality filtering and candidate selection | ✅ Complete |
| **Stage 4.5** | Multi-horizon intelligence | Future-only 1D/3D/5D/7D/20D forecasting and horizon evaluation | ✅ Complete |
| **Stage 5** | Evaluation & learning | Predicted-vs-actual evaluation, direction accuracy and model-performance tracking | ✅ Complete |
| **Stage 6** | Reliability intelligence | Stock-specific reliability, horizon reliability and evidence-based weighting | ✅ Complete |
| **Stage 7** | Market-context intelligence | Market regime, sector context, volatility and benchmark-relative analysis | ✅ Complete |
| **Stage 8** | Risk & decision intelligence | Uncertainty controls, downside analysis, cost/slippage awareness and safer trade decisions | ✅ Complete |
| **Stage 9** | Adaptive learning | Calibration, decision ledgers, drift/health monitoring and champion/challenger controls | ✅ Complete |
| **Stage 10** | Advanced decision layer | Integrated probability, reliability, risk, cost and outcome-aware decision framework | ✅ Complete |
| **Stage 10.4** | Adaptive Probability, Risk & Decision AI | Calibrated confidence, target-hit probability, abstention, adaptive thresholds, reliability and live-health protection | ✅ Complete |
| **Stage 10.5** | Portfolio Timing, Averaging & Exit Intelligence | Sell windows, averaging windows, recovery timing, profit targets, position-aware sizing and dynamic protection | 🚀 **Current** |

### Stage architecture

```text
Stage 1
  ↓
Stage 2
  ↓
Stage 3
  ↓
Stage 4 ── Price buckets + selection
  ↓
Stage 4.5 ── Multi-horizon forecasts
  ↓
Stages 5–9 ── Evaluation + reliability + context + risk + learning
  ↓
Stage 10 ── Integrated decision intelligence
  ↓
Stage 10.4 ── Probability + calibration + adaptive risk/decision AI
  ↓
Stage 10.5 ── Portfolio timing + averaging + exit intelligence
```

Stages are cumulative. A later stage consumes and strengthens the earlier layers; Stage 10.5 does **not** replace the underlying prediction model.

## 3. Current Stage 10.5 architecture

```text
NSE EQUITY UNIVERSE
        ↓
DATA QUALITY + LIQUIDITY FILTER
        ↓
CAUSAL FEATURES + LEAKAGE CONTROLS
        ↓
AI OHLCV PREDICTION
        ↓
MULTI-HORIZON 1D / 3D / 5D / 7D / 20D
        ↓
PRICE-BUCKET RANKING
        ↓
MARKET + SECTOR + VOLATILITY + BENCHMARK CONTEXT
        ↓
CONFIDENCE CALIBRATION
        ↓
TARGET-HIT + DOWNSIDE PROBABILITIES
        ↓
STOCK + HORIZON RELIABILITY
        ↓
UNCERTAINTY + RISK + COST ADJUSTMENT
        ↓
ADAPTIVE FINAL DECISION
        ↓
PORTFOLIO POSITION CONTEXT
        ↓
SELL / HOLD / AVG / DO NOT AVG
        ↓
SELL WINDOW + AVERAGING WINDOW + RECOVERY TIMING
        ↓
EVENING OUTCOME EVALUATION
        ↓
LEARNING + MODEL HEALTH + ROLLBACK PROTECTION
```

## 4. Prediction engine

### Core prediction

The model predicts the next available trading session's:

- Open
- High
- Low
- Close

Targets are future-session aligned to avoid using future observations as current features.

### Multi-horizon prediction

The production horizon set is:

```text
1 trading day
3 trading days
5 trading days
7 trading days
20 trading days
```

A forecast is only evaluated after its corresponding horizon has matured.

### Feature families

The pipeline uses combinations of:

- OHLCV market history
- Lagged prices
- Returns
- Volatility
- Trend indicators
- Momentum indicators
- Moving averages
- MACD
- RSI
- ADX
- Bollinger Bands
- ATR
- OBV
- Fundamental/context features where available
- Market benchmark context
- Sector-relative context

Features are designed around historical information available at prediction time.

## 5. Universe and stock selection

The current configuration supports the **full NSE equity universe** with data-quality and liquidity screening. The current run uses approximately **4 months of historical market data** to keep the production scan practical.

Important selection rules:

- No global five-stock cap.
- `TOP_N = None` in the current configuration.
- Price buckets are:

```text
>1000
500-999
100-499
50-99
10-49
```

- Qualified stocks can be retained within the scanned universe.
- The report separately identifies the **best pick in each price bucket**.
- Liquidity, minimum history and minimum price checks are applied before final AI selection.

## 6. Stage 10.4 intelligence

Stage 10.4 transformed raw predictions into risk-aware decisions.

### Probability and confidence

- Confidence calibration.
- Target-hit probabilities for multiple return levels.
- Downside probability.
- Evidence/sample-aware calibration.

### Reliability

Reliability is tracked at multiple levels:

- Stock-specific reliability.
- Horizon-specific reliability.
- Historical error / MAPE.
- Direction accuracy.
- Data quality.

### Risk controls

- Prediction uncertainty filtering.
- Volatility-aware scoring.
- Benchmark-relative expected return.
- Transaction-cost awareness.
- Slippage awareness.
- Adaptive decision thresholds.
- Explicit **NO TRADE** abstention when evidence is insufficient.

### Model governance

- Walk-forward validation.
- Champion/challenger comparison.
- Statistical promotion gates.
- Live degradation monitoring.
- Rollback protection.
- Decision-outcome ledger.

## 7. Stage 10.5 portfolio intelligence

Stage 10.5 adds a position-aware layer without changing the core stock prediction model.

### Profit target

The default portfolio objective is a **10% profit target based on the actual/estimated average purchase price**, rather than blindly using the market price.

### Sell timing

The engine evaluates the multi-horizon forecasts to estimate when a profit target may become reachable.

Possible outputs include:

- `NOW`
- A specific expected trading date.
- A date range.
- `NOT REACHED IN 20D`
- `NO AI TARGET`

### Averaging timing

Averaging is not triggered simply because a stock is down.

The engine looks for evidence of:

- Current weakness.
- A favourable future recovery signal.
- Adequate AI confidence.
- A sensible recovery path.
- Position/capital constraints.

Possible output is an **averaging window → expected recovery date** rather than an unconditional buy instruction.

### Position-aware quantity

The portfolio layer considers:

- Quantity held.
- Average price.
- Current price.
- Current return/P&L.
- AI target.
- Multi-horizon forecasts.
- Confidence.
- Recovery potential.
- Capital already committed.

Averaging capital is capped by the portfolio engine's risk control rather than allowing unlimited averaging.

### Dynamic protection

Stage 10.5 includes dynamic trailing protection and recovery triggers so that a profitable/recovering position is not treated as an unlimited upside prediction.

## 8. Morning report

The morning report is designed for Telegram/mobile consumption and includes:

1. Market overview.
2. Number of stocks scanned.
3. Data-qualified/liquid/AI-qualified counts.
4. Previous and current model accuracy.
5. Price-bucket-wise stock tables.
6. Best pick from every price bucket.
7. Current OHLCV.
8. Predicted next-session OHLCV.
9. Expected return/direction.
10. Multi-horizon outlook.
11. Confidence/reliability/risk information.
12. Jump-watch candidates.
13. Intraday candidates.
14. IPO intelligence.
15. AI Portfolio Manager decisions.
16. Sell alerts and profit-booking opportunities.
17. Averaging/recovery plans where supported by the model.

## 9. Evening evaluation and learning

After the market closes, the system compares forecasts with realised outcomes.

```text
PREDICTED OHLCV
      ↓
ACTUAL OHLCV
      ↓
ERROR / MAPE / DIRECTION
      ↓
1D / 3D / 5D / 7D / 20D MATURITY
      ↓
BUCKET + HORIZON + CONFIDENCE ANALYSIS
      ↓
DECISION OUTCOME
      ↓
RELIABILITY / LEARNING STATE
      ↓
MODEL HEALTH
      ↓
CHAMPION / CHALLENGER DECISION
```

The learning layer tracks more than a single headline accuracy number. It can analyse performance by stock, horizon, price bucket, confidence and decision outcome.

## 10. Model accuracy and health

The system tracks:

- OHLC MAPE.
- Open/High/Low/Close error.
- Direction accuracy.
- Close-error bands.
- Multi-horizon performance.
- Price-bucket performance.
- Bucket × horizon performance.
- Confidence calibration.
- Decision win rate / realised return.
- Stock reliability.
- Horizon reliability.
- Drift and live degradation.
- Champion/challenger health.

**Accuracy means closeness of the model prediction to the realised market outcome; it is not a guarantee of future returns.**

## 11. Intraday, jump and IPO modules

### Intraday

A dedicated intraday engine scans short-interval market data and scores candidates for potential intraday movement.

### Jump watch

The jump engine identifies stocks with sufficient probability, confidence and expected upside to enter a jump-watch list.

### IPO

The IPO module maintains IPO-related intelligence as a separate report component rather than mixing it into the core OHLC prediction target.

## 12. Portfolio Manager boundary

The Portfolio Manager is a **decision layer on top of the prediction engine**.

```text
CORE AI MODEL
     ↓
PREDICTIONS + PROBABILITIES + RELIABILITY
     ↓
PORTFOLIO MANAGER
     ↓
POSITION-AWARE ACTION
```

It does not retrain or redefine the core stock prediction model. This separation allows the prediction engine to be evaluated independently from portfolio-specific objectives such as averaging, recovery and profit booking.

## 13. Persistence and learning state

The project intentionally uses **GitHub as persistent state** rather than a local database.

Important state areas include:

```text
data/stage2/predictions/
data/stage2/evaluations/
data/stage2/metrics/
data/stage2/state/
data/stage2/jump/
data/stage2/intraday/
data/stage2/ipo/
```

The production workflows commit required state updates back to the repository.

## 14. Automation

Exactly **3 production workflows** are maintained:

```text
🌅 Morning   08:15 IST  Monday–Friday
🌙 Evening   16:30 IST  Monday–Friday
📊 Weekly    10:00 IST  Saturday
```

Morning is one hour before the 09:15 IST NSE cash-market open. The production setup intentionally does **not** use a separate regression-test workflow.

### Manual execution

```bash
python -m src.morning_runner
python -m src.evening
python -m src.weekly_report
```

## 15. Repository structure

```text
src/
├── config.py
├── features.py
├── market_data.py
├── models.py
├── prediction.py
├── multihorizon.py
├── selection.py
├── stage4_engine.py
├── stage45_engine.py
├── final_intelligence.py
├── evaluation.py
├── retraining.py
├── ledger.py
├── morning_runner.py
├── evening.py
├── telegram_report.py
├── weekly_report.py
└── portfolio_report.py

.github/workflows/
├── stage10_4_morning.yml
├── stage10_4_evening.yml
└── stage10_4_weekly.yml

data/stage2/
├── predictions/
├── evaluations/
├── metrics/
├── state/
├── jump/
├── intraday/
└── ipo/

portfolio_manager/
tests/
requirements.txt
README.md
```

## 16. Current production configuration

| Setting | Current value |
|---|---|
| Current stage | **10.5** |
| Model version | **stage10.5-v1.0** |
| Forecast horizons | **1 / 3 / 5 / 7 / 20 days** |
| Global stock cap | **None** |
| Price-bucket cap | **1000 per bucket** |
| Universe cap | **None / full screened NSE universe** |
| History window | **4 months** |
| Minimum history | **60 rows** |
| Minimum price | **₹10** |
| Minimum average traded value | **₹2 crore** |
| Morning | **08:15 IST** |
| Evening | **16:30 IST** |
| Weekly | **Saturday 10:00 IST** |
| Persistent state | **GitHub** |
| Local database | **None** |

## 17. Development philosophy

The model is deliberately built as a **learning and decision system**, not merely a price-regression script.

Priority order:

```text
DATA QUALITY
   ↓
LEAKAGE CONTROL
   ↓
PREDICTION QUALITY
   ↓
PROBABILITY / CONFIDENCE
   ↓
RELIABILITY
   ↓
RISK CONTROL
   ↓
DECISION QUALITY
   ↓
REAL-WORLD OUTCOME EVALUATION
   ↓
LEARNING
```

A more complex model is not automatically considered better. Changes should improve validated out-of-sample performance, calibration, decision quality or operational reliability.

## 18. Current status

**Stage 10.5 is the current production stage.**

The next improvement should be promoted only when it demonstrates measurable out-of-sample improvement and does not weaken the existing health, reliability or rollback safeguards.
