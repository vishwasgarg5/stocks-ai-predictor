from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
STAGE2_DIR = DATA_DIR / "stage2"

PREDICTIONS_DIR = STAGE2_DIR / "predictions"
EVALUATIONS_DIR = STAGE2_DIR / "evaluations"
JUMP_DIR = STAGE2_DIR / "jump"
INTRADAY_DIR = STAGE2_DIR / "intraday"
METRICS_DIR = STAGE2_DIR / "metrics"
STATE_DIR = STAGE2_DIR / "state"

for directory in [
    PREDICTIONS_DIR,
    EVALUATIONS_DIR,
    JUMP_DIR,
    INTRADAY_DIR,
    METRICS_DIR,
    STATE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# GENERAL
# ---------------------------------------------------------

STAGE_NAME = "Stage 2"
MODEL_VERSION = "stage2-v2.0"

TOP_N = 5

# Broad scan.
# Your existing Nifty-150 universe is preferred when found.
MAX_UNIVERSE = 1000

# Number of stocks receiving expensive ML training after
# liquidity/technical screening.
PRESCREEN_N = 60

HISTORY_PERIOD = "2y"

MIN_HISTORY_ROWS = 180

# Minimum average daily traded value in INR.
MIN_AVG_TRADED_VALUE = 2_000_000

# Minimum price.
MIN_PRICE = 20

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

VALIDATION_FRACTION = 0.20

# Minimum improvement required before Challenger
# becomes Champion.
MIN_RELATIVE_IMPROVEMENT = 0.02

# ---------------------------------------------------------
# 7-DAY JUMP ENGINE
# ---------------------------------------------------------

JUMP_THRESHOLD = 0.05

JUMP_HORIZON_DAYS = 7

JUMP_TOP_N = 5

# Candidates considered for jump model.
JUMP_CANDIDATE_N = 30

# Minimum model confidence.
MIN_JUMP_CONFIDENCE = 50

# ---------------------------------------------------------
# INTRADAY
# ---------------------------------------------------------

INTRADAY_TOP_N = 5

INTRADAY_PERIOD = "5d"

INTRADAY_INTERVAL = "15m"

INTRADAY_MIN_ROWS = 80

# Minimum expected intraday move.
INTRADAY_MIN_MOVE = 0.012

# ---------------------------------------------------------
# MARKET REGIME
# ---------------------------------------------------------

NIFTY_SYMBOL = "^NSEI"

# ---------------------------------------------------------
# UNIVERSE FILES
# ---------------------------------------------------------

UNIVERSE_FILES = [
    DATA_DIR / "nifty150_symbols.csv",
    DATA_DIR / "nifty150.csv",
    DATA_DIR / "nifty150_list.csv",
    DATA_DIR / "universe.csv",
]

# ---------------------------------------------------------
# METRIC FILES
# ---------------------------------------------------------

DAILY_METRICS_FILE = METRICS_DIR / "daily_metrics.csv"

STOCK_RELIABILITY_FILE = METRICS_DIR / "stock_reliability.csv"

JUMP_METRICS_FILE = METRICS_DIR / "jump_metrics.csv"

INTRADAY_METRICS_FILE = METRICS_DIR / "intraday_metrics.csv"

MODEL_STATE_FILE = STATE_DIR / "model_state.json"

# ---------------------------------------------------------
# SCHEDULE
# ---------------------------------------------------------

MORNING_HOUR = 5
MORNING_MINUTE = 0

EVENING_HOUR = 16
EVENING_MINUTE = 30

WEEKLY_HOUR = 18
WEEKLY_MINUTE = 0

TIMEZONE = "Asia/Kolkata"

# ---------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------

TELEGRAM_MAX_LENGTH = 3900
