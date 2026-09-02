from pathlib import Path

# =============================================================================
# CORE PATHS — GitHub-only persistence. No local database is used.
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
STAGE2_DIR = DATA_DIR / "stage2"
PREDICTIONS_DIR = STAGE2_DIR / "predictions"
EVALUATIONS_DIR = STAGE2_DIR / "evaluations"
JUMP_DIR = STAGE2_DIR / "jump"
INTRADAY_DIR = STAGE2_DIR / "intraday"
METRICS_DIR = STAGE2_DIR / "metrics"
STATE_DIR = STAGE2_DIR / "state"
for directory in [PREDICTIONS_DIR, EVALUATIONS_DIR, JUMP_DIR, INTRADAY_DIR, METRICS_DIR, STATE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STAGE 4 — Market + Sector Intelligence
# Easy access: change the stage/version here when a future stage is released.
# Stage 4 is additive to the Stage 2 prediction engine.
# =============================================================================
STAGE_NAME = "Stage 4"
MODEL_VERSION = "stage4-v4.0"
STAGE4_SECTOR_MAP_FILE = METRICS_DIR / "sector_map.csv"

TOP_N = 5

# =============================================================================
# UNIVERSE / DATA
# =============================================================================
MAX_UNIVERSE = 1000
PRESCREEN_N = 60
HISTORY_PERIOD = "2y"
MIN_HISTORY_ROWS = 180
MIN_AVG_TRADED_VALUE = 20_000_000
MIN_PRICE = 10
VALIDATION_FRACTION = 0.20
MIN_RELATIVE_IMPROVEMENT = 0.02

# =============================================================================
# STAGE 2 ENGINES — kept intact
# =============================================================================
JUMP_THRESHOLD = 0.05
JUMP_HORIZON_DAYS = 7
JUMP_TOP_N = 5
JUMP_CANDIDATE_N = 30
MIN_JUMP_CONFIDENCE = 50
INTRADAY_TOP_N = 5
INTRADAY_PERIOD = "5d"
INTRADAY_INTERVAL = "15m"
INTRADAY_MIN_ROWS = 80
INTRADAY_MIN_MOVE = 0.012
NIFTY_SYMBOL = "^NSEI"

# Broad NSE list preferred; repository lists are fallback sources only.
UNIVERSE_FILES = [
    DATA_DIR / "universe.csv",
    DATA_DIR / "nifty150_symbols.csv",
    DATA_DIR / "nifty150.csv",
    DATA_DIR / "nifty150_list.csv",
]

DAILY_METRICS_FILE = METRICS_DIR / "daily_metrics.csv"
STOCK_RELIABILITY_FILE = METRICS_DIR / "stock_reliability.csv"
JUMP_METRICS_FILE = METRICS_DIR / "jump_metrics.csv"
INTRADAY_METRICS_FILE = METRICS_DIR / "intraday_metrics.csv"
MODEL_STATE_FILE = STATE_DIR / "model_state.json"

# =============================================================================
# SCHEDULE / TELEGRAM
# =============================================================================
MORNING_HOUR = 5
MORNING_MINUTE = 0
EVENING_HOUR = 16
EVENING_MINUTE = 30
WEEKLY_HOUR = 18
WEEKLY_MINUTE = 0
TIMEZONE = "Asia/Kolkata"
TELEGRAM_MAX_LENGTH = 3900
