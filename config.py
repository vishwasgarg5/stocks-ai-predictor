from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OHLCV_DIR = DATA_DIR / "ohlcv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
NIFTY_CSV = DATA_DIR / "nifty.csv"

ROLLING_DAYS = 92
INTERVAL = "1d"
MIN_ROWS = 35
TOP_N = 5
RANDOM_STATE = 42

# Fallback NIFTY 50 universe. The data engine can refresh the list when an NSE source is available.
NIFTY50 = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK",
    "BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV","BEL","BHARTIARTL",
    "CIPLA","COALINDIA","DRREDDY","EICHERMOT","ETERNAL",
    "GRASIM","HCLTECH","HDFCBANK","HDFCLIFE","HEROMOTOCO",
    "HINDALCO","HINDUNILVR","ICICIBANK","INDUSINDBK","INFY",
    "ITC","JIOFIN","JSWSTEEL","KOTAKBANK","LT",
    "M&M","MARUTI","MAXHEALTH","NESTLEIND","NTPC",
    "ONGC","POWERGRID","RELIANCE","SBILIFE","SBIN",
    "SHRIRAMFIN","SUNPHARMA","TATACONSUM","TATAMOTORS","TATASTEEL",
    "TCS","TECHM","TITAN","TRENT","ULTRACEMCO"
]

FEATURES = [
    "ret_1d","ret_3d","ret_5d","ret_10d","ret_20d",
    "rsi_14","macd","macd_signal","ema_20_ratio","ema_50_ratio",
    "sma_20_ratio","adx_14","atr_pct","bb_width","volume_ratio",
    "nifty_ret_1d","nifty_ret_5d","relative_ret_5d","relative_ret_20d",
]

for d in (DATA_DIR, OHLCV_DIR, MODEL_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)
