import sqlite3
from datetime import datetime, timezone
from config import DB_PATH


def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            target_date TEXT,
            symbol TEXT NOT NULL,
            rank INTEGER,
            score REAL,
            base_close REAL,
            pred_open REAL, pred_high REAL, pred_low REAL, pred_close REAL,
            actual_open REAL, actual_high REAL, actual_low REAL, actual_close REAL,
            open_error REAL, high_error REAL, low_error REAL, close_error REAL,
            direction_correct INTEGER
        )""")


def save_prediction(p: dict, rank: int, score: float, target_date=None):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""INSERT INTO predictions
        (created_at,target_date,symbol,rank,score,base_close,pred_open,pred_high,pred_low,pred_close)
        VALUES (?,?,?,?,?,?,?,?,?,?)""", (
            datetime.now(timezone.utc).isoformat(), target_date, p["symbol"], rank, score,
            p["base_close"], p["pred_open"], p["pred_high"], p["pred_low"], p["pred_close"]))


def evaluate_pending(actuals: dict[str, dict]) -> int:
    updated = 0
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute("SELECT id,symbol,pred_open,pred_high,pred_low,pred_close,base_close FROM predictions WHERE actual_close IS NULL").fetchall()
        for pid, symbol, po, ph, pl, pc, base in rows:
            a = actuals.get(symbol)
            if not a:
                continue
            direction = int((pc > base) == (a["close"] > base))
            con.execute("""UPDATE predictions SET target_date=COALESCE(target_date,?), actual_open=?,actual_high=?,actual_low=?,actual_close=?, open_error=?,high_error=?,low_error=?,close_error=?,direction_correct=? WHERE id=?""",
              (a.get("date"), a["open"],a["high"],a["low"],a["close"], a["open"]-po,a["high"]-ph,a["low"]-pl,a["close"]-pc,direction,pid))
            updated += 1
    return updated
