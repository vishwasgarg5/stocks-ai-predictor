from datetime import datetime, time
from zoneinfo import ZoneInfo
import os

IST = ZoneInfo("Asia/Kolkata")
START = time(4, 50)
END = time(5, 10)


def enforce_morning_window():
    """Allow scheduled morning jobs only near 05:00 IST; manual runs are allowed."""
    if os.getenv("GITHUB_EVENT_NAME") != "schedule":
        return
    now = datetime.now(IST)
    if not (START <= now.time() <= END):
        raise SystemExit(
            f"Delayed morning schedule at {now:%Y-%m-%d %H:%M:%S} IST; "
            "outside 04:50-05:10 IST. No prediction generated or Telegram sent."
        )
