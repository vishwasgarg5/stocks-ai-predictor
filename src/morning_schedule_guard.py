from datetime import datetime, time
from zoneinfo import ZoneInfo
import os

IST = ZoneInfo("Asia/Kolkata")
START = time(8, 5)
END = time(8, 25)


def enforce_morning_window():
    if os.getenv("GITHUB_EVENT_NAME") != "schedule":
        return
    now = datetime.now(IST)
    if not (START <= now.time() <= END):
        raise SystemExit(
            f"Delayed morning schedule at {now:%Y-%m-%d %H:%M:%S} IST; "
            "outside 08:05-08:25 IST. No prediction generated or Telegram sent."
        )
