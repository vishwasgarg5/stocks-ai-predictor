"""Guarded entrypoint for the scheduled Stage 1.5 morning prediction workflow."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.morning_schedule_guard import enforce_morning_window
from stage15_morning import run

if __name__ == "__main__":
    enforce_morning_window()
    run()
