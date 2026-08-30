"""Guarded entrypoint for the scheduled Stage 1.5 morning prediction workflow."""
from src.morning_schedule_guard import enforce_morning_window
from stage15_morning import run

if __name__ == "__main__":
    enforce_morning_window()
    run()
