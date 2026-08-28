"""Guarded entrypoint for the scheduled morning prediction workflow."""
from src.morning_schedule_guard import enforce_morning_window
from morning import run

if __name__ == "__main__":
    enforce_morning_window()
    run()
