"""Stage 28 command-center contract.

Stage 28 is a superset layer: existing Stage 1-10.5 engines remain the
calculation engines, while this module provides one versioned capability
contract, validation gate, and machine-readable manifest for the final stack.
It deliberately does not fabricate predictions or bypass existing safety gates.
"""
from datetime import datetime, timezone
from typing import Any, Mapping

STAGE28_VERSION = "stage28-v1.0"

STAGES = {
    1: "Core market data and baseline prediction",
    2: "Feature engineering and model training",
    3: "Universe screening and stock selection",
    4: "Sector and market context",
    5: "Prediction and selection hardening",
    6: "Multi-horizon forecasting",
    7: "Jump and intraday intelligence",
    8: "Uncertainty, risk and adaptive scoring",
    9: "Learning, calibration and model governance",
    10: "Portfolio and decision intelligence",
    10.5: "Portfolio timing, averaging and exit intelligence",
    11: "Data intelligence and freshness",
    12: "Advanced ensemble prediction",
    13: "Multi-horizon intelligence",
    14: "Market-regime intelligence",
    15: "Cross-sectional stock ranking",
    16: "Portfolio intelligence",
    17: "Trade intelligence",
    18: "Champion-challenger learning",
    19: "Walk-forward backtesting and simulation",
    20: "Prediction calibration and uncertainty",
    21: "Event and corporate-action intelligence",
    22: "Intraday AI",
    23: "Explainable AI",
    24: "Adaptive risk engine",
    25: "Autonomous learning",
    26: "Production reliability and recovery",
    27: "Unified decision intelligence",
    28: "AI investment command center",
}

CAPABILITIES = (
    "github_only_state",
    "canonical_data_snapshot",
    "data_freshness_and_quarantine",
    "nifty150_universe",
    "technical_fundamental_scoring",
    "market_regime",
    "ensemble_prediction",
    "multi_horizon_1d_to_365d",
    "cross_sectional_ranking",
    "prediction_uncertainty",
    "risk_adjusted_return",
    "transaction_cost_and_slippage",
    "portfolio_position_intelligence",
    "averaging_and_exit_intelligence",
    "jump_watchlist",
    "intraday_watchlist",
    "ipo_intelligence",
    "prediction_actual_lineage",
    "baseline_comparison",
    "walk_forward_validation",
    "calibration",
    "champion_challenger_governance",
    "drift_detection",
    "rollback_controls",
    "report_integrity",
    "telegram_command_center",
    "duplicate_run_protection",
)


def manifest() -> dict[str, Any]:
    """Return a stable manifest consumed by reports/tests/automation."""
    return {
        "Stage": "Stage 28",
        "Version": STAGE28_VERSION,
        "GeneratedAtUTC": datetime.now(timezone.utc).isoformat(),
        "PreviousStagesIncluded": list(STAGES.keys()),
        "Capabilities": list(CAPABILITIES),
        "PredictionHorizonsDays": [1, 3, 5, 7, 10, 20, 60, 90, 180, 365],
        "Architecture": [
            "DATA", "QUALITY", "FEATURES", "REGIME", "MODEL",
            "PREDICTION", "CALIBRATION", "RISK", "RANKING", "PORTFOLIO",
            "DECISION", "EVALUATION", "LEARNING", "REPORT",
        ],
    }


def validate_contract(config: Mapping[str, Any] | None = None) -> tuple[bool, list[str]]:
    """Validate non-negotiable Stage 28 invariants without network access."""
    cfg = config or {}
    errors: list[str] = []
    horizons = tuple(cfg.get("MULTI_HORIZONS", (1, 3, 5, 7, 10, 20, 60, 90, 180, 365)))
    if not horizons or max(horizons) < 365:
        errors.append("MULTI_HORIZONS must extend through 365 days")
    if int(cfg.get("PREDICTION_TOP_N", 10)) != 10:
        errors.append("PREDICTION_TOP_N must remain 10")
    if int(cfg.get("PRESCREEN_N", 150)) < 150:
        errors.append("PRESCREEN_N must cover at least 150 stocks")
    if float(cfg.get("TRANSACTION_COST_BPS", 20)) < 0 or float(cfg.get("SLIPPAGE_BPS", 10)) < 0:
        errors.append("transaction costs and slippage cannot be negative")
    return not errors, errors


def stage_summary() -> str:
    """Compact human-readable status for Telegram/CI logs."""
    return "Stage 28 | 1-365D | Nifty 150 | Prediction→Risk→Portfolio→Decision→Learning"
