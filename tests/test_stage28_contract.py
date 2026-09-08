from src.stage28 import STAGES, CAPABILITIES, manifest, validate_contract


def test_stage28_includes_all_previous_stages():
    assert 1 in STAGES and 10.5 in STAGES and 28 in STAGES
    assert len(STAGES) == 29


def test_stage28_core_capabilities_are_present():
    required = {
        "canonical_data_snapshot",
        "prediction_actual_lineage",
        "multi_horizon_1d_to_365d",
        "portfolio_position_intelligence",
        "champion_challenger_governance",
        "report_integrity",
        "telegram_command_center",
    }
    assert required.issubset(set(CAPABILITIES))


def test_stage28_contract_defaults_pass():
    ok, errors = validate_contract()
    assert ok, errors


def test_stage28_manifest_is_machine_readable():
    data = manifest()
    assert data["Stage"] == "Stage 28"
    assert data["Version"] == "stage28-v1.0"
    assert data["PredictionHorizonsDays"][-1] == 365
