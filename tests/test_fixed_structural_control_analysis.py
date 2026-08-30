from scripts.analyze_fixed_structural_controls import METHODS, _validate


def _payload():
    return {
        "status": "P0_A2_FIXED_STRUCTURAL_CONTROLS_COMPLETE",
        "dataset": "tiny",
        "test": {method: {"recall@5": 0.1} for method in METHODS},
        "alignment": {"status": "BIT_EXACT_A2_INPUT_ALIGNMENT"},
        "a1_reproduction": {"maximum_absolute_difference": 0.0},
        "test_access_audit": {
            "validation_selected_A2_weights_or_rules": False,
            "all_locked_methods_reported": True,
            "test_selected_models_or_features": False,
        },
    }


def test_analysis_accepts_complete_locked_result():
    _validate(_payload(), "tiny")


def test_analysis_rejects_validation_selected_structural_rules():
    payload = _payload()
    payload["test_access_audit"]["validation_selected_A2_weights_or_rules"] = True
    try:
        _validate(payload, "tiny")
    except ValueError as exc:
        assert "test-access audit failed" in str(exc)
    else:
        raise AssertionError("Expected validation-selected A2 rules to fail")
