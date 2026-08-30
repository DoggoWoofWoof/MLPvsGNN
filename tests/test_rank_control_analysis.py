from scripts.analyze_rank_controls import _confirmation_r5, _validate_rank_result


def _rank_payload():
    method = {"candidate_ceiling": 0.75, "recall@5": 0.5}
    return {
        "status": "P0_A1_RANK_CONTROLS_COMPLETE",
        "dataset": "tiny",
        "selection": {"selected_dense_weight": 0.25},
        "test": {
            "dense": dict(method),
            "splade": dict(method),
            "equal_rrf": dict(method),
            "weighted_rrf_selected": dict(method),
        },
        "test_access_audit": {
            "weighted_test_weights_computed": [0.25, 0.5],
            "unselected_weighted_test_results_computed": False,
        },
    }


def test_rank_result_audit_accepts_only_selected_and_equal_weight():
    _validate_rank_result(_rank_payload(), "tiny")


def test_rank_result_audit_rejects_unselected_test_cells():
    payload = _rank_payload()
    payload["test_access_audit"]["weighted_test_weights_computed"].append(0.75)
    try:
        _validate_rank_result(payload, "tiny")
    except ValueError as exc:
        assert "Unexpected weighted-RRF test cells" in str(exc)
    else:
        raise AssertionError("Expected an unselected weighted test cell to fail")


def test_confirmation_mean_is_read_from_frozen_aggregate():
    payload = {"models": {"sa_mlp": {"aggregate": {"test_metrics": {"recall@5": {"mean": 0.7}}}}}}
    assert _confirmation_r5(payload, "sa_mlp") == 0.7
