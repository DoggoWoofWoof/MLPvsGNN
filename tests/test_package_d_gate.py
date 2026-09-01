from scripts.check_package_d_gate import _check_one

SEEDS = [0, 1, 2, 3, 4]


def _payload(**overrides: object) -> dict:
    payload = {
        "status": "CANDIDATE_BUDGET_DATASET_COMPLETE",
        "dataset": "squad_clean",
        "budget": 400,
        "comparison_contract": {"test_selected_budget": False},
        "models": {
            model: {
                "seeds": {
                    str(seed): {
                        "checkpoint_path": f"/vol/{model}_{seed}.pt",
                        "checkpoint_file_sha256": "a" * 64,
                    }
                    for seed in SEEDS
                }
            }
            for model in ("sa_mlp", "seed_aware_gnn")
        },
    }
    payload.update(overrides)
    return payload


def test_a_complete_condition_opens_its_share_of_the_gate() -> None:
    check = _check_one("squad_clean", _payload(), SEEDS)
    assert check["passes"] is True
    assert check["reasons"] == []


def test_an_absent_result_is_reported_rather_than_treated_as_complete() -> None:
    check = _check_one("squad_clean", None, SEEDS)
    assert check["passes"] is False
    assert check["present"] is False


def test_an_in_progress_condition_is_refused() -> None:
    check = _check_one(
        "squad_clean", _payload(status="CANDIDATE_BUDGET_IN_PROGRESS"), SEEDS
    )
    assert check["passes"] is False


def test_a_missing_seed_is_refused_even_when_the_status_claims_complete() -> None:
    payload = _payload()
    del payload["models"]["seed_aware_gnn"]["seeds"]["4"]
    check = _check_one("squad_clean", payload, SEEDS)
    assert check["passes"] is False
    assert any("seed_aware_gnn" in reason for reason in check["reasons"])


def test_a_seed_without_a_checkpoint_hash_is_refused() -> None:
    # Package D verifies checkpoint hashes before loading, so a condition that
    # never recorded one cannot satisfy the gate.
    payload = _payload()
    payload["models"]["sa_mlp"]["seeds"]["2"]["checkpoint_file_sha256"] = ""
    check = _check_one("squad_clean", payload, SEEDS)
    assert check["passes"] is False
    assert any("checkpoint hash" in reason for reason in check["reasons"])


def test_a_test_selected_budget_is_refused() -> None:
    check = _check_one(
        "squad_clean",
        _payload(comparison_contract={"test_selected_budget": True}),
        SEEDS,
    )
    assert check["passes"] is False


def test_a_result_filed_under_the_wrong_dataset_is_refused() -> None:
    check = _check_one("metaqa", _payload(), SEEDS)
    assert check["passes"] is False
    assert any("dataset field" in reason for reason in check["reasons"])
