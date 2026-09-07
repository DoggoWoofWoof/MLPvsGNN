"""What the M2B smoke verification has to refuse in order to be worth a gate.

``engineering_smoke_passes`` and ``parameter_accounting_matches`` are flipped by
this script alone, so the interesting question is never "does it pass on a good
result" -- almost anything does. It is what it refuses. Nearly every test below
takes a result that passes and breaks exactly one thing, because a check that
cannot fail is not a check.

Three of them are about the checker rather than the smoke:

*   the denominator. If a declared item stops matching a check, the run must
    stop rather than report "11 of 11" against a declaration that asked for 12.
*   the parameter counts. They must come from the declaration, so the script
    cannot agree with a wrong number by carrying its own copy. The test moves
    the declaration and requires the checker to move with it.
*   the vacuous pass. Two rungs that both saw an empty NODE_ROLE column agree
    perfectly, which is the shape of an accident that looks like a control.

The fixture is a synthetic result built to the runner's real shape. It is not a
recorded run: no M2B fit exists yet, and the last test says so.
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys
from typing import Any

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import m2b_smoke_verification as verifier  # noqa: E402

DECLARATION_PATH = REPO_ROOT / "configs" / "m2b_semantic_minimality.yaml"
SMOKE_RESULT_PATH = (
    REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "smoke" / "2wiki_clean.json"
)
VERIFICATION_PATH = (
    REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "smoke_verification.json"
)

PANEL = 6


@pytest.fixture(scope="module")
def declaration() -> dict[str, Any]:
    return yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))


def _digest(text: str) -> str:
    return text.encode().hex().ljust(64, "0")[:64]


def _fit(
    declaration: dict[str, Any], rung: str, *, seconds: float, reused: bool = False
) -> dict[str, Any]:
    """One fit shaped like the runner's, weighing what the declaration says.

    ``reused`` builds the S3 shape: weights loaded from M2, no epochs, zero
    seconds, and everything downstream of the model measured here anyway.
    """

    declared = declaration["semantic_candidates"][rung]
    width = int(declared["semantic_output_width"])
    training = (
        {
            "reused": True,
            "reused_from": "/vol/m2_fits/R3/qls-universal/checkpoint.pt",
            "training_seconds": 0.0,
            "history": [],
        }
        if reused
        else {"training_seconds": seconds, "history": [{"loss": 1.0}, {"loss": 0.5}]}
    )
    return {
        "dataset": "2wiki_clean",
        "regime": "R3",
        "semantic_rung": rung,
        "seed": 0,
        "reused_from_m2": reused,
        "semantic_rung_fingerprint": {
            "rung": rung,
            "class": f"{rung}Head",
            "semantic_columns": width,
            "embedding_dim": 1536,
            "projection_dim": 64 if rung == "S4" else None,
            "semantic_parameters": int(declared["semantic_trainable_parameters"]),
            "sha256": _digest(f"rung{rung}"),
        },
        # One store, so this is the same string for every rung of the cell.
        "shared_inputs_sha256": _digest("shared"),
        "parameters": {
            "semantic": int(declared["semantic_trainable_parameters"]),
            "scorer": int(declared["scorer_trainable_parameters"]),
            "total": int(declared["total_trainable_parameters"]),
        },
        "metrics": {"recall@5": 0.5},
        "training": training,
        "uncached_inference": {
            "measured_span": (
                "raw query and candidate embeddings -> semantic rung -> frozen precomputed "
                "structural block -> scorer -> score"
            ),
            "total_model_ms": {"p50": 1.0, "p95": 2.0, "p99": 3.0},
            "peak_inference_gpu_memory_mb": 12.0,
        },
        "held_out_scores": {"scored_values": PANEL * 20, "all_finite": True},
        "per_query_recall_at_5": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        "checkpoint_round_trip": {
            "loaded_strict": True,
            "max_absolute_score_difference": 0.0,
            "compared_on_query": "q0",
        },
        "systems": {
            "train_time_seconds": seconds,
            "uncached_inference_p50_ms": 1.0,
            "uncached_inference_p95_ms": 2.0,
            "uncached_inference_p99_ms": 3.0,
            "tie_break_orders_on": "uncached_inference_p95_ms",
        },
        "instrumentation": {
            field: f"<{field}>" for field in declaration["instrumentation_requirement"]["fields"]
        }
        | {
            "query_ids": PANEL,
            "aggregate_metrics_reconstructed_from_rows": True,
            "peak_vram_mb": 12.0,
            "peak_rss_mb": 900.0,
            "train_time_seconds": seconds,
        },
    }


@pytest.fixture
def result(declaration) -> dict[str, Any]:
    """A smoke result that passes every declared item, to be broken one at a time."""

    return {
        "status": "M2B_SEMANTIC_MINIMALITY_COMPLETE",
        "dataset": "2wiki_clean",
        "seed": 0,
        "features_were_loaded_not_rebuilt": "M2B never calls _cell_master_local.",
        "cells": {
            "R3": {
                "regime": "R3",
                "store_admission": {
                    "admitted": True,
                    "evidence": "reconstructed",
                    "why": "",
                    "scientific_key_differences": [],
                    "feature_build_contract_sha256": _digest("contract"),
                    "original_full_config_sha256": _digest("oldconfig"),
                },
                "shared_inputs": {
                    "cell_features_sha256": _digest("cells"),
                    "arm_store_sha256": _digest("arm"),
                    "feature_build_contract_sha256": _digest("contract"),
                    "precomputed_width": 12,
                    "held_out_query_ids_sha256": _digest("ids"),
                    "held_out_query_count": PANEL,
                    "structural_family_digests": {
                        family: _digest(family) for family in verifier.ARM_STRUCTURAL_FAMILIES
                    },
                    "candidate_contract_sha256": _digest("candidates"),
                    "candidate_id_order_sha256": _digest("order"),
                    "candidate_normalisation": "per_query_minmax",
                    "sha256": _digest("shared"),
                },
                "held_out_query_ids": [f"q{index}" for index in range(PANEL)],
                "rungs": {
                    "S2": _fit(declaration, "S2", seconds=40.0),
                    # S3 enters the smoke the way it enters the fan-out.
                    "S3": _fit(declaration, "S3", seconds=0.0, reused=True),
                    "S4": _fit(declaration, "S4", seconds=92.0),
                },
            }
        },
    }


def _built(result: dict[str, Any], tmp_path: pathlib.Path) -> dict[str, Any]:
    path = tmp_path / "2wiki_clean.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return verifier.build(path, DECLARATION_PATH)


# --------------------------------------------------------------------------
# The checker itself
# --------------------------------------------------------------------------


def test_the_declared_items_each_match_exactly_one_check(declaration):
    """Against the live declaration, not a fixture: this is the real binding."""

    items = verifier.declared_items(declaration)
    assert len(items) == 12
    assert set(items) == set(verifier.ITEM_KEYS.values())


def test_a_declared_item_with_no_check_stops_the_run(declaration, monkeypatch, tmp_path):
    """The denominator has to come from the declaration.

    Ten checks against eleven declared items would report "10 of 10" and read
    like a pass, which is exactly the arithmetic this refusal exists to stop.
    """

    keys = dict(verifier.ITEM_KEYS)
    del keys["the measured S4 fit time"]
    monkeypatch.setattr(verifier, "ITEM_KEYS", keys)
    with pytest.raises(SystemExit, match="matches no check"):
        verifier.declared_items(declaration)


def test_an_item_matching_two_checks_stops_the_run(declaration, monkeypatch):
    """An ambiguous phrase would silently settle one item with another's check."""

    keys = dict(verifier.ITEM_KEYS) | {"the": "heads_instantiate_at_the_frozen_width"}
    monkeypatch.setattr(verifier, "ITEM_KEYS", keys)
    with pytest.raises(SystemExit, match="must be settled by exactly one check"):
        verifier.declared_items(declaration)


def test_a_fixture_that_meets_the_declaration_passes_all_twelve(result, tmp_path):
    built = _built(result, tmp_path)
    assert built["verdict"] == verifier.PASSED
    assert built["failed_items"] == []
    assert built["passed_items"] == built["declared_items"] == 12


def test_the_result_must_be_the_declared_cell_and_seed(result, tmp_path):
    result["dataset"] = "hotpotqa_clean"
    with pytest.raises(SystemExit, match="the declaration smokes 2wiki_clean"):
        _built(result, tmp_path)


def test_a_seed_other_than_zero_is_refused(result, tmp_path):
    # M2B is a one-seed screen. A seed-1 result verified into the gate would
    # put a fit nobody authorised behind the fan-out.
    result["seed"] = 1
    with pytest.raises(SystemExit, match="smokes seed 0"):
        _built(result, tmp_path)


def test_a_missing_result_leaves_the_gate_false(tmp_path):
    with pytest.raises(SystemExit, match="no smoke to verify"):
        verifier.build(tmp_path / "absent.json", DECLARATION_PATH)


def test_a_missing_rung_is_refused_rather_than_scored_as_a_failure(result, tmp_path):
    # Not "S4 failed its items" -- there is no S4, and reporting 11 of 12 would
    # invite a rerun of the wrong thing.
    del result["cells"]["R3"]["rungs"]["S4"]
    with pytest.raises(SystemExit, match="no S4 fit"):
        _built(result, tmp_path)


def test_a_smoke_without_the_reused_rung_is_refused(result, tmp_path, monkeypatch):
    """Leaving S3 out would defer the reuse path to the fan-out.

    That is the arrangement the amendment corrected: fourteen cells depend on
    reuse, and a failure discovered during the fan-out costs six containers
    instead of one.
    """

    smoke = dict(yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8")))
    smoke["smoke_before_fanout"] = dict(smoke["smoke_before_fanout"])
    smoke["smoke_before_fanout"]["rungs"] = ["S2", "S4"]
    path = tmp_path / "without_s3.yaml"
    path.write_text(yaml.safe_dump(smoke), encoding="utf-8")
    result_path = tmp_path / "2wiki_clean.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(SystemExit, match="is not among them"):
        verifier.build(result_path, path)


# --------------------------------------------------------------------------
# Item 1: the heads
# --------------------------------------------------------------------------


def test_a_head_built_at_the_old_768_width_is_refused(result, tmp_path):
    """The 98,304 trap in its other form: the width, not the parameter count."""

    result["cells"]["R3"]["rungs"]["S4"]["semantic_rung_fingerprint"]["embedding_dim"] = 768
    built = _built(result, tmp_path)
    assert built["verdict"] == verifier.FAILED
    assert "heads_instantiate_at_the_frozen_width" in built["failed_items"]


def test_a_wrong_column_count_fails_the_head_item(result, tmp_path):
    result["cells"]["R3"]["rungs"]["S4"]["semantic_rung_fingerprint"]["semantic_columns"] = 256
    built = _built(result, tmp_path)
    assert "heads_instantiate_at_the_frozen_width" in built["failed_items"]


# --------------------------------------------------------------------------
# Item 2: the store
# --------------------------------------------------------------------------


def test_a_store_admitted_with_a_differing_identity_field_is_refused(result, tmp_path):
    admission = result["cells"]["R3"]["store_admission"]
    admission["scientific_key_differences"] = ["per_seed_cap"]
    built = _built(result, tmp_path)
    assert "store_loads_under_its_contract_and_is_not_rebuilt" in built["failed_items"]


def test_the_seven_identity_fields_are_the_contracts_not_a_local_count(result, tmp_path):
    """Item 2 says seven. The number is reported from the contract module."""

    built = _built(result, tmp_path)
    entry = built["items"]["store_loads_under_its_contract_and_is_not_rebuilt"]
    assert entry["identity_fields_compared"] == 7
    assert len(verifier.SCIENTIFIC_BUILD_KEY_FIELDS) == 7


def test_the_declaration_hash_drift_is_reported_and_is_not_a_refusal(result, tmp_path):
    """Amendment 2's whole point, checked here rather than only asserted there."""

    built = _built(result, tmp_path)
    entry = built["items"]["store_loads_under_its_contract_and_is_not_rebuilt"]
    assert entry["passed"] is True
    assert entry["observed"]["original_full_config_sha256"] != entry["observed"][
        "feature_build_contract_sha256"
    ]


def test_a_rebuilt_cell_master_fails_the_store_item(result, tmp_path):
    # A rebuild produces byte-identical arrays and spends the largest line in
    # the estimate to do it, so "loaded" is checked rather than inferred.
    result["features_were_loaded_not_rebuilt"] = ""
    built = _built(result, tmp_path)
    assert "store_loads_under_its_contract_and_is_not_rebuilt" in built["failed_items"]


# --------------------------------------------------------------------------
# Item 3: the controlled comparison
# --------------------------------------------------------------------------


def test_two_rungs_that_read_different_shared_inputs_are_refused(result, tmp_path):
    result["cells"]["R3"]["rungs"]["S4"]["shared_inputs_sha256"] = _digest("other")
    built = _built(result, tmp_path)
    assert "only_the_semantic_rung_differs" in built["failed_items"]


def test_two_rungs_with_the_same_semantic_fingerprint_are_refused(result, tmp_path):
    """If the rung digests agree, the two fits are the same fit twice.

    That reads as a perfect control -- identical everything -- and is instead
    the one arrangement under which the comparison measures nothing.
    """

    same = result["cells"]["R3"]["rungs"]["S2"]["semantic_rung_fingerprint"]["sha256"]
    result["cells"]["R3"]["rungs"]["S4"]["semantic_rung_fingerprint"]["sha256"] = same
    built = _built(result, tmp_path)
    assert "only_the_semantic_rung_differs" in built["failed_items"]


def test_an_empty_structural_family_digest_is_not_a_pass(result, tmp_path):
    """The vacuous agreement. Two rungs that both saw no NODE_ROLE agree.

    The digests would match, the shared input digest would match, and the item
    would pass while the arm's defining family was absent.
    """

    result["cells"]["R3"]["shared_inputs"]["structural_family_digests"]["NODE_ROLE"] = ""
    built = _built(result, tmp_path)
    entry = built["items"]["only_the_semantic_rung_differs"]
    assert entry["passed"] is False
    assert entry["families_without_a_real_digest"] == ["NODE_ROLE"]


def test_every_family_the_arm_carries_is_checked(result, tmp_path):
    built = _built(result, tmp_path)
    checked = built["structural_families_checked"]
    assert set(checked) == {"BASE", "NODE_ROLE", "SUPPORT", "PATH"}
    # GEOMETRY is not in this arm, so requiring a digest for it would refuse
    # every honest result.
    assert "GEOMETRY" not in checked


def test_the_panel_count_must_agree_with_the_panel(result, tmp_path):
    result["cells"]["R3"]["shared_inputs"]["held_out_query_count"] = PANEL + 1
    built = _built(result, tmp_path)
    assert "only_the_semantic_rung_differs" in built["failed_items"]


# --------------------------------------------------------------------------
# Items 4 and 5: the parameter accounting
# --------------------------------------------------------------------------


def test_the_counts_are_read_from_the_declaration_not_from_this_script(
    result, tmp_path, declaration
):
    """Move the declaration; the checker must move with it.

    If the counts were hard-coded here, a fit weighing what an amended
    declaration asked for would be refused, and the script would be enforcing
    its own opinion about the ladder rather than the filed one.
    """

    amended = copy.deepcopy(declaration)
    amended["semantic_candidates"]["S4"]["total_trainable_parameters"] = 205218
    path = tmp_path / "amended.yaml"
    path.write_text(yaml.safe_dump(amended), encoding="utf-8")

    smoke = tmp_path / "2wiki_clean.json"
    smoke.write_text(json.dumps(result), encoding="utf-8")

    # The unamended fixture now disagrees with the amended declaration.
    assert "s4_parameter_accounting" in verifier.build(smoke, path)["failed_items"]
    # And agrees with the real one.
    assert verifier.build(smoke, DECLARATION_PATH)["failed_items"] == []


def test_the_old_98304_count_is_refused(result, tmp_path):
    """2 * 768 * 64. The number this track's planning prose carried for months."""

    result["cells"]["R3"]["rungs"]["S4"]["parameters"]["semantic"] = 98304
    built = _built(result, tmp_path)
    entry = built["items"]["s4_parameter_accounting"]
    assert entry["passed"] is False
    assert entry["expected"]["semantic"] == 196608


def test_s2_must_report_no_semantic_parameters_at_all(result, tmp_path):
    """449 total and zero semantic is the claim the paper would rest on."""

    entry = _built(result, tmp_path)["items"]["s2_parameter_accounting"]
    assert entry["expected"] == {
        "semantic": 0,
        "scorer": 449,
        "total": 449,
        "semantic_columns": 3,
    }
    result["cells"]["R3"]["rungs"]["S2"]["parameters"]["semantic"] = 1
    assert "s2_parameter_accounting" in _built(result, tmp_path)["failed_items"]


def test_a_scorer_count_that_drifts_is_caught_even_when_the_total_is_right(
    result, tmp_path
):
    # Compensating errors: semantic down by one, scorer up by one, total
    # unchanged. Checking only the total would pass this.
    parameters = result["cells"]["R3"]["rungs"]["S4"]["parameters"]
    parameters["semantic"] -= 1
    parameters["scorer"] += 1
    built = _built(result, tmp_path)
    assert "s4_parameter_accounting" in built["failed_items"]


# --------------------------------------------------------------------------
# Items 6 to 11
# --------------------------------------------------------------------------


def test_a_checkpoint_that_does_not_reproduce_its_scores_is_refused(result, tmp_path):
    round_trip = result["cells"]["R3"]["rungs"]["S2"]["checkpoint_round_trip"]
    round_trip["max_absolute_score_difference"] = 1e-9
    built = _built(result, tmp_path)
    assert "checkpoint_round_trips" in built["failed_items"]


def test_a_non_strict_load_is_refused(result, tmp_path):
    result["cells"]["R3"]["rungs"]["S4"]["checkpoint_round_trip"]["loaded_strict"] = False
    assert "checkpoint_round_trips" in _built(result, tmp_path)["failed_items"]


def test_per_query_rows_shorter_than_the_panel_are_refused(result, tmp_path):
    """The bootstrap resamples these. A short vector would silently misalign."""

    result["cells"]["R3"]["rungs"]["S4"]["per_query_recall_at_5"] = [1.0, 0.0]
    assert "aggregate_rebuilt_from_rows" in _built(result, tmp_path)["failed_items"]


def test_an_aggregate_not_rebuilt_from_rows_is_refused(result, tmp_path):
    instrumentation = result["cells"]["R3"]["rungs"]["S2"]["instrumentation"]
    instrumentation["aggregate_metrics_reconstructed_from_rows"] = False
    assert "aggregate_rebuilt_from_rows" in _built(result, tmp_path)["failed_items"]


def test_a_nan_loss_is_refused(result, tmp_path):
    result["cells"]["R3"]["rungs"]["S4"]["training"]["history"] = [{"loss": float("nan")}]
    assert "losses_and_scores_are_finite" in _built(result, tmp_path)["failed_items"]


def test_a_fit_with_no_epochs_at_all_fails_rather_than_passing_vacuously(result, tmp_path):
    # all() over an empty list is True. A fit that trained nothing must not
    # inherit a pass from that.
    result["cells"]["R3"]["rungs"]["S2"]["training"]["history"] = []
    assert "losses_and_scores_are_finite" in _built(result, tmp_path)["failed_items"]


def test_a_non_finite_held_out_score_is_refused(result, tmp_path):
    result["cells"]["R3"]["rungs"]["S4"]["held_out_scores"]["all_finite"] = False
    assert "losses_and_scores_are_finite" in _built(result, tmp_path)["failed_items"]


def test_scoring_nothing_is_not_scoring_everything_finitely(result, tmp_path):
    result["cells"]["R3"]["rungs"]["S2"]["held_out_scores"]["scored_values"] = 0
    assert "losses_and_scores_are_finite" in _built(result, tmp_path)["failed_items"]


def test_a_null_instrumentation_field_is_named(result, tmp_path):
    result["cells"]["R3"]["rungs"]["S4"]["instrumentation"]["peak_rss_mb"] = None
    built = _built(result, tmp_path)
    entry = built["items"]["instrumentation_is_populated"]
    assert entry["passed"] is False
    assert entry["missing"]["S4"] == ["peak_rss_mb"]
    assert entry["declared_fields"] == 18


def test_a_zero_latency_percentile_is_refused(result, tmp_path):
    # Zero is what an unmeasured percentile looks like, not a fast one.
    result["cells"]["R3"]["rungs"]["S2"]["systems"]["uncached_inference_p95_ms"] = 0.0
    assert "uncached_latency_and_peak_vram" in _built(result, tmp_path)["failed_items"]


def test_the_reported_span_is_the_whole_stack(result, tmp_path):
    """Requirement 5: a benchmark starting after the projection undercharges S4."""

    span = _built(result, tmp_path)["items"]["uncached_latency_and_peak_vram"]["observed"][
        "measured_span"
    ]
    assert span.startswith("raw query and candidate embeddings")
    assert "semantic rung" in span and "scorer" in span


def test_a_reused_s4_measures_nothing_and_is_refused(result, tmp_path):
    """The failure that would look like a very cheap S4.

    A reused fit reports zero training seconds. The ratio would come out as
    0.0, the fan-out would be projected at nothing, and the 2.5x multiplier
    would have been replaced by an artifact of not having trained.
    """

    fit = result["cells"]["R3"]["rungs"]["S4"]
    fit["reused_from_m2"] = True
    fit["systems"]["train_time_seconds"] = 0.0
    fit["training"] = {"training_seconds": 0.0, "history": []}
    built = _built(result, tmp_path)
    assert "s4_fit_time_measured" in built["failed_items"]


def test_the_measured_ratio_is_reported_next_to_the_guess_it_replaces(result, tmp_path):
    observed = _built(result, tmp_path)["items"]["s4_fit_time_measured"]["observed"]
    assert observed["measured_s4_over_s2"] == pytest.approx(92.0 / 40.0)
    assert observed["the_multiplier_this_replaces"] == 2.5


def test_the_artifact_says_which_gates_it_earns_and_which_it_does_not(result, tmp_path):
    earns = _built(result, tmp_path)["what_this_earns"]
    assert "engineering_smoke_passes" in earns
    assert "parameter_accounting_matches" in earns
    # The cost gate is a separate calculation against a ceiling that has not
    # been refiled yet, and a passing smoke does not confer it.
    assert "measured_cost_within_ceiling" not in earns
    assert "cost gate is earned separately" in earns


# --------------------------------------------------------------------------
# Item 12: the reuse path
# --------------------------------------------------------------------------


def test_a_refitted_s3_is_refused(result, tmp_path):
    """The failure this item exists for, and it looks like a healthy smoke.

    Same shape, same fields, plausible numbers -- and an S3 column that is no
    longer the fit M2 filed, in a phase whose whole design is that fourteen of
    its forty-two cells come from M2 unchanged.
    """

    s3 = result["cells"]["R3"]["rungs"]["S3"]
    s3["reused_from_m2"] = False
    s3["training"] = {"reused": False, "training_seconds": 31.0, "history": [{"loss": 0.4}]}
    s3["systems"]["train_time_seconds"] = 31.0
    built = _built(result, tmp_path)
    assert built["verdict"] == verifier.FAILED
    assert "s3_is_reused_not_refitted" in built["failed_items"]


def test_zero_seconds_is_the_pass_for_the_reused_rung(result, tmp_path):
    """The opposite reading from item 11, and worth stating as a test.

    Zero seconds means "did not train", which is correct for S3 and a failure
    for S4. A checker written with `value or default` gets this backwards,
    because 0.0 is falsy -- so the passing case is asserted directly.
    """

    entry = _built(result, tmp_path)["items"]["s3_is_reused_not_refitted"]
    assert entry["passed"] is True
    assert entry["observed"]["train_time_seconds"] == 0.0
    assert entry["observed"]["training_epochs"] == 0
    assert "opposite of the S4 timing item" in entry["why_zero_seconds_is_the_pass_here"]


def test_a_reused_rung_with_training_epochs_is_refused(result, tmp_path):
    # Zero seconds and a training history at once: the seconds were not
    # recorded, and something did train.
    result["cells"]["R3"]["rungs"]["S3"]["training"]["history"] = [{"loss": 0.4}]
    assert "s3_is_reused_not_refitted" in _built(result, tmp_path)["failed_items"]


def test_a_reused_rung_must_name_the_checkpoint_it_reused(result, tmp_path):
    result["cells"]["R3"]["rungs"]["S3"]["training"]["reused_from"] = ""
    assert "s3_is_reused_not_refitted" in _built(result, tmp_path)["failed_items"]


def test_the_reused_rung_is_still_held_to_the_shared_inputs(result, tmp_path):
    """Reuse exempts S3 from training, not from the controlled comparison."""

    result["cells"]["R3"]["rungs"]["S3"]["shared_inputs_sha256"] = _digest("elsewhere")
    assert "only_the_semantic_rung_differs" in _built(result, tmp_path)["failed_items"]


def test_the_reused_rung_is_still_held_to_finite_scores(result, tmp_path):
    """It trains no epochs, so the loss half is vacuous; the score half is not.

    M2 filed S3's effectiveness, but the scores being re-measured here are this
    container's, and a NaN in them would be this container's problem.
    """

    result["cells"]["R3"]["rungs"]["S3"]["held_out_scores"]["all_finite"] = False
    assert "losses_and_scores_are_finite" in _built(result, tmp_path)["failed_items"]


def test_the_reused_rung_is_still_benchmarked_here(result, tmp_path):
    """Its latency is a property of this container, not of M2's."""

    result["cells"]["R3"]["rungs"]["S3"]["systems"]["uncached_inference_p95_ms"] = 0.0
    assert "uncached_latency_and_peak_vram" in _built(result, tmp_path)["failed_items"]


def test_the_reused_rung_still_weighs_3585(result, tmp_path):
    built = _built(result, tmp_path)
    columns = built["items"]["heads_instantiate_at_the_frozen_width"]["expected"][
        "semantic_columns"
    ]
    assert columns == {"S2": 3, "S3": 5, "S4": 258}
    result["cells"]["R3"]["rungs"]["S3"]["semantic_rung_fingerprint"]["semantic_columns"] = 4
    assert "heads_instantiate_at_the_frozen_width" in _built(result, tmp_path)["failed_items"]


# --------------------------------------------------------------------------
# Where the phase actually stands
# --------------------------------------------------------------------------


def test_the_gate_this_script_flips_is_still_false(declaration):
    gates = declaration["launch_authorization"]["gates"]
    assert gates["engineering_smoke_passes"] is False
    assert gates["parameter_accounting_matches"] is False


@pytest.mark.skipif(SMOKE_RESULT_PATH.is_file(), reason="the smoke has run")
def test_no_m2b_smoke_has_been_recorded_yet():
    """True until the real run lands, and then this skips rather than lies."""

    assert not SMOKE_RESULT_PATH.exists()
    assert not VERIFICATION_PATH.exists()
