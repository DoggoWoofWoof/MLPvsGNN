"""Stage D0b: the multivariate bridge diagnostic, and the harness that runs it.

D0 scored one quantity at a time. D0b asks whether the bridge signal survives
being placed beside the ordinary retrieval and static features, which is the
question that decides whether it is worth a place in QLS-v2 at all.

A comparison between feature sets is only as trustworthy as its harness, so the
central test here is a positive control: a signal planted in one column and
nowhere else must be found by the arm that carries that column and missed by
every arm that does not. If the harness could not tell those apart, no number it
produced about real features would mean anything.

The rest pin the properties the protocol depends on -- that the arms share their
base columns, that the strata partition the queries, that the added column obeys
the frozen count schema, that the reported epoch is not selected on the split it
is reported from, and that the test split is refused before any data is opened.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.run_graph_context_d0b import (  # noqa: E402
    ARMS,
    BASE_FEATURE_NAMES,
    BLOCK_WIDTH,
    BRIDGE_FEATURE_NAME,
    METRICS,
    QUESTION_ARMS,
    STRATA,
    PackedFeatures,
    arm_feature_names,
    evaluate,
    log1p_candidate_normalised,
    query_stratum,
    run,
    train_arm,
)

BLOCKS = dict(ARMS)


def packed(count, *, seed, separation, degrees=None):
    """A synthetic split whose only signal sits in the bridge column.

    Every other block is noise drawn from the same distribution for positives
    and negatives, so an arm without the bridge column has nothing to learn and
    an arm with it has one thing to learn.
    """
    rng = np.random.default_rng(seed)
    split = PackedFeatures()
    for query in range(count):
        size = int(rng.integers(6, 12))
        ids = np.sort(rng.choice(400, size=size, replace=False)).astype(np.int64)
        positive = np.zeros(size, dtype=bool)
        positive[rng.integers(0, size)] = True
        bridge = rng.random((size, 1)).astype(np.float32)
        bridge[positive, 0] += separation
        degree = (
            rng.integers(0, 8, size=size)
            if degrees is None
            else np.asarray(degrees, dtype=np.int64)
        )
        split.append(
            base=rng.random((size, 9)).astype(np.float32),
            local_cand=rng.random((size, 10)).astype(np.float32),
            local_h1=rng.random((size, 10)).astype(np.float32),
            bridge=bridge,
            ids=ids,
            positive=positive,
            golds=ids[positive],
            stratum=query_stratum(degree, positive),
            query_index=query,
        )
    split.finalise()
    return split


@pytest.fixture(scope="module")
def trained():
    train, validation = packed(300, seed=0, separation=3.0), packed(120, seed=1, separation=3.0)
    return {
        name: train_arm(blocks, train, validation, seed=0, learning_rate=0.05)
        for name, blocks in ARMS
    }


# --- the positive control -------------------------------------------------


def test_the_arm_holding_the_planted_column_finds_the_planted_signal(trained):
    """The harness can detect a feature that works. Everything else rests on it."""
    assert trained["TARGET_H1_BRIDGE"]["validation"]["overall"]["recall@1"] > 0.5


@pytest.mark.parametrize("arm", ("RETRIEVAL_ONLY", "CAND", "TARGET_H1"))
def test_an_arm_without_the_planted_column_does_not_find_it(trained, arm):
    """...and cannot detect one it was not given, so a win is attributable.

    Without this the first test would be satisfied by a harness that leaked the
    label into every arm.
    """
    assert trained[arm]["validation"]["overall"]["recall@1"] < 0.3
    assert (
        trained[arm]["validation"]["overall"]["recall@1"]
        < trained["TARGET_H1_BRIDGE"]["validation"]["overall"]["recall@1"]
    )


# --- the schema -----------------------------------------------------------


@pytest.mark.parametrize(
    "arm,width", (("RETRIEVAL_ONLY", 9), ("CAND", 19), ("TARGET_H1", 19), ("TARGET_H1_BRIDGE", 20))
)
def test_each_arm_has_the_parameter_count_its_schema_implies(trained, arm, width):
    """19 for the two context arms: the frozen A3 count, not a coincidence."""
    assert trained[arm]["parameters"] == width
    assert len(arm_feature_names(BLOCKS[arm])) == width
    assert len(trained[arm]["weights"]) == width


@pytest.mark.parametrize("arm", [name for name, _ in ARMS])
def test_every_arm_starts_from_the_same_nine_base_columns(arm):
    """The arms differ by what is appended, never by what is replaced."""
    assert arm_feature_names(BLOCKS[arm])[:9] == list(BASE_FEATURE_NAMES)


def test_the_two_context_arms_differ_only_in_the_context_their_block_came_from():
    cand = arm_feature_names(BLOCKS["CAND"])
    target = arm_feature_names(BLOCKS["TARGET_H1"])
    assert [name.removesuffix("__cand") for name in cand] == [
        name.removesuffix("__h1") for name in target
    ]


def test_the_bridge_arm_adds_exactly_one_column_to_the_target_arm():
    target = arm_feature_names(BLOCKS["TARGET_H1"])
    bridge = arm_feature_names(BLOCKS["TARGET_H1_BRIDGE"])
    assert bridge[:-1] == target
    assert bridge[-1] == BRIDGE_FEATURE_NAME


def test_the_floor_arm_is_not_one_of_the_arms_the_question_is_about():
    """`RETRIEVAL_ONLY` interprets the others; it is not a fourth design."""
    assert "RETRIEVAL_ONLY" not in QUESTION_ARMS
    assert set(QUESTION_ARMS) | {"RETRIEVAL_ONLY"} == {name for name, _ in ARMS}


def test_the_block_widths_agree_with_the_names_they_generate():
    for block, width in BLOCK_WIDTH.items():
        assert len(arm_feature_names((block,))) - len(BASE_FEATURE_NAMES) == width


# --- the added column's shape --------------------------------------------


def test_the_bridge_column_follows_the_frozen_count_schema():
    """log1p, then divide by the maximum over the scored candidates.

    Matching the kernel's own treatment of a count is what makes the arm a test
    of the quantity rather than of its units.
    """
    values = np.array([0.0, 1.0, 3.0, 7.0])
    out = log1p_candidate_normalised(values)
    assert out.max() == pytest.approx(1.0)
    assert out[0] == 0.0
    assert out[1] == pytest.approx(np.log1p(1) / np.log1p(7), abs=1e-6)


def test_a_bridge_column_that_is_zero_everywhere_stays_zero():
    """The kernel's guard, kept: an absent quantity must not become a NaN."""
    out = log1p_candidate_normalised(np.zeros(5))
    assert np.array_equal(out, np.zeros(5, dtype=np.float32))
    assert np.isfinite(out).all()


def test_the_bridge_column_is_normalised_over_the_candidates_it_is_read_on():
    """Candidate-readout, for the one column the frozen kernel does not produce.

    The whole point of Section 1's fix is that the normalising population is the
    scored population. A column added beside the frozen ten has to obey the same
    rule or it reintroduces the confound the fix removed.
    """
    assert log1p_candidate_normalised(np.array([2.0, 4.0])).max() == pytest.approx(1.0)
    assert log1p_candidate_normalised(np.array([2.0, 4.0, 9.0])).max() == pytest.approx(1.0)


# --- the strata -----------------------------------------------------------


def test_the_strata_partition_the_queries(trained):
    counts = trained["CAND"]["validation"]["by_gold_stratum"]
    assert sum(counts[name]["queries"] for name in STRATA) == 120


def test_a_query_takes_the_stratum_of_its_hardest_gold():
    """Minimum, not maximum. A query holding one stranded gold is a query the
    repair could help, and calling it `ordinary` would hide it."""
    degree = np.array([0, 9, 9])
    positive = np.array([True, True, False])
    assert query_stratum(degree, positive) == "isolated"


def test_a_gold_that_is_well_connected_does_not_rescue_a_stranded_one():
    degree = np.array([3, 40])
    assert query_stratum(degree, np.array([True, True])) == "low_degree"
    assert query_stratum(degree, np.array([False, True])) == "ordinary"


def test_a_query_with_no_in_pool_gold_gets_its_own_bucket():
    """Not folded into `isolated`. "The gold is not here" and "the gold is here
    with no neighbours" are different findings and D0 kept them apart."""
    assert query_stratum(np.array([0, 1, 2]), np.zeros(3, dtype=bool)) == "no_gold_in_pool"


def test_an_out_of_range_degree_is_an_error_not_a_silent_bucket():
    with pytest.raises(ValueError):
        query_stratum(np.array([-1]), np.array([True]))


# --- the protocol ---------------------------------------------------------


def test_the_reported_numbers_are_the_final_epoch_and_not_a_selected_one(trained):
    """No checkpoint is chosen on the split the result is read from.

    A3 selects its epoch on validation and reports on test. D0b has no test
    split, so selecting on validation and reporting on validation would be
    reading the same number twice. Three epochs are run and the third is
    reported; the trajectory is kept so a reader can see whether another epoch
    would have reordered the arms.
    """
    for arm in trained:
        history = trained[arm]["history"]
        assert [row["epoch"] for row in history] == [1, 2, 3]
        for metric in METRICS:
            assert history[-1]["validation"][metric] == pytest.approx(
                trained[arm]["validation"]["overall"][metric]
            )


def test_the_learning_rate_is_carried_not_chosen(trained):
    """Every arm is fit at the same rate, the one A3's protocol already selected.

    An arm allowed its own rate would be a small architecture search, which is
    exactly what this stage was told not to do.
    """
    assert {trained[arm]["learning_rate"] for arm in trained} == {0.05}


def test_weights_start_at_zero_so_the_arms_differ_by_features_alone():
    train, validation = packed(40, seed=3, separation=0.0), packed(20, seed=4, separation=0.0)
    first = train_arm((), train, validation, seed=0, learning_rate=0.0)
    assert first["weights"] == [0.0] * 9


def test_two_runs_of_the_same_arm_agree():
    """Determinism, so a difference between arms is a difference in features."""
    train, validation = packed(40, seed=5, separation=1.0), packed(20, seed=6, separation=1.0)
    a = train_arm(("bridge",), train, validation, seed=0, learning_rate=0.05)
    b = train_arm(("bridge",), train, validation, seed=0, learning_rate=0.05)
    assert a["weights"] == b["weights"]


def test_only_queries_with_an_in_pool_gold_are_fit():
    """The frozen loss requires a positive in every packed query; A3 excludes the
    rest from the loss rather than fabricating one."""
    train, validation = packed(40, seed=7, separation=1.0), packed(20, seed=8, separation=1.0)
    result = train_arm(("bridge",), train, validation, seed=0, learning_rate=0.05)
    assert result["train_queries_with_in_pool_gold"] <= result["train_queries"]
    assert result["train_queries"] == 40


# --- what the runner refuses ---------------------------------------------


def _args(splits):
    return argparse.Namespace(splits=splits)


def test_the_test_split_is_refused_before_anything_is_opened():
    with pytest.raises(ValueError, match="test split is not read"):
        run(_args(["train", "validation", "test"]))


def test_a_run_without_a_training_split_is_refused():
    with pytest.raises(ValueError, match="both are required"):
        run(_args(["validation"]))


def test_a_run_without_the_reporting_split_is_refused():
    with pytest.raises(ValueError, match="both are required"):
        run(_args(["train"]))


# --- metrics --------------------------------------------------------------


def test_ranking_ties_break_on_ascending_global_id():
    """The convention every sealed result in this project uses, so a number here
    is comparable with one from A3 rather than merely similar to it."""
    split = PackedFeatures()
    split.append(
        base=np.zeros((3, 9), dtype=np.float32),
        local_cand=np.zeros((3, 10), dtype=np.float32),
        local_h1=np.zeros((3, 10), dtype=np.float32),
        bridge=np.zeros((3, 1), dtype=np.float32),
        ids=np.array([11, 22, 33], dtype=np.int64),
        positive=np.array([False, False, True]),
        golds=np.array([33], dtype=np.int64),
        stratum="ordinary",
        query_index=0,
    )
    split.finalise()
    # Every score is zero, so the whole ranking is decided by the tie-break.
    metrics = evaluate(np.zeros(9, dtype=np.float32), split, ())
    assert metrics["overall"]["recall@1"] == 0.0
    assert metrics["overall"]["mrr"] == pytest.approx(1.0 / 3.0)


def test_recall_counts_golds_that_are_absent_from_the_pool():
    """R@k divides by all golds, not by the in-pool ones, matching the sealed
    semantics. A stage that quietly switched to the conditional form would look
    like an improvement over A3 that was not one."""
    split = PackedFeatures()
    split.append(
        base=np.zeros((2, 9), dtype=np.float32),
        local_cand=np.zeros((2, 10), dtype=np.float32),
        local_h1=np.zeros((2, 10), dtype=np.float32),
        bridge=np.zeros((2, 1), dtype=np.float32),
        ids=np.array([5, 6], dtype=np.int64),
        positive=np.array([True, False]),
        golds=np.array([5, 999], dtype=np.int64),
        stratum="ordinary",
        query_index=0,
    )
    split.finalise()
    metrics = evaluate(np.zeros(9, dtype=np.float32), split, ())
    assert metrics["overall"]["recall@1"] == pytest.approx(0.5)
    assert metrics["overall"]["full_coverage@20"] == 0.0


# --- the launcher ---------------------------------------------------------


def _launcher():
    return pytest.importorskip(
        "scripts.modal_graph_context_pilot", reason="modal is not installed here"
    )


def test_the_launcher_gives_d0b_both_development_splits_and_neither_more():
    module = _launcher()
    args = module._d0b_runner_args(module._jobs(["2wiki_clean"], "stage_d0b", 0)[0])
    assert args.splits == ["train", "validation"]
    assert "test" not in args.splits


def test_the_launcher_carries_a3s_selected_learning_rate_rather_than_choosing_one():
    """The rate is read off the sealed A3 result, so no arm can be given its own.

    A per-arm rate would be a small architecture search wearing a different name.
    """
    import json

    module = _launcher()
    sealed = json.loads(
        (module.HOST_REPO_ROOT / "outputs" / "p0_linear_rank_structure" / "2wiki_clean.json")
        .read_text(encoding="utf-8")
    )
    args = module._d0b_runner_args(module._jobs(["2wiki_clean"], "stage_d0b", 0)[0])
    assert args.learning_rate == float(sealed["selected_learning_rate"])


def test_a_stage_that_trains_nothing_is_not_blocked_by_a_missing_a3_result():
    """Only D0b needs the sealed linear control, so only D0b may fail without it."""
    module = _launcher()
    assert module._selected_learning_rate("a_dataset_with_no_a3_result", "stage_c") is None
    with pytest.raises(FileNotFoundError):
        module._selected_learning_rate("a_dataset_with_no_a3_result", "stage_d0b")


def test_the_launcher_gives_d0b_the_sealed_static_features():
    module = _launcher()
    args = module._d0b_runner_args(module._jobs(["2wiki_clean"], "stage_d0b", 0)[0])
    assert args.feature_cache.name == "fixed_structural_features_v1"
    assert args.rrf_constant == 60


def test_the_window_the_gate_reads_is_the_window_modal_will_enforce():
    """A per-stage timeout resolved at import time has to reach the gate.

    Gating D0b against the shared one-hour default would refuse a job that fits
    the two-hour window its function was actually decorated with.
    """
    module = _launcher()
    assert module.TIMEOUT_SECONDS == int(
        module.MODAL_CONFIG.get("stage_timeout_seconds", {}).get(
            module.STAGE, module.MODAL_CONFIG["timeout_seconds"]
        )
    )


def test_d0bs_cost_model_does_not_collapse_to_the_load_term():
    """`query_cap` is 0 for a whole-split stage, and the shared per-query model
    multiplies by it. Left alone, the gate would wave through a job of any size."""
    from scripts.spawn_modal_jobs import (
        GRAPH_CONTEXT_LOAD_SECONDS,
        _graph_context_d0b_seconds,
    )

    module = _launcher()
    job = module._jobs(["2wiki_clean"], "stage_d0b", 0)[0]
    seconds = _graph_context_d0b_seconds(module, job)
    assert seconds > GRAPH_CONTEXT_LOAD_SECONDS * 2
    assert seconds < module.TIMEOUT_SECONDS


def test_the_declared_ceiling_is_not_below_what_the_gate_projects():
    """A declaration the launcher's own model already exceeds is not a ceiling."""
    import yaml

    module = _launcher()
    from scripts.spawn_modal_jobs import gate_launch

    declared = yaml.safe_load(module.CONFIG_PATH.read_text(encoding="utf-8"))["stages"]["D0B"]
    report = gate_launch(
        "graph-context", module, module._jobs(["2wiki_clean"], "stage_d0b", 0)
    )
    assert report["total_hours"] <= declared["projected_cost"]["cpu_hours_ceiling"]
    assert report["expected_spend_usd"] <= declared["projected_cost"]["cost_ceiling_usd"]
