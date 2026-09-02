"""D0c is exploratory, and these tests are what keeps that honest.

A stage with a post-hoc threshold in it is one careless edit away from being
cited as confirmation. So the tests here check two different kinds of thing: the
mechanics (does the conditional block actually contain what its name says, is
the epoch selected on the tail and not on validation) and the labelling (does
the result say, in the file itself, that the threshold was chosen after seeing
the strata it was chosen from).

The positive control is the one behavioural test, and it is a miniature of what
D0b actually found: the restored block helps on starved rows and *hurts* on
connected ones. Under that fixture neither unconditional arm can win, because
each is right on one population and wrong on the other, so the selective arm has
to beat both or it is not doing what its name claims. An earlier version planted
signal only on the starved rows, which made `SELECTIVE_SUBSTITUTE` and
`TARGET_H1` see identical data and could not have failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.run_graph_context_d0b import PackedFeatures, query_stratum  # noqa: E402
from scripts.run_graph_context_d0c import (  # noqa: E402
    BLOCK_WIDTH,
    CONFIRMATORY_ARMS,
    D0C_ARMS,
    EXPLORATORY_ARMS,
    SELECTIVE_DEGREE_THRESHOLD,
    STRATUM_NAMES,
    attach_selective_blocks,
    build_parser,
    candidate_degrees,
    delta_by_degree_stratum,
    fit_arm,
    monotone,
    run,
)

ARM_NAMES = tuple(name for name, _ in D0C_ARMS)


def packed(count, *, seed, separation):
    """A split that reproduces D0b's finding in miniature.

    Where the candidate had little in-pool structure, the restored block carries
    the signal and the historical block is noise. Where it had plenty, the
    historical block carries the signal and the restored block carries an
    *anti*-signal -- it lifts the negatives. That is the shape D0b measured:
    restored context helps the starved and actively misleads on the connected.

    An arm reading one block everywhere therefore wins on half the queries and
    loses on the other half; only an arm that switches on the same condition
    gets both. Queries come in all-starved, all-connected and mixed kinds, so the
    row-wise conditioning is exercised rather than assumed.
    """
    rng = np.random.default_rng(seed)
    split = PackedFeatures()
    degrees = []
    for query in range(count):
        size = int(rng.integers(8, 14))
        ids = np.sort(rng.choice(400, size=size, replace=False)).astype(np.int64)
        positive = np.zeros(size, dtype=bool)
        positive[rng.integers(0, size)] = True
        kind = query % 3
        if kind == 0:
            degree = rng.integers(0, SELECTIVE_DEGREE_THRESHOLD, size=size)
        elif kind == 1:
            degree = rng.integers(SELECTIVE_DEGREE_THRESHOLD, 8, size=size)
        else:
            degree = np.where(
                np.arange(size) % 2 == 0,
                rng.integers(0, SELECTIVE_DEGREE_THRESHOLD, size=size),
                rng.integers(SELECTIVE_DEGREE_THRESHOLD, 8, size=size),
            )
        degree = degree.astype(np.int64)
        starved = degree < SELECTIVE_DEGREE_THRESHOLD

        local_cand = rng.random((size, 10)).astype(np.float32)
        local_h1 = rng.random((size, 10)).astype(np.float32)
        local_h1[starved & positive, 0] += separation
        local_cand[~starved & positive, 0] += separation
        local_h1[~starved & ~positive, 0] += separation

        degrees.append(degree)
        split.append(
            base=rng.random((size, 9)).astype(np.float32),
            local_cand=local_cand,
            local_h1=local_h1,
            bridge=rng.random((size, 1)).astype(np.float32),
            ids=ids,
            positive=positive,
            golds=ids[positive],
            stratum=query_stratum(degree, positive),
            query_index=query,
        )
    split.finalise()
    degree = np.concatenate(degrees)
    attach_selective_blocks(split, degree, SELECTIVE_DEGREE_THRESHOLD)
    return split, degree


# --- the conditional blocks contain what their names say -----------------------


def test_the_substitute_block_takes_h1_only_where_evidence_was_scarce():
    split, degree = packed(40, seed=0, separation=2.0)
    starved = degree < SELECTIVE_DEGREE_THRESHOLD
    assert np.array_equal(split.local_substitute[starved], split.local_h1[starved])
    assert np.array_equal(split.local_substitute[~starved], split.local_cand[~starved])


def test_the_mask_block_leaves_the_connected_rows_empty():
    split, degree = packed(40, seed=0, separation=2.0)
    starved = degree < SELECTIVE_DEGREE_THRESHOLD
    assert np.array_equal(split.local_mask[starved], split.local_h1[starved])
    assert not split.local_mask[~starved].any()


def test_the_condition_is_per_candidate_not_per_query():
    """A connected candidate in a starved query keeps its historical context."""

    split, degree = packed(40, seed=1, separation=2.0)
    rows_per_query = np.diff(split.ptr)
    mixed = [
        q
        for q in range(split.query_count)
        if len(
            set(
                (degree[int(split.ptr[q]) : int(split.ptr[q + 1])] < SELECTIVE_DEGREE_THRESHOLD)
            )
        ) == 2
    ]
    assert mixed, "the fixture must contain a query spanning the threshold"
    assert rows_per_query.sum() == degree.size


def test_a_degree_vector_that_does_not_cover_the_rows_is_an_error():
    split, degree = packed(10, seed=2, separation=1.0)
    with pytest.raises(ValueError):
        attach_selective_blocks(split, degree[:-1], SELECTIVE_DEGREE_THRESHOLD)


def test_every_arm_has_the_width_its_blocks_imply():
    for name, blocks in D0C_ARMS:
        assert sum(BLOCK_WIDTH[block] for block in blocks) + 9 == {
            "RETRIEVAL_ONLY": 9, "CAND": 19, "TARGET_H1": 19, "BOTH": 29,
            "SELECTIVE_SUBSTITUTE": 19, "SELECTIVE_MASK": 19,
        }[name]


# --- the positive control ------------------------------------------------------


@pytest.fixture(scope="module")
def trained():
    train, _ = packed(400, seed=0, separation=6.0)
    tail, _ = packed(120, seed=1, separation=6.0)
    validation, _ = packed(160, seed=2, separation=6.0)
    return {
        name: fit_arm(
            blocks, train, tail, validation, seed=0, learning_rate=0.05, epochs=4
        )
        for name, blocks in D0C_ARMS
    }


def test_the_conditional_arm_recovers_signal_that_is_split_across_both_blocks(trained):
    assert trained["SELECTIVE_SUBSTITUTE"]["validation"]["overall"]["recall@1"] > 0.6


def test_the_floor_arm_has_nothing_to_find(trained):
    assert trained["RETRIEVAL_ONLY"]["validation"]["overall"]["recall@1"] < 0.3


@pytest.mark.parametrize("arm", ["CAND", "TARGET_H1"])
def test_reading_one_block_everywhere_is_weaker_than_switching(trained, arm):
    """The hypothesis in miniature: neither context is right for every candidate."""

    assert (
        trained["SELECTIVE_SUBSTITUTE"]["validation"]["overall"]["recall@1"]
        > trained[arm]["validation"]["overall"]["recall@1"]
    ), arm


def test_masking_the_connected_rows_discards_what_substitution_keeps(trained):
    """Why both selective forms are run: they are not the same ablation."""

    assert (
        trained["SELECTIVE_SUBSTITUTE"]["validation"]["overall"]["recall@1"]
        > trained["SELECTIVE_MASK"]["validation"]["overall"]["recall@1"]
    )


# --- epoch selection ------------------------------------------------------------


def test_the_selected_epoch_is_the_tails_argmax(trained):
    for arm in ARM_NAMES:
        history = trained[arm]["history"]
        recalls = [entry["train_tail"]["recall@5"] for entry in history]
        assert trained[arm]["selected_epoch"] == int(np.argmax(recalls)) + 1, arm


def test_the_trajectory_is_reported_on_the_tail_and_never_on_validation(trained):
    for arm in ARM_NAMES:
        for entry in trained[arm]["history"]:
            assert set(entry) == {"epoch", "mean_train_loss", "train_tail"}


def test_validation_is_read_once_per_arm(trained, monkeypatch):
    """Counted, not asserted: the reporting split must be touched exactly once."""

    import scripts.run_graph_context_d0c as module

    train, _ = packed(60, seed=3, separation=2.0)
    tail, _ = packed(30, seed=4, separation=2.0)
    validation, _ = packed(30, seed=5, separation=2.0)
    seen: list[int] = []
    original = module.evaluate

    def counting(weight, packed_split, blocks):
        if packed_split is validation:
            seen.append(1)
        return original(weight, packed_split, blocks)

    monkeypatch.setattr(module, "evaluate", counting)
    module.fit_arm(
        ("local_cand",), train, tail, validation, seed=0, learning_rate=0.05, epochs=5
    )
    assert sum(seen) == 1


def test_the_train_objective_is_recorded_for_every_epoch(trained):
    for arm in ARM_NAMES:
        losses = [entry["mean_train_loss"] for entry in trained[arm]["history"]]
        assert len(losses) == 4
        assert all(np.isfinite(losses))


# --- the monotonicity diagnostic -------------------------------------------------


def test_the_delta_is_reported_against_the_frozen_strata_in_order(trained):
    rows = delta_by_degree_stratum(trained, "TARGET_H1", "CAND")
    assert list(rows) == [name for name in STRATUM_NAMES if name in rows]
    for stratum, values in rows.items():
        assert values["recall@5"] == pytest.approx(
            trained["TARGET_H1"]["validation"]["by_gold_stratum"][stratum]["recall@5"]
            - trained["CAND"]["validation"]["by_gold_stratum"][stratum]["recall@5"]
        )


def test_monotone_reports_its_own_values_rather_than_a_bare_verdict():
    rows = {name: {"recall@5": value} for name, value in zip(STRATUM_NAMES, [3.0, 2.0, 1.0, 0.0])}
    verdict = monotone(rows, "recall@5")
    assert verdict["non_increasing"] is True
    assert verdict["values_in_stratum_order"] == [3.0, 2.0, 1.0, 0.0]
    assert verdict["first_minus_last"] == 3.0


def test_a_non_monotone_delta_is_reported_as_such():
    rows = {name: {"recall@5": value} for name, value in zip(STRATUM_NAMES, [1.0, 3.0, 0.0, 2.0])}
    assert monotone(rows, "recall@5")["non_increasing"] is False


def test_the_strata_are_the_frozen_buckets_and_not_fitted_here():
    assert STRATUM_NAMES == ("isolated", "degree_1", "low_degree", "ordinary")


# --- the labelling that keeps this exploratory ------------------------------------


def test_the_selective_arms_are_not_in_the_confirmatory_set():
    assert set(EXPLORATORY_ARMS) == {"SELECTIVE_SUBSTITUTE", "SELECTIVE_MASK"}
    assert set(CONFIRMATORY_ARMS).isdisjoint(EXPLORATORY_ARMS)
    assert set(CONFIRMATORY_ARMS) | set(EXPLORATORY_ARMS) | {"RETRIEVAL_ONLY"} == set(ARM_NAMES)


def test_the_threshold_is_named_once_rather_than_scattered():
    source = (REPO_ROOT / "scripts" / "run_graph_context_d0c.py").read_text(encoding="utf-8")
    assert source.count("SELECTIVE_DEGREE_THRESHOLD") >= 4
    assert SELECTIVE_DEGREE_THRESHOLD == 2


def test_the_module_records_that_the_threshold_was_chosen_after_the_fact():
    from scripts.run_graph_context_d0c import POST_HOC

    for phrase in ("after observing", "not preregistered", "not a QLS-v2 rule"):
        assert phrase in POST_HOC


@pytest.mark.parametrize(
    "phrase",
    [
        "EXPLORATORY",
        "not preregistered",
        "must **not** be used for untouched confirmation",
        "must **not** be frozen as part of QLS-v2",
        "no claim of universal transfer",
    ],
)
def test_the_docstring_carries_every_restriction_the_stage_runs_under(phrase):
    import scripts.run_graph_context_d0c as module

    assert phrase in module.__doc__


# --- refusals ---------------------------------------------------------------------


def _args(splits):
    return argparse.Namespace(splits=splits)


@pytest.mark.parametrize(
    "splits", [["train", "validation", "test"], ["test"], ["train"], ["validation"]]
)
def test_only_the_two_development_splits_are_accepted(splits):
    with pytest.raises(ValueError):
        run(_args(splits))


def test_the_learning_rate_must_be_supplied_rather_than_defaulted():
    """Carried from A3, never chosen here -- so there is no default to fall back on."""

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--data", ".", "--feature-cache", ".", "--expected-queries", "10",
                "--baseline", "{}", "--data-fingerprint-sha256", "0", "--output", "o.json",
            ]
        )


def test_the_runner_declares_no_gnn_and_no_bridge():
    import scripts.run_graph_context_d0c as module

    source = (REPO_ROOT / "scripts" / "run_graph_context_d0c.py").read_text(encoding="utf-8")
    assert '"bridge_feature_used": False' in source
    assert '"gnn_trained": False' in source
    assert "local_bridge" not in {block for _n, blocks in module.D0C_ARMS for block in blocks}


def test_the_arms_never_receive_the_reliability_variable_as_a_feature():
    """Exposed as an axis, not trained on. The blocks are the proof."""

    for _name, blocks in D0C_ARMS:
        assert all(block in BLOCK_WIDTH for block in blocks)
        assert "degree" not in " ".join(blocks)
