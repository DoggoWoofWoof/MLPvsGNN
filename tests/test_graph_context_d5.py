"""D5's value is entirely in two equivalences, so those are what is tested hardest.

`Z2R - Z1R` is a geometry comparison only if D5's retrieval prior is D4's to the
last bit, and `Z2R - Z2` is a prior comparison only if D5's distance group is
D3's. The prior now sits in columns 4-6, which `candidate_readout` rescales, so
the ordering of injection against normalisation is load-bearing in a way it was
not in D4.

That ordering is tested from both sides here: the equivalence is asserted, and
the normaliser is shown to be capable of moving a column, so a green result is
not just a normaliser that happens to be a no-op on everything.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mp_retrieval.complete_data import CompleteRetrievalDataset  # noqa: E402
from mp_retrieval.graph_context import NORMALISED_COLUMNS, candidate_readout  # noqa: E402
from mp_retrieval.linear_control import LOCAL_FEATURE_NAMES  # noqa: E402
from scripts.run_graph_context_d3 import DISTANCE_COLUMNS, masked_local  # noqa: E402
from scripts.run_graph_context_d4 import (  # noqa: E402
    RETRIEVAL_COLUMNS as D4_RETRIEVAL_COLUMNS,
    build_graded_retrieval_block,
)
from scripts.run_graph_context_d5 import (  # noqa: E402
    COMPLETE_STATUS,
    CONTEXT,
    D3_REUSED_FROM_D2,
    D3_TRAINED_THERE,
    D5_ARMS,
    INCREMENTS,
    NEW_ARM,
    PRIOR_COLUMNS,
    REUSED_FROM_D3,
    REUSED_FROM_D4,
    ZERO_COLUMNS,
    assert_geometry_matches_d3,
    assert_prior_matches_d4,
    build_parser,
    prior_value_columns,
    run,
)
from scripts.run_graph_context_pilot import SPLITS  # noqa: E402
from test_graph_context_d1 import args_for, dataset as build_dataset, local_block  # noqa: E402
from test_graph_context_d4 import (  # noqa: E402
    DENSE_SLICE,
    RRF_CONSTANT,
    SPLADE_SLICE,
    write_rank_lists,
)

METRICS = ("recall@1", "recall@5", "recall@20", "mrr")
NODES = 48


@pytest.fixture(scope="module")
def data() -> CompleteRetrievalDataset:
    return build_dataset()


@contextmanager
def synthetic(data):
    import scripts.run_graph_context_d2 as d2_module
    import scripts.run_graph_context_d3 as d3_module
    import scripts.run_graph_context_d4 as d4_module
    import scripts.run_graph_context_d5 as module

    static = np.linspace(0.0, 1.0, NODES * 7, dtype=np.float32).reshape(NODES, 7)
    patched = (module, d4_module, d3_module, d2_module)
    original = [(m.load_complete_dataset, m.load_or_build_static) for m in patched]
    for m in patched:
        m.load_complete_dataset = lambda *a, **k: data
        m.load_or_build_static = lambda *a, **k: (static, {"source": "synthetic"})
    try:
        yield module
    finally:
        for m, (loader, builder) in zip(patched, original, strict=True):
            m.load_complete_dataset, m.load_or_build_static = loader, builder


def d5_args(tmp_path: Path, data, d3_result: Path, d4_result: Path, **overrides):
    write_rank_lists(tmp_path, data)
    parsed = vars(args_for(tmp_path, data, **overrides))
    parsed["output"] = Path(parsed["output"]).with_name("stage_d5.json")
    parsed["d3_result"] = d3_result
    parsed["d4_result"] = d4_result
    parsed["rrf_constant"] = RRF_CONSTANT
    known = {action.dest for action in build_parser()._actions}
    return argparse.Namespace(**{k: v for k, v in parsed.items() if k in known})


@pytest.fixture(scope="module")
def chain(tmp_path_factory, data) -> tuple[Path, Path]:
    """A real D2, D3 and D4, so D5's four reused rows are genuine output."""

    import scripts.run_graph_context_d2 as d2_module
    import scripts.run_graph_context_d3 as d3_module
    import scripts.run_graph_context_d4 as d4_module
    from test_graph_context_d2 import d2_args
    from test_graph_context_d3 import d3_args
    from test_graph_context_d4 import d4_args

    with synthetic(data):
        two = d2_args(tmp_path_factory.mktemp("d5_d2"), data)
        d2_module.run(two)
        three = d3_args(tmp_path_factory.mktemp("d5_d3"), data, Path(two.output))
        d3_module.run(three)
        four = d4_args(tmp_path_factory.mktemp("d5_d4"), data, Path(three.output))
        d4_module.run(four)
    return Path(three.output), Path(four.output)


@pytest.fixture(scope="module")
def completed(tmp_path_factory, data, chain):
    with synthetic(data):
        return run(d5_args(tmp_path_factory.mktemp("d5"), data, *chain))


@pytest.fixture(scope="module")
def blocks(data):
    """Z2R and the two blocks it must agree with, built as the runner builds them."""

    local, ptr = local_block(data, CONTEXT)
    local = np.asarray(local)
    dense = np.stack([q.candidate_index.numpy()[DENSE_SLICE] for q in data.queries])
    splade = np.stack([q.candidate_index.numpy()[SPLADE_SLICE] for q in data.queries])
    values, _prior = prior_value_columns(
        data.queries, ptr, dense, splade, constant=RRF_CONSTANT, dtype=local.dtype
    )
    treatment = masked_local(local, DISTANCE_COLUMNS)
    treatment[:, list(PRIOR_COLUMNS)] = values
    d4_block, _ = build_graded_retrieval_block(
        data.queries, local, ptr, dense, splade, constant=RRF_CONSTANT
    )
    return local, ptr, dense, splade, values, treatment, d4_block


# --- the correction: the prior is not renormalised -------------------------------


def test_the_prior_sits_in_slots_the_normaliser_would_rescale():
    """Stated rather than avoided: this is exactly why the ordering is proven."""

    assert set(PRIOR_COLUMNS) <= set(NORMALISED_COLUMNS)
    assert not set(DISTANCE_COLUMNS) & set(NORMALISED_COLUMNS)


def test_the_normaliser_really_can_move_a_column():
    """Without this, the equivalence below could pass on a vacuous no-op."""

    probe = np.zeros((5, 10), dtype=np.float32)
    probe[:, PRIOR_COLUMNS[0]] = [0.5, 0.4, 0.3, 0.2, 0.1]
    rescaled = candidate_readout(probe)[:, PRIOR_COLUMNS[0]]
    assert not np.allclose(rescaled, probe[:, PRIOR_COLUMNS[0]])
    assert rescaled.max() == pytest.approx(1.0)


def test_the_prior_columns_are_bit_identical_to_d4s(blocks):
    """The invariant the whole stage rests on."""

    _local, _ptr, _dense, _splade, values, treatment, d4_block = blocks
    theirs = d4_block[:, list(D4_RETRIEVAL_COLUMNS)]
    ours = treatment[:, list(PRIOR_COLUMNS)]
    assert np.array_equal(ours, theirs)
    assert np.array_equal(ours, values)
    difference = np.abs(ours.astype(np.float64) - theirs.astype(np.float64))
    assert float(difference.max()) == 0.0


def test_the_runner_proves_the_equivalence_on_the_real_path(blocks, data):
    _local, ptr, dense, splade, values, _treatment, _d4 = blocks
    proof = assert_prior_matches_d4(
        data.queries, _local, ptr, dense, splade, values, constant=RRF_CONSTANT
    )
    assert proof["max_abs_diff"] == 0.0
    assert proof["elementwise_identical"] is True
    assert proof["prior_lands_in_normalised_slots"] is True
    assert proof["candidate_rows_compared"] == values.shape[0]


def test_a_perturbed_prior_is_refused(blocks, data):
    _local, ptr, dense, splade, values, _treatment, _d4 = blocks
    broken = values.copy()
    broken[0, 0] = np.float16(0.5) if broken[0, 0] != np.float16(0.5) else np.float16(0.25)
    with pytest.raises(RuntimeError, match="not D4's"):
        assert_prior_matches_d4(
            data.queries, _local, ptr, dense, splade, broken, constant=RRF_CONSTANT
        )


def test_a_different_constant_is_refused(blocks, data):
    """A changed K would be a different prior, and D5 is a composition experiment."""

    _local, ptr, dense, splade, _values, _treatment, _d4 = blocks
    other, _ = prior_value_columns(
        data.queries, ptr, dense, splade, constant=10, dtype=_local.dtype
    )
    with pytest.raises(RuntimeError, match="not D4's"):
        assert_prior_matches_d4(
            data.queries, _local, ptr, dense, splade, other, constant=RRF_CONSTANT
        )


def test_the_naive_layout_would_have_been_harmless_here_but_is_not_relied_on(blocks):
    """Recorded because it is true and load-bearing for how much this fix earned.

    Each prior column's per-query maximum is exactly 1.0 -- rank 0 is always in
    `Cq`, and agreement reaches 1 whenever the two lists overlap -- so
    `candidate_readout` divides by one and a pre-normalisation injection would
    have produced the same values on this data. That is a coincidence of the
    reciprocal-rank transform, not a guarantee: a different K, a percentile
    transform, or a pool missing rank 0 would each break it. D5 proves equality
    rather than depending on it.
    """

    _local, ptr, _dense, _splade, values, _treatment, _d4 = blocks
    worst = 0.0
    for start, end in zip(ptr[:-1], ptr[1:], strict=True):
        block = np.zeros((int(end - start), 10), dtype=np.float32)
        block[:, list(PRIOR_COLUMNS)] = values[int(start) : int(end)]
        after = candidate_readout(block)
        worst = max(worst, float(np.abs(after[:, list(PRIOR_COLUMNS)] - block[:, list(PRIOR_COLUMNS)]).max()))
        assert block[:, list(PRIOR_COLUMNS)].max() == pytest.approx(1.0)
    assert worst == 0.0


# --- the geometry is D3's ---------------------------------------------------------


def test_the_distance_columns_are_d3s_distance_only(blocks):
    local, _ptr, _dense, _splade, _values, treatment, _d4 = blocks
    theirs = masked_local(local, DISTANCE_COLUMNS)
    assert np.array_equal(
        treatment[:, list(DISTANCE_COLUMNS)], theirs[:, list(DISTANCE_COLUMNS)]
    )
    proof = assert_geometry_matches_d3(treatment, local)
    assert proof["elementwise_identical"] is True
    assert proof["complete_one_hot"] is True


def test_the_remaining_columns_stay_zero(blocks):
    _local, _ptr, _dense, _splade, _values, treatment, _d4 = blocks
    assert not treatment[:, list(ZERO_COLUMNS)].any()
    assert ZERO_COLUMNS == (7, 8, 9)


def test_leaked_structure_is_refused(blocks):
    local, _ptr, _dense, _splade, _values, treatment, _d4 = blocks
    leaky = treatment.copy()
    leaky[:, 8] = local[:, 8]
    with pytest.raises(RuntimeError, match="outside its declared slots"):
        assert_geometry_matches_d3(leaky, local)


def test_perturbed_geometry_is_refused(blocks):
    """A candidate moved to the wrong distance bucket, still a valid one-hot."""

    local, _ptr, _dense, _splade, _values, treatment, _d4 = blocks
    broken = treatment.copy()
    occupied = int(np.argmax(np.asarray(broken[0, list(DISTANCE_COLUMNS)])))
    broken[0, DISTANCE_COLUMNS[occupied]] = 0
    broken[0, DISTANCE_COLUMNS[(occupied + 1) % len(DISTANCE_COLUMNS)]] = 1
    with pytest.raises(RuntimeError, match="not D3's DISTANCE_ONLY geometry"):
        assert_geometry_matches_d3(broken, local)


def test_a_broken_one_hot_is_refused(blocks):
    """Distinct from the check above: this would pass an elementwise diff of a
    block whose geometry had been rebuilt rather than copied."""

    local, _ptr, _dense, _splade, _values, treatment, _d4 = blocks
    rebuilt = masked_local(np.zeros_like(local), DISTANCE_COLUMNS)
    rebuilt[:, list(PRIOR_COLUMNS)] = treatment[:, list(PRIOR_COLUMNS)]
    with pytest.raises(RuntimeError, match="not D3's DISTANCE_ONLY geometry"):
        assert_geometry_matches_d3(rebuilt, local)
    with pytest.raises(RuntimeError, match="not a complete one-hot"):
        assert_geometry_matches_d3(rebuilt, np.zeros_like(local))


def test_z2r_is_exactly_d3_geometry_plus_d4_prior_and_nothing_else(blocks):
    """The composition claim, stated as a single reconstruction."""

    local, _ptr, _dense, _splade, _values, treatment, d4_block = blocks
    rebuilt = np.zeros_like(treatment)
    rebuilt[:, list(DISTANCE_COLUMNS)] = masked_local(local, DISTANCE_COLUMNS)[
        :, list(DISTANCE_COLUMNS)
    ]
    rebuilt[:, list(PRIOR_COLUMNS)] = d4_block[:, list(D4_RETRIEVAL_COLUMNS)]
    assert np.array_equal(rebuilt, treatment)


# --- five rows, one run -----------------------------------------------------------


def test_the_table_is_the_declared_five_arms(completed):
    assert D5_ARMS == (
        "SEED_ID_ONLY", "DISTANCE_ONLY", "FULL_LOCAL",
        "SEED_PLUS_GRADED_RETRIEVAL", "DISTANCE_PLUS_GRADED_RETRIEVAL",
    )
    assert set(completed["ladder"]) == set(D5_ARMS)
    assert completed["arms_trained_here"] == [NEW_ARM]
    assert completed["reuse"]["new_runs"] == 1


def test_only_the_new_arm_was_trained(completed):
    for arm in (*REUSED_FROM_D3, *REUSED_FROM_D4):
        assert completed["results"][arm]["retrained_here"] is False, arm
    assert completed["results"][NEW_ARM]["retrained_here"] is True
    for arm in D3_TRAINED_THERE:
        assert completed["results"][arm]["measured_in"] == "stage_d3", arm
    for arm in REUSED_FROM_D4:
        assert completed["results"][arm]["measured_in"] == "stage_d4", arm


def test_the_two_hop_row_still_points_at_where_it_was_measured(completed):
    """D3 did not train FULL_LOCAL either; it reused D2's FULL_CAND.

    Relabelling it `stage_d3` here would launder a two-hop provenance into a
    one-hop one, and the ten-column reference row is the one most likely to be
    quoted later.
    """

    assert D3_REUSED_FROM_D2 == {"FULL_LOCAL": ("stage_d2", "FULL_CAND")}
    row = completed["results"]["FULL_LOCAL"]
    assert row["measured_in"] == "stage_d2"
    assert row["measured_as"] == "FULL_CAND"
    assert row["reused_via"] == "stage_d3"
    assert row["retrained_here"] is False
    assert completed["arms"]["FULL_LOCAL"]["measured_in"] == "stage_d2"
    assert completed["reuse"]["reused_arms"]["FULL_LOCAL"] == "stage_d2 via stage_d3"


def test_the_reused_rows_are_the_earlier_stages_own_numbers(completed, chain):
    d3 = json.loads(chain[0].read_text(encoding="utf-8"))
    d4 = json.loads(chain[1].read_text(encoding="utf-8"))
    for arm in REUSED_FROM_D3:
        assert completed["ladder"][arm] == d3["ladder"][arm], arm
    for arm in REUSED_FROM_D4:
        assert completed["ladder"][arm] == d4["ladder"][arm], arm


def test_every_arm_has_the_same_parameter_count(completed):
    counts = {arm: completed["results"][arm]["parameters"] for arm in D5_ARMS}
    assert len(set(counts.values())) == 1, counts
    assert completed["ablation"]["architecture_changed"] is False


def test_the_reuse_guard_covers_both_sources(completed):
    checks = completed["reuse"]["conditions_checked"]
    assert completed["reuse"]["all_conditions_match"] is True
    assert all(block["match"] for block in checks.values())
    assert any(name.startswith("d3_") for name in checks)
    assert any(name.startswith("d4_") for name in checks)
    for required in (
        "d3_dataset", "d4_dataset", "d3_candidate_rows", "d4_candidate_rows",
        "d3_training_epochs", "d4_training_epochs", "d4_rrf_constant",
        "d4_seed_bit_not_replaced", "chain_seed_id_only_agrees",
        "d3_arms_trained_here", "d4_arms_trained_here",
        "d3_SEED_ID_ONLY_trained_there", "d3_DISTANCE_ONLY_trained_there",
        "d3_FULL_LOCAL_reused_from_d2",
    ):
        assert required in checks, required


@pytest.mark.parametrize(
    "which, field, value",
    [
        ("d3", "status", "GRAPH_CONTEXT_D3_IN_PROGRESS"),
        ("d4", "status", "GRAPH_CONTEXT_D4_IN_PROGRESS"),
        ("d3", "dataset", "some_other_dataset"),
        ("d4", "dataset", "some_other_dataset"),
        ("d3", "data_fingerprint_sha256", "f" * 64),
        ("d4", "data_fingerprint_sha256", "f" * 64),
        ("d3", "context", "TARGET_H1"),
        ("d4", "num_nodes", 1),
        ("d3", "local_feature_schema", ["a", "b"]),
        ("d3", "training.epochs", 99),
        ("d4", "training.seed", 7),
        ("d3", "splits.validation_reported", 1),
        ("d4", "splits.train_fit", 1),
        ("d3", "ablation.architecture_changed", True),
        ("d4", "a3_rank_feature_audit.constant_K", 10),
        ("d4", "ablation.seed_bit_replaced", True),
        ("d3", "arms_trained_here", ["SEED_ID_ONLY"]),
        ("d4", "arms_trained_here", ["SEED_ID_ONLY"]),
        ("d3", "seed_identity_proof.candidate_rows_compared", 1),
        ("d4", "seed_identity_proof.mismatches", 3),
        ("d3", "results.SEED_ID_ONLY.retrained_here", False),
        ("d3", "results.FULL_LOCAL.measured_in", "stage_d3"),
        ("d3", "results.FULL_LOCAL.measured_as", "FULL_LOCAL"),
        ("d3", "results.FULL_LOCAL.retrained_here", True),
    ],
)
def test_reuse_is_refused_when_a_condition_differs(tmp_path, data, chain, which, field, value):
    payloads = {
        "d3": json.loads(chain[0].read_text(encoding="utf-8")),
        "d4": json.loads(chain[1].read_text(encoding="utf-8")),
    }
    node = payloads[which]
    *parents, leaf = field.split(".")
    for key in parents:
        node = node[key]
    assert node[leaf] != value, f"{which}.{field} was already {value!r}; this proves nothing"
    node[leaf] = value

    paths = {}
    for name, payload in payloads.items():
        paths[name] = tmp_path / f"stage_{name}.json"
        paths[name].write_text(json.dumps(payload), encoding="utf-8")
    with synthetic(data):
        with pytest.raises(RuntimeError, match="cannot be reused|is not complete"):
            run(d5_args(tmp_path, data, paths["d3"], paths["d4"]))


# --- the two conditionals and the interaction -------------------------------------


def test_the_increments_are_the_declared_pair(completed):
    assert [name for name, _, _ in INCREMENTS] == [
        "delta_prior_given_geometry", "delta_geometry_given_prior"
    ]
    for name, upper, lower in INCREMENTS:
        delta = completed["increments"][name]
        assert delta["to"] == upper and delta["from"] == lower
        for metric in completed["ladder"][upper]:
            expected = completed["ladder"][upper][metric] - completed["ladder"][lower][metric]
            assert delta[metric] == pytest.approx(expected, abs=1e-12), (name, metric)


def test_the_interaction_is_the_difference_of_the_two_geometry_increments(completed):
    interaction = completed["interaction"]
    assert interaction["definition"] == "(Z2R - Z1R) - (Z2 - Z1)"
    assert interaction["descriptive_only"] is True
    for metric in completed["ladder"][NEW_ARM]:
        conditional = (
            completed["ladder"][NEW_ARM][metric]
            - completed["ladder"]["SEED_PLUS_GRADED_RETRIEVAL"][metric]
        )
        plain = (
            completed["ladder"]["DISTANCE_ONLY"][metric]
            - completed["ladder"]["SEED_ID_ONLY"][metric]
        )
        assert interaction[metric] == pytest.approx(conditional - plain, abs=1e-12), metric


def test_the_interaction_matches_d3s_distance_increment(completed, chain):
    """(Z2 - Z1) is delta_distance; the interaction is measured against it."""

    d3 = json.loads(chain[0].read_text(encoding="utf-8"))
    for metric in completed["ladder"][NEW_ARM]:
        plain = (
            completed["ladder"]["DISTANCE_ONLY"][metric]
            - completed["ladder"]["SEED_ID_ONLY"][metric]
        )
        assert plain == pytest.approx(d3["increments"]["delta_distance"][metric], abs=1e-12)


def test_both_increments_are_reported_per_stratum(completed):
    for name, upper, lower in INCREMENTS:
        rows = completed["increments_by_stratum"][name]
        assert rows, name
        for stratum, row in rows.items():
            assert row["queries"] > 0
            for metric in METRICS:
                if metric in row:
                    expected = (
                        completed["results"][upper]["validation_by_stratum"][stratum][metric]
                        - completed["results"][lower]["validation_by_stratum"][stratum][metric]
                    )
                    assert row[metric] == pytest.approx(expected, abs=1e-12), (name, stratum)


def test_the_earlier_increments_are_carried_forward(completed, chain):
    d3 = json.loads(chain[0].read_text(encoding="utf-8"))
    d4 = json.loads(chain[1].read_text(encoding="utf-8"))
    for name in ("delta_seed", "delta_distance", "delta_remaining_structure"):
        assert completed["comparators"][f"{name}_d3"] == d3["increments"][name]
    assert completed["comparators"]["delta_retrieval_quality_d4"] == (
        d4["increments"]["delta_retrieval_quality"]
    )


# --- the contract -----------------------------------------------------------------


def test_the_contract_holds(completed):
    contract = completed["contract"]
    for key in (
        "gnn_trained", "message_passing", "test_split_read",
        "epoch_selected_on_validation", "architecture_changed",
    ):
        assert contract[key] is False, key
    assert contract["scored_nodes"] == "exactly Cq in every arm"
    assert completed["splits"]["test_read"] is False
    assert completed["ablation"]["injection_is_post_normalisation"] is True
    assert completed["status"] == COMPLETE_STATUS


def test_the_test_split_is_refused_outright(tmp_path, data, chain):
    with synthetic(data):
        args = d5_args(tmp_path, data, *chain)
        args.splits = ["train", "validation", "test"]
        with pytest.raises(ValueError, match="test split is not read"):
            run(args)


@pytest.mark.parametrize("missing", [0, 1])
def test_an_absent_source_result_is_refused(tmp_path, data, chain, missing):
    paths = list(chain)
    paths[missing] = tmp_path / "nothing.json"
    with synthetic(data):
        with pytest.raises(FileNotFoundError, match="four earlier arms"):
            run(d5_args(tmp_path, data, *paths))


def test_the_operating_point_is_the_frozen_one():
    defaults = {action.dest: action.default for action in build_parser()._actions}
    assert defaults["epochs"] == 3
    assert defaults["batch_size"] == 16
    assert defaults["learning_rate"] == 0.001
    assert defaults["weight_decay"] == 0.0001
    assert defaults["seed"] == 0
    assert defaults["rrf_constant"] == 60
    assert defaults["holdout_fraction"] == 0.1


def test_the_reported_split_is_validation(completed, data):
    assert completed["splits"]["validation_reported"] == len(
        [q for q in data.queries if q.split == SPLITS["validation"]]
    )
    assert completed["column_semantics"]["0"].startswith("distance_0")
    assert LOCAL_FEATURE_NAMES[0] == "distance_0"
