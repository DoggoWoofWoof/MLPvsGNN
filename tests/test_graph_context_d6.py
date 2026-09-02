"""D6 measures an effect that may be around one R@5 point, so the confounds it
has to exclude are the ones a smaller effect would hide behind.

Two of them are structural and are tested hardest here. The arms must differ in
exactly the six columns under test and in nothing else -- not in the prior, not
in the distance group, not in width. And the widening must not re-solve the head
width, because a narrower head on a wider input would change capacity and
information together.

The third is the reason the stage exists: a one-arm design cannot represent
`FULL_LOCAL` plus the prior in ten columns without overwriting three of the six
features it means to measure. That impossibility is asserted rather than left as
prose.
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
from mp_retrieval.graph_context import NORMALISED_COLUMNS  # noqa: E402
from mp_retrieval.linear_control import LOCAL_FEATURE_NAMES  # noqa: E402
from mp_retrieval.operator_models import (  # noqa: E402
    build_explicit_feature_mlp,
    explicit_feature_input_dim,
)
from scripts.run_graph_context_d3 import DISTANCE_COLUMNS, masked_local  # noqa: E402
from scripts.run_graph_context_d4 import (  # noqa: E402
    RETRIEVAL_COLUMNS as D4_RETRIEVAL_COLUMNS,
    build_graded_retrieval_block,
)
from scripts.run_graph_context_d5 import prior_value_columns  # noqa: E402
from scripts.run_graph_context_d6 import (  # noqa: E402
    BASE_ARM,
    COMPLETE_STATUS,
    CONTEXT,
    D6_ARMS,
    D6_LOCAL_FEATURE_NAMES,
    FULL_ARM,
    HISTORICAL_COLUMNS,
    INCREMENT,
    LOCAL_DIM,
    PRIOR_COLUMNS,
    RESIDUAL_COLUMNS,
    assert_blocks_are_exact,
    assert_normalisation_unchanged,
    build_parser,
    d6_local_block,
    parameter_accounting,
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
    import scripts.run_graph_context_d5 as d5_module
    import scripts.run_graph_context_d6 as module

    static = np.linspace(0.0, 1.0, NODES * 7, dtype=np.float32).reshape(NODES, 7)
    patched = (module, d5_module, d4_module, d3_module, d2_module)
    original = [(m.load_complete_dataset, m.load_or_build_static) for m in patched]
    for m in patched:
        m.load_complete_dataset = lambda *a, **k: data
        m.load_or_build_static = lambda *a, **k: (static, {"source": "synthetic"})
    try:
        yield module
    finally:
        for m, (loader, builder) in zip(patched, original, strict=True):
            m.load_complete_dataset, m.load_or_build_static = loader, builder


def d6_args(tmp_path: Path, data, d5_result: Path, **overrides):
    write_rank_lists(tmp_path, data)
    parsed = vars(args_for(tmp_path, data, **overrides))
    parsed["output"] = Path(parsed["output"]).with_name("stage_d6.json")
    parsed["d5_result"] = d5_result
    parsed["rrf_constant"] = RRF_CONSTANT
    known = {action.dest for action in build_parser()._actions}
    return argparse.Namespace(**{k: v for k, v in parsed.items() if k in known})


@pytest.fixture(scope="module")
def d5_result(tmp_path_factory, data) -> Path:
    """A real D2, D3, D4 and D5, so the reference D6 quotes is genuine output."""

    import scripts.run_graph_context_d2 as d2_module
    import scripts.run_graph_context_d3 as d3_module
    import scripts.run_graph_context_d4 as d4_module
    import scripts.run_graph_context_d5 as d5_module
    from test_graph_context_d2 import d2_args
    from test_graph_context_d3 import d3_args
    from test_graph_context_d4 import d4_args
    from test_graph_context_d5 import d5_args

    with synthetic(data):
        two = d2_args(tmp_path_factory.mktemp("d6_d2"), data)
        d2_module.run(two)
        three = d3_args(tmp_path_factory.mktemp("d6_d3"), data, Path(two.output))
        d3_module.run(three)
        four = d4_args(tmp_path_factory.mktemp("d6_d4"), data, Path(three.output))
        d4_module.run(four)
        five = d5_args(
            tmp_path_factory.mktemp("d6_d5"), data, Path(three.output), Path(four.output)
        )
        d5_module.run(five)
    return Path(five.output)


@pytest.fixture(scope="module")
def completed(tmp_path_factory, data, d5_result):
    with synthetic(data):
        return run(d6_args(tmp_path_factory.mktemp("d6"), data, d5_result))


@pytest.fixture(scope="module")
def blocks(data):
    local, ptr = local_block(data, CONTEXT)
    local = np.asarray(local)
    dense = np.stack([q.candidate_index.numpy()[DENSE_SLICE] for q in data.queries])
    splade = np.stack([q.candidate_index.numpy()[SPLADE_SLICE] for q in data.queries])
    values, _prior = prior_value_columns(
        data.queries, ptr, dense, splade, constant=RRF_CONSTANT, dtype=local.dtype
    )
    base = d6_local_block(local, values, keep=DISTANCE_COLUMNS)
    full = d6_local_block(local, values, keep=HISTORICAL_COLUMNS)
    d4_block, _ = build_graded_retrieval_block(
        data.queries, local, ptr, dense, splade, constant=RRF_CONSTANT
    )
    return local, values, base, full, d4_block


# --- why a one-arm design is impossible --------------------------------------


def test_the_historical_block_leaves_no_room_for_the_prior():
    """The premise of the two-arm design, asserted rather than argued.

    All ten slots are occupied by `FULL_LOCAL`, so three retrieval columns
    cannot be added without displacing three historical ones -- and the three
    that a D5-style layout would displace are among the six under test.
    """

    assert len(LOCAL_FEATURE_NAMES) == 10
    assert len(HISTORICAL_COLUMNS) == 10
    assert len(PRIOR_COLUMNS) == 3
    assert LOCAL_DIM == 13
    displaced = {4, 5, 6}
    assert displaced <= set(RESIDUAL_COLUMNS)
    assert [LOCAL_FEATURE_NAMES[c] for c in sorted(displaced)] == [
        "seed_connections", "paths_length_1", "paths_length_2"
    ]


def test_the_six_columns_under_test_are_the_declared_six():
    assert [LOCAL_FEATURE_NAMES[c] for c in RESIDUAL_COLUMNS] == [
        "seed_connections",
        "paths_length_1",
        "paths_length_2",
        "paths_length_3",
        "personalized_pagerank",
        "common_out_neighbors_with_seed_neighborhood",
    ]
    assert set(RESIDUAL_COLUMNS).isdisjoint(DISTANCE_COLUMNS)
    assert set(RESIDUAL_COLUMNS) | set(DISTANCE_COLUMNS) == set(HISTORICAL_COLUMNS)


def test_the_schema_appends_rather_than_substitutes():
    assert D6_LOCAL_FEATURE_NAMES[:10] == tuple(LOCAL_FEATURE_NAMES)
    assert D6_LOCAL_FEATURE_NAMES[10:] == (
        "dense_reciprocal_rank", "splade_reciprocal_rank", "retriever_agreement"
    )


# --- the tensor invariants ----------------------------------------------------


def test_the_base_arm_is_d3_geometry_plus_the_prior(blocks):
    local, values, base, _full, _d4 = blocks
    assert base.shape[1] == LOCAL_DIM
    assert np.array_equal(
        base[:, list(DISTANCE_COLUMNS)],
        masked_local(local, DISTANCE_COLUMNS)[:, list(DISTANCE_COLUMNS)],
    )
    assert np.array_equal(base[:, list(PRIOR_COLUMNS)], values)


def test_the_full_arm_is_the_historical_block_plus_the_prior(blocks):
    local, values, _base, full, _d4 = blocks
    assert np.array_equal(full[:, list(HISTORICAL_COLUMNS)], local)
    assert np.array_equal(full[:, list(PRIOR_COLUMNS)], values)


def test_the_base_arm_holds_no_residual_structure(blocks):
    _local, _values, base, _full, _d4 = blocks
    assert not base[:, list(RESIDUAL_COLUMNS)].any()


def test_both_arms_carry_d4s_prior_exactly(blocks):
    _local, _values, base, full, d4_block = blocks
    theirs = d4_block[:, list(D4_RETRIEVAL_COLUMNS)]
    for name, block in ((BASE_ARM, base), (FULL_ARM, full)):
        ours = block[:, list(PRIOR_COLUMNS)]
        assert np.array_equal(ours, theirs), name
        difference = np.abs(ours.astype(np.float64) - theirs.astype(np.float64))
        assert float(difference.max()) == 0.0, name


def test_the_arms_differ_in_exactly_the_six_columns(blocks):
    """The single most important property of the design."""

    _local, _values, base, full, _d4 = blocks
    differing = [c for c in range(LOCAL_DIM) if not np.array_equal(base[:, c], full[:, c])]
    assert differing == list(RESIDUAL_COLUMNS)


def test_the_runner_proves_all_of_it(blocks):
    local, values, base, full, _d4 = blocks
    proof = assert_blocks_are_exact(base, full, local, values)
    assert proof["max_abs_diff"] == 0.0
    assert proof["base_residual_columns_are_zero"] is True
    assert proof["full_residual_nonzero_entries"] > 0
    assert proof["differing_column_indices"] == list(RESIDUAL_COLUMNS)
    for name, block in proof["columns"].items():
        assert block["elementwise_identical"] is True, name
        assert block["max_abs_diff"] == 0.0, name


@pytest.mark.parametrize(
    "corrupt, message",
    [
        ("base_residual", "residual structure"),
        ("base_distance", "not exact"),
        ("full_historical", "not exact"),
        ("base_prior", "not exact"),
        ("full_prior", "not exact"),
        ("full_residual_zeroed", "entirely zero"),
    ],
)
def test_a_corrupt_block_is_refused(blocks, corrupt, message):
    local, values, base, full, _d4 = blocks
    base, full = base.copy(), full.copy()
    if corrupt == "base_residual":
        base[0, RESIDUAL_COLUMNS[0]] = 1
    elif corrupt == "base_distance":
        occupied = int(np.argmax(np.asarray(base[0, list(DISTANCE_COLUMNS)])))
        base[0, DISTANCE_COLUMNS[occupied]] = 0
        base[0, DISTANCE_COLUMNS[(occupied + 1) % len(DISTANCE_COLUMNS)]] = 1
    elif corrupt == "full_historical":
        full[0, RESIDUAL_COLUMNS[0]] = np.float16(0.5) + full[0, RESIDUAL_COLUMNS[0]]
    elif corrupt == "base_prior":
        base[0, PRIOR_COLUMNS[0]] = np.float16(0.5) + base[0, PRIOR_COLUMNS[0]]
    elif corrupt == "full_prior":
        full[0, PRIOR_COLUMNS[0]] = np.float16(0.5) + full[0, PRIOR_COLUMNS[0]]
    else:
        full[:, list(RESIDUAL_COLUMNS)] = 0
        local = local.copy()
        local[:, list(RESIDUAL_COLUMNS)] = 0
    with pytest.raises(RuntimeError, match=message):
        assert_blocks_are_exact(base, full, local, values)


def test_a_wrongly_shaped_input_is_refused(blocks):
    local, values, _base, _full, _d4 = blocks
    with pytest.raises(RuntimeError, match="10-column historical block"):
        d6_local_block(np.zeros((local.shape[0], 9), dtype=local.dtype), values,
                       keep=DISTANCE_COLUMNS)
    with pytest.raises(RuntimeError, match="3-column retrieval prior"):
        d6_local_block(local, values[:, :2], keep=DISTANCE_COLUMNS)


# --- normalisation is not extended -------------------------------------------


def test_the_normaliser_was_not_widened():
    proof = assert_normalisation_unchanged()
    assert proof["normalised_columns"] == [4, 5, 6, 7, 8, 9]
    assert proof["extended_to_the_new_columns"] is False
    assert set(PRIOR_COLUMNS).isdisjoint(NORMALISED_COLUMNS)


def test_widening_the_normaliser_would_be_refused(monkeypatch):
    """A negative control on the guard itself."""

    import scripts.run_graph_context_d6 as module

    monkeypatch.setattr(module, "NORMALISED_COLUMNS", (4, 5, 6, 7, 8, 9, 10, 11, 12))
    with pytest.raises(RuntimeError, match="must not extend it"):
        module.assert_normalisation_unchanged()


# --- the architecture is matched, and the widening is disclosed ---------------


def test_the_head_width_is_the_historical_one_not_a_re_solve(completed, data):
    """A re-solved head would narrow as the input widens, changing capacity and
    information at once. The stage holds the historical width instead."""

    accounting = completed["parameter_accounting"]
    historical_input = explicit_feature_input_dim(
        "sa_mlp", 64, 7, len(HISTORICAL_COLUMNS)
    )
    assert accounting["historical_input_dim"] == historical_input
    assert accounting["d6_input_dim"] == historical_input + len(PRIOR_COLUMNS)
    rematched = build_explicit_feature_mlp(
        "sa_mlp", data.node_array.shape[1], 64,
        static_dim=7, local_dim=LOCAL_DIM,
        target_parameters=4000, dropout=0.2, temperature=0.07,
    )
    rematched_width = rematched.scorer[0].out_features
    assert accounting["head_width"] >= rematched_width, (
        "the held width should not be narrower than a re-solve"
    )


def test_the_widening_cost_is_reported(completed):
    accounting = completed["parameter_accounting"]
    assert accounting["d6_parameters"] > accounting["historical_parameters"]
    expected = accounting["d6_parameters"] - accounting["historical_parameters"]
    assert accounting["increase"] == expected
    assert accounting["increase"] == len(PRIOR_COLUMNS) * accounting["head_width"]
    assert accounting["increase_percent"] == pytest.approx(
        100.0 * expected / accounting["historical_parameters"], abs=1e-4
    )


def test_both_arms_have_the_identical_architecture(completed):
    counts = {arm: completed["results"][arm]["parameters"] for arm in D6_ARMS}
    widths = {arm: completed["results"][arm]["head_width"] for arm in D6_ARMS}
    dims = {arm: completed["results"][arm]["local_dim"] for arm in D6_ARMS}
    assert len(set(counts.values())) == 1, counts
    assert len(set(widths.values())) == 1, widths
    assert set(dims.values()) == {LOCAL_DIM}, dims
    assert set(counts.values()) == {completed["parameter_accounting"]["d6_parameters"]}
    assert completed["ablation"]["architecture_changed_between_arms"] is False
    assert completed["ablation"]["architecture_changed_vs_historical"] is True


def test_both_arms_were_trained_here(completed):
    assert completed["arms_trained_here"] == list(D6_ARMS)
    for arm in D6_ARMS:
        assert completed["results"][arm]["retrained_here"] is True
        assert completed["results"][arm]["measured_in"] == "stage_d6"
    assert completed["descriptive_reference"]["new_runs"] == 2


# --- the increment ------------------------------------------------------------


def test_the_increment_is_the_declared_one(completed):
    name, upper, lower = INCREMENT
    assert upper == FULL_ARM and lower == BASE_ARM
    block = completed["increments"][name]
    assert block["to"] == FULL_ARM and block["from"] == BASE_ARM
    assert block["columns"] == [LOCAL_FEATURE_NAMES[c] for c in RESIDUAL_COLUMNS]
    for metric in completed["ladder"][FULL_ARM]:
        expected = (
            completed["ladder"][FULL_ARM][metric] - completed["ladder"][BASE_ARM][metric]
        )
        assert block[metric] == pytest.approx(expected, abs=1e-12), metric


def test_the_increment_is_reported_per_stratum(completed):
    name, upper, lower = INCREMENT
    rows = completed["increments_by_stratum"][name]
    assert rows
    for stratum, row in rows.items():
        assert row["queries"] > 0
        for metric in METRICS:
            if metric not in row:
                continue
            expected = (
                completed["results"][upper]["validation_by_stratum"][stratum][metric]
                - completed["results"][lower]["validation_by_stratum"][stratum][metric]
            )
            assert row[metric] == pytest.approx(expected, abs=1e-12), (stratum, metric)


def test_the_unconditional_version_is_carried_forward(completed, d5_result):
    d5 = json.loads(d5_result.read_text(encoding="utf-8"))
    assert completed["comparators"]["delta_remaining_structure_d3"] == (
        d5["comparators"]["delta_remaining_structure_d3"]
    )
    assert completed["comparators"]["delta_geometry_given_prior_d5"] == (
        d5["increments"]["delta_geometry_given_prior"]
    )


# --- D5 is a reference, not a control ----------------------------------------


def test_d5_is_labelled_descriptive_everywhere(completed):
    assert completed["descriptive_reference"]["role"] == (
        "DESCRIPTIVE REFERENCE, NOT CAUSAL CONTROL"
    )
    assert completed["widening_check"]["role"] == (
        "DESCRIPTIVE REFERENCE, NOT CAUSAL CONTROL"
    )
    assert BASE_ARM not in completed["d5_ladder_for_reference"]
    assert FULL_ARM not in completed["d5_ladder_for_reference"]


def test_the_widening_check_is_arithmetic(completed):
    check = completed["widening_check"]
    for metric, reference in check["d5_reference"].items():
        assert check[metric] == pytest.approx(
            completed["ladder"][BASE_ARM][metric] - reference, abs=1e-12
        ), metric


def test_no_d5_arm_is_used_as_a_control(completed):
    """The increment must be a difference of two arms trained in this stage."""

    name, upper, lower = INCREMENT
    assert {upper, lower} <= set(completed["arms_trained_here"])
    assert set(completed["ladder"]) == set(D6_ARMS)


@pytest.mark.parametrize(
    "field, value",
    [
        ("status", "GRAPH_CONTEXT_D5_IN_PROGRESS"),
        ("dataset", "some_other_dataset"),
        ("data_fingerprint_sha256", "f" * 64),
        ("context", "TARGET_H1"),
        ("num_nodes", 1),
        ("local_feature_schema", ["a", "b"]),
        ("training.epochs", 99),
        ("training.seed", 7),
        ("splits.validation_reported", 1),
        ("a3_rank_feature_audit.constant_K", 10),
        ("prior_equivalence_with_d4.max_abs_diff", 0.5),
        ("feature_build.candidate_rows", 3),
    ],
)
def test_a_drifted_reference_is_refused(tmp_path, data, d5_result, field, value):
    payload = json.loads(d5_result.read_text(encoding="utf-8"))
    node = payload
    *parents, leaf = field.split(".")
    for key in parents:
        node = node[key]
    assert node[leaf] != value, f"{field} was already {value!r}; this proves nothing"
    node[leaf] = value
    path = tmp_path / "stage_d5.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with synthetic(data):
        with pytest.raises(RuntimeError, match="cannot be quoted|is not complete"):
            run(d6_args(tmp_path, data, path))


def test_an_absent_reference_is_refused(tmp_path, data):
    with synthetic(data):
        with pytest.raises(FileNotFoundError, match="descriptive reference"):
            run(d6_args(tmp_path, data, tmp_path / "nothing.json"))


# --- the contract -------------------------------------------------------------


def test_the_contract_holds(completed):
    contract = completed["contract"]
    for key in ("gnn_trained", "message_passing", "test_split_read",
                "epoch_selected_on_validation", "architecture_changed_between_arms"):
        assert contract[key] is False, key
    assert contract["scored_nodes"] == "exactly Cq in every arm"
    assert completed["splits"]["test_read"] is False
    assert completed["status"] == COMPLETE_STATUS


def test_the_test_split_is_refused_outright(tmp_path, data, d5_result):
    with synthetic(data):
        args = d6_args(tmp_path, data, d5_result)
        args.splits = ["train", "validation", "test"]
        with pytest.raises(ValueError, match="test split is not read"):
            run(args)


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
    assert completed["local_dim"] == LOCAL_DIM
    assert completed["local_feature_schema"] == list(D6_LOCAL_FEATURE_NAMES)
