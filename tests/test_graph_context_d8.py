"""D8 replaces one column and claims the replacement is the only thing that moved.

That claim is what this file attacks. The replacement arm must differ from the
historical arm in column 4 and in nothing else, in both directions -- an arm
that also nudged the prior would still "carry the corrected column". The
injected scalar must reach the learner as `count / |Sq|`, not as that value
divided a second time by a per-query maximum, which is the specific mistake that
would silently turn the replacement back into the thing it replaces. And because
`B` and `H` are reused from other containers rather than refitted, the guards
standing in for a reproduction run are tested by breaking them.

The verdict thresholds are exercised as a table. They were registered in the
declaration before the stage ran, and the tests pin the boundaries so that a
later edit to the rule shows up as a failing test rather than a nicer label.
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
from mp_retrieval.distinct_support import HISTORICAL_NAME, REPLACEMENT_NAME  # noqa: E402
from mp_retrieval.graph_context import NORMALISED_COLUMNS  # noqa: E402
from scripts.run_graph_context_d3 import DISTANCE_COLUMNS  # noqa: E402
from scripts.run_graph_context_d5 import prior_value_columns  # noqa: E402
from scripts.run_graph_context_d6 import (  # noqa: E402
    HISTORICAL_COLUMNS,
    LOCAL_DIM,
    PRIOR_COLUMNS,
    RESIDUAL_COLUMNS,
    d6_local_block,
)
from scripts.run_graph_context_d7 import residual_column_occupancy  # noqa: E402
from scripts.run_graph_context_d8 import (  # noqa: E402
    ABORTED_STATUS,
    BASE,
    COMPLETE_STATUS,
    CONTEXT,
    DELTA_H_MINUS_B,
    DELTA_V_MINUS_B,
    DELTA_V_MINUS_H,
    FAILS,
    HISTORICAL_ARM,
    IMPROVES,
    MATERIAL,
    PARETO_MATCHES,
    REPLACEMENT_ARM,
    REUSED_ARMS,
    SUPPORT_COLUMN,
    assert_d8_block_is_exact,
    assert_the_injection_was_not_rescaled,
    build_distinct_support,
    build_parser,
    classify,
    cost_gate,
    d8_local_block,
    historical_support_audit,
    mechanistic_comparison,
    replacement_definition,
    run,
    verify_reuse,
    verify_substrate_reproduces_d7,
)
from test_graph_context_d1 import args_for, dataset as build_dataset, local_block  # noqa: E402
from test_graph_context_d4 import (  # noqa: E402
    DENSE_SLICE,
    RRF_CONSTANT,
    SPLADE_SLICE,
    write_rank_lists,
)
from test_graph_context_d7 import d6_result, synthetic as d7_synthetic  # noqa: E402, F401

METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")


@contextmanager
def synthetic(data):
    """D7's patch set, extended to D8's own module."""
    import scripts.run_graph_context_d8 as module

    static = np.linspace(0.0, 1.0, int(data.num_nodes) * 7, dtype=np.float32).reshape(
        int(data.num_nodes), 7
    )
    original = (module.load_complete_dataset, module.load_or_build_static)
    module.load_complete_dataset = lambda *a, **k: data
    module.load_or_build_static = lambda *a, **k: (static, {"source": "synthetic"})
    try:
        with d7_synthetic(data):
            yield module
    finally:
        module.load_complete_dataset, module.load_or_build_static = original


@pytest.fixture(scope="module")
def data() -> CompleteRetrievalDataset:
    return build_dataset()


def d8_args(tmp_path: Path, data, d7_result: Path, **overrides):
    write_rank_lists(tmp_path, data)
    parsed = vars(args_for(tmp_path, data, **overrides))
    parsed["output"] = Path(parsed["output"]).with_name("stage_d8.json")
    parsed["d7_result"] = d7_result
    parsed["rrf_constant"] = RRF_CONSTANT
    known = {action.dest for action in build_parser()._actions}
    return argparse.Namespace(**{k: v for k, v in parsed.items() if k in known})


@pytest.fixture(scope="module")
def d7_result(tmp_path_factory, data, d6_result) -> Path:  # noqa: F811
    """A real D7, so the historical arm D8 reuses is genuine output."""
    import scripts.run_graph_context_d7 as d7_module
    from test_graph_context_d7 import d7_args

    with synthetic(data):
        seven = d7_args(tmp_path_factory.mktemp("d8_d7"), data, d6_result)
        d7_module.run(seven)
    return Path(seven.output)


@pytest.fixture(scope="module")
def completed(tmp_path_factory, data, d7_result):
    with synthetic(data):
        return run(d8_args(tmp_path_factory.mktemp("d8"), data, d7_result))


@pytest.fixture(scope="module")
def blocks(data):
    """B, H and V over the synthetic substrate, plus the pieces they came from."""
    local, ptr = local_block(data, CONTEXT)
    local = np.asarray(local)
    dense = np.stack([q.candidate_index.numpy()[DENSE_SLICE] for q in data.queries])
    splade = np.stack([q.candidate_index.numpy()[SPLADE_SLICE] for q in data.queries])
    values, _prior = prior_value_columns(
        data.queries, ptr, dense, splade, constant=RRF_CONSTANT, dtype=local.dtype
    )
    keep = tuple(sorted(DISTANCE_COLUMNS + (SUPPORT_COLUMN,)))
    base = d6_local_block(local, values, keep=DISTANCE_COLUMNS)
    full = d6_local_block(local, values, keep=HISTORICAL_COLUMNS)
    historical = d6_local_block(local, values, keep=keep)
    # A support scalar that is genuinely different from the historical column,
    # so the "V differs from H" invariant is not satisfied by accident.
    rng = np.random.default_rng(0)
    fraction = (rng.integers(0, 4, size=local.shape[0]) / 3.0).astype(local.dtype)
    replacement = d8_local_block(local, values, fraction)
    return {
        "local": local,
        "ptr": ptr,
        "values": values,
        "base": base,
        "full": full,
        "historical": historical,
        "replacement": replacement,
        "fraction": fraction,
        "occupancy": residual_column_occupancy(full),
    }


def _exact(blocks, **overrides):
    payload = {
        "replacement": blocks["replacement"],
        "historical": blocks["historical"],
        "base": blocks["base"],
        "values": blocks["values"],
        "support": blocks["fraction"],
        "historical_column": blocks["local"][:, SUPPORT_COLUMN],
        "occupancy": blocks["occupancy"],
    }
    payload.update(overrides)
    return assert_d8_block_is_exact(**payload)


# --- the audit ---------------------------------------------------------------


def test_the_audit_reports_the_formula_not_the_name():
    audit = historical_support_audit()
    assert audit["column_index"] == SUPPORT_COLUMN
    assert audit["column_name"] == HISTORICAL_NAME
    assert audit["hop_radius"] == 1
    assert audit["multiplicity_is_possible"] is True
    assert audit["reverse_edges_count_separately"] is True
    assert audit["duplicate_edges_count_separately"] is True
    assert audit["a_seed_adjacent_to_a_seed_accrues_support"] is True
    assert "EDGE count" in audit["raw_quantity"]
    assert "log1p" in audit["transform"]
    assert "per-query maximum" in audit["transform"]


def test_the_audit_names_every_step_of_the_computation_path():
    path = historical_support_audit()["computation_path"]
    assert any("build_local_features" in step for step in path)
    assert any("qls_local_features" in step for step in path)
    assert any("_local_feature_chunk" in step for step in path)
    assert any("candidate_readout" in step for step in path)


def test_the_audit_records_every_route_to_multiplicity():
    routes = " ".join(historical_support_audit()["how_multiplicity_arises"]).lower()
    assert "reciprocal" in routes
    assert "parallel" in routes
    assert "self-loop" in routes


def test_the_historical_column_is_one_of_the_normalised_columns():
    """The audit rests on this; if the frozen kernel changed, D8's premise is wrong."""
    assert SUPPORT_COLUMN in NORMALISED_COLUMNS


# --- the replacement definition ----------------------------------------------


def test_exactly_one_replacement_is_defined():
    definition = replacement_definition()
    assert definition["name"] == REPLACEMENT_NAME
    assert definition["replaces"] == HISTORICAL_NAME
    assert definition["canonical_scalar"] == "support_fraction"
    banned = definition["only_one_replacement_here"].lower()
    for absent in ("support@1", "weighted support", "path diversity", "diffusion"):
        assert absent in banned


def test_the_relation_is_the_historical_one():
    definition = replacement_definition()
    assert "identical to the historical relation" in definition["relation"]
    unchanged = " ".join(definition["what_does_not_change"]).lower()
    assert "hop radius, still 1" in unchanged
    assert "seed privilege" in unchanged
    assert definition["what_changes"] == "the counting rule, and only the counting rule"


def test_the_algorithm_is_one_bounded_pass():
    algorithm = replacement_definition()["algorithm"]
    assert algorithm["fixed_graph_passes"] == 1
    assert algorithm["iterates_to_convergence"] is False
    assert "O(|E[Cq]|)" in algorithm["time_complexity"]
    assert "uint64" in algorithm["temporary_memory"]
    assert "no SIMD" in algorithm["optimisation_deferred"]


# --- the verdict table -------------------------------------------------------


def _delta(**overrides):
    delta = dict.fromkeys(METRICS, 0.0)
    delta.update(overrides)
    return delta


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (_delta(), PARETO_MATCHES),
        (_delta(**{"recall@5": MATERIAL}), IMPROVES),
        (_delta(**{"recall@5": MATERIAL - 1e-9}), PARETO_MATCHES),
        (_delta(**{"recall@5": -MATERIAL}), FAILS),
        (_delta(**{"recall@5": -MATERIAL + 1e-9}), PARETO_MATCHES),
        (_delta(**{"full_coverage@20": MATERIAL}), IMPROVES),
        (_delta(**{"mrr": MATERIAL}), IMPROVES),
        (_delta(**{"recall@1": -MATERIAL}), FAILS),
        (_delta(**{"recall@5": MATERIAL, "recall@1": -MATERIAL}), PARETO_MATCHES),
        (_delta(**{"recall@5": 0.02, "mrr": -0.02}), PARETO_MATCHES),
        (_delta(**{"recall@5": 0.002, "mrr": -0.001}), PARETO_MATCHES),
        (_delta(**{"recall@5": 0.02, "recall@20": 0.01}), IMPROVES),
        (_delta(**{"recall@5": -0.02, "mrr": -0.01}), FAILS),
    ],
)
def test_the_verdict_table_holds_at_every_boundary(delta, expected):
    assert classify(delta)["label"] == expected


def test_a_material_gain_with_a_material_loss_is_recorded_as_mixed():
    verdict = classify(_delta(**{"recall@5": 0.02, "mrr": -0.02}))
    assert verdict["label"] == PARETO_MATCHES
    assert verdict["mixed"] is True
    assert verdict["all_inside_the_band"] is False
    assert verdict["materially_better_on"] == ["recall@5"]
    assert verdict["materially_worse_on"] == ["mrr"]


def test_a_true_tie_is_not_recorded_as_mixed():
    verdict = classify(_delta(**{"recall@5": 0.001}))
    assert verdict["label"] == PARETO_MATCHES
    assert verdict["mixed"] is False
    assert verdict["all_inside_the_band"] is True


def test_rule_a_is_reported_separately_from_the_trichotomy():
    """The user's stated criterion, visible independent of the Pareto reading."""
    clean = classify(_delta(**{"recall@5": 0.01, "recall@1": -0.001, "mrr": 0.001}))
    assert clean["label"] == IMPROVES
    assert clean["rule_a_r5_gain_with_head_maintained"] is True

    traded = classify(_delta(**{"recall@5": 0.01, "mrr": -0.02}))
    assert traded["rule_a_r5_gain_with_head_maintained"] is False
    assert traded["label"] == PARETO_MATCHES


def test_a_pareto_match_does_not_demand_an_accuracy_increase():
    verdict = classify(_delta())
    assert "not required" in verdict["if_pareto_matches"]
    assert "bounded [0,1]" in verdict["if_pareto_matches"]


def test_failing_does_not_send_us_back_to_raw_edge_counts():
    verdict = classify(_delta(**{"recall@5": -0.02}))
    assert verdict["label"] == FAILS
    assert "do NOT immediately revert to raw edge counts" in verdict["if_fails"]
    assert "PATHS" in verdict["if_fails"]


def test_the_verdict_is_never_a_significance_claim():
    assert "not a" in classify(_delta())["not_a_significance_claim"]


# --- the tensor invariants ---------------------------------------------------


def test_the_exact_blocks_pass(blocks):
    checks = _exact(blocks)
    assert checks["max_abs_diff"] == 0.0
    assert checks["differs_from_historical_in"] == [HISTORICAL_NAME]
    assert checks["differs_from_base_in"] == [HISTORICAL_NAME]
    assert checks["other_residual_columns_are_zero"] is True
    assert checks["historical_support_column_nonzero_rows"] > 0


def test_the_replacement_block_is_the_declared_width(blocks):
    assert blocks["replacement"].shape[1] == LOCAL_DIM == 13


def test_an_arm_that_also_leaked_a_path_column_is_refused(blocks):
    leaked = np.array(blocks["replacement"], copy=True)
    leaked[:, 5] = blocks["full"][:, 5]
    with pytest.raises(RuntimeError, match="not exact"):
        _exact(blocks, replacement=leaked)


def test_an_arm_whose_prior_moved_is_refused(blocks):
    moved = np.array(blocks["replacement"], copy=True)
    moved[:, PRIOR_COLUMNS[0]] = 0.0
    with pytest.raises(RuntimeError, match="not exact"):
        _exact(blocks, replacement=moved)


def test_an_arm_whose_distance_geometry_moved_is_refused(blocks):
    rotated = np.array(blocks["replacement"], copy=True)
    rotated[:, list(DISTANCE_COLUMNS)] = rotated[:, list(DISTANCE_COLUMNS)][:, ::-1]
    with pytest.raises(RuntimeError, match="not exact"):
        _exact(blocks, replacement=rotated)


def test_an_arm_carrying_the_historical_column_unchanged_is_refused(blocks):
    """If V == H there is nothing for V - H to measure, so the stage must stop."""
    unchanged = np.array(blocks["historical"], copy=True)
    with pytest.raises(RuntimeError, match="not exact"):
        _exact(
            blocks,
            replacement=unchanged,
            support=blocks["local"][:, SUPPORT_COLUMN],
        )


def test_a_historical_arm_that_is_not_the_frozen_column_is_refused(blocks):
    drifted = np.array(blocks["historical"], copy=True)
    drifted[:, SUPPORT_COLUMN] = drifted[:, SUPPORT_COLUMN] * 0.5
    with pytest.raises(RuntimeError, match="not exact"):
        _exact(blocks, historical=drifted)


def test_an_empty_historical_support_column_is_refused(blocks):
    hollow = dict(blocks["occupancy"])
    hollow["_counts"] = dict(hollow["_counts"])
    hollow["_counts"][SUPPORT_COLUMN] = 0
    with pytest.raises(RuntimeError, match="not exact"):
        _exact(blocks, occupancy=hollow)


def test_the_invariant_is_stated_in_both_directions(blocks):
    requirement = _exact(blocks)["requirement"]
    assert "every other" in requirement
    assert "column 4 alone" in requirement


# --- the injection discipline ------------------------------------------------


@pytest.fixture(scope="module")
def support(data):
    latency: list[float] = []
    shared: list[float] = []
    rowptr = data.rowptr.numpy().astype(np.int64, copy=False)
    col = data.col.numpy().astype(np.int64, copy=False)
    from mp_retrieval.graph_context import build_operators

    operators = build_operators(rowptr, col, int(data.num_nodes))
    return build_distinct_support(
        data.queries, rowptr, col, int(data.num_nodes), operators,
        latency=latency, shared_latency=shared,
    )


def test_the_learner_receives_count_over_num_seeds(data, support):
    local, ptr = local_block(data, CONTEXT)
    local = np.asarray(local)
    dense = np.stack([q.candidate_index.numpy()[DENSE_SLICE] for q in data.queries])
    splade = np.stack([q.candidate_index.numpy()[SPLADE_SLICE] for q in data.queries])
    values, _prior = prior_value_columns(
        data.queries, ptr, dense, splade, constant=RRF_CONSTANT, dtype=local.dtype
    )
    block = d8_local_block(local, values, support["fraction"])
    checks = assert_the_injection_was_not_rescaled(block, support, ptr)
    assert checks["elementwise_identical"] is True
    assert checks["max_abs_diff_against_count_over_num_seeds"] == 0.0
    assert 0.0 <= checks["column_minimum"] <= checks["column_maximum"] <= 1.0
    assert checks["injected_after"] == "candidate_readout"
    assert checks["normalised_columns_extended_to_the_replacement"] is False


def test_a_second_per_query_max_rescaling_is_caught(data, support):
    """The exact mistake the discipline exists to prevent."""
    local, ptr = local_block(data, CONTEXT)
    local = np.asarray(local)
    dense = np.stack([q.candidate_index.numpy()[DENSE_SLICE] for q in data.queries])
    splade = np.stack([q.candidate_index.numpy()[SPLADE_SLICE] for q in data.queries])
    values, _prior = prior_value_columns(
        data.queries, ptr, dense, splade, constant=RRF_CONSTANT, dtype=local.dtype
    )
    rescaled = np.asarray(support["fraction"], dtype=np.float64).copy()
    moved = False
    for start, end in zip(ptr[:-1], ptr[1:], strict=True):
        window = rescaled[start:end]
        maximum = window.max() if window.size else 0.0
        if 0.0 < maximum < 1.0:
            rescaled[start:end] = window / maximum
            moved = True
    if not moved:
        pytest.skip("no query on this substrate has a sub-maximal support column")
    block = d8_local_block(local, values, rescaled.astype(local.dtype))
    with pytest.raises(RuntimeError, match="not receiving"):
        assert_the_injection_was_not_rescaled(block, support, ptr)


def test_the_gate_is_the_elementwise_check_not_the_maxima(data, support):
    local, ptr = local_block(data, CONTEXT)
    local = np.asarray(local)
    dense = np.stack([q.candidate_index.numpy()[DENSE_SLICE] for q in data.queries])
    splade = np.stack([q.candidate_index.numpy()[SPLADE_SLICE] for q in data.queries])
    values, _prior = prior_value_columns(
        data.queries, ptr, dense, splade, constant=RRF_CONSTANT, dtype=local.dtype
    )
    block = d8_local_block(local, values, support["fraction"])
    checks = assert_the_injection_was_not_rescaled(block, support, ptr)
    assert "elementwise equality" in checks["which_check_is_the_gate"]


# --- the replacement kernel over real query objects --------------------------


def test_the_support_columns_line_up_with_the_candidate_rows(data, support):
    _local, ptr = local_block(data, CONTEXT)
    rows = int(ptr[-1])
    for key in ("fraction", "count", "connections", "degree"):
        assert support[key].shape[0] == rows, key
    assert support["seeds_per_query"].size == len(data.queries)


def test_the_distinct_count_never_exceeds_the_edge_count(support):
    assert np.all(support["count"] <= support["connections"])


def test_the_fraction_is_the_count_over_the_seeds(data, support):
    _local, ptr = local_block(data, CONTEXT)
    widths = np.diff(ptr)
    seeds = support["seeds_per_query"].astype(np.float64)
    denominator = np.repeat(np.where(seeds > 0, seeds, 1.0), widths)
    expected = (support["count"].astype(np.float64) / denominator).astype(np.float16)
    np.testing.assert_array_equal(support["fraction"], expected)


def test_the_workspace_is_bounded_and_recorded(support):
    assert support["temporary_workspace_bytes"] > 0


# --- the mechanistic comparison ----------------------------------------------


def test_the_comparison_is_not_a_selection_criterion(data, support):
    local, ptr = local_block(data, CONTEXT)
    comparison = mechanistic_comparison(
        support, np.asarray(local)[:, SUPPORT_COLUMN], ptr
    )
    disclaimer = comparison["is_not_a_selection_criterion"]
    assert "before the effectiveness numbers are interpreted" in disclaimer
    assert "nearly identical" in disclaimer


def test_the_comparison_reports_every_declared_quantity(data, support):
    local, ptr = local_block(data, CONTEXT)
    comparison = mechanistic_comparison(
        support, np.asarray(local)[:, SUPPORT_COLUMN], ptr
    )
    assert set(comparison["by_induced_degree"]) == {
        "degree_0", "degree_1", "degree_2_to_4", "degree_5_plus"
    }
    assert comparison["rows"] > 0
    assert "historical_column_vs_support_fraction" in comparison["correlation"]
    assert "rows_where_edges_exceed_distinct_seeds" in comparison["multiplicity"]
    assert comparison["multiplicity"]["max_historical_raw_edge_count"] >= 0.0
    ordering = comparison["ordering"]
    assert ordering["within_query_candidate_pairs"] > 0
    assert 0.0 <= ordering["fraction_ordered_differently"] <= 1.0


def test_every_pair_is_accounted_for_exactly_once(data, support):
    """The ordering split has to partition the pairs, or the fractions lie."""
    local, ptr = local_block(data, CONTEXT)
    ordering = mechanistic_comparison(
        support, np.asarray(local)[:, SUPPORT_COLUMN], ptr
    )["ordering"]
    total = (
        ordering["concordant"]
        + ordering["strictly_reversed"]
        + ordering["tied_by_the_historical_column_only"]
        + ordering["tied_by_distinct_support_only"]
        + ordering["tied_by_both"]
    )
    assert total == ordering["within_query_candidate_pairs"]


def test_the_degree_buckets_partition_the_rows(data, support):
    local, ptr = local_block(data, CONTEXT)
    buckets = mechanistic_comparison(
        support, np.asarray(local)[:, SUPPORT_COLUMN], ptr
    )["by_induced_degree"]
    assert sum(entry["rows"] for entry in buckets.values()) == int(ptr[-1])


def test_the_reported_comparison_is_the_validation_block(completed):
    """The effectiveness numbers are validation numbers; so is the explanation."""
    comparison = completed["mechanistic_comparison"]
    assert comparison["measured_on"] == "validation feature construction"
    assert comparison["seeds_per_query"]["queries"] == completed["splits"][
        "validation_reported"
    ]
    everything = comparison["also_over_every_opened_query"]
    assert everything["measured_on"] == "every opened query"
    assert everything["seeds_per_query"]["queries"] == sum(
        completed["splits"][key]
        for key in (
            "train_fit",
            "train_holdout_for_epoch_selection",
            "validation_reported",
        )
    )
    assert comparison["rows"] < everything["rows"]
    assert "The validation figures above are the reported ones." in everything[
        "why_reported"
    ]


def test_slicing_the_whole_range_reproduces_the_unsliced_statistic(data, support):
    local, ptr = local_block(data, CONTEXT)
    column = np.asarray(local)[:, SUPPORT_COLUMN]
    whole = mechanistic_comparison(support, column, ptr)
    sliced = mechanistic_comparison(
        support, column, ptr, queries=slice(0, len(data.queries) + 1)
    )
    for key in ("rows", "rows_with_any_support", "multiplicity", "by_induced_degree"):
        assert sliced[key] == whole[key], key
    assert sliced["ordering"]["within_query_candidate_pairs"] == whole["ordering"][
        "within_query_candidate_pairs"
    ]


def test_a_tail_slice_reads_only_its_own_rows(data, support):
    """A slice that dropped the pointer rebase would read the wrong candidates."""
    local, ptr = local_block(data, CONTEXT)
    column = np.asarray(local)[:, SUPPORT_COLUMN]
    start = len(data.queries) // 2
    tail = mechanistic_comparison(support, column, ptr, queries=slice(start, None))
    expected_rows = int(ptr[-1]) - int(ptr[start])
    assert tail["rows"] == expected_rows
    assert tail["seeds_per_query"]["queries"] == len(data.queries) - start
    buckets = tail["by_induced_degree"]
    assert sum(entry["rows"] for entry in buckets.values()) == expected_rows
    ordering = tail["ordering"]
    assert (
        ordering["concordant"]
        + ordering["strictly_reversed"]
        + ordering["tied_by_the_historical_column_only"]
        + ordering["tied_by_distinct_support_only"]
        + ordering["tied_by_both"]
    ) == ordering["within_query_candidate_pairs"]


# --- reuse and reproduction --------------------------------------------------


def test_the_stage_completes_and_trains_exactly_one_arm(completed):
    assert completed["status"] == COMPLETE_STATUS
    assert completed["arms_trained_here"] == [REPLACEMENT_ARM]
    assert sorted(completed["arms_reused"]) == sorted(REUSED_ARMS)
    assert completed["reuse"]["new_runs"] == 1
    trained = [
        arm for arm, row in completed["results"].items() if row.get("retrained_here")
    ]
    assert trained == [REPLACEMENT_ARM]


def test_every_arm_shares_the_architecture(completed):
    rows = completed["results"]
    assert len({row["parameters"] for row in rows.values()}) == 1
    assert {row["local_dim"] for row in rows.values()} == {LOCAL_DIM}
    assert completed["ablation"]["architecture_changed_between_arms"] is False
    assert completed["ablation"]["architecture_changed_vs_d7"] is False


def test_the_three_increments_are_reported(completed):
    increments = completed["increments"]
    assert set(increments) == {DELTA_H_MINUS_B, DELTA_V_MINUS_B, DELTA_V_MINUS_H}
    assert increments[DELTA_V_MINUS_H]["from"] == HISTORICAL_ARM
    assert increments[DELTA_V_MINUS_H]["to"] == REPLACEMENT_ARM
    assert increments[DELTA_V_MINUS_B]["from"] == BASE
    for key in METRICS:
        assert key in increments[DELTA_V_MINUS_H]


def test_the_increments_are_consistent_with_the_ladder(completed):
    ladder = completed["ladder"]
    for key in METRICS:
        assert completed["increments"][DELTA_V_MINUS_H][key] == pytest.approx(
            ladder[REPLACEMENT_ARM][key] - ladder[HISTORICAL_ARM][key]
        )
        assert completed["increments"][DELTA_V_MINUS_B][key] == pytest.approx(
            ladder[REPLACEMENT_ARM][key] - ladder[BASE][key]
        )


def test_the_carried_h_minus_b_reproduces_d7(completed, d7_result):
    d7 = json.loads(Path(d7_result).read_text(encoding="utf-8"))
    for key in METRICS:
        assert completed["increments"][DELTA_H_MINUS_B][key] == pytest.approx(
            d7["increments"]["delta_support"][key]
        )


def test_the_verdict_is_recomputed_from_the_increment(completed):
    assert completed["verdict"]["label"] == classify(
        completed["increments"][DELTA_V_MINUS_H]
    )["label"]
    assert completed["verdict"]["increment"] == DELTA_V_MINUS_H


def test_the_reuse_conditions_all_matched(completed):
    reuse = completed["reuse"]
    assert reuse["all_conditions_match"] is True
    assert reuse["source"] == "stage_d7.json"
    assert all(block["match"] for block in reuse["conditions_checked"].values())
    assert reuse["role"][BASE].startswith("EXACT MATCHED CAUSAL CONTROL")
    assert "HISTORICAL PROXY" in reuse["role"][HISTORICAL_ARM]


def test_the_substrate_reproduces_d7(completed):
    reproduction = completed["deterministic_reproduction"]
    assert reproduction["all_conditions_match"] is True
    assert "stop and inspect" in reproduction["on_failure"]
    assert "historical_support_column_nonzero_rows" in reproduction["conditions_checked"]


def test_a_d7_that_did_not_complete_is_refused(completed, d7_result):
    d7 = json.loads(Path(d7_result).read_text(encoding="utf-8"))
    d7["status"] = "GRAPH_CONTEXT_D7_IN_PROGRESS"
    with pytest.raises(RuntimeError, match="not complete"):
        verify_reuse(d7, completed, _args(completed))


def _args(completed):
    return argparse.Namespace(
        dataset=completed["dataset"],
        seed=completed["training"]["seed"],
        data_fingerprint_sha256=completed["data_fingerprint_sha256"],
        rrf_constant=completed["a3_rank_feature_audit"]["constant_K"],
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset", "some_other_dataset"),
        ("seed", 7),
        ("data_fingerprint_sha256", "f" * 64),
        ("rrf_constant", 11),
    ],
)
def test_a_drifted_condition_refuses_the_reuse(completed, d7_result, field, value):
    d7 = json.loads(Path(d7_result).read_text(encoding="utf-8"))
    args = _args(completed)
    setattr(args, field, value)
    with pytest.raises(RuntimeError, match="cannot reuse"):
        verify_reuse(d7, completed, args)


def test_a_d7_whose_own_reuse_failed_cannot_be_chained_onto(completed, d7_result):
    d7 = json.loads(Path(d7_result).read_text(encoding="utf-8"))
    d7["reuse"]["all_conditions_match"] = False
    with pytest.raises(RuntimeError, match="cannot chain onto it"):
        verify_reuse(d7, completed, _args(completed))


def test_a_d7_whose_substrate_check_failed_cannot_be_chained_onto(completed, d7_result):
    d7 = json.loads(Path(d7_result).read_text(encoding="utf-8"))
    d7["deterministic_reproduction"]["all_conditions_match"] = False
    with pytest.raises(RuntimeError, match="cannot chain onto it"):
        verify_reuse(d7, completed, _args(completed))


def test_a_reused_arm_that_was_never_fitted_is_refused(completed, d7_result):
    d7 = json.loads(Path(d7_result).read_text(encoding="utf-8"))
    d7["results"][HISTORICAL_ARM]["retrained_here"] = False
    with pytest.raises(RuntimeError, match="was not fitted in D7"):
        verify_reuse(d7, completed, _args(completed))


@pytest.mark.parametrize(
    "field",
    [
        "candidate_rows",
        "residual_nonzero_entries",
        "historical_support_column_nonzero_rows",
        "prior_rows_ranked_by_both",
    ],
)
def test_a_substrate_that_does_not_reproduce_d7_stops_the_stage(
    completed, d7_result, blocks, field
):
    """A drifted substrate must stop the stage, never be refit around."""
    d7 = json.loads(Path(d7_result).read_text(encoding="utf-8"))
    occupancy = completed["deterministic_reproduction"]["conditions_checked"]
    assert occupancy[field]["match"] is True
    if field == "candidate_rows":
        d7["feature_build"]["candidate_rows"] += 1
    elif field == "residual_nonzero_entries":
        d7["tensor_equivalence"]["residual_column_occupancy"][
            "total_nonzero_entries"
        ] += 1
    elif field == "historical_support_column_nonzero_rows":
        d7["tensor_equivalence"]["residual_column_occupancy"][
            "nonzero_rows_by_column"
        ][HISTORICAL_NAME] += 1
    else:
        d7["graded_retrieval_prior"]["rows_ranked_by_both"] += 1
    with pytest.raises(RuntimeError, match="STOP AND INSPECT"):
        verify_substrate_reproduces_d7(d7, completed, blocks["occupancy"])


# --- the cost gate -----------------------------------------------------------


def test_the_cost_gate_fires_only_when_construction_dominates():
    assert cost_gate(1.0, 10.0)["construction_dominates_the_build"] is False
    assert cost_gate(10.0, 10.0)["construction_dominates_the_build"] is False
    assert cost_gate(10.001, 10.0)["construction_dominates_the_build"] is True
    assert cost_gate(1.0, 10.0)["trained"] is True
    assert cost_gate(20.0, 10.0)["trained"] is False


def test_the_stage_stops_at_the_microbenchmark_when_the_gate_fires(
    tmp_path, data, d7_result
):
    """The abort path is a reported outcome, not a crash, and it does not train."""
    import scripts.run_graph_context_d8 as module

    original = module.cost_gate
    module.cost_gate = lambda support, build: original(build + 1.0, build)
    try:
        with synthetic(data):
            aborted = run(d8_args(tmp_path, data, d7_result))
    finally:
        module.cost_gate = original

    assert aborted["status"] == ABORTED_STATUS
    assert aborted["cost_gate"]["construction_dominates_the_build"] is True
    assert aborted["cost_gate"]["trained"] is False
    assert aborted["results"] == {}
    assert "no effectiveness claim is made" in aborted["outcome"]
    assert "verdict" not in aborted
    # The measurement it did make still stands and is still reported.
    assert aborted["mechanistic_comparison"]["rows"] > 0
    assert aborted["deterministic_reproduction"]["all_conditions_match"] is True


def test_the_cost_gate_is_recorded_and_did_not_fire(completed):
    gate = completed["cost_gate"]
    assert gate["construction_dominates_the_build"] is False
    assert gate["trained"] is True
    assert "before this stage ran" in gate["registered"]
    assert completed["status"] != ABORTED_STATUS


def test_the_incremental_cost_is_not_the_whole_pipeline(completed):
    construction = completed["distinct_support_construction"]
    assert construction["fixed_graph_passes"] == 1
    assert "not attributed to one feature" in construction["measures"]
    assert "would share" in construction["what_is_incremental"]
    for key in ("p50", "p95", "p99"):
        assert key in construction["incremental_latency_ms_per_query"]
    assert construction["temporary_workspace_bytes"] > 0


# --- the contract ------------------------------------------------------------


def test_no_gnn_no_test_split_no_pool_modification(completed):
    contract = completed["contract"]
    assert contract["gnn_trained"] is False
    assert contract["message_passing"] is False
    assert contract["test_split_read"] is False
    assert contract["candidate_pools_modified"] is False
    assert contract["epoch_selected_on_validation"] is False
    assert completed["splits"]["test_read"] is False


def test_the_test_split_cannot_be_requested(tmp_path, data, d7_result):
    args = d8_args(tmp_path, data, d7_result)
    args.splits = ["train", "validation", "test"]
    with pytest.raises(ValueError, match="test split is not read"):
        run(args)


def test_the_normaliser_was_not_extended(completed):
    normalisation = completed["normalisation"]
    assert normalisation["normalised_columns"] == list(NORMALISED_COLUMNS)
    assert normalisation["normalised_columns_extended_to_the_replacement"] is False
    assert normalisation["path_and_ppr_definitions_changed"] is False


def test_the_selection_rule_correction_is_recorded_before_the_outcome(completed):
    rule = completed["selection_rule"]
    assert rule["the_rule_is_not"] == "choose the largest R@5"
    assert "Pareto frontier" in rule["the_rule_is"]
    assert "before this stage ran" in rule["recorded"]
    assert "before any D8 number existed" in rule["not_outcome_driven"]


def test_paths_is_deferred_not_killed(completed):
    paths = completed["paths_status"]
    assert paths["classification"] == "PROMISING / DEFERRED"
    assert paths["measured_in_d7"] == pytest.approx(0.0082)
    assert "not because it lost" in paths["why_deferred"]


def test_the_stage_does_not_claim_more_than_it_measured(completed):
    text = completed["not_established"].lower()
    assert "one dataset, one seed" in text
    assert "weighted support" in text
    assert "path diversity" in text
