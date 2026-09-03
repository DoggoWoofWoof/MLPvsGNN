"""D9 replaces three columns, and can decide not to spend a training run at all.

Two claims are under attack here. The first is the usual one: the replacement
arm must differ from the historical arm in columns 5-7 and in nothing else, in
both directions, and the injected scalars must reach the learner as
`branch/(branch+1)` rather than as that value divided a second time by a
per-query maximum -- the mistake that would quietly restore the query-relative
rescaling the replacement exists to remove.

The second is new. D9 carries a gate that stops before GPU work when the two
blocks are representation-equivalent, and a gate that can decide anything can
also decide wrongly. Its thresholds were filed in the declaration before the
comparison existed, so they are pinned here as a table; both outcomes are
exercised end to end; and the ordering statistic -- the one condition with
discriminating power, since `branch > 0` implies `walk > 0` by construction --
is checked against an independent count rather than trusted.

REACH is computed and reported but never injected. That is the correction this
stage was built around: "distinct seeds within k hops" is D8's support feature
at a larger radius, and a test asserts the injected block is BRANCH.
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
from mp_retrieval.path_diversity import (  # noqa: E402
    HISTORICAL_NAMES,
    HOPS,
    REPLACEMENT_NAMES,
)
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
from scripts.run_graph_context_d9 import (  # noqa: E402
    BASE,
    COMPLETE_STATUS,
    CONTEXT,
    COST_STATUS,
    DELTA_H_MINUS_B,
    DELTA_V_MINUS_B,
    DELTA_V_MINUS_H,
    FAILS,
    GATE_STATUS,
    HISTORICAL_ARM,
    IMPROVES,
    MATERIAL,
    NONZERO_AGREEMENT_MIN,
    ORDERING_CHANGE_MAX,
    PARETO_MATCHES,
    PATH_COLUMNS,
    REPLACEMENT_ARM,
    REUSED_ARMS,
    SPEARMAN_ABS_MIN,
    UNINFORMATIVE,
    apply_the_frozen_gate,
    assert_d9_block_is_exact,
    assert_the_injection_was_not_rescaled,
    build_parser,
    build_path_diversity,
    classify,
    cost_gate,
    d9_local_block,
    historical_paths_audit,
    mechanistic_gate,
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
    """D7's patch set, extended to D9's own module."""
    import scripts.run_graph_context_d9 as module

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


def d9_args(tmp_path: Path, data, d7_result: Path, **overrides):
    write_rank_lists(tmp_path, data)
    parsed = vars(args_for(tmp_path, data, **overrides))
    parsed["output"] = Path(parsed["output"]).with_name("stage_d9.json")
    parsed["d7_result"] = d7_result
    parsed["rrf_constant"] = RRF_CONSTANT
    known = {action.dest for action in build_parser()._actions}
    return argparse.Namespace(**{k: v for k, v in parsed.items() if k in known})


@pytest.fixture(scope="module")
def d7_result(tmp_path_factory, data, d6_result) -> Path:  # noqa: F811
    """A real D7, so the historical arm D9 reuses is genuine output."""
    import scripts.run_graph_context_d7 as d7_module
    from test_graph_context_d7 import d7_args

    with synthetic(data):
        seven = d7_args(tmp_path_factory.mktemp("d9_d7"), data, d6_result)
        d7_module.run(seven)
    return Path(seven.output)


@contextmanager
def _forced(name, decide):
    """Force one of the two aborts to a fixed answer, keeping the real numbers.

    At synthetic scale the branch-diversity construction and the historical
    build are within milliseconds of each other -- the pure-Python `njit`
    fallback is used here, numba is not installed -- so which side of the cost
    gate a run lands on is a race. The branch is therefore chosen explicitly in
    the fixtures and the gate's own logic is tested directly against its
    thresholds, rather than left to the clock.
    """
    import scripts.run_graph_context_d9 as module

    original = getattr(module, name)

    def patched(*a, **k):
        return decide(original(*a, **k))

    setattr(module, name, patched)
    try:
        yield
    finally:
        setattr(module, name, original)


def _cheap(report):
    return report | {"construction_dominates_the_build": False}


@pytest.fixture(scope="module")
def completed(tmp_path_factory, data, d7_result):
    with synthetic(data), _forced("cost_gate", _cheap):
        return run(d9_args(tmp_path_factory.mktemp("d9"), data, d7_result))


@pytest.fixture(scope="module")
def aborted_by_cost(tmp_path_factory, data, d7_result):
    with synthetic(data), _forced(
        "cost_gate", lambda report: report | {"construction_dominates_the_build": True}
    ):
        return run(d9_args(tmp_path_factory.mktemp("d9_cost"), data, d7_result))


@pytest.fixture(scope="module")
def stopped_at_the_gate(tmp_path_factory, data, d7_result):
    with synthetic(data), _forced("cost_gate", _cheap), _forced(
        "apply_the_frozen_gate", lambda report: report | {"fired": True}
    ):
        return run(d9_args(tmp_path_factory.mktemp("d9_gate"), data, d7_result))


@pytest.fixture(scope="module")
def quantities(data):
    """The path-family quantities over the synthetic substrate."""
    from mp_retrieval.graph_context import build_operators

    rowptr = data.rowptr.numpy()
    col = data.col.numpy()
    size = int(data.num_nodes)
    operators = build_operators(rowptr, col, size)
    return build_path_diversity(
        data.queries, rowptr, col, size, operators,
        latency=[], shared_latency=[], support_latency=[],
    )


@pytest.fixture(scope="module")
def blocks(data, quantities):
    """B, H and V over the synthetic substrate, plus the pieces they came from."""
    local, ptr = local_block(data, CONTEXT)
    local = np.asarray(local)
    dense = np.stack([q.candidate_index.numpy()[DENSE_SLICE] for q in data.queries])
    splade = np.stack([q.candidate_index.numpy()[SPLADE_SLICE] for q in data.queries])
    values, _prior = prior_value_columns(
        data.queries, ptr, dense, splade, constant=RRF_CONSTANT, dtype=local.dtype
    )
    keep = tuple(sorted(DISTANCE_COLUMNS + tuple(PATH_COLUMNS)))
    base = d6_local_block(local, values, keep=DISTANCE_COLUMNS)
    full = d6_local_block(local, values, keep=HISTORICAL_COLUMNS)
    historical = d6_local_block(local, values, keep=keep)
    diversity = quantities["diversity"]
    replacement = d9_local_block(local, values, diversity)
    return {
        "local": local,
        "ptr": ptr,
        "values": values,
        "base": base,
        "full": full,
        "historical": historical,
        "replacement": replacement,
        "diversity": diversity,
        "occupancy": residual_column_occupancy(full),
    }


def _exact(blocks, **overrides):
    payload = {
        "replacement": blocks["replacement"],
        "historical": blocks["historical"],
        "base": blocks["base"],
        "prior": blocks["values"],
        "diversity": blocks["diversity"],
        "historical_columns": blocks["local"][:, list(PATH_COLUMNS)],
        "occupancy": blocks["occupancy"],
    }
    payload.update(overrides)
    return assert_d9_block_is_exact(**payload)


# --- the audit ---------------------------------------------------------------


def test_the_audit_reports_the_recursion_not_the_names():
    audit = historical_paths_audit()
    assert audit["column_indices"] == list(PATH_COLUMNS)
    assert audit["column_names"] == list(HISTORICAL_NAMES)
    assert "w_h[target] += w_{h-1}[source]" in audit["recursion"]
    assert "WALKS" in audit["raw_quantity"]
    assert "not a path count" in audit["raw_quantity"]
    assert "not a seed count" in audit["raw_quantity"]
    assert audit["lengths_are_exact_not_cumulative"] is True
    assert "not_relied_on_by_name" in audit


def test_the_audit_answers_every_question_that_was_asked_of_it():
    audit = historical_paths_audit()
    for key in (
        "what_constitutes_a_path",
        "source_nodes",
        "target_nodes",
        "direction_convention",
        "seeds_initiate_all_walks",
        "endpoints_can_be_seeds",
        "repeated_vertices_allowed",
        "repeated_edges_allowed",
        "reciprocal_edges_double_count",
        "parallel_edges_multiply_counts",
        "self_loops_contribute",
        "walks_through_the_same_predecessor_counted_separately",
        "normalisation",
        "dtype",
        "overflow_behaviour",
        "computation_path",
    ):
        assert audit[key], key


def test_the_audit_names_every_step_of_the_computation_path():
    path = historical_paths_audit()["computation_path"]
    assert any("build_local_features" in step for step in path)
    assert any("qls_local_features" in step for step in path)
    assert any("_local_feature_chunk" in step for step in path)
    assert any("candidate_readout" in step for step in path)


def test_the_audit_records_the_three_routes_to_inflation():
    audit = historical_paths_audit()
    assert audit["parallel_edges_multiply_counts"] is True
    assert "hop 2" in audit["reciprocal_edges_double_count"]
    assert "every hop" in audit["self_loops_contribute"]
    assert "no distinctness" in audit["walks_through_the_same_predecessor_counted_separately"]


def test_the_audit_records_the_asymmetry_d7_could_not_see():
    """PATHS is directed; seed_connections is not. D7's decomposition hid that."""
    audit = historical_paths_audit()
    assert "undirected in effect" in audit["direction_differs_from_seed_connections"]
    assert "D7" in audit["direction_differs_from_seed_connections"]


def test_the_historical_columns_are_all_normalised_columns():
    """The audit rests on this; if the frozen kernel changed, D9's premise is wrong."""
    for column in PATH_COLUMNS:
        assert column in NORMALISED_COLUMNS


def test_the_overflow_boundary_is_stated_rather_than_assumed_safe():
    audit = historical_paths_audit()
    assert "2**24" in audit["overflow_behaviour"]
    assert "float32" == audit["dtype"]


# --- the replacement definition ----------------------------------------------


def test_the_three_concepts_are_kept_separate():
    concepts = replacement_definition()["three_concepts_kept_separate"]
    assert set(concepts) == {"REACH", "BRANCH", "WALK"}
    assert "SEEDS" in concepts["REACH"]
    assert "PREDECESSORS" in concepts["BRANCH"]
    assert "walks" in concepts["WALK"]


def test_the_definition_refuses_multi_hop_seed_reach_by_name():
    """The specific correction this stage exists to honour."""
    why = replacement_definition()["why_not_reach"]
    assert "distinct seeds within k hops" in why
    assert "SUPPORT" in why
    assert "larger radius" in why


def test_the_injected_scalar_is_branch_not_reach():
    definition = replacement_definition()
    assert definition["names"] == list(REPLACEMENT_NAMES)
    assert definition["replaces"] == list(HISTORICAL_NAMES)
    assert definition["canonical_scalar"] == (
        "branch_diversity_h(d) = branch_h(d) / (branch_h(d) + 1)"
    )
    assert "distinct predecessors" in definition["masks"]


def test_the_transform_is_defended_as_strictly_monotone():
    """If the transform reordered anything, divergence would not be attributable."""
    why = replacement_definition()["why_the_saturating_form"]
    assert "Strictly monotone" in why
    assert "attributable to the counting rule" in why
    assert "[0,1]" in why


def test_the_rejected_alternatives_are_recorded_with_reasons():
    rejected = replacement_definition()["alternatives_rejected"]
    assert "branch_over_indegree" in rejected
    assert "purity" in rejected["branch_over_indegree"]
    assert "unbounded" in rejected["raw_branch_count"]


def test_the_algorithm_is_three_bounded_passes_and_forbids_enumeration():
    algorithm = replacement_definition()["algorithm"]
    assert algorithm["fixed_graph_passes"] == HOPS == 3
    assert algorithm["iterates_to_convergence"] is False
    forbidden = " ".join(algorithm["forbidden_and_not_implemented"]).lower()
    for banned in ("simple-path enumeration", "dfs", "max-flow", "motif"):
        assert banned in forbidden


def test_only_one_replacement_is_defined():
    banned = replacement_definition()["only_one_replacement_here"].lower()
    for absent in ("weighted support", "support-plus-path", "ppr replacement", "motif"):
        assert absent in banned


# --- the verdict table -------------------------------------------------------


def _delta(**overrides):
    delta = dict.fromkeys(METRICS, 0.0)
    delta.update(overrides)
    return delta


@pytest.mark.parametrize(
    ("delta", "label"),
    [
        (_delta(), PARETO_MATCHES),
        (_delta(**{"recall@5": MATERIAL}), IMPROVES),
        (_delta(**{"recall@5": MATERIAL - 1e-9}), PARETO_MATCHES),
        (_delta(**{"recall@5": -MATERIAL}), FAILS),
        (_delta(**{"recall@5": -MATERIAL + 1e-9}), PARETO_MATCHES),
        (_delta(**{"recall@5": MATERIAL, "mrr": -MATERIAL}), PARETO_MATCHES),
        (_delta(**{"recall@1": MATERIAL, "recall@20": MATERIAL}), IMPROVES),
        (_delta(**{"recall@1": -MATERIAL, "mrr": -MATERIAL}), FAILS),
    ],
)
def test_the_verdict_boundaries_are_where_they_were_registered(delta, label):
    assert classify(delta)["label"] == label


def test_a_mixed_result_is_never_reported_as_an_improvement():
    verdict = classify(_delta(**{"recall@5": 0.02, "mrr": -0.02}))
    assert verdict["label"] == PARETO_MATCHES
    assert verdict["mixed"] is True
    assert verdict["materially_better_on"] == ["recall@5"]
    assert verdict["materially_worse_on"] == ["mrr"]


def test_every_verdict_carries_the_reading_registered_for_it():
    verdict = classify(_delta())
    assert "QLS-v2 candidate" in verdict["if_improves"]
    assert "Pareto grounds" in verdict["if_pareto_matches"]
    assert "Do NOT add walk counts back" in verdict["if_fails"]
    assert "not a" in verdict["not_a_significance_claim"]
    assert verdict["increment"] == DELTA_V_MINUS_H


def test_the_failure_reading_says_what_two_failures_would_mean():
    """§11(C): SUPPORT and PATHS both failing under correction is itself evidence."""
    reading = classify(_delta(**{"recall@5": -0.02}))["if_fails"]
    assert "SUPPORT and PATHS" in reading
    assert "further structural engineering" in reading


# --- the block, column by column ---------------------------------------------


def test_the_replacement_block_is_the_historical_block_with_three_columns_swapped(blocks):
    report = _exact(blocks)
    assert report["differs_from_historical_in"] == list(HISTORICAL_NAMES)
    assert report["max_abs_diff"] == 0.0
    assert report["other_residual_columns_are_zero"] is True


def test_the_block_keeps_the_frozen_width(blocks):
    assert blocks["replacement"].shape[1] == LOCAL_DIM
    assert blocks["historical"].shape[1] == LOCAL_DIM
    assert blocks["base"].shape[1] == LOCAL_DIM


def test_the_distance_geometry_is_bit_identical_across_all_three_arms(blocks):
    for column in DISTANCE_COLUMNS:
        assert np.array_equal(blocks["replacement"][:, column], blocks["base"][:, column])
        assert np.array_equal(blocks["replacement"][:, column], blocks["historical"][:, column])


def test_the_prior_is_bit_identical_across_all_three_arms(blocks):
    for offset, column in enumerate(PRIOR_COLUMNS):
        assert np.array_equal(blocks["replacement"][:, column], blocks["values"][:, offset])
        assert np.array_equal(blocks["replacement"][:, column], blocks["historical"][:, column])


def test_the_columns_the_replacement_does_not_touch_are_zero_in_both_arms(blocks):
    untouched = [c for c in RESIDUAL_COLUMNS if c not in PATH_COLUMNS]
    for column in untouched:
        assert not blocks["replacement"][:, column].any()
        assert not blocks["historical"][:, column].any()


def test_the_injected_columns_are_the_branch_saturations(blocks):
    stored = blocks["diversity"].astype(blocks["replacement"].dtype)
    for offset, column in enumerate(PATH_COLUMNS):
        assert np.array_equal(blocks["replacement"][:, column], stored[:, offset])


def test_the_storage_precision_is_measured_rather_than_absorbed(blocks):
    """float16 storage is a real ceiling; it is reported, not hidden in a tolerance."""
    storage = _exact(blocks)["storage_precision"]
    assert storage["block_dtype"] == str(blocks["replacement"].dtype)
    assert storage["max_abs_cast_error"] > 0.0
    assert storage["distinct_scalar_values_after_cast"] <= storage[
        "distinct_scalar_values_before_cast"
    ]
    assert storage["largest_branch_count_still_distinguishable_from_its_successor"] > 0
    assert "does not advantage" in storage["the_historical_columns_pay_the_same_cost"]


def test_a_replacement_that_also_moved_the_prior_is_refused(blocks):
    broken = np.array(blocks["replacement"], copy=True)
    broken[0, PRIOR_COLUMNS[0]] += 0.5
    with pytest.raises(RuntimeError, match="prior"):
        _exact(blocks, replacement=broken)


def test_a_replacement_that_also_moved_the_geometry_is_refused(blocks):
    broken = np.array(blocks["replacement"], copy=True)
    broken[0, DISTANCE_COLUMNS[0]] += 0.5
    with pytest.raises(RuntimeError, match="distance"):
        _exact(blocks, replacement=broken)


def test_a_replacement_that_woke_a_column_it_does_not_own_is_refused(blocks):
    untouched = [c for c in RESIDUAL_COLUMNS if c not in PATH_COLUMNS]
    broken = np.array(blocks["replacement"], copy=True)
    broken[0, untouched[0]] = 0.5
    with pytest.raises(RuntimeError, match="not exactly"):
        _exact(blocks, replacement=broken)


def test_a_replacement_carrying_the_historical_values_is_refused(blocks):
    """An arm that quietly kept the walk counts would still be 'a 13-column arm'."""
    with pytest.raises(RuntimeError, match=r"differs from D7_PATHS_13 in \[\]"):
        _exact(blocks, replacement=blocks["historical"])


# --- injection discipline ----------------------------------------------------


def _prior_for(data, ptr, dtype):
    dense = np.stack([q.candidate_index.numpy()[DENSE_SLICE] for q in data.queries])
    splade = np.stack([q.candidate_index.numpy()[SPLADE_SLICE] for q in data.queries])
    values, _prior = prior_value_columns(
        data.queries, ptr, dense, splade, constant=RRF_CONSTANT, dtype=dtype
    )
    return values


def test_the_injection_reaches_the_learner_unrescaled(blocks, data):
    checks = assert_the_injection_was_not_rescaled(
        blocks["replacement"], blocks["diversity"], blocks["ptr"]
    )
    assert checks["elementwise_identical"] is True
    assert checks["the_replacement_bypasses_the_normaliser"] is True
    assert checks["normalised_columns_extended_to_the_replacement"] is False
    assert set(checks["columns"]) == set(REPLACEMENT_NAMES)


def test_a_second_per_query_max_rescaling_is_caught(data, quantities):
    """The exact mistake the discipline exists to prevent."""
    local, ptr = local_block(data, CONTEXT)
    local = np.asarray(local)
    values = _prior_for(data, ptr, local.dtype)
    rescaled = np.asarray(quantities["diversity"], dtype=np.float64).copy()
    moved = False
    for offset in range(HOPS):
        for start, end in zip(ptr[:-1], ptr[1:], strict=True):
            window = rescaled[start:end, offset]
            maximum = window.max() if window.size else 0.0
            if 0.0 < maximum < 1.0:
                rescaled[start:end, offset] = window / maximum
                moved = True
    if not moved:
        pytest.skip("no query on this substrate has a sub-maximal diversity column")
    block = d9_local_block(local, values, rescaled)
    with pytest.raises(RuntimeError, match="not receiving"):
        assert_the_injection_was_not_rescaled(block, quantities["diversity"], ptr)


def test_the_gate_is_the_elementwise_check_not_the_maxima(blocks):
    checks = assert_the_injection_was_not_rescaled(
        blocks["replacement"], blocks["diversity"], blocks["ptr"]
    )
    assert "elementwise equality" in checks["which_check_is_the_gate"]
    assert "column maximum of" in checks["how_a_second_rescaling_would_show"]


def test_a_query_relative_column_would_report_every_maximum_at_one(blocks):
    """The corroborating statistic: an intrinsic column does not saturate per query."""
    checks = assert_the_injection_was_not_rescaled(
        blocks["replacement"], blocks["diversity"], blocks["ptr"]
    )
    below = sum(
        entry["queries_whose_column_maximum_is_below_one"]
        for entry in checks["columns"].values()
    )
    occupied = sum(
        entry["queries_with_any_diversity"] for entry in checks["columns"].values()
    )
    assert occupied > 0
    assert below == occupied, "branch/(branch+1) can never reach 1.0"


# --- the replacement kernel over real query objects --------------------------


def test_the_quantities_line_up_with_the_candidate_rows(data, quantities, blocks):
    rows = int(blocks["ptr"][-1])
    for key in ("branch", "reach", "walks", "diversity"):
        assert quantities[key].shape == (rows, HOPS), key
    assert quantities["distinct_indegree"].shape == (rows,)
    assert quantities["d8_distinct_support"].shape == (rows,)
    assert quantities["seeds_per_query"].shape == (len(data.queries),)


def test_branch_never_exceeds_the_walk_count(quantities):
    assert np.all(quantities["branch"] <= quantities["walks"] + 1e-9)


def test_branch_never_exceeds_the_distinct_indegree(quantities):
    for offset in range(HOPS):
        assert np.all(quantities["branch"][:, offset] <= quantities["distinct_indegree"])


def test_a_branch_implies_a_walk(quantities):
    """Structural, and the reason the nonzero-agreement gate condition is weak."""
    assert np.all(quantities["walks"][quantities["branch"] > 0] > 0)


def test_reach_never_exceeds_the_seed_count(data, quantities, blocks):
    ptr = blocks["ptr"]
    for index in range(len(data.queries)):
        window = quantities["reach"][ptr[index] : ptr[index + 1]]
        assert np.all(window <= quantities["seeds_per_query"][index])


def test_the_diversity_is_the_saturation_of_the_branch_count(quantities):
    branch = quantities["branch"].astype(np.float64)
    assert np.allclose(quantities["diversity"], branch / (branch + 1.0))


def test_the_workspace_is_bounded_and_recorded(quantities):
    assert quantities["temporary_workspace_bytes"] > 0
    assert quantities["temporary_workspace_bytes"] < 1 << 20


# --- the mechanistic comparison ----------------------------------------------


@pytest.fixture(scope="module")
def comparison(blocks, quantities):
    return mechanistic_gate(
        blocks["diversity"],
        blocks["local"][:, list(PATH_COLUMNS)],
        quantities,
        blocks["ptr"],
    )


def test_the_comparison_reports_every_declared_quantity(comparison):
    assert set(comparison["columns"]) == {
        f"{historical}_vs_{replacement}"
        for historical, replacement in zip(HISTORICAL_NAMES, REPLACEMENT_NAMES, strict=True)
    }
    for entry in comparison["columns"].values():
        for key in (
            "pooled_spearman",
            "pearson",
            "per_query_spearman",
            "spearman_on_supported_rows_only",
            "rows_differing_numerically",
            "nonzero_agreement",
            "ordering",
            "value_distribution",
        ):
            assert key in entry, key


def test_the_comparison_is_made_on_the_post_transform_values(blocks, comparison):
    """§7: the values actually presented to the learner, not the raw counts."""
    entry = comparison["columns"]["paths_length_1_vs_branch_diversity_1"]
    assert entry["column_index"] == PATH_COLUMNS[0]
    presented = blocks["local"][:, PATH_COLUMNS[0]]
    assert entry["rows_nonzero_historical"] == int((np.asarray(presented) > 0).sum())


def test_every_pair_is_accounted_for_exactly_once(comparison):
    for entry in comparison["columns"].values():
        ordering = entry["ordering"]
        assert (
            ordering["concordant"]
            + ordering["strictly_reversed"]
            + ordering["tied_by_the_historical_column_only"]
            + ordering["tied_by_distinct_support_only"]
            + ordering["tied_by_both"]
        ) == ordering["within_query_candidate_pairs"]


def test_the_ordering_statistic_agrees_with_a_direct_count(blocks, quantities):
    """The one gate condition with teeth, checked against a brute-force count."""
    ptr = np.asarray(blocks["ptr"], dtype=np.int64)
    walks = np.asarray(quantities["walks"], dtype=np.float64)[:, 0]
    branch = np.asarray(quantities["branch"], dtype=np.int64)[:, 0]
    pairs = differ = 0
    for start, end in zip(ptr[:-1], ptr[1:], strict=True):
        for i in range(start, end):
            for j in range(i + 1, end):
                pairs += 1
                left = np.sign(walks[i] - walks[j])
                right = np.sign(branch[i] - branch[j])
                if left != right:
                    differ += 1
    entry = mechanistic_gate(
        quantities["diversity"], blocks["local"][:, list(PATH_COLUMNS)],
        quantities, blocks["ptr"],
    )["columns"]["paths_length_1_vs_branch_diversity_1"]["ordering"]
    assert entry["within_query_candidate_pairs"] == pairs
    assert entry["ordered_differently"] == differ


def test_the_block_level_comparison_does_not_force_a_column_mapping(comparison):
    """§9: report the 3-d block as a block, since the semantics are not one-to-one."""
    block = comparison["block_level"]
    assert "one-to-one column mapping" in block["measures"]
    for key in (
        "pooled_spearman_of_the_l2_norm",
        "per_query_spearman_of_the_l2_norm",
        "rows_where_exactly_one_block_is_empty",
    ):
        assert key in block


def test_reach_and_branch_are_reported_separately(comparison):
    """§3: the three concepts must not collapse into one statistic."""
    concepts = comparison["concept_separation"]
    assert set(concepts["reach_vs_branch"]) == set(HISTORICAL_NAMES)
    assert "D8's support feature" in concepts["why"]
    for entry in concepts["reach_vs_branch"].values():
        assert "mean_reach" in entry
        assert "mean_branch" in entry


def test_the_overlap_with_d8_support_is_diagnostic_only(comparison):
    """§12: report the correlation, do not train the composition."""
    overlap = comparison["concept_separation"]["branch_vs_d8_distinct_support"]
    assert "Diagnostic only" in overlap["why"]
    assert "does not train support and diversity together" in overlap["why"]
    # "computed" records whether D8's column was rebuilt at all; a caller that
    # skipped it gets None per column rather than a correlation against zeros.
    assert set(overlap) - {"why", "computed"} == set(REPLACEMENT_NAMES)
    assert overlap["computed"] is True


def test_the_diamond_case_separates_reach_from_branch():
    """If BRANCH were REACH the replacement would be D8 at a larger radius.

    One seed, two disjoint two-hop routes into the sink: one seed reaches it,
    through two distinct predecessors. REACH cannot tell that apart from a
    single route; BRANCH can, and that is the whole reason a path family is
    separate from a support family.
    """
    from mp_retrieval.path_diversity import path_diversity

    edges = np.array([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=np.int64)
    result = path_diversity(edges, 4, np.array([0], dtype=np.int64), hops=2)
    assert result["reach"][1, 3] == 1
    assert result["branch"][1, 3] == 2


def test_the_pilot_substrate_does_not_exercise_that_separation(quantities):
    """Recorded, not hidden: on this synthetic graph REACH and BRANCH coincide.

    Two seeds and a sparse graph give every candidate at most one predecessor
    carrying evidence, so the separation above is real but invisible here. The
    stage reports both quantities on the real substrate rather than assuming
    2Wiki behaves like either graph.
    """
    assert np.array_equal(quantities["reach"], quantities["branch"])


# --- the frozen gate ---------------------------------------------------------


def _gate_entry(**overrides):
    entry = {
        "pooled_spearman": 1.0,
        "ordering": {"fraction_ordered_differently": 0.0},
        "nonzero_agreement": 1.0,
    }
    entry.update(overrides)
    return entry


def _gate(**columns):
    return apply_the_frozen_gate({"columns": columns or {"a": _gate_entry()}})


def test_the_thresholds_are_the_ones_that_were_filed():
    """Registered in configs/graph_context_pilot.yaml before the gate was run."""
    assert SPEARMAN_ABS_MIN == 0.995
    assert ORDERING_CHANGE_MAX == 0.0025
    assert NONZERO_AGREEMENT_MIN == 0.995
    assert _gate()["thresholds"] == {
        "spearman_abs_min": SPEARMAN_ABS_MIN,
        "ordering_change_max": ORDERING_CHANGE_MAX,
        "nonzero_agreement_min": NONZERO_AGREEMENT_MIN,
    }


def test_the_gate_fires_only_when_all_three_conditions_hold():
    assert _gate(a=_gate_entry())["fired"] is True


@pytest.mark.parametrize(
    "override",
    [
        {"pooled_spearman": SPEARMAN_ABS_MIN - 1e-6},
        {"ordering": {"fraction_ordered_differently": ORDERING_CHANGE_MAX}},
        {"nonzero_agreement": NONZERO_AGREEMENT_MIN},
    ],
)
def test_one_failing_condition_is_enough_to_train(override):
    result = _gate(a=_gate_entry(**override))
    assert result["fired"] is False
    assert result["columns_that_diverge_materially"] == ["a"]


@pytest.mark.parametrize(
    "override",
    [
        {"pooled_spearman": SPEARMAN_ABS_MIN},
        {"ordering": {"fraction_ordered_differently": ORDERING_CHANGE_MAX - 1e-9}},
        {"nonzero_agreement": NONZERO_AGREEMENT_MIN + 1e-9},
    ],
)
def test_the_boundaries_are_where_they_were_registered(override):
    assert _gate(a=_gate_entry(**override))["fired"] is True


def test_one_diverging_column_out_of_three_is_enough_to_train():
    result = _gate(
        a=_gate_entry(),
        b=_gate_entry(ordering={"fraction_ordered_differently": 0.5}),
        c=_gate_entry(),
    )
    assert result["fired"] is False
    assert result["columns_that_diverge_materially"] == ["b"]


def test_a_negative_spearman_is_read_as_divergence_not_as_agreement():
    """A perfectly reversed column is maximally different, not maximally similar."""
    assert _gate(a=_gate_entry(pooled_spearman=-1.0))["fired"] is True
    entry = _gate(a=_gate_entry(pooled_spearman=-1.0))["per_column"]["a"]
    assert entry["abs_pooled_spearman"] == 1.0


def test_an_undefined_spearman_never_fires_the_gate():
    """A constant column cannot be evidence that two signals agree."""
    assert _gate(a=_gate_entry(pooled_spearman=None))["fired"] is False
    assert _gate(a=_gate_entry(pooled_spearman=float("nan")))["fired"] is False


def test_the_weakness_of_two_of_the_three_conditions_was_recorded_in_advance():
    """§8 honesty: a near-1.0 figure must not read as evidence of equivalence."""
    note = _gate()["expected_weakness_recorded_in_advance"]
    assert "implies" in note
    assert "zero mass" in note
    assert "ordering condition is the one with teeth" in note


def test_the_gate_names_which_spearman_it_reads():
    """The filed threshold did not say; the runner names it before measuring."""
    assert "pooled" in _gate()["which_spearman"]
    assert "validation" in _gate()["which_spearman"]


# --- the run, end to end -----------------------------------------------------


def test_the_stage_completes_and_trains_exactly_one_arm(completed):
    assert completed["status"] == COMPLETE_STATUS
    assert completed["arms_trained_here"] == [REPLACEMENT_ARM]
    assert completed["arms_reused"] == list(REUSED_ARMS)
    assert completed["results"][REPLACEMENT_ARM]["retrained_here"] is True
    for arm in REUSED_ARMS:
        assert completed["results"][arm]["retrained_here"] is False


def test_the_gate_did_not_fire_on_this_substrate(completed):
    gate = completed["frozen_gate"]
    assert gate["fired"] is False
    assert gate["columns_that_diverge_materially"]


def test_every_arm_shares_the_architecture(completed):
    parameters = {
        arm: completed["results"][arm]["parameters"]
        for arm in (BASE, HISTORICAL_ARM, REPLACEMENT_ARM)
    }
    assert len(set(parameters.values())) == 1
    widths = {
        arm: completed["results"][arm]["local_dim"]
        for arm in (BASE, HISTORICAL_ARM, REPLACEMENT_ARM)
    }
    assert set(widths.values()) == {LOCAL_DIM}
    assert completed["ablation"]["columns_added"] == 0


def test_the_three_increments_are_reported(completed):
    assert set(completed["increments"]) == {
        DELTA_H_MINUS_B, DELTA_V_MINUS_B, DELTA_V_MINUS_H
    }
    assert completed["increments"][DELTA_V_MINUS_H]["from"] == HISTORICAL_ARM
    assert completed["increments"][DELTA_V_MINUS_H]["to"] == REPLACEMENT_ARM


def test_the_increments_are_consistent_with_the_ladder(completed):
    ladder = completed["ladder"]
    for name, frm, to in (
        (DELTA_H_MINUS_B, BASE, HISTORICAL_ARM),
        (DELTA_V_MINUS_B, BASE, REPLACEMENT_ARM),
        (DELTA_V_MINUS_H, HISTORICAL_ARM, REPLACEMENT_ARM),
    ):
        for key in METRICS:
            if key in ladder[BASE]:
                assert completed["increments"][name][key] == pytest.approx(
                    ladder[to][key] - ladder[frm][key]
                )


def test_the_carried_h_minus_b_reproduces_d7(completed, d7_result):
    d7 = json.loads(Path(d7_result).read_text(encoding="utf-8"))
    for key in METRICS:
        if key in d7["increments"]["delta_paths"]:
            assert completed["increments"][DELTA_H_MINUS_B][key] == pytest.approx(
                d7["increments"]["delta_paths"][key], abs=1e-9
            )


def test_the_verdict_is_recomputed_from_the_increment(completed):
    assert completed["verdict"]["label"] == classify(
        completed["increments"][DELTA_V_MINUS_H]
    )["label"]
    assert completed["verdict"]["trained"] is True


def test_the_reuse_conditions_all_matched(completed):
    assert completed["reuse"]["all_conditions_match"] is True
    assert completed["deterministic_reproduction"]["all_conditions_match"] is True


def test_the_reported_comparison_is_the_validation_block(completed):
    comparison = completed["mechanistic_comparison"]
    assert comparison["measured_on"] == "validation feature construction"
    assert comparison["seeds_per_query"]["queries"] == completed["splits"][
        "validation_reported"
    ]
    everything = comparison["also_over_every_opened_query"]
    assert everything["measured_on"] == "every opened query"
    assert everything["seeds_per_query"]["queries"] == sum(
        completed["splits"][key]
        for key in ("train_fit", "train_holdout_for_epoch_selection", "validation_reported")
    )
    assert comparison["rows"] < everything["rows"]


def test_the_stage_reports_by_stratum_for_both_comparisons(completed):
    assert set(completed["increments_by_stratum"]) == {DELTA_V_MINUS_H, DELTA_V_MINUS_B}
    for block in completed["increments_by_stratum"].values():
        assert block
        for stratum in block.values():
            assert stratum["queries"] > 0


# --- the two aborts ----------------------------------------------------------


def test_the_cost_gate_fires_only_when_construction_dominates():
    assert cost_gate(1.0, 2.0)["construction_dominates_the_build"] is False
    assert cost_gate(2.0, 2.0)["construction_dominates_the_build"] is False
    assert cost_gate(2.0 + 1e-9, 2.0)["construction_dominates_the_build"] is True
    assert "regardless of representation divergence" in cost_gate(1.0, 2.0)["rule"]


def test_the_microbenchmark_abort_spends_no_training_run(aborted_by_cost):
    assert aborted_by_cost["status"] == COST_STATUS
    assert aborted_by_cost["arms_trained_here"] == []
    assert aborted_by_cost["results"] == {}
    assert aborted_by_cost["verdict"]["label"] == UNINFORMATIVE
    assert aborted_by_cost["verdict"]["trained"] is False
    assert "No effectiveness claim" in aborted_by_cost["not_established"]


def test_the_microbenchmark_abort_still_reports_the_mechanism(aborted_by_cost):
    """Stopping is a result; the representation measurement is not discarded."""
    assert aborted_by_cost["mechanistic_comparison"]["columns"]
    assert aborted_by_cost["historical_paths_audit"]["column_names"] == list(HISTORICAL_NAMES)
    assert aborted_by_cost["tensor_equivalence"]["max_abs_diff"] == 0.0


def test_the_cost_gate_is_read_before_the_mechanistic_gate(aborted_by_cost):
    """§18 step 4 precedes step 5: a construction that costs too much never trains."""
    assert "frozen_gate" not in aborted_by_cost


def test_the_mechanistic_gate_abort_spends_no_training_run(stopped_at_the_gate):
    assert stopped_at_the_gate["status"] == GATE_STATUS
    assert stopped_at_the_gate["arms_trained_here"] == []
    assert stopped_at_the_gate["results"] == {}
    assert stopped_at_the_gate["verdict"]["label"] == UNINFORMATIVE
    assert stopped_at_the_gate["verdict"]["trained"] is False


def test_the_gate_abort_says_it_is_about_the_dataset_not_the_definitions(
    stopped_at_the_gate,
):
    why = stopped_at_the_gate["verdict"]["why"]
    assert "statement about this dataset" in why
    assert "the semantics differ, the graph does not exercise the difference" in why
    assert "parallel edges" in stopped_at_the_gate["verdict"]["what_would_change_it"]


def test_the_gate_abort_refuses_to_report_an_effectiveness_result(stopped_at_the_gate):
    assert "Nothing about effectiveness" in stopped_at_the_gate["not_established"]
    assert "could not have found one" in stopped_at_the_gate["not_established"]
    assert "ladder" not in stopped_at_the_gate
    assert "increments" not in stopped_at_the_gate


def test_the_gate_abort_still_carries_d7s_ladder_for_context(stopped_at_the_gate):
    assert stopped_at_the_gate["d7_ladder_for_reference"]


def test_the_cost_gate_is_recorded_and_did_not_fire(completed):
    assert completed["cost_gate"]["construction_dominates_the_build"] is False
    assert completed["cost_gate"]["rule"]


def test_the_incremental_cost_is_not_the_whole_pipeline(completed):
    """§13: shared graph preparation is reported separately, not charged here."""
    construction = completed["branch_diversity_construction"]
    assert construction["shared_graph_preparation_seconds"] >= 0.0
    assert construction["fixed_graph_passes"] == HOPS
    assert construction["convergence_loop"] is False
    assert "would share" in construction["what_is_incremental"]
    assert "overlap diagnostic" in construction["what_is_incremental"]
    for key in ("p50", "p95", "p99"):
        assert key in construction["incremental_latency_ms_per_query"]


def test_the_overlap_diagnostic_is_timed_apart_from_the_feature(completed):
    construction = completed["branch_diversity_construction"]
    assert "overlap_diagnostic_seconds" in construction
    assert "is not part of D9's feature" in construction["what_is_incremental"]


# --- reuse and reproduction --------------------------------------------------


def _args(completed):
    return argparse.Namespace(
        dataset=completed["dataset"],
        seed=completed["training"]["seed"],
        data_fingerprint_sha256=completed["data_fingerprint_sha256"],
        rrf_constant=completed["a3_rank_feature_audit"]["constant_K"],
    )


def test_a_d7_that_did_not_complete_is_refused(completed, d7_result):
    d7 = json.loads(Path(d7_result).read_text(encoding="utf-8"))
    d7["status"] = "GRAPH_CONTEXT_D7_IN_PROGRESS"
    with pytest.raises(RuntimeError, match="not complete"):
        verify_reuse(d7, completed, _args(completed))


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


@pytest.mark.parametrize("name", HISTORICAL_NAMES)
def test_a_drift_in_any_one_path_column_stops_the_stage(
    completed, d7_result, blocks, name
):
    """A drift confined to one replaced column must not hide in the six-column total."""
    d7 = json.loads(Path(d7_result).read_text(encoding="utf-8"))
    d7["tensor_equivalence"]["residual_column_occupancy"]["nonzero_rows_by_column"][
        name
    ] += 1
    with pytest.raises(RuntimeError, match="STOP AND INSPECT"):
        verify_substrate_reproduces_d7(d7, completed, blocks["occupancy"])


@pytest.mark.parametrize(
    "field", ["candidate_rows", "residual_nonzero_entries", "prior_rows_ranked_by_both"]
)
def test_a_substrate_that_does_not_reproduce_d7_stops_the_stage(
    completed, d7_result, blocks, field
):
    d7 = json.loads(Path(d7_result).read_text(encoding="utf-8"))
    assert completed["deterministic_reproduction"]["conditions_checked"][field]["match"]
    if field == "candidate_rows":
        d7["feature_build"]["candidate_rows"] += 1
    elif field == "residual_nonzero_entries":
        d7["tensor_equivalence"]["residual_column_occupancy"][
            "total_nonzero_entries"
        ] += 1
    else:
        d7["graded_retrieval_prior"]["rows_ranked_by_both"] += 1
    with pytest.raises(RuntimeError, match="STOP AND INSPECT"):
        verify_substrate_reproduces_d7(d7, completed, blocks["occupancy"])


# --- the contract ------------------------------------------------------------


def test_no_gnn_no_test_split_no_pool_modification(completed):
    contract = completed["contract"]
    assert contract["gnn_trained"] is False
    assert contract["message_passing"] is False
    assert contract["test_split_read"] is False
    assert contract["candidate_pools_modified"] is False
    assert contract["epoch_selected_on_validation"] is False
    assert contract["architecture_changed_between_arms"] is False
    assert completed["splits"]["test_read"] is False


def test_the_test_split_cannot_be_requested(tmp_path, data, d7_result):
    args = d9_args(tmp_path, data, d7_result)
    args.splits = ["train", "validation", "test"]
    with pytest.raises(ValueError, match="test split is not read"):
        run(args)


def test_the_normaliser_was_not_extended(completed):
    normalisation = completed["normalisation"]
    assert normalisation["normalised_columns_extended_to_the_replacement"] is False
    assert normalisation["path_and_ppr_definitions_changed"] is False
    assert "non-monotone across queries" in normalisation["why_the_replacement_is_outside_it"]


def test_the_stage_says_what_it_is_not(completed):
    """§1's correction, carried into the result file rather than only the plan."""
    assert "distinct seeds within k hops" in completed["what_this_is_not"]
    assert "D8's support feature at a larger radius" in completed["what_this_is_not"]
    assert "only BRANCH is injected" in completed["what_this_is_not"]
    assert set(completed["three_quantities_kept_separate"]) == {"REACH_h", "BRANCH_h", "WALK_h"}


def test_the_stage_does_not_claim_more_than_it_measured(completed):
    not_established = completed["not_established"]
    assert "one dataset" in not_established.lower()
    assert "one seed" in not_established.lower()
    assert "not trained here" in not_established
    assert "G[Cq]" in not_established
