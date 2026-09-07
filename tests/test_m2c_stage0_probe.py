"""The Stage-0 probe's own logic, tested where it can be tested locally.

The probe reads sealed artifacts that live on the Modal volume, so the loading
path cannot run here. What CAN run here is everything that turns loaded arrays
into a reported number -- the metrics, the residual construction, and the
admission diagnostic -- and those are where a silent error would be most
expensive, because a wrong number that runs looks exactly like a right one.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.run_m2c_stage0_probe as probe  # noqa: E402
from mp_retrieval.candidate_expansion_v2 import ExpansionBudget  # noqa: E402


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260907)


def _csr(edges, num_nodes):
    rowptr = np.zeros(num_nodes + 1, dtype=np.int64)
    for source, _ in edges:
        rowptr[source + 1] += 1
    np.cumsum(rowptr, out=rowptr)
    col = np.zeros(len(edges), dtype=np.int64)
    cursor = rowptr[:-1].copy()
    for source, target in sorted(edges):
        col[cursor[source]] = target
        cursor[source] += 1
    return rowptr, col


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_recall_is_the_share_of_relevant_items_found_not_whether_any_was():
    """A query with two golds, one of them at rank 3, is half recalled at 5."""

    ranks = np.array([3, 7, 1, 9])
    relevant = np.array([True, True, False, False])
    row = probe._metrics(ranks, relevant)

    assert row["recall@1"] == pytest.approx(0.0)
    assert row["recall@5"] == pytest.approx(0.5)
    assert row["recall@20"] == pytest.approx(1.0)
    assert row["mrr"] == pytest.approx(1 / 3), "MRR uses the best relevant rank"


def test_a_query_with_no_relevant_candidate_scores_zero_rather_than_dividing_by_zero():
    row = probe._metrics(np.array([1, 2]), np.zeros(2, dtype=bool))
    assert row["recall@5"] == 0.0 and row["mrr"] == 0.0
    assert row["scored"] == 0.0, "it is still counted, and marked as unwinnable"


def test_a_gold_at_rank_one_is_perfect_at_every_cutoff():
    row = probe._metrics(np.array([1]), np.array([True]))
    assert row["recall@1"] == row["recall@5"] == row["recall@20"] == 1.0
    assert row["mrr"] == 1.0


def test_the_mean_over_an_empty_panel_is_not_an_error():
    summary = probe._mean_metrics([])
    assert summary["queries"] == 0 and summary["recall@5"] == 0.0


def test_the_mean_is_over_queries_not_over_relevant_items():
    rows = [
        {"recall@1": 1.0, "recall@5": 1.0, "recall@20": 1.0, "mrr": 1.0},
        {"recall@1": 0.0, "recall@5": 0.0, "recall@20": 0.5, "mrr": 0.1},
    ]
    summary = probe._mean_metrics(rows)
    assert summary["queries"] == 2
    assert summary["recall@5"] == pytest.approx(0.5)
    assert summary["mrr"] == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# The three residual controls
# ---------------------------------------------------------------------------


def test_the_three_residuals_are_three_different_vectors(rng):
    query = rng.normal(size=32)
    seeds = rng.normal(size=(4, 32))
    anchor = rng.normal(size=32)

    built = probe._residual_vectors(query, seeds, anchor)
    assert set(built) == set(probe.RESIDUALS)
    vectors = [built[name]["vector"] for name in probe.RESIDUALS]
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            assert not np.allclose(vectors[i], vectors[j]), (
                f"{probe.RESIDUALS[i]} and {probe.RESIDUALS[j]} coincide; "
                "one of them is not a control"
            )


def test_the_legacy_arm_calls_the_frozen_function_rather_than_a_lookalike(rng):
    """If this drifts, the arm is no longer the mechanism M0A nulled."""

    from mp_retrieval.candidate_expansion_v2 import query_residual

    query = rng.normal(size=32)
    anchor = rng.normal(size=32)
    built = probe._residual_vectors(query, rng.normal(size=(3, 32)), anchor)
    assert np.allclose(
        built["R1_LEGACY_DIRECTIONAL"]["vector"], query_residual(query, anchor)
    )


def test_a_fallback_is_recorded_as_a_fallback_not_as_a_control(rng):
    """R2 degenerating to normalize(q) coincides with R0. The records must not."""

    seeds = rng.normal(size=(3, 32))
    query = seeds.T @ rng.normal(size=3)  # lies in the seed span
    built = probe._residual_vectors(query, seeds, rng.normal(size=32))

    assert np.allclose(
        built["R2_SEED_SUBSPACE"]["vector"], built["R0_RAW_QUERY_CONTROL"]["vector"]
    )
    assert built["R2_SEED_SUBSPACE"]["fell_back_to_query"] is True
    assert built["R0_RAW_QUERY_CONTROL"]["fell_back_to_query"] is False


# ---------------------------------------------------------------------------
# The +64 admission diagnostic
# ---------------------------------------------------------------------------


def _query(index, pool, golds, seed_positions):
    return SimpleNamespace(
        query_index=index,
        query_id=f"q{index}",
        candidate_index=torch.tensor(pool, dtype=torch.long),
        relevant_global=torch.tensor(golds, dtype=torch.long),
        retrieval_seed_local=torch.tensor(seed_positions, dtype=torch.long),
        anchor_global=int(pool[0]),
        split=1,
    )


@pytest.fixture
def admission_fixture(rng):
    num_nodes = 80
    pool = np.arange(20)
    edges = [
        (int(s), int(v))
        for s in (0, 1, 2)
        for v in range(20, num_nodes)
        if rng.random() < 0.25
    ]
    rowptr, col = _csr(edges, num_nodes)
    panel = [_query(i, pool, [25, 40], [0, 1, 2]) for i in range(6)]
    return {
        "panel": panel,
        "node_array": rng.normal(size=(num_nodes, 24)),
        "query_array": rng.normal(size=(len(panel), 24)),
        "rowptr": rowptr,
        "col": col,
        "num_nodes": num_nodes,
        "budget": ExpansionBudget(
            per_seed_cap=16, graph_expansion_cap=64, neighbour_scan_cap_per_seed=4096
        ),
        "cap": 0,
        "mainline_family": "structural_only",
        "stored_was_symmetric": False,
    }


def test_the_diagnostic_reports_all_three_arms_at_one_budget(admission_fixture):
    result = probe._admission_diagnostic(**admission_fixture)

    assert result["arms"] == list(probe.ADMISSION_ARMS)
    assert result["budget"] == 64
    assert "budget-matched rather than budget-buying" in result["why_64_only"]
    assert set(result["recall_ceiling_at_5"]) == set(probe.ADMISSION_ARMS)
    assert result["queries"] == len(admission_fixture["panel"])


def test_the_bound_reported_is_the_k_aware_ceiling_not_pool_coverage(admission_fixture):
    """Substituting candidate_ceiling here is the exact error §3 exists to stop."""

    result = probe._admission_diagnostic(**admission_fixture)
    assert "recall_ceiling_at_5" in result
    assert "candidate_ceiling" not in result["recall_ceiling_at_5"]
    assert "K-aware" in result["candidate_ceiling_is_not_used_as_the_r5_bound"]
    for value in result["recall_ceiling_at_5"].values():
        assert 0.0 <= value <= 1.0


def test_the_a64_arm_is_its_own_baseline_so_its_delta_is_exactly_zero(admission_fixture):
    result = probe._admission_diagnostic(**admission_fixture)
    assert result["delta_recall_ceiling_at_5_pp_versus_a64"]["A64"] == 0.0


def test_overlap_is_reported_for_every_pair_including_the_two_residual_arms(
    admission_fixture,
):
    """legacy vs seed-subspace is the pair that carries the new variable."""

    result = probe._admission_diagnostic(**admission_fixture)
    overlaps = result["pairwise_overlap"]
    assert set(overlaps) == {
        "A64|legacy_PF64",
        "A64|seed_subspace_PF64",
        "legacy_PF64|seed_subspace_PF64",
    }
    for pair in overlaps.values():
        assert 0.0 <= pair["jaccard"] <= 1.0
        assert pair["intersection"] <= min(pair["left_size"], pair["right_size"])
        assert pair["union"] == pair["left_size"] + pair["right_size"] - pair["intersection"]


def test_the_cap_truncates_the_panel_and_is_reported(admission_fixture):
    capped = probe._admission_diagnostic(**{**admission_fixture, "cap": 2})
    assert capped["queries"] == 2


def test_relevant_admissions_are_counted_against_the_query_own_golds(admission_fixture):
    result = probe._admission_diagnostic(**admission_fixture)
    counts = result["unique_relevant_admissions"]
    assert set(counts) == set(probe.ADMISSION_ARMS)
    for arm, value in counts.items():
        assert 0 <= value, arm
        assert value <= result["unique_admitted_nodes"][arm] * len(
            admission_fixture["panel"]
        )


def test_the_diagnostic_states_the_graph_all_three_arms_expanded_over(admission_fixture):
    """A64 is defined on ONE graph. Running the residual arms on another would
    make them differ from the control in the adjacency as well as the scoring
    rule, and the scoring rule is the whole variable."""

    graph = probe._admission_diagnostic(**admission_fixture)["graph"]
    assert graph["family"] == "structural_only"
    assert graph["symmetrised"] is True
    assert graph["stored_already_symmetric"] is False


# ---------------------------------------------------------------------------
# The contract with the sealed artifacts
# ---------------------------------------------------------------------------


SOURCE = (REPO_ROOT / "scripts" / "run_m2c_stage0_probe.py").read_text(encoding="utf-8")


def test_the_mainline_family_is_a64s_own_and_is_not_typed_here():
    """The value M2 recorded in the R3 cell master's build key.

    A wrong string here does not merely mislabel an arm: ``load_cell_under_
    contract`` compares this field against the persisted build key and refuses
    the master outright. The first Stage-0 submission died on exactly that,
    with ``['a64_mainline_family'] differ`` on both R3 cells.
    """

    default = next(
        action.default
        for action in probe.build_parser()._actions
        if action.dest == "a64_mainline_family"
    )
    assert default == probe._m1a.MAINLINE_FAMILY == "structural_only"
    assert '"baseline_a_simple"' not in SOURCE.split("build_parser")[1]


def test_the_admission_budget_is_a64s_constant_and_not_a_flag():
    """64 is not a choice this probe gets to make -- it is what makes the
    comparison budget-matched. A flag beside it could only disagree."""

    dests = {action.dest for action in probe.build_parser()._actions}
    assert "admission_budget" not in dests
    assert "_m1a._a64_budget(" in SOURCE
    assert probe._m1a._a64_budget(per_seed_cap=16, neighbour_scan_cap_per_seed=4096) == (
        ExpansionBudget(per_seed_cap=16, graph_expansion_cap=64, neighbour_scan_cap_per_seed=4096)
    )


def test_the_feature_store_is_sized_by_the_dataset_not_by_the_panel():
    """``context_feature_store`` indexes by ``query.query_index``, which is a
    position in the dataset's query array and not in the split. Sizing it to
    the panel made every query past the split length an IndexError, which is
    how the first squad_clean submission died."""

    assert "query_count=len(dataset.queries)" in SOURCE
    assert "query_count=len(widened)" not in SOURCE


def test_a_family_graph_is_symmetrised_before_anything_reads_it(tmp_path):
    """Stored asymmetric; choosing an orientation is not a probe's call.

    It matters twice: the admission arms must expand over the adjacency A64
    expanded over, and the directional scores must be computed over the
    adjacency M0A's null was measured on.
    """

    path = tmp_path / "graph.pt"
    torch.save(
        {"edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]), "num_nodes": 4}, path
    )
    rowptr, col, was_symmetric = probe._load_family_csr(path, 4)

    assert was_symmetric is False, "the fixture is deliberately one-directional"
    pairs = {
        (source, int(target))
        for source in range(4)
        for target in col[rowptr[source] : rowptr[source + 1]]
    }
    assert pairs == {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)}


def test_the_provenance_views_resolve_to_the_audited_families():
    """G_FULL is baseline_a_simple. full_union_c would add NER on top."""

    assert probe.PROVENANCE_FAMILIES == {
        "G_STRUCT": "structural_only",
        "G_KNN": "knn_only",
        "G_FULL": "baseline_a_simple",
    }
    from mp_retrieval.edge_provenance import EDGE_FAMILY_NAMES

    for family in probe.PROVENANCE_FAMILIES.values():
        assert family in EDGE_FAMILY_NAMES
    assert "full_union_c" not in probe.PROVENANCE_FAMILIES.values()


def test_the_ranking_signal_is_declared_rather_than_chosen_per_cell():
    assert probe.RANKING_SIGNAL == "dir_max"
    assert probe.ARMS == ("S4", "direction_only", "S4_plus_direction_rrf")
    assert probe.RESIDUALS[0] == "R0_RAW_QUERY_CONTROL"
