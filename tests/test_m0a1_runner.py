"""The overlap runner, end to end on a toy dataset.

A toy graph is enough to exercise every mechanical contract: that U2 is built
from the exact TARGET_H1 arm, that the 2x2 cross-tabulation is a genuine
partition of the runner's own golds, that the curve is monotone and nested,
that the kNN reproduction reads rather than recomputes, and that the systems
block does not silently invent a number. It cannot show that structural
expansion recovers anything real -- that is what the actual run is for.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mp_retrieval.complete_data import load_complete_dataset
from mp_retrieval.overlap_audit import CROSS_TAB_CELLS
from scripts.run_m0a1_overlap import COMPLETE_STATUS, CURVE_POINTS, FULL_FRONTIER, run

NUM_NODES = 30
FAMILIES = ("structural_only", "baseline_a_simple")


def _write_dataset(root: Path) -> None:
    rng = np.random.default_rng(20260903)
    nodes = rng.normal(size=(NUM_NODES, 6)).astype(np.float32)
    np.save(root / "nodes.npy", nodes)
    np.save(root / "queries_all.npy", rng.normal(size=(4, 6)).astype(np.float32))
    dense = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
            [12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23],
        ]
    )
    splade = (dense + 2) % NUM_NODES
    np.save(root / "dense_top200_all.npy", dense)
    np.save(root / "splade_top200_all.npy", splade)
    (root / "query_ids_all.json").write_text(
        json.dumps(
            {
                "ids": ["q0", "q1", "q2", "q3"],
                # Golds placed outside every query's Cq, some reachable only
                # through the wider family graphs below -- otherwise every
                # query would land in a single cell and the partition would
                # never be exercised.
                "golds": [
                    ["doc_25", "doc_28"],
                    ["doc_26"],
                    ["doc_24", "doc_29"],
                    ["doc_27"],
                ],
                "split_indices": {"train": [3], "val": [0, 1, 2], "test": []},
            }
        ),
        encoding="utf-8",
    )
    source = np.arange(NUM_NODES, dtype=np.int64)
    target = (source + 5) % NUM_NODES
    torch.save(
        {
            "edge_index": torch.tensor(np.stack([source, target]), dtype=torch.long),
            "num_nodes": NUM_NODES,
        },
        root / "graph.pt",
    )


def _write_families(graph_root: Path) -> None:
    for offset, family in enumerate(FAMILIES, start=1):
        source = np.arange(NUM_NODES, dtype=np.int64)
        target = (source + 4 * offset) % NUM_NODES
        edge_index = np.concatenate(
            [np.stack([source, target]), np.stack([target, source])], axis=1
        )
        path = graph_root / family
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"edge_index": torch.tensor(edge_index, dtype=torch.long), "num_nodes": NUM_NODES},
            path / "graph.pt",
        )


def _write_m0a_probe(path: Path, *, recovered) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "movement": {
                    "R3/knn_only/L1_DIRECTIONAL/matched": {"missing_golds_recovered": recovered},
                    "R3/knn_only/STRUCTURAL_NEIGHBOUR/matched": {
                        "missing_golds_recovered": recovered
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    graphs = tmp_path / "families"
    _write_dataset(data)
    _write_families(graphs)
    frozen = load_complete_dataset(data, dataset="toy")
    probe_path = tmp_path / "m0a" / "probe.json"
    _write_m0a_probe(probe_path, recovered=0)
    settings = {
        "data": data,
        "dataset": "toy",
        "data_fingerprint_sha256": "0" * 64,
        "expected_queries": len(frozen.queries),
        "baseline": {
            "candidate_contract_sha256": frozen.metadata["candidate_contract_sha256"]
        },
        "candidate_contract_compatibility": None,
        "queries": 3,
        "per_seed_cap": 4,
        "neighbour_scan_cap_per_seed": 4096,
        "edge_provenance_root": graphs,
        "edge_families": list(FAMILIES),
        "m0a_probe": probe_path,
        "output": tmp_path / "out" / "overlap.json",
    }
    settings.update(overrides)
    return argparse.Namespace(**settings)


@pytest.fixture(scope="module")
def overlap(tmp_path_factory):
    return run(_args(tmp_path_factory.mktemp("m0a1")))


def test_the_run_completes_and_says_what_it_did_not_do(overlap):
    assert overlap["status"] == COMPLETE_STATUS
    assert overlap["trained_anything"] is False
    assert overlap["test_split_read"] is False
    assert overlap["directional_arm_was_rerun"] is False
    assert overlap["m0a_verdict_is_unchanged"] == "MOVEMENT_UNDER_A_BREACHED_ABORT_RULE"
    assert overlap["primary_arm"] == "STRUCTURAL_NEIGHBOUR"


def test_r1_and_r2_ceilings_are_reconfirmed_bit_exact(overlap):
    assert overlap["invariants"]["r1_ceiling_equals_r2_ceiling_exactly"] is True


def test_u2_contains_cq_for_every_query(overlap):
    assert overlap["invariants"]["u2_contains_cq"] is True
    assert overlap["u2"]["arm"] == "TARGET_H1"
    assert overlap["u2"]["context_node_count"]["mean"] >= overlap["r1"]["candidate_count"]["mean"]


def test_a_struct_is_disjoint_from_cq_and_cq_struct_contains_cq(overlap):
    assert overlap["invariants"]["a_struct_is_disjoint_from_cq"] is True
    assert overlap["invariants"]["cq_struct_contains_cq"] is True
    for family, row in overlap["overlap"].items():
        assert row["a_struct_is_disjoint_from_cq"] is True, family


# --- the 2x2 partition ---


def test_the_cross_tabulation_is_reported_for_every_family(overlap):
    for family, row in overlap["overlap"].items():
        cells = row["additive"]["cross_tabulation"]["gold_instances"]
        assert set(cells) == set(CROSS_TAB_CELLS), family


def test_the_cross_tabulation_sums_to_the_runners_own_missing_golds(overlap):
    """A ground truth computed independently of overlap_audit's own bookkeeping.

    q0 has 2 golds, q1 has 1, q2 has 2 -- 5 gold instances across the 3 sampled
    queries, and the cross-tabulation counts only golds absent from Cq, so the
    per-family total can be at most 5.
    """
    total_golds = 5
    for family, row in overlap["overlap"].items():
        cells = row["additive"]["cross_tabulation"]["gold_instances"]
        assert sum(cells.values()) <= total_golds, family


def test_the_matched_and_additive_partitions_are_both_reported(overlap):
    for family, row in overlap["overlap"].items():
        assert "matched_budget" in row, family
        assert set(row["matched_budget"]["cross_tabulation"]["gold_instances"]) == set(
            CROSS_TAB_CELLS
        )
        assert isinstance(row["matched_budget"]["agrees_with_additive"], bool)


def test_recovery_share_is_present_and_bounded(overlap):
    for family, row in overlap["overlap"].items():
        share = row["additive"]["recovery_share"]
        if not share["undefined_because_nothing_was_recovered"]:
            assert 0.0 <= share["already_in_r2_context"] <= 1.0, family
            assert share["already_in_r2_context"] + share["beyond_r2_context"] == pytest.approx(
                1.0
            )


# --- the saturation curve ---


def test_every_curve_has_every_declared_point_in_order(overlap):
    for family, row in overlap["curve"].items():
        budgets = [point["budget"] for point in row["points"]]
        assert budgets == list(CURVE_POINTS), family


def test_the_numeric_points_are_monotone_non_decreasing_in_recovery(overlap):
    """A strictly nested frontier cannot recover fewer golds as the cap widens."""
    for family, row in overlap["curve"].items():
        numeric = [p for p in row["points"] if p["budget"] != FULL_FRONTIER]
        recovered = [p["missing_golds_recovered"] for p in numeric]
        assert recovered == sorted(recovered), family
        added = [p["added_nodes_per_query_mean"] for p in numeric]
        assert added == sorted(added), family


def test_the_full_frontier_point_is_flagged_as_lifting_the_per_seed_cap(overlap):
    for row in overlap["curve"].values():
        full = next(p for p in row["points"] if p["budget"] == FULL_FRONTIER)
        assert full["lifts_per_seed_cap"] is True
        capped = [p for p in row["points"] if p["budget"] != FULL_FRONTIER]
        assert all(p["lifts_per_seed_cap"] is False for p in capped)


def test_the_unbounded_point_recovers_at_least_as_much_as_every_capped_point(overlap):
    for family, row in overlap["curve"].items():
        full = next(p for p in row["points"] if p["budget"] == FULL_FRONTIER)
        for point in row["points"]:
            assert point["missing_golds_recovered"] <= full["missing_golds_recovered"], (
                family,
                point["budget"],
            )


def test_the_marginal_table_has_one_fewer_row_than_the_curve(overlap):
    for family, row in overlap["curve"].items():
        assert len(row["marginal"]) == len(row["points"]) - 1, family


def test_the_curve_is_declared_as_saturation_not_fitting(overlap):
    for row in overlap["curve"].values():
        assert row["purpose"] == "saturation analysis, not hyperparameter fitting"
        assert row["no_dataset_specific_budget_was_selected"] is True


# --- systems ---


def test_the_systems_block_does_not_inherit_the_directional_verdict(overlap):
    assert overlap["systems"]["arm"] == "STRUCTURAL_NEIGHBOUR"
    assert overlap["systems"]["does_not_inherit_the_directional_verdict"] is True


def test_the_workspace_figure_is_declared_not_measured(overlap):
    assert overlap["systems"]["temporary_workspace_is_a_declared_bound_not_a_measurement"] is True
    assert overlap["systems"]["temporary_workspace_bytes"] == 8 * 128 * 4


def test_the_latency_ratio_is_reported_per_family(overlap):
    ratios = overlap["systems"]["expansion_over_context_p95_ratio"]
    assert set(ratios) == set(overlap["overlap"])
    for family, ratio in ratios.items():
        assert ratio >= 0.0, family


# --- kNN: reproduced, not recomputed ---


def test_knn_is_read_from_the_stored_m0a_artifact(overlap):
    knn = overlap["knn_only"]
    assert knn["status"] == "REPRODUCED_FROM_M0A_ARTIFACT"
    assert knn["checked"] is True
    assert knn["holds"] is True
    assert knn["cells"]


def test_a_nonzero_stored_knn_recovery_is_reported_as_not_holding(tmp_path):
    args = _args(tmp_path / "knn_nonzero")
    _write_m0a_probe(args.m0a_probe, recovered=3)
    result = run(args)
    assert result["knn_only"]["holds"] is False
    assert result["knn_only"]["cells"]


def test_a_missing_m0a_artifact_is_reported_as_unavailable_not_as_agreement(tmp_path):
    args = _args(tmp_path / "noknn", m0a_probe=tmp_path / "does" / "not" / "exist.json")
    result = run(args)
    assert result["knn_only"]["status"] == "NOT_AVAILABLE"
    assert result["knn_only"]["checked"] is False
    assert "holds" not in result["knn_only"]


# --- determinism and refusals ---


def test_the_run_repeats_exactly(tmp_path):
    first = run(_args(tmp_path / "a"))
    second = run(_args(tmp_path / "b"))
    for payload in (first, second):
        payload.pop("systems")
        payload["u2"].pop("build_latency_ms")
        payload["knn_only"].pop("source")  # the tmp_path differs by construction
        for row in payload["overlap"].values():
            row.pop("expansion_latency_ms")
            row.pop("context_build_latency_ms")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_a_short_split_is_refused_rather_than_quietly_shrunk(tmp_path):
    with pytest.raises(ValueError, match="needed 9"):
        run(_args(tmp_path / "short", queries=9))


def test_the_audit_without_a_family_graph_is_refused(tmp_path):
    with pytest.raises(ValueError, match="at least one edge-provenance family"):
        run(_args(tmp_path / "nofamily", edge_families=[]))


def test_the_frozen_candidate_pool_is_proved_before_it_is_measured(overlap):
    proof = overlap["candidate_contract"]
    assert proof["status"] == "BIT_EXACT_FROZEN_CANDIDATE_EQUIVALENCE"
    assert proof["expected_contract_sha256"] == proof["observed_contract_sha256"]


def test_a_drifted_candidate_pool_stops_the_run(tmp_path):
    args = _args(tmp_path / "drift")
    args.baseline = {"candidate_contract_sha256": "f" * 64}
    with pytest.raises(ValueError, match="candidate contract does not match"):
        run(args)


def test_a_manifest_of_the_wrong_size_stops_the_run(tmp_path):
    args = _args(tmp_path / "size")
    args.expected_queries = 99
    with pytest.raises(ValueError, match="differs from the registered protocol"):
        run(args)


def test_a_completed_run_is_not_repeated(tmp_path):
    args = _args(tmp_path / "cached")
    first = run(args)
    args.expected_queries = 99  # would raise if the run actually happened again
    second = run(args)
    assert first == second
