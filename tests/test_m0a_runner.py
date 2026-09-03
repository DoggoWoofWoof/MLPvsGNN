"""The probe runs end to end, and the invariants it claims are the ones it checks.

A toy dataset is enough for every contract in the declaration: the R1/R2
ceiling identity, the matched-budget size rule, determinism, and the refusal to
report a regime the config did not declare. What a toy cannot show is whether
expansion recovers anything, and this file does not pretend otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mp_retrieval.complete_data import load_complete_dataset
from scripts.run_m0a_probe import COMPLETE_STATUS, run

NUM_NODES = 24
FAMILIES = ("structural_only", "knn_only")


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
                "golds": [
                    ["doc_1", "doc_20"],
                    ["doc_7"],
                    ["doc_14", "doc_23"],
                    ["doc_19"],
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
        target = (source + 3 * offset) % NUM_NODES
        edge_index = np.concatenate(
            [np.stack([source, target]), np.stack([target, source])], axis=1
        )
        path = graph_root / family
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"edge_index": torch.tensor(edge_index, dtype=torch.long), "num_nodes": NUM_NODES},
            path / "graph.pt",
        )


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    graphs = tmp_path / "families"
    _write_dataset(data)
    _write_families(graphs)
    frozen = load_complete_dataset(data, dataset="toy")
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
        "graph_expansion_cap": 8,
        "neighbour_scan_cap_per_seed": 4096,
        "edge_provenance_root": graphs,
        "edge_families": list(FAMILIES),
        "output": tmp_path / "out" / "probe.json",
    }
    settings.update(overrides)
    return argparse.Namespace(**settings)


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    return run(_args(tmp_path_factory.mktemp("m0a")))


def test_the_probe_completes_and_says_what_it_did_not_do(probe):
    assert probe["status"] == COMPLETE_STATUS
    assert probe["trained_anything"] is False
    assert probe["test_split_read"] is False
    assert probe["split"] == "validation"
    assert probe["selection"] == "deterministic_prefix_of_the_split_order"


def test_r1_and_r2_report_the_identical_ceiling(probe):
    """Asserted in the runner too; if it ever fails there the run stops."""
    assert probe["invariants"]["r1_ceiling_equals_r2_ceiling_exactly"] is True
    assert probe["regimes"]["R1"]["headroom"] == probe["regimes"]["R2"]["headroom"]


def test_r2_sees_more_than_r1_without_scoring_more(probe):
    r1, r2 = probe["regimes"]["R1"], probe["regimes"]["R2"]
    assert r2["context_node_count"]["mean"] > r1["context_node_count"]["mean"]
    assert r2["candidate_count"] == r1["candidate_count"]


def test_every_declared_r3_cell_is_present(probe):
    expected = {
        f"R3/{family}/{method}/{rule}"
        for family in FAMILIES
        for method in ("L1_DIRECTIONAL", "STRUCTURAL_NEIGHBOUR")
        for rule in ("matched", "additive")
    }
    assert expected <= set(probe["regimes"])
    assert set(probe["regimes"]) == expected | {"R1", "R2"}


def test_only_the_matched_cells_are_the_headline(probe):
    for key, cell in probe["regimes"].items():
        if key.startswith("R3/"):
            assert cell["is_the_headline"] is key.endswith("/matched")


def test_the_matched_pools_are_the_size_of_the_historical_pool(probe):
    baseline = probe["regimes"]["R1"]["candidate_count"]
    for key, cell in probe["regimes"].items():
        if key.endswith("/matched"):
            assert cell["candidate_count"] == baseline, key


def test_the_additive_pools_are_larger_and_marked_as_diagnostic(probe):
    baseline = probe["regimes"]["R1"]["candidate_count"]["mean"]
    additive = [
        cell["candidate_count"]["mean"]
        for key, cell in probe["regimes"].items()
        if key.endswith("/additive")
    ]
    assert additive and all(value > baseline for value in additive)


def test_a_matched_cell_reports_no_added_slots(probe):
    for key, movement in probe["movement"].items():
        if key.endswith("/matched"):
            assert movement["pool_size_is_matched"] is True, key
            assert movement["pool_slots_added_total"] == 0, key


def test_the_diagnostic_cosine_is_reported_and_is_not_a_score(probe):
    """Recorded so a reader can see whether admitted nodes were merely similar."""
    cell = probe["regimes"]["R3/structural_only/L1_DIRECTIONAL/matched"]
    cosine = cell["admitted_node_query_cosine"]
    assert set(cosine) == {"p50", "p95", "p99", "mean", "max"}
    assert -1.0 <= cosine["p50"] <= 1.0


def test_the_scan_cap_and_the_degenerate_cases_are_reported(probe):
    cell = probe["regimes"]["R3/knn_only/L1_DIRECTIONAL/matched"]
    for key in (
        "neighbour_scan_cap_fired_queries",
        "degenerate_residual_queries",
        "zero_displacement_edges",
    ):
        assert isinstance(cell[key], int), key


def test_the_stored_graph_asymmetry_is_recorded(probe):
    """The frontier is undirected precisely because the stored graph is not."""
    assert probe["stored_graph_was_symmetric"] is False
    assert probe["queries_without_frozen_seeds"] == 0


def test_the_budget_that_ran_is_the_budget_that_is_reported(probe):
    assert probe["budget"]["hop_cap"] == 1
    assert probe["budget"]["per_seed_cap"] == 4
    assert probe["budget"]["graph_expansion_cap"] == 8
    assert probe["budget"]["tie_break"] == "ascending_global_node_id"


def test_the_run_repeats_exactly(tmp_path):
    first = run(_args(tmp_path / "a"))
    second = run(_args(tmp_path / "b"))
    for payload in (first, second):
        payload.pop("systems")
        for cell in payload["regimes"].values():
            for key in list(cell):
                if key.endswith("_latency_ms"):
                    cell.pop(key)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_a_short_split_is_refused_rather_than_quietly_shrunk(tmp_path):
    with pytest.raises(ValueError, match="needed 9"):
        run(_args(tmp_path / "short", queries=9))


def test_r3_without_a_family_graph_is_refused(tmp_path):
    with pytest.raises(ValueError, match="at least one edge-provenance family"):
        run(_args(tmp_path / "nofamily", edge_families=[]))


def test_the_frozen_candidate_pool_is_proved_before_it_is_measured(probe):
    """R1 is the historical object only if Cq is bit-exact against the artifact."""
    proof = probe["candidate_contract"]
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
