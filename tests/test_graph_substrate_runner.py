"""End-to-end Phase -1 audit over a toy dataset with a known bridge deletion.

The fixture encodes the exact failure the audit exists to detect. Query ``q0``
has two golds:

``doc_11``  reachable from a retrieval seed by a single kNN edge, both endpoints
            inside the candidate pool -- the relationship survives induction.
``doc_5``   reachable from a retrieval seed only through node 12, which is a
            real node of the graph but was never retrieved -- so the induced
            subgraph deletes the only connecting path.

Every number asserted below is derived from that fixture by hand, so a change in
the audit's semantics fails the test rather than silently reporting a different
quantity under the same key.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mp_retrieval.complete_data import load_complete_dataset
from scripts.run_graph_substrate_audit import completed_audit, run

NUM_NODES = 13  # 0..11 are retrievable; 12 is the bridge that is never retrieved
BRIDGE_EDGES = [(0, 12), (12, 5)]
KNN_EDGES = [(6, 11)]


def _bidirectional(pairs: list[tuple[int, int]]) -> torch.Tensor:
    source = [a for a, _ in pairs] + [b for _, b in pairs]
    target = [b for _, b in pairs] + [a for a, _ in pairs]
    return torch.tensor([source, target], dtype=torch.long)


def _dataset(root: Path) -> None:
    np.save(root / "nodes.npy", np.eye(NUM_NODES, 4, dtype=np.float32))
    np.save(root / "queries_all.npy", np.ones((3, 4), dtype=np.float32))
    np.save(
        root / "dense_top200_all.npy",
        np.array([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]),
    )
    np.save(
        root / "splade_top200_all.npy",
        np.array([[6, 7, 8, 9, 10, 11], [7, 8, 9, 10, 11, 0], [8, 9, 10, 11, 0, 1]]),
    )
    (root / "query_ids_all.json").write_text(
        json.dumps(
            {
                "ids": ["q0", "q1", "q2"],
                "golds": [["doc_5", "doc_11"], ["doc_0"], ["doc_1"]],
                "split_indices": {"train": [1], "val": [2], "test": [0]},
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {"edge_index": _bidirectional(BRIDGE_EDGES + KNN_EDGES), "num_nodes": NUM_NODES},
        root / "graph.pt",
    )


def _families(root: Path) -> Path:
    """Package B style sidecars: the native edges and the kNN edges separately."""

    for name, pairs in (("structural_only", BRIDGE_EDGES), ("knn_only", KNN_EDGES)):
        family = root / name
        family.mkdir(parents=True)
        torch.save(
            {"edge_index": _bidirectional(pairs), "num_nodes": NUM_NODES},
            family / "graph.pt",
        )
    return root


def _prepared(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    root = tmp_path / "data"
    root.mkdir()
    _dataset(root)
    dataset = load_complete_dataset(root, dataset="toy")
    args = argparse.Namespace(
        data=root,
        dataset="toy",
        expected_queries=3,
        baseline={
            "candidate_contract_sha256": dataset.metadata["candidate_contract_sha256"]
        },
        candidate_contract_compatibility=None,
        data_fingerprint_sha256="fingerprint",
        graphs=["dataset_default"],
        splits=["test"],
        edge_families=None,
        max_hops=3,
        pooled_query_cap=1000,
        operator_kind="gcn",
        expansion_query_cap=1000,
        output=tmp_path / "substrate.json",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# --------------------------------------------------------------------------
# The audit runs, stays read-only, and writes what it claims to write.
# --------------------------------------------------------------------------


def test_audit_writes_a_complete_read_only_diagnostic(tmp_path: Path) -> None:
    result = run(_prepared(tmp_path))
    contract = result["diagnostic_contract"]

    assert result["status"] == "GRAPH_SUBSTRATE_AUDIT_COMPLETE"
    assert contract["read_only"] is True
    assert contract["candidate_pools_modified"] is False
    assert contract["candidate_pools_expanded"] is False
    assert contract["graph_expansion_performed"] is False
    assert contract["candidate_admission_performed"] is False
    assert contract["models_trained"] is False
    assert result["candidate_contract"]["status"] == "BIT_EXACT_FROZEN_CANDIDATE_EQUIVALENCE"
    assert set(result["graphs"]) == {"dataset_default"}
    assert set(result["graphs"]["dataset_default"]["splits"]) == {"test"}


def test_audit_result_is_persisted_atomically(tmp_path: Path) -> None:
    args = _prepared(tmp_path)
    result = run(args)
    written = json.loads(args.output.read_text(encoding="utf-8"))
    assert written["status"] == result["status"]
    assert written["graphs"] == result["graphs"]


# --------------------------------------------------------------------------
# What the induced substrate actually looks like.
# --------------------------------------------------------------------------


def _test_split(result: dict, graph: str = "dataset_default") -> dict:
    return result["graphs"][graph]["splits"]["test"]


def test_connectivity_of_the_induced_pool_is_reported(tmp_path: Path) -> None:
    # q0's pool is nodes 0..11. Only 6 - 11 survives induction, so ten of the
    # twelve candidates are isolated and there are eleven components.
    summary = _test_split(run(_prepared(tmp_path)))["query_level"]["connectivity"]

    assert summary["candidates"] == pytest.approx(12.0)
    assert summary["isolated_fraction"] == pytest.approx(10 / 12)
    assert summary["components"] == pytest.approx(11.0)
    assert summary["largest_component_fraction"] == pytest.approx(2 / 12)


def test_neighbourhood_retention_is_measured_against_the_true_global_degree(
    tmp_path: Path,
) -> None:
    # Four pool nodes have a global neighbour: 0 and 5 (through the bridge, both
    # cut) and 6 and 11 (through the kNN edge, both kept).
    split = _test_split(run(_prepared(tmp_path)))

    assert split["query_level"]["retention"]["retention_mean"] == pytest.approx(0.5)
    assert split["query_level"]["retention"]["boundary_cut_ratio"] == pytest.approx(0.5)
    assert split["node_level"]["retention_mean"] == pytest.approx(0.5)
    assert split["node_level"]["pooled_from_queries"] == 1


def test_seed_reachability_is_lower_on_the_induced_substrate(tmp_path: Path) -> None:
    """The substrate, not the traversal, is what loses the gold at two hops."""

    split = _test_split(run(_prepared(tmp_path)))["query_level"]
    induced = split["seed_reachability_induced"]
    global_ = split["seed_reachability_global"]

    # Ten of twelve candidates are seeds; node 11 is one hop away on both.
    assert induced["reachable_at_1"] == pytest.approx(11 / 12)
    assert global_["reachable_at_1"] == pytest.approx(11 / 12)
    # Node 5 is two hops away in the real graph and unreachable once induced.
    assert global_["reachable_at_2"] == pytest.approx(1.0)
    assert induced["reachable_at_2"] == pytest.approx(11 / 12)


# --------------------------------------------------------------------------
# The failure this whole phase exists to quantify, end to end.
# --------------------------------------------------------------------------


def test_bridge_loss_is_detected_end_to_end(tmp_path: Path) -> None:
    split = _test_split(run(_prepared(tmp_path)))["query_level"]
    preservation = split["gold_path_preservation"]
    loss = split["gold_bridge_loss"]

    # doc_11 is one hop away and survives; doc_5 is two hops away and does not.
    assert preservation["connected_globally_fraction"] == pytest.approx(1.0)
    assert preservation["connected_induced_fraction"] == pytest.approx(0.5)
    assert preservation["globally_connected_but_induced_disconnected"] == pytest.approx(1.0)
    assert preservation["path_preservation_at_1"] == pytest.approx(1.0)
    assert preservation["path_preservation_at_2"] == pytest.approx(0.5)

    assert loss["bridge_loss_at_1"] == pytest.approx(0.0)
    assert loss["bridge_loss_at_2"] == pytest.approx(0.5)
    assert loss["bridge_loss_at_3"] == pytest.approx(0.5)


def test_provenance_families_are_audited_separately(tmp_path: Path) -> None:
    """Native and kNN edges carry different relationships and must not be pooled."""

    args = _prepared(
        tmp_path,
        graphs=["dataset_default", "structural_only", "knn_only"],
        edge_families=str(_families(tmp_path / "families")),
    )
    result = run(args)

    assert set(result["graphs"]) == {"dataset_default", "structural_only", "knn_only"}
    assert result["graphs"]["structural_only"]["undirected_edges"] == 2
    assert result["graphs"]["knn_only"]["undirected_edges"] == 1
    assert (
        result["graphs"]["structural_only"]["undirected_edge_key_sha256"]
        != result["graphs"]["knn_only"]["undirected_edge_key_sha256"]
    )

    # The destroyed relationship lives entirely in the native edges; the
    # surviving one lives entirely in the kNN edges.
    native = _test_split(result, "structural_only")["query_level"]["gold_bridge_loss"]
    knn = _test_split(result, "knn_only")["query_level"]["gold_bridge_loss"]
    assert native["bridge_loss_at_2"] == pytest.approx(1.0)
    assert knn["bridge_loss_at_2"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_audit_reuses_a_complete_diagnostic(tmp_path: Path) -> None:
    args = _prepared(tmp_path)
    first = run(args)
    reused = completed_audit(args)
    assert reused is not None
    assert reused["graphs"] == first["graphs"]


@pytest.mark.parametrize(
    "override",
    [
        {"data_fingerprint_sha256": "different"},
        {"max_hops": 2},
        {"graphs": ["dataset_default", "knn_only"]},
        {"splits": ["train"]},
    ],
)
def test_audit_refuses_to_reuse_a_different_contract(
    tmp_path: Path, override: dict[str, object]
) -> None:
    args = _prepared(tmp_path)
    run(args)
    for key, value in override.items():
        setattr(args, key, value)
    with pytest.raises(ValueError, match="different diagnostic contract"):
        completed_audit(args)


def test_audit_rejects_a_mismatched_frozen_candidate_contract(tmp_path: Path) -> None:
    args = _prepared(tmp_path)
    args.baseline = {"candidate_contract_sha256": "0" * 64}
    with pytest.raises(ValueError, match="candidate contract does not match"):
        run(args)


def test_audit_rejects_an_unexpected_query_count(tmp_path: Path) -> None:
    args = _prepared(tmp_path, expected_queries=99)
    with pytest.raises(ValueError, match="query count differs"):
        run(args)


def test_provenance_families_require_a_family_root(tmp_path: Path) -> None:
    args = _prepared(tmp_path, graphs=["knn_only"])
    with pytest.raises(ValueError, match="require --edge-families"):
        run(args)


# --------------------------------------------------------------------------
# Message flow is measured on the stored orientation, not the symmetrised view.
# --------------------------------------------------------------------------


def _one_way_dataset(root: Path) -> None:
    """Identical to `_dataset` except the kNN edge points AWAY from the seed."""

    _dataset(root)
    torch.save(
        {
            "edge_index": torch.tensor(
                [[0, 12, 12, 5, 11], [12, 0, 5, 12, 6]], dtype=torch.long
            ),
            "num_nodes": NUM_NODES,
        },
        root / "graph.pt",
    )


def test_symmetrised_reach_overstates_what_the_operator_can_propagate(
    tmp_path: Path,
) -> None:
    """Seed 6 and gold 11 are adjacent, but the message travels 11 -> 6.

    Symmetrised, the relationship is intact and every connectivity statistic
    says so. Along the direction messages actually move, the seed's signal never
    reaches the gold, so a one-layer GNN cannot use the edge at all.
    """

    root = tmp_path / "data"
    root.mkdir()
    _one_way_dataset(root)
    dataset = load_complete_dataset(root, dataset="toy")
    args = argparse.Namespace(
        data=root,
        dataset="toy",
        expected_queries=3,
        baseline={
            "candidate_contract_sha256": dataset.metadata["candidate_contract_sha256"]
        },
        candidate_contract_compatibility=None,
        data_fingerprint_sha256="fingerprint",
        graphs=["dataset_default"],
        splits=["test"],
        edge_families=None,
        max_hops=3,
        pooled_query_cap=1000,
        operator_kind="gcn",
        expansion_query_cap=1000,
        output=tmp_path / "substrate.json",
    )
    split = _test_split(run(args))["query_level"]

    # Ten of twelve candidates are seeds. Symmetrised, gold 11 joins them.
    assert split["seed_reachability_induced"]["reachable_at_1"] == pytest.approx(11 / 12)
    # Along the stored direction it never does, at any depth.
    flow = split["seed_reachability_induced_message_flow"]
    assert flow["reachable_at_1"] == pytest.approx(10 / 12)
    assert flow["reachable_at_3"] == pytest.approx(10 / 12)


def test_operator_edge_load_is_reported_for_the_selected_family(tmp_path: Path) -> None:
    load = _test_split(run(_prepared(tmp_path)))["query_level"]["operator_edge_load"]

    # The pool keeps only the bidirectional kNN pair 6 <-> 11.
    assert load["unique_non_self_edges"] == pytest.approx(2.0)
    assert load["duplicate_messages"] == pytest.approx(0.0)
    assert load["stored_self_loops"] == pytest.approx(0.0)
    # GCN inserts one self-loop per candidate on top of the two stored messages.
    assert load["operator_inserted_self_loops"] == pytest.approx(12.0)
    assert load["messages_consumed_by_operator"] == pytest.approx(14.0)
    assert load["duplicate_sensitive"] == pytest.approx(1.0)


def test_both_expansion_definitions_are_reported(tmp_path: Path) -> None:
    expansion = _test_split(run(_prepared(tmp_path)))["expansion"]

    assert expansion["queries_measured"] == 1
    # The pool is nodes 0..11; one hop pulls in the bridge node 12 either way.
    assert expansion["symmetric"]["candidates"] == pytest.approx(12.0)
    assert expansion["symmetric"]["U_seed_1_nodes"] == pytest.approx(13.0)
    assert expansion["symmetric"]["U_target_1_nodes"] == pytest.approx(13.0)
    assert expansion["symmetric"]["U_target_1_expansion"] == pytest.approx(13 / 12)
    assert "U_seed_3_nodes" in expansion["symmetric"]
    assert expansion["message_flow"]["candidates"] == pytest.approx(12.0)


def test_expansion_admits_nothing_to_the_candidate_pool(tmp_path: Path) -> None:
    """Expansion measures sizes only; the scoring set and its hash must not move."""

    args = _prepared(tmp_path)
    result = run(args)
    proof = result["candidate_contract"]
    assert proof["status"] == "BIT_EXACT_FROZEN_CANDIDATE_EQUIVALENCE"
    contract = result["diagnostic_contract"]
    assert contract["expansion_definitions"]["admits_nothing_to_the_pool"] is True
    assert contract["candidate_pools_expanded"] is False
