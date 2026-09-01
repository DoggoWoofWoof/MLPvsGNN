import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mp_retrieval.complete_data import load_complete_dataset
from scripts.run_candidate_headroom import completed_headroom, run

NUM_NODES = 12


def _dataset(root: Path) -> None:
    np.save(root / "nodes.npy", np.eye(NUM_NODES, 4, dtype=np.float32))
    np.save(root / "queries_all.npy", np.ones((3, 4), dtype=np.float32))
    np.save(root / "dense_top200_all.npy", np.array([[0, 1, 2], [5, 6, 7], [0, 5, 10]]))
    np.save(root / "splade_top200_all.npy", np.array([[2, 3, 4], [7, 8, 9], [10, 11, 1]]))
    (root / "query_ids_all.json").write_text(
        json.dumps(
            {
                "ids": ["q0", "q1", "q2"],
                "golds": [["doc_3", "doc_9"], ["doc_5", "doc_6", "doc_7"], ["doc_11", "doc_7"]],
                "split_indices": {"train": [1], "val": [2], "test": [0]},
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "edge_index": torch.tensor([[4, 6, 8, 5], [6, 8, 9, 7]], dtype=torch.long),
            "num_nodes": NUM_NODES,
        },
        root / "graph.pt",
    )


def _args(root: Path, output: Path, fingerprint: str, contract: str) -> argparse.Namespace:
    return argparse.Namespace(
        data=root,
        dataset="toy",
        expected_queries=3,
        baseline={"candidate_contract_sha256": contract},
        candidate_contract_compatibility=None,
        ks=[1, 5],
        budgets=[3, 6],
        rrf_constant=60,
        max_hops=3,
        max_visited=2_000_000,
        reachability_splits=["test"],
        output=output,
        data_fingerprint_sha256=fingerprint,
    )


def _prepared(tmp_path: Path) -> argparse.Namespace:
    root = tmp_path / "data"
    root.mkdir()
    _dataset(root)
    dataset = load_complete_dataset(root, dataset="toy")
    return _args(
        root,
        tmp_path / "headroom.json",
        "fingerprint",
        dataset.metadata["candidate_contract_sha256"],
    )


def test_runner_writes_a_complete_read_only_diagnostic(tmp_path: Path) -> None:
    args = _prepared(tmp_path)
    result = run(args)

    assert result["status"] == "CANDIDATE_HEADROOM_DIAGNOSTIC_COMPLETE"
    assert result["diagnostic_contract"]["candidate_pools_modified"] is False
    assert result["diagnostic_contract"]["graph_expansion_performed"] is False
    assert result["diagnostic_contract"]["candidate_admission_performed"] is False
    assert result["candidate_contract"]["status"] == "BIT_EXACT_FROZEN_CANDIDATE_EQUIVALENCE"
    assert set(result["headroom"]) == {"train", "validation", "test"}
    assert set(result["reachability"]) == {"test"}
    for split in result["headroom"].values():
        assert "equal_rrf_budget_3" in split
        assert "equal_rrf_budget_6" in split
        assert "frozen_union" in split
        assert "recall_ceiling@5" in split["frozen_union"]
    assert json.loads(args.output.read_text(encoding="utf-8"))["status"] == result["status"]


def test_runner_reports_the_test_split_ceiling_and_reachability(tmp_path: Path) -> None:
    args = _prepared(tmp_path)
    result = run(args)
    union = result["headroom"]["test"]["frozen_union"]
    reach = result["reachability"]["test"]

    assert union["gold_fraction_at_pool_macro"] == pytest.approx(0.5)
    assert union["recall_ceiling@1"] == pytest.approx(0.5)
    assert union["recall_ceiling@5"] == pytest.approx(0.5)
    assert union["recall_headroom_lost_to_candidate_generation@5"] == pytest.approx(0.5)
    assert reach["queries_with_missing_gold"] == 1
    assert reach["buckets"]["missing_golds_at_distance_3"] == 1
    assert reach["candidate_pools_modified"] is False


def test_runner_reuses_a_complete_diagnostic(tmp_path: Path) -> None:
    args = _prepared(tmp_path)
    first = run(args)
    reused = completed_headroom(args)
    assert reused is not None
    assert reused["headroom"] == first["headroom"]


@pytest.mark.parametrize(
    "override",
    [
        {"data_fingerprint_sha256": "different"},
        {"ks": [1, 20]},
        {"budgets": [3]},
        {"rrf_constant": 42},
        {"max_hops": 2},
    ],
)
def test_runner_refuses_to_overwrite_a_different_contract(
    tmp_path: Path, override: dict[str, object]
) -> None:
    args = _prepared(tmp_path)
    run(args)
    for key, value in override.items():
        setattr(args, key, value)
    with pytest.raises(ValueError, match="different diagnostic contract"):
        completed_headroom(args)


def test_runner_rejects_a_mismatched_frozen_candidate_contract(tmp_path: Path) -> None:
    args = _prepared(tmp_path)
    args.baseline = {"candidate_contract_sha256": "0" * 64}
    with pytest.raises(ValueError, match="candidate contract does not match"):
        run(args)
