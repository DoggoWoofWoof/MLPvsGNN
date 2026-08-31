import numpy as np
import torch

from mp_retrieval.complete_data import CompleteQuery
from mp_retrieval.local_topology_perturbations import perturb_packed_topologies
from mp_retrieval.topology_store import PackedLocalTopologies


def _query(index: int, size: int) -> CompleteQuery:
    return CompleteQuery(
        query_index=index,
        query_id=f"q{index}",
        candidate_index=torch.arange(size),
        relevant_local=torch.tensor([0]),
        relevant_global=torch.tensor([0]),
        anchor_global=0,
        split=0,
    )


def _topologies() -> tuple[PackedLocalTopologies, list[CompleteQuery]]:
    first = np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int32)
    second = np.asarray([[0, 1, 2], [1, 2, 0]], dtype=np.int32)
    return (
        PackedLocalTopologies(
            edge_ptr=np.asarray([0, 4, 7], dtype=np.int64),
            edge_index=np.concatenate((first, second), axis=1),
            query_position=np.asarray([0, 1], dtype=np.int64),
            build_seconds=0.0,
        ),
        [_query(0, 3), _query(1, 3)],
    )


def test_degree_rewire_is_deterministic_and_degree_preserving() -> None:
    clean, queries = _topologies()
    left, metadata = perturb_packed_topologies(
        clean, queries, kind="degree_rewire", rate=1.0, seed=7
    )
    right, _ = perturb_packed_topologies(
        clean, queries, kind="degree_rewire", rate=1.0, seed=7
    )
    assert np.array_equal(left.edge_index, right.edge_index)
    assert metadata["directed_in_and_out_degree_preserved"] is True
    for query in range(2):
        start, end = clean.edge_ptr[query : query + 2]
        before = clean.edge_index[:, start:end]
        after = left.edge_index[:, start:end]
        assert np.array_equal(np.bincount(before[0], minlength=3), np.bincount(after[0], minlength=3))
        assert np.array_equal(np.bincount(before[1], minlength=3), np.bincount(after[1], minlength=3))


def test_random_add_and_hub_injection_preserve_query_alignment() -> None:
    clean, queries = _topologies()
    added, add_metadata = perturb_packed_topologies(
        clean, queries, kind="random_add", rate=1.0, seed=3
    )
    assert added.edge_ptr.tolist() == [0, 8, 14]
    assert add_metadata["added_edges"] == 7
    assert add_metadata["self_loops_after"] == 0
    hubs, hub_metadata = perturb_packed_topologies(
        clean, queries, kind="hub_injection", rate=0.5, seed=3
    )
    assert hubs.edge_index.shape == clean.edge_index.shape
    assert hub_metadata["directed_edge_count_preserved"] is True
