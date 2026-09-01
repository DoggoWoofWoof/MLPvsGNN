"""Phase -1 graph-substrate diagnostics.

The decisive test is `test_bridge_node_deletion_is_detected`: it encodes the
exact failure the audit exists to quantify -- a seed and a gold that are two
hops apart in the real graph but disconnected once the intermediate node is
excluded from the candidate pool.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mp_retrieval.complete_data import CompleteQuery, CompleteRetrievalDataset
from mp_retrieval.graph_substrate import (
    UNREACHED,
    bridge_loss,
    connectivity_summary,
    hop_distances,
    induced_view,
    path_preservation,
    receptive_field_sizes,
    retention_summary,
)


def csr_from_edges(edges: list[tuple[int, int]], size: int) -> tuple[np.ndarray, np.ndarray]:
    """Directed CSR from an explicit edge list."""

    source = np.array([edge[0] for edge in edges], dtype=np.int64)
    target = np.array([edge[1] for edge in edges], dtype=np.int64)
    order = np.argsort(source, kind="stable")
    source, target = source[order], target[order]
    counts = np.bincount(source, minlength=size)
    rowptr = np.zeros(size + 1, dtype=np.int64)
    np.cumsum(counts, out=rowptr[1:])
    return rowptr, target


def symmetric_edges(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(a, b) for a, b in pairs] + [(b, a) for a, b in pairs]


# --------------------------------------------------------------------------
# The induced view must reproduce what the models were actually given.
# --------------------------------------------------------------------------


def test_induced_view_matches_the_frozen_induced_subgraph():
    """`induced_view` must agree edge-for-edge with the shipped implementation.

    If these ever disagree the audit is measuring a graph the models never saw,
    which would make every Phase -1 number meaningless.
    """

    rng = np.random.default_rng(20260902)
    size = 60
    pairs = {
        (int(a), int(b))
        for a, b in rng.integers(0, size, size=(400, 2))
        if int(a) != int(b)
    }
    rowptr, col = csr_from_edges(sorted(pairs), size)
    candidates = np.sort(rng.choice(size, size=25, replace=False)).astype(np.int64)

    dataset = CompleteRetrievalDataset(
        root=Path("."),
        dataset="synthetic",
        node_array=np.zeros((size, 1), dtype=np.float32),
        query_array=np.zeros((1, 1), dtype=np.float32),
        rowptr=torch.from_numpy(rowptr),
        col=torch.from_numpy(col),
        queries=[],
        metadata={},
    )
    query = CompleteQuery(
        query_index=0,
        query_id="q0",
        candidate_index=torch.from_numpy(candidates),
        relevant_local=torch.empty(0, dtype=torch.long),
        relevant_global=torch.empty(0, dtype=torch.long),
        anchor_global=0,
        split=0,
    )

    reference = dataset.induced_subgraph(query).numpy()
    observed = induced_view(rowptr, col, candidates).edges

    reference_set = {(int(a), int(b)) for a, b in zip(*reference) if int(a) != int(b)}
    observed_set = {(int(a), int(b)) for a, b in zip(*observed)}
    assert observed_set == reference_set


def test_induced_view_keeps_only_edges_with_both_endpoints_in_the_pool():
    # 0 -> 1 survives (both candidates); 1 -> 2 does not (2 is outside).
    rowptr, col = csr_from_edges([(0, 1), (1, 2), (2, 3)], 4)
    counts = induced_view(rowptr, col, np.array([0, 1], dtype=np.int64))

    assert counts.edges.shape[1] == 1
    assert (counts.edges[:, 0] == np.array([0, 1])).all()
    # Node 1's global out-degree is 1, and that one edge left the pool.
    assert counts.global_degree.tolist() == [1, 1]
    assert counts.boundary_edges == 1


def test_self_loops_are_excluded_from_the_substrate_statistics():
    rowptr, col = csr_from_edges([(0, 0), (0, 1), (1, 0)], 2)
    counts = induced_view(rowptr, col, np.array([0, 1], dtype=np.int64))

    assert counts.self_loops == 1
    assert counts.edges.shape[1] == 2
    summary = connectivity_summary(counts)
    assert summary["isolated_fraction"] == 0.0
    assert summary["self_loops"] == 1.0


# --------------------------------------------------------------------------
# Neighbourhood retention
# --------------------------------------------------------------------------


def test_retention_measures_the_fraction_of_real_neighbours_that_survived():
    # Node 0 is a hub with ten global neighbours; only node 1 is also a
    # candidate, so it keeps 10% of the context a normal GNN would aggregate.
    # Node 1's single global neighbour is node 0, so it keeps all of its own.
    edges = [(0, target) for target in range(1, 11)] + [(1, 0)]
    rowptr, col = csr_from_edges(edges, 11)
    counts = induced_view(rowptr, col, np.array([0, 1], dtype=np.int64))

    summary = retention_summary(counts)
    assert counts.global_degree.tolist() == [10, 1]
    assert counts.induced_out_degree.tolist() == [1, 1]
    assert summary["retention_median"] == pytest.approx(0.55)  # mean of 0.1 and 1.0
    assert summary["retention_below_25pct_fraction"] == pytest.approx(0.5)
    assert summary["retention_zero_fraction"] == 0.0
    # Nine of the eleven incident edges leave the pool.
    assert summary["boundary_cut_ratio"] == pytest.approx(9 / 11)


def test_retention_is_one_when_the_pool_is_the_whole_graph():
    rowptr, col = csr_from_edges(symmetric_edges([(0, 1), (1, 2)]), 3)
    counts = induced_view(rowptr, col, np.array([0, 1, 2], dtype=np.int64))

    summary = retention_summary(counts)
    assert summary["retention_mean"] == pytest.approx(1.0)
    assert summary["boundary_cut_ratio"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Connectivity and receptive field
# --------------------------------------------------------------------------


def test_connectivity_reports_isolates_and_components():
    # A triangle plus two isolated candidates.
    rowptr, col = csr_from_edges(symmetric_edges([(0, 1), (1, 2), (0, 2)]), 5)
    counts = induced_view(rowptr, col, np.array([0, 1, 2, 3, 4], dtype=np.int64))

    summary = connectivity_summary(counts)
    assert summary["isolated_fraction"] == pytest.approx(2 / 5)
    assert summary["components"] == 3.0
    assert summary["largest_component_fraction"] == pytest.approx(3 / 5)
    assert summary["second_component_fraction"] == pytest.approx(1 / 5)


def test_isolated_candidates_have_an_empty_receptive_field():
    """A candidate with R1 = 0 receives no message; a 1-layer GNN scores it as an MLP."""

    rowptr, col = csr_from_edges(symmetric_edges([(0, 1)]), 4)
    counts = induced_view(rowptr, col, np.array([0, 1, 2, 3], dtype=np.int64))

    summary = receptive_field_sizes(counts, max_hops=3)
    assert summary["R1_zero_fraction"] == pytest.approx(0.5)
    assert summary["R1_median"] == pytest.approx(0.5)


def test_receptive_field_grows_along_a_path_and_excludes_self():
    # Path 0 - 1 - 2 - 3: node 0 reaches 1, then 2, then 3.
    rowptr, col = csr_from_edges(symmetric_edges([(0, 1), (1, 2), (2, 3)]), 4)
    counts = induced_view(rowptr, col, np.array([0, 1, 2, 3], dtype=np.int64))

    summary = receptive_field_sizes(counts, max_hops=3)
    assert summary["R1_zero_fraction"] == 0.0
    # R1 is 1,2,2,1 along the path; by three hops every node reaches all others.
    assert summary["R1_mean"] == pytest.approx(1.5)
    assert summary["R3_mean"] == pytest.approx(3.0)


# --------------------------------------------------------------------------
# The failure this whole phase exists to quantify
# --------------------------------------------------------------------------


def test_bridge_node_deletion_is_detected():
    """seed - bridge - gold, with the bridge outside the candidate pool.

    Globally the gold is two hops from the seed. Inside the induced graph it is
    unreachable, because the only connecting path leaves the pool. Neither a
    2-layer GNN nor a QLS hop feature can recover this.
    """

    global_rowptr, global_col = csr_from_edges(
        symmetric_edges([(0, 1), (1, 2)]), 3
    )  # 0 = seed, 1 = bridge, 2 = gold
    candidates = np.array([0, 2], dtype=np.int64)  # the bridge is NOT retrieved

    counts = induced_view(global_rowptr, global_col, candidates)
    assert counts.edges.shape[1] == 0, "induction must delete the bridged path"

    global_distance = hop_distances(global_rowptr, global_col, np.array([0]), 3, max_hops=3)
    assert global_distance[2] == 2

    induced_rowptr, induced_col = csr_from_edges([], 2)
    induced_distance = hop_distances(
        induced_rowptr, induced_col, np.array([0]), 2, max_hops=3
    )
    assert induced_distance[1] == UNREACHED  # local index 1 is the gold

    gold_global = np.array([global_distance[2]])
    gold_induced = np.array([induced_distance[1]])

    preservation = path_preservation(gold_global, gold_induced, max_hops=3)
    assert preservation["path_preservation_at_2"] == 0.0
    assert preservation["globally_connected_but_induced_disconnected"] == 1.0

    loss = bridge_loss(gold_global, gold_induced, max_hops=3)
    assert loss["bridge_loss_at_2"] == 1.0
    assert loss["bridge_loss_at_3"] == 1.0


def test_preserved_paths_report_no_bridge_loss():
    """The same shape, but with the bridge retrieved: nothing is destroyed."""

    rowptr, col = csr_from_edges(symmetric_edges([(0, 1), (1, 2)]), 3)
    candidates = np.array([0, 1, 2], dtype=np.int64)
    counts = induced_view(rowptr, col, candidates)

    global_distance = hop_distances(rowptr, col, np.array([0]), 3, max_hops=3)
    induced_rowptr, induced_col = csr_from_edges(
        [(int(a), int(b)) for a, b in zip(*counts.edges)], 3
    )
    induced_distance = hop_distances(
        induced_rowptr, induced_col, np.array([0]), 3, max_hops=3
    )

    loss = bridge_loss(global_distance, induced_distance, max_hops=3)
    assert loss["bridge_loss_at_2"] == 0.0
    assert loss["bridge_loss_at_3"] == 0.0


def test_distance_inflation_is_reported_separately_from_disconnection():
    """Induction can lengthen a path without severing it; that is a distinct harm."""

    global_distance = np.array([1, 2, 2])
    induced_distance = np.array([1, 3, UNREACHED])

    preservation = path_preservation(global_distance, induced_distance, max_hops=3)
    assert preservation["globally_connected_but_induced_disconnected"] == 1.0
    # Of the two still connected, one was lengthened.
    assert preservation["distance_inflated_fraction"] == pytest.approx(0.5)
    assert preservation["mean_distance_inflation"] == pytest.approx(0.5)
    # At h=2 all three were eligible; only the first still qualifies.
    assert preservation["path_preservation_at_2"] == pytest.approx(1 / 3)


def test_empty_candidate_pool_is_handled():
    rowptr, col = csr_from_edges([(0, 1)], 2)
    counts = induced_view(rowptr, col, np.array([], dtype=np.int64))
    assert counts.edges.shape == (2, 0)
    assert connectivity_summary(counts)["candidates"] == 0.0
