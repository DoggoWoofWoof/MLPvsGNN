"""Directed message flow, operator edge semantics, and the two expansion graphs.

Symmetrised connectivity answers *"are these nodes related at all"*. It does not
answer *"can this candidate receive that node's signal"*, because every frozen
operator aggregates ``source_to_target``. A path that is intact under
symmetrisation can be unusable in the direction messages actually travel, and
these tests pin that distinction down.

`test_operator_edge_semantics_table_matches_the_real_convolutions` is the
load-bearing one: `OPERATOR_EDGE_SEMANTICS` is a hand-written table, and it is
only trustworthy while it agrees with the shipped PyG layers.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from mp_retrieval.graph_substrate import (
    MESSAGE_FLOW,
    OPERATOR_EDGE_SEMANTICS,
    directed_adjacency,
    expansion_sizes,
    hop_distances,
    induced_view,
    message_flow_receptive_field,
    operator_edge_load,
    receptive_field_sizes,
)
from test_graph_substrate import csr_from_edges, symmetric_edges

DIM = 4


def _conv(kind: str):
    import torch.nn as nn
    from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, SAGEConv

    torch.manual_seed(0)
    factories = {
        "gcn": lambda: GCNConv(DIM, DIM),
        "sage": lambda: SAGEConv(DIM, DIM),
        "gat": lambda: GATv2Conv(DIM, DIM, heads=1, concat=False),
        "gin": lambda: GINConv(
            nn.Sequential(nn.Linear(DIM, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        ),
    }
    layer = factories[kind]()
    layer.eval()
    return layer


# --------------------------------------------------------------------------
# The table must describe the layers that are actually shipped.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(OPERATOR_EDGE_SEMANTICS))
def test_operator_edge_semantics_table_matches_the_real_convolutions(kind: str) -> None:
    layer = _conv(kind)
    x = torch.eye(3, DIM)
    forward = torch.tensor([[0], [1]])          # a single edge 0 -> 1
    duplicated = torch.tensor([[0, 0], [1, 1]])  # the same edge twice
    empty = torch.empty(2, 0, dtype=torch.long)

    with torch.no_grad():
        once = layer(x, forward)
        twice = layer(x, duplicated)
        none = layer(x, empty)

    moved = [i for i in range(3) if not torch.allclose(once[i], none[i], atol=1e-6)]
    assert moved == [1], f"{kind}: edge (0,1) must move node 1, not node 0"
    assert MESSAGE_FLOW == "source_to_target"

    semantics = OPERATOR_EDGE_SEMANTICS[kind]
    duplicate_changed = not torch.allclose(once, twice, atol=1e-6)
    assert duplicate_changed is bool(semantics["duplicate_sensitive"]), (
        f"{kind}: duplicate sensitivity in the table disagrees with the layer"
    )
    # Node 2 has no edge at all and is still assigned a representation, so a
    # candidate with an empty receptive field is scored as a plain MLP.
    assert bool(none[2].abs().max() > 1e-6) is bool(
        semantics["isolated_node_still_scored"]
    )
    declared = getattr(layer, "add_self_loops", False)
    assert bool(declared) is bool(semantics["adds_self_loops"])


def test_every_frozen_selected_family_is_covered_by_the_table() -> None:
    # 2wiki gat, musique gcn, webqsp gat, hotpotqa gin, squad gcn, metaqa gat.
    for kind in ("gat", "gcn", "gin"):
        assert kind in OPERATOR_EDGE_SEMANTICS
        assert OPERATOR_EDGE_SEMANTICS[kind]["duplicate_sensitive"] is True


# --------------------------------------------------------------------------
# Symmetric connectivity and message flow are different questions.
# --------------------------------------------------------------------------


def test_a_path_intact_under_symmetrisation_can_carry_no_message() -> None:
    """Stored orientation ``candidate -> bridge -> seed`` reverses the flow.

    Symmetrised, the seed and the candidate are two hops apart and every
    connectivity statistic calls the path preserved. But messages travel
    source_to_target, so the candidate aggregates nothing and the seed's signal
    never reaches it -- at any depth.
    """

    # 0 = seed, 1 = bridge, 2 = candidate. Edges point AWAY from the candidate.
    rowptr, col = csr_from_edges([(2, 1), (1, 0)], 3)
    candidates = np.array([0, 1, 2], dtype=np.int64)
    counts = induced_view(rowptr, col, candidates)
    assert counts.edges.shape[1] == 2, "both edges are inside the pool"

    symmetric = receptive_field_sizes(counts, max_hops=3)
    flow = message_flow_receptive_field(counts, max_hops=3)

    # Symmetrically, every node reaches both others within two hops.
    assert symmetric["R2_mean"] == pytest.approx(2.0)
    assert symmetric["R2_zero_fraction"] == 0.0
    # Along the real direction the candidate receives nothing, ever.
    assert flow["flow_R3_zero_fraction"] == pytest.approx(1 / 3)
    # The seed receives from both; the bridge from one; the candidate from none.
    assert flow["flow_R3_mean"] == pytest.approx((2 + 1 + 0) / 3)


def test_seed_signal_cannot_travel_against_the_stored_orientation() -> None:
    rowptr, col = csr_from_edges([(2, 1), (1, 0)], 3)
    counts = induced_view(rowptr, col, np.array([0, 1, 2], dtype=np.int64))

    forward_rowptr, forward_col = directed_adjacency(counts.edges, 3)
    reached = hop_distances(forward_rowptr, forward_col, np.array([0]), 3, max_hops=3)
    assert reached[1] == np.iinfo(np.int32).max
    assert reached[2] == np.iinfo(np.int32).max

    # Reverse the stored edges and the same seed reaches both.
    back_rowptr, back_col = directed_adjacency(counts.edges, 3, reverse=True)
    back = hop_distances(back_rowptr, back_col, np.array([0]), 3, max_hops=3)
    assert back[1] == 1
    assert back[2] == 2


def test_message_flow_matches_symmetric_reach_on_a_symmetric_graph() -> None:
    """When the stored graph is already symmetric the two notions coincide."""

    rowptr, col = csr_from_edges(symmetric_edges([(0, 1), (1, 2)]), 3)
    counts = induced_view(rowptr, col, np.array([0, 1, 2], dtype=np.int64))

    symmetric = receptive_field_sizes(counts, max_hops=2)
    flow = message_flow_receptive_field(counts, max_hops=2)
    assert flow["flow_R1_mean"] == pytest.approx(symmetric["R1_mean"])
    assert flow["flow_R2_mean"] == pytest.approx(symmetric["R2_mean"])


# --------------------------------------------------------------------------
# Edge multiplicity is a real message, not a bookkeeping detail.
# --------------------------------------------------------------------------


def test_duplicate_edges_are_counted_as_the_extra_messages_they_are() -> None:
    rowptr, col = csr_from_edges([(0, 1), (0, 1), (1, 0)], 2)
    counts = induced_view(rowptr, col, np.array([0, 1], dtype=np.int64))

    assert counts.kept_messages == 3
    assert counts.unique_non_self_edges == 2

    load = operator_edge_load(counts, "gcn")
    assert load["stored_non_self_messages"] == 3.0
    assert load["unique_non_self_edges"] == 2.0
    assert load["duplicate_messages"] == 1.0
    assert load["duplicate_message_fraction"] == pytest.approx(1 / 3)
    # GCN inserts one self-loop per candidate on top of the stored messages.
    assert load["operator_inserted_self_loops"] == 2.0
    assert load["messages_consumed_by_operator"] == 5.0


def test_sage_is_the_only_family_that_ignores_multiplicity() -> None:
    rowptr, col = csr_from_edges([(0, 1), (0, 1)], 2)
    counts = induced_view(rowptr, col, np.array([0, 1], dtype=np.int64))

    assert operator_edge_load(counts, "sage")["duplicate_sensitive"] == 0.0
    assert operator_edge_load(counts, "gin")["duplicate_sensitive"] == 1.0
    # SAGE adds no self-loop, so it consumes exactly the stored messages.
    assert operator_edge_load(counts, "sage")["messages_consumed_by_operator"] == 2.0


def test_stored_self_loops_are_reported_separately_from_inserted_ones() -> None:
    rowptr, col = csr_from_edges([(0, 0), (0, 1)], 2)
    counts = induced_view(rowptr, col, np.array([0, 1], dtype=np.int64))

    load = operator_edge_load(counts, "gat")
    assert load["stored_self_loops"] == 1.0
    assert load["operator_inserted_self_loops"] == 2.0
    # Node 0 ends up with a doubled self-loop: one stored, one inserted.
    assert load["messages_consumed_by_operator"] == 4.0


def test_retention_ignores_a_stored_self_loop_in_both_terms() -> None:
    """A self-loop is not neighbourhood context, so it belongs in neither term."""

    from mp_retrieval.graph_substrate import retention_summary

    rowptr, col = csr_from_edges([(0, 0), (0, 1), (1, 0)], 2)
    counts = induced_view(rowptr, col, np.array([0, 1], dtype=np.int64))
    # Node 0 stores two edges, one of them a self-loop; its real neighbour is 1.
    assert retention_summary(counts)["retention_mean"] == pytest.approx(1.0)


def test_unknown_operator_is_rejected() -> None:
    rowptr, col = csr_from_edges([(0, 1)], 2)
    counts = induced_view(rowptr, col, np.array([0, 1], dtype=np.int64))
    with pytest.raises(ValueError, match="Unknown message-passing operator"):
        operator_edge_load(counts, "transformer")


# --------------------------------------------------------------------------
# U_seed and U_target are not the same graph.
# --------------------------------------------------------------------------


def test_seed_and_target_expansions_diverge() -> None:
    """The pool's own neighbourhood grows far faster than the seeds' does."""

    # Seed 0 is a leaf. Candidate 1 is a hub with ten outside neighbours.
    edges = symmetric_edges([(0, 1)] + [(1, t) for t in range(2, 12)])
    rowptr, col = csr_from_edges(edges, 12)
    pool = np.array([0, 1], dtype=np.int64)
    seeds = np.array([0], dtype=np.int64)

    sizes = expansion_sizes(rowptr, col, pool, seeds, max_hops=2)
    assert sizes["candidates"] == 2.0
    # One hop from the seed reaches only the hub, which is already a candidate.
    assert sizes["U_seed_1_nodes"] == 2.0
    # One hop from the pool pulls in all ten of the hub's neighbours.
    assert sizes["U_target_1_nodes"] == 12.0
    assert sizes["U_target_1_expansion"] == pytest.approx(6.0)
    # By two hops the seed expansion catches up through the hub.
    assert sizes["U_seed_2_nodes"] == 12.0


def test_expansion_never_shrinks_and_always_contains_the_pool() -> None:
    rowptr, col = csr_from_edges(symmetric_edges([(0, 1), (1, 2), (2, 3)]), 4)
    pool = np.array([0], dtype=np.int64)
    sizes = expansion_sizes(rowptr, col, pool, pool, max_hops=3)
    assert (
        sizes["candidates"]
        <= sizes["U_seed_1_nodes"]
        <= sizes["U_seed_2_nodes"]
        <= sizes["U_seed_3_nodes"]
    )
    assert sizes["U_seed_3_nodes"] == 4.0


def test_expansion_reports_sizes_without_touching_the_pool() -> None:
    rowptr, col = csr_from_edges(symmetric_edges([(0, 1)]), 2)
    pool = np.array([0], dtype=np.int64)
    before = pool.copy()
    expansion_sizes(rowptr, col, pool, pool, max_hops=1)
    assert np.array_equal(pool, before)
