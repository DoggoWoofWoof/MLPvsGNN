import torch

from mp_retrieval.perturbations import (
    add_random_edges,
    corrupt_edge_types,
    degrade_features,
    degree_preserving_rewire,
    drop_edges,
    inject_hubs,
    remove_edge_types,
)


def cycle_graph(n=20):
    src = torch.arange(n)
    dst = (src + 1) % n
    return torch.stack([src, dst])


def test_drop_and_add_are_seed_deterministic():
    edges = cycle_graph()
    left, _ = drop_edges(edges, 0.25, seed=7)
    right, _ = drop_edges(edges, 0.25, seed=7)
    assert torch.equal(left, right)
    added, _ = add_random_edges(edges, 20, 0.5, seed=4)
    assert added.shape[1] == 30
    assert len(set(zip(added[0].tolist(), added[1].tolist()))) == 30


def test_directed_rewire_preserves_degrees():
    edges = torch.cat([cycle_graph(), torch.stack([torch.arange(20), (torch.arange(20) + 3) % 20])], dim=1)
    rewired = degree_preserving_rewire(edges, 0.5, seed=3)
    assert torch.equal(torch.bincount(edges[0], minlength=20), torch.bincount(rewired[0], minlength=20))
    assert torch.equal(torch.bincount(edges[1], minlength=20), torch.bincount(rewired[1], minlength=20))
    changed_fraction = float((rewired != edges).any(dim=0).float().mean())
    assert 0.5 <= changed_fraction <= 0.5 + 1 / edges.shape[1]


def test_directed_rewire_reaches_full_rate_and_is_nested():
    edges = torch.cat(
        [cycle_graph(), torch.stack([torch.arange(20), (torch.arange(20) + 3) % 20])],
        dim=1,
    )
    low = degree_preserving_rewire(edges, 0.25, seed=17)
    high = degree_preserving_rewire(edges, 1.0, seed=17)
    low_changed = (low != edges).any(dim=0)
    high_changed = (high != edges).any(dim=0)
    assert float(high_changed.float().mean()) == 1.0
    assert torch.all(high_changed[low_changed])
    assert torch.equal(
        torch.bincount(edges[0], minlength=20),
        torch.bincount(high[0], minlength=20),
    )
    assert torch.equal(
        torch.bincount(edges[1], minlength=20),
        torch.bincount(high[1], minlength=20),
    )


def test_directed_rewire_handles_duplicate_input_edges():
    base = torch.cat(
        [cycle_graph(), torch.stack([torch.arange(20), (torch.arange(20) + 3) % 20])],
        dim=1,
    )
    edges = torch.cat([base, base[:, :4]], dim=1)
    rewired = degree_preserving_rewire(edges, 0.25, seed=13)
    assert rewired.shape == edges.shape
    assert torch.equal(torch.bincount(edges[0], minlength=20), torch.bincount(rewired[0], minlength=20))
    assert torch.equal(torch.bincount(edges[1], minlength=20), torch.bincount(rewired[1], minlength=20))


def test_type_corruption_preserves_histogram():
    types = torch.tensor([0, 0, 0, 1, 1, 2])
    corrupted = corrupt_edge_types(types, 1.0, seed=9)
    assert torch.equal(types.sort().values, corrupted.sort().values)


def test_hub_injection_preserves_edge_count_and_increases_target_concentration():
    edges = cycle_graph()
    changed = inject_hubs(edges, 20, 0.5, seed=11, num_hubs=2)
    assert changed.shape == edges.shape
    before = torch.bincount(edges[1], minlength=20).max()
    after = torch.bincount(changed[1], minlength=20).max()
    assert after > before


def test_hub_injection_handles_duplicate_input_edges():
    base = cycle_graph()
    edges = torch.cat([base, base[:, :4]], dim=1)
    changed = inject_hubs(edges, 20, 0.5, seed=11, num_hubs=2)
    assert changed.shape == edges.shape
    assert torch.equal(
        torch.bincount(changed[0], minlength=20),
        torch.bincount(edges[0], minlength=20),
    )


def test_feature_degradation_and_type_removal_are_deterministic():
    features = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    assert torch.equal(
        degrade_features(features, 0.5, seed=5),
        degrade_features(features, 0.5, seed=5),
    )
    edges = cycle_graph(6)
    edge_type = torch.tensor([0, 1, 2, 0, 1, 2])
    kept_edges, kept_types = remove_edge_types(edges, edge_type, {1})
    assert kept_edges.shape[1] == 4
    assert 1 not in kept_types.tolist()
