"""The M3B pool functions against the filed headroom helpers, on toy graphs.

``expand_hops`` must be the headroom's ``expand`` for hops <= 2 -- the frozen
contract cells are read from rows that function produced, and a contract
measured with a different walk would not be comparable to them. ``FamilyStore``
must order its columns exactly as ``FamilyCSR`` does. The carve rule must be a
pure function of the sorted ids with disjoint outputs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mp_retrieval import m3b_pools  # noqa: E402


@pytest.fixture(scope="module")
def headroom_runner():
    spec = importlib.util.spec_from_file_location("m3a_headroom", ROOT / "scripts" / "m3a_headroom.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _random_graph(rng, n, m, weighted, typed=False):
    src = rng.integers(0, n, size=m).astype(np.int32)
    dst = rng.integers(0, n, size=m).astype(np.int32)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    weight = rng.random(src.size).astype(np.float32) if weighted else None
    rel = rng.integers(0, 4, size=src.size).astype(np.int16) if typed else None
    return SimpleNamespace(src=src, dst=dst, weight=weight, rel=rel, n_nodes=n, directed=not weighted)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_family_store_orders_columns_exactly_like_family_csr(headroom_runner, seed):
    rng = np.random.default_rng(seed)
    for weighted in (False, True):
        g = _random_graph(rng, 60, 400, weighted, typed=not weighted)
        csr = headroom_runner.FamilyCSR(g, "x")
        store = m3b_pools.FamilyStore.from_graph(g, "x")
        np.testing.assert_array_equal(csr.indptr, store.indptr)
        np.testing.assert_array_equal(csr.col, store.col)
        if not weighted:
            # typed entries cover every stored edge twice, and rel_count counts the stored ones once
            assert store.typed_col.size == 2 * g.src.size
            assert store.rel_count.sum() == g.src.size


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_expand_hops_is_the_filed_expansion_for_one_and_two_hops(headroom_runner, seed):
    rng = np.random.default_rng(seed)
    fams_g = [_random_graph(rng, 80, 300, False, True), _random_graph(rng, 80, 500, True), _random_graph(rng, 80, 500, True)]
    csrs = [headroom_runner.FamilyCSR(g, f) for g, f in zip(fams_g, ("structural", "ner", "knn"))]
    stores = [m3b_pools.FamilyStore.from_graph(g, f) for g, f in zip(fams_g, ("structural", "ner", "knn"))]
    seeds = np.unique(rng.integers(0, 80, size=6)).astype(np.int64)
    for setting in ({"name": "h1_c3", "hops": 1, "per_seed_cap": 3},
                    {"name": "h2", "hops": 2, "per_seed_cap": 3, "per_frontier_cap": 2, "visited_cap": 30},
                    {"name": "h2big", "hops": 2, "per_seed_cap": 25, "per_frontier_cap": 25, "visited_cap": 2000}):
        for fam_sel in ([0], [1], [0, 1, 2]):
            expected = headroom_runner.expand(seeds, [csrs[i] for i in fam_sel], setting)
            got = m3b_pools.expand_hops(seeds, [stores[i] for i in fam_sel], setting)
            np.testing.assert_array_equal(expected, got)


def test_three_hops_extends_two_hops_and_respects_the_visited_cap():
    rng = np.random.default_rng(7)
    g = _random_graph(rng, 200, 600, True)
    store = m3b_pools.FamilyStore.from_graph(g, "knn")
    seeds = np.asarray([0, 1], dtype=np.int64)
    two = m3b_pools.expand_hops(seeds, [store], {"hops": 2, "per_seed_cap": 5, "per_frontier_cap": 5, "visited_cap": 100})
    three = m3b_pools.expand_hops(seeds, [store], {"hops": 3, "per_seed_cap": 5, "per_frontier_cap": 5, "visited_cap": 100})
    np.testing.assert_array_equal(two, three[: two.size])
    assert np.union1d(seeds, three).size <= 100
    assert np.unique(three).size == three.size
    assert not np.isin(three, seeds).any()


def test_carves_are_disjoint_deterministic_and_capped():
    ids = [f"q{i:06d}" for i in range(20000)]
    fit, select = m3b_pools.carve_ids(ids)
    assert not set(fit) & set(select)
    assert len(fit) <= 6000 and len(select) <= 1600
    fit2, select2 = m3b_pools.carve_ids(list(ids))
    assert fit == fit2 and select == select2
    small_fit, small_select = m3b_pools.carve_ids([f"w{i}" for i in range(1549)])
    assert len(small_select) == 310 and len(small_fit) == 1239
