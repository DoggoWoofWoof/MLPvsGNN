"""The F(q, v) compiler on toy substrates: shapes, masks, and a handful of
columns recomputed by brute force. The compiler is deterministic and reads no
gold; these tests hold the formulas the declaration names."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mp_retrieval import m3b_features as F  # noqa: E402
from mp_retrieval import m3b_pools as P  # noqa: E402


class _Nodes:
    def __init__(self, matrix):
        self.matrix = matrix

    def read(self, rows):
        return self.matrix[np.asarray(rows)].astype(np.float32)


def _graph(rng, n, m, weighted, typed, n_rel=4):
    src = rng.integers(0, n, size=m).astype(np.int32)
    dst = rng.integers(0, n, size=m).astype(np.int32)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    return SimpleNamespace(src=src, dst=dst, weight=rng.random(src.size).astype(np.float32) if weighted else None,
                           rel=rng.integers(0, n_rel, size=src.size).astype(np.int16) if typed else None, n_nodes=n, directed=not weighted)


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    n = 300
    stores = {"structural": P.FamilyStore.from_graph(_graph(rng, n, 900, False, True), "structural"),
              "ner": P.FamilyStore.from_graph(_graph(rng, n, 700, True, False), "ner"),
              "knn": P.FamilyStore.from_graph(_graph(rng, n, 1200, True, False), "knn")}
    emb = rng.normal(size=(n, F.DIM)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q = rng.normal(size=F.DIM).astype(np.float32)
    q /= np.linalg.norm(q)
    dense_ids = rng.permutation(n)[:100].astype(np.int64)
    splade_ids = rng.permutation(n)[:100].astype(np.int64)
    inp = F.QueryInputs(q=q, dense_ids=dense_ids, dense_scores=np.sort(rng.random(100))[::-1].astype(np.float32),
                        splade_ids=splade_ids, splade_scores=np.sort(rng.random(100) * 20)[::-1].astype(np.float32))
    seeds = P.seeds_of(dense_ids, splade_ids)
    expansion = P.expand_hops(seeds, list(stores.values()), {"hops": 2, "per_seed_cap": 10, "per_frontier_cap": 5, "visited_cap": 120})
    pool = np.union1d(np.union1d(dense_ids[:30], splade_ids[:30]), expansion)
    pool = np.union1d(pool, seeds)
    rel_emb = rng.normal(size=(4, F.DIM)).astype(np.float32)
    rel_emb /= np.linalg.norm(rel_emb, axis=1, keepdims=True)
    rel = F.RelationTable.from_arrays(rel_emb, stores["structural"].rel_count, n)
    return SimpleNamespace(stores=stores, nodes=_Nodes(emb), emb=emb, inp=inp, seeds=seeds, pool=pool, rel=rel, q=q)


def test_shapes_masks_and_finiteness(toy):
    c = F.compile_query(toy.inp, toy.pool, toy.seeds, toy.stores, toy.nodes, toy.rel)
    n = toy.pool.size
    assert c.scalars.shape == (n, F.N_COLUMNS) and np.isfinite(c.scalars).all()
    assert c.seedw.shape == (n, F.MAX_SEEDS)
    for fam, (u, v, attr) in c.edges.items():
        assert u.shape == v.shape and attr.shape == (u.size, len(F.EDGE_ATTR))
        assert (u < n).all() and (v < n).all() and (u != v).all()
    X = c.scalars
    for view in F.TOPOLOGY_VIEWS:
        buckets = sum(X[:, F.IDX[f"dist{b}_{view}"]] for b in ("0", "1", "2", "3", "unreached"))
        np.testing.assert_array_equal(buckets, 1.0)
        assert (X[:, F.IDX[f"reach_{view}"]] == 1 - X[:, F.IDX[f"dist{'unreached'}_{view}"]]).all()
    assert X[:, F.IDX["is_seed"]].sum() == toy.seeds.size
    assert X[:, F.IDX["typed_available"]].all()


def test_retrieval_columns_by_brute_force(toy):
    c = F.compile_query(toy.inp, toy.pool, toy.seeds, toy.stores, toy.nodes, toy.rel)
    X = c.scalars
    np.testing.assert_allclose(X[:, F.IDX["dense_cos"]], toy.emb[toy.pool] @ toy.q, atol=1e-5)
    d_rank = {int(v): r + 1 for r, v in enumerate(toy.inp.dense_ids)}
    s_rank = {int(v): r + 1 for r, v in enumerate(toy.inp.splade_ids)}
    for i, v in enumerate(toy.pool):
        v = int(v)
        expected = (1 / (60 + d_rank[v]) if v in d_rank else 0) + (1 / (60 + s_rank[v]) if v in s_rank else 0)
        assert abs(X[i, F.IDX["rrf"]] - expected) < 1e-6
        assert X[i, F.IDX["dense_rr"]] == pytest.approx(1 / d_rank[v] if v in d_rank else 0)
        assert X[i, F.IDX["agreement"]] == float(v in d_rank and v in s_rank)


def test_pool_edges_are_in_pool_capped_and_symmetric_in_kind(toy):
    c = F.compile_query(toy.inp, toy.pool, toy.seeds, toy.stores, toy.nodes, toy.rel)
    n = toy.pool.size
    for fam, (u, v, attr) in c.edges.items():
        store = toy.stores[fam]
        # every message edge u -> v is a stored neighbour relation between the two positions
        for uu, vv in zip(toy.pool[u[:50]], toy.pool[v[:50]]):
            nbrs = store.col[store.indptr[vv]:store.indptr[vv + 1]]
            assert uu in nbrs
        counts = np.bincount(v, minlength=n)
        assert counts.max() <= F.IN_POOL_CAP
        if fam != "structural":
            assert attr[:, 0].max() == pytest.approx(1.0)
        else:
            assert (attr[:, 2] == 1).all() and ((attr[:, 3] + attr[:, 4]) >= 1).all()


def test_gcs_dominates_the_base_and_seed_distance_zero_is_the_seed_set(toy):
    c = F.compile_query(toy.inp, toy.pool, toy.seeds, toy.stores, toy.nodes, toy.rel)
    X = c.scalars
    s = X[:, F.IDX["rrf"]] / X[:, F.IDX["rrf"]].max()
    assert (X[:, F.IDX["gcs_full"]] >= s - 1e-6).all() and (X[:, F.IDX["gcs_struct"]] >= s - 1e-6).all()
    seeds_local = np.searchsorted(toy.pool, toy.seeds)
    assert set(np.flatnonzero(X[:, F.IDX["dist0_FULL"]])) == set(seeds_local.tolist())


def test_typed_columns_without_a_relation_table_are_zero_with_mask_zero(toy):
    c = F.compile_query(toy.inp, toy.pool, toy.seeds, toy.stores, toy.nodes, None)
    X = c.scalars
    typed_cols = [F.IDX[name] for name, g in F.GROUP_OF.items() if g == "typed_relations_B"]
    assert not X[:, typed_cols].any()
    # the structural message edges still exist, with rel_compat 0 and mask 0
    u, v, attr = c.edges["structural"]
    assert u.size > 0 and not attr[:, 1].any() and not attr[:, 2].any()


def test_contract_json_lists_every_column_once():
    j = F.contract_json()
    names = [c["name"] for c in j["columns"]]
    assert names == F.COLUMNS and len(set(names)) == len(names) == j["n_columns"]
