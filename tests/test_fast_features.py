"""The fast feature compiler is the reference compiler, bit for bit: on random
worlds with typed multi-relation pairs, both stored directions, hubs past the
in-pool cap, duplicate and absent cache entries, empty families and no
relation table, FastCompiler.compile returns exactly what compile_query_v2
(v2=True) or compile_query (v2=False) returns in the same process -- every
scalar column, every edge and edge attribute, the seed weights -- on the
parallel and on the single-thread kernels, at pool sizes on both sides of the
BLAS threading threshold and of every K-block split of the ring sums."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("numba")

from mp_retrieval import fast_features as FF  # noqa: E402
from mp_retrieval import m3b_features as F  # noqa: E402
from mp_retrieval import m3b_pools as P  # noqa: E402
from mp_retrieval import universal_v2_features as V  # noqa: E402


class _Nodes16:
    """The served embeddings as DenseNodes holds a small corpus: float16 in RAM, read as float32."""

    def __init__(self, matrix16):
        self._matrix = matrix16

    def read(self, rows, dtype=np.float32):
        return self._matrix[np.asarray(rows, dtype=np.int64)].astype(dtype)


class _Nodes32:
    """Embeddings served as float32 (24-bit significands, not float16's 11): read through ``read``."""

    _matrix = None

    def __init__(self, matrix32):
        self._m = matrix32

    def read(self, rows, dtype=np.float32):
        return self._m[np.asarray(rows, dtype=np.int64)].astype(dtype)


def _unit(rng, shape):
    x = rng.normal(size=shape).astype(np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _edges(rng, n, m, hubs=()):
    src = rng.integers(0, n, size=m)
    dst = rng.integers(0, n, size=m)
    for h in hubs:   # hubs past the in-pool cap
        k = rng.integers(0, n, size=400)
        src, dst = np.r_[src, np.full(k.size, h)], np.r_[dst, k]
    keep = src != dst
    return src[keep].astype(np.int32), dst[keep].astype(np.int32)


def _world(seed, n_nodes=2400, m_struct=90_000, m_ner=40_000, m_knn=60_000, n_rel=7, typed=True, empty_ner=False):
    rng = np.random.default_rng(seed)
    hubs = tuple(rng.integers(0, n_nodes, size=3))
    s_src, s_dst = _edges(rng, n_nodes, m_struct, hubs)
    # repeated pairs with other relations and reversed copies: several typed entries per pair, both directions
    rep = rng.integers(0, s_src.size, size=s_src.size // 4)
    s_src, s_dst = np.r_[s_src, s_src[rep], s_dst[rep[: rep.size // 2]]], np.r_[s_dst, s_dst[rep], s_src[rep[: rep.size // 2]]]
    rel = rng.integers(0, n_rel, size=s_src.size).astype(np.int16)
    struct = SimpleNamespace(src=s_src, dst=s_dst, weight=None, rel=rel, n_nodes=n_nodes, directed=True)
    stores = {"structural": P.FamilyStore.from_graph(struct, "structural")}
    for fam, m in (("ner", 0 if empty_ner else m_ner), ("knn", m_knn)):
        src, dst = _edges(rng, n_nodes, m, hubs if fam == "knn" else ())
        w = rng.random(src.size).astype(np.float32) * 3
        w[rng.random(src.size) < 0.2] = 1.0   # ties in the stored weight order
        stores[fam] = P.FamilyStore.from_graph(SimpleNamespace(src=src, dst=dst, weight=w, rel=None, n_nodes=n_nodes, directed=False), fam)
    emb16 = _unit(rng, (n_nodes, F.DIM)).astype(np.float16)
    rel_table = F.RelationTable.from_arrays(_unit(rng, (n_rel, F.DIM)), stores["structural"].rel_count, n_nodes) if typed else None
    return SimpleNamespace(rng=rng, n=n_nodes, stores=stores, nodes=_Nodes16(emb16), emb=emb16.astype(np.float32), rel=rel_table)


def _query(w, pool_size):
    rng = w.rng
    q = _unit(rng, F.DIM)
    scores = w.emb @ q
    dense_ids = np.argsort(-scores, kind="stable")[:1000].astype(np.int64)
    splade_ids = rng.permutation(w.n)[:1000].astype(np.int64)
    splade_ids[7] = splade_ids[3]                        # a repeated cache entry: the first occurrence counts
    inp = F.QueryInputs(q=q, dense_ids=dense_ids, dense_scores=scores[dense_ids].astype(np.float32), splade_ids=splade_ids,
                        splade_scores=np.sort(rng.random(1000).astype(np.float32) * 30)[::-1].copy())
    seeds = P.seeds_of(dense_ids, splade_ids)
    others = rng.choice(np.setdiff1d(np.arange(w.n), seeds), size=pool_size - seeds.size, replace=False)
    pool = np.unique(np.r_[others, seeds]).astype(np.int64)   # exactly pool_size nodes, every seed among them
    return inp, pool, seeds


def _assert_same(fast, ref):
    assert np.array_equal(fast.pool, ref.pool) and fast.pool.dtype == ref.pool.dtype
    assert np.array_equal(fast.seeds_local, ref.seeds_local)
    assert fast.scalars.shape == ref.scalars.shape and fast.scalars.dtype == ref.scalars.dtype
    diff = fast.scalars.view(np.uint32) != ref.scalars.view(np.uint32)
    names = [V.COLUMNS[j] for j in np.flatnonzero(diff.any(axis=0))]
    assert not names, f"columns not bit-identical: {names}"
    assert np.array_equal(fast.seedw.view(np.uint32), ref.seedw.view(np.uint32))
    assert fast.edges.keys() == ref.edges.keys()
    for fam in ref.edges:
        for a, b in zip(fast.edges[fam], ref.edges[fam]):
            assert a.dtype == b.dtype and a.shape == b.shape, fam
            assert np.array_equal(np.ascontiguousarray(a).view(np.uint32) if a.dtype == np.float32 else a,
                                  np.ascontiguousarray(b).view(np.uint32) if b.dtype == np.float32 else b), fam


@pytest.fixture(scope="module")
def world():
    return _world(20260930)


@pytest.mark.parametrize("pool_size", [37, 333, 351, 362, 363, 700, 1450])
def test_v2_bit_identical_to_the_reference(world, pool_size):
    fc = FF.FastCompiler(world.stores, world.nodes, world.rel)
    for _ in range(2):
        inp, pool, seeds = _query(world, pool_size)
        counts_ref, counts_fast = {}, {}
        ref = V.compile_query_v2(inp, pool, seeds, world.stores, world.nodes, world.rel, pair_counts=counts_ref)
        _assert_same(fc.compile(inp, pool, seeds, pair_counts=counts_fast), ref)
        assert counts_fast == counts_ref
        # the gathered embeddings passed in, as a batch compiler does
        _assert_same(fc.compile(inp, pool, seeds, embeddings=world.nodes.read(pool)), ref)


@pytest.mark.parametrize("pool_size", [120, 900])
def test_m3b_layout_bit_identical_to_compile_query(world, pool_size):
    fc = FF.FastCompiler(world.stores, world.nodes, world.rel)
    inp, pool, seeds = _query(world, pool_size)
    ref = F.compile_query(inp, pool, seeds, world.stores, world.nodes, world.rel)
    _assert_same(fc.compile(inp, pool, seeds, v2=False), ref)


def test_single_thread_kernels_agree(world):
    fc = FF.FastCompiler(world.stores, world.nodes, world.rel, threads=1)
    inp, pool, seeds = _query(world, 500)
    _assert_same(fc.compile(inp, pool, seeds), V.compile_query_v2(inp, pool, seeds, world.stores, world.nodes, world.rel))


def test_untyped_world_and_an_empty_family():
    w = _world(11, n_nodes=1200, typed=False, empty_ner=True)
    fc = FF.FastCompiler(w.stores, w.nodes, w.rel)
    inp, pool, seeds = _query(w, 400)
    ref = V.compile_query_v2(inp, pool, seeds, w.stores, w.nodes, w.rel)
    _assert_same(fc.compile(inp, pool, seeds), ref)
    assert not ref.scalars[:, V.IDX["typed_available"]].any()


@pytest.mark.parametrize("size", [1000, 20])
def test_memory_mapped_shards(world, tmp_path, size):
    """A corpus too large for RAM is gathered from its float16 shards directly, on every thread (a small pool too);
    more than 100 shards too (hotpotqa has 131), which a tuple could not carry into a parallel region."""
    paths = []
    for k in range(0, world.n, size):
        paths.append(tmp_path / f"shard{k // size}.npy")
        np.save(paths[-1], world.nodes._matrix[k:k + size])
    spec = SimpleNamespace(n_rows=world.n, shard_size=size, n_shards=len(paths), _path=lambda k: paths[k])
    nodes = F.DenseNodes(spec, ram_limit_bytes=0)
    assert nodes._matrix is None and len(nodes._maps) == -(-world.n // size)
    fc = FF.FastCompiler(world.stores, nodes, world.rel)
    assert fc._shards is not None and len(fc._shards) == len(paths)
    for pool_size in (90, 700):
        inp, pool, seeds = _query(world, pool_size)
        _assert_same(fc.compile(inp, pool, seeds), V.compile_query_v2(inp, pool, seeds, world.stores, nodes, world.rel))
    del fc, nodes


def test_seed_outside_the_pool_is_refused(world):
    fc = FF.FastCompiler(world.stores, world.nodes, world.rel)
    inp, pool, seeds = _query(world, 100)
    outside = np.setdiff1d(np.arange(world.n), pool)[:1]
    with pytest.raises(ValueError):
        fc.compile(inp, pool, np.r_[seeds, outside])
    assert (fc._lookup == -1).all()   # the lookup table is released on every exit


def test_seed_exact_rule():
    """_seed_exact passes float16-served seed rows, whose {0, 0.5, 1}-weighted sums are then the same float32 in any
    order (here numpy's product at 3000 row positions against a sequential sum, 10 seeds: the regime where
    OpenBLAS's order depends on the position), and refuses a column whose finest term sits too far below its sum and
    float32 rows."""
    K = FF._kernels(False)
    rng = np.random.default_rng(5)
    E_S = _unit(rng, (10, F.DIM)).astype(np.float16).astype(np.float32)
    assert K.seed_exact(E_S, E_S.view(np.uint32))
    W = rng.choice(np.asarray([0.0, 0.5, 1.0], np.float32), size=(3000, 10), p=[0.5, 0.2, 0.3])
    seq = np.zeros((3000, F.DIM), np.float32)
    for s in range(10):
        seq = seq + W[:, s:s + 1] * E_S[s]
    assert np.array_equal(W @ E_S, seq)
    bad = E_S.copy()
    bad[:, 7] = np.float32(0.25)            # float16 values whose sum, 2.25, needs a grid no finer than 2^-23 ...
    bad[3, 7] = np.float32(2.0 ** -24)      # ... and float16's smallest subnormal, halved by the weight 0.5: 2^-25
    assert not K.seed_exact(bad, bad.view(np.uint32))
    bad[3, 7] = np.float32(2.0 ** -8)
    assert K.seed_exact(bad, bad.view(np.uint32))
    f32 = _unit(rng, (10, F.DIM))
    assert not K.seed_exact(f32, f32.view(np.uint32))
    none = np.zeros((0, F.DIM), np.float32)
    assert K.seed_exact(none, none.view(np.uint32))


def test_both_seed_product_paths_on_one_query(world, monkeypatch):
    """A query whose seed products pass _seed_exact (10 seeds) is compiled on the fused rows; forced onto numpy's
    products, the same query gives the same bits."""
    fc = FF.FastCompiler(world.stores, world.nodes, world.rel)
    inp, pool, seeds = _query(world, 1450)
    E_S = world.nodes.read(seeds)
    assert seeds.size >= 8 and fc.K.seed_exact(E_S, E_S.view(np.uint32))
    ref = V.compile_query_v2(inp, pool, seeds, world.stores, world.nodes, world.rel)
    _assert_same(fc.compile(inp, pool, seeds), ref)
    monkeypatch.setattr(fc.K, "seed_exact", lambda E_S, bits: False)
    _assert_same(fc.compile(inp, pool, seeds), ref)


def test_float32_embeddings_take_numpys_seed_products():
    w = _world(13, n_nodes=1200)
    nodes32 = _Nodes32(_unit(np.random.default_rng(1), (w.n, F.DIM)))
    fc = FF.FastCompiler(w.stores, nodes32, w.rel)
    inp, pool, seeds = _query(w, 700)
    E_S = nodes32.read(seeds)
    assert not fc.K.seed_exact(E_S, E_S.view(np.uint32))
    _assert_same(fc.compile(inp, pool, seeds), V.compile_query_v2(inp, pool, seeds, w.stores, nodes32, w.rel))


def test_ring_rule_matches_the_live_blas():
    rule = FF.ring_rule()
    assert rule.dense or (rule.q > 0 and rule.unroll > 0)


def test_blas_thread_count_changed_after_the_compiler_was_built(world):
    """The ring sums follow the BLAS thread count at compile time, not at construction: a compiler built under the
    default count still matches the reference run under one BLAS thread, on both sides of the threaded driver."""
    threadpoolctl = pytest.importorskip("threadpoolctl")
    fc = FF.FastCompiler(world.stores, world.nodes, world.rel)
    for pool_size in (363, 700):
        inp, pool, seeds = _query(world, pool_size)
        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            assert FF.blas_threads() == 1
            _assert_same(fc.compile(inp, pool, seeds), V.compile_query_v2(inp, pool, seeds, world.stores, world.nodes, world.rel))
        _assert_same(fc.compile(inp, pool, seeds), V.compile_query_v2(inp, pool, seeds, world.stores, world.nodes, world.rel))


def test_drop_ins_match_the_reference_and_keep_one_compiler(world):
    inp, pool, seeds = _query(world, 450)
    counts_ref, counts_fast = {}, {}
    ref = V.compile_query_v2(inp, pool, seeds, world.stores, world.nodes, world.rel, pair_counts=counts_ref)
    _assert_same(FF.compile_query_v2_fast(inp, pool, seeds, world.stores, world.nodes, world.rel, pair_counts=counts_fast), ref)
    assert counts_fast == counts_ref
    _assert_same(FF.compile_query_fast(inp, pool, seeds, world.stores, world.nodes, world.rel),
                 F.compile_query(inp, pool, seeds, world.stores, world.nodes, world.rel))
    fc = FF.compiler_for(world.stores, world.nodes, world.rel)
    assert fc is FF.compiler_for(world.stores, world.nodes, world.rel)
    FF.release_compilers()
    assert FF.compiler_for(world.stores, world.nodes, world.rel) is not fc
    FF.release_compilers()
