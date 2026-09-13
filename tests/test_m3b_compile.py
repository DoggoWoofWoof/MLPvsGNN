"""The compile script's helpers: pools per the contract, gold localisation, the
univariate ranking used by the screen and the base-score rule, duplicates."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mp_retrieval.m3b_features import N_COLUMNS, MAX_SEEDS  # noqa: E402
from mp_retrieval.m3b_train import CacheWriter, rank_metrics  # noqa: E402


def load_compile():
    spec = importlib.util.spec_from_file_location("m3b_compile", ROOT / "scripts" / "m3b_compile.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["m3b_compile"] = module          # dataclasses resolve the defining module through sys.modules
    spec.loader.exec_module(module)
    return module


class _Compiled:
    def __init__(self, pool, scalars, seeds_local, seedw=None):
        self.pool = pool
        self.scalars = scalars
        self.seeds_local = seeds_local
        self.seedw = np.zeros((pool.size, MAX_SEEDS), dtype=np.float32) if seedw is None else seedw
        self.n_edges = {"structural": 0, "ner": 0, "knn": 0}


def test_build_pool_is_the_sorted_union_and_counts_the_seeds_it_adds():
    m = load_compile()
    base = np.asarray([10, 3, 7])
    seeds = np.asarray([3, 42])
    pool, added = m.build_pool(base, seeds, np.asarray([7, 99]))
    assert pool.tolist() == [3, 7, 10, 42, 99]
    assert added == 1
    pool2, added2 = m.build_pool(base, seeds, None)
    assert pool2.tolist() == [3, 7, 10, 42] and added2 == 1


def test_gold_local_of_returns_only_the_golds_present():
    m = load_compile()
    pool = np.asarray([2, 5, 9, 14])
    assert m.gold_local_of(pool, np.asarray([5, 14, 100])).tolist() == [1, 3]
    assert m.gold_local_of(pool, np.asarray([1, 3, 15])).tolist() == []
    assert m.gold_local_of(pool, np.asarray([], dtype=np.int64)).tolist() == []


def test_univariate_recall5_matches_the_metric_function_on_a_written_cache(tmp_path):
    m = load_compile()
    rng = np.random.default_rng(3)
    writer = CacheWriter(tmp_path / "c", {})
    queries = []
    for qi in range(12):
        n = int(rng.integers(6, 40))
        pool = np.sort(rng.choice(5000, n, replace=False))
        X = rng.normal(size=(n, N_COLUMNS)).astype(np.float32)
        X[:, 3] = 0.0                                # a constant column
        gold_local = np.sort(rng.choice(n, int(rng.integers(1, 4)), replace=False))
        writer.add(f"q{qi}", qi, rng.normal(size=1536).astype(np.float32), _Compiled(pool, X, np.asarray([0])), gold_local, gold_local.size + 1)
        queries.append((X, gold_local, gold_local.size + 1))
    writer.write()
    got = m.univariate_recall5(tmp_path / "c")
    assert got.shape == (N_COLUMNS, 2)
    for col in (0, 3, 7, N_COLUMNS - 1):
        for s_i, sign in enumerate((1.0, -1.0)):
            expect = np.mean([rank_metrics(sign * X[:, col], g, t)["recall@5"] for X, g, t in queries])
            assert abs(got[col, s_i] - expect) < 1e-9, (col, sign)


def test_spearman_duplicates_flags_monotone_copies_only():
    m = load_compile()
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    sample = np.column_stack([a, np.exp(a), rng.normal(size=5000), np.zeros(5000), -a + 1e-3 * rng.normal(size=5000)])
    C, pairs = m.spearman_duplicates(sample, 0.98)
    flagged = {(i, j) for i, j, _ in pairs}
    assert (0, 1) in flagged and (0, 4) in flagged and (1, 4) in flagged
    assert not any(2 in p or 3 in p for p in flagged)
    assert np.isfinite(C).all()


def test_cache_writer_streams_the_row_arrays_and_reads_back_identically(tmp_path):
    from mp_retrieval.m3b_train import NpyAppender
    rng = np.random.default_rng(11)
    writer = CacheWriter(tmp_path / "c", {"kind": "toy"})
    pools, scalars, seedws = [], [], []
    for q in range(7):
        n = int(rng.integers(1, 40))
        pool = np.sort(rng.choice(500, size=n, replace=False)).astype(np.int64)
        x = rng.standard_normal((n, N_COLUMNS)).astype(np.float16)
        w = rng.random((n, MAX_SEEDS)).astype(np.float16)
        gold = np.asarray([0], dtype=np.int64) if n else np.empty(0, np.int64)
        writer.add(f"q{q}", q, rng.standard_normal(1536), _Compiled(pool, x, np.asarray([0], dtype=np.int64), w), gold, 1)
        pools.append(pool); scalars.append(x); seedws.append(w)
    meta = writer.write()
    assert meta["n_queries"] == 7 and meta["n_rows"] == sum(p.size for p in pools)
    for name, ref in (("pool", np.concatenate(pools).astype(np.int32)), ("scalars", np.concatenate(scalars)), ("seedw", np.concatenate(seedws))):
        got = np.load(tmp_path / "c" / f"{name}.npy", mmap_mode="r")
        assert got.shape == ref.shape and got.dtype == ref.dtype, name
        np.testing.assert_array_equal(np.asarray(got), ref)
    ptr = np.load(tmp_path / "c" / "pool_ptr.npy")
    assert ptr.tolist() == np.r_[0, np.cumsum([p.size for p in pools])].tolist()
    # an empty stream still reads back as a well-formed (0, k) array
    a = NpyAppender(tmp_path / "e.npy", np.float16, 3)
    a.close()
    assert np.load(tmp_path / "e.npy").shape == (0, 3)
