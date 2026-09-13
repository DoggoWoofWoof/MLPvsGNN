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


def test_trim_keeps_every_requested_value_and_refuses_the_full_layout_stages(tmp_path, monkeypatch):
    import json
    from mp_retrieval.m3b_features import COLUMNS
    from mp_retrieval.m3b_train import CarveData
    m = load_compile()
    rng = np.random.default_rng(5)
    writer = CacheWriter(tmp_path / "fit", {"kind": "fit"})
    for q in range(5):
        n = int(rng.integers(3, 30))
        pool = np.sort(rng.choice(200, size=n, replace=False)).astype(np.int64)
        writer.add(f"q{q}", q, rng.standard_normal(1536), _Compiled(pool, rng.standard_normal((n, N_COLUMNS)).astype(np.float16), np.asarray([0])),
                   np.asarray([1], dtype=np.int64), 1)
    writer.write()
    surviving = [COLUMNS[k] for k in (0, 7, 3, 50, 110)]          # declared order is whatever the screen keeps; positions are mapped by name
    requested = np.asarray([50, 3, 110], dtype=np.int64)
    before = CarveData(tmp_path / "fit", context=None, columns=requested)
    ref = [before.query(i)["x"] for i in range(before.n_queries)]
    full_before = np.asarray(np.load(tmp_path / "fit" / "scalars.npy"))
    del before                                        # a reader's mapping would hold the file open on Windows
    import gc
    gc.collect()
    meta = m.trim_cache(tmp_path / "fit", surviving, "sha")
    assert meta["columns_stored"] == surviving and meta["scalars_bytes"] < meta["scalars_bytes_before_trim"]
    trimmed = np.load(tmp_path / "fit" / "scalars.npy", mmap_mode="r")
    assert trimmed.shape == (full_before.shape[0], 5)
    np.testing.assert_array_equal(np.asarray(trimmed), full_before[:, [0, 7, 3, 50, 110]])
    after = CarveData(tmp_path / "fit", context=None, columns=requested)
    for i in range(after.n_queries):
        np.testing.assert_array_equal(after.query(i)["x"], ref[i])
    del trimmed, after
    gc.collect()
    # the same call again is a no-op, a missing column refuses, and the full-layout stages refuse
    assert m.trim_cache(tmp_path / "fit", surviving, "sha")["columns_stored"] == surviving
    import pytest
    with pytest.raises(ValueError):
        CarveData(tmp_path / "fit", context=None, columns=np.asarray([1]))
    with pytest.raises(ValueError):
        CarveData(tmp_path / "fit", context=None)
    with pytest.raises(SystemExit):
        m.univariate_recall5(tmp_path / "fit")
    with pytest.raises(SystemExit):
        m.column_stats(tmp_path / "fit")


def test_disk_guard_reads_amendment_2_and_halts_at_its_floor(tmp_path, monkeypatch):
    import pytest
    m = load_compile()
    with pytest.raises(SystemExit):
        m.DiskGuard({})
    guard = m.DiskGuard({"amendment_2_2026_09_13_systems": {"disk_guard": {"halt_below_free_gb": 0.0, "cache_bound_gb": 10}}})
    monkeypatch.setattr(m, "CACHE", tmp_path / "cache")
    guard.check()                                    # nothing written, plenty of disk above a 0 GB floor
    tight = m.DiskGuard({"amendment_2_2026_09_13_systems": {"disk_guard": {"halt_below_free_gb": 10**6, "cache_bound_gb": 10}}})
    with pytest.raises(SystemExit):
        tight.check()
    (tmp_path / "cache" / "x" / "fit").mkdir(parents=True)
    (tmp_path / "cache" / "x" / "fit" / "scalars.npy").write_bytes(b"0" * 2048)
    bound = m.DiskGuard({"amendment_2_2026_09_13_systems": {"disk_guard": {"halt_below_free_gb": 0.0, "cache_bound_gb": 1e-6}}})
    with pytest.raises(SystemExit):
        bound.check(writing=tmp_path / "cache" / "x" / "fit")
    assert not (tmp_path / "cache" / "x" / "fit").exists()


def test_stage_file_appends_each_block_once_and_refuses_a_second_filing(tmp_path, monkeypatch):
    import json
    import yaml
    m = load_compile()
    monkeypatch.setattr(m, "CONFIG", tmp_path / "decl.yaml")
    monkeypatch.setattr(m, "OUT", tmp_path / "out")
    monkeypatch.setattr(m, "CACHE", tmp_path / "out" / "cache")
    monkeypatch.setattr(m, "RELATIONS", tmp_path / "out" / "relations")
    m.OUT.mkdir()
    (tmp_path / "decl.yaml").write_text("status: DECLARED_NOT_RUN\npopulations:\n  eval_splits:\n    squad: dev\n", encoding="utf-8")
    carves = {"utc": "u", "rule": {"select": "s"}, "per_dataset": {"squad": {"N": 10, "select": 2, "fit": 4, "select_sha256": "S", "fit_sha256": "F"}}}
    (m.OUT / "carves.json").write_text(json.dumps(carves), encoding="utf-8")
    for kind, sha in (("select", "S"), ("fit", "F")):
        d = m.CACHE / "squad" / kind
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"population": {"ids_sha256": sha, "zero_gold_excluded": 0}, "n_queries": 2, "n_rows": 9, "candidates_mean": 4.5,
                                                 "gold_in_pool_mean": 1.0, "queries_with_no_gold_in_pool": 0, "seeds_added_mean": 0.0, "ms_per_query": 1.0,
                                                 "bytes": 100, "relation_table": False, "pool": "p", "contract_block": "c"}), encoding="utf-8")
    (m.OUT / "base_score.json").write_text(json.dumps({"utc": "u", "selected": "rrf", "base_index": 1, "candidates": ["dense_cos", "rrf"],
                                                       "macro_select_recall5": {"dense_cos": 0.1, "rrf": 0.2}, "per_dataset": {"squad": {"dense_cos": 0.1, "rrf": 0.2}},
                                                       "rule": "r"}), encoding="utf-8")
    (m.OUT / "feature_screen.json").write_text("{}", encoding="utf-8")
    (m.OUT / "qls_u_core_contract_block.yaml").write_text(yaml.safe_dump({"name": "QLS_U_CORE_CONTRACT", "surviving": ["rrf"]}), encoding="utf-8")
    cfg = yaml.safe_load((tmp_path / "decl.yaml").read_text(encoding="utf-8"))
    m.stage_file(cfg, "2026_09_13", ["carves", "base", "core"])
    reloaded = yaml.safe_load((tmp_path / "decl.yaml").read_text(encoding="utf-8"))
    assert reloaded["carve_record_2026_09_13"]["per_dataset"]["squad"]["fit_compiled"]["rows"] == 9
    assert reloaded["fixed_base_score_selected_2026_09_13"]["selected"] == "rrf"
    assert reloaded["qls_u_core_contract_2026_09_13"]["surviving"] == ["rrf"]
    assert reloaded["status"] == "DECLARED_NOT_RUN"           # nothing above the appended blocks is touched
    import pytest
    with pytest.raises(SystemExit, match="already filed"):
        m.stage_file(reloaded, "2026_09_14", ["base"])
    # a compiled population that is not the carve refuses before anything is appended
    (m.CACHE / "squad" / "fit" / "meta.json").write_text(json.dumps({"population": {"ids_sha256": "X"}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not the carve"):
        m.stage_file({"populations": {"eval_splits": {"squad": "dev"}}}, "2026_09_15", ["carves"])
