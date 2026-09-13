"""The run script's eval sharding: shards k::N of a population merge back into
population order, and every merged quantity is recomputed from the arrays."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mp_retrieval.m3b_train import METRIC_NAMES, rank_metrics  # noqa: E402


def load_run():
    spec = importlib.util.spec_from_file_location("m3b_run", ROOT / "scripts" / "m3b_run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["m3b_run"] = module
    spec.loader.exec_module(module)
    return module


def _population(rng, n=23, scorers=("gat", "fixed:rrf")):
    ids = [f"q{i}" for i in range(n)]
    pools = [rng.integers(30, 60) for _ in range(n)]
    golds = [rng.integers(1, 4) for _ in range(n)]
    metrics = {s: {m: np.zeros(n) for m in METRIC_NAMES} for s in scorers}
    for i in range(n):
        in_pool = rng.integers(0, golds[i] + 1)
        gold_local = rng.choice(pools[i], size=in_pool, replace=False).astype(np.int64)
        for s in scorers:
            r = rank_metrics(rng.standard_normal(pools[i]), gold_local, int(golds[i]))
            for m in METRIC_NAMES:
                metrics[s][m][i] = r[m]
    return ids, np.asarray(pools), metrics


def _write_shards(run, eval_dir, ids, sizes, metrics, N):
    n = len(ids)
    digest = hashlib.sha256(",".join(ids).encode("utf-8")).hexdigest()
    scorers = list(metrics)
    for k in range(N):
        sl = slice(k, n, N)
        arrays = {f"{s}/{m}": metrics[s][m][sl] for s in scorers for m in METRIC_NAMES}
        arrays["pool_size"] = sizes[sl].astype(np.int64)
        np.savez_compressed(eval_dir / f"toy__shard{k}of{N}.npz", **arrays)
        gip, gt = metrics[scorers[0]]["gold_in_pool"][sl], metrics[scorers[0]]["gold_total"][sl]
        ceiling = run.ceiling_from_arrays(gip, gt, sizes[sl])
        record = {"dataset": "toy", "utc": "t", "queries": int(len(ids[sl])), "zero_gold_excluded": 0, "ids_sha256": "x",
                  "shard": {"k": k, "N": N, "population_queries": n, "population_ids_sha256": digest},
                  "contract_block": "cb", "pool": "p", "construction": {"a": 1}, "ceiling_as_compiled": ceiling, "seeds_added_mean": float(k),
                  "scorers": scorers, "chunk_queries": 4, "seconds": 1.5, "ms_per_query": 10.0 * (k + 1), "latency": {"compile": {"p50": 1}},
                  "peak_rss_bytes": 100 + k, "threads": 2, "freeze_RECORD_SHA256": "f", "mrr_audit": {},
                  "summary": {s: {m: float(metrics[s][m][sl].mean()) for m in ("recall@1", "recall@5", "recall@20", "hit@1", "mrr")} for s in scorers}}
        (eval_dir / f"toy__shard{k}of{N}.json").write_text(json.dumps(record), encoding="utf-8")
        (eval_dir / f"toy__shard{k}of{N}_query_ids.json").write_text(json.dumps(ids[sl]), encoding="utf-8")
    return digest


def test_merge_restores_population_order_and_recomputes_every_summary(tmp_path, monkeypatch):
    run = load_run()
    monkeypatch.setattr(run, "EVAL", tmp_path)
    rng = np.random.default_rng(3)
    ids, sizes, metrics = _population(rng)
    digest = _write_shards(run, tmp_path, ids, sizes, metrics, N=3)
    record = run.merge_shards("toy", log=lambda m: None)
    assert record is not None and record["queries"] == len(ids) and record["ids_sha256"] == digest
    assert json.loads((tmp_path / "toy_query_ids.json").read_text()) == ids
    with np.load(tmp_path / "toy.npz") as z:
        for s in metrics:
            for m in METRIC_NAMES:
                np.testing.assert_array_equal(z[f"{s}/{m}"], metrics[s][m])
        np.testing.assert_array_equal(z["pool_size"], sizes)
    gip, gt = metrics["gat"]["gold_in_pool"], metrics["gat"]["gold_total"]
    expected = run.ceiling_from_arrays(gip, gt, sizes)
    for key in ("recall_ceiling@1", "recall_ceiling@5", "recall_ceiling@20", "any_gold_at_pool", "all_gold_at_pool", "candidates_mean", "candidates_max"):
        assert abs(record["ceiling_as_compiled"][key] - expected[key]) < 1e-12, key
    assert abs(record["ceiling_as_compiled"]["fraction_of_attainable@5"]
               - expected["recall_ceiling@5"] / float((np.minimum(gt, 5) / gt).mean())) < 1e-12
    for s in metrics:
        assert record["summary"][s]["recall@5"] == round(float(metrics[s]["recall@5"].mean()), 4)
        assert record["mrr_audit"][s]["ok"]
    assert record["peak_rss_bytes"] == 102 and record["seconds"] == 4.5 and record["latency"] == {"compile": {"p50": 1}}
    assert record["shard"] is None and [m["k"] for m in record["merged_from"]] == [0, 1, 2]
    assert (tmp_path / "toy.json").exists()


def test_merge_waits_for_every_shard(tmp_path, monkeypatch):
    run = load_run()
    monkeypatch.setattr(run, "EVAL", tmp_path)
    rng = np.random.default_rng(4)
    ids, sizes, metrics = _population(rng, n=10)
    _write_shards(run, tmp_path, ids, sizes, metrics, N=2)
    (tmp_path / "toy__shard1of2.json").unlink()
    assert run.merge_shards("toy", log=lambda m: None) is None
    assert not (tmp_path / "toy.json").exists()


def test_shard_suffix_and_slices_partition_the_population():
    run = load_run()
    assert run.shard_suffix(None) == "" and run.shard_suffix((2, 5)) == "__shard2of5"
    n = 101
    parts = [list(range(k, n, 4)) for k in range(4)]
    assert sorted(sum(parts, [])) == list(range(n))
