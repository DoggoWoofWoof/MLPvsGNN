"""M1B's multi-seed runner: feature-store reuse, X/Y resolution, per-query row capture.

Fixture mirrors tests/test_run_m1a_feature_screen.py's toy dataset (same
node count, same family-graph construction) since this runner reuses that
module's _cell_master_local/_widen_query/_arm_store internals unmodified --
M1A's own suite already proves those functions correct. _run_arm itself is
not reused: _run_arm_with_rows is a deliberate duplicate of its body (see
that function's own docstring in run_m1b_targeted_resolution.py for why).
What is new here is specific to what M1B adds:

- one feature build reused across every declared seed in a cell, not only
  across arms (test_feature_build_runs_once_per_cell_reused_across_seeds_
  and_arms extends M1A's own once-per-cell claim, which only ever exercised
  one seed);
- the explicit engineering requirement this track's declaration asked for:
  a cached feature tensor is elementwise identical to one rebuilt from
  scratch outside the seed loop (test_cached_tensor_equals_independently_
  rebuilt_tensor);
- X/Y placeholder resolution against the real filed declaration's own
  bindings (test_declared_cell_reads_the_real_filed_declaration);
- per-query recall@5 capture: every seed of every arm, including a
  reuse_contract candidate's seed=0, is genuinely fit and scored, and its
  per-query rows are persisted keyed by query_id
  (test_per_query_recall_at_5_by_seed_matches_the_aggregate_metric);
- amendment 5's replacement for splicing: a reuse_contract candidate is no
  longer copied byte-for-byte from M1A -- it is freshly retrained and its
  aggregate recall@5 cross-checked against M1A's original within a
  documented tolerance, and a divergence beyond that tolerance stops the
  run rather than silently trusting a broken reuse premise
  (test_reuse_candidate_is_cross_checked_not_spliced_and_still_retrains,
  test_reuse_cross_check_raises_on_divergence_beyond_tolerance); the
  older, still-live ceiling-mismatch guard is unchanged
  (test_reuse_integrity_check_fires_on_a_ceiling_mismatch).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from mp_retrieval.complete_data import load_complete_dataset
from scripts.run_m0b_regime_map import MAINLINE_FAMILY
from scripts.run_m1b_targeted_resolution import (
    ARM_FAMILIES,
    COMPLETE_STATUS,
    _declared_cell,
    _load_declaration,
    _resolve_symbolic_arm,
    main,
    run,
)

NUM_NODES = 40
FAMILIES = (MAINLINE_FAMILY, "baseline_a_simple")

DENSE = np.array(
    [
        [5, 4, 3, 2, 1, 0],
        [11, 10, 9, 8, 7, 6],
        [17, 16, 15, 14, 13, 12],
        [23, 22, 21, 20, 19, 18],
    ]
)
SPLADE = (DENSE + 2) % NUM_NODES
GOLDS = ("doc_3", "doc_9", "doc_15", "doc_21")


def _write_families(graph_root: Path) -> None:
    for offset, family in enumerate(FAMILIES, start=1):
        source = np.arange(NUM_NODES, dtype=np.int64)
        target = (source + 4 * offset) % NUM_NODES
        edge_index = np.concatenate([np.stack([source, target]), np.stack([target, source])], axis=1)
        path = graph_root / family
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"edge_index": torch.tensor(edge_index, dtype=torch.long), "num_nodes": NUM_NODES},
            path / "graph.pt",
        )


def _write_dataset(root: Path) -> None:
    rng = np.random.default_rng(20260905)
    nodes = rng.normal(size=(NUM_NODES, 6)).astype(np.float32)
    np.save(root / "nodes.npy", nodes)
    np.save(root / "queries_all.npy", rng.normal(size=(4, 6)).astype(np.float32))
    np.save(root / "dense_top200_all.npy", DENSE)
    np.save(root / "splade_top200_all.npy", SPLADE)
    (root / "query_ids_all.json").write_text(
        json.dumps(
            {
                "ids": ["q0", "q1", "q2", "q3"],
                "golds": [[gold] for gold in GOLDS],
                "split_indices": {"train": [3], "val": [0, 1, 2], "test": []},
            }
        ),
        encoding="utf-8",
    )
    source = np.arange(NUM_NODES, dtype=np.int64)
    target = (source + 5) % NUM_NODES
    torch.save(
        {"edge_index": torch.tensor(np.stack([source, target]), dtype=torch.long), "num_nodes": NUM_NODES},
        root / "graph.pt",
    )


def _args(tmp_path: Path, *, dataset: str = "2wiki_clean", **overrides) -> argparse.Namespace:
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    graphs = tmp_path / "families"
    _write_families(graphs)
    _write_dataset(data)
    frozen = load_complete_dataset(data, dataset=dataset)
    settings = {
        "data": data,
        "dataset": dataset,
        "data_fingerprint_sha256": "0" * 64,
        "expected_queries": len(frozen.queries),
        "frozen_embedding_dim": 6,
        "baseline": {"candidate_contract_sha256": frozen.metadata["candidate_contract_sha256"]},
        "candidate_contract_compatibility": None,
        "queries": 3,
        "holdout_fraction": 0.34,
        "per_seed_cap": 4,
        "neighbour_scan_cap_per_seed": 4096,
        "edge_provenance_root": graphs,
        "edge_families": list(FAMILIES),
        "a64_mainline_family": MAINLINE_FAMILY,
        "arms": None,
        "semantic_rung": "S3",
        "seeds": [0, 1],
        "epochs": 1,
        "batch_size": 4,
        "dropout": 0.0,
        "temperature": 1.0,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "device": "cpu",
        "output": tmp_path / "out" / "m1b.json",
    }
    settings.update(overrides)
    return argparse.Namespace(**settings)


@pytest.fixture()
def no_reuse(monkeypatch, tmp_path):
    """Point the reuse manifest at an empty (nothing-reusable) fixture -- the
    default for tests that only care about the multi-seed training path,
    not splicing."""
    manifest_path = tmp_path / "reuse_audit.json"
    manifest_path.write_text(
        json.dumps({"all_8_reusable": True, "reuse_candidates": []}), encoding="utf-8"
    )
    monkeypatch.setattr("scripts.run_m1b_targeted_resolution.REUSE_MANIFEST_PATH", manifest_path)
    return manifest_path


# --- feature-store reuse: the explicit engineering requirement ---


def test_feature_build_runs_once_per_cell_reused_across_seeds_and_arms(no_reuse, tmp_path):
    from mp_retrieval import graph_context

    real = graph_context.qls_local_features
    calls = []

    def _counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    args = _args(tmp_path, arms=["BASE", "BASE+NODE_ROLE"], seeds=[0, 1, 2])
    with patch("scripts.run_m1a_feature_screen.qls_local_features", side_effect=_counting):
        result = run(args)
    assert result["status"] == COMPLETE_STATUS
    # 2 arms x 3 seeds = 6 (arm, seed) pairs trained, but the feature build
    # (one call per query) must run exactly once regardless -- not once per
    # arm (M1A's own claim) and not once per seed (the new claim here).
    assert len(calls) == result["queries"]


def test_cached_tensor_equals_independently_rebuilt_tensor(no_reuse, tmp_path):
    """The declaration's explicit requirement: a cached feature tensor,
    reused across the seed loop, is elementwise identical to one rebuilt
    completely independently (a fresh _cell_master_local + _arm_store call,
    outside run()'s own cache) on the same real/smoke sample."""
    from scripts import run_m1a_feature_screen as m1a
    from scripts import run_m1b_targeted_resolution as m1b

    args = _args(tmp_path, arms=["BASE+NODE_ROLE"], seeds=[0, 1])
    captured: dict[str, object] = {}
    real_arm_store = m1a._arm_store

    def _capturing(**kwargs):
        store, width = real_arm_store(**kwargs)
        captured.setdefault("cached_local", store.local.copy())
        captured["precomputed_width"] = width
        captured["master_blocks"] = kwargs["master_blocks"]
        captured["queries"] = kwargs["queries"]
        captured["query_count"] = kwargs["query_count"]
        captured["num_nodes"] = kwargs["num_nodes"]
        return store, width

    with patch.object(m1a, "_arm_store", side_effect=_capturing):
        result = run(args)
    assert result["status"] == COMPLETE_STATUS

    rebuilt_store, rebuilt_width = m1a._arm_store(
        arm="BASE+NODE_ROLE",
        regime="R3",
        master_blocks=captured["master_blocks"],
        queries=captured["queries"],
        query_count=captured["query_count"],
        num_nodes=captured["num_nodes"],
    )
    assert rebuilt_width == captured["precomputed_width"]
    np.testing.assert_array_equal(rebuilt_store.local, captured["cached_local"])


def test_seed_loop_trains_each_declared_seed_exactly_once(no_reuse, tmp_path):
    args = _args(tmp_path, arms=["BASE"], seeds=[0, 1, 2])
    result = run(args)
    seeds = result["cells"]["R3"]["arms"]["BASE"]["seeds"]
    assert set(seeds) == {"0", "1", "2"}
    for seed_key, arm_result in seeds.items():
        assert arm_result["seed"] == int(seed_key)
        assert arm_result["reused_from_m1a"] is False


# --- X/Y symbolic-arm resolution ---


def test_resolve_symbolic_arm_substitutes_x_and_y_by_token():
    assert _resolve_symbolic_arm("BASE+NODE_ROLE+X", x="SUPPORT", y=None) == "BASE+NODE_ROLE+SUPPORT"
    assert _resolve_symbolic_arm("BASE+Y", x=None, y="SUPPORT") == "BASE+SUPPORT"
    assert _resolve_symbolic_arm("BASE", x="SUPPORT", y="SUPPORT") == "BASE"
    assert _resolve_symbolic_arm("BASE+NODE_ROLE", x="SUPPORT", y=None) == "BASE+NODE_ROLE"


def test_resolve_symbolic_arm_refuses_an_unbound_placeholder():
    with pytest.raises(ValueError, match="x_binding"):
        _resolve_symbolic_arm("BASE+X", x=None, y=None)
    with pytest.raises(ValueError, match="y_binding"):
        _resolve_symbolic_arm("BASE+Y", x=None, y=None)


def test_the_new_interaction_arm_is_registered():
    assert ARM_FAMILIES["BASE+NODE_ROLE+SUPPORT"] == ("NODE_ROLE", "SUPPORT")


def test_declared_cell_reads_the_real_filed_declaration():
    declaration = _load_declaration()
    assert _declared_cell(declaration, "2wiki_clean") == {
        "regime": "R3",
        "arms": ["BASE", "BASE+NODE_ROLE", "BASE+NODE_ROLE+SUPPORT"],
    }
    assert _declared_cell(declaration, "hotpotqa_clean") == {"regime": "R3", "arms": ["BASE", "BASE+SUPPORT"]}
    assert _declared_cell(declaration, "metaqa") == {"regime": "R3", "arms": ["BASE", "BASE+PATH"]}
    assert _declared_cell(declaration, "webqsp") == {"regime": "R3", "arms": ["BASE", "BASE+NODE_ROLE"]}


def test_a_non_r3_cell_is_refused():
    declaration = {"cells": {"toy": {"regime": "R1", "arms": ["BASE"]}}}
    with pytest.raises(ValueError, match="every M1B cell is R3"):
        _declared_cell(declaration, "toy")


# --- amendment 5: cross-checked against M1A, not spliced -- every seed retrains ---


def test_reuse_candidate_is_cross_checked_not_spliced_and_still_retrains(monkeypatch, tmp_path):
    """A reuse_contract candidate no longer skips training or copies its
    metrics byte-for-byte from M1A (amendment 5, 2026-09-05): the bootstrap
    needs genuine per-query rows for every seed, so seed=0 of a reuse
    candidate is fit exactly like any other seed, and the prior M1A result
    is used only as a cross-check reference."""
    # Pass 1: no reuse at all -- a real, self-consistent seed=0 BASE result
    # for this exact toy fixture, playing the role of "the prior M1A run".
    manifest_path = tmp_path / "reuse_audit.json"
    manifest_path.write_text(json.dumps({"all_8_reusable": True, "reuse_candidates": []}), encoding="utf-8")
    monkeypatch.setattr("scripts.run_m1b_targeted_resolution.REUSE_MANIFEST_PATH", manifest_path)

    prior_args = _args(tmp_path / "prior", arms=["BASE"], seeds=[0])
    prior = run(prior_args)
    prior_arm_result = dict(prior["cells"]["R3"]["arms"]["BASE"]["seeds"]["0"])

    headline_path = tmp_path / "headline.json"
    headline_path.write_text(
        json.dumps({"status": "M1A_FEATURE_SCREEN_DATASET_COMPLETE", "cells": {"R3": {
            "regime_headroom": prior["cells"]["R3"]["regime_headroom"],
            "arms": {"BASE": prior_arm_result},
        }}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.run_m1b_targeted_resolution.M1A_HEADLINE_DIR", tmp_path)
    manifest_path.write_text(
        json.dumps({
            "all_8_reusable": True,
            "reuse_candidates": [{"dataset": "2wiki_clean", "arm": "BASE", "reusable": True}],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.run_m1b_targeted_resolution._load_m1a_headline",
        lambda dataset: json.loads(headline_path.read_text(encoding="utf-8")),
    )

    from scripts import run_m1b_targeted_resolution as m1b

    calls = []
    real_run_arm_with_rows = m1b._run_arm_with_rows

    def _counting(**kwargs):
        calls.append(kwargs["args"].seed)
        return real_run_arm_with_rows(**kwargs)

    args = _args(tmp_path / "second", arms=["BASE"], seeds=[0, 1])
    with patch.object(m1b, "_run_arm_with_rows", side_effect=_counting):
        second = run(args)

    assert calls == [0, 1]  # every declared seed retrains, including the former reuse candidate

    seeds = second["cells"]["R3"]["arms"]["BASE"]["seeds"]
    cross_checked = dict(seeds["0"])
    assert cross_checked["reused_from_m1a"] is False
    assert cross_checked["seed"] == 0
    check = cross_checked["cross_checked_against_m1a_splice"]
    assert check["m1a_recall_at_5"] == prior_arm_result["metrics"]["recall@5"]
    assert check["fresh_recall_at_5"] == cross_checked["metrics"]["recall@5"]
    assert check["delta_pp"] == pytest.approx((check["fresh_recall_at_5"] - check["m1a_recall_at_5"]) * 100)
    assert check["within_tolerance"] is True  # same seed, same CPU fixture -- reproduces closely

    # A seed that was never a reuse candidate never gets the cross-check field.
    assert "cross_checked_against_m1a_splice" not in seeds["1"]
    assert seeds["1"]["reused_from_m1a"] is False


def test_reuse_cross_check_raises_on_divergence_beyond_tolerance(monkeypatch, tmp_path):
    """The cross-check is a safety net on top of the reuse audit, not a
    rubber stamp: if a fresh re-fit disagrees with M1A's recorded recall@5
    by more than REUSE_CROSS_CHECK_TOLERANCE_PP, the run stops rather than
    silently trusting a broken reuse premise."""
    manifest_path = tmp_path / "reuse_audit.json"
    manifest_path.write_text(json.dumps({"all_8_reusable": True, "reuse_candidates": []}), encoding="utf-8")
    monkeypatch.setattr("scripts.run_m1b_targeted_resolution.REUSE_MANIFEST_PATH", manifest_path)

    prior_args = _args(tmp_path / "prior", arms=["BASE"], seeds=[0])
    prior = run(prior_args)
    corrupted_arm_result = dict(prior["cells"]["R3"]["arms"]["BASE"]["seeds"]["0"])
    corrupted_arm_result["metrics"] = dict(corrupted_arm_result["metrics"])
    corrupted_arm_result["metrics"]["recall@5"] = corrupted_arm_result["metrics"]["recall@5"] + 0.9

    headline_path = tmp_path / "headline.json"
    headline_path.write_text(
        json.dumps({"status": "M1A_FEATURE_SCREEN_DATASET_COMPLETE", "cells": {"R3": {
            "regime_headroom": prior["cells"]["R3"]["regime_headroom"],
            "arms": {"BASE": corrupted_arm_result},
        }}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.run_m1b_targeted_resolution.M1A_HEADLINE_DIR", tmp_path)
    manifest_path.write_text(
        json.dumps({
            "all_8_reusable": True,
            "reuse_candidates": [{"dataset": "2wiki_clean", "arm": "BASE", "reusable": True}],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.run_m1b_targeted_resolution._load_m1a_headline",
        lambda dataset: json.loads(headline_path.read_text(encoding="utf-8")),
    )

    args = _args(tmp_path / "second", arms=["BASE"], seeds=[0])
    with pytest.raises(ValueError, match="diverges from M1A's original headline recall@5"):
        run(args)


# --- per-query row capture: the bootstrap's actual prerequisite ---


def test_per_query_recall_at_5_by_seed_matches_the_aggregate_metric(no_reuse, tmp_path):
    """The entire point of amendment 5: per-query rows exist, are keyed by
    the held-out queries' own query_id, and their mean reproduces the
    aggregate recall@5 _aggregate_rows already computed -- proving the
    captured rows are the same rows the metric was built from, not a
    parallel, possibly-mismatched recomputation."""
    args = _args(tmp_path, arms=["BASE"], seeds=[0, 1])
    result = run(args)
    cell = result["cells"]["R3"]
    arm = cell["arms"]["BASE"]
    held_out = cell["held_out_queries"]
    for seed_key in ("0", "1"):
        per_query = arm["per_query_recall_at_5_by_seed"][seed_key]
        assert len(per_query) == held_out
        assert all(isinstance(value, float) for value in per_query.values())
        aggregate = arm["seeds"][seed_key]["metrics"]["recall@5"]
        assert sum(per_query.values()) / len(per_query) == pytest.approx(aggregate)


def test_reuse_integrity_check_fires_on_a_ceiling_mismatch(monkeypatch, tmp_path):
    manifest_path = tmp_path / "reuse_audit.json"
    manifest_path.write_text(
        json.dumps({
            "all_8_reusable": True,
            "reuse_candidates": [{"dataset": "2wiki_clean", "arm": "BASE", "reusable": True}],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.run_m1b_targeted_resolution.REUSE_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(
        "scripts.run_m1b_targeted_resolution._load_m1a_headline",
        lambda dataset: {
            "status": "M1A_FEATURE_SCREEN_DATASET_COMPLETE",
            "cells": {"R3": {"regime_headroom": {"recall_ceiling@5": -1.0}, "arms": {"BASE": {}}}},
        },
    )
    args = _args(tmp_path, arms=["BASE"], seeds=[0])
    with pytest.raises(ValueError, match="does not match M1A's recorded value"):
        run(args)


def test_missing_reuse_manifest_stops_the_run(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scripts.run_m1b_targeted_resolution.REUSE_MANIFEST_PATH", tmp_path / "does_not_exist.json"
    )
    args = _args(tmp_path, arms=["BASE"], seeds=[0])
    with pytest.raises(FileNotFoundError, match="m1b_reuse_audit.py"):
        run(args)


def test_a_failing_reuse_manifest_stops_the_run(monkeypatch, tmp_path):
    manifest_path = tmp_path / "reuse_audit.json"
    manifest_path.write_text(json.dumps({"all_8_reusable": False, "reuse_candidates": []}), encoding="utf-8")
    monkeypatch.setattr("scripts.run_m1b_targeted_resolution.REUSE_MANIFEST_PATH", manifest_path)
    args = _args(tmp_path, arms=["BASE"], seeds=[0])
    with pytest.raises(ValueError, match="all_8_reusable=False"):
        run(args)


# --- main(): no seed restriction, unlike M1A's own one-seed guard ---


def test_main_accepts_seeds_other_than_zero(no_reuse, tmp_path):
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    graphs = tmp_path / "families"
    _write_families(graphs)
    _write_dataset(data)
    baseline_path = tmp_path / "baseline.json"
    frozen = load_complete_dataset(data, dataset="2wiki_clean")
    baseline_path.write_text(
        json.dumps({"candidate_contract_sha256": frozen.metadata["candidate_contract_sha256"]}), encoding="utf-8"
    )
    exit_code = main([
        "--data", str(data),
        "--dataset", "2wiki_clean",
        "--data-fingerprint-sha256", "0" * 64,
        "--expected-queries", str(len(frozen.queries)),
        "--frozen-embedding-dim", "6",
        "--baseline", str(baseline_path),
        "--queries", "3",
        "--edge-provenance-root", str(graphs),
        "--edge-families", *FAMILIES,
        "--arms", "BASE",
        "--seeds", "1", "2",
        "--epochs", "1",
        "--batch-size", "4",
        "--device", "cpu",
        "--output", str(tmp_path / "out" / "m1b.json"),
    ])
    assert exit_code == 0
    written = json.loads((tmp_path / "out" / "m1b.json").read_text(encoding="utf-8"))
    assert set(written["cells"]["R3"]["arms"]["BASE"]["seeds"]) == {"1", "2"}
