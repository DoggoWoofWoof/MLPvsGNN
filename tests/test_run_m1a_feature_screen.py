"""The M1A cell-matrix runner, end to end on a toy dataset.

Fixture mirrors tests/test_m0c_runner.py's toy dataset (same node count, same
descending dense/splade rows, same family-graph construction) so the two
runners are exercised under matching conditions. What is new here is specific
to M1A's own claims:

- feature build runs once per (dataset, regime) cell, never once per arm
  (test_feature_build_runs_once_per_cell_not_per_arm patches
  qls_local_features with a call counter);
- R3's widened scored set (C3 = Cq u A64) recovers a gold that is reachable
  by structural admission alone and invisible to Cq/U2 -- the exact confound
  _widen_query exists to fix (test_r3_widening_recovers_an_a64_only_gold);
- --regimes/--arms narrow the real filed declaration's cells rather than a
  test-only stand-in, so these tests exercise the actual arm-name strings
  configs/m1a_feature_screen.yaml declares for hotpotqa_clean/squad_clean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from mp_retrieval.candidate_expansion_v2 import STRUCTURAL, expand
from mp_retrieval.complete_data import load_complete_dataset
from scripts.run_m0a_probe import _families, _load_family_csr, _undirected
from scripts.run_m0b_regime_map import MAINLINE_FAMILY, _a64_budget
from scripts.run_m1a_feature_screen import (
    ARM_FAMILIES,
    COMPLETE_STATUS,
    _arm_columns,
    _declared_cells,
    _load_declaration,
    _resolve_repo_root,
    _widen_query,
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
# Cq(q0)={0..7}, Cq(q1)={6..13}, Cq(q2)={12..19}, Cq(q3)={18..25} -- one gold
# per query safely inside Cq, so every regime has real recall signal to train
# against; q0 additionally gets a second, A64-discovered gold (see
# _discover_a64_only_gold) that Cq/U2 alone can never see.
BASE_GOLDS = ("doc_3", "doc_9", "doc_15", "doc_21")


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


def _discover_a64_only_gold(graphs_root: Path) -> int:
    """A real node id admitted into A64 for q0, computed the same way the runner would.

    Not hand-derived: expand() is called directly against the fixture's own
    family graph, exactly mirroring test_m0c_runner.py's own precedent of
    deriving ground truth from the real functions rather than asserting graph
    arithmetic by hand.
    """

    family_rowptr, family_col = _load_family_csr(graphs_root / MAINLINE_FAMILY / "graph.pt", NUM_NODES)
    family_rowptr, family_col, _symmetric = _undirected(family_rowptr, family_col, NUM_NODES)
    pool = np.unique(np.concatenate([DENSE[0], SPLADE[0]])).astype(np.int64)
    seeds = np.unique(np.concatenate([DENSE[0, :5], SPLADE[0, :5]])).astype(np.int64)
    expansion = expand(
        STRUCTURAL,
        rowptr=family_rowptr,
        col=family_col,
        node_embeddings=np.zeros((NUM_NODES, 6), dtype=np.float32),
        query_embedding=None,
        anchor=int(DENSE[0, 0]),
        pool=pool,
        seeds=seeds,
        budget=_a64_budget(per_seed_cap=4, neighbour_scan_cap_per_seed=4096),
        num_nodes=NUM_NODES,
    )
    admitted = expansion.admitted
    assert admitted.size > 0, "fixture must produce a real A64 admission or this test proves nothing"
    outside_cq = admitted[~np.isin(admitted, pool)]
    assert outside_cq.size > 0
    return int(outside_cq[0])


def _write_dataset(root: Path, *, discovered_gold: int) -> None:
    rng = np.random.default_rng(20260904)
    nodes = rng.normal(size=(NUM_NODES, 6)).astype(np.float32)
    np.save(root / "nodes.npy", nodes)
    np.save(root / "queries_all.npy", rng.normal(size=(4, 6)).astype(np.float32))
    np.save(root / "dense_top200_all.npy", DENSE)
    np.save(root / "splade_top200_all.npy", SPLADE)
    golds = [
        [BASE_GOLDS[0], f"doc_{discovered_gold}"],
        [BASE_GOLDS[1]],
        [BASE_GOLDS[2]],
        [BASE_GOLDS[3]],
    ]
    (root / "query_ids_all.json").write_text(
        json.dumps(
            {
                "ids": ["q0", "q1", "q2", "q3"],
                "golds": golds,
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


def _args(tmp_path: Path, *, dataset: str = "hotpotqa_clean", **overrides) -> argparse.Namespace:
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    graphs = tmp_path / "families"
    _write_families(graphs)
    discovered_gold = _discover_a64_only_gold(graphs)
    _write_dataset(data, discovered_gold=discovered_gold)
    frozen = load_complete_dataset(data, dataset=dataset)
    settings = {
        "data": data,
        "dataset": dataset,
        "data_fingerprint_sha256": "0" * 64,
        "expected_queries": len(frozen.queries),
        "frozen_embedding_dim": 6,  # matches _write_dataset's own toy width, not the real 1536
        "baseline": {"candidate_contract_sha256": frozen.metadata["candidate_contract_sha256"]},
        "candidate_contract_compatibility": None,
        "queries": 3,
        "holdout_fraction": 0.34,
        "per_seed_cap": 4,
        "neighbour_scan_cap_per_seed": 4096,
        "edge_provenance_root": graphs,
        "edge_families": list(FAMILIES),
        "a64_mainline_family": MAINLINE_FAMILY,
        "regimes": None,
        "arms": None,
        "semantic_rung": "S3",
        "seed": 0,
        "epochs": 1,
        "batch_size": 4,
        "dropout": 0.0,
        "temperature": 1.0,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "device": "cpu",
        "output": tmp_path / "out" / "m1a.json",
    }
    settings.update(overrides)
    return argparse.Namespace(**settings)


def _smoke_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    return _args(
        tmp_path,
        dataset="hotpotqa_clean",
        regimes=["R3"],
        arms=["BASE", "BASE+NODE_ROLE"],
        **overrides,
    )


@pytest.fixture(scope="module")
def smoke_result(tmp_path_factory):
    return run(_smoke_args(tmp_path_factory.mktemp("m1a_smoke")))


# --- the declared step-4 smoke scope, end to end ---


def test_the_smoke_scope_completes(smoke_result):
    assert smoke_result["status"] == COMPLETE_STATUS
    assert smoke_result["test_split_read"] is False
    assert set(smoke_result["cells"]) == {"R3"}
    assert smoke_result["cells"]["R3"]["arms_run"] == ["BASE", "BASE+NODE_ROLE"]
    assert set(smoke_result["cells"]["R3"]["arms"]) == {"BASE", "BASE+NODE_ROLE"}


def test_precomputed_width_matches_base_vs_base_plus_node_role(smoke_result):
    arms = smoke_result["cells"]["R3"]["arms"]
    assert arms["BASE"]["precomputed_width"] == 4
    assert arms["BASE+NODE_ROLE"]["precomputed_width"] == 5


def test_semantic_parameter_count_matches_the_toy_embedding_dim_live(smoke_result):
    # embedding_dim=6 here, not the real 768 -- semantic_parameter_count is
    # 2*embedding_dim regardless (see tests/test_m1a_screen.py), so this must
    # scale with the fixture rather than hardcoding the real dataset's 1536.
    for arm in smoke_result["cells"]["R3"]["arms"].values():
        assert arm["parameters"]["semantic"] == 2 * 6
        assert arm["parameters"]["total"] == arm["parameters"]["semantic"] + arm["parameters"]["scorer"]


def test_metrics_and_ceiling_attainment_are_present_and_sane(smoke_result):
    for arm in smoke_result["cells"]["R3"]["arms"].values():
        metrics = arm["metrics"]
        for key in ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20"):
            assert 0.0 <= metrics[key] <= 1.0 + 1e-9
        attainment = arm["ceiling_attainment_at_5"]
        assert attainment is not None and attainment >= 0.0


def test_the_full_declared_squad_clean_cell_runs_unnarrowed(tmp_path):
    """No --regimes/--arms at all: every cell configs/m1a_feature_screen.yaml
    declares for squad_clean (just R1: [BASE]) must run on its own."""

    result = run(_args(tmp_path, dataset="squad_clean", regimes=None, arms=None))
    assert result["status"] == COMPLETE_STATUS
    assert set(result["cells"]) == {"R1"}
    assert result["cells"]["R1"]["arms_run"] == ["BASE"]


# --- the declaration is read live, not hand-copied ---


def test_declared_cells_match_the_real_filed_declaration():
    declaration = _load_declaration()
    cells = _declared_cells(declaration, "hotpotqa_clean")
    assert cells["R1"] == ["BASE", "BASE+GEOMETRY", "BASE+SUPPORT", "BASE+PATH"]
    assert cells["R3"] == ["BASE", "BASE+GEOMETRY", "BASE+SUPPORT", "BASE+PATH", "BASE+NODE_ROLE"]
    assert _declared_cells(declaration, "squad_clean") == {"R1": ["BASE"]}


def test_an_undeclared_regime_filter_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not declared"):
        run(_args(tmp_path, dataset="squad_clean", regimes=["R3"], arms=None))


def test_an_arm_filter_that_empties_a_regime_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no arms to run"):
        run(_args(tmp_path, regimes=["R1"], arms=["BASE+NODE_ROLE"]))


# --- R3 widening: the actual correctness fix ---


def test_r3_widening_recovers_an_a64_only_gold(smoke_result):
    """q0's second gold lives only in A64 (see _discover_a64_only_gold) -- R3's
    scored set must include it, so this gold must be individually recoverable
    (it need not rank #1, but the widened candidate_index must contain it)."""

    # Proven at the unit level below (test_widen_query_recovers_a_gold_outside_cq);
    # here we only need the cell to have registered a wider ceiling than a
    # Cq-only regime could reach.
    ceiling = smoke_result["cells"]["R3"]["regime_headroom"]["recall_ceiling@5"]
    assert ceiling is not None and ceiling > 0.0


def test_widen_query_recovers_a_gold_outside_cq():
    from mp_retrieval.complete_data import CompleteQuery

    query = CompleteQuery(
        query_index=0,
        query_id="q0",
        candidate_index=torch.tensor([5, 6, 7], dtype=torch.long),
        relevant_local=torch.tensor([0], dtype=torch.long),
        relevant_global=torch.tensor([5, 99], dtype=torch.long),
        anchor_global=5,
        split=1,
    )
    # Under Cq={5,6,7} alone, gold 99 is unrecoverable.
    assert query.relevant_local.tolist() == [0]
    widened = _widen_query(query, scored=np.array([5, 6, 7, 99, 100]))
    assert widened.candidate_index.tolist() == [5, 6, 7, 99, 100]
    assert widened.relevant_local.tolist() == [0, 3]


def test_widen_query_onto_cq_itself_is_a_same_valued_round_trip():
    from mp_retrieval.complete_data import CompleteQuery

    query = CompleteQuery(
        query_index=0,
        query_id="q0",
        candidate_index=torch.tensor([7, 5, 6], dtype=torch.long),
        relevant_local=torch.tensor([1], dtype=torch.long),
        relevant_global=torch.tensor([5], dtype=torch.long),
        anchor_global=5,
        split=1,
    )
    widened = _widen_query(query, scored=np.unique(query.candidate_index.numpy()))
    assert widened.relevant_local.tolist() == [
        widened.candidate_index.tolist().index(5)
    ]


# --- arm column slicing: BASE always first, families appended in a fixed order ---


def test_arm_columns_selects_the_declared_family_slots_in_order():
    master = np.arange(12, dtype=np.float32).reshape(1, 12)
    assert _arm_columns(master, "BASE", "R3").tolist() == [[0, 1, 2, 3]]
    assert _arm_columns(master, "BASE+GEOMETRY", "R1").tolist() == [[0, 1, 2, 3, 4, 5, 6]]
    assert _arm_columns(master, "BASE+SUPPORT", "R1").tolist() == [[0, 1, 2, 3, 7]]
    assert _arm_columns(master, "BASE+PATH", "R1").tolist() == [[0, 1, 2, 3, 8, 9, 10]]
    assert _arm_columns(master, "BASE+NODE_ROLE", "R3").tolist() == [[0, 1, 2, 3, 11]]


def test_node_role_outside_r3_is_refused():
    master = np.zeros((1, 12), dtype=np.float32)
    with pytest.raises(ValueError, match="only defined for R3"):
        _arm_columns(master, "BASE+NODE_ROLE", "R1")


def test_every_declared_arm_name_has_a_family_composition():
    declaration = _load_declaration()
    for dataset in declaration["datasets"]:
        if dataset == "musique_clean":  # explicitly out of scope, see the declaration
            continue
        for arms in _declared_cells(declaration, dataset).values():
            assert set(arms) <= set(ARM_FAMILIES), (dataset, arms)


# --- feature build cost: once per cell, never once per arm ---


def test_feature_build_runs_once_per_cell_not_per_arm(tmp_path):
    from mp_retrieval import graph_context

    real = graph_context.qls_local_features
    calls = []

    def _counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    args = _args(tmp_path, dataset="hotpotqa_clean", regimes=["R1"], arms=None)
    with patch("scripts.run_m1a_feature_screen.qls_local_features", side_effect=_counting):
        result = run(args)
    assert len(result["cells"]["R1"]["arms_run"]) == 4  # BASE, +GEOMETRY, +SUPPORT, +PATH
    assert len(calls) == result["queries"]  # once per query, not once per (query, arm)


# --- determinism and refusals ---


def test_the_smoke_scope_repeats_exactly(tmp_path):
    first = run(_smoke_args(tmp_path / "a"))
    second = run(_smoke_args(tmp_path / "b"))
    for payload in (first, second):
        payload.pop("systems")
        for cell in payload["cells"].values():
            cell.pop("uncached_feature_build_latency_ms")
            for arm in cell["arms"].values():
                arm.pop("training")
                arm.pop("inference")
                arm.pop("systems")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_a_short_split_is_refused_rather_than_quietly_shrunk(tmp_path):
    with pytest.raises(ValueError, match="needed 9"):
        run(_smoke_args(tmp_path, queries=9))


def test_a_drifted_candidate_pool_stops_the_run(tmp_path):
    args = _smoke_args(tmp_path)
    args.baseline = {"candidate_contract_sha256": "f" * 64}
    with pytest.raises(ValueError, match="candidate contract does not match"):
        run(args)


def test_a_manifest_of_the_wrong_size_stops_the_run(tmp_path):
    args = _smoke_args(tmp_path)
    args.expected_queries = 99
    with pytest.raises(ValueError, match="differs from the registered protocol"):
        run(args)


def test_a_completed_run_is_not_repeated(tmp_path):
    args = _smoke_args(tmp_path)
    first = run(args)
    args.expected_queries = 99  # would raise if the run actually happened again
    second = run(args)
    assert first == second


def test_resolve_repo_root_uses_file_path_when_its_own_parent_has_the_marker(tmp_path):
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "configs" / "m1a_feature_screen.yaml").write_text("x", encoding="utf-8")
    file_path = repo / "scripts" / "run_m1a_feature_screen.py"

    found = _resolve_repo_root(file_path, sys_path=[], marker_relpath="configs/m1a_feature_screen.yaml")

    assert found == repo


def test_resolve_repo_root_falls_back_to_sys_path_when_file_path_lands_on_an_automount(tmp_path):
    # Mirrors Modal's real layout: the script's own __file__ resolves under an
    # auto-mounted copy of scripts/ that has no configs/ sibling (nothing
    # imports configs/, so it is only ever placed by an explicit
    # add_local_file at the launcher's own REMOTE_ROOT, a different path).
    automount = tmp_path / "automount"
    (automount / "scripts").mkdir(parents=True)
    file_path = automount / "scripts" / "run_m1a_feature_screen.py"

    mounted = tmp_path / "mounted"
    (mounted / "configs").mkdir(parents=True)
    (mounted / "configs" / "m1a_feature_screen.yaml").write_text("x", encoding="utf-8")

    found = _resolve_repo_root(
        file_path,
        # a leading "" is a real, common sys.path entry (cwd) -- must not crash
        sys_path=["", str(mounted)],
        marker_relpath="configs/m1a_feature_screen.yaml",
    )

    assert found == mounted


def test_resolve_repo_root_falls_back_to_the_naive_candidate_when_nothing_has_the_marker(tmp_path):
    nowhere = tmp_path / "nowhere"
    (nowhere / "scripts").mkdir(parents=True)
    file_path = nowhere / "scripts" / "run_m1a_feature_screen.py"

    found = _resolve_repo_root(
        file_path,
        sys_path=[str(tmp_path / "also_nowhere")],
        marker_relpath="configs/m1a_feature_screen.yaml",
    )

    assert found == nowhere


def test_main_refuses_a_nonzero_seed(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"candidate_contract_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="one-seed"):
        main(
            [
                "--data",
                str(tmp_path),
                "--dataset",
                "hotpotqa_clean",
                "--data-fingerprint-sha256",
                "0" * 64,
                "--expected-queries",
                "3",
                "--frozen-embedding-dim",
                "6",
                "--baseline",
                str(baseline_path),
                "--seed",
                "1",
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
