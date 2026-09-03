"""The M0C bounded-R3 runner, end to end on a toy dataset.

Mirrors tests/test_m0b_runner.py's toy fixture exactly (same node count, same
descending dense/splade rows to catch a caller that forgets to sort a pool
before it becomes qls_local_features' nodes= argument, same two-family setup)
so the two runners are exercised under matching conditions. What is new here
is specific to M0C's own claim: U3_bounded = stable_union(U2, A64), never
TARGET_H1(Cq_struct), and every node admitted into C3 lands inside U3_bounded
without a second graph walk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mp_retrieval.complete_data import load_complete_dataset
from mp_retrieval.data import QuerySplit
from mp_retrieval.graph_context import build_operators, context_nodes
from mp_retrieval.overlap_audit import NODE_ROLES
from scripts.run_m0a_probe import QueryView, _families, _load_family_csr, _undirected
from scripts.run_m0b_regime_map import CONTEXT_ARM, _a64_budget
from scripts.run_m0c_bounded_r3 import (
    A64_GRAPH_EXPANSION_CAP,
    COMPLETE_STATUS,
    MAINLINE_FAMILY,
    run,
)
from mp_retrieval.candidate_expansion_v2 import STRUCTURAL, expand

NUM_NODES = 40
FAMILIES = (MAINLINE_FAMILY, "baseline_a_simple")


DEFAULT_GOLDS = (
    ("doc_30", "doc_33"),
    ("doc_31",),
    ("doc_29", "doc_34"),
    ("doc_32",),
)


def _write_dataset(root: Path, *, golds=DEFAULT_GOLDS) -> None:
    rng = np.random.default_rng(20260903)
    nodes = rng.normal(size=(NUM_NODES, 6)).astype(np.float32)
    np.save(root / "nodes.npy", nodes)
    np.save(root / "queries_all.npy", rng.normal(size=(4, 6)).astype(np.float32))
    dense = np.array(
        [
            [5, 4, 3, 2, 1, 0],
            [11, 10, 9, 8, 7, 6],
            [17, 16, 15, 14, 13, 12],
            [23, 22, 21, 20, 19, 18],
        ]
    )
    splade = (dense + 2) % NUM_NODES
    np.save(root / "dense_top200_all.npy", dense)
    np.save(root / "splade_top200_all.npy", splade)
    (root / "query_ids_all.json").write_text(
        json.dumps(
            {
                "ids": ["q0", "q1", "q2", "q3"],
                "golds": [list(row) for row in golds],
                "split_indices": {"train": [3], "val": [0, 1, 2], "test": []},
            }
        ),
        encoding="utf-8",
    )
    source = np.arange(NUM_NODES, dtype=np.int64)
    target = (source + 5) % NUM_NODES
    torch.save(
        {
            "edge_index": torch.tensor(np.stack([source, target]), dtype=torch.long),
            "num_nodes": NUM_NODES,
        },
        root / "graph.pt",
    )


def _write_families(graph_root: Path) -> None:
    for offset, family in enumerate(FAMILIES, start=1):
        source = np.arange(NUM_NODES, dtype=np.int64)
        target = (source + 4 * offset) % NUM_NODES
        edge_index = np.concatenate(
            [np.stack([source, target]), np.stack([target, source])], axis=1
        )
        path = graph_root / family
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"edge_index": torch.tensor(edge_index, dtype=torch.long), "num_nodes": NUM_NODES},
            path / "graph.pt",
        )


def _args(tmp_path: Path, *, golds=DEFAULT_GOLDS, **overrides) -> argparse.Namespace:
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    graphs = tmp_path / "families"
    _write_dataset(data, golds=golds)
    _write_families(graphs)
    frozen = load_complete_dataset(data, dataset="toy")
    settings = {
        "data": data,
        "dataset": "toy",
        "data_fingerprint_sha256": "0" * 64,
        "expected_queries": len(frozen.queries),
        "baseline": {
            "candidate_contract_sha256": frozen.metadata["candidate_contract_sha256"]
        },
        "candidate_contract_compatibility": None,
        "queries": 3,
        "per_seed_cap": 4,
        "neighbour_scan_cap_per_seed": 4096,
        "edge_provenance_root": graphs,
        "edge_families": list(FAMILIES),
        "a64_mainline_family": MAINLINE_FAMILY,
        "output": tmp_path / "out" / "bounded_r3.json",
    }
    settings.update(overrides)
    return argparse.Namespace(**settings)


@pytest.fixture(scope="module")
def bounded_map(tmp_path_factory):
    return run(_args(tmp_path_factory.mktemp("m0c")))


def test_the_run_completes_and_says_what_it_did_not_do(bounded_map):
    assert bounded_map["status"] == COMPLETE_STATUS
    assert bounded_map["trained_anything"] is False
    assert bounded_map["test_split_read"] is False
    assert bounded_map["a64_mainline_family"] == MAINLINE_FAMILY
    assert bounded_map["a64_graph_expansion_cap"] == A64_GRAPH_EXPANSION_CAP == 64
    assert "R3_FULL_REEXPANSION" in bounded_map["m0b_r3_full_is_unchanged"]


# --- invariants: R1/R2 unchanged, then M0C's own gated invariants ---


def test_r1_and_r2_ceilings_are_bit_exact(bounded_map):
    assert bounded_map["invariants"]["oracle_r1_equals_oracle_r2_bit_exact"] is True
    assert bounded_map["invariants"]["scored_r1_equals_scored_r2"] is True


def test_u2_contains_cq_for_every_query(bounded_map):
    assert bounded_map["invariants"]["u2_contains_cq"] is True
    assert bounded_map["r2"]["arm"] == "TARGET_H1"


def test_m0c_invariants_hold_for_every_family(bounded_map):
    for family in FAMILIES:
        assert bounded_map["invariants"][f"{family}.a64_disjoint_from_cq"] is True
        assert bounded_map["invariants"][f"{family}.admitted_delta_within_universal_cap"] is True
        assert bounded_map["invariants"][f"{family}.cq_struct_equals_cq_union_a64"] is True
        assert bounded_map["invariants"][f"{family}.scored_r1_subset_scored_r3"] is True
        # M0C's own two new invariants: bounded U3 contains everything it must,
        # by construction, and this checks that construction rather than
        # assuming it.
        assert bounded_map["invariants"][f"{family}.u2_subset_u3_bounded"] is True
        assert bounded_map["invariants"][f"{family}.c3_subset_u3_bounded"] is True


# --- R3_bounded: reported per family, context is the bounded definition ---


def test_r3_bounded_is_reported_for_every_family_with_the_mainline_flagged(bounded_map):
    assert set(bounded_map["r3_bounded"]) == set(FAMILIES)
    for family, row in bounded_map["r3_bounded"].items():
        assert row["is_mainline"] == (family == MAINLINE_FAMILY)
        assert row["admitted_per_query"]["mean"] >= 0.0
        assert 0 <= row["context_node_count"]["mean"]
        assert "TARGET_H1(Cq_struct)" not in row["context_definition"] or "not" in row["context_definition"]


def test_containment_and_gold_overlap_are_reported_same_shape_as_m0b(bounded_map):
    from mp_retrieval.overlap_audit import CROSS_TAB_CELLS

    for row in bounded_map["r3_bounded"].values():
        containment = row["admitted_node_containment_in_u2"]
        if not containment["undefined_because_nothing_was_admitted"]:
            assert 0.0 <= containment["containment_rate"] <= 1.0
        cells = row["gold_overlap_vs_u2"]["cross_tabulation"]["gold_instances"]
        assert set(cells) == set(CROSS_TAB_CELLS)


# --- NODE_ROLE, under the bounded regime ---


def test_node_roles_cover_r1_r2_and_r3_bounded_only_for_the_mainline_family(bounded_map):
    assert set(bounded_map["node_roles"]) == {"R1", "R2", "R3_BOUNDED"}
    for tag, counts in bounded_map["node_roles"].items():
        assert set(counts) == set(NODE_ROLES), tag


def test_r3_bounded_node_role_totals_match_admitted_bookkeeping(bounded_map):
    roles = bounded_map["node_roles"]["R3_BOUNDED"]
    mainline = bounded_map["r3_bounded"][MAINLINE_FAMILY]
    admitted_total = round(mainline["admitted_per_query"]["mean"] * bounded_map["queries"])
    assert roles["STRUCTURAL_SCORED_CANDIDATE"] == admitted_total


# --- the actual scientific claim: bounded is a genuine restriction of full ---


def test_u3_bounded_is_a_subset_of_what_full_reexpansion_would_have_produced(tmp_path):
    """U3_bounded = U2 union A64 is provably a subset of TARGET_H1(Cq_struct):

    U2 = TARGET_H1(Cq) subset TARGET_H1(Cq_struct) because Cq subset Cq_struct
    and TARGET_H1 is monotone in its pool argument; A64 subset Cq_struct
    subset TARGET_H1(Cq_struct) because TARGET_H1(X) always contains X. This
    reconstructs U3_full directly (bypassing the runner, which deliberately
    never computes it) to check that mathematical claim against the same
    admissions the runner itself derives, not just assert it in prose.
    """

    args = _args(tmp_path)
    dataset = load_complete_dataset(args.data, dataset=args.dataset, require_embeddings=False)
    queries = dataset.split(QuerySplit.VALIDATION)[: args.queries]
    views = [QueryView(query) for query in queries]
    num_nodes = int(dataset.num_nodes)
    rowptr = dataset.rowptr.numpy().astype(np.int64, copy=False)
    col = dataset.col.numpy().astype(np.int64, copy=False)
    operators = build_operators(rowptr, col, num_nodes)

    families = _families(args)
    family_rowptr, family_col = _load_family_csr(families[MAINLINE_FAMILY], num_nodes)
    family_rowptr, family_col, _symmetric = _undirected(family_rowptr, family_col, num_nodes)
    budget = _a64_budget(
        per_seed_cap=args.per_seed_cap, neighbour_scan_cap_per_seed=args.neighbour_scan_cap_per_seed
    )

    strict_reduction_seen = False
    for view in views:
        u2 = context_nodes(CONTEXT_ARM, operators=operators, pool=view.pool, seeds=view.seeds)
        expansion = expand(
            STRUCTURAL,
            rowptr=family_rowptr,
            col=family_col,
            node_embeddings=dataset.node_array,
            query_embedding=None,
            anchor=view.anchor,
            pool=view.pool,
            seeds=view.seeds,
            budget=budget,
            num_nodes=num_nodes,
        )
        cq_struct = expansion.additive_pool
        a64 = expansion.admitted
        u3_bounded = np.union1d(u2, a64)
        u3_full = context_nodes(CONTEXT_ARM, operators=operators, pool=cq_struct, seeds=view.seeds)

        assert np.isin(u3_bounded, u3_full).all()
        assert u3_bounded.size <= u3_full.size
        if u3_bounded.size < u3_full.size:
            strict_reduction_seen = True

    assert strict_reduction_seen, (
        "the toy fixture never exercised a real reduction -- this fixture "
        "would silently pass even if U3_bounded had accidentally been wired "
        "to recompute TARGET_H1(Cq_struct)"
    )


# --- determinism and refusals ---


def test_the_run_repeats_exactly(tmp_path):
    first = run(_args(tmp_path / "a"))
    second = run(_args(tmp_path / "b"))
    for payload in (first, second):
        payload.pop("systems")
        payload["r1"].pop("feature_latency_ms")
        payload["r2"].pop("build_latency_ms")
        payload["r2"].pop("feature_latency_ms")
        for row in payload["r3_bounded"].values():
            row.pop("admission_latency_ms")
            row.pop("context_build_latency_ms")
            row.pop("feature_latency_ms")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_a_short_split_is_refused_rather_than_quietly_shrunk(tmp_path):
    with pytest.raises(ValueError, match="needed 9"):
        run(_args(tmp_path, queries=9))


def test_the_map_without_a_family_graph_is_refused(tmp_path):
    with pytest.raises(ValueError, match="at least one edge-provenance family"):
        run(_args(tmp_path, edge_families=[]))


def test_an_undeclared_mainline_family_is_refused(tmp_path):
    with pytest.raises(ValueError, match="is not among"):
        run(_args(tmp_path, a64_mainline_family="not_a_real_family"))


def test_the_frozen_candidate_pool_is_proved_before_it_is_measured(bounded_map):
    proof = bounded_map["candidate_contract"]
    assert proof["status"] == "BIT_EXACT_FROZEN_CANDIDATE_EQUIVALENCE"
    assert proof["expected_contract_sha256"] == proof["observed_contract_sha256"]


def test_a_drifted_candidate_pool_stops_the_run(tmp_path):
    args = _args(tmp_path)
    args.baseline = {"candidate_contract_sha256": "f" * 64}
    with pytest.raises(ValueError, match="candidate contract does not match"):
        run(args)


def test_a_manifest_of_the_wrong_size_stops_the_run(tmp_path):
    args = _args(tmp_path)
    args.expected_queries = 99
    with pytest.raises(ValueError, match="differs from the registered protocol"):
        run(args)


def test_a_completed_run_is_not_repeated(tmp_path):
    args = _args(tmp_path)
    first = run(args)
    args.expected_queries = 99  # would raise if the run actually happened again
    second = run(args)
    assert first == second
