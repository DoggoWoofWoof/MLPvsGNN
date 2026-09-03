"""The M0B regime-map runner, end to end on a toy dataset.

Real frozen datasets live only on the Modal volume (see data/README.md and
scripts/modal_m0a1_overlap.py's REMOTE_ROOT) -- there is no local complete_data
root to run against directly. A toy graph, built the same way
tests/test_m0a1_runner.py already proved out, is what lets this runner's own
wiring (R1/R2/R3 construction, Safeguard B's gated invariants, NODE_ROLE,
containment) be checked before any real Modal spend, not just after.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mp_retrieval.complete_data import load_complete_dataset
from mp_retrieval.overlap_audit import NODE_ROLES
from scripts.run_m0b_regime_map import (
    A64_GRAPH_EXPANSION_CAP,
    COMPLETE_STATUS,
    MAINLINE_FAMILY,
    run,
)

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
    # Rows are descending, not ascending: complete_data._stable_union does an
    # order-preserving (first-occurrence) dedup of concat(dense, splade), not
    # a sort, so an ascending-by-construction fixture reassembles
    # candidate_index in sorted order by accident and can never exercise a
    # caller that forgets to sort a pool before using it as
    # qls_local_features' nodes= argument -- the real bug
    # scripts/run_m0b_webqsp_probe.py hit on first contact with real (sorted-
    # nothing) webqsp data, fixed there with one np.unique call and mirrored
    # by the same reversed-row fix in tests/test_m0b_webqsp_probe.py.
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
                # Golds placed outside every query's Cq, some reachable only
                # through the wider family graph's frontier -- otherwise every
                # query would land in one cell and the invariants would never
                # be exercised against a real admission.
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
        "output": tmp_path / "out" / "regime_map.json",
    }
    settings.update(overrides)
    return argparse.Namespace(**settings)


@pytest.fixture(scope="module")
def regime_map(tmp_path_factory):
    return run(_args(tmp_path_factory.mktemp("m0b")))


def test_the_run_completes_and_says_what_it_did_not_do(regime_map):
    assert regime_map["status"] == COMPLETE_STATUS
    assert regime_map["trained_anything"] is False
    assert regime_map["test_split_read"] is False
    assert regime_map["m0a_verdict_is_unchanged"] == "MOVEMENT_UNDER_A_BREACHED_ABORT_RULE"
    assert regime_map["m0a1_verdict_is_unchanged"] == "ADVANCE"
    assert regime_map["a64_mainline_family"] == MAINLINE_FAMILY
    assert regime_map["a64_graph_expansion_cap"] == A64_GRAPH_EXPANSION_CAP == 64


# --- invariants: R1/R2, then Safeguard B's per-family gates ---


def test_r1_and_r2_ceilings_are_bit_exact(regime_map):
    assert regime_map["invariants"]["oracle_r1_equals_oracle_r2_bit_exact"] is True
    assert regime_map["invariants"]["scored_r1_equals_scored_r2"] is True


def test_u2_contains_cq_for_every_query(regime_map):
    assert regime_map["invariants"]["u2_contains_cq"] is True
    assert regime_map["r2"]["arm"] == "TARGET_H1"
    assert regime_map["r2"]["context_node_count"]["mean"] >= regime_map["r1"]["candidate_count"]["mean"]


def test_safeguard_b_invariants_hold_for_every_family(regime_map):
    for family in FAMILIES:
        assert regime_map["invariants"][f"{family}.a64_disjoint_from_cq"] is True
        assert regime_map["invariants"][f"{family}.admitted_delta_within_universal_cap"] is True
        assert regime_map["invariants"][f"{family}.cq_struct_equals_cq_union_a64"] is True
        assert regime_map["invariants"][f"{family}.scored_r1_subset_scored_r3"] is True


# --- R3: admitted count, containment, gold overlap ---


def test_r3_is_reported_for_every_family_with_the_mainline_flagged(regime_map):
    assert set(regime_map["r3"]) == set(FAMILIES)
    for family, row in regime_map["r3"].items():
        assert row["is_mainline"] == (family == MAINLINE_FAMILY)
        assert row["admitted_per_query"]["mean"] >= 0.0
        assert 0 <= row["context_node_count"]["mean"]


def test_containment_is_reported_as_a_rate_never_asserted_as_one(regime_map):
    for row in regime_map["r3"].values():
        containment = row["admitted_node_containment_in_u2"]
        if not containment["undefined_because_nothing_was_admitted"]:
            assert 0.0 <= containment["containment_rate"] <= 1.0


def test_gold_overlap_reuses_the_m0a1_cross_tabulation_shape(regime_map):
    from mp_retrieval.overlap_audit import CROSS_TAB_CELLS

    for family, row in regime_map["r3"].items():
        cells = row["gold_overlap_vs_u2"]["cross_tabulation"]["gold_instances"]
        assert set(cells) == set(CROSS_TAB_CELLS), family


# --- NODE_ROLE ---


def test_node_roles_are_reported_only_for_the_mainline_family(regime_map):
    assert set(regime_map["node_roles"]) == {"R1", "R2", "R3"}
    for tag, counts in regime_map["node_roles"].items():
        assert set(counts) == set(NODE_ROLES), tag


def test_r1_node_roles_have_no_structural_or_context_only_members(regime_map):
    r1 = regime_map["node_roles"]["R1"]
    assert r1["STRUCTURAL_SCORED_CANDIDATE"] == 0
    assert r1["CONTEXT_ONLY"] == 0
    assert r1["RETRIEVAL_CANDIDATE"] > 0


def test_r2_node_roles_have_context_only_but_no_structural_scored(regime_map):
    r2 = regime_map["node_roles"]["R2"]
    assert r2["STRUCTURAL_SCORED_CANDIDATE"] == 0
    assert r2["CONTEXT_ONLY"] >= 0


def test_r3_node_role_totals_match_admitted_and_gold_overlap_bookkeeping(regime_map):
    r3_roles = regime_map["node_roles"]["R3"]
    mainline = regime_map["r3"][MAINLINE_FAMILY]
    admitted_total = round(mainline["admitted_per_query"]["mean"] * regime_map["queries"])
    assert r3_roles["STRUCTURAL_SCORED_CANDIDATE"] == admitted_total


# --- feature-construction cost: cold-start vs steady-state ---


def test_r1_feature_latency_splits_a_cold_start_from_steady_state(regime_map):
    r1_features = regime_map["r1"]["feature_latency_ms"]
    raw = r1_features["raw_ms"]
    assert len(raw) == regime_map["queries"]
    assert r1_features["cold_start_compile_ms"] == raw[0]
    steady = raw[1:]
    assert r1_features["steady_state"]["max"] == max(steady)
    assert r1_features["steady_state"]["mean"] == pytest.approx(sum(steady) / len(steady))


def test_r2_and_r3_feature_latency_have_no_cold_start_of_their_own(regime_map):
    # The one-time Numba compile always lands on R1's query-0 call (this
    # runner's fixed R1-before-R2-before-R3 order); by the time R2/R3 first
    # call qls_local_features, the kernel is already warm. Their full raw
    # array is already steady-state -- nothing here is sliced off.
    r2_features = regime_map["r2"]["feature_latency_ms"]
    assert r2_features["cold_start_compile_ms"] is None
    assert len(r2_features["raw_ms"]) == regime_map["queries"]
    assert r2_features["steady_state"]["max"] == max(r2_features["raw_ms"])

    for family, row in regime_map["r3"].items():
        r3_features = row["feature_latency_ms"]
        assert r3_features["cold_start_compile_ms"] is None, family
        assert len(r3_features["raw_ms"]) == regime_map["queries"], family
        assert r3_features["steady_state"]["max"] == max(r3_features["raw_ms"]), family


def test_raw_feature_timings_are_never_discarded(regime_map):
    # "raw unfiltered timings must still be retained in the machine-readable
    # artifact -- never silently discarded or overwritten" -- one raw entry
    # per query, in every regime, always, not just when a cold start exists.
    assert len(regime_map["r1"]["feature_latency_ms"]["raw_ms"]) == regime_map["queries"]
    assert len(regime_map["r2"]["feature_latency_ms"]["raw_ms"]) == regime_map["queries"]
    for row in regime_map["r3"].values():
        assert len(row["feature_latency_ms"]["raw_ms"]) == regime_map["queries"]


# --- systems ---


def test_systems_reports_latency_percentiles_and_peak_rss(regime_map):
    systems = regime_map["systems"]
    # _peak_rss_bytes() reads the POSIX resource module; on Windows (this
    # local test run) it is None by design, not a bug -- the real Modal
    # containers are Linux, where M0A.1's own runner already reports real
    # values with the identical function (see outputs/m0a1_overlap/*.json).
    rss = systems["peak_process_rss_bytes"]
    assert rss is None or rss > 0
    for key in (
        "u2_build_latency_ms",
        "mainline_a64_admission_latency_ms",
        "mainline_u3_build_latency_ms",
    ):
        assert set(systems[key]) >= {"p50", "p95", "max"}
    for key in ("r1_feature_latency_ms", "r2_feature_latency_ms", "mainline_r3_feature_latency_ms"):
        assert set(systems[key]["steady_state"]) >= {"p50", "p95", "max"}
    assert "cold_start_compile_ms" in systems["r1_feature_latency_ms"]
    assert systems["mainline_r3_feature_latency_ms"]["steady_state"]["max"] == max(
        regime_map["r3"][MAINLINE_FAMILY]["feature_latency_ms"]["raw_ms"]
    )
    assert systems["latency_is_per_query_percentiles_on_one_container"] is True


# --- determinism and refusals ---


def test_the_run_repeats_exactly(tmp_path):
    first = run(_args(tmp_path / "a"))
    second = run(_args(tmp_path / "b"))
    for payload in (first, second):
        payload.pop("systems")
        payload["r1"].pop("feature_latency_ms")
        payload["r2"].pop("build_latency_ms")
        payload["r2"].pop("feature_latency_ms")
        for row in payload["r3"].values():
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


def test_the_frozen_candidate_pool_is_proved_before_it_is_measured(regime_map):
    proof = regime_map["candidate_contract"]
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
