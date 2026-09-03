"""The M0B Safeguard C webqsp probe, end to end on a toy dataset.

Same technique tests/test_m0b_runner.py already proved out for the sibling
regime-map runner: a synthetic complete_data root, built locally, lets the
probe's own wiring (R1/R2/R3 feature+diagnostic timing, A64 admission, the
10-20 query bound, Safeguard B's gates) be checked before any real Modal
spend on webqsp.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mp_retrieval.complete_data import load_complete_dataset
from scripts.run_m0b_webqsp_probe import (
    A64_GRAPH_EXPANSION_CAP,
    COMPLETE_STATUS,
    MAINLINE_FAMILY,
    MAX_QUERIES,
    MIN_QUERIES,
    REGIMES,
    run,
)

NUM_NODES = 300
NUM_QUERIES = 25
NUM_TRAIN = 5
CANDIDATES_PER_QUERY = 6


def _write_dataset(root: Path) -> None:
    rng = np.random.default_rng(20260903)
    nodes = rng.normal(size=(NUM_NODES, 6)).astype(np.float32)
    np.save(root / "nodes.npy", nodes)
    np.save(root / "queries_all.npy", rng.normal(size=(NUM_QUERIES, 6)).astype(np.float32))

    # Descending within each row, not ascending: complete_data._stable_union
    # dedupes dense+splade by first occurrence, not by sorting, so an
    # ascending row (tried first) reassembles into a fully sorted
    # candidate_index by construction and silently hides any bug that only
    # shows up on an unsorted pool -- which is exactly what real webqsp
    # candidate order is (retrieval-rank, not node-id order) and exactly
    # what broke run_m0b_webqsp_probe.py's R1 arm the first time it ran on
    # real data (qls_local_features requires its `nodes=` argument
    # pre-sorted and does not sort it itself). Same value set per query as
    # the ascending version, so the disjointness-from-golds range below is
    # unaffected -- only the order changes.
    dense = np.array(
        [
            [
                (CANDIDATES_PER_QUERY * i + j) % NUM_NODES
                for j in reversed(range(CANDIDATES_PER_QUERY))
            ]
            for i in range(NUM_QUERIES)
        ]
    )
    splade = (dense + 2) % NUM_NODES
    np.save(root / "dense_top200_all.npy", dense)
    np.save(root / "splade_top200_all.npy", splade)

    # Gold ids drawn from the node range candidate pools never touch (pools
    # only ever cover 0..NUM_QUERIES*CANDIDATES_PER_QUERY-1) -- this probe
    # never reads gold content, only that the dataset loads validly.
    golds = [[f"doc_{(200 + i) % NUM_NODES}"] for i in range(NUM_QUERIES)]
    (root / "query_ids_all.json").write_text(
        json.dumps(
            {
                "ids": [f"q{i}" for i in range(NUM_QUERIES)],
                "golds": golds,
                "split_indices": {
                    "train": list(range(NUM_TRAIN)),
                    "val": list(range(NUM_TRAIN, NUM_QUERIES)),
                    "test": [],
                },
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
    source = np.arange(NUM_NODES, dtype=np.int64)
    target = (source + 4) % NUM_NODES
    edge_index = np.concatenate([np.stack([source, target]), np.stack([target, source])], axis=1)
    path = graph_root / MAINLINE_FAMILY
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"edge_index": torch.tensor(edge_index, dtype=torch.long), "num_nodes": NUM_NODES},
        path / "graph.pt",
    )


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    graphs = tmp_path / "families"
    _write_dataset(data)
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
        "queries": 15,
        "per_seed_cap": 4,
        "neighbour_scan_cap_per_seed": 4096,
        "edge_provenance_root": graphs,
        "edge_families": [MAINLINE_FAMILY],
        "a64_mainline_family": MAINLINE_FAMILY,
        "output": tmp_path / "out" / "probe.json",
    }
    settings.update(overrides)
    return argparse.Namespace(**settings)


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    return run(_args(tmp_path_factory.mktemp("m0b_probe")))


def test_the_probe_completes_and_says_what_it_did_not_do(probe):
    assert probe["status"] == COMPLETE_STATUS
    assert probe["safeguard"] == "C"
    assert probe["trained_anything"] is False
    assert probe["test_split_read"] is False
    assert probe["no_meaningful_retrieval_statistics_by_design"] is True
    assert probe["a64_graph_expansion_cap"] == A64_GRAPH_EXPANSION_CAP == 64


def test_queries_outside_the_10_20_bound_are_refused(tmp_path):
    with pytest.raises(ValueError, match=f"{MIN_QUERIES}-{MAX_QUERIES}"):
        run(_args(tmp_path / "too_few", queries=MIN_QUERIES - 1))
    with pytest.raises(ValueError, match=f"{MIN_QUERIES}-{MAX_QUERIES}"):
        run(_args(tmp_path / "too_many", queries=MAX_QUERIES + 1))


def test_safeguard_b_invariants_all_hold(probe):
    for name in (
        "a64_disjoint_from_cq",
        "admitted_delta_within_universal_cap",
        "cq_struct_equals_cq_union_a64",
        "scored_r1_subset_scored_r3",
    ):
        assert probe["invariants"][name] is True


def test_all_three_regimes_are_reported_with_full_timing_shape(probe):
    assert set(probe["regimes"]) == set(REGIMES) == {"R1", "R2", "R3"}
    for tag, row in probe["regimes"].items():
        # _percentiles() (latency) reports p50; _counts() (sizes) reports
        # median instead -- same underlying stat, different helper, not a
        # typo. See scripts/run_m0a_probe.py's two functions.
        for key in (
            "build_latency_ms",
            "feature_latency_ms",
            "seed_distance_diagnostic_latency_ms",
            "total_construction_plus_feature_latency_ms",
        ):
            assert set(row[key]) >= {"p50", "p95", "max"}, (tag, key)
        for key in ("context_node_count", "context_edge_count"):
            assert set(row[key]) >= {"median", "p95", "max"}, (tag, key)


def test_seed_distance_is_labelled_a_diagnostic_not_a_serving_cost(probe):
    for row in probe["regimes"].values():
        assert row["seed_distance_is_a_measurement_instrument_never_paid_at_serving_time"] is True


def test_r2_context_is_never_smaller_than_r1_context(probe):
    # U2 = TARGET_H1(Cq) always contains Cq, so its node count can only be
    # greater or equal, in mean, to R1's (context == pool exactly for R1).
    r1_mean = probe["regimes"]["R1"]["context_node_count"]["mean"]
    r2_mean = probe["regimes"]["R2"]["context_node_count"]["mean"]
    assert r2_mean >= r1_mean


def test_a64_admission_and_containment_are_reported(probe):
    assert probe["admitted_per_query"]["mean"] >= 0.0
    assert set(probe["admission_latency_ms"]) >= {"p50", "p95", "max"}
    containment = probe["admitted_node_containment_in_u2"]
    if not containment["undefined_because_nothing_was_admitted"]:
        assert 0.0 <= containment["containment_rate"] <= 1.0


def test_systems_reports_the_declared_workspace_bound_and_platform_aware_rss(probe):
    systems = probe["systems"]
    assert systems["temporary_workspace_bytes"] == 8 * A64_GRAPH_EXPANSION_CAP * 4 == 2048
    assert systems["temporary_workspace_is_a_declared_bound_not_a_measurement"] is True
    # _peak_rss_bytes() is POSIX-only; None on this local Windows run is
    # expected, not a bug -- see tests/test_m0b_runner.py's identical note.
    rss = systems["peak_process_rss_bytes"]
    assert rss is None or rss > 0
    assert systems["latency_is_per_query_percentiles_on_one_container"] is True


def test_reused_prior_measurement_cites_the_graph_context_pilot(probe):
    reused = probe["reused_prior_measurement"]
    assert reused["source"] == "docs/GRAPH_CONTEXT_PILOT_RESULTS.md"
    assert reused["queries_measured"] == 300
    assert "webqsp_r2_target_h1_total_p95_ms" in reused


def test_the_probe_repeats_exactly(tmp_path):
    first = run(_args(tmp_path / "a"))
    second = run(_args(tmp_path / "b"))
    for payload in (first, second):
        payload.pop("systems")
        payload.pop("admission_latency_ms")
        for row in payload["regimes"].values():
            row.pop("build_latency_ms")
            row.pop("feature_latency_ms")
            row.pop("seed_distance_diagnostic_latency_ms")
            row.pop("total_construction_plus_feature_latency_ms")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_a_completed_run_is_not_repeated(tmp_path):
    args = _args(tmp_path)
    first = run(args)
    args.expected_queries = 99  # would raise if the run actually happened again
    second = run(args)
    assert first == second


def test_a_drifted_candidate_pool_stops_the_run(tmp_path):
    args = _args(tmp_path)
    args.baseline = {"candidate_contract_sha256": "f" * 64}
    with pytest.raises(ValueError, match="candidate contract does not match"):
        run(args)


def test_a_manifest_of_the_wrong_size_stops_the_run(tmp_path):
    args = _args(tmp_path)
    args.expected_queries = 12345
    with pytest.raises(ValueError, match="differs from the registered protocol"):
        run(args)


def test_the_map_without_a_family_graph_is_refused(tmp_path):
    with pytest.raises(ValueError, match="is not among"):
        run(_args(tmp_path, edge_families=[]))


def test_an_undeclared_mainline_family_is_refused(tmp_path):
    with pytest.raises(ValueError, match="is not among"):
        run(_args(tmp_path, a64_mainline_family="not_a_real_family"))
