#!/usr/bin/env python
"""Collect per-query and per-dataset statistics for an L2 artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.data import QuerySplit  # noqa: E402
from mp_retrieval.l2_data import L2CandidateDataset  # noqa: E402
from mp_retrieval.l2_features import build_candidate_features, present_only_stats  # noqa: E402
from mp_retrieval.l2_interventions import apply_intervention  # noqa: E402
from mp_retrieval.l2_stats import aggregate_query_statistics, candidate_graph_statistics  # noqa: E402
from mp_retrieval.protocol import sha256_file  # noqa: E402


def _pilot_resplit(artifact: L2CandidateDataset, seed: int) -> None:
    order = list(range(len(artifact.queries)))
    random.Random(seed).shuffle(order)
    train_end = int(0.7 * len(order))
    validation_end = int(0.85 * len(order))
    for position, query_idx in enumerate(order):
        artifact.queries[query_idx].split = int(
            QuerySplit.TRAIN
            if position < train_end
            else QuerySplit.VALIDATION
            if position < validation_end
            else QuerySplit.TEST
        )


def collect(
    data: Path,
    output: Path,
    *,
    allow_pilot_resplit: bool,
    split_seed: int,
    perturbation: str = "clean",
    perturbation_rate: float = 0.0,
    perturbation_seed: int = 31415,
    remove_edge_types: set[int] | None = None,
) -> dict:
    artifact = L2CandidateDataset.load(data)
    train = [query for query in artifact.queries if query.split == int(QuerySplit.TRAIN)]
    if not train:
        if not allow_pilot_resplit:
            raise RuntimeError("Artifact has no training split; use --allow-pilot-resplit for pilot stats")
        _pilot_resplit(artifact, split_seed)
        train = [query for query in artifact.queries if query.split == int(QuerySplit.TRAIN)]
    mean, std = present_only_stats(train)
    features = {query: build_candidate_features(query, mean, std) for query in artifact.queries}
    edge_pairs = {query: artifact.induced_subgraph(query) for query in artifact.queries}
    edges = {query: pair[0] for query, pair in edge_pairs.items()}
    edge_types = {query: pair[1] for query, pair in edge_pairs.items()}
    features, edges, edge_types, intervention = apply_intervention(
        artifact.queries,
        features,
        edges,
        edge_types,
        kind=perturbation,
        rate=perturbation_rate,
        seed=perturbation_seed,
        removed_edge_types=remove_edge_types or set(),
    )
    rows = []
    for query_idx, query in enumerate(artifact.queries):
        rows.append(
            candidate_graph_statistics(
                query,
                features[query],
                edges[query],
                edge_types[query],
                seed=split_seed + query_idx,
            )
        )
    source_status = str(artifact.metadata.get("status", "unknown"))
    pilot_reasons = []
    if source_status != "canonical":
        pilot_reasons.append(f"source_artifact_status={source_status}")
    if allow_pilot_resplit:
        pilot_reasons.append("test_cache_was_resplit")
    result = {
        "status": "NOT_PAPER_VALID_PILOT" if pilot_reasons else "canonical",
        "pilot_reasons": pilot_reasons,
        "dataset": artifact.dataset,
        "data_sha256": sha256_file(data),
        "feature_basis": "shared standardized frozen L2 expert evidence",
        "typed_edge_policy": "unavailable is null; never encode an untyped graph as entropy zero",
        "intervention": intervention,
        "aggregate": aggregate_query_statistics(rows),
        "per_query": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-pilot-resplit", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--perturbation",
        choices=[
            "clean",
            "drop",
            "add_random",
            "degree_rewire",
            "hub_injection",
            "feature_mask",
            "feature_gaussian",
            "typed_edge_removal",
        ],
        default="clean",
    )
    parser.add_argument("--perturbation-rate", type=float, default=0.0)
    parser.add_argument("--perturbation-seed", type=int, default=31415)
    parser.add_argument("--remove-edge-types", nargs="*", type=int, default=[])
    args = parser.parse_args()
    result = collect(
        args.data,
        args.output,
        allow_pilot_resplit=args.allow_pilot_resplit,
        split_seed=args.split_seed,
        perturbation=args.perturbation,
        perturbation_rate=args.perturbation_rate,
        perturbation_seed=args.perturbation_seed,
        remove_edge_types=set(args.remove_edge_types),
    )
    print(
        json.dumps(
            {"status": result["status"], "dataset": result["dataset"], "aggregate": result["aggregate"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
