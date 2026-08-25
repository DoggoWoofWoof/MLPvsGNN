#!/usr/bin/env python
"""Run audited paired candidate-MLP versus candidate-GNN L2 experiments."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.data import QuerySplit  # noqa: E402
from mp_retrieval.l2_data import CandidateQuery, L2CandidateDataset  # noqa: E402
from mp_retrieval.l2_features import build_candidate_features, present_only_stats  # noqa: E402
from mp_retrieval.l2_interventions import apply_intervention  # noqa: E402
from mp_retrieval.l2_models import (  # noqa: E402
    build_gnn_scorer,
    build_mlp_scorer,
    compute_matched_width,
    count_parameters,
    parameter_matched_width,
)
from mp_retrieval.l2_protocol import comparison_contract  # noqa: E402
from mp_retrieval.protocol import seed_everything, sha256_file  # noqa: E402
from mp_retrieval.representation import assert_message_passing_gradients  # noqa: E402
from mp_retrieval.retrieval import paired_bootstrap_delta  # noqa: E402


def _pilot_subset(queries: list[CandidateQuery], limit: int | None, seed: int) -> list[CandidateQuery]:
    if limit is None or limit >= len(queries):
        return queries
    if limit < 10:
        raise ValueError("--max-queries must be at least 10 to retain train/validation/test splits")
    order = list(range(len(queries)))
    random.Random(seed).shuffle(order)
    selected = set(order[:limit])
    return [query for idx, query in enumerate(queries) if idx in selected]


def _pilot_resplit(queries: list[CandidateQuery], seed: int) -> None:
    """Engineering-only split of an old test cache; never valid paper evidence."""

    order = list(range(len(queries)))
    random.Random(seed).shuffle(order)
    train_end = int(0.7 * len(order))
    validation_end = int(0.85 * len(order))
    for position, query_idx in enumerate(order):
        queries[query_idx].split = int(
            QuerySplit.TRAIN
            if position < train_end
            else QuerySplit.VALIDATION
            if position < validation_end
            else QuerySplit.TEST
        )


def _metric_rows(scores: torch.Tensor, query: CandidateQuery, ks: tuple[int, ...]) -> dict[str, float]:
    order = torch.argsort(scores, descending=True).cpu()
    relevant = set(query.relevant_local.tolist())
    total_global = max(int(query.relevant_global.unique().numel()), 1)
    row: dict[str, float] = {"candidate_ceiling": query.candidate_ceiling}
    first = next((rank + 1 for rank, idx in enumerate(order.tolist()) if idx in relevant), None)
    row["mrr"] = 0.0 if first is None else 1.0 / first
    for k in ks:
        hits = len(relevant & set(order[:k].tolist()))
        row[f"recall@{k}"] = hits / total_global
        row[f"conditional_recall@{k}"] = hits / max(len(relevant), 1)
        row[f"full_coverage@{k}"] = float(hits == total_global)
    return row


def _listwise_loss(scores: torch.Tensor, relevant_local: torch.Tensor) -> torch.Tensor:
    target = torch.zeros_like(scores)
    positives = relevant_local.to(scores.device)
    target[positives] = 1.0 / positives.numel()
    return -(F.log_softmax(scores, dim=0) * target).sum()


def _evaluate(model, examples, features, edges, device, ks, *, uses_topology):
    model.eval()
    rows = []
    start = time.perf_counter()
    with torch.no_grad():
        for query in examples:
            x = features[query].to(device)
            edge_index = edges[query].to(device) if uses_topology else None
            rows.append(_metric_rows(model(x, edge_index).cpu(), query, ks))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    aggregate = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]} if rows else {}
    aggregate["milliseconds_per_query"] = elapsed * 1000 / max(len(rows), 1)
    return aggregate, rows


def _train(model, train, validation, features, edges, device, epochs, lr, *, uses_topology):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state, best_value = None, -float("inf")
    gradient_checked = False
    started = time.perf_counter()
    for _epoch in range(epochs):
        model.train()
        for query in sorted(train, key=lambda item: item.query_id):
            if query.relevant_local.numel() == 0:
                continue
            x = features[query].to(device)
            edge_index = edges[query].to(device) if uses_topology else None
            scores = model(x, edge_index)
            loss = _listwise_loss(scores, query.relevant_local)
            optimizer.zero_grad()
            loss.backward()
            if uses_topology and not gradient_checked:
                assert_message_passing_gradients(model)
                gradient_checked = True
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation_metrics, _ = _evaluate(
            model,
            validation,
            features,
            edges,
            device,
            (5,),
            uses_topology=uses_topology,
        )
        value = validation_metrics.get("recall@5", -float("inf"))
        if value > best_value:
            best_value = value
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("No trainable queries or validation examples were available")
    model.load_state_dict(best_state)
    return model, {"training_seconds": train_seconds, "best_validation_recall@5": best_value}


def _calibration_queries(
    train: list[CandidateQuery],
    count: int,
    features: dict[CandidateQuery, torch.Tensor],
    edges: dict[CandidateQuery, torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    eligible = sorted(
        [query for query in train if query.relevant_local.numel()],
        key=lambda query: (query.candidate_index.numel(), query.query_id),
    )
    if not eligible:
        raise RuntimeError("No positive training queries are available for compute calibration")
    positions = np.linspace(0, len(eligible) - 1, min(count, len(eligible)), dtype=int)
    return [
        (features[eligible[pos]], edges[eligible[pos]], eligible[pos].relevant_local)
        for pos in positions
    ]


def _paired_results(
    gnn_rows: list[list[dict[str, float]]],
    mlp_rows: list[list[dict[str, float]]],
    *,
    split_seed: int,
) -> dict[str, object]:
    metrics = [key for key in mlp_rows[0][0] if key != "candidate_ceiling"]
    paired: dict[str, object] = {}
    for metric in metrics:
        mlp_by_query = np.mean(
            [[row[metric] for row in seed_rows] for seed_rows in mlp_rows], axis=0
        )
        gnn_by_query = np.mean(
            [[row[metric] for row in seed_rows] for seed_rows in gnn_rows], axis=0
        )
        paired[metric] = paired_bootstrap_delta(
            gnn_by_query, mlp_by_query, seed=split_seed
        )
    return paired


def run(args) -> dict:
    artifact = L2CandidateDataset.load(args.data)
    artifact.queries = _pilot_subset(artifact.queries, args.max_queries, args.split_seed)
    has_canonical_train = any(query.split == int(QuerySplit.TRAIN) for query in artifact.queries)
    if not has_canonical_train:
        if not args.allow_pilot_resplit:
            raise RuntimeError(
                "Artifact has no training split. Re-export canonical splits or pass --allow-pilot-resplit "
                "for an explicitly non-paper-valid engineering run."
            )
        _pilot_resplit(artifact.queries, args.split_seed)
    splits = {
        split: [query for query in artifact.queries if query.split == int(split)]
        for split in (QuerySplit.TRAIN, QuerySplit.VALIDATION, QuerySplit.TEST)
    }
    mean, std = present_only_stats(splits[QuerySplit.TRAIN])
    features = {query: build_candidate_features(query, mean, std) for query in artifact.queries}
    edge_pairs = {query: artifact.induced_subgraph(query) for query in artifact.queries}
    edges = {query: pair[0] for query, pair in edge_pairs.items()}
    edge_types = {query: pair[1] for query, pair in edge_pairs.items()}
    features, edges, edge_types, intervention = apply_intervention(
        artifact.queries,
        features,
        edges,
        edge_types,
        kind=args.perturbation,
        rate=args.perturbation_rate,
        seed=args.perturbation_seed,
        removed_edge_types=set(args.remove_edge_types),
    )
    input_dim = next(iter(features.values())).shape[1]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    match_reports: dict[str, object] = {}
    widths: dict[str, int] = {}
    seed_everything(args.compute_seed)
    if "parameter" in args.match_modes:
        width, report = parameter_matched_width(
            args.gnn,
            input_dim,
            args.hidden_dim,
            num_layers=args.layers,
            dropout=args.dropout,
        )
        if float(report["relative_parameter_gap"]) > args.parameter_tolerance:
            raise RuntimeError(f"Parameter match exceeds tolerance: {report}")
        widths["parameter_matched_mlp"] = width
        match_reports["parameter"] = report
    if "compute" in args.match_modes:
        calibration = _calibration_queries(
            splits[QuerySplit.TRAIN], args.compute_calibration_queries, features, edges
        )
        width, report = compute_matched_width(
            args.gnn,
            input_dim,
            args.hidden_dim,
            calibration,
            device=device,
            num_layers=args.layers,
            dropout=args.dropout,
            widths=args.compute_widths,
            warmups=args.compute_warmups,
            repeats=args.compute_repeats,
        )
        if float(report["relative_compute_gap"]) > args.compute_tolerance:
            raise RuntimeError(
                "Compute match exceeds tolerance. Expand --compute-widths or relax only with a "
                f"pre-registered justification: {report}"
            )
        widths["compute_matched_mlp"] = width
        match_reports["compute"] = report
    contract = comparison_contract(
        artifact.queries, features, edges, seeds=args.seeds
    )
    model_names = list(widths) + [args.gnn]
    results = {name: [] for name in model_names}
    per_query = {name: [] for name in model_names}
    telemetry = {name: [] for name in model_names}
    for seed in args.seeds:
        factories = {
            name: (
                lambda width=width: build_mlp_scorer(
                    input_dim,
                    width,
                    num_layers=args.layers,
                    dropout=args.dropout,
                )
            )
            for name, width in widths.items()
        }
        factories[args.gnn] = lambda: build_gnn_scorer(
            args.gnn,
            input_dim,
            args.hidden_dim,
            num_layers=args.layers,
            dropout=args.dropout,
        )
        for name in model_names:
            seed_everything(seed)
            model = factories[name]()
            uses_topology = name == args.gnn
            trained, train_telemetry = _train(
                model,
                splits[QuerySplit.TRAIN],
                splits[QuerySplit.VALIDATION],
                features,
                edges,
                device,
                args.epochs,
                args.lr,
                uses_topology=uses_topology,
            )
            aggregate, rows = _evaluate(
                trained,
                splits[QuerySplit.TEST],
                features,
                edges,
                device,
                tuple(args.ks),
                uses_topology=uses_topology,
            )
            results[name].append(aggregate)
            per_query[name].append(rows)
            telemetry[name].append(train_telemetry | {"parameters": count_parameters(model)})
    paired = {
        match_mode: _paired_results(
            per_query[args.gnn],
            per_query[f"{match_mode}_matched_mlp"],
            split_seed=args.split_seed,
        )
        for match_mode in args.match_modes
    }
    artifact_status = str(artifact.metadata.get("status", "unknown"))
    pilot_reasons = []
    if artifact_status != "canonical":
        pilot_reasons.append(f"source_artifact_status={artifact_status}")
    if args.allow_pilot_resplit:
        pilot_reasons.append("test_cache_was_resplit")
    if args.max_queries is not None:
        pilot_reasons.append(f"query_limit={args.max_queries}")
    status = "NOT_PAPER_VALID_PILOT" if pilot_reasons else "canonical"
    config = vars(args).copy()
    config.update({"data": str(args.data), "output": str(args.output), "device": str(device)})
    summary = {
        "status": status,
        "pilot_reasons": pilot_reasons,
        "dataset": artifact.dataset,
        "source_artifact_status": artifact_status,
        "data_sha256": sha256_file(args.data),
        "splits": {split.name.lower(): len(rows) for split, rows in splits.items()},
        "config": config,
        "comparison_contract": contract,
        "matching": match_reports,
        "intervention": intervention,
        "paired_gnn_minus_mlp": paired,
        "aggregate": {
            model: {
                metric: {
                    "mean": float(np.mean([run_metrics[metric] for run_metrics in runs])),
                    "std": float(np.std([run_metrics[metric] for run_metrics in runs])),
                }
                for metric in runs[0]
            }
            for model, runs in results.items()
        },
        "training_telemetry": telemetry,
        "runs": {"aggregate": results, "per_query": per_query},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gnn", choices=["gcn", "sage", "gat", "gin"], default="gcn")
    parser.add_argument(
        "--match-modes",
        nargs="+",
        choices=["parameter", "compute"],
        default=["parameter", "compute"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20, 50])
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-pilot-resplit", action="store_true")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--parameter-tolerance", type=float, default=0.05)
    parser.add_argument("--compute-tolerance", type=float, default=0.10)
    parser.add_argument("--compute-seed", type=int, default=2718)
    parser.add_argument("--compute-calibration-queries", type=int, default=3)
    parser.add_argument(
        "--compute-widths",
        nargs="+",
        type=int,
        default=list(range(4, 513)),
    )
    parser.add_argument("--compute-warmups", type=int, default=1)
    parser.add_argument("--compute-repeats", type=int, default=3)
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
    if len(set(args.match_modes)) != len(args.match_modes):
        parser.error("--match-modes cannot contain duplicates")
    summary = run(args)
    print(json.dumps({"status": summary["status"], "aggregate": summary["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
