#!/usr/bin/env python
"""Run a paired candidate-MLP versus candidate-GNN L2 experiment."""

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
from mp_retrieval.l2_models import parameter_matched_scorers  # noqa: E402
from mp_retrieval.protocol import seed_everything, sha256_file  # noqa: E402
from mp_retrieval.representation import assert_message_passing_gradients  # noqa: E402
from mp_retrieval.retrieval import paired_bootstrap_delta  # noqa: E402


def _pilot_resplit(queries: list[CandidateQuery], seed: int) -> None:
    """Engineering-only split of an old test cache; never valid paper evidence."""

    order = list(range(len(queries)))
    random.Random(seed).shuffle(order)
    train_end = int(0.7 * len(order))
    val_end = int(0.85 * len(order))
    for position, query_idx in enumerate(order):
        split = QuerySplit.TRAIN if position < train_end else (
            QuerySplit.VALIDATION if position < val_end else QuerySplit.TEST
        )
        queries[query_idx].split = int(split)


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


def _evaluate(model, examples, features, edges, device, ks):
    model.eval()
    rows = []
    start = time.perf_counter()
    with torch.no_grad():
        for idx in examples:
            x = features[idx].to(device)
            e = edges[idx].to(device)
            rows.append(_metric_rows(model(x, e).cpu(), idx, ks))
    elapsed = time.perf_counter() - start
    aggregate = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]} if rows else {}
    aggregate["milliseconds_per_query"] = elapsed * 1000 / max(len(rows), 1)
    return aggregate, rows


def _train(model, train, validation, features, edges, device, epochs, lr, is_gnn):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state, best_value = None, -float("inf")
    gradient_checked = False
    for epoch in range(epochs):
        model.train()
        for query in sorted(train, key=lambda q: q.query_id):
            if query.relevant_local.numel() == 0:
                continue
            x = features[query].to(device)
            edge_index = edges[query].to(device)
            scores = model(x, edge_index)
            target = torch.zeros_like(scores)
            target[query.relevant_local.to(device)] = 1.0 / query.relevant_local.numel()
            loss = -(F.log_softmax(scores, dim=0) * target).sum()
            optimizer.zero_grad()
            loss.backward()
            if is_gnn and not gradient_checked:
                assert_message_passing_gradients(model)
                gradient_checked = True
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val, _ = _evaluate(model, validation, features, edges, device, (5,))
        value = val.get("recall@5", -float("inf"))
        if value > best_value:
            best_value = value
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best_state is None:
        raise RuntimeError("No trainable queries or validation examples were available")
    model.load_state_dict(best_state)
    return model


def run(args) -> dict:
    artifact = L2CandidateDataset.load(args.data)
    if not any(query.split == int(QuerySplit.TRAIN) for query in artifact.queries):
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
    edges = {query: artifact.induced_subgraph(query)[0] for query in artifact.queries}
    input_dim = next(iter(features.values())).shape[1]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    results = {"mlp": [], args.gnn: [], "per_query": {"mlp": [], args.gnn: []}}
    parameter_report = None
    for seed in args.seeds:
        seed_everything(seed)
        mlp, gnn, parameter_report = parameter_matched_scorers(
            args.gnn, input_dim, args.hidden_dim, num_layers=args.layers, dropout=args.dropout
        )
        for name, model in (("mlp", mlp), (args.gnn, gnn)):
            seed_everything(seed)
            trained = _train(
                model,
                splits[QuerySplit.TRAIN],
                splits[QuerySplit.VALIDATION],
                features,
                edges,
                device,
                args.epochs,
                args.lr,
                name != "mlp",
            )
            aggregate, rows = _evaluate(
                trained,
                splits[QuerySplit.TEST],
                features,
                edges,
                device,
                tuple(args.ks),
            )
            results[name].append(aggregate)
            results["per_query"][name].append(rows)
    paired = {}
    comparable_metrics = [
        key
        for key in results["mlp"][0]
        if key not in {"milliseconds_per_query", "candidate_ceiling"}
    ]
    for metric in comparable_metrics:
        mlp_by_query = np.mean(
            [[row[metric] for row in seed_rows] for seed_rows in results["per_query"]["mlp"]],
            axis=0,
        )
        gnn_by_query = np.mean(
            [[row[metric] for row in seed_rows] for seed_rows in results["per_query"][args.gnn]],
            axis=0,
        )
        paired[metric] = paired_bootstrap_delta(
            gnn_by_query,
            mlp_by_query,
            seed=args.split_seed,
        )
    summary = {
        "status": "NOT_PAPER_VALID_PILOT" if args.allow_pilot_resplit else "canonical",
        "dataset": artifact.dataset,
        "data_sha256": sha256_file(args.data),
        "splits": {split.name.lower(): len(rows) for split, rows in splits.items()},
        "config": vars(args) | {"data": str(args.data), "output": str(args.output)},
        "parameters": parameter_report,
        "paired_gnn_minus_mlp": paired,
        "aggregate": {
            model: {
                metric: {
                    "mean": float(np.mean([run[metric] for run in runs])),
                    "std": float(np.std([run[metric] for run in runs])),
                }
                for metric in runs[0]
            }
            for model, runs in results.items()
            if model != "per_query"
        },
        "runs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gnn", choices=["gcn", "sage", "gat", "gin"], default="gcn")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20, 50])
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-pilot-resplit", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({"status": summary["status"], "aggregate": summary["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
