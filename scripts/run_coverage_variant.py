#!/usr/bin/env python
"""Run the single preregistered K-direction set-coverage Offset variant."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import time
from typing import Any
import warnings

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.complete_data import CompleteQuery, load_complete_dataset  # noqa: E402
from mp_retrieval.data import QuerySplit  # noqa: E402
from mp_retrieval.operator_models import (  # noqa: E402
    COVERAGE_OFFSET_MODEL,
    SetCoverageOffsetMLP,
    build_operator_model,
    model_parameter_counts,
)
from mp_retrieval.protocol import seed_everything, sha256_file  # noqa: E402
from mp_retrieval.set_coverage import set_coverage_loss  # noqa: E402
from scripts.run_confirmation import _aggregate_runs, _repeated_score  # noqa: E402
from scripts.run_operator_screen import (  # noqa: E402
    _hash_tensor_contract,
    _prepare_batch,
    _score_queries,
    _synchronize,
)


def _train_set_model(
    model: SetCoverageOffsetMLP,
    train_queries: list[CompleteQuery],
    validation_queries: list[CompleteQuery],
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    diversity_weight: float,
    diversity_cosine_margin: float,
    seed: int,
) -> tuple[SetCoverageOffsetMLP, dict[str, Any]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        learning_rate,
        weight_decay=weight_decay,
    )
    eligible = [query for query in train_queries if query.relevant_local.numel()]
    max_positives = max(int(query.relevant_local.numel()) for query in eligible)
    if max_positives > model.directions:
        raise ValueError(
            f"Observed {max_positives} in-pool positives but model has {model.directions} directions"
        )
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = -float("inf")
    history: list[dict[str, float]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    empty_topologies: dict[CompleteQuery, torch.Tensor] = {}
    for epoch in range(epochs):
        order = eligible[:]
        random.Random(seed + epoch * 1_000_003).shuffle(order)
        model.train()
        losses: list[float] = []
        assignment_losses: list[float] = []
        diversity_losses: list[float] = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            nodes, queries, anchors, batch_index, _edges, lengths = _prepare_batch(
                batch,
                node_embeddings,
                query_embeddings,
                empty_topologies,
                device,
                uses_topology=False,
            )
            directional_scores, targets = model.directional_scores(
                nodes,
                queries,
                anchors,
                batch_index,
            )
            loss, components = set_coverage_loss(
                directional_scores,
                targets,
                batch,
                lengths,
                diversity_weight=diversity_weight,
                diversity_cosine_margin=diversity_cosine_margin,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            assignment_losses.append(float(components["assignment"].detach()))
            diversity_losses.append(float(components["diversity"].detach()))
        validation, _rows, _telemetry = _score_queries(
            model,
            validation_queries,
            node_embeddings,
            query_embeddings,
            empty_topologies,
            device,
            batch_size=batch_size,
            ks=(5,),
            timed=False,
        )
        value = float(validation["recall@5"])
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "assignment_loss": float(np.mean(assignment_losses)),
                "diversity_loss": float(np.mean(diversity_losses)),
                "validation_recall@5": value,
            }
        )
        if value > best_validation:
            best_validation = value
            best_state = deepcopy(
                {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
            )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("No coverage-variant checkpoint was selected")
    model.load_state_dict(best_state)
    return model, {
        "training_seconds": elapsed,
        "peak_training_gpu_memory_mb_total": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
        ),
        "best_validation_recall@5": best_validation,
        "history": history,
        "train_queries": len(train_queries),
        "train_queries_with_candidate_gold": len(eligible),
        "max_in_pool_positives": max_positives,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_complete_dataset(args.data, dataset=args.dataset)
    if len(dataset.queries) != args.expected_queries:
        raise ValueError(
            f"Expected {args.expected_queries} queries for {args.dataset}, got {len(dataset.queries)}"
        )
    splits = {
        split: dataset.split(split)
        for split in (QuerySplit.TRAIN, QuerySplit.VALIDATION, QuerySplit.TEST)
    }
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        nodes = torch.from_numpy(np.asarray(dataset.node_array)).to(device=device, dtype=torch.float32)
        query_embeddings = torch.from_numpy(np.asarray(dataset.query_array)).to(
            device=device,
            dtype=torch.float32,
        )
    empty_topologies: dict[CompleteQuery, torch.Tensor] = {}
    seed_results: dict[str, Any] = {}
    parameters: dict[str, int] | None = None
    for seed in args.seeds:
        seed_everything(seed)
        built = build_operator_model(
            COVERAGE_OFFSET_MODEL,
            dataset.feature_dim,
            args.hidden_dim,
            offset_directions=args.directions,
            dropout=args.dropout,
            temperature=args.temperature,
        )
        if not isinstance(built, SetCoverageOffsetMLP):
            raise TypeError("Coverage model factory returned the wrong model class")
        current_parameters = model_parameter_counts(built)
        if parameters is None:
            parameters = current_parameters
        elif current_parameters != parameters:
            raise RuntimeError("Parameter count changed across seeds")
        trained, training = _train_set_model(
            built,
            splits[QuerySplit.TRAIN],
            splits[QuerySplit.VALIDATION],
            nodes,
            query_embeddings,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            diversity_weight=args.diversity_weight,
            diversity_cosine_margin=args.diversity_cosine_margin,
            seed=seed,
        )
        metrics, rows, inference = _repeated_score(
            trained,
            splits[QuerySplit.TEST],
            nodes,
            query_embeddings,
            empty_topologies,
            device,
            batch_size=args.batch_size,
            ks=tuple(args.ks),
            repeats=args.inference_repeats,
        )
        seed_results[str(seed)] = {
            "metrics": metrics,
            "training": training,
            "inference": inference,
            "per_query": {
                query.query_id: {
                    **row,
                    "gold_count": int(query.relevant_global.numel()),
                    "in_pool_gold_count": int(query.relevant_local.numel()),
                }
                for query, row in zip(splits[QuerySplit.TEST], rows)
            },
        }
        del trained, built
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert parameters is not None
    if parameters["parameters"] != args.expected_parameters:
        raise ValueError(
            f"Expected {args.expected_parameters} parameters, observed {parameters['parameters']}"
        )
    source_files = {
        name: {
            "bytes": (args.data / name).stat().st_size,
            "sha256": (
                sha256_file(args.data / name)
                if name != "nodes.npy" or args.nodes_sha256 == "computed-by-launcher"
                else args.nodes_sha256
            ),
        }
        for name in (
            "nodes.npy",
            "queries_all.npy",
            "dense_top200_all.npy",
            "splade_top200_all.npy",
            "query_ids_all.json",
            "graph.pt",
        )
    }
    result = {
        "status": "PREREGISTERED_COVERAGE_VARIANT_GATE",
        "dataset": dataset.dataset,
        "data": {
            "queries": len(dataset.queries),
            "nodes": dataset.num_nodes,
            "edges": dataset.metadata["num_edges"],
            "feature_dim": dataset.feature_dim,
            "splits": {split.name.lower(): len(values) for split, values in splits.items()},
            "source_files": source_files,
            "candidate_contract_sha256": dataset.metadata["candidate_contract_sha256"],
            "topology_preprocessing_seconds": 0.0,
        },
        "comparison_contract": {
            "same_frozen_query_embeddings": True,
            "same_frozen_node_embeddings": True,
            "same_candidate_pool": True,
            "same_labels_and_all_in_pool_positives": True,
            "same_canonical_splits": True,
            "same_seeds": args.seeds,
            "adjacency_at_train_or_inference": False,
            "sha256": _hash_tensor_contract(dataset.queries),
        },
        "objective": {
            "assignment": "hard minimum over injective direction-to-positive assignments",
            "directions": args.directions,
            "diversity_weight": args.diversity_weight,
            "diversity_cosine_margin": args.diversity_cosine_margin,
            "inference": "max over directional cosine scores",
        },
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "model": {
            "key": f"{COVERAGE_OFFSET_MODEL}_h{args.hidden_dim}",
            "parameters": parameters,
            "uses_topology": False,
            "seeds": seed_results,
            "aggregate": _aggregate_runs(seed_results),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument("--expected-parameters", type=int, default=221504)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--diversity-weight", type=float, default=0.1)
    parser.add_argument("--diversity-cosine-margin", type=float, default=0.2)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 20])
    parser.add_argument("--inference-repeats", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--nodes-sha256", default="computed-by-launcher")
    args = parser.parse_args()
    result = run(args)
    metrics = result["model"]["aggregate"]["test_metrics"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset": result["dataset"],
                "parameters": result["model"]["parameters"],
                "R@1": metrics["recall@1"],
                "R@5": metrics["recall@5"],
                "R@20": metrics["recall@20"],
                "MRR": metrics["mrr"],
                "FC@20": metrics["full_coverage@20"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
