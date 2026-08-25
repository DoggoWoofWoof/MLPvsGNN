#!/usr/bin/env python
"""Export CRAG L2 signal caches as query-local candidate graph data.

The script reads CRAG but writes only to this repository (or an explicitly
selected output path). Current ``signals_*`` caches contain test queries only,
so exported artifacts are marked ``pilot_test_only`` and must not be used as a
paper training/test split. Canonical train/validation/test caches will replace
them later without changing the data contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.data import QuerySplit  # noqa: E402
from mp_retrieval.l2_data import CandidateQuery, L2CandidateDataset, edge_index_to_csr  # noqa: E402
from mp_retrieval.protocol import sha256_file  # noqa: E402


def _stable_pool(orders: np.ndarray) -> list[int]:
    """Round-robin expert rankings so no expert controls candidate ordering."""

    result: list[int] = []
    seen: set[int] = set()
    for rank in range(orders.shape[1]):
        for expert in range(orders.shape[0]):
            node = int(orders[expert, rank])
            if node >= 0 and node not in seen:
                seen.add(node)
                result.append(node)
    return result


def _queries_from_signals(path: Path, dataset: str) -> tuple[list[str], list[CandidateQuery]]:
    cache = np.load(path, allow_pickle=True)
    orders = cache["orders"]
    scores = cache["scores"]
    golds = cache["golds"]
    names = [str(x) for x in cache["names"].tolist()]
    queries: list[CandidateQuery] = []
    for query_idx in range(orders.shape[1]):
        query_orders = orders[:, query_idx, :]
        pool = _stable_pool(query_orders)
        local = {node: idx for idx, node in enumerate(pool)}
        evidence = torch.zeros((len(pool), len(names)), dtype=torch.float32)
        mask = torch.zeros((len(pool), len(names)), dtype=torch.bool)
        for expert_idx in range(len(names)):
            for rank, node_raw in enumerate(query_orders[expert_idx]):
                node = int(node_raw)
                if node < 0:
                    continue
                position = local[node]
                evidence[position, expert_idx] = float(scores[expert_idx, query_idx, rank])
                mask[position, expert_idx] = True
        gold_values = np.asarray(golds[query_idx]).reshape(-1)
        gold_global = sorted({int(x) for x in gold_values if int(x) >= 0})
        gold_local = [local[node] for node in gold_global if node in local]
        queries.append(
            CandidateQuery(
                query_id=f"{dataset}:pilot_test:{query_idx}",
                candidate_index=torch.tensor(pool, dtype=torch.long),
                expert_scores=evidence,
                expert_mask=mask,
                relevant_local=torch.tensor(gold_local, dtype=torch.long),
                relevant_global=torch.tensor(gold_global, dtype=torch.long),
                split=int(QuerySplit.TEST),
                metadata={
                    "candidate_ceiling": len(gold_local) / max(len(gold_global), 1),
                    "source_query_offset": query_idx,
                },
            )
        )
    return names, queries


def export(crag_root: Path, dataset: str, output: Path, index_subdir: str) -> None:
    signal_path = crag_root / "results" / "L2" / f"signals_{dataset}_gte_qwen.npz"
    graph_path = crag_root / "data" / "ukb_storage" / dataset / index_subdir / "graph.pt"
    if not graph_path.exists():
        graph_path = crag_root / "data" / "ukb_storage" / dataset / "graph.pt"
    if not signal_path.exists() or not graph_path.exists():
        raise FileNotFoundError(f"Missing signal or graph artifact for {dataset}")
    names, queries = _queries_from_signals(signal_path, dataset)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    num_nodes = int(getattr(graph, "num_nodes"))
    rowptr, col, _ = edge_index_to_csr(graph.edge_index, num_nodes)
    artifact = L2CandidateDataset(
        dataset=dataset,
        num_nodes=num_nodes,
        signal_names=names,
        rowptr=rowptr,
        col=col,
        queries=queries,
        metadata={
            "status": "pilot_test_only",
            "warning": "No training split is present; do not report as paper evidence.",
            "crag_root_read_only": str(crag_root.resolve()),
            "signals_source": str(signal_path.resolve()),
            "signals_sha256": sha256_file(signal_path),
            "graph_source": str(graph_path.resolve()),
            "graph_sha256": sha256_file(graph_path),
            "edge_semantics": "untyped CRAG graph; canonical typed-edge export still required",
        },
    )
    artifact.save(output)
    summary = {
        "dataset": dataset,
        "queries": len(queries),
        "nodes": num_nodes,
        "edges": int(col.numel()),
        "mean_pool": float(np.mean([q.candidate_index.numel() for q in queries])),
        "mean_candidate_ceiling": float(np.mean([q.candidate_ceiling for q in queries])),
        "output": str(output.resolve()),
    }
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crag-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-subdir", default="gte_qwen")
    args = parser.parse_args()
    export(args.crag_root, args.dataset, args.output, args.index_subdir)


if __name__ == "__main__":
    main()
