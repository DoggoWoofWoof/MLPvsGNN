#!/usr/bin/env python
"""Compute read-only candidate-generation headroom for a frozen dataset.

Every number here is diagnostic. The script loads the frozen complete dataset,
proves the candidate contract is bit-exact against the registered confirmation,
and then measures what the reported metrics could ever have reached. It never
admits a node into a pool, never expands a pool along graph edges, never
reorders a pool, and never rewrites a frozen artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.candidate_headroom import (
    missing_gold_reachability,
    source_headroom,
    symmetric_csr,
)
from mp_retrieval.complete_data import load_complete_dataset
from mp_retrieval.data import QuerySplit
from scripts.run_edge_provenance import _atomic_json
from scripts.run_sa_mlp_confirmation import validate_candidate_contract

COMPLETE_STATUS = "CANDIDATE_HEADROOM_DIAGNOSTIC_COMPLETE"
SPLITS = {
    "train": QuerySplit.TRAIN,
    "validation": QuerySplit.VALIDATION,
    "test": QuerySplit.TEST,
}


def completed_headroom(args: argparse.Namespace) -> dict[str, Any] | None:
    """Return an already-written diagnostic, or ``None`` to compute it."""

    if not args.output.is_file():
        return None
    existing = json.loads(args.output.read_text(encoding="utf-8"))
    if existing.get("status") != COMPLETE_STATUS:
        return None
    contract = existing.get("diagnostic_contract", {})
    if (
        existing.get("dataset") != args.dataset
        or existing.get("data_fingerprint_sha256") != args.data_fingerprint_sha256
        or list(existing.get("ks", [])) != list(args.ks)
        or list(existing.get("budgets", [])) != list(args.budgets)
        or int(existing.get("rrf_constant", -1)) != int(args.rrf_constant)
        or int(contract.get("max_hops", -1)) != int(args.max_hops)
        or contract.get("candidate_pools_modified") is not False
    ):
        raise ValueError("Existing headroom diagnostic has a different diagnostic contract")
    return existing


def _edge_index_from_csr(rowptr: np.ndarray, col: np.ndarray) -> np.ndarray:
    """Recover the stored directed edge list without re-reading the graph file."""

    degrees = np.diff(rowptr)
    source = np.repeat(np.arange(degrees.size, dtype=np.int64), degrees)
    return np.stack((source, col.astype(np.int64, copy=False)))


def run(
    args: argparse.Namespace,
    checkpoint_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    finished = completed_headroom(args)
    if finished is not None:
        return finished

    dataset = load_complete_dataset(args.data, dataset=args.dataset)
    if len(dataset.queries) != args.expected_queries:
        raise ValueError("Complete dataset query count differs from the registered protocol")
    contract_before = dataset.metadata["candidate_contract_sha256"]
    candidate_contract = validate_candidate_contract(
        args.baseline,
        dataset,
        args.candidate_contract_compatibility,
    )

    dense = np.load(args.data / "dense_top200_all.npy", mmap_mode="r")
    splade = np.load(args.data / "splade_top200_all.npy", mmap_mode="r")
    ks = tuple(int(k) for k in args.ks)
    budgets = tuple(int(budget) for budget in args.budgets)

    result: dict[str, Any] = {
        "status": "CANDIDATE_HEADROOM_IN_PROGRESS",
        "dataset": args.dataset,
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract": candidate_contract,
        "ks": list(ks),
        "budgets": list(budgets),
        "rrf_constant": int(args.rrf_constant),
        "queries": len(dataset.queries),
        "num_nodes": int(dataset.num_nodes),
        "num_stored_directed_edges": int(dataset.metadata["num_edges"]),
        "diagnostic_contract": {
            "read_only": True,
            "candidate_pools_modified": False,
            "candidate_pools_expanded": False,
            "candidate_pools_reordered": False,
            "frozen_hashes_changed": False,
            "graph_expansion_performed": False,
            "candidate_admission_performed": False,
            "max_hops": int(args.max_hops),
            "ceiling_formula": "min(gold_in_pool, K) / gold_total",
            "coverage_is_not_oracle_recall_at_k": True,
            "budget_fusion_mirrors": "candidate_budget.build_budget_dataset",
        },
        "headroom": {},
        "reachability": {},
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key != "baseline"
        },
    }

    for name, split in SPLITS.items():
        queries = dataset.split(split)
        if not queries:
            continue
        result["headroom"][name] = source_headroom(
            queries,
            dense,
            splade,
            num_nodes=dataset.num_nodes,
            ks=ks,
            budgets=budgets,
            rrf_constant=int(args.rrf_constant),
        )
        _atomic_json(args.output, result)
        if checkpoint_hook is not None:
            checkpoint_hook()

    if args.reachability_splits:
        edge_index = _edge_index_from_csr(dataset.rowptr.numpy(), dataset.col.numpy())
        rowptr, col, was_symmetric = symmetric_csr(
            torch.from_numpy(edge_index), dataset.num_nodes
        )
        del edge_index
        result["diagnostic_contract"]["stored_graph_was_symmetric"] = bool(was_symmetric)
        for name in args.reachability_splits:
            queries = dataset.split(SPLITS[name])
            if not queries:
                continue
            result["reachability"][name] = missing_gold_reachability(
                queries,
                rowptr,
                col,
                num_nodes=dataset.num_nodes,
                max_hops=int(args.max_hops),
                max_visited=int(args.max_visited),
            )
            _atomic_json(args.output, result)
            if checkpoint_hook is not None:
                checkpoint_hook()

    if dataset.metadata["candidate_contract_sha256"] != contract_before:
        raise RuntimeError("Candidate contract changed while computing a read-only diagnostic")
    result["status"] = COMPLETE_STATUS
    _atomic_json(args.output, result)
    if checkpoint_hook is not None:
        checkpoint_hook()
    return result


if __name__ == "__main__":
    raise SystemExit("Use scripts/modal_candidate_headroom.py for the registered execution")
