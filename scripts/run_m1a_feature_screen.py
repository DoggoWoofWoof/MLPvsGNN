#!/usr/bin/env python
"""M1A: train BASE +/- one feature family per declared (dataset, regime, arm) cell.

The filed declaration is configs/m1a_feature_screen.yaml; the protocol is
docs/M1A_FEATURE_SCREEN_PROTOCOL.md. This runner covers step 4 (real smoke)
and step 5 (the real one-seed screen) of the plan in that doc's section 13 --
nothing here decides which cells are authorised to run; that gate is external
to this file (see the declaration's own status/authorises fields).

One dataset per invocation, matching every other M0x/D-series runner in this
repo (run_m0b_regime_map.py, run_m0c_bounded_r3.py, run_sa_mlp_confirmation.py).
Which (regime, arm) cells apply to a dataset is read from the filed
declaration's own datasets[dataset].cells block, not re-derived here --
tests/test_m1a_declaration.py is what keeps that block honest against this
runner's expectations. --regimes/--arms optionally narrow that set further,
for a smoke run smaller than the full declared cell list.

Two things this runner gets right that a naive port of run_m0b_regime_map.py
or run_sa_mlp_confirmation.py would not:

1. Feature build (qls_local_features, plus R3's A64 admission) runs exactly
   once per (dataset, regime) *cell*, not once per arm -- see
   configs/m1a_feature_screen.yaml#compute.feature_build_runs_once_per_cell_not_per_arm.
   _cell_master_local computes every family's columns together for a cell;
   _arm_columns only slices them per arm, never recomputes.

2. qls_local_features is called with normalisation="candidate", not the
   "context" default. graph_context.py's own docstring names "context"
   normalisation as confounding "the arm restored structure" with "the
   normaliser's denominator moved" whenever the context set's size changes
   -- which is exactly what happens across R1 (context=Cq) -> R2 (context=U2)
   -> R3 (context=U3, wider still). Since M1A's whole question is a
   regime-by-regime comparison, that confound cannot be allowed into the
   inputs a trained model sees. "candidate" removes it by normalising
   against the scored candidates alone, at the cost of no longer being
   comparable to any figure M0B/M0C reported (their normalisation was
   "context", but they never trained on the values -- only timed the call).

R3's scored set (C3 = Cq union A64) is strictly wider than Cq. The frozen
CompleteQuery.relevant_local (gold positions local to Cq) is computed once,
at dataset-load time, against Cq alone -- reused unchanged it would silently
undercount recall under R3 for every gold node A64 alone admits (M0C
measured 78 such gold instances on this track; see the declaration's
datasets.*.cells R3 rationale). _widen_query re-keys candidate_index and
relevant_local onto a regime's own scored set, mirroring
complete_data.load_complete_dataset's own gold-position construction. R1/R2
widen onto Cq itself, a same-valued round trip -- applied uniformly rather
than special-cased so there is exactly one code path to get right.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def _resolve_repo_root(file_path: Path, sys_path: list[str], marker_relpath: str) -> Path:
    """Find the repo root containing ``marker_relpath``, robust to Modal's mount layout.

    ``file_path``-relative resolution is correct for local/CLI use. It is
    NOT correct for a Modal container: Modal auto-mounts the local
    `scripts`/`src` packages under a container path derived from import
    analysis, separate from this repo's own configs/ (which nothing
    imports, so it is only ever placed by an explicit add_local_file in the
    launcher) -- ``file_path`` then resolves under the auto-mount, not the
    launcher's REMOTE_ROOT mount, so the configs/ sibling this module
    expects is missing next to it. Recover the root from whichever
    ``sys_path`` entry actually has the marker instead.
    """
    candidate = file_path.resolve().parents[1]
    if (candidate / marker_relpath).is_file():
        return candidate
    for entry in sys_path:
        if entry and (Path(entry) / marker_relpath).is_file():
            return Path(entry)
    return candidate


REPO_ROOT = _resolve_repo_root(Path(__file__), sys.path, "configs/m1a_feature_screen.yaml")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.complete_data import CompleteQuery, load_complete_dataset  # noqa: E402
from mp_retrieval.data import QuerySplit  # noqa: E402
from mp_retrieval.graph_context import build_operators, context_nodes, qls_local_features  # noqa: E402
from mp_retrieval.headroom_v2 import ragged_from_rows, regime_headroom  # noqa: E402
from mp_retrieval.m1a_screen import (  # noqa: E402
    FAMILY_COLUMNS,
    base_columns,
    build_m1a_model,
    node_role_column,
    r3_bounded_context,
)
from mp_retrieval.protocol import seed_everything  # noqa: E402
from mp_retrieval.structural_features import StructuralFeatureStore  # noqa: E402
from scripts.run_edge_provenance import _atomic_json  # noqa: E402
from scripts.run_graph_context_d1 import context_feature_store, holdout_split  # noqa: E402
from scripts.run_m0a_probe import (  # noqa: E402
    QueryView,
    _families,
    _load_family_csr,
    _peak_rss_bytes,
    _percentiles,
    _undirected,
)
from scripts.run_m0b_regime_map import CONTEXT_ARM, MAINLINE_FAMILY, _a64_budget  # noqa: E402
from scripts.run_m0b_webqsp_probe import FEATURE_DAMPING, FEATURE_PPR_ITERATIONS  # noqa: E402
from scripts.run_sa_mlp_confirmation import _fit, _score_once, validate_candidate_contract  # noqa: E402

COMPLETE_STATUS = "M1A_FEATURE_SCREEN_DATASET_COMPLETE"
IN_PROGRESS_STATUS = "M1A_FEATURE_SCREEN_DATASET_IN_PROGRESS"
DECLARATION_PATH = REPO_ROOT / "configs" / "m1a_feature_screen.yaml"
PROTOCOL_PATH = "docs/M1A_FEATURE_SCREEN_PROTOCOL.md"
REGIMES = ("R1", "R2", "R3")
KS = (1, 5, 20)

#: Column ranges into one cell's master block (BASE[4] GEOMETRY[3] SUPPORT[1]
#: PATH[3] NODE_ROLE[1]). NODE_ROLE is always present: real under R3,
#: identically zero under R1/R2 -- see _cell_master_local. FAMILY_COLUMNS
#: (m1a_screen.py) slices qls_local_features' own 10-column output; this
#: slices the master block _cell_master_local assembles from it.
MASTER_COLUMNS: dict[str, slice] = {
    "BASE": slice(0, 4),
    "GEOMETRY": slice(4, 7),
    "SUPPORT": slice(7, 8),
    "PATH": slice(8, 11),
    "NODE_ROLE": slice(11, 12),
}

#: M2's universal schema, configs/m2_qls_v2_freeze.yaml#qls_universal.
UNIVERSAL_ARM = "BASE+NODE_ROLE+SUPPORT+PATH"

#: Arm name -> extra families beyond BASE. Mirrors the declaration's arm
#: vocabulary (configs/m1a_feature_screen.yaml#feature_catalog); not read
#: from the yaml itself since this is composition logic, not declared data.
ARM_FAMILIES: dict[str, tuple[str, ...]] = {
    "BASE": (),
    "BASE+GEOMETRY": ("GEOMETRY",),
    "BASE+SUPPORT": ("SUPPORT",),
    "BASE+PATH": ("PATH",),
    "BASE+NODE_ROLE": ("NODE_ROLE",),
    UNIVERSAL_ARM: ("NODE_ROLE", "SUPPORT", "PATH"),
}

#: The only arms allowed a NODE_ROLE column under R1/R2, where that column
#: is identically zero (no scored candidate is structurally admitted when the
#: scored set is Cq). M1A/M1B's historical arms keep NODE_ROLE R3-only.
ZERO_NODE_ROLE_WHEN_NOT_APPLICABLE: frozenset[str] = frozenset({UNIVERSAL_ARM})


def _load_declaration() -> dict[str, Any]:
    return yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))


def _declared_cells(declaration: dict[str, Any], dataset: str) -> dict[str, list[str]]:
    try:
        entry = declaration["datasets"][dataset]
    except KeyError as exc:
        raise ValueError(f"{dataset!r} has no declared cells in {DECLARATION_PATH}") from exc
    cells = {regime: list(cell["arms"]) for regime, cell in entry["cells"].items()}
    for regime, arms in cells.items():
        if regime not in REGIMES:
            raise ValueError(f"Declared regime {regime!r} for {dataset!r} is not one of {REGIMES}")
        unknown = sorted(set(arms) - set(ARM_FAMILIES))
        if unknown:
            raise ValueError(f"Declared arm(s) {unknown} for {dataset!r}/{regime} are not in ARM_FAMILIES")
    return cells


def _widen_query(query: CompleteQuery, scored: np.ndarray) -> CompleteQuery:
    """Re-key ``candidate_index``/``relevant_local`` onto a regime's own scored set.

    Mirrors ``complete_data.load_complete_dataset``'s gold-position
    construction exactly, over ``scored`` instead of the frozen dense/splade
    union. See the module docstring for why this matters under R3.
    """

    scored = np.asarray(scored, dtype=np.int64)
    local = {int(node_id): position for position, node_id in enumerate(scored.tolist())}
    relevant_local = torch.tensor(
        [local[int(gold)] for gold in query.relevant_global.tolist() if int(gold) in local],
        dtype=torch.long,
    )
    return dataclasses.replace(
        query,
        candidate_index=torch.from_numpy(scored),
        relevant_local=relevant_local,
    )


def _cell_master_local(
    *,
    regime: str,
    views: list[QueryView],
    queries: list[CompleteQuery],
    dense: np.ndarray,
    splade: np.ndarray,
    rowptr: np.ndarray,
    col: np.ndarray,
    num_nodes: int,
    operators,
    family_rowptr: np.ndarray | None,
    family_col: np.ndarray | None,
    node_embeddings,
    budget,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """One (dataset, regime) cell's per-query ``(scored_ids, master_columns, latency_ms)``.

    ``scored_ids`` is always ``np.unique``-d before being used anywhere:
    ``qls_local_features`` internally sorts its own ``pool`` argument and
    returns rows in that sorted order, so using the sorted-unique array as
    the single canonical candidate order (for the feature call, for
    ``base_columns``, and for the widened query's own ``candidate_index``)
    means every row lines up by construction -- no second reindex, and
    nothing to get silently out of sync.

    Called once per cell; arms slice the returned master columns, never
    rebuild them (see module docstring point 1).
    """

    scored_sets: list[np.ndarray] = []
    master_blocks: list[np.ndarray] = []
    latencies: list[float] = []
    for view, query in zip(views, queries, strict=True):
        if regime == "R1":
            scored = np.unique(view.pool)
            context_ids = scored
        elif regime == "R2":
            scored = np.unique(view.pool)
            context_ids = context_nodes(CONTEXT_ARM, operators=operators, pool=view.pool, seeds=view.seeds)
        elif regime == "R3":
            u2 = context_nodes(CONTEXT_ARM, operators=operators, pool=view.pool, seeds=view.seeds)
            u3_bounded, c3, _a64, _expansion = r3_bounded_context(
                family_rowptr=family_rowptr,
                family_col=family_col,
                node_embeddings=node_embeddings,
                anchor=view.anchor,
                pool=view.pool,
                seeds=view.seeds,
                budget=budget,
                num_nodes=num_nodes,
                u2=u2,
            )
            scored = np.unique(c3)
            context_ids = u3_bounded
        else:
            raise ValueError(f"Unknown regime {regime!r}")

        started = time.perf_counter()
        local10 = qls_local_features(
            rowptr=rowptr,
            col=col,
            nodes=context_ids,
            pool=scored,
            seeds=view.seeds,
            size=num_nodes,
            damping=FEATURE_DAMPING,
            ppr_iterations=FEATURE_PPR_ITERATIONS,
            edge_source=operators.edge_source,
            normalisation="candidate",
        )
        latencies.append((time.perf_counter() - started) * 1000.0)

        base = base_columns(
            local10,
            dense=np.asarray(dense[query.query_index]),
            splade=np.asarray(splade[query.query_index]),
            candidates=scored,
        )
        family_block = np.concatenate(
            [
                local10[:, FAMILY_COLUMNS["GEOMETRY"]],
                local10[:, FAMILY_COLUMNS["SUPPORT"]],
                local10[:, FAMILY_COLUMNS["PATH"]],
            ],
            axis=1,
        )
        if regime == "R3":
            node_role = node_role_column(cq=np.unique(view.pool), scored=scored, context=context_ids)
        else:
            # scored IS Cq above, so NODE_ROLE's definition (1 iff v in C3 \ Cq)
            # is identically 0 here -- the feature's value, not a placeholder.
            node_role = np.zeros((len(scored), 1), dtype=np.float32)
        master_blocks.append(np.concatenate([base, family_block, node_role], axis=1).astype(np.float32))
        scored_sets.append(scored)
    return scored_sets, master_blocks, latencies


def _arm_columns(master: np.ndarray, arm: str, regime: str) -> np.ndarray:
    blocks = [master[:, MASTER_COLUMNS["BASE"]]]
    for family in ARM_FAMILIES[arm]:
        block = master[:, MASTER_COLUMNS[family]]
        if family == "NODE_ROLE" and regime != "R3":
            if arm not in ZERO_NODE_ROLE_WHEN_NOT_APPLICABLE:
                raise ValueError("NODE_ROLE is only defined for R3 cells")
            if block.any():
                raise ValueError(f"NODE_ROLE must be identically zero under {regime} for {arm!r}")
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


def _arm_store(
    *,
    arm: str,
    regime: str,
    master_blocks: list[np.ndarray],
    queries: list[CompleteQuery],
    query_count: int,
    num_nodes: int,
) -> tuple[StructuralFeatureStore, int]:
    blocks = [_arm_columns(master, arm, regime).astype(np.float16) for master in master_blocks]
    local = np.concatenate(blocks, axis=0) if blocks else np.zeros((0, 0), dtype=np.float16)
    candidate_ptr = np.zeros(len(blocks) + 1, dtype=np.int64)
    for index, block in enumerate(blocks):
        candidate_ptr[index + 1] = candidate_ptr[index] + block.shape[0]
    static = np.zeros((num_nodes, 0), dtype=np.float32)
    store = context_feature_store(queries, static, local, candidate_ptr, query_count, arm=arm)
    return store, int(local.shape[1])


def _run_arm(
    *,
    arm: str,
    regime: str,
    store: StructuralFeatureStore,
    precomputed_width: int,
    train_queries: list[CompleteQuery],
    validation_queries: list[CompleteQuery],
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    seed_everything(args.seed)
    model = build_m1a_model(
        precomputed_width=precomputed_width,
        semantic_rung=args.semantic_rung,
        dropout=args.dropout,
        temperature=args.temperature,
        embedding_dim=node_embeddings.shape[1],
    )
    fitted, training = _fit(
        "sa_mlp",
        model,
        train_queries,
        validation_queries,
        node_embeddings,
        query_embeddings,
        None,
        store,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    metrics, _rows, inference = _score_once(
        "sa_mlp",
        fitted,
        validation_queries,
        node_embeddings,
        query_embeddings,
        None,
        store,
        device,
        batch_size=args.batch_size,
        ks=KS,
        timed=True,
    )
    return {
        "arm": arm,
        "regime": regime,
        "semantic_rung": args.semantic_rung,
        "precomputed_width": precomputed_width,
        "parameters": {
            "total": fitted.trainable_parameter_count(),
            "semantic": fitted.semantic_parameter_count(),
            "scorer": fitted.scorer_parameter_count(),
        },
        "metrics": metrics,
        "training": training,
        "inference": inference,
        "eligible_train_queries": sum(1 for query in train_queries if query.relevant_local.numel()),
        "systems": {
            "train_time_seconds": training["training_seconds"],
            "peak_train_vram_mb": training["peak_training_gpu_memory_mb_total"],
            "peak_train_rss_mb": inference["peak_cpu_rss_mb_total"],
            "peak_train_rss_mb_provenance": (
                "measured during post-fit validation scoring, not the training loop itself -- "
                "_fit is reused unmodified (see reuse contract) and does not instrument CPU RSS"
            ),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("status") == COMPLETE_STATUS:
            return existing

    declaration = _load_declaration()
    cells = _declared_cells(declaration, args.dataset)
    if args.regimes is not None:
        unknown = sorted(set(args.regimes) - set(cells))
        if unknown:
            raise ValueError(f"--regimes {unknown} are not declared for {args.dataset!r}: {sorted(cells)}")
        cells = {regime: cells[regime] for regime in args.regimes}
    if args.arms is not None:
        cells = {regime: [arm for arm in arms if arm in args.arms] for regime, arms in cells.items()}
        empty = [regime for regime, arms in cells.items() if not arms]
        if empty:
            raise ValueError(f"--arms {args.arms} leaves regime(s) {empty} with no arms to run")

    dataset = load_complete_dataset(args.data, dataset=args.dataset, require_embeddings=True)
    if len(dataset.queries) != args.expected_queries:
        raise ValueError("Complete dataset query count differs from the registered protocol")
    if dataset.node_array.shape[1] != args.frozen_embedding_dim:
        raise ValueError(
            f"{args.dataset}: node embeddings are {dataset.node_array.shape[1]}-dimensional, "
            f"--frozen-embedding-dim says {args.frozen_embedding_dim} -- SemanticHead would "
            "silently build at the wrong width and every downstream parameter count would be wrong"
        )
    candidate_contract = validate_candidate_contract(
        args.baseline, dataset, args.candidate_contract_compatibility
    )
    queries = dataset.split(QuerySplit.VALIDATION)[: args.queries]
    if len(queries) < args.queries:
        raise ValueError(f"validation split holds {len(queries)} queries, needed {args.queries}")
    views = [QueryView(query) for query in queries]
    num_nodes = int(dataset.num_nodes)
    query_count = len(dataset.queries)

    rowptr = dataset.rowptr.numpy().astype(np.int64, copy=False)
    col = dataset.col.numpy().astype(np.int64, copy=False)
    operators = build_operators(rowptr, col, num_nodes)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # np.array(..., copy=True) rather than np.asarray: node_array/query_array
    # are read-only mmaps (load_complete_dataset's zero-copy convention), and
    # a same-dtype np.asarray on one is still a read-only view -- torch.from_numpy
    # on that produces a tensor whose writes are undefined behaviour.
    node_embeddings = torch.from_numpy(np.array(dataset.node_array, dtype=np.float32, copy=True)).to(device)
    query_embeddings = torch.from_numpy(np.array(dataset.query_array, dtype=np.float32, copy=True)).to(device)
    dense = np.load(Path(args.data) / "dense_top200_all.npy", mmap_mode="r")
    splade = np.load(Path(args.data) / "splade_top200_all.npy", mmap_mode="r")

    family_rowptr = family_col = None
    if "R3" in cells:
        families = _families(args)
        if args.a64_mainline_family not in families:
            raise ValueError(
                f"declared mainline family {args.a64_mainline_family!r} is not among {list(families)}"
            )
        family_rowptr, family_col = _load_family_csr(families[args.a64_mainline_family], num_nodes)
        family_rowptr, family_col, _symmetric = _undirected(family_rowptr, family_col, num_nodes)
    budget = _a64_budget(per_seed_cap=args.per_seed_cap, neighbour_scan_cap_per_seed=args.neighbour_scan_cap_per_seed)

    result: dict[str, Any] = {
        "status": IN_PROGRESS_STATUS,
        "stage": "m1a_feature_screen",
        "dataset": args.dataset,
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "declaration": "configs/m1a_feature_screen.yaml",
        "protocol": PROTOCOL_PATH,
        "candidate_contract": candidate_contract,
        "queries": len(views),
        "split": "validation",
        "selection": "deterministic_prefix_of_the_split_order",
        "test_split_read": False,
        "holdout_fraction": args.holdout_fraction,
        "seed": args.seed,
        "semantic_rung": args.semantic_rung,
        "num_nodes": num_nodes,
        "a64_mainline_family": args.a64_mainline_family if "R3" in cells else None,
        "cells": {},
        "systems": {},
    }

    golds = ragged_from_rows([view.golds for view in views])
    for regime, arms in cells.items():
        scored_sets, master_blocks, latencies = _cell_master_local(
            regime=regime,
            views=views,
            queries=queries,
            dense=dense,
            splade=splade,
            rowptr=rowptr,
            col=col,
            num_nodes=num_nodes,
            operators=operators,
            family_rowptr=family_rowptr,
            family_col=family_col,
            node_embeddings=dataset.node_array,
            budget=budget,
        )
        headroom, _present, _gold_counts = regime_headroom(
            ragged_from_rows(scored_sets), golds, num_nodes=num_nodes, ks=KS
        )
        widened = [
            _widen_query(query, scored) for query, scored in zip(queries, scored_sets, strict=True)
        ]
        train_queries, held_out_queries = holdout_split(widened, args.holdout_fraction)

        cell_result: dict[str, Any] = {
            "arms_run": list(arms),
            "regime_headroom": headroom,
            "scored_node_count": _percentiles([float(scored.size) for scored in scored_sets]),
            "uncached_feature_build_latency_ms": _percentiles(latencies),
            "train_queries": len(train_queries),
            "held_out_queries": len(held_out_queries),
            "arms": {},
        }
        for arm in arms:
            store, precomputed_width = _arm_store(
                arm=arm,
                regime=regime,
                master_blocks=master_blocks,
                queries=widened,
                query_count=query_count,
                num_nodes=num_nodes,
            )
            arm_result = _run_arm(
                arm=arm,
                regime=regime,
                store=store,
                precomputed_width=precomputed_width,
                train_queries=train_queries,
                validation_queries=held_out_queries,
                node_embeddings=node_embeddings,
                query_embeddings=query_embeddings,
                device=device,
                args=args,
            )
            achieved = arm_result["metrics"].get("recall@5")
            ceiling = headroom.get("recall_ceiling@5")
            arm_result["ceiling_attainment_at_5"] = (
                float(achieved) / float(ceiling) if achieved is not None and ceiling else None
            )
            cell_result["arms"][arm] = arm_result
        result["cells"][regime] = cell_result

    result["systems"] = {"peak_process_rss_bytes": _peak_rss_bytes()}
    result["status"] = COMPLETE_STATUS
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-fingerprint-sha256", required=True)
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--frozen-embedding-dim", type=int, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate-contract-compatibility", default=None)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--per-seed-cap", type=int, default=16)
    parser.add_argument("--neighbour-scan-cap-per-seed", type=int, default=4096)
    parser.add_argument("--edge-provenance-root", type=Path, default=None)
    parser.add_argument("--edge-families", nargs="+", default=[MAINLINE_FAMILY])
    parser.add_argument("--a64-mainline-family", default=MAINLINE_FAMILY)
    parser.add_argument("--regimes", nargs="+", default=None, choices=REGIMES)
    parser.add_argument("--arms", nargs="+", default=None, choices=list(ARM_FAMILIES))
    parser.add_argument("--semantic-rung", default="S3", choices=("S2", "S3"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if args.seed != 0:
        raise ValueError("M1A is declared as a one-seed (seed=0) screen; see configs/m1a_feature_screen.yaml#sampling")
    run(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
