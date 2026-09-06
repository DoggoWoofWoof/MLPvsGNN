#!/usr/bin/env python
"""M1B: targeted uncertainty + minimality resolution -- the multi-seed runner.

configs/m1b_targeted_resolution.yaml is the declaration. It is its own
protocol document for this phase -- M1B's scope is four cells, not the
eight-step plan M1A's docs/M1A_FEATURE_SCREEN_PROTOCOL.md needed.

This is a new script, not an edit to run_m1a_feature_screen.py, because
that file is hard-declared a one-seed (seed=0) screen (its own main():
``if args.seed != 0: raise ValueError(...)``) and the M1B reuse_contract's
git-source-identity check depends on that file staying byte-unchanged
since the M1A step-5 launch, not just until the next convenient edit. This
runner instead imports and reuses its feature-build, arm-column, and
per-seed-fit internals unmodified -- _cell_master_local, _widen_query,
_arm_store, _run_arm, MASTER_COLUMNS -- and adds exactly the two things
M1A never needed:

1. A seed loop per cell. Feature build is confirmed seed-independent (the
   declaration's own uncertainty_procedure.design.query_level: neither
   _cell_master_local nor holdout_split takes a seed argument) and runs
   exactly once per (dataset, R3) cell; _arm_store slices it once per arm.
   Both are reused across every development seed in that cell -- only
   _run_arm's seed_everything call ever depends on seed.

2. X/Y symbolic-arm resolution. configs/m1b_targeted_resolution.yaml
   declares 2wiki_clean's third arm as "BASE+NODE_ROLE+X" and
   hotpotqa_clean's second arm as "BASE+Y", bound to a concrete family
   only in that file's own mechanical_selection_of_x_and_y (derived after
   M1A, mechanically, from the filed Pareto tie-break). M1A's static
   ARM_FAMILIES vocabulary has no reason to know a family that did not
   exist as an arm until this file; _resolve_symbolic_arm substitutes the
   bound family in before any of M1A's arm-composition code ever sees the
   name. The one genuinely new arm this produces, BASE+NODE_ROLE+SUPPORT,
   is registered into run_m1a_feature_screen's own ARM_FAMILIES dict at
   import time (see below) rather than duplicating _arm_columns' body --
   that function reads the module-global ARM_FAMILIES by name, not a
   parameter, so extending the dict in place is the reuse-preserving way
   to teach it one more arm without touching its source.

Where the reuse_contract audit (scripts/m1b_reuse_audit.py, outputs/
m1b_targeted_resolution/reuse_audit.json) proves a (dataset, R3, arm)
seed=0 result scientifically identical to what this runner would produce,
that (arm, seed=0) pair used to be spliced in from the real M1A headline
JSON instead of retrained. Amendment 5 (2026-09-05) ended that: the
preregistered paired query-level bootstrap needs genuine per-query
outcomes, and neither this runner nor run_m1a_feature_screen.py ever
persisted them or a model checkpoint -- _score_once computes them
internally and _run_arm discards them before returning. There is therefore
no way to recover per-query rows for an already-spliced cell without
re-fitting it. All 27 logical (dataset, arm, seed) fits are now trained
here for real; the 8 reuse_contract candidates are additionally
cross-checked against M1A's original headline recall@5 at seed=0
(within REUSE_CROSS_CHECK_TOLERANCE_PP, not exact equality -- GPU training
is not bit-deterministic across separate processes the way the CPU-only
feature/headroom computation the recall_ceiling@5 check below guards is)
rather than trusted blindly. A missing or non-passing reuse manifest still
stops the run: the cross-check is a safety net on top of the audit, not a
replacement for it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def _resolve_repo_root(file_path: Path, sys_path: list[str], marker_relpath: str) -> Path:
    """Find the repo root containing ``marker_relpath``, robust to Modal's mount layout.

    Identical in behaviour to run_m1a_feature_screen._resolve_repo_root --
    copied rather than imported because this module must compute its own
    REPO_ROOT and populate sys.path *before* ``scripts.*`` is importable at
    all, the same bootstrapping constraint every runner script in this repo
    resolves independently (run_sa_mlp_confirmation.py, run_m0b_regime_map.py
    each set their own module-level REPO_ROOT the same way). See that
    function's docstring for why the naive ``parents[1]`` guess is not
    always correct under Modal's auto-mount.
    """
    candidate = file_path.resolve().parents[1]
    if (candidate / marker_relpath).is_file():
        return candidate
    for entry in sys_path:
        if entry and (Path(entry) / marker_relpath).is_file():
            return Path(entry)
    return candidate


REPO_ROOT = _resolve_repo_root(Path(__file__), sys.path, "configs/m1b_targeted_resolution.yaml")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.complete_data import load_complete_dataset  # noqa: E402
from mp_retrieval.data import QuerySplit  # noqa: E402
from mp_retrieval.graph_context import build_operators  # noqa: E402
from mp_retrieval.headroom_v2 import ragged_from_rows, regime_headroom  # noqa: E402
from scripts import run_m1a_feature_screen as _m1a  # noqa: E402
from scripts.run_edge_provenance import _atomic_json  # noqa: E402
from scripts.run_graph_context_d1 import holdout_split  # noqa: E402
from scripts.run_m0a_probe import QueryView, _families, _load_family_csr, _peak_rss_bytes, _percentiles, _undirected  # noqa: E402
from scripts.run_m0b_regime_map import MAINLINE_FAMILY, _a64_budget  # noqa: E402
from scripts.run_sa_mlp_confirmation import _fit, _score_once, validate_candidate_contract  # noqa: E402

# _arm_columns (inside run_m1a_feature_screen) reads the module-global
# ARM_FAMILIES by name, not as a parameter -- extending that dict in place
# teaches every M1A internal this runner reuses (_arm_columns, _arm_store)
# about the one new interaction arm M1B introduces, without touching
# run_m1a_feature_screen.py's own source. NODE_ROLE before SUPPORT,
# matching MASTER_COLUMNS' declared order (NODE_ROLE=11:12 is appended
# after SUPPORT=7:8 in the master block, but _arm_columns concatenates in
# ARM_FAMILIES[arm] order, and configs/m1b_targeted_resolution.yaml's own
# primary_comparison names the arm "BASE+NODE_ROLE+SUPPORT" in this order).
_m1a.ARM_FAMILIES.setdefault("BASE+NODE_ROLE+SUPPORT", ("NODE_ROLE", "SUPPORT"))

DECLARATION_PATH = REPO_ROOT / "configs" / "m1b_targeted_resolution.yaml"
M1A_HEADLINE_DIR = REPO_ROOT / "outputs" / "m1a_feature_screen" / "headline"
REUSE_MANIFEST_PATH = REPO_ROOT / "outputs" / "m1b_targeted_resolution" / "reuse_audit.json"
COMPLETE_STATUS = "M1B_TARGETED_RESOLUTION_DATASET_COMPLETE"
IN_PROGRESS_STATUS = "M1B_TARGETED_RESOLUTION_DATASET_IN_PROGRESS"
# Reuses the track's own >0.50pp "material" threshold (configs/
# m1b_targeted_resolution.yaml#promotion_rule_for_m1b) as the cross-check
# tolerance: a re-fit that disagrees with M1A's original by less than the
# amount this track already treats as noise is not a reuse-premise failure.
REUSE_CROSS_CHECK_TOLERANCE_PP = 0.50
KS = _m1a.KS
MASTER_COLUMNS = _m1a.MASTER_COLUMNS
ARM_FAMILIES = _m1a.ARM_FAMILIES


def _load_declaration() -> dict[str, Any]:
    return yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))


def _resolve_symbolic_arm(arm: str, *, x: str | None, y: str | None) -> str:
    """Substitute the bound family for a literal ``X``/``Y`` token in an arm name.

    Token-based (split on "+"), not substring replacement, so a family name
    that happened to contain the letter X or Y could never be mismatched --
    none currently does, but this does not rely on that staying true.
    """
    resolved = []
    for token in arm.split("+"):
        if token == "X":
            if x is None:
                raise ValueError(f"arm {arm!r} uses the X placeholder but no x_binding is declared")
            resolved.append(x)
        elif token == "Y":
            if y is None:
                raise ValueError(f"arm {arm!r} uses the Y placeholder but no y_binding is declared")
            resolved.append(y)
        else:
            resolved.append(token)
    return "+".join(resolved)


def _declared_cell(declaration: dict[str, Any], dataset: str) -> dict[str, Any]:
    try:
        entry = declaration["cells"][dataset]
    except KeyError as exc:
        raise ValueError(f"{dataset!r} has no declared cell in {DECLARATION_PATH}") from exc
    regime = entry["regime"]
    if regime != "R3":
        raise ValueError(f"{dataset!r}'s declared regime is {regime!r}, but every M1B cell is R3 -- re-check the declaration")
    x = entry.get("x_binding")
    y = entry.get("y_binding")
    arms = [_resolve_symbolic_arm(arm, x=x, y=y) for arm in entry["arms"]]
    unknown = sorted(set(arms) - set(ARM_FAMILIES))
    if unknown:
        raise ValueError(f"Resolved arm(s) {unknown} for {dataset!r} are not in ARM_FAMILIES")
    return {"regime": regime, "arms": arms}


def _load_reuse_manifest() -> dict[tuple[str, str], bool]:
    if not REUSE_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"{REUSE_MANIFEST_PATH} does not exist -- run scripts/m1b_reuse_audit.py "
            "before this runner; seed=0 reuse must be proven, never assumed"
        )
    manifest = json.loads(REUSE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not manifest.get("all_8_reusable"):
        raise ValueError(
            f"{REUSE_MANIFEST_PATH} reports all_8_reusable={manifest.get('all_8_reusable')!r} -- "
            "re-run scripts/m1b_reuse_audit.py and resolve why before trusting any reuse from it. "
            "This runner refuses to guess which subset is still safe."
        )
    return {(c["dataset"], c["arm"]): bool(c["reusable"]) for c in manifest["reuse_candidates"]}


def _load_m1a_headline(dataset: str) -> dict[str, Any]:
    path = M1A_HEADLINE_DIR / f"{dataset}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "M1A_FEATURE_SCREEN_DATASET_COMPLETE":
        raise ValueError(f"{path} is not a complete M1A headline artifact")
    return data


def _splice_from_m1a(headline: dict[str, Any], *, regime: str, arm: str) -> dict[str, Any]:
    """A shallow copy of M1A's own arm_result for (regime, arm) -- never mutated in place."""
    return dict(headline["cells"][regime]["arms"][arm])


def _run_arm_with_rows(
    *,
    arm: str,
    regime: str,
    store: Any,
    precomputed_width: int,
    train_queries: list[Any],
    validation_queries: list[Any],
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Amendment 5's row-capturing twin of run_m1a_feature_screen._run_arm.

    Duplicated rather than called, and duplicated rather than imported and
    wrapped: _run_arm's own _score_once call already computes the per-query
    rows this needs, but discards them before returning, and it is one of
    the git-source-identity-frozen functions this track's reuse contract
    depends on staying byte-unchanged -- so there is no way to get the rows
    back out of it. Every other line below is copied from it unmodified.
    """
    _m1a.seed_everything(args.seed)
    model = _m1a.build_m1a_model(
        precomputed_width=precomputed_width,
        semantic_rung=args.semantic_rung,
        dropout=args.dropout,
        temperature=args.temperature,
        embedding_dim=node_embeddings.shape[1],
    )
    fitted, training = _fit(
        "sa_mlp", model, train_queries, validation_queries,
        node_embeddings, query_embeddings, None, store, device,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        seed=args.seed,
    )
    metrics, rows, inference = _score_once(
        "sa_mlp", fitted, validation_queries, node_embeddings, query_embeddings,
        None, store, device, batch_size=args.batch_size, ks=KS, timed=True,
    )
    per_query_recall_at_5 = {
        query.query_id: float(row["recall@5"])
        for query, row in zip(validation_queries, rows, strict=True)
    }
    arm_result = {
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
    return arm_result, per_query_recall_at_5


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("status") == COMPLETE_STATUS:
            return existing

    declaration = _load_declaration()
    cell_decl = _declared_cell(declaration, args.dataset)
    regime = cell_decl["regime"]
    arms = cell_decl["arms"]
    if args.arms is not None:
        arms = [arm for arm in arms if arm in args.arms]
        if not arms:
            raise ValueError(f"--arms {args.arms} leaves {args.dataset!r} with no arms to run")
    seeds = list(args.seeds) if args.seeds is not None else list(declaration["seeds"]["development_seed_values"])

    reuse_lookup = _load_reuse_manifest()
    needs_m1a_headline = any(reuse_lookup.get((args.dataset, arm), False) and 0 in seeds for arm in arms)
    m1a_headline = _load_m1a_headline(args.dataset) if needs_m1a_headline else None

    dataset = load_complete_dataset(args.data, dataset=args.dataset, require_embeddings=True)
    if len(dataset.queries) != args.expected_queries:
        raise ValueError("Complete dataset query count differs from the registered protocol")
    if dataset.node_array.shape[1] != args.frozen_embedding_dim:
        raise ValueError(
            f"{args.dataset}: node embeddings are {dataset.node_array.shape[1]}-dimensional, "
            f"--frozen-embedding-dim says {args.frozen_embedding_dim} -- SemanticHead would "
            "silently build at the wrong width and every downstream parameter count would be wrong"
        )
    candidate_contract = validate_candidate_contract(args.baseline, dataset, args.candidate_contract_compatibility)
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
    node_embeddings = torch.from_numpy(np.array(dataset.node_array, dtype=np.float32, copy=True)).to(device)
    query_embeddings = torch.from_numpy(np.array(dataset.query_array, dtype=np.float32, copy=True)).to(device)
    dense = np.load(Path(args.data) / "dense_top200_all.npy", mmap_mode="r")
    splade = np.load(Path(args.data) / "splade_top200_all.npy", mmap_mode="r")

    families = _families(args)
    if args.a64_mainline_family not in families:
        raise ValueError(f"declared mainline family {args.a64_mainline_family!r} is not among {list(families)}")
    family_rowptr, family_col = _load_family_csr(families[args.a64_mainline_family], num_nodes)
    family_rowptr, family_col, _symmetric = _undirected(family_rowptr, family_col, num_nodes)
    budget = _a64_budget(per_seed_cap=args.per_seed_cap, neighbour_scan_cap_per_seed=args.neighbour_scan_cap_per_seed)

    golds = ragged_from_rows([view.golds for view in views])
    scored_sets, master_blocks, latencies = _m1a._cell_master_local(
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
    headroom, _present, _gold_counts = regime_headroom(ragged_from_rows(scored_sets), golds, num_nodes=num_nodes, ks=KS)
    widened = [_m1a._widen_query(query, scored) for query, scored in zip(queries, scored_sets, strict=True)]
    train_queries, held_out_queries = holdout_split(widened, args.holdout_fraction)

    if needs_m1a_headline:
        m1a_ceiling = m1a_headline["cells"][regime]["regime_headroom"].get("recall_ceiling@5")
        fresh_ceiling = headroom.get("recall_ceiling@5")
        if m1a_ceiling != fresh_ceiling:
            raise ValueError(
                f"{args.dataset}/{regime}: M1B's freshly-measured recall_ceiling@5 "
                f"({fresh_ceiling!r}) does not match M1A's recorded value ({m1a_ceiling!r}) -- "
                "the reuse premise (same code, same data, same result) has broken; do not "
                "splice any M1A row for this cell until this is resolved"
            )

    cell_result: dict[str, Any] = {
        "regime": regime,
        "regime_headroom": headroom,
        "scored_node_count": _percentiles([float(scored.size) for scored in scored_sets]),
        "uncached_feature_build_latency_ms": _percentiles(latencies),
        "train_queries": len(train_queries),
        "held_out_queries": len(held_out_queries),
        "development_seed_values": seeds,
        "arms_run": list(arms),
        "arms": {},
    }
    for arm in arms:
        store, precomputed_width = _m1a._arm_store(
            arm=arm, regime=regime, master_blocks=master_blocks, queries=widened,
            query_count=query_count, num_nodes=num_nodes,
        )
        seed_results: dict[str, Any] = {}
        per_query_by_seed: dict[str, dict[str, float]] = {}
        for seed in seeds:
            is_reuse_candidate = reuse_lookup.get((args.dataset, arm), False) and seed == 0
            args.seed = seed
            arm_result, per_query_recall_at_5 = _run_arm_with_rows(
                arm=arm, regime=regime, store=store, precomputed_width=precomputed_width,
                train_queries=train_queries, validation_queries=held_out_queries,
                node_embeddings=node_embeddings, query_embeddings=query_embeddings,
                device=device, args=args,
            )
            arm_result = dict(arm_result)
            achieved = arm_result["metrics"].get("recall@5")
            ceiling = headroom.get("recall_ceiling@5")
            arm_result["ceiling_attainment_at_5"] = (
                float(achieved) / float(ceiling) if achieved is not None and ceiling else None
            )
            arm_result["seed"] = seed
            # No longer literally spliced (amendment 5): every seed is
            # freshly fit so its per-query rows exist. A reuse_contract
            # candidate is instead cross-checked against M1A's original.
            arm_result["reused_from_m1a"] = False
            if is_reuse_candidate:
                spliced = _splice_from_m1a(m1a_headline, regime=regime, arm=arm)
                m1a_recall = spliced["metrics"].get("recall@5")
                fresh_recall = arm_result["metrics"].get("recall@5")
                delta_pp = (
                    None if m1a_recall is None or fresh_recall is None
                    else (fresh_recall - m1a_recall) * 100
                )
                within_tolerance = delta_pp is not None and abs(delta_pp) <= REUSE_CROSS_CHECK_TOLERANCE_PP
                if not within_tolerance:
                    raise ValueError(
                        f"{args.dataset}/{regime}/{arm} seed=0: freshly re-fit recall@5 "
                        f"({fresh_recall!r}) diverges from M1A's original headline recall@5 "
                        f"({m1a_recall!r}) by {delta_pp!r}pp, beyond the "
                        f"{REUSE_CROSS_CHECK_TOLERANCE_PP}pp tolerance -- the reuse audit's "
                        "identical-result premise has not reproduced; do not trust this cell "
                        "until this is resolved"
                    )
                arm_result["cross_checked_against_m1a_splice"] = {
                    "m1a_recall_at_5": m1a_recall,
                    "fresh_recall_at_5": fresh_recall,
                    "delta_pp": delta_pp,
                    "tolerance_pp": REUSE_CROSS_CHECK_TOLERANCE_PP,
                    "within_tolerance": within_tolerance,
                }
            seed_results[str(seed)] = arm_result
            per_query_by_seed[str(seed)] = per_query_recall_at_5
        cell_result["arms"][arm] = {
            "seeds": seed_results,
            "per_query_recall_at_5_by_seed": per_query_by_seed,
        }

    result: dict[str, Any] = {
        "status": COMPLETE_STATUS,
        "stage": "m1b_targeted_resolution",
        "dataset": args.dataset,
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "declaration": "configs/m1b_targeted_resolution.yaml",
        "candidate_contract": candidate_contract,
        "queries": len(views),
        "split": "validation",
        "selection": "deterministic_prefix_of_the_split_order",
        "test_split_read": False,
        "holdout_fraction": args.holdout_fraction,
        "development_seed_values": seeds,
        "semantic_rung": args.semantic_rung,
        "num_nodes": num_nodes,
        "a64_mainline_family": args.a64_mainline_family,
        "cells": {regime: cell_result},
        "systems": {"peak_process_rss_bytes": _peak_rss_bytes()},
    }
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
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--semantic-rung", default="S3", choices=("S2", "S3"))
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
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
    run(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
