#!/usr/bin/env python
"""Rank each blocker's frozen pool by ONE semantic primitive at a time.

This exists because M2D's Stage-0 gate returned STOP_PENDING_B: conditions A
and C failed on their filed terms, and condition B -- "one missing S3 or raw
semantic primitive systematically ranks the relevant item above S4's wrong top
item on BOTH failure cells" -- had never been asked. The Stage-0 probe ranks
four whole models and emits no ranking for any single primitive, so nothing it
produced bears on B either way. A STOP resting on an unmeasured condition is
not a STOP, so B is measured here and the same gate is re-applied unchanged.

**It trains nothing and fits nothing.** Every primitive is either parameter-free
geometry on the raw 1536-dimensional embeddings or a frozen weight read out of
M2B's own S3 checkpoint. The panel is the same development portion of the same
validation split Stage 0 read, so M2B's held-out surface stays untouched, and
the candidate pool is the one M2's sealed master already scored.

What it measures, per query in the error population:

    S4 is wrong at rank 1, and a relevant candidate is in the pool. Does this
    primitive, ranking the pool ALONE, put some relevant candidate above the
    item S4 put first?

The two exclusions are counted separately because they mean opposite things. A
query whose gold was never a candidate is an ADMISSION failure and is not
evidence about ranking at all; a query S4 already gets right is not in B's
population. Both are reported, so the share below is a share of something
stated rather than of whatever was left.

Two choices are fixed here rather than after the numbers:

**Each primitive's direction is its definition's, declared in the table below.**
A similarity ranks descending and a distance ascending. Taking whichever
direction scored better would be choosing the comparison from the result.

**dot_qd_pct is measured once and its identity with the raw dot is stated.**
The within-query percentile is a monotone transform of the raw query-document
dot within one query, so as a SOLE ranker the two produce the same order. That
is a fact about the transform, not a finding, and measuring it twice would
report one number as two.

``normalized_state_dot`` is measured as a negative control and marked as one:
S4 already computes it, so it cannot satisfy B however it ranks. It is here to
show what S4's own geometry does on the queries S4 gets wrong.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.run_m1a_feature_screen as _m1a
import scripts.run_m2b_semantic_minimality as _m2b
from mp_retrieval import qls_v2_semantic, run_artifacts
from mp_retrieval.complete_data import load_complete_dataset
from mp_retrieval.data import QuerySplit
from mp_retrieval.m2c_structural_offset import rank_positions
from scripts.run_m2d_stage0_probe import (
    _baseline_block,
    _family,
    _load_model,
)

PHASE = "m2d"
ARM = "stage0_primitive_probe"
COMPLETE_STATUS = "M2D_PRIMITIVE_PROBE_COMPLETE"

#: Every primitive this probe ranks by, with the direction its definition
#: implies and whether S4 is missing it. ``missing_from_s4`` is the gate's
#: admission test for condition B and is the archaeology's finding, not this
#: file's: ``archaeology`` names the row it came from in
#: outputs/m2d_s4_semantic_repair/semantic_archaeology.json#S3_NOT_IN_S4, and a
#: test holds the two together.
PRIMITIVES: dict[str, dict[str, Any]] = {
    "cosine_qd": {
        "higher_is_better": True,
        "missing_from_s4": True,
        "parameter_free": True,
        "archaeology": "cosine_qd at raw 1536",
        "what_it_is": "the cosine between the raw 1536-dim query and candidate vectors",
    },
    "dot_qd_pct": {
        "higher_is_better": True,
        "missing_from_s4": True,
        "parameter_free": True,
        "archaeology": "dot_qd_pct",
        "what_it_is": (
            "the within-query percentile of the raw query-document dot. As a sole "
            "ranker this is the raw dot's own order, because the percentile is a "
            "monotone transform of it within one query"
        ),
    },
    "mean_abs_diff": {
        "higher_is_better": False,
        "missing_from_s4": True,
        "parameter_free": True,
        "archaeology": "mean_abs_diff at raw 1536",
        "what_it_is": "the mean absolute difference over all 1536 raw coordinates",
    },
    "semantic_product": {
        "higher_is_better": True,
        "missing_from_s4": True,
        "parameter_free": False,
        "archaeology": "semantic_product (learned full-rank diagonal)",
        "what_it_is": (
            "S3's learned full-rank diagonal bilinear form, read out of M2B's frozen "
            "S3 checkpoint and not refit"
        ),
    },
    "semantic_difference": {
        "higher_is_better": False,
        "missing_from_s4": True,
        "parameter_free": False,
        "archaeology": "semantic_difference (learned weighted L1 at raw 1536)",
        "what_it_is": (
            "S3's learned weighted L1 at raw 1536, from the same frozen checkpoint. "
            "Ranked as a distance because that is its form; the scorer's sign is a "
            "property of the whole model, not of this column alone"
        ),
    },
    "normalized_state_dot": {
        "higher_is_better": True,
        "missing_from_s4": False,
        "parameter_free": False,
        "archaeology": None,
        "what_it_is": (
            "NEGATIVE CONTROL. S4's own cosine between its projected, GELU-warped "
            "64-dim states. S4 already computes it, so it cannot satisfy condition B; "
            "it is measured to show what S4's geometry does where S4 fails"
        ),
    },
}


def _primitive_scores(
    query_vector: torch.Tensor,
    candidate_vectors: torch.Tensor,
    s3: torch.nn.Module,
    s4: torch.nn.Module,
) -> dict[str, np.ndarray]:
    """One score per candidate per primitive, all on the same pool.

    The three parameter-free columns come from qls_v2_semantic's own
    ``parameter_free_scalars`` rather than a second implementation here: a
    reimplemented cosine that differed slightly would produce a B answer about
    a quantity S2 and S3 never computed.
    """

    scalars = qls_v2_semantic.parameter_free_scalars(query_vector, candidate_vectors)
    names = list(qls_v2_semantic.PARAMETER_FREE_FEATURE_NAMES)
    scores = {name: scalars[:, names.index(name)] for name in names}

    s3_head = s3.semantic_head
    scores["semantic_product"] = (candidate_vectors * query_vector) @ s3_head.product_weight
    scores["semantic_difference"] = (
        (candidate_vectors - query_vector).abs() @ s3_head.difference_weight
    )

    # S4's own column, recomputed the way ProjectionSemanticHead.forward
    # computes it -- GELU, then normalize, then the dot -- so the control is
    # S4's quantity and not a similar one.
    s4_head = s4.semantic_head
    raw_nodes = torch.nn.functional.gelu(s4_head.node_projection(candidate_vectors))
    raw_query = torch.nn.functional.gelu(s4_head.query_projection(query_vector))
    node_state = torch.nn.functional.normalize(raw_nodes, dim=-1)
    query_state = torch.nn.functional.normalize(raw_query, dim=-1)
    scores["normalized_state_dot"] = (node_state * query_state).sum(dim=-1)

    return {
        name: value.detach().cpu().numpy().astype(np.float64).reshape(-1)
        for name, value in scores.items()
    }


def ranks_by(score: np.ndarray, pool: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Rank the pool by one primitive, in the direction its definition implies.

    A named function rather than a sign inline in the loop, because the choice
    of direction is the choice that decides what B measures: reading a distance
    descending would report the share of errors a primitive makes WORSE and
    call it a repair rate.
    """

    return rank_positions(score if higher_is_better else -score, pool)


def reorders(ranks: np.ndarray, relevant: np.ndarray, s4_top_position: int) -> bool:
    """Does this primitive alone put SOME relevant candidate above S4's top item?

    "Some" and "above" are both load-bearing. The best relevant candidate is
    what counts, because B asks whether the primitive rescues the query at all,
    not whether it recovers every relevant item; and the comparison is strict,
    because tying with S4's wrong item is not ranking above it.
    """

    return int(ranks[relevant].min()) < int(ranks[s4_top_position])


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    dataset = load_complete_dataset(args.data, dataset=args.dataset)
    if dataset.node_array is None or dataset.query_array is None:
        raise RuntimeError(f"{args.data} was opened topology-only; the raw vectors are needed")
    if len(dataset.queries) != args.expected_queries:
        raise RuntimeError(f"{args.dataset} holds {len(dataset.queries)} queries")

    candidate_contract = _m2b.validate_candidate_contract(
        _baseline_block(args.baseline), dataset, args.candidate_contract_compatibility
    )
    validation = dataset.split(QuerySplit.VALIDATION)[: args.queries]
    if len(validation) < args.queries:
        raise RuntimeError(f"validation split holds {len(validation)}, needed {args.queries}")
    if any(query.split == int(QuerySplit.TEST) for query in validation):
        raise RuntimeError("a test-split query reached the primitive panel; refusing")

    scored_sets, master_blocks, _latency, _fp, _decision = _m2b.load_cell_under_contract(
        args.cell_features, args, args.regime, candidate_contract=candidate_contract
    )
    widened = [
        _m1a._widen_query(query, scored)
        for query, scored in zip(validation, scored_sets, strict=True)
    ]
    panel, holdout = _m2b.holdout_split(widened, args.holdout_fraction)
    if args.panel_cap:
        panel = panel[: args.panel_cap]

    store, precomputed_width = _m1a._arm_store(
        arm=_m2b.RUNNER_UNIVERSAL_ARM,
        regime=args.regime,
        master_blocks=master_blocks,
        queries=widened,
        query_count=len(dataset.queries),
        num_nodes=int(dataset.num_nodes),
    )

    device = torch.device("cpu")
    node_embeddings = torch.from_numpy(np.asarray(dataset.node_array)).to(device).float()
    query_embeddings = torch.from_numpy(np.asarray(dataset.query_array)).to(device).float()
    embedding_dim = int(node_embeddings.shape[1])
    models = {
        rung: _load_model(
            checkpoint,
            rung=rung,
            precomputed_width=precomputed_width,
            embedding_dim=embedding_dim,
            dropout=args.dropout,
            temperature=args.temperature,
            device=device,
        )
        for rung, checkpoint in (("S4", args.s4_checkpoint), ("S3", args.s3_checkpoint))
    }

    reordered = dict.fromkeys(PRIMITIVES, 0)
    population = 0
    excluded = {"no_relevant_in_pool": 0, "s4_top1_already_right": 0}
    kernel_ms: list[float] = []

    with torch.no_grad():
        for query in panel:
            pool = query.candidate_index.numpy().astype(np.int64, copy=False)
            if pool.size == 0:
                continue
            relevant = np.isin(pool, query.relevant_global.numpy())
            if not relevant.any():
                excluded["no_relevant_in_pool"] += 1
                continue

            s4_scores = (
                _m2b._one_query_scores(
                    models["S4"], query, node_embeddings, query_embeddings, store, device
                )
                .cpu()
                .numpy()
                .astype(np.float64)
                .reshape(-1)
            )
            s4_ranks = rank_positions(s4_scores, pool)
            s4_top = int(np.argmin(s4_ranks))
            if relevant[s4_top]:
                excluded["s4_top1_already_right"] += 1
                continue

            population += 1
            tick = time.perf_counter()
            scores = _primitive_scores(
                query_embeddings[query.query_index],
                node_embeddings[torch.from_numpy(pool)],
                models["S3"],
                models["S4"],
            )
            kernel_ms.append((time.perf_counter() - tick) * 1000.0)

            for name, spec in PRIMITIVES.items():
                ranks = ranks_by(scores[name], pool, spec["higher_is_better"])
                if reorders(ranks, relevant, s4_top):
                    reordered[name] += 1

    result: dict[str, Any] = {
        "status": COMPLETE_STATUS,
        "stage": "m2d_primitive_probe",
        "measures": "advance_gate_condition_B",
        "dataset": args.dataset,
        "regime": args.regime,
        "cell": f"{args.dataset}/{args.regime}",
        "family": _family(args.dataset),
        "declaration": "configs/m2d_s4_semantic_repair.yaml",
        "protocol": "docs/M2D_S4_SEMANTIC_REPAIR_PROTOCOL.md",
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract": candidate_contract,
        "source_commit": args.source_commit,
        "trained_anything": False,
        "fits_anything_new": False,
        "test_split_read": False,
        "split": "validation",
        "panel": {
            "queries": len(panel),
            "portion": "development (fit) portion of the validation split",
            "holdout_left_unexamined": len(holdout),
            "holdout_fraction": args.holdout_fraction,
        },
        "question": (
            "on a query where S4 is wrong at rank 1 and a relevant candidate is in "
            "the pool, does this primitive alone rank some relevant candidate above "
            "the item S4 put first?"
        ),
        "population": population,
        "excluded": excluded,
        "primitives_measured": sorted(PRIMITIVES),
        "primitives": {
            name: {
                "share_reordered": (reordered[name] / population) if population else 0.0,
                "reordered": reordered[name],
                "population": population,
                "missing_from_s4": spec["missing_from_s4"],
                "parameter_free": spec["parameter_free"],
                "ranked": "descending" if spec["higher_is_better"] else "ascending",
                "archaeology_primitive": spec["archaeology"],
                "what_it_is": spec["what_it_is"],
            }
            for name, spec in PRIMITIVES.items()
        },
        "systems": {
            "primitive_kernel_ms_per_query_p50": (
                float(np.percentile(kernel_ms, 50)) if kernel_ms else None
            ),
            "note": (
                "the cost of computing six primitives at once on one pool, not a "
                "systems measurement of any candidate model"
            ),
            "wall_seconds": round(time.perf_counter() - started, 1),
        },
    }

    identity = run_artifacts.ArtifactIdentity(
        phase=PHASE,
        dataset=args.dataset,
        regime=args.regime,
        arm=ARM,
        source_commit=args.source_commit,
        run_id=args.run_id or run_artifacts.current_run_id(),
    )
    receipt = run_artifacts.write_artifact(
        args.output_root,
        identity,
        result,
        config_fingerprint=args.config_fingerprint,
        rows_at="primitives_measured",
    )
    result["artifact"] = receipt.as_dict()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--regime", required=True, choices=["R1", "R2", "R3"])
    parser.add_argument("--data-fingerprint-sha256", required=True)
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--queries", type=int, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate-contract-compatibility", required=True)
    parser.add_argument("--cell-features", type=Path, required=True)
    parser.add_argument("--s4-checkpoint", type=Path, required=True)
    parser.add_argument("--s3-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-fingerprint", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--panel-cap", type=int, default=0)
    # M2's frozen build key, exactly as the Stage-0 probe takes it.
    parser.add_argument("--per-seed-cap", type=int, default=16)
    parser.add_argument("--neighbour-scan-cap-per-seed", type=int, default=4096)
    parser.add_argument("--a64-mainline-family", default=_m1a.MAINLINE_FAMILY)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    print(f"{result['cell']}: {result['population']} queries in the error population")
    for name, row in sorted(result["primitives"].items()):
        flag = "" if row["missing_from_s4"] else "  (S4 already has it)"
        print(f"  {name:<22} {row['share_reordered']:.4f}{flag}")
    print(f"wrote {result['artifact']['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
