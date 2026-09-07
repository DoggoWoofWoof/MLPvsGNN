#!/usr/bin/env python
"""M2D Stage-0 probe: is anything already ranking S4's passage mistakes correctly?

Trains nothing. Fits nothing. Sweeps nothing. Reads no test split. Every number
it produces comes from four rankings over the same frozen candidate pool --
S4's, S3's, Dense's and SPLADE's -- and the only constant involved is the RRF
60 this project froze in ``configs/candidate_budget.yaml`` years of phases ago.

Three questions, all declared before this ran:

  1. **Complementarity.** S4 loses recall@1 in every passage cell. Is it losing
     where S3 wins *and* winning where S3 loses, or is it simply dominated? If
     S4-wrong/S3-right is large and the reverse is near zero there is nothing to
     fuse and no complementary signal to integrate.
  2. **Fixed fusion.** At the inherited constant and equal weights, does adding
     an existing ranking to S4 repair the blockers? Z4 (S4 + S3) is diagnostic
     only and can never be a result: running two semantic models at inference
     would spend the exact latency advantage this phase exists to protect.
  3. **Error-conditioned rescue.** On the queries S4 gets top-1 wrong, which
     signal already ranks a relevant candidate at 1, 5 or 20? That table is what
     a repair may be chosen from, and section 6 forbids choosing one first.

What is held identical across all four rankings is everything except the
ranking: the same panel, the same sealed cell master, the same scored pool, the
same relevance mask. S4's and S3's weights are M2B's own sealed checkpoints,
loaded and never refitted -- a refit here would not be the model that produced
the deltas this phase was opened for.

The panel is the development (fit) portion of each cell's validation split, the
same ``holdout_split`` division M2B made, so M2B's held-out 20% stays unexamined
and the dataset's test split is never opened.

Results are written through :mod:`mp_retrieval.run_artifacts`, to a path that
carries the phase, cell, arm, commit and run id. That is not tidiness: a write
that overwrites an existing file on the result volume is silently discarded at
commit while the container reads back its own bytes and reports success, so the
only safe rerun is one that cannot address the same path.
"""

from __future__ import annotations

import argparse
import json
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
from mp_retrieval import m2d_semantic_fusion as fusion
from mp_retrieval import run_artifacts
from mp_retrieval.complete_data import load_complete_dataset
from mp_retrieval.data import QuerySplit
from mp_retrieval.m2c_structural_offset import rank_positions

COMPLETE_STATUS = "M2D_STAGE0_COMPLETE"
PHASE = "m2d"
ARM = "stage0_diagnostic"

#: The four rankings, and which of them cover the whole pool. Dense and SPLADE
#: return a top-200 each; the pool is their stable union, so a pool member can
#: be missing from either list and is unranked BY THAT SOURCE when it is.
FULL_COVERAGE = ("S4", "S3")
PARTIAL_COVERAGE = ("Dense", "SPLADE")
RANKERS = FULL_COVERAGE + PARTIAL_COVERAGE

#: Section 5's arms. Z4 is diagnostic only and is labelled so in the output
#: itself, not only in the protocol, because the label has to travel with the
#: number into whatever reads it next.
FUSION_ARMS: dict[str, tuple[str, ...]] = {
    "Z0_S4": ("S4",),
    "Z1_S4_DENSE": ("S4", "Dense"),
    "Z2_S4_SPLADE": ("S4", "SPLADE"),
    "Z3_S4_DENSE_SPLADE": ("S4", "Dense", "SPLADE"),
    "Z4_S4_S3_DIAGNOSTIC_ONLY": ("S4", "S3"),
}

#: Context arms, section 5's "also if cheap from stored ranks". They say
#: whether the incumbent is improvable by the same parameter-free move, which
#: is the only way to tell an S4-specific gain from one any model would get.
CONTEXT_ARMS: dict[str, tuple[str, ...]] = {
    "C0_S3": ("S3",),
    "C1_S3_DENSE": ("S3", "Dense"),
    "C2_S3_SPLADE": ("S3", "SPLADE"),
}

DIAGNOSTIC_ONLY_ARMS = ("Z4_S4_S3_DIAGNOSTIC_ONLY",)


def _load_model(
    checkpoint: Path,
    *,
    rung: str,
    precomputed_width: int,
    embedding_dim: int,
    dropout: float,
    temperature: float,
    device: torch.device,
) -> torch.nn.Module:
    """One of M2B's sealed fits, loaded strictly.

    ``strict=True`` is the load-bearing argument. A checkpoint whose semantic
    branch does not match the rung being built would otherwise load with the
    mismatched tensors silently dropped, and the probe would compare S4 against
    a partially randomised S3 while reporting it as M2B's.
    """

    model = _m1a.build_m1a_model(
        precomputed_width=precomputed_width,
        semantic_rung=rung,
        dropout=dropout,
        temperature=temperature,
        embedding_dim=embedding_dim,
        semantic_head=_m2b.build_semantic_head(rung, embedding_dim),
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    model.eval()
    return model


#: The passage-graph family, exactly as the declaration's
#: family_split_seed_0_mean_s4_minus_s3_pp.definition names it. Everything
#: else in the frozen universe is KB. Kept as one constant, and checked against
#: the declared text by test, so the split cannot drift between the probe that
#: reports it and the record that prices it.
PASSAGE_FAMILY = frozenset(
    {"2wiki_clean", "hotpotqa_clean", "musique_clean", "squad_clean"}
)


def _family(dataset: str) -> str:
    """passage or kb, by the split M2C measured and M2D inherited."""

    return "passage" if dataset in PASSAGE_FAMILY else "kb"


def _summarise_complementarity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Section 4's counts, plus the ratio the declaration says it turns on."""

    total = len(rows)
    with_gold = [row for row in rows if row["has_relevant_in_pool"]]
    counted = {
        name: int(sum(1 for row in rows if row[name]))
        for name in (
            "s4_right_at_1",
            "s3_right_at_1",
            "s4_wrong_s3_right",
            "s3_wrong_s4_right",
            "both_right_at_1",
            "neither_right_at_1",
            "same_top1_node",
        )
    }
    cross_s3_under_s4 = [
        row["s3_best_relevant_rank_under_s4"] for row in with_gold
    ]
    cross_s4_under_s3 = [
        row["s4_best_relevant_rank_under_s3"] for row in with_gold
    ]

    def distribution(values: list[int]) -> dict[str, float] | None:
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90)),
            "at_1": float((array <= 1).mean()),
            "at_5": float((array <= 5).mean()),
            "at_20": float((array <= 20).mean()),
        }

    disagreements = counted["s4_wrong_s3_right"] + counted["s3_wrong_s4_right"]
    return {
        "queries": total,
        "queries_with_a_relevant_candidate_in_the_pool": len(with_gold),
        "counts": counted,
        "shares": {name: (count / total if total else 0.0) for name, count in counted.items()},
        "disagreement_at_1": {
            "total": disagreements,
            "share_of_queries": disagreements / total if total else 0.0,
            "s4_wrong_s3_right_share_of_disagreements": (
                counted["s4_wrong_s3_right"] / disagreements if disagreements else 0.0
            ),
            "reading": (
                "one-sided if S4-wrong/S3-right dominates: S4 is then dominated on this "
                "cell and there is nothing to fuse. Two-sided means the representations "
                "trade errors."
            ),
        },
        "cross_ranks": {
            "s3_best_relevant_rank_under_s4": distribution(cross_s3_under_s4),
            "s4_best_relevant_rank_under_s3": distribution(cross_s4_under_s3),
            "meaning": (
                "where the OTHER model put the relevant candidate this model found "
                "first. Low values on both sides mean the two orderings differ near the "
                "top without either losing the candidate."
            ),
        },
        "top_k_overlap": {
            str(k): {
                "mean": float(np.mean([row["overlap"][k] for row in rows])) if rows else 0.0,
                "mean_share": (
                    float(np.mean([row["overlap"][k] for row in rows])) / k if rows else 0.0
                ),
            }
            for k in fusion.KS
        },
        "same_best_relevant_node_share": (
            float(np.mean([bool(row["same_best_relevant_node"]) for row in with_gold]))
            if with_gold
            else 0.0
        ),
        "both_found_same_relevant_ordered_differently_share": (
            float(
                np.mean(
                    [bool(row["both_found_same_relevant_ordered_differently"]) for row in with_gold]
                )
            )
            if with_gold
            else 0.0
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    dataset = load_complete_dataset(args.data, dataset=args.dataset)
    if dataset.node_array is None or dataset.query_array is None:
        raise ValueError(
            f"{args.data} was opened topology-only. Both semantic branches are "
            "functions of embeddings, so this probe cannot run without them."
        )
    if len(dataset.queries) != args.expected_queries:
        raise ValueError("Complete dataset query count differs from the registered protocol")

    candidate_contract = _m2b.validate_candidate_contract(
        _baseline_block(args.baseline), dataset, args.candidate_contract_compatibility
    )
    validation = dataset.split(QuerySplit.VALIDATION)[: args.queries]
    if len(validation) < args.queries:
        raise ValueError(
            f"validation split holds {len(validation)} queries, needed {args.queries}"
        )
    if any(query.split == int(QuerySplit.TEST) for query in validation):
        raise RuntimeError("a test-split query reached the Stage-0 panel; refusing")

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
        "S4": _load_model(
            args.s4_checkpoint,
            rung="S4",
            precomputed_width=precomputed_width,
            embedding_dim=embedding_dim,
            dropout=args.dropout,
            temperature=args.temperature,
            device=device,
        ),
        "S3": _load_model(
            args.s3_checkpoint,
            rung="S3",
            precomputed_width=precomputed_width,
            embedding_dim=embedding_dim,
            dropout=args.dropout,
            temperature=args.temperature,
            device=device,
        ),
    }

    # The stored retrieval lists, reopened from the frozen payload rather than
    # reconstructed. The pool IS their stable union -- complete_data builds it
    # that way -- so every pool member is ranked by at least one of them, and
    # asserting that below is a real check on the two staying aligned.
    dense_lists = np.load(args.data / "dense_top200_all.npy", mmap_mode="r")
    splade_lists = np.load(args.data / "splade_top200_all.npy", mmap_mode="r")

    per_ranker: dict[str, list[dict[str, float]]] = {name: [] for name in RANKERS}
    per_arm: dict[str, list[dict[str, float]]] = {
        name: [] for name in (*FUSION_ARMS, *CONTEXT_ARMS)
    }
    complementarity_rows: list[dict[str, Any]] = []
    rescue_rows: list[dict[str, Any]] = []
    excluded = {"no_relevant_in_pool": 0, "s4_top1_already_right": 0}
    source_coverage = {name: [] for name in PARTIAL_COVERAGE}
    kernel_ms: list[float] = []

    with torch.no_grad():
        for query in panel:
            pool = query.candidate_index.numpy().astype(np.int64, copy=False)
            if pool.size == 0:
                continue
            relevant = np.isin(pool, query.relevant_global.numpy())

            tick = time.perf_counter()
            ranks: dict[str, np.ndarray] = {}
            contribution: dict[str, np.ndarray] = {}
            for name, model in models.items():
                scores = (
                    _m2b._one_query_scores(
                        model, query, node_embeddings, query_embeddings, store, device
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                    .reshape(-1)
                )
                ranks[name] = rank_positions(scores, pool)
                contribution[name] = fusion.full_coverage_contribution(ranks[name])

            for name, lists in (("Dense", dense_lists), ("SPLADE", splade_lists)):
                stored = np.asarray(lists[query.query_index], dtype=np.int64)
                contribution[name] = fusion.source_contribution(pool, stored)
                ranks[name] = fusion.unranked_last(contribution[name], pool)
                source_coverage[name].append(
                    float(np.count_nonzero(contribution[name])) / pool.size
                )
            kernel_ms.append((time.perf_counter() - tick) * 1000.0)

            if not np.any(contribution["Dense"] + contribution["SPLADE"]):
                raise RuntimeError(
                    f"query {query.query_id!r} has a scored pool no stored source ranks. "
                    "The pool is built as the union of the two lists, so this means the "
                    "sealed master and the frozen payload disagree."
                )

            for name in RANKERS:
                per_ranker[name].append(fusion.metrics(ranks[name], relevant))
            for arm, sources in (*FUSION_ARMS.items(), *CONTEXT_ARMS.items()):
                fused = fusion.fuse_ranks([contribution[name] for name in sources], pool)
                per_arm[arm].append(fusion.metrics(fused, relevant))

            complementarity_rows.append(
                fusion.complementarity(ranks["S4"], ranks["S3"], relevant, pool)
            )
            rescue = fusion.rescue_row(
                ranks["S4"],
                {name: ranks[name] for name in ("Dense", "SPLADE", "S3")},
                relevant,
            )
            if rescue is None:
                if not relevant.any():
                    excluded["no_relevant_in_pool"] += 1
                else:
                    excluded["s4_top1_already_right"] += 1
            else:
                rescue_rows.append(rescue)

    result: dict[str, Any] = {
        "status": COMPLETE_STATUS,
        "stage": "m2d_stage0_probe",
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
            "why": (
                "M2B divided each cell's validation split 80/20 and reported on the "
                "20%. Stage 0 reads only the 80% that fitting already spent, so the "
                "surface M2B's filed numbers come from stays clean."
            ),
        },
        "fusion": {
            "method": "reciprocal_rank_fusion",
            "constant": fusion.RRF_CONSTANT,
            "constant_status": "REUSED_NOT_NEW",
            "constant_source": "configs/candidate_budget.yaml#candidate_contract.rrf_constant",
            "weights": "equal",
            "swept": False,
            "unranked_candidate_rule": (
                "a pool member absent from a source's stored top-200 is unranked by "
                "that source and contributes exactly zero, the same treatment "
                "linear_control.rank_feature_rows gives it"
            ),
        },
        "rankers": {name: fusion.mean_metrics(rows) for name, rows in per_ranker.items()},
        "comparability": (
            "These absolute numbers are on the FIT portion of the validation split. "
            "M2B's filed table reports the HOLDOUT portion, so a row here is not "
            "comparable with a row there and neither is the S4-minus-S3 delta. What "
            "the advance gate compares is arm against Z0 (S4) WITHIN this panel, "
            "which is why the gate is written that way."
        ),
        "arms": {
            name: {
                **fusion.mean_metrics(rows),
                "sources": list({**FUSION_ARMS, **CONTEXT_ARMS}[name]),
                "eligible_as_a_final_model": name not in DIAGNOSTIC_ONLY_ARMS
                and name not in CONTEXT_ARMS,
                "role": (
                    "diagnostic only: running two semantic models would defeat the "
                    "systems object this phase exists to protect"
                    if name in DIAGNOSTIC_ONLY_ARMS
                    else "context: whether the incumbent is improvable by the same "
                    "parameter-free move"
                    if name in CONTEXT_ARMS
                    else "candidate"
                ),
            }
            for name, rows in per_arm.items()
        },
        "deltas_vs_s4_pp": {
            name: {
                metric: (
                    fusion.mean_metrics(rows)[metric]
                    - fusion.mean_metrics(per_arm["Z0_S4"])[metric]
                )
                * 100.0
                for metric in (*(f"recall@{k}" for k in fusion.KS), "mrr")
            }
            for name, rows in per_arm.items()
        },
        "complementarity": _summarise_complementarity(complementarity_rows),
        "rescue": fusion.rescue_table(
            rescue_rows, names=("Dense", "SPLADE", "S3"), excluded=excluded
        ),
        "source_coverage_of_the_pool": {
            name: {
                "mean": float(np.mean(values)) if values else 0.0,
                "median": float(np.median(values)) if values else 0.0,
            }
            for name, values in source_coverage.items()
        },
        "systems": {
            "ranking_kernel_ms_per_query_p50": (
                float(np.percentile(kernel_ms, 50)) if kernel_ms else None
            ),
            "ranking_kernel_ms_per_query_p95": (
                float(np.percentile(kernel_ms, 95)) if kernel_ms else None
            ),
            "note": (
                "this is the probe's own cost of producing four rankings, not a "
                "systems measurement of any candidate model. Section 13's gate is "
                "measured by the systems harness, not here."
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
        rows_at="rankers.S4",
    )
    result["artifact"] = receipt.as_dict()
    return result


def _baseline_block(baseline: Any) -> dict[str, Any]:
    """M2C's own accessor, reused rather than reimplemented."""

    from scripts.run_m2c_stage0_probe import _baseline_block as m2c_baseline_block

    return m2c_baseline_block(baseline)


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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    print(f"{result['cell']}: {result['panel']['queries']} queries")
    for name in RANKERS:
        row = result["rankers"][name]
        print(
            f"  {name:<7} R@1 {row['recall@1']:.4f}  R@5 {row['recall@5']:.4f}  "
            f"MRR {row['mrr']:.4f}"
        )
    print(f"wrote {result['artifact']['path']}")
    print(json.dumps(result["rescue"]["by_cutoff"][1], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
