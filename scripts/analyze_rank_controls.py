#!/usr/bin/env python
"""Compile the frozen P0 A1 rank controls and compare them descriptively."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "2wiki_clean",
    "musique_clean",
    "webqsp",
    "hotpotqa_clean",
    "squad_clean",
    "metaqa",
)
METHODS = ("dense", "splade", "equal_rrf", "weighted_rrf_selected")
METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_rank_result(payload: dict[str, Any], dataset: str) -> None:
    if payload.get("status") != "P0_A1_RANK_CONTROLS_COMPLETE":
        raise ValueError(f"Incomplete rank-control result for {dataset}")
    if payload.get("dataset") != dataset:
        raise ValueError(f"Rank-control dataset mismatch for {dataset}")
    if set(payload.get("test", {})) != set(METHODS):
        raise ValueError(f"Unexpected test methods for {dataset}")
    audit = payload.get("test_access_audit", {})
    if audit.get("unselected_weighted_test_results_computed") is not False:
        raise ValueError(f"Weighted-RRF test-access audit failed for {dataset}")
    selected = float(payload["selection"]["selected_dense_weight"])
    allowed_test_weights = {0.5, selected}
    if set(map(float, audit["weighted_test_weights_computed"])) != allowed_test_weights:
        raise ValueError(f"Unexpected weighted-RRF test cells for {dataset}")
    ceilings = [float(payload["test"][method]["candidate_ceiling"]) for method in METHODS]
    if max(ceilings) - min(ceilings) > 1e-12:
        raise ValueError(f"Rank-only methods changed the candidate ceiling for {dataset}")


def _confirmation_r5(payload: dict[str, Any], model: str) -> float:
    return float(payload["models"][model]["aggregate"]["test_metrics"]["recall@5"]["mean"])


def compile_analysis(rank_root: Path, confirmation_root: Path) -> dict[str, Any]:
    rows = []
    for dataset in DATASETS:
        rank = _load(rank_root / f"{dataset}.json")
        confirmation = _load(confirmation_root / f"{dataset}.json")
        _validate_rank_result(rank, dataset)
        selected = rank["test"]["weighted_rrf_selected"]
        dense_r5 = float(rank["test"]["dense"]["recall@5"])
        splade_r5 = float(rank["test"]["splade"]["recall@5"])
        learned = {
            model: _confirmation_r5(confirmation, model)
            for model in ("seed_only", "sa_mlp", "seed_aware_gnn")
        }
        rows.append(
            {
                "dataset": dataset,
                "test_queries": int(rank["split_counts"]["test"]),
                "identity_source": rank["identity_source"],
                "selected_dense_weight": float(rank["selection"]["selected_dense_weight"]),
                "selected_splade_weight": float(rank["selection"]["selected_splade_weight"]),
                "rank_controls": rank["test"],
                "selected_metrics": {metric: float(selected[metric]) for metric in METRICS},
                "candidate_ceiling": float(selected["candidate_ceiling"]),
                "candidate_available": float(selected["candidate_available"]),
                "rrf_gain_over_best_single_r5": float(selected["recall@5"])
                - max(dense_r5, splade_r5),
                "selected_minus_equal_rrf_r5": float(selected["recall@5"])
                - float(rank["test"]["equal_rrf"]["recall@5"]),
                "confirmation_r5": learned,
                "selected_rrf_minus_confirmation_r5": {
                    model: float(selected["recall@5"]) - value for model, value in learned.items()
                },
                "timing": rank["timing"],
                "source_sha256": rank["source_sha256"],
                "test_access_audit": rank["test_access_audit"],
            }
        )
    return {
        "status": "P0_A1_RANK_CONTROLS_ANALYZED",
        "datasets": rows,
        "claims": {
            "candidate_contract_unchanged": True,
            "validation_only_weight_selection": True,
            "unselected_weighted_test_cells_computed": False,
            "timing_is_service_latency": False,
            "comparison_to_learned_models_is_descriptive": True,
        },
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def _signed_pct(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def render_markdown(analysis: dict[str, Any]) -> str:
    rows = analysis["datasets"]
    lines = [
        "# P0 A1 Dense/SPLADE rank-control results",
        "",
        "Status: **complete on all six frozen datasets**.",
        "",
        (
            "Version 2 correction: MRR is computed over each method's full available ranking. "
            "Version 1 accidentally reported MRR@20; its R@K, FullCov, ceiling, selection, and "
            "rankings were unaffected."
        ),
        "",
        (
            "These are deterministic, training-free controls over the unchanged Dense-top-200 "
            "union SPLADE-top-200 candidate contract. Weighted RRF was selected by validation "
            "R@5 only. The test split was evaluated only for Dense, SPLADE, locked equal RRF, "
            "and the single selected weight."
        ),
        "",
        "## Primary R@5 result",
        "",
        "| Dataset | Dense | SPLADE | Equal RRF | Selected Dense weight | Selected RRF | Gain over best single |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        controls = row["rank_controls"]
        lines.append(
            f"| {row['dataset']} | {_pct(controls['dense']['recall@5'])} | "
            f"{_pct(controls['splade']['recall@5'])} | "
            f"{_pct(controls['equal_rrf']['recall@5'])} | "
            f"{row['selected_dense_weight']:.2f} | "
            f"{_pct(controls['weighted_rrf_selected']['recall@5'])} | "
            f"{_signed_pct(row['rrf_gain_over_best_single_r5'])} |"
        )

    lines.extend(
        [
            "",
            (
                "Validation-selected RRF improves on the stronger single ranker on five datasets "
                "and ties it on MetaQA, where validation selects Dense weight 0.0 (pure SPLADE). "
                "The selected weights vary substantially by dataset, so an equal-fusion assumption "
                "is not universally optimal."
            ),
            "",
            "## Selected rank-control metrics",
            "",
            "| Dataset | Candidate ceiling | R@1 | R@5 | R@20 | MRR | FullCov@20 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        metrics = row["selected_metrics"]
        lines.append(
            f"| {row['dataset']} | {_pct(row['candidate_ceiling'])} | "
            f"{_pct(metrics['recall@1'])} | {_pct(metrics['recall@5'])} | "
            f"{_pct(metrics['recall@20'])} | {_pct(metrics['mrr'])} | "
            f"{_pct(metrics['full_coverage@20'])} |"
        )

    lines.extend(
        [
            "",
            "## Descriptive comparison with the frozen learned models",
            "",
            (
                "The three learned columns are five-seed means from the already sealed fairness "
                "confirmation. These differences are descriptive cross-artifact comparisons, not "
                "new confirmatory significance tests."
            ),
            "",
            "| Dataset | Selected RRF | Seed-only MLP | QLS-MLP | Seed-aware GNN | RRF − QLS | RRF − GNN |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        controls = row["rank_controls"]["weighted_rrf_selected"]
        learned = row["confirmation_r5"]
        deltas = row["selected_rrf_minus_confirmation_r5"]
        lines.append(
            f"| {row['dataset']} | {_pct(controls['recall@5'])} | "
            f"{_pct(learned['seed_only'])} | {_pct(learned['sa_mlp'])} | "
            f"{_pct(learned['seed_aware_gnn'])} | {_signed_pct(deltas['sa_mlp'])} | "
            f"{_signed_pct(deltas['seed_aware_gnn'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Rank fusion is a necessary control, not the paper's replacement model. It already "
                "matches QLS-MLP R@5 within 0.08 points on 2Wiki and SQuAD, which means those two "
                "datasets cannot by themselves establish a fixed-structure mechanism. In contrast, "
                "QLS-MLP exceeds selected RRF by 4.89 points on HotpotQA, 11.04 on MuSiQue, 16.36 "
                "on MetaQA, and 23.17 on WebQSP. Those are the regimes where learned semantic "
                "interaction and/or query-local graph summaries add material value beyond rank fusion."
            ),
            "",
            (
                "The candidate ceiling is identical for every rank-only method because all methods "
                "rerank the same frozen union. These controls cannot repair missing gold nodes."
            ),
            "",
            "## Leakage and systems audit",
            "",
            (
                "- No node or query embeddings, graph edges, partitions, or model checkpoints are "
                "loaded by the rank-control evaluator."
            ),
            (
                "- MetaQA entity identity is restored from the frozen local SPLADE `id_to_idx` "
                "bijection; the sparse SPLADE matrix and graph are not used for scoring."
            ),
            (
                "- Per-query metric arrays and SHA-256 source fingerprints are retained under "
                "`outputs/p0_rank_controls/`."
            ),
            (
                "- The recorded seconds measure offline artifact evaluation, including source "
                "fingerprinting. They are not batch-1 service latency and must not be compared with "
                "QLS/GNN online latency."
            ),
            "",
            "## Next frozen boundary",
            "",
            (
                "A1 answers the semantic rank-fusion control. The next experiment must separately "
                "freeze structural-only and linear-combination controls before any of their test "
                "results are computed: PPR, distance, path/connectivity, RRF plus structure, and a "
                "linear QLS control. No completed QLS/GNN architecture may be tuned against A1."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rank-root", type=Path, default=REPO_ROOT / "outputs" / "p0_rank_controls"
    )
    parser.add_argument(
        "--confirmation-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "sa_mlp_confirmation",
    )
    parser.add_argument(
        "--json-output", type=Path, default=REPO_ROOT / "outputs" / "p0_rank_controls_analysis.json"
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPO_ROOT / "docs" / "P0_RANK_CONTROLS_RESULTS.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = compile_analysis(args.rank_root, args.confirmation_root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(analysis), encoding="utf-8")
    print(json.dumps({"status": analysis["status"], "datasets": len(analysis["datasets"])}))


if __name__ == "__main__":
    main()
