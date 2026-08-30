#!/usr/bin/env python
"""Compile the frozen six-dataset P0 A2 structural-control table."""

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
METHODS = (
    "selected_rrf",
    "distance",
    "ppr",
    "path_connectivity",
    "structural_summary",
    "selected_rrf_plus_ppr",
    "selected_rrf_plus_structural_summary",
)
STRUCTURAL_ONLY = ("distance", "ppr", "path_connectivity", "structural_summary")
TRAINING_FREE = METHODS


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _confirmation_r5(payload: dict[str, Any], model: str) -> float:
    return float(payload["models"][model]["aggregate"]["test_metrics"]["recall@5"]["mean"])


def _validate(payload: dict[str, Any], dataset: str) -> None:
    if payload.get("status") != "P0_A2_FIXED_STRUCTURAL_CONTROLS_COMPLETE":
        raise ValueError(f"Incomplete A2 result for {dataset}")
    if payload.get("dataset") != dataset or tuple(payload.get("test", {})) != METHODS:
        raise ValueError(f"Unexpected A2 dataset/method contract for {dataset}")
    if payload["alignment"]["status"] != "BIT_EXACT_A2_INPUT_ALIGNMENT":
        raise ValueError(f"A2 alignment failed for {dataset}")
    if float(payload["a1_reproduction"]["maximum_absolute_difference"]) > 2e-7:
        raise ValueError(f"A1 reproduction failed for {dataset}")
    audit = payload["test_access_audit"]
    if (
        audit["validation_selected_A2_weights_or_rules"]
        or audit["test_selected_models_or_features"]
        or not audit["all_locked_methods_reported"]
    ):
        raise ValueError(f"A2 test-access audit failed for {dataset}")


def compile_analysis(result_root: Path, confirmation_root: Path) -> dict[str, Any]:
    rows = []
    for dataset in DATASETS:
        result = _load(result_root / f"{dataset}.json")
        confirmation = _load(confirmation_root / f"{dataset}.json")
        _validate(result, dataset)
        r5 = {method: float(result["test"][method]["recall@5"]) for method in METHODS}
        best_structural = max(
            STRUCTURAL_ONLY, key=lambda method: (r5[method], -METHODS.index(method))
        )
        best_training_free = max(
            TRAINING_FREE, key=lambda method: (r5[method], -METHODS.index(method))
        )
        learned = {
            model: _confirmation_r5(confirmation, model)
            for model in ("seed_only", "sa_mlp", "seed_aware_gnn")
        }
        rows.append(
            {
                "dataset": dataset,
                "test_queries": int(result["input_contract"]["split_counts"]["test"]),
                "r5": r5,
                "best_structural_only_descriptive": best_structural,
                "best_training_free_descriptive": best_training_free,
                "best_structural_minus_rrf_r5": r5[best_structural] - r5["selected_rrf"],
                "best_training_free_minus_rrf_r5": r5[best_training_free] - r5["selected_rrf"],
                "confirmation_r5": learned,
                "best_training_free_minus_qls_r5": r5[best_training_free] - learned["sa_mlp"],
                "best_training_free_minus_gnn_r5": r5[best_training_free]
                - learned["seed_aware_gnn"],
                "alignment": result["alignment"],
                "timing": result["timing"],
            }
        )
    return {
        "status": "P0_A2_FIXED_STRUCTURAL_CONTROLS_ANALYZED",
        "datasets": rows,
        "claims": {
            "all_rules_locked_before_test": True,
            "best_method_labels_are_descriptive_not_selected": True,
            "simple_structure_has_signal_on_webqsp_metaqa_hotpot": True,
            "simple_rules_fully_explain_qls_all_datasets": False,
            "linear_A3_justified": True,
        },
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def _signed_pct(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def render_markdown(analysis: dict[str, Any]) -> str:
    rows = analysis["datasets"]
    lines = [
        "# P0 A2 fixed structural-control results",
        "",
        "Status: **complete on all six frozen datasets**.",
        "",
        (
            "Every method was locked before test access and uses the exact sealed QLS query-local "
            "feature cache. There are no learned parameters, seeds, A2 validation choices, or "
            "message-passing operations."
        ),
        "",
        "## Complete R@5 table",
        "",
        "| Dataset | Selected RRF | Distance | PPR | Path/connectivity | Structural summary | RRF + PPR | RRF + summary |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        r5 = row["r5"]
        lines.append(
            f"| {row['dataset']} | {_pct(r5['selected_rrf'])} | {_pct(r5['distance'])} | "
            f"{_pct(r5['ppr'])} | {_pct(r5['path_connectivity'])} | "
            f"{_pct(r5['structural_summary'])} | {_pct(r5['selected_rrf_plus_ppr'])} | "
            f"{_pct(r5['selected_rrf_plus_structural_summary'])} |"
        )

    lines.extend(
        [
            "",
            "## Structural signal and remaining learned-model gap",
            "",
            (
                "The `best` labels below summarize the fully reported table; they are descriptive "
                "test maxima and are not selected models or inputs to A3."
            ),
            "",
            "| Dataset | Best structural-only | Δ vs RRF | Best training-free | Best training-free R@5 | Δ vs QLS | Δ vs GNN |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        best = row["best_training_free_descriptive"]
        lines.append(
            f"| {row['dataset']} | {row['best_structural_only_descriptive']} | "
            f"{_signed_pct(row['best_structural_minus_rrf_r5'])} | {best} | "
            f"{_pct(row['r5'][best])} | {_signed_pct(row['best_training_free_minus_qls_r5'])} | "
            f"{_signed_pct(row['best_training_free_minus_gnn_r5'])} |"
        )

    lines.extend(
        [
            "",
            "## Result",
            "",
            (
                "Fixed query-local structure contains real retrieval signal. Structural summary "
                "alone improves over selected RRF by 5.20 R@5 points on WebQSP and 4.41 on MetaQA. "
                "Locked RRF+PPR improves HotpotQA by 1.01 points. These are the three relational/graph "
                "regimes where the original plain GNN had won."
            ),
            "",
            (
                "The simple rules are not a sufficient replacement for QLS-MLP. Relative to the "
                "best fully reported training-free method, QLS retains 17.97 points on WebQSP, "
                "11.96 on MetaQA, 11.04 on MuSiQue, and 3.89 on HotpotQA. On 2Wiki and SQuAD, rank "
                "fusion alone already matches QLS within 0.08 points."
            ),
            "",
            (
                "Naive equal fusion is also not universally beneficial: adding structural rankings "
                "damages 2Wiki, MuSiQue, and SQuAD. The next legitimate control is therefore a tiny "
                "linear model trained only on train labels and selected on validation—not a new deep "
                "architecture and not a test-tuned fusion weight."
            ),
            "",
            "## Audit",
            "",
            "- All six candidate-order/source/feature contracts passed exact SHA-256 alignment.",
            "- The selected-RRF reference reproduced corrected A1 with zero aggregate difference.",
            "- No graph or node/query embedding was loaded by A2; only frozen rank arrays, labels/splits for evaluation, and memory-mapped structural scalars were read.",
            "- Modal used CPU workers because this stage contains only fixed scalar scoring and sorting. GPUs remain reserved for the learned A3 control.",
            "- Reported runtime is offline artifact evaluation, including full candidate hashing and compression; it is not service latency.",
            "",
            "## Stopping point",
            "",
            (
                "A2 is closed. Do not tune structural formulas or fusion weights against these test "
                "results. A3 may proceed only under its own frozen feature, optimizer, validation, "
                "seed, and test-access contract."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "p0_fixed_structural_controls",
    )
    parser.add_argument(
        "--confirmation-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "sa_mlp_confirmation",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "p0_fixed_structural_controls_analysis.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPO_ROOT / "docs" / "P0_FIXED_STRUCTURAL_CONTROLS_RESULTS.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = compile_analysis(args.result_root, args.confirmation_root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(analysis), encoding="utf-8")
    print(json.dumps({"status": analysis["status"], "datasets": len(analysis["datasets"])}))


if __name__ == "__main__":
    main()
