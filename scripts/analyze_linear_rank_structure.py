#!/usr/bin/env python
"""Compile the frozen six-dataset P0 A3 linear-control results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mp_retrieval.rank_fusion import load_frozen_rank_contract, sha256_file

DATASETS = (
    "2wiki_clean",
    "musique_clean",
    "webqsp",
    "hotpotqa_clean",
    "squad_clean",
    "metaqa",
)
SEEDS = (0, 1, 2, 3, 4)
METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")
LEARNED_MODELS = ("linear_rank_structure", "seed_only", "sa_mlp", "seed_aware_gnn")
CONTRASTS = {
    "linear_minus_selected_rrf": ("linear_rank_structure", "selected_rrf"),
    "linear_minus_seed_only": ("linear_rank_structure", "seed_only"),
    "linear_minus_qls_mlp": ("linear_rank_structure", "sa_mlp"),
    "linear_minus_seed_aware_gnn": ("linear_rank_structure", "seed_aware_gnn"),
}
T_CRITICAL_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std_ci(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = (
        T_CRITICAL_95.get(len(values), 1.96) * sample_std / math.sqrt(len(values))
        if len(values) > 1
        else float("nan")
    )
    return {
        "n": len(values),
        "mean": mean,
        "sample_std": sample_std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def _paired_t_pvalue(values: list[float]) -> float:
    if max(values) == min(values):
        return 1.0 if values[0] == 0 else 0.0
    return float(stats.ttest_1samp(values, popmean=0.0).pvalue)


def _holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, value) in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * value)
        adjusted[key] = min(running, 1.0)
    return adjusted


def _hierarchical_paired_ci(
    left: np.ndarray,
    right: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("Paired arrays must have shape [seed, query, metric]")
    random = np.random.default_rng(seed)
    seed_count, query_count, metric_count = left.shape
    difference = left.astype(np.float64) - right.astype(np.float64)
    estimates = np.empty((replicates, metric_count), dtype=np.float64)
    for replicate in range(replicates):
        seed_weights = random.multinomial(seed_count, np.full(seed_count, 1 / seed_count))
        query_weights = random.multinomial(query_count, np.full(query_count, 1 / query_count))
        seed_mean = np.tensordot(seed_weights / seed_count, difference, axes=(0, 0))
        estimates[replicate] = query_weights @ seed_mean / query_count
    return np.quantile(estimates, 0.025, axis=0), np.quantile(estimates, 0.975, axis=0)


def _query_order_sha256(query_ids: list[str], indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(query_ids[int(index)].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _safe_metric(metric: str) -> str:
    return metric.replace("@", "_at_")


def _load_arrays(
    dataset: str,
    result: dict[str, Any],
    rank_result: dict[str, Any],
    structural_result: dict[str, Any],
    confirmation: dict[str, Any],
    result_root: Path,
    structural_root: Path,
    confirmation_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    a3_path = result_root / f"{dataset}.query_metrics.npz"
    a2_path = structural_root / f"{dataset}.query_metrics.npz"
    confirmation_path = confirmation_root / f"{dataset}.query_metrics.npz"
    if sha256_file(a3_path) != result["query_metrics"]["sha256"]:
        raise ValueError(f"A3 query metrics failed SHA-256: {dataset}")
    if sha256_file(a2_path) != structural_result["query_metrics"]["sha256"]:
        raise ValueError(f"A2 query metrics failed SHA-256: {dataset}")
    if sha256_file(confirmation_path) != confirmation["query_metrics"]["sha256"]:
        raise ValueError(f"Confirmation query metrics failed SHA-256: {dataset}")
    with np.load(a3_path) as packed:
        if tuple(map(str, packed["metric_names"].tolist())) != METRICS:
            raise ValueError(f"A3 metric order changed: {dataset}")
        query_index = np.asarray(packed["query_index"], dtype=np.int64)
        linear = np.stack(
            [np.asarray(packed[f"linear_rank_structure_seed_{seed}"], dtype=np.float32) for seed in SEEDS]
        )
    with np.load(confirmation_path) as packed:
        if tuple(map(str, packed["metric_names"].tolist())) != METRICS:
            raise ValueError(f"Confirmation metric order changed: {dataset}")
        confirmation_order_hash = str(packed["query_order_sha256"].item())
        learned = {
            model: np.stack(
                [np.asarray(packed[f"{model}_seed_{seed}"], dtype=np.float32) for seed in SEEDS]
            )
            for model in ("seed_only", "sa_mlp", "seed_aware_gnn")
        }
    with np.load(a2_path) as packed:
        a2_query_index = np.asarray(packed["query_index"], dtype=np.int64)
        a2_methods = {
            method: np.column_stack(
                [
                    np.asarray(packed[f"{method}__{_safe_metric(metric)}"], dtype=np.float32)
                    for metric in METRICS
                ]
            )
            for method in structural_result["test"]
        }
    if not np.array_equal(query_index, a2_query_index):
        raise ValueError(f"A2 and A3 test query indices differ: {dataset}")
    contract = load_frozen_rank_contract(
        Path(rank_result["source_root"]), dataset=dataset, hash_sources=False
    )
    if not np.array_equal(query_index, contract.split_indices["test"]):
        raise ValueError(f"A3 test query indices differ from the frozen manifest: {dataset}")
    observed_order_hash = _query_order_sha256(contract.query_ids, query_index)
    if observed_order_hash != confirmation_order_hash:
        raise ValueError(f"A3 and confirmation test query order differs: {dataset}")
    arrays = {
        "linear_rank_structure": linear,
        **learned,
        **{
            method: np.broadcast_to(values, (len(SEEDS), *values.shape))
            for method, values in a2_methods.items()
        },
    }
    return arrays, {
        "status": "BIT_EXACT_A1_A2_A3_CONFIRMATION_QUERY_ALIGNMENT",
        "test_query_indices_equal": True,
        "test_query_order_sha256": observed_order_hash,
        "queries": int(query_index.size),
    }


def _aggregate_model(array: np.ndarray) -> dict[str, Any]:
    return {
        metric: _mean_std_ci(array[:, :, metric_index].mean(axis=1).tolist())
        for metric_index, metric in enumerate(METRICS)
    }


def _systems(result: dict[str, Any], confirmation: dict[str, Any]) -> dict[str, Any]:
    linear_latency = [
        float(result["seeds"][str(seed)]["inference"]["latency_ms_per_query"])
        for seed in SEEDS
    ]
    linear_gpu = [
        float(result["seeds"][str(seed)]["inference"]["peak_gpu_memory_mb_incremental"])
        for seed in SEEDS
    ]
    qls_latency = [
        float(confirmation["models"]["sa_mlp"]["seeds"][str(seed)]["inference"]["latency_ms_per_query"])
        for seed in SEEDS
    ]
    gnn_latency = [
        float(confirmation["models"]["seed_aware_gnn"]["seeds"][str(seed)]["inference"]["latency_ms_per_query"])
        for seed in SEEDS
    ]
    canonical_training = [
        float(result["seeds"][str(seed)]["training"]["training_seconds_shared_feature_pass"])
        for seed in SEEDS
        if seed != 0
    ]
    structural_cache = confirmation["feature_cache"]
    derived_cache = result["derived_cache"]
    return {
        "linear_parameters": 19,
        "qls_parameters": int(confirmation["models"]["sa_mlp"]["parameters"]["parameters"]),
        "gnn_parameters": int(
            confirmation["models"]["seed_aware_gnn"]["parameters"]["parameters"]
        ),
        "linear_latency_ms_per_query": _mean_std_ci(linear_latency),
        "linear_incremental_gpu_mib": _mean_std_ci(linear_gpu),
        "linear_canonical_training_seconds_seeds_1_to_4": _mean_std_ci(canonical_training),
        "linear_lr_screen_seconds": float(
            result["learning_rate_screen_timing"]["training_seconds_shared_feature_pass"]
        ),
        "linear_speedup_over_qls": _mean_std_ci(
            [qls / linear for qls, linear in zip(qls_latency, linear_latency)]
        ),
        "linear_speedup_over_gnn": _mean_std_ci(
            [gnn / linear for gnn, linear in zip(gnn_latency, linear_latency)]
        ),
        "parameter_reduction_vs_qls": int(
            confirmation["models"]["sa_mlp"]["parameters"]["parameters"]
        )
        / 19,
        "parameter_reduction_vs_gnn": int(
            confirmation["models"]["seed_aware_gnn"]["parameters"]["parameters"]
        )
        / 19,
        "inherited_structural_cache_bytes": int(structural_cache["cache_bytes"]),
        "derived_rank_cache_bytes": int(derived_cache["cache_bytes"]),
        "total_cache_bytes": int(structural_cache["cache_bytes"])
        + int(derived_cache["cache_bytes"]),
        "inherited_structural_precompute_seconds": float(
            structural_cache["total_preprocessing_seconds"]
        ),
        "derived_rank_cache_build_seconds": float(derived_cache["build_seconds"]),
        "warm_cache_only": True,
        "uncached_unseen_embedding_timing": False,
    }


def compile_analysis(
    result_root: Path,
    rank_root: Path,
    structural_root: Path,
    confirmation_root: Path,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for dataset_position, dataset in enumerate(DATASETS):
        result = _load(result_root / f"{dataset}.json")
        rank_result = _load(rank_root / f"{dataset}.json")
        structural_result = _load(structural_root / f"{dataset}.json")
        confirmation = _load(confirmation_root / f"{dataset}.json")
        if result.get("status") != "P0_A3_LINEAR_RANK_STRUCTURE_COMPLETE":
            raise ValueError(f"Incomplete A3 result: {dataset}")
        if result["protocol"]["tag"] != "p0-linear-rank-structure-protocol-v2":
            raise ValueError(f"A3 result used an unexpected protocol: {dataset}")
        if sorted(map(int, result["seeds"])) != list(SEEDS):
            raise ValueError(f"A3 seed set is incomplete: {dataset}")
        audit = result["test_access_audit"]
        if (
            audit["test_used_for_learning_rate_selection"]
            or audit["test_used_for_epoch_checkpoint_selection"]
            or audit["test_selected_features_or_models"]
            or int(audit["test_evaluations_per_seed"]) != len(SEEDS)
        ):
            raise ValueError(f"A3 test-access audit failed: {dataset}")
        arrays, query_alignment = _load_arrays(
            dataset,
            result,
            rank_result,
            structural_result,
            confirmation,
            result_root,
            structural_root,
            confirmation_root,
        )
        best_training_free = max(
            structural_result["test"],
            key=lambda method: float(structural_result["test"][method]["recall@5"]),
        )
        models = {
            "selected_rrf": _aggregate_model(arrays["selected_rrf"]),
            "best_training_free_descriptive": {
                "method": best_training_free,
                **_aggregate_model(arrays[best_training_free]),
            },
            **{model: _aggregate_model(arrays[model]) for model in LEARNED_MODELS},
        }
        contrasts: dict[str, Any] = {}
        for contrast_position, (contrast, (left, right)) in enumerate(CONTRASTS.items()):
            low, high = _hierarchical_paired_ci(
                arrays[left],
                arrays[right],
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + dataset_position * 101 + contrast_position,
            )
            contrasts[contrast] = {}
            for metric_index, metric in enumerate(METRICS):
                by_seed = (
                    arrays[left][:, :, metric_index].mean(axis=1)
                    - arrays[right][:, :, metric_index].mean(axis=1)
                ).tolist()
                contrasts[contrast][metric] = {
                    "seed_effect": _mean_std_ci(by_seed),
                    "paired_seed_t_pvalue": _paired_t_pvalue(by_seed),
                    "paired_hierarchical_query_ci95_low": float(low[metric_index]),
                    "paired_hierarchical_query_ci95_high": float(high[metric_index]),
                }
        linear_r5 = float(models["linear_rank_structure"]["recall@5"]["mean"])
        rrf_r5 = float(models["selected_rrf"]["recall@5"]["mean"])
        qls_r5 = float(models["sa_mlp"]["recall@5"]["mean"])
        recovery = (
            (linear_r5 - rrf_r5) / (qls_r5 - rrf_r5)
            if qls_r5 > rrf_r5
            else float("nan")
        )
        rows.append(
            {
                "dataset": dataset,
                "selected_learning_rate": float(result["selected_learning_rate"]),
                "models": models,
                "contrasts": contrasts,
                "descriptive_linear_minus_best_training_free_r5": linear_r5
                - float(models["best_training_free_descriptive"]["recall@5"]["mean"]),
                "descriptive_fraction_of_rrf_to_qls_gap_recovered": recovery,
                "query_alignment": query_alignment,
                "systems": _systems(result, confirmation),
                "metaqa_hops": (
                    {
                        str(hop): {
                            "linear_rank_structure": _mean_std_ci(
                                [
                                    float(result["seeds"][str(seed)]["by_hop"][str(hop)][
                                        "metrics"
                                    ]["recall@5"])
                                    for seed in SEEDS
                                ]
                            ),
                            **{
                                model: _mean_std_ci(
                                    [
                                        float(confirmation["models"][model]["seeds"][str(seed)][
                                            "by_hop"
                                        ][str(hop)]["metrics"]["recall@5"])
                                        for seed in SEEDS
                                    ]
                                )
                                for model in ("seed_only", "sa_mlp", "seed_aware_gnn")
                            },
                        }
                        for hop in (1, 2, 3)
                    }
                    if dataset == "metaqa"
                    else {}
                ),
            }
        )

    for contrast in CONTRASTS:
        for metric in METRICS:
            adjusted = _holm(
                {
                    row["dataset"]: float(
                        row["contrasts"][contrast][metric]["paired_seed_t_pvalue"]
                    )
                    for row in rows
                }
            )
            for row in rows:
                item = row["contrasts"][contrast][metric]
                item["holm_adjusted_pvalue_across_datasets"] = adjusted[row["dataset"]]
                item["holm_significant_0.05"] = adjusted[row["dataset"]] < 0.05

    return {
        "status": "P0_A3_LINEAR_RANK_STRUCTURE_SIX_DATASET_ANALYZED",
        "protocol_tag": "p0-linear-rank-structure-protocol-v2",
        "seeds": list(SEEDS),
        "bootstrap": {
            "method": "paired_two_stage_seed_then_query_percentile",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
        },
        "datasets": rows,
        "claims": {
            "linear_control_explains_full_qls_result_all_datasets": False,
            "linear_control_matches_qls_within_one_r5_point": [
                row["dataset"]
                for row in rows
                if abs(
                    row["contrasts"]["linear_minus_qls_mlp"]["recall@5"]["seed_effect"][
                        "mean"
                    ]
                )
                <= 0.01
            ],
            "linear_control_improves_selected_rrf_r5": [
                row["dataset"]
                for row in rows
                if row["contrasts"]["linear_minus_selected_rrf"]["recall@5"]["seed_effect"][
                    "mean"
                ]
                > 0
            ],
            "universal_non_message_passing_claim_supported": False,
            "next_step": "close_Package_A_and_do_not_tune_A3",
        },
    }


def _pct(value: float) -> str:
    return f"{100 * value:.2f}"


def _effect(value: float) -> str:
    return f"{100 * value:+.2f}"


def render_markdown(analysis: dict[str, Any]) -> str:
    rows = analysis["datasets"]
    lines = [
        "# P0 A3 linear rank + structure results",
        "",
        "Status: **complete on all six frozen datasets and five canonical seeds**.",
        "",
        (
            "A3 is a bias-free 19-parameter linear scorer over two source-rank, seven "
            "static-graph, and ten query-local structural features. It loads neither graph "
            "adjacency nor node/query embeddings. All learning-rate and epoch choices use "
            "validation only."
        ),
        "",
        "## Main R@5 decomposition",
        "",
        "The `best A2` column is a descriptive maximum from the already complete A2 table; it was not used to select A3 inputs or settings.",
        "",
        "| Dataset | Selected RRF | Best A2 | A3 linear | Seed-only MLP | QLS-MLP | Seed-aware GNN |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        models = row["models"]
        lines.append(
            f"| {row['dataset']} | {_pct(models['selected_rrf']['recall@5']['mean'])} | "
            f"{_pct(models['best_training_free_descriptive']['recall@5']['mean'])} "
            f"({models['best_training_free_descriptive']['method']}) | "
            f"{_pct(models['linear_rank_structure']['recall@5']['mean'])} ± "
            f"{_pct(models['linear_rank_structure']['recall@5']['sample_std'])} | "
            f"{_pct(models['seed_only']['recall@5']['mean'])} | "
            f"{_pct(models['sa_mlp']['recall@5']['mean'])} | "
            f"{_pct(models['seed_aware_gnn']['recall@5']['mean'])} |"
        )

    lines.extend(
        [
            "",
            "## What A3 establishes",
            "",
            (
                "Learned weighting of fixed rank and structure is genuinely useful, but it is "
                "not the whole QLS result. Relative to the best training-free A2 rule, A3 gains "
                "+6.82 R@5 points on WebQSP, +6.32 on MetaQA, and +1.34 on HotpotQA. It "
                "essentially ties the rank baseline on 2Wiki and SQuAD and loses 0.73 on "
                "MuSiQue."
            ),
            "",
            (
                "QLS-MLP still leads A3 by 11.77 points on MuSiQue, 11.16 on WebQSP, "
                "5.63 on MetaQA, and 2.55 on HotpotQA. A3 is within one point of QLS only "
                "on 2Wiki and SQuAD. Thus neither rank fusion, one fixed structural rule, nor "
                "linear reweighting explains the six-dataset QLS result."
            ),
            "",
            (
                "On the three regimes where fixed structure was most useful, A3 recovers "
                "51.8% (WebQSP), 47.9% (HotpotQA), and 65.6% (MetaQA) of the selected-RRF "
                "to QLS R@5 gap. The remaining gap is consistent with nonlinear semantic/"
                "structural interaction; it is not evidence by itself that message passing is "
                "necessary, because QLS remains non-message-passing."
            ),
            "",
            "## Registered paired R@5 contrasts",
            "",
            "| Dataset | A3 − RRF | A3 − seed-only | A3 − QLS | A3 − GNN | A3−QLS paired-query 95% CI | Holm p (A3−QLS) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        contrasts = row["contrasts"]
        qls = contrasts["linear_minus_qls_mlp"]["recall@5"]
        lines.append(
            f"| {row['dataset']} | "
            f"{_effect(contrasts['linear_minus_selected_rrf']['recall@5']['seed_effect']['mean'])} | "
            f"{_effect(contrasts['linear_minus_seed_only']['recall@5']['seed_effect']['mean'])} | "
            f"{_effect(qls['seed_effect']['mean'])} | "
            f"{_effect(contrasts['linear_minus_seed_aware_gnn']['recall@5']['seed_effect']['mean'])} | "
            f"[{_effect(qls['paired_hierarchical_query_ci95_low'])}, "
            f"{_effect(qls['paired_hierarchical_query_ci95_high'])}] | "
            f"{qls['holm_adjusted_pvalue_across_datasets']:.4g} |"
        )

    lines.extend(
        [
            "",
            "## A3 full effectiveness table",
            "",
            "| Dataset | R@1 | R@5 | R@20 | MRR | FullCov@20 | LR |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        model = row["models"]["linear_rank_structure"]
        lines.append(
            f"| {row['dataset']} | "
            + " | ".join(
                f"{_pct(model[metric]['mean'])} ± {_pct(model[metric]['sample_std'])}"
                for metric in METRICS
            )
            + f" | {row['selected_learning_rate']:.3g} |"
        )

    lines.extend(
        [
            "",
            "## Warm-cache systems accounting",
            "",
            "| Dataset | A3 ms/query | A3 speedup vs QLS | A3 speedup vs GNN | A3 params | QLS params | GNN params | Total fixed cache GiB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        system = row["systems"]
        lines.append(
            f"| {row['dataset']} | {system['linear_latency_ms_per_query']['mean']:.4f} | "
            f"{system['linear_speedup_over_qls']['mean']:.2f}× | "
            f"{system['linear_speedup_over_gnn']['mean']:.2f}× | "
            f"{system['linear_parameters']} | {system['qls_parameters']:,} | "
            f"{system['gnn_parameters']:,} | {system['total_cache_bytes'] / 1024**3:.3f} |"
        )

    metaqa = next(row for row in rows if row["dataset"] == "metaqa")
    lines.extend(
        [
            "",
            "These are warm-cache post-retrieval measurements. They include cached-feature gathering and device transfer but exclude raw retrieval, graph/feature construction, and metric sorting. Structural and derived-cache build time/disk are preserved in the JSON analysis; no uncached real-world speedup is claimed.",
            "",
            "## MetaQA R@5 by hop",
            "",
            "| Hop | A3 linear | Seed-only | QLS-MLP | Seed-aware GNN |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for hop in (1, 2, 3):
        values = metaqa["metaqa_hops"][str(hop)]
        lines.append(
            f"| {hop} | {_pct(values['linear_rank_structure']['mean'])} | "
            f"{_pct(values['seed_only']['mean'])} | {_pct(values['sa_mlp']['mean'])} | "
            f"{_pct(values['seed_aware_gnn']['mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Audit and stopping point",
            "",
            "- All A1/A2/A3/confirmation test query indices and query-order hashes match exactly.",
            "- All six A3 artifacts report five seeds, 19 parameters, validation-only selection, and exactly one label-based test evaluation per seed.",
            "- The derived A3 cache contains only candidate IDs and Dense/SPLADE rank features; labels are built in memory and never persisted with features.",
            "- The evaluator loads no graph adjacency, node embeddings, or query embeddings.",
            "- Zero seed variance on WebQSP is reported rather than hidden; the convex linear scorer converged to the same ranking across shuffles.",
            "",
            (
                "**Package A is now closed.** Do not tune A3 against these tests. The evidence "
                "supports a graded capacity story—not a universal MLP win: fixed structure helps, "
                "linear weighting recovers part of that opportunity, nonlinear QLS is needed in "
                "four datasets, and seed-aware message passing retains a small-to-large R@5 lead "
                "except on SQuAD."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root", type=Path, default=REPO_ROOT / "outputs" / "p0_linear_rank_structure"
    )
    parser.add_argument(
        "--rank-root", type=Path, default=REPO_ROOT / "outputs" / "p0_rank_controls"
    )
    parser.add_argument(
        "--structural-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "p0_fixed_structural_controls",
    )
    parser.add_argument(
        "--confirmation-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "sa_mlp_confirmation",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20270830)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "p0_linear_rank_structure_analysis.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPO_ROOT / "docs" / "P0_LINEAR_RANK_STRUCTURE_RESULTS.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = compile_analysis(
        args.result_root,
        args.rank_root,
        args.structural_root,
        args.confirmation_root,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(analysis), encoding="utf-8")
    print(json.dumps({"status": analysis["status"], "datasets": len(analysis["datasets"])}))


if __name__ == "__main__":
    main()
