#!/usr/bin/env python
"""Compile the frozen six-dataset SA-MLP confirmation without partial peeking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "sa_mlp_confirmation.yaml"
METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")
MODEL_ORDER = ("plain_mlp", "seed_only", "sa_mlp", "seed_aware_gnn")
CONTRASTS = {
    "seed_only_minus_plain_mlp": ("seed_only", "plain_mlp"),
    "sa_mlp_minus_seed_only": ("sa_mlp", "seed_only"),
    "sa_mlp_minus_seed_aware_gnn": ("sa_mlp", "seed_aware_gnn"),
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
    if len(values) < 2:
        return float("nan")
    if max(values) == min(values):
        return 1.0 if values[0] == 0 else 0.0
    return float(stats.ttest_1samp(values, popmean=0.0).pvalue)


def _holm(pvalues: dict[str, float]) -> dict[str, float]:
    finite = [(key, value) for key, value in pvalues.items() if math.isfinite(value)]
    ordered = sorted(finite, key=lambda item: item[1])
    adjusted: dict[str, float] = {key: float("nan") for key in pvalues}
    running = 0.0
    count = len(ordered)
    for rank, (key, value) in enumerate(ordered):
        running = max(running, (count - rank) * value)
        adjusted[key] = min(running, 1.0)
    return adjusted


def _query_order_sha256(ids: list[str]) -> str:
    digest = hashlib.sha256()
    for query_id in ids:
        digest.update(query_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _baseline_keys(payload: dict[str, Any]) -> tuple[str, str]:
    if payload.get("status") == "PAPER_MAIN_TABLE_DATASET_COMPLETE":
        selected = payload["selection_validation_only"]["selected"]
        return "plain_mlp", selected
    if payload.get("status") == "CONFIRMATION_GATE_NOT_PAPER_FINAL":
        selected = payload["frozen_best_gnn"]
        return "plain_mlp_h64", f"{selected}_h64"
    raise ValueError("Unsupported frozen baseline artifact")


def _plain_query_arrays(
    baseline: dict[str, Any],
    metric_names: list[str],
    expected_order_sha256: str,
    seeds: list[int],
) -> np.ndarray:
    plain_key, _gnn_key = _baseline_keys(baseline)
    first = baseline["models"][plain_key]["seeds"][str(seeds[0])]["per_query"]
    query_ids = list(first)
    observed = _query_order_sha256(query_ids)
    if observed != expected_order_sha256:
        raise ValueError("Frozen plain-MLP query order differs from packed confirmation order")
    output = np.empty((len(seeds), len(query_ids), len(metric_names)), dtype=np.float32)
    for seed_position, seed in enumerate(seeds):
        rows = baseline["models"][plain_key]["seeds"][str(seed)]["per_query"]
        if list(rows) != query_ids:
            raise ValueError("Frozen plain-MLP query order changes across seeds")
        output[seed_position] = np.asarray(
            [[float(rows[query_id][metric]) for metric in metric_names] for query_id in query_ids],
            dtype=np.float32,
        )
    return output


def _packed_new_arrays(
    packed_path: Path,
    expected_sha256: str,
    seeds: list[int],
) -> tuple[list[str], dict[str, np.ndarray], str]:
    digest = hashlib.sha256(packed_path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Packed query metrics failed SHA-256: {packed_path}")
    with np.load(packed_path) as packed:
        metric_names = [str(value) for value in packed["metric_names"].tolist()]
        order_hash = str(packed["query_order_sha256"].item())
        models = {
            model: np.stack(
                [np.asarray(packed[f"{model}_seed_{seed}"], dtype=np.float32) for seed in seeds]
            )
            for model in ("seed_only", "sa_mlp", "seed_aware_gnn")
        }
    return metric_names, models, order_hash


def _hierarchical_paired_ci(
    left: np.ndarray,
    right: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-stage paired bootstrap: optimizer seeds, then shared query IDs."""

    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("Paired query matrices must have shape [seed, query, metric]")
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


def _seed_records(
    new: dict[str, Any],
    baseline: dict[str, Any],
    seeds: list[int],
) -> dict[str, dict[str, Any]]:
    plain_key, _gnn_key = _baseline_keys(baseline)
    return {
        "plain_mlp": baseline["models"][plain_key]["seeds"],
        "seed_only": new["models"]["seed_only"]["seeds"],
        "sa_mlp": new["models"]["sa_mlp"]["seeds"],
        "seed_aware_gnn": new["models"]["seed_aware_gnn"]["seeds"],
    }


def _aggregate_models(records: dict[str, dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model in MODEL_ORDER:
        output[model] = {
            metric: _mean_std_ci(
                [float(records[model][str(seed)]["metrics"][metric]) for seed in seeds]
            )
            for metric in METRICS
        }
    return output


def _aggregate_systems(
    new: dict[str, Any],
    records: dict[str, dict[str, Any]],
    seeds: list[int],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model in MODEL_ORDER:
        model_runs = records[model]
        available_training = all("training" in model_runs[str(seed)] for seed in seeds)
        available_cpu = all(
            "peak_cpu_rss_mb_total" in model_runs[str(seed)]["inference"] for seed in seeds
        )
        output[model] = {
            "parameters": (
                new["baseline"]["plain_mlp"]["parameters"]
                if model == "plain_mlp"
                else new["models"][model]["parameters"]
            ),
            "latency_ms_per_query": _mean_std_ci(
                [
                    float(model_runs[str(seed)]["inference"]["latency_ms_per_query"])
                    for seed in seeds
                ]
            ),
            "peak_gpu_memory_mb_incremental": _mean_std_ci(
                [
                    float(model_runs[str(seed)]["inference"]["peak_gpu_memory_mb_incremental"])
                    for seed in seeds
                ]
            ),
        }
        if available_training:
            output[model]["training_seconds"] = _mean_std_ci(
                [float(model_runs[str(seed)]["training"]["training_seconds"]) for seed in seeds]
            )
        if available_cpu:
            output[model]["peak_cpu_rss_mb_total"] = _mean_std_ci(
                [
                    float(model_runs[str(seed)]["inference"]["peak_cpu_rss_mb_total"])
                    for seed in seeds
                ]
            )
    sa_latency = [
        float(records["sa_mlp"][str(seed)]["inference"]["latency_ms_per_query"]) for seed in seeds
    ]
    gnn_latency = [
        float(records["seed_aware_gnn"][str(seed)]["inference"]["latency_ms_per_query"])
        for seed in seeds
    ]
    output["sa_mlp_latency_divided_by_seed_aware_gnn"] = _mean_std_ci(
        [left / right for left, right in zip(sa_latency, gnn_latency)]
    )
    cache = new["feature_cache"]
    output["fixed_feature_cache"] = {
        "disk_bytes": cache["cache_bytes"],
        "disk_gib": cache["cache_bytes"] / 1024**3,
        "precomputation_seconds": cache["total_preprocessing_seconds"],
        "candidate_rows": cache["candidate_rows"],
        "static_dtype": cache["static_dtype"],
        "local_dtype": cache["local_dtype"],
    }
    return output


def _metaqa_hops(records: dict[str, dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for hop in (1, 2, 3):
        output[str(hop)] = {}
        for model in MODEL_ORDER:
            output[str(hop)][model] = {
                metric: _mean_std_ci(
                    [
                        float(records[model][str(seed)]["by_hop"][str(hop)]["metrics"][metric])
                        for seed in seeds
                    ]
                )
                for metric in METRICS
            }
    return output


def analyze(
    root: Path,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["training"]["seeds"]]
    dataset_rows: list[dict[str, Any]] = []
    for dataset_position, (dataset, spec) in enumerate(config["datasets"].items()):
        result_path = root / f"{dataset}.json"
        packed_path = root / f"{dataset}.query_metrics.npz"
        if not result_path.is_file() or not packed_path.is_file():
            raise FileNotFoundError(f"Confirmation is incomplete; missing {dataset} outputs")
        new = _load(result_path)
        if new.get("status") != "SA_MLP_CONFIRMATION_DATASET_COMPLETE":
            raise ValueError(f"Confirmation result is incomplete: {dataset}")
        baseline = _load(REPO_ROOT / spec["baseline"])
        records = _seed_records(new, baseline, seeds)
        if any(sorted(map(int, records[model])) != seeds for model in MODEL_ORDER):
            raise ValueError(f"Incomplete five-seed model record: {dataset}")
        metric_names, new_arrays, order_hash = _packed_new_arrays(
            packed_path,
            new["query_metrics"]["sha256"],
            seeds,
        )
        if metric_names != list(METRICS):
            raise ValueError(f"Packed metric order changed: {dataset}")
        plain = _plain_query_arrays(baseline, metric_names, order_hash, seeds)
        arrays = {"plain_mlp": plain, **new_arrays}
        contrasts: dict[str, Any] = {}
        for contrast_position, (contrast, (left, right)) in enumerate(CONTRASTS.items()):
            low, high = _hierarchical_paired_ci(
                arrays[left],
                arrays[right],
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + dataset_position * 101 + contrast_position,
            )
            contrasts[contrast] = {}
            for metric_position, metric in enumerate(METRICS):
                differences = [
                    float(records[left][str(seed)]["metrics"][metric])
                    - float(records[right][str(seed)]["metrics"][metric])
                    for seed in seeds
                ]
                contrasts[contrast][metric] = {
                    "seed_effect": _mean_std_ci(differences),
                    "paired_seed_t_pvalue": _paired_t_pvalue(differences),
                    "paired_hierarchical_query_ci95_low": float(low[metric_position]),
                    "paired_hierarchical_query_ci95_high": float(high[metric_position]),
                }
        dataset_rows.append(
            {
                "dataset": dataset,
                "models": _aggregate_models(records, seeds),
                "contrasts": contrasts,
                "systems": _aggregate_systems(new, records, seeds),
                "metaqa_hops": _metaqa_hops(records, seeds) if dataset == "metaqa" else {},
                "candidate_compatibility_proof": new["comparison_contract"][
                    "candidate_compatibility_proof"
                ],
            }
        )

    for contrast in CONTRASTS:
        for metric in METRICS:
            adjusted = _holm(
                {
                    row["dataset"]: row["contrasts"][contrast][metric]["paired_seed_t_pvalue"]
                    for row in dataset_rows
                }
            )
            for row in dataset_rows:
                item = row["contrasts"][contrast][metric]
                item["holm_adjusted_pvalue_across_datasets"] = adjusted[row["dataset"]]
                item["holm_significant_0.05"] = adjusted[row["dataset"]] < 0.05

    return {
        "status": "SA_MLP_CONFIRMATION_SIX_DATASET_COMPLETE",
        "protocol_tag": "sa-mlp-confirmation-protocol-v1",
        "compatibility_tag": "sa-mlp-confirmation-compat-v1",
        "seeds": seeds,
        "bootstrap": {
            "method": "paired_two_stage_seed_then_query_percentile",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
        },
        "datasets": dataset_rows,
    }


def _percentage(value: float) -> str:
    return f"{100 * value:.2f}"


def _effect(value: float) -> str:
    return f"{100 * value:+.2f}"


def _markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# SA-MLP six-dataset fairness confirmation",
        "",
        (
            "Status: **all six datasets and five paired seeds complete**. SA-MLP is a "
            "non-message-passing fixed-structure model, not a topology-free model."
        ),
        "",
    ]
    for row in analysis["datasets"]:
        lines.extend(
            [
                f"## {row['dataset']}",
                "",
                "| Model | R@1 | R@5 | R@20 | MRR | FullCov@20 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for model in MODEL_ORDER:
            values = row["models"][model]
            cells = [
                f"{_percentage(values[metric]['mean'])} ± "
                f"{_percentage(values[metric]['sample_std'])}"
                for metric in METRICS
            ]
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
        lines.extend(
            [
                "",
                "| Contrast | Metric | Mean effect | Seed 95% CI | Paired-query 95% CI | Holm p |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for contrast in CONTRASTS:
            for metric in METRICS:
                value = row["contrasts"][contrast][metric]
                seed_effect = value["seed_effect"]
                lines.append(
                    f"| {contrast} | {metric} | {_effect(seed_effect['mean'])} | "
                    f"[{_effect(seed_effect['ci95_low'])}, "
                    f"{_effect(seed_effect['ci95_high'])}] | "
                    f"[{_effect(value['paired_hierarchical_query_ci95_low'])}, "
                    f"{_effect(value['paired_hierarchical_query_ci95_high'])}] | "
                    f"{value['holm_adjusted_pvalue_across_datasets']:.4g} |"
                )
        systems = row["systems"]
        cache = systems["fixed_feature_cache"]
        lines.extend(
            [
                "",
                "| Model | Params | Train s | Latency ms/q | GPU peak MiB | CPU RSS MiB |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for model in MODEL_ORDER:
            item = systems[model]
            train = item.get("training_seconds", {}).get("mean")
            cpu = item.get("peak_cpu_rss_mb_total", {}).get("mean")
            lines.append(
                f"| {model} | {item['parameters']['trainable_parameters']:,} | "
                f"{'n/a' if train is None else f'{train:.1f}'} | "
                f"{item['latency_ms_per_query']['mean']:.4f} | "
                f"{item['peak_gpu_memory_mb_incremental']['mean']:.1f} | "
                f"{'n/a' if cpu is None else f'{cpu:.1f}'} |"
            )
        lines.extend(
            [
                "",
                "| Fixed-structure cost | Value |",
                "|---|---:|",
                (
                    "| SA latency / seed-aware GNN latency | "
                    f"{systems['sa_mlp_latency_divided_by_seed_aware_gnn']['mean']:.3f} |"
                ),
                f"| Feature cache | {cache['disk_gib']:.3f} GiB |",
                f"| Feature precomputation | {cache['precomputation_seconds']:.1f} s |",
                "",
            ]
        )
        if row["dataset"] == "metaqa":
            lines.extend(
                [
                    "### MetaQA hop breakdown",
                    "",
                    "| Hop | Model | R@1 | R@5 | R@20 | MRR | FullCov@20 |",
                    "|---:|---|---:|---:|---:|---:|---:|",
                ]
            )
            for hop in (1, 2, 3):
                for model in MODEL_ORDER:
                    values = row["metaqa_hops"][str(hop)][model]
                    cells = [
                        f"{_percentage(values[metric]['mean'])} ± "
                        f"{_percentage(values[metric]['sample_std'])}"
                        for metric in METRICS
                    ]
                    lines.append(f"| {hop} | {model} | " + " | ".join(cells) + " |")
            lines.append("")
    lines.extend(
        [
            "## Interpretation contract",
            "",
            (
                "`seed-only - plain` measures the retrieval prior; `SA - seed-only` measures "
                "fixed graph computation; `SA - seed-aware GNN` compares fixed structural "
                "summaries with learned message passing. No contrast is collapsed into another."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "outputs" / "sa_mlp_confirmation",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20270826)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "sa_mlp_confirmation_analysis.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPO_ROOT / "docs" / "SA_MLP_CONFIRMATION_RESULTS.md",
    )
    args = parser.parse_args()
    result = analyze(
        args.root,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.markdown_output.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "datasets": 6}, indent=2))


if __name__ == "__main__":
    main()
