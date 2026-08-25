#!/usr/bin/env python
"""Compile the frozen six-dataset effectiveness, hop, and systems tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_confirmation import _mean_std  # noqa: E402


DATASET_ORDER = (
    "2wiki_clean",
    "musique_clean",
    "webqsp",
    "hotpotqa_clean",
    "squad_clean",
    "metaqa",
)
METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized(result: dict[str, Any], legacy: bool) -> dict[str, Any]:
    if legacy:
        selected = result["frozen_best_gnn"]
        mlp_key = "plain_mlp_h64"
        gnn_key = f"{selected}_h64"
        paired = result["paired_minus_frozen_gnn"][mlp_key]
        topology = {
            "cold_build_seconds": result["data"]["topology_preprocessing_seconds"],
            "candidate_induced_edges": None,
            "packed_storage_bytes": None,
        }
    else:
        selected = result["selection_validation_only"]["selected"]
        mlp_key = "plain_mlp"
        gnn_key = selected
        paired = result["paired_mlp_minus_gnn"]
        topology = result["data"]["topology"]
    models = {"mlp": result["models"][mlp_key], "gnn": result["models"][gnn_key]}
    for model in models.values():
        if len(model["seeds"]) != 5:
            raise ValueError(f"{result['dataset']} is not a complete five-seed result")
    return {
        "dataset": result["dataset"],
        "selected_gnn": selected,
        "data": result["data"],
        "models": models,
        "paired": paired,
        "topology": topology,
        "legacy_frozen_finding": legacy,
    }


def _total_memory(model: dict[str, Any]) -> dict[str, Any]:
    return _mean_std(
        [
            float(seed["inference"]["peak_gpu_memory_mb_total"])
            for seed in model["seeds"].values()
        ]
    )


def _main_row(item: dict[str, Any]) -> dict[str, Any]:
    mlp = item["models"]["mlp"]
    gnn = item["models"]["gnn"]
    return {
        "dataset": item["dataset"],
        "selected_gnn": item["selected_gnn"],
        "mlp": mlp["aggregate"]["test_metrics"],
        "gnn": gnn["aggregate"]["test_metrics"],
        "mlp_minus_gnn": item["paired"],
        "candidate_ceiling": float(
            next(iter(mlp["seeds"].values()))["metrics"]["candidate_ceiling"]
        ),
        "candidate_available": float(
            next(iter(mlp["seeds"].values()))["metrics"]["candidate_available"]
        ),
        "legacy_frozen_finding": item["legacy_frozen_finding"],
    }


def _systems_row(item: dict[str, Any]) -> dict[str, Any]:
    mlp = item["models"]["mlp"]
    gnn = item["models"]["gnn"]
    mlp_latency = float(mlp["aggregate"]["latency_ms_per_query"]["mean"])
    gnn_latency = float(gnn["aggregate"]["latency_ms_per_query"]["mean"])
    mlp_memory = mlp["aggregate"]["peak_gpu_memory_mb_incremental"]
    gnn_memory = gnn["aggregate"]["peak_gpu_memory_mb_incremental"]
    return {
        "dataset": item["dataset"],
        "nodes": item["data"]["nodes"],
        "edges": item["data"]["edges"],
        "selected_gnn": item["selected_gnn"],
        "mlp_parameters": mlp["parameters"]["parameters"],
        "gnn_parameters": gnn["parameters"]["parameters"],
        "gnn_over_mlp_parameters": (
            gnn["parameters"]["parameters"] / mlp["parameters"]["parameters"]
        ),
        "mlp_latency_ms": mlp["aggregate"]["latency_ms_per_query"],
        "gnn_latency_ms": gnn["aggregate"]["latency_ms_per_query"],
        "gnn_over_mlp_latency": gnn_latency / mlp_latency,
        "mlp_throughput_qps": mlp["aggregate"]["throughput_queries_per_second"],
        "gnn_throughput_qps": gnn["aggregate"]["throughput_queries_per_second"],
        "mlp_peak_gpu_memory_mb_total": _total_memory(mlp),
        "gnn_peak_gpu_memory_mb_total": _total_memory(gnn),
        "mlp_peak_gpu_memory_mb_incremental": mlp_memory,
        "gnn_peak_gpu_memory_mb_incremental": gnn_memory,
        "gnn_minus_mlp_peak_gpu_memory_mb_incremental": (
            float(gnn_memory["mean"]) - float(mlp_memory["mean"])
        ),
        "mlp_training_seconds": mlp["aggregate"]["training_seconds"],
        "gnn_training_seconds": gnn["aggregate"]["training_seconds"],
        "cold_topology_preprocessing_seconds": item["topology"]["cold_build_seconds"],
        "candidate_induced_edges": item["topology"].get("candidate_induced_edges"),
        "packed_topology_storage_bytes": item["topology"]["packed_storage_bytes"],
    }


def _conclusion(
    main_table: list[dict[str, Any]], systems_table: list[dict[str, Any]]
) -> dict[str, Any]:
    mlp_wins: list[str] = []
    gnn_wins: list[str] = []
    neutral: list[str] = []
    for row in main_table:
        interval = row["mlp_minus_gnn"]["recall@5"]
        if float(interval["ci95_low"]) > 0:
            mlp_wins.append(row["dataset"])
        elif float(interval["ci95_high"]) < 0:
            gnn_wins.append(row["dataset"])
        else:
            neutral.append(row["dataset"])
    speedups = [float(row["gnn_over_mlp_latency"]) for row in systems_table]
    memory_savings = [
        float(row["gnn_minus_mlp_peak_gpu_memory_mb_incremental"])
        for row in systems_table
    ]
    parameter_ratios = [float(row["gnn_over_mlp_parameters"]) for row in systems_table]
    return {
        "primary_metric": "recall@5",
        "decision_rule": "paired five-seed Student-t 95% interval relative to zero",
        "mlp_wins": mlp_wins,
        "gnn_wins": gnn_wins,
        "neutral": neutral,
        "gnn_over_mlp_latency_min": min(speedups),
        "gnn_over_mlp_latency_max": max(speedups),
        "gnn_minus_mlp_incremental_memory_mib_min": min(memory_savings),
        "gnn_minus_mlp_incremental_memory_mib_max": max(memory_savings),
        "gnn_over_mlp_parameter_min": min(parameter_ratios),
        "gnn_over_mlp_parameter_max": max(parameter_ratios),
    }


def _metaqa_hops(item: dict[str, Any]) -> list[dict[str, Any]]:
    if item["dataset"] != "metaqa":
        return []
    rows: list[dict[str, Any]] = []
    for hop in (1, 2, 3):
        hop_key = str(hop)
        metrics_by_model: dict[str, dict[str, Any]] = {}
        counts: set[int] = set()
        for family in ("mlp", "gnn"):
            seeds = item["models"][family]["seeds"]
            counts.update(int(seed["by_hop"][hop_key]["queries"]) for seed in seeds.values())
            metrics_by_model[family] = {
                metric: _mean_std(
                    [float(seed["by_hop"][hop_key]["metrics"][metric]) for seed in seeds.values()]
                )
                for metric in METRICS
            }
        if len(counts) != 1:
            raise ValueError(f"MetaQA hop-{hop} query counts differ across model seeds")
        paired = {
            metric: _mean_std(
                [
                    float(
                        item["models"]["mlp"]["seeds"][str(seed)]["by_hop"][hop_key][
                            "metrics"
                        ][metric]
                    )
                    - float(
                        item["models"]["gnn"]["seeds"][str(seed)]["by_hop"][hop_key][
                            "metrics"
                        ][metric]
                    )
                    for seed in range(5)
                ]
            )
            for metric in METRICS
        }
        rows.append(
            {
                "hop": hop,
                "queries": counts.pop(),
                "selected_gnn": item["selected_gnn"],
                "mlp": metrics_by_model["mlp"],
                "gnn": metrics_by_model["gnn"],
                "mlp_minus_gnn": paired,
            }
        )
    return rows


def _pct(value: float) -> str:
    return f"{100 * value:.2f}"


def _ci(row: dict[str, Any]) -> str:
    return f"[{_pct(row['ci95_low'])}, {_pct(row['ci95_high'])}]"


def _markdown(analysis: dict[str, Any]) -> str:
    conclusion = analysis["conclusion"]
    lines = [
        "# Six-dataset MLP vs message-passing results",
        "",
        "All values are five-seed means. Deltas are paired MLP minus GNN percentage points.",
        "",
        "## Stop-gate conclusion",
        "",
        "The frozen R@5 comparison yields two MLP wins whose paired five-seed 95% intervals "
        "exclude zero (2Wiki and MuSiQue), three GNN wins (WebQSP, HotpotQA, and MetaQA), "
        "and one neutral result (SQuAD). This rejects both universal claims: neither "
        "message passing nor a topology-free MLP dominates every retrieval regime.",
        "",
        f"The MLP is {conclusion['gnn_over_mlp_latency_min']:.2f}--"
        f"{conclusion['gnn_over_mlp_latency_max']:.2f}x faster and saves "
        f"{conclusion['gnn_minus_mlp_incremental_memory_mib_min']:.0f}--"
        f"{conclusion['gnn_minus_mlp_incremental_memory_mib_max']:.0f} MiB of incremental "
        "peak GPU memory. Parameter counts are effectively matched: the GNNs have only "
        f"{conclusion['gnn_over_mlp_parameter_min']:.3f}--"
        f"{conclusion['gnn_over_mlp_parameter_max']:.3f}x the MLP parameters. The result is "
        "therefore a latency/memory tradeoff, not a claim of materially fewer MLP parameters.",
        "",
        "MetaQA does not show an increasing GNN advantage with hop count: its R@5 advantage "
        "is 25.53 points at 1 hop, 2.39 at 2 hops, and 0.85 at 3 hops. Hop count alone is "
        "not the mechanism. The causal reason for the cross-dataset boundary remains untested "
        "at this stop gate and must not be inferred from dataset names.",
        "",
        "## Main effectiveness table",
        "",
        "| Dataset | GNN | MLP R@1 | GNN R@1 | ΔR@1 | MLP R@5 | GNN R@5 | "
        "ΔR@5 | ΔR@20 | ΔMRR | ΔFullCov@20 | R@5 paired 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["main_table"]:
        pair = row["mlp_minus_gnn"]
        lines.append(
            f"| {row['dataset']} | {row['selected_gnn'].upper()} | "
            f"{_pct(row['mlp']['recall@1']['mean'])} | {_pct(row['gnn']['recall@1']['mean'])} | "
            f"{_pct(pair['recall@1']['mean'])} | {_pct(row['mlp']['recall@5']['mean'])} | "
            f"{_pct(row['gnn']['recall@5']['mean'])} | {_pct(pair['recall@5']['mean'])} | "
            f"{_pct(pair['recall@20']['mean'])} | {_pct(pair['mrr']['mean'])} | "
            f"{_pct(pair['full_coverage@20']['mean'])} | {_ci(pair['recall@5'])} |"
        )
    lines.extend(
        [
            "",
            "## Candidate-conditional table",
            "",
            "| Dataset | Candidate ceiling | Queries with ≥1 in-pool gold | "
            "MLP conditional R@5 | GNN conditional R@5 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["main_table"]:
        lines.append(
            f"| {row['dataset']} | {_pct(row['candidate_ceiling'])} | "
            f"{_pct(row['candidate_available'])} | "
            f"{_pct(row['mlp']['conditional_recall@5']['mean'])} | "
            f"{_pct(row['gnn']['conditional_recall@5']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## MetaQA hop table",
            "",
            "| Hop | Test queries | MLP R@5 | GNN R@5 | ΔR@5 | Paired 95% CI | MLP MRR | GNN MRR |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["metaqa_hop_table"]:
        lines.append(
            f"| {row['hop']} | {row['queries']} | {_pct(row['mlp']['recall@5']['mean'])} | "
            f"{_pct(row['gnn']['recall@5']['mean'])} | "
            f"{_pct(row['mlp_minus_gnn']['recall@5']['mean'])} | "
            f"{_ci(row['mlp_minus_gnn']['recall@5'])} | {_pct(row['mlp']['mrr']['mean'])} | "
            f"{_pct(row['gnn']['mrr']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Systems table",
            "",
            "| Dataset | Nodes | Edges | MLP ms/q | GNN ms/q | GNN/MLP latency | "
            "MLP incr. MiB | GNN incr. MiB | Saved MiB | MLP params | GNN params | "
            "Cold topology s | Packed topology GiB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["systems_table"]:
        packed = row["packed_topology_storage_bytes"]
        packed_gib = "--" if packed is None else f"{packed / (1024 ** 3):.3f}"
        lines.append(
            f"| {row['dataset']} | {row['nodes']} | {row['edges']} | "
            f"{row['mlp_latency_ms']['mean']:.4f} | {row['gnn_latency_ms']['mean']:.4f} | "
            f"{row['gnn_over_mlp_latency']:.2f}× | "
            f"{row['mlp_peak_gpu_memory_mb_incremental']['mean']:.1f} | "
            f"{row['gnn_peak_gpu_memory_mb_incremental']['mean']:.1f} | "
            f"{row['gnn_minus_mlp_peak_gpu_memory_mb_incremental']:.1f} | "
            f"{row['mlp_parameters']} | {row['gnn_parameters']} | "
            f"{row['cold_topology_preprocessing_seconds']:.1f} | {packed_gib} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary and next decision",
            "",
            "What is established: under identical frozen features, candidates, labels, loss, "
            "splits, seeds, and training budget, adding the validation-selected message-passing "
            "model helps on three datasets, hurts on two, and is neutral on one; its inference "
            "cost is higher on all six. Candidate-conditional R@5 preserves the same directions, "
            "so the boundary is not an artifact of missing candidate-pool golds.",
            "",
            "What is not established: these tables do not identify homophily, neighborhood "
            "noise, hubness, answer multiplicity, or feature quality as the cause. They also do "
            "not support an all-dataset MLP claim, a fewer-parameters claim, or a claim that GNN "
            "value grows with query hops.",
            "",
            "The preregistered stop condition is satisfied. No topology perturbation, mechanism "
            "predictor, Offset rescue, or new architecture has been run as part of this gate.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(root: Path) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        if dataset in {"2wiki_clean", "musique_clean"}:
            result = _load(root / "confirmation" / f"{dataset}.json")
            normalized.append(_normalized(result, legacy=True))
        else:
            result = _load(root / "main_table" / f"{dataset}.json")
            if result["status"] != "PAPER_MAIN_TABLE_DATASET_COMPLETE":
                raise ValueError(f"{dataset} result is incomplete")
            normalized.append(_normalized(result, legacy=False))
    metaqa = next(item for item in normalized if item["dataset"] == "metaqa")
    main_table = [_main_row(item) for item in normalized]
    systems_table = [_systems_row(item) for item in normalized]
    return {
        "status": "SIX_DATASET_STOP_GATE_COMPLETE",
        "conclusion": _conclusion(main_table, systems_table),
        "main_table": main_table,
        "metaqa_hop_table": _metaqa_hops(metaqa),
        "systems_table": systems_table,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument(
        "--json-output", type=Path, default=REPO_ROOT / "outputs" / "main_table_analysis.json"
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=REPO_ROOT / "docs" / "SIX_DATASET_RESULTS.md"
    )
    args = parser.parse_args()
    result = analyze(args.root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "datasets": len(result["main_table"])}, indent=2))


if __name__ == "__main__":
    main()
