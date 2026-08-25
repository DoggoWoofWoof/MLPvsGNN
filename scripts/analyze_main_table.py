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
    return {
        "dataset": item["dataset"],
        "nodes": item["data"]["nodes"],
        "edges": item["data"]["edges"],
        "selected_gnn": item["selected_gnn"],
        "mlp_parameters": mlp["parameters"]["parameters"],
        "gnn_parameters": gnn["parameters"]["parameters"],
        "mlp_latency_ms": mlp["aggregate"]["latency_ms_per_query"],
        "gnn_latency_ms": gnn["aggregate"]["latency_ms_per_query"],
        "gnn_over_mlp_latency": gnn_latency / mlp_latency,
        "mlp_throughput_qps": mlp["aggregate"]["throughput_queries_per_second"],
        "gnn_throughput_qps": gnn["aggregate"]["throughput_queries_per_second"],
        "mlp_peak_gpu_memory_mb_total": _total_memory(mlp),
        "gnn_peak_gpu_memory_mb_total": _total_memory(gnn),
        "mlp_peak_gpu_memory_mb_incremental": mlp["aggregate"][
            "peak_gpu_memory_mb_incremental"
        ],
        "gnn_peak_gpu_memory_mb_incremental": gnn["aggregate"][
            "peak_gpu_memory_mb_incremental"
        ],
        "mlp_training_seconds": mlp["aggregate"]["training_seconds"],
        "gnn_training_seconds": gnn["aggregate"]["training_seconds"],
        "cold_topology_preprocessing_seconds": item["topology"]["cold_build_seconds"],
        "packed_topology_storage_bytes": item["topology"]["packed_storage_bytes"],
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
    lines = [
        "# Six-dataset MLP vs message-passing results",
        "",
        "All values are five-seed means. Deltas are paired MLP minus GNN percentage points.",
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
            "MLP incr. MiB | GNN incr. MiB | MLP params | GNN params | Cold topology s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["systems_table"]:
        lines.append(
            f"| {row['dataset']} | {row['nodes']} | {row['edges']} | "
            f"{row['mlp_latency_ms']['mean']:.4f} | {row['gnn_latency_ms']['mean']:.4f} | "
            f"{row['gnn_over_mlp_latency']:.2f}× | "
            f"{row['mlp_peak_gpu_memory_mb_incremental']['mean']:.1f} | "
            f"{row['gnn_peak_gpu_memory_mb_incremental']['mean']:.1f} | "
            f"{row['mlp_parameters']} | {row['gnn_parameters']} | "
            f"{row['cold_topology_preprocessing_seconds']:.1f} |"
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
    return {
        "status": "SIX_DATASET_STOP_GATE_COMPLETE",
        "main_table": [_main_row(item) for item in normalized],
        "metaqa_hop_table": _metaqa_hops(metaqa),
        "systems_table": [_systems_row(item) for item in normalized],
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
