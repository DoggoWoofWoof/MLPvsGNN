#!/usr/bin/env python
"""Apply the locked validation-only crossover selection rule."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "phase_screen.yaml"
MODEL_NAMES = ("sa_mlp", "seed_aware_gnn")


def _rate_key(rate: float) -> str:
    return f"{rate:.2f}".replace(".", "p")


def select_confirmation_rates(points: list[tuple[float, float]]) -> list[float]:
    """Select endpoints/crossover brackets without inspecting test metrics."""

    if len(points) < 2 or points[0][0] != 0.0:
        raise ValueError("Phase points must start with the clean rate-zero condition")
    ordered = sorted((float(rate), float(gap)) for rate, gap in points)
    if len({rate for rate, _gap in ordered}) != len(ordered):
        raise ValueError("Phase-screen rates must be unique")
    selected = {ordered[0][0], ordered[-1][0]}
    exact = [index for index, (_rate, gap) in enumerate(ordered) if gap == 0.0]
    if exact:
        for index in exact:
            selected.add(ordered[index][0])
            if index > 0:
                selected.add(ordered[index - 1][0])
            if index + 1 < len(ordered):
                selected.add(ordered[index + 1][0])
        return sorted(selected)
    for left, right in pairwise(ordered):
        if (left[1] < 0 < right[1]) or (right[1] < 0 < left[1]):
            selected.update((left[0], right[0]))
    return sorted(selected)


def _clean_gap(confirmation: dict[str, Any]) -> tuple[float, dict[str, float]]:
    metrics = {}
    for model in MODEL_NAMES:
        training = confirmation["models"][model]["seeds"]["0"]["training"]
        metrics[model] = float(training["best_validation_recall@5"])
    return metrics["seed_aware_gnn"] - metrics["sa_mlp"], metrics


def compile_analysis(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    merged = {axis: {0.0, float(spec["rates"][-1])} for axis, spec in config["axes"].items()}
    for dataset, dataset_spec in config["datasets"].items():
        confirmation = json.loads(
            (REPO_ROOT / dataset_spec["confirmation"]).read_text(encoding="utf-8")
        )
        clean_gap, clean_metrics = _clean_gap(confirmation)
        axes: dict[str, Any] = {}
        for axis, axis_spec in config["axes"].items():
            points = [(0.0, clean_gap)]
            conditions = []
            for rate in map(float, axis_spec["rates"]):
                path = root / dataset / axis / f"rate_{_rate_key(rate)}.json"
                if not path.is_file():
                    raise FileNotFoundError(f"Phase screen is incomplete: {path}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (
                    payload.get("status") != "PHASE_SCREEN_VALIDATION_ONLY_COMPLETE"
                    or payload.get("dataset") != dataset
                    or payload.get("axis") != axis
                    or float(payload.get("rate", -1)) != rate
                    or payload["screen_contract"].get("test_metrics_computed") is not False
                    or int(payload["screen_contract"].get("training_seed", -1)) != 0
                ):
                    raise ValueError(f"Phase-screen contract failed: {path}")
                gap = float(payload["validation_gnn_minus_qls"]["recall@5"])
                points.append((rate, gap))
                conditions.append(
                    {
                        "rate": rate,
                        "gnn_minus_qls_validation_recall@5": gap,
                        "models": {
                            model: float(payload["models"][model]["validation_metrics"]["recall@5"])
                            for model in MODEL_NAMES
                        },
                        "intervention": payload["intervention"],
                    }
                )
            selected = select_confirmation_rates(points)
            merged[axis].update(selected)
            axes[axis] = {
                "clean": {
                    "rate": 0.0,
                    "gnn_minus_qls_validation_recall@5": clean_gap,
                    "models": clean_metrics,
                    "source": "sealed_confirmation_seed_0_best_validation",
                },
                "conditions": conditions,
                "selected_rates": selected,
                "has_sign_change": any(
                    (left[1] < 0 < right[1]) or (right[1] < 0 < left[1])
                    for left, right in pairwise(points)
                ),
                "has_exact_zero": any(gap == 0.0 for _rate, gap in points),
            }
        rows.append({"dataset": dataset, "axes": axes})
    selected_by_axis = {axis: sorted(rates) for axis, rates in merged.items()}
    analysis = {
        "status": "PHASE_SCREEN_VALIDATION_ONLY_ANALYZED",
        "metric": "validation_recall@5_seed_aware_gnn_minus_qls_mlp",
        "test_metrics_computed": False,
        "datasets": rows,
        "merged_confirmation_rates": selected_by_axis,
        "selection_rule": config["screen_selection_rule"],
        "predictor_training_allowed": False,
        "stopping_point": "RATES_SELECTED_REQUIRES_PROTOCOL_COMMIT_BEFORE_TEST",
    }
    confirmation = {
        "experiment": "five_seed_phase_crossover_confirmation",
        "status": "GENERATED_FROM_LOCKED_VALIDATION_RULE_REQUIRES_COMMIT_BEFORE_TEST",
        "protocol_ancestor": "phase-screen-protocol-v1",
        "selection_source": str(root),
        "axes": {
            axis: {
                "rates": rates,
                "perturbation_seed": int(config["perturbation_seeds"][axis]),
            }
            for axis, rates in selected_by_axis.items()
        },
        "datasets": config["datasets"],
        "training": {
            **config["training"],
            "seeds": [0, 1, 2, 3, 4],
            "evaluation_split": "test_once_after_validation_checkpoint_selection",
        },
        "parameter_regime": config["parameter_regime"],
        "inference_repeats": 5,
        "analysis": {
            "primary_contrast": "seed_aware_gnn_minus_sa_mlp_recall_at_5",
            "paired_seed_and_query": True,
            "holm_scope": "datasets_within_axis_and_rate",
            "all_selected_cells_reported": True,
            "predictor_training_before_confirmation": False,
        },
        "modal": {
            **config["modal"],
            "app": "message-passing-retrieval-phase-confirmation",
        },
    }
    return analysis, confirmation


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Validation-only phase screen",
        "",
        "Status: **complete only when every registered cell below is present**.",
        "",
        (
            "Positive values mean the seed-aware GNN exceeds QLS-MLP on validation R@5. "
            "No test metric is read by this analysis."
        ),
        "",
    ]
    for row in analysis["datasets"]:
        lines.extend([f"## {row['dataset']}", ""])
        for axis, values in row["axes"].items():
            points = [values["clean"], *values["conditions"]]
            curve = ", ".join(
                f"{point['rate']:.2f}: {100 * point['gnn_minus_qls_validation_recall@5']:+.2f}"
                for point in points
            )
            rates = ", ".join(f"{rate:.2f}" for rate in values["selected_rates"])
            lines.append(f"- `{axis}` — gaps [{curve}]; selected [{rates}]")
        lines.append("")
    lines.extend(
        [
            "## Frozen next boundary",
            "",
            (
                "The union of selected rates is written to the generated phase-confirmation "
                "configuration. That file must be reviewed, committed, and tagged before any "
                "selected test cell runs. Predictor fitting remains prohibited until the five-seed "
                "confirmation establishes reproducible help, neutral, and harm regions."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "outputs" / "phase_screen")
    parser.add_argument(
        "--json-output", type=Path, default=REPO_ROOT / "outputs" / "phase_screen_analysis.json"
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=REPO_ROOT / "docs" / "PHASE_SCREEN_RESULTS.md"
    )
    parser.add_argument(
        "--confirmation-config",
        type=Path,
        default=REPO_ROOT / "configs" / "phase_confirmation.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis, confirmation = compile_analysis(args.root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(analysis), encoding="utf-8")
    args.confirmation_config.parent.mkdir(parents=True, exist_ok=True)
    args.confirmation_config.write_text(
        yaml.safe_dump(confirmation, sort_keys=False), encoding="utf-8"
    )
    print(json.dumps({"status": analysis["status"], "rates": analysis["merged_confirmation_rates"]}, indent=2))


if __name__ == "__main__":
    main()
