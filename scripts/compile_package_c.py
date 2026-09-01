#!/usr/bin/env python
"""Join Package C effectiveness with the candidate-headroom ceilings.

Package C reports what the rankers achieved. The headroom diagnostic reports
what their candidate pools allowed. Neither is interpretable alone: the ceiling
rises with budget, so a raw gain across the sweep is partly the ceiling moving
and partly the ranker improving. This compiler reports both beside each other
and separates those two causes explicitly.

Read-only. It consumes finished Package C and headroom outputs and writes a
combined report; it trains nothing and modifies no candidate pool.
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "candidate_budget.yaml"
MODEL_NAMES = ("sa_mlp", "seed_aware_gnn")
PUBLICATION_NAMES = {"sa_mlp": "QLS-MLP", "seed_aware_gnn": "Seed-aware GNN"}
RECALL_KS = (1, 5, 20)
HEADROOM_COMPLETE = "CANDIDATE_HEADROOM_DIAGNOSTIC_COMPLETE"
BUDGET_ANALYZED = "CANDIDATE_BUDGET_ALL_DATASETS_ANALYZED"


def attainment(achieved: float, ceiling: float) -> float | None:
    """Fraction of what the candidate pool allowed that the ranker actually got.

    ``None`` when the ceiling is zero, because no ranking could score there and
    a ratio would be an artefact rather than a measurement.
    """
    if ceiling <= 0.0:
        return None
    return float(achieved) / float(ceiling)


def decompose_budget_step(
    lower_recall: float,
    lower_ceiling: float,
    upper_recall: float,
    upper_ceiling: float,
) -> dict[str, float | None]:
    """Split a recall change across a budget step into ceiling and ranking parts.

    Recall is ``attainment * ceiling``, so between two budgets

        d_recall = a0 * (c1 - c0)  +  c1 * (a1 - a0)
                   \\___________/     \\___________/
                    ceiling effect     ranking effect

    The two terms are exact and sum to the observed change, with no residual.
    The split is order-dependent by construction -- the ceiling effect is
    evaluated at the lower attainment and the ranking effect at the upper
    ceiling -- and that ordering is fixed here so it cannot be chosen after
    seeing which attribution looks better.
    """
    lower_attainment = attainment(lower_recall, lower_ceiling)
    upper_attainment = attainment(upper_recall, upper_ceiling)
    observed = float(upper_recall) - float(lower_recall)
    if lower_attainment is None or upper_attainment is None:
        return {
            "observed_recall_change": observed,
            "ceiling_effect": None,
            "ranking_effect": None,
            "attributable": False,
        }
    ceiling_effect = lower_attainment * (float(upper_ceiling) - float(lower_ceiling))
    ranking_effect = float(upper_ceiling) * (upper_attainment - lower_attainment)
    return {
        "observed_recall_change": observed,
        "ceiling_effect": ceiling_effect,
        "ranking_effect": ranking_effect,
        "attributable": True,
    }


def _headroom_for_budget(headroom: dict[str, Any], budget: int) -> dict[str, Any]:
    key = f"equal_rrf_budget_{budget}"
    test = headroom["headroom"]["test"]
    if key not in test:
        raise KeyError(f"Headroom diagnostic has no {key} for {headroom['dataset']}")
    return test[key]


def compile_joint(analysis: dict[str, Any], headroom_root: Path) -> dict[str, Any]:
    """Combine a finished Package C analysis with the headroom ceilings."""
    if analysis.get("status") != BUDGET_ANALYZED:
        raise ValueError("Package C analysis is not complete")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    budgets = list(map(int, config["candidate_contract"]["budgets"]))

    headrooms: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for row in analysis["datasets"]:
        dataset = row["dataset"]
        if dataset not in headrooms:
            path = headroom_root / f"{dataset}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Headroom diagnostic is missing: {path}")
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if (
                loaded.get("status") != HEADROOM_COMPLETE
                or loaded.get("dataset") != dataset
                or loaded["diagnostic_contract"].get("candidate_pools_modified") is not False
            ):
                raise ValueError(f"Headroom diagnostic contract failed: {path}")
            headrooms[dataset] = loaded
        headroom = headrooms[dataset]
        budget = int(row["budget"])
        pool = _headroom_for_budget(headroom, budget)

        achieved: dict[str, dict[str, float]] = {}
        gaps: dict[str, dict[str, float | None]] = {}
        for model in MODEL_NAMES:
            achieved[model] = {
                f"recall@{k}": float(row["models"][model][f"recall@{k}"]["mean"])
                for k in RECALL_KS
            }
            achieved[model]["mrr"] = float(row["models"][model]["mrr"]["mean"])
            achieved[model]["full_coverage@20"] = float(
                row["models"][model]["full_coverage@20"]["mean"]
            )
            gaps[model] = {}
            for k in RECALL_KS:
                ceiling = float(pool[f"recall_ceiling@{k}"])
                value = achieved[model][f"recall@{k}"]
                gaps[model][f"ceiling_minus_recall@{k}"] = ceiling - value
                gaps[model][f"attainment@{k}"] = attainment(value, ceiling)

        rows.append(
            {
                "dataset": dataset,
                "budget": budget,
                "candidate_pool": {
                    "coverage_micro": float(pool["coverage_micro"]),
                    "gold_fraction_at_pool_macro": float(
                        pool["gold_fraction_at_pool_macro"]
                    ),
                    "any_gold_at_pool": float(pool["any_gold_at_pool"]),
                    "all_gold_at_pool": float(pool["all_gold_at_pool"]),
                    "queries_with_no_gold_in_pool": int(
                        pool["queries_with_no_gold_in_pool"]
                    ),
                    **{
                        f"recall_ceiling@{k}": float(pool[f"recall_ceiling@{k}"])
                        for k in RECALL_KS
                    },
                    "full_coverage_ceiling@20": float(pool["hit_ceiling@20"]),
                },
                "achieved": achieved,
                "ceiling_gaps": gaps,
                "contrast": row["seed_aware_gnn_minus_sa_mlp"],
                "structural_context": row["context"],
                "cached_operator_cost": row["systems"],
            }
        )

    steps: list[dict[str, Any]] = []
    for dataset in dict.fromkeys(row["dataset"] for row in rows):
        by_budget = {
            row["budget"]: row for row in rows if row["dataset"] == dataset
        }
        for lower, upper in pairwise(budgets):
            if lower not in by_budget or upper not in by_budget:
                continue
            low_row, high_row = by_budget[lower], by_budget[upper]
            steps.append(
                {
                    "dataset": dataset,
                    "from_budget": lower,
                    "to_budget": upper,
                    "models": {
                        model: {
                            f"recall@{k}": decompose_budget_step(
                                low_row["achieved"][model][f"recall@{k}"],
                                low_row["candidate_pool"][f"recall_ceiling@{k}"],
                                high_row["achieved"][model][f"recall@{k}"],
                                high_row["candidate_pool"][f"recall_ceiling@{k}"],
                            )
                            for k in RECALL_KS
                        }
                        for model in MODEL_NAMES
                    },
                }
            )

    return {
        "status": "CANDIDATE_BUDGET_WITH_HEADROOM_COMPILED",
        "budgets": budgets,
        "rows": rows,
        "budget_steps": steps,
        "bootstrap": analysis["bootstrap"],
        "claims": {
            **analysis["claims"],
            "ceilings_reported_beside_every_metric": True,
            "budget_gain_decomposed_into_ceiling_and_ranking": True,
            "coverage_is_not_an_oracle_recall": True,
            "candidate_pools_modified": False,
        },
    }


def _fmt(value: float | None, places: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def render_markdown(joint: dict[str, Any]) -> str:
    lines = [
        "# Package C: candidate budgets against their candidate ceilings",
        "",
        (
            "Every effectiveness number appears beside the ceiling its candidate pool "
            "allowed. For a query with `g` golds, `p` of them in the pool, and cut-off "
            "`K`, the achievable Recall@K is `min(p, K) / g`. Pool coverage `p / g` is "
            "not an oracle Recall@K and is reported as a separate column."
        ),
        "",
        "## Pools and attainment",
        "",
        (
            "| Dataset | Budget | Coverage | GoldFrac | AnyGold | AllGold | Ceil@5 | "
            "QLS R@5 | GNN R@5 | Ceil-QLS | Ceil-GNN | QLS att. | GNN att. |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in joint["rows"]:
        pool, got, gap = row["candidate_pool"], row["achieved"], row["ceiling_gaps"]
        lines.append(
            f"| {row['dataset']} | {row['budget']} | "
            f"{pool['coverage_micro']:.4f} | "
            f"{pool['gold_fraction_at_pool_macro']:.4f} | "
            f"{pool['any_gold_at_pool']:.4f} | {pool['all_gold_at_pool']:.4f} | "
            f"{pool['recall_ceiling@5']:.4f} | "
            f"{got['sa_mlp']['recall@5']:.4f} | "
            f"{got['seed_aware_gnn']['recall@5']:.4f} | "
            f"{gap['sa_mlp']['ceiling_minus_recall@5']:.4f} | "
            f"{gap['seed_aware_gnn']['ceiling_minus_recall@5']:.4f} | "
            f"{_fmt(gap['sa_mlp']['attainment@5'], 3)} | "
            f"{_fmt(gap['seed_aware_gnn']['attainment@5'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## What a budget increase actually bought",
            "",
            (
                "Recall is `attainment x ceiling`, so a change across a budget step "
                "splits exactly into the ceiling moving and the ranker improving. A "
                "raw gain is not a ranking improvement when the ceiling rose with it."
            ),
            "",
            "| Dataset | Step | Model | d Recall@5 | from ceiling | from ranking |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for step in joint["budget_steps"]:
        for model, metrics in step["models"].items():
            entry = metrics["recall@5"]
            lines.append(
                f"| {step['dataset']} | {step['from_budget']}->{step['to_budget']} | "
                f"{PUBLICATION_NAMES[model]} | "
                f"{entry['observed_recall_change']:+.4f} | "
                f"{_fmt(entry['ceiling_effect'])} | {_fmt(entry['ranking_effect'])} |"
            )
    lines.extend(
        [
            "",
            (
                "Attainment is blank where the ceiling is zero: no ranking can score "
                "against an empty pool, so a ratio there would be an artefact."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        type=Path,
        default=REPO_ROOT / "outputs" / "candidate_budget_analysis.json",
        help="Package C analysis produced by analyze_candidate_budget.py",
    )
    parser.add_argument(
        "--headroom-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "candidate_headroom",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "candidate_budget_headroom.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=REPO_ROOT / "docs" / "CANDIDATE_BUDGET_AND_HEADROOM_RESULTS.md",
        help=(
            "Joint report. Kept separate from CANDIDATE_BUDGET_RESULTS.md, which "
            "analyze_candidate_budget.py owns, so neither overwrites the other."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    joint = compile_joint(analysis, args.headroom_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(joint, indent=2), encoding="utf-8")
    args.markdown.write_text(render_markdown(joint), encoding="utf-8")
    print(f"Wrote {args.output} and {args.markdown}")


if __name__ == "__main__":
    main()
