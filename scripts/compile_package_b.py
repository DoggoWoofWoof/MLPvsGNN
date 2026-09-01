#!/usr/bin/env python
"""Join Package B edge families to the candidate ceiling they all share.

Package B changes only the global edge family. Every family therefore receives
the same frozen candidate pool, so one ceiling per dataset governs all of them.
That has a direct consequence for how B may be read:

* the absolute level a family reaches is bounded by an upstream candidate
  constraint that no edge family can move, and
* the difference between families is a topology and ranking effect, because the
  ceiling cancels between them.

The report also puts ``symbolic_b`` beside ``knn_only`` explicitly. ``symbolic_b``
is structural and NER edges -- genuine relational topology. ``knn_only`` is the
embedding-similarity edges alone. Reporting them side by side is what keeps the
question "is message passing exploiting relational structure, or embedding
similarity reintroduced as edges?" answerable rather than assumed.

Read-only. Consumes finished outputs and writes a report; it computes no metric
and trains nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("sa_mlp", "seed_aware_gnn")
PUBLICATION_NAMES = {"sa_mlp": "QLS-MLP", "seed_aware_gnn": "Seed-aware GNN"}
RECALL_KS = (1, 5, 20)
HEADROOM_COMPLETE = "CANDIDATE_HEADROOM_DIAGNOSTIC_COMPLETE"
PROVENANCE_ANALYZED = "EDGE_PROVENANCE_ALL_DATASETS_ANALYZED"

# Package B trains on the frozen union pool itself, not on a budget subset.
POOL_KEY = "frozen_union"

# The pair that answers the provenance question.
RELATIONAL_FAMILY = "symbolic_b"
SIMILARITY_FAMILY = "knn_only"


def attainment(achieved: float, ceiling: float) -> float | None:
    """Fraction of the reachable ceiling a model actually reached."""
    if ceiling <= 0.0:
        return None
    return float(achieved) / float(ceiling)


def _fmt(value: float | None, places: int = 4) -> str:
    return "" if value is None else f"{value:.{places}f}"


def _contract_sha(root: Path, dataset: str) -> str:
    """The candidate contract Package B actually trained against."""
    results = sorted(root.glob(f"{dataset}/*/result.json"))
    if not results:
        raise FileNotFoundError(f"No Package B result for {dataset} under {root}")
    shas = set()
    for path in results:
        payload = json.loads(path.read_text(encoding="utf-8"))
        shas.add(payload["candidate_contract"]["observed_contract_sha256"])
    if len(shas) != 1:
        raise ValueError(
            f"{dataset} Package B families disagree on the candidate contract: {sorted(shas)}"
        )
    return shas.pop()


def _pool(headroom_root: Path, dataset: str, contract_sha: str) -> dict[str, Any]:
    """The frozen-union ceiling, refused unless it describes B's own pool."""
    payload = json.loads((headroom_root / f"{dataset}.json").read_text(encoding="utf-8"))
    if payload["status"] != HEADROOM_COMPLETE:
        raise ValueError(f"{dataset} headroom diagnostic is not complete")
    observed = payload["candidate_contract"]["observed_contract_sha256"]
    if observed != contract_sha:
        raise ValueError(
            f"{dataset} headroom measured pool {observed[:12]} but Package B trained "
            f"against {contract_sha[:12]}; the ceiling would not describe these results"
        )
    return payload["headroom"]["test"][POOL_KEY]


def compile_joint(analysis: dict[str, Any], root: Path, headroom_root: Path) -> dict[str, Any]:
    if analysis.get("status") != PROVENANCE_ANALYZED:
        raise ValueError(
            f"Refusing to compile Package B from a {analysis.get('status')!r} analysis; "
            f"only {PROVENANCE_ANALYZED} may be reported"
        )
    pools: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for row in analysis["datasets"]:
        dataset = row["dataset"]
        if dataset not in pools:
            pools[dataset] = _pool(headroom_root, dataset, _contract_sha(root, dataset))
        pool = pools[dataset]
        achieved = {
            model: {
                metric: float(row["models"][model][metric]["mean"])
                for metric in (*(f"recall@{k}" for k in RECALL_KS), "mrr", "full_coverage@20")
            }
            for model in MODEL_NAMES
        }
        gaps = {
            model: {
                **{
                    f"ceiling_minus_recall@{k}": pool[f"recall_ceiling@{k}"]
                    - achieved[model][f"recall@{k}"]
                    for k in RECALL_KS
                },
                **{
                    f"attainment@{k}": attainment(
                        achieved[model][f"recall@{k}"], pool[f"recall_ceiling@{k}"]
                    )
                    for k in RECALL_KS
                },
            }
            for model in MODEL_NAMES
        }
        rows.append(
            {
                "dataset": dataset,
                "family": row["family"],
                "directed_edges": row["directed_edges"],
                "candidate_pool": {
                    "pool": POOL_KEY,
                    "coverage_micro": pool["coverage_micro"],
                    "gold_fraction_at_pool_macro": pool["gold_fraction_at_pool_macro"],
                    "any_gold_at_pool": pool["any_gold_at_pool"],
                    "all_gold_at_pool": pool["all_gold_at_pool"],
                    **{f"recall_ceiling@{k}": pool[f"recall_ceiling@{k}"] for k in RECALL_KS},
                },
                "achieved": achieved,
                "ceiling_gaps": gaps,
                "contrast": row["seed_aware_gnn_minus_sa_mlp"],
            }
        )

    provenance = []
    for dataset in dict.fromkeys(row["dataset"] for row in rows):
        by_family = {row["family"]: row for row in rows if row["dataset"] == dataset}
        relational = by_family.get(RELATIONAL_FAMILY)
        similarity = by_family.get(SIMILARITY_FAMILY)
        if relational is None or similarity is None:
            continue
        provenance.append(
            {
                "dataset": dataset,
                "relational_family": RELATIONAL_FAMILY,
                "similarity_family": SIMILARITY_FAMILY,
                "models": {
                    model: {
                        f"recall@{k}": {
                            "relational": relational["achieved"][model][f"recall@{k}"],
                            "similarity": similarity["achieved"][model][f"recall@{k}"],
                            "relational_minus_similarity": (
                                relational["achieved"][model][f"recall@{k}"]
                                - similarity["achieved"][model][f"recall@{k}"]
                            ),
                        }
                        for k in RECALL_KS
                    }
                    for model in MODEL_NAMES
                },
                "directed_edges": {
                    "relational": relational["directed_edges"],
                    "similarity": similarity["directed_edges"],
                },
            }
        )

    return {
        "status": "EDGE_PROVENANCE_AND_HEADROOM_JOINED",
        "source_analysis_status": analysis["status"],
        "pool": POOL_KEY,
        "rows": rows,
        "provenance_contrast": provenance,
        "claims": {
            "ceiling_is_common_across_edge_families": True,
            "family_differences_are_topology_effects_not_ceiling_effects": True,
            "ceiling_is_diagnostic_only_and_never_given_to_a_model": True,
            "candidate_pools_unchanged_by_this_report": True,
        },
    }


def render_markdown(joint: dict[str, Any]) -> str:
    lines = [
        "# Package B: edge families against their common candidate ceiling",
        "",
        (
            "Package B changes only the global edge family, so every family receives the "
            "same frozen candidate pool and one ceiling per dataset governs all of them. "
            "The absolute level is therefore an upstream candidate constraint that no edge "
            "family can move, while the difference between families is a topology and "
            "ranking effect, because the ceiling cancels between them."
        ),
        "",
        "## Families against the shared ceiling",
        "",
        (
            "| Dataset | Family | Edges | Ceil@5 | QLS R@5 | GNN R@5 | Ceil-QLS | Ceil-GNN | "
            "QLS att. | GNN att. |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in joint["rows"]:
        pool, got, gap = row["candidate_pool"], row["achieved"], row["ceiling_gaps"]
        lines.append(
            f"| {row['dataset']} | {row['family']} | {row['directed_edges']} | "
            f"{pool['recall_ceiling@5']:.4f} | "
            f"{got['sa_mlp']['recall@5']:.4f} | {got['seed_aware_gnn']['recall@5']:.4f} | "
            f"{gap['sa_mlp']['ceiling_minus_recall@5']:.4f} | "
            f"{gap['seed_aware_gnn']['ceiling_minus_recall@5']:.4f} | "
            f"{_fmt(gap['sa_mlp']['attainment@5'], 3)} | "
            f"{_fmt(gap['seed_aware_gnn']['attainment@5'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## Every reported cut-off",
            "",
            (
                "| Dataset | Family | Ceil@1 | QLS R@1 | GNN R@1 | Ceil@5 | QLS R@5 | GNN R@5 | "
                "Ceil@20 | QLS R@20 | GNN R@20 | QLS MRR | GNN MRR |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in joint["rows"]:
        pool, got = row["candidate_pool"], row["achieved"]
        qls, gnn = got["sa_mlp"], got["seed_aware_gnn"]
        lines.append(
            f"| {row['dataset']} | {row['family']} | "
            f"{pool['recall_ceiling@1']:.4f} | {qls['recall@1']:.4f} | {gnn['recall@1']:.4f} | "
            f"{pool['recall_ceiling@5']:.4f} | {qls['recall@5']:.4f} | {gnn['recall@5']:.4f} | "
            f"{pool['recall_ceiling@20']:.4f} | {qls['recall@20']:.4f} | "
            f"{gnn['recall@20']:.4f} | {qls['mrr']:.4f} | {gnn['mrr']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Relational topology against embedding similarity",
            "",
            (
                f"`{RELATIONAL_FAMILY}` is structural and NER edges: genuine relational "
                f"topology. `{SIMILARITY_FAMILY}` is the embedding-similarity edges alone. "
                "A positive difference means the family built from relations outperforms the "
                "family built from embedding neighborhoods on the same candidates and the "
                "same ceiling. A difference near zero means the message passing was "
                "exploiting embedding similarity reintroduced as edges rather than relational "
                "structure, and must be reported as such."
            ),
            "",
            (
                "| Dataset | Model | Relational R@5 | Similarity R@5 | Relational - Similarity | "
                "Relational edges | Similarity edges |"
            ),
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for entry in joint["provenance_contrast"]:
        for model in MODEL_NAMES:
            cell = entry["models"][model]["recall@5"]
            lines.append(
                f"| {entry['dataset']} | {PUBLICATION_NAMES[model]} | "
                f"{cell['relational']:.4f} | {cell['similarity']:.4f} | "
                f"{cell['relational_minus_similarity']:+.4f} | "
                f"{entry['directed_edges']['relational']} | "
                f"{entry['directed_edges']['similarity']} |"
            )

    lines.extend(
        [
            "",
            (
                "The ceiling is a diagnostic and is never given to a model. It is identical "
                "across the families of a dataset by construction, so it explains none of the "
                "differences in this report."
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
        default=REPO_ROOT / "outputs" / "edge_provenance_analysis.json",
        help="Package B analysis produced by analyze_edge_provenance.py",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "outputs" / "edge_provenance",
        help="Package B results, read only for their candidate contract hash",
    )
    parser.add_argument(
        "--headroom-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "candidate_headroom",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "edge_provenance_headroom.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=REPO_ROOT / "docs" / "EDGE_PROVENANCE_AND_HEADROOM_RESULTS.md",
        help=(
            "Joint report. Kept separate from EDGE_PROVENANCE_RESULTS.md, which "
            "analyze_edge_provenance.py owns, so neither overwrites the other."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    joint = compile_joint(analysis, args.root, args.headroom_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(joint, indent=2), encoding="utf-8")
    args.markdown.write_text(render_markdown(joint), encoding="utf-8")
    print(f"Wrote {args.output} and {args.markdown}")


if __name__ == "__main__":
    main()
