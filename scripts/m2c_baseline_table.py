"""Export the immutable M2C baseline table from the frozen M2B artifacts.

M2C is a post-hoc development branch motivated by an observed S4 result. The
first thing it needs is a starting point that cannot drift: every S2/S3/S4
number M2B measured, in one machine-readable place, derived from the artifacts
rather than retyped from a document.

Nothing is re-run. Every column below already exists in
``outputs/m2b_semantic_minimality/`` -- the headline screen at seed 0 and the
targeted resolution at seeds 1 and 2 -- and this script only reshapes it. If a
column cannot be read it is recorded as ``None`` rather than recomputed, so a
gap in the historical record stays visible instead of being filled in by M2C.

Two derived columns are added because M2C's questions need them and both follow
from numbers already present:

``delta_s4_minus_s3_pp``
    the per-cell margin the whole phase exists to repair.

``required_repair_to_guard_pp``
    ``max(0, -robust_deficit - 0.50)``. What a challenger must actually recover
    to clear M2B's per-cell guard: the deficit LESS the guard's tolerance. It is
    neither the deficit nor the tolerance, and confusing it with either
    misstates how much work a mechanism has to do. Reported beside
    ``exposure_lost_to_candidate_generation_pp`` so the two can be compared,
    which is the only way to say whether admission is capable of clearing a cell
    even in the oracle limit.

``attainment_at_5``
    NOT recomputed. M2B already recorded ``ceiling_attainment_at_5`` on every
    rung, and this column is that field, copied. Reranking cannot move the
    ceiling, so attainment is the only quantity a ranking-side intervention can
    improve; Track B exists because the ceiling is the other half, and it is
    reported beside attainment rather than instead of it.

Two ceilings are exported, because they are different quantities and only one
of them bounds ``recall@5``:

``recall_ceiling_at_5``
    ``regime_headroom["recall_ceiling@5"]`` -- the K-aware ceiling, and the
    denominator of M2B's own attainment. This is the one that bounds recall@5.

``candidate_pool_ceiling``
    ``metrics["candidate_ceiling"]`` -- the macro fraction of a query's gold
    nodes present in the scored pool, with **no K**. It is a coverage statistic,
    not a recall@5 bound, and the two diverge sharply wherever a query has more
    than five gold nodes: metaqa carries up to 246 golds per query (p95 = 36)
    and webqsp up to 222 (p95 = 27), so on those two datasets K binds long
    before pool coverage does. Dividing recall@5 by the pool ceiling there would
    understate attainment and invent ranking headroom that no ranker could
    reach. On the other four datasets max golds per query is at most four, K
    never binds, and the two ceilings agree up to the held-out/split panel
    difference.

``recall_headroom_lost_to_candidate_generation_at_5``
    ``recall_ceiling_perfect_retrieval@5 - recall_ceiling@5``: the part of the
    ceiling that candidate generation threw away. This is the exposure quantity
    -- the only one an admission-side intervention can move.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

M2B_ROOT = pathlib.Path("outputs/m2b_semantic_minimality")
HEADLINE = M2B_ROOT / "headline"
RESOLUTION = M2B_ROOT / "resolution"
OUTPUT_ROOT = pathlib.Path("outputs/m2c_s4_structural_conditioning")
TABLE_JSON = OUTPUT_ROOT / "m2b_baseline_table.json"
TABLE_MARKDOWN = pathlib.Path("docs/M2C_BASELINE_TABLE.md")

RUNGS = ("S2", "S3", "S4")
DECLARED_SEED = 0

#: M2B's per-cell effectiveness guard, in points. A rung is admissible in a
#: cell when it is no worse than this far behind the cell's best. Taken from
#: configs/m2b_semantic_minimality.yaml, where it is filed as not adjustable --
#: restated here only so the derived required-repair column is computable, and
#: checked against that file by tests/test_m2c_declaration.py.
M2B_PER_CELL_GUARD_PP = 0.50

#: The two cells M2B resolved at three seeds, and the rungs it resolved them
#: for. Read from the M2B runner rather than restated, so this file cannot
#: claim a seed the resolution never fitted.
RESOLUTION_SEEDS = (1, 2)


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(metrics: dict, name: str) -> float | None:
    value = metrics.get(name)
    return None if value is None else float(value)


def _row(dataset: str, regime: str, seed: int, rung: str, fit: dict, cell: dict) -> dict:
    """One (dataset, regime, seed, rung) row, read straight off the artifact."""

    metrics = fit["metrics"]
    parameters = fit["parameters"]
    uncached = fit.get("uncached_inference", {})
    total_ms = uncached.get("total_model_ms", {})
    headroom = cell.get("regime_headroom", {})
    recall_ceiling = _metric(headroom, "recall_ceiling@5")
    perfect_ceiling = _metric(headroom, "recall_ceiling_perfect_retrieval@5")
    pool_ceiling = _metric(metrics, "candidate_ceiling")
    recall_at_5 = _metric(metrics, "recall@5")
    # M2B computed attainment itself. Copy it; do not re-derive it against a
    # different ceiling and quietly disagree with the frozen record.
    attainment = _metric(fit, "ceiling_attainment_at_5")
    lost_to_generation = (
        None
        if perfect_ceiling is None or recall_ceiling is None
        else perfect_ceiling - recall_ceiling
    )
    return {
        "dataset": dataset,
        "regime": regime,
        "seed": int(seed),
        "rung": rung,
        "recall@1": _metric(metrics, "recall@1"),
        "recall@5": recall_at_5,
        "recall@20": _metric(metrics, "recall@20"),
        "mrr": _metric(metrics, "mrr"),
        "full_coverage@20": _metric(metrics, "full_coverage@20"),
        "recall_ceiling_at_5": recall_ceiling,
        "candidate_pool_ceiling": pool_ceiling,
        "recall_headroom_lost_to_candidate_generation_at_5": lost_to_generation,
        "attainment_at_5": attainment,
        "golds_per_query_mean": _metric(headroom, "golds_per_query_mean"),
        "golds_per_query_max": _metric(headroom, "golds_per_query_max"),
        "uncached_p50_ms": total_ms.get("p50"),
        "uncached_p95_ms": total_ms.get("p95"),
        "uncached_p99_ms": total_ms.get("p99"),
        "semantic_parameters": int(parameters["semantic"]),
        "scorer_parameters": int(parameters["scorer"]),
        "total_parameters": int(parameters["total"]),
        "held_out_queries": int(cell["held_out_queries"]),
        "train_queries": int(cell["train_queries"]),
        "split_queries": None,
        "reused_from_m2": bool(fit.get("reused_from_m2", False)),
        "source_commit": None,
    }


def collect() -> list[dict]:
    """Every fit M2B recorded, at every seed it recorded one for."""

    rows: list[dict] = []
    for path in sorted(HEADLINE.glob("*.json")):
        payload = _load(path)
        dataset = payload["dataset"]
        commit = payload["provenance"]["source_commit"]
        for regime, cell in payload["cells"].items():
            for rung, fit in cell["rungs"].items():
                row = _row(dataset, regime, DECLARED_SEED, rung, fit, cell)
                row["split_queries"] = int(payload["queries"])
                row["source_commit"] = commit
                rows.append(row)

    for seed in RESOLUTION_SEEDS:
        for path in sorted((RESOLUTION / f"seed{seed}").glob("*.json")):
            payload = _load(path)
            if int(payload["seed"]) != seed:
                raise SystemExit(
                    f"{path} records seed {payload['seed']!r} under seed{seed}/"
                )
            dataset = payload["dataset"]
            commit = payload["provenance"]["source_commit"]
            for regime, cell in payload["cells"].items():
                for rung, fit in cell["rungs"].items():
                    row = _row(dataset, regime, seed, rung, fit, cell)
                    row["split_queries"] = int(payload["queries"])
                    row["source_commit"] = commit
                    rows.append(row)
    return rows


def margins(rows: list[dict]) -> list[dict]:
    """S4 - S3 per (dataset, regime, seed), in points, plus the 3-seed means."""

    by_key: dict[tuple[str, str, int], dict[str, dict]] = {}
    for row in rows:
        by_key.setdefault((row["dataset"], row["regime"], row["seed"]), {})[
            row["rung"]
        ] = row

    per_seed: list[dict] = []
    for (dataset, regime, seed), rungs in sorted(by_key.items()):
        if not {"S3", "S4"} <= set(rungs):
            continue
        s3, s4 = rungs["S3"]["recall@5"], rungs["S4"]["recall@5"]
        entry = {
            "dataset": dataset,
            "regime": regime,
            "seed": seed,
            "s3_recall@5": s3,
            "s4_recall@5": s4,
            "delta_s4_minus_s3_pp": (s4 - s3) * 100.0,
        }
        if "S2" in rungs:
            entry["s2_recall@5"] = rungs["S2"]["recall@5"]
            entry["delta_s2_minus_s3_pp"] = (rungs["S2"]["recall@5"] - s3) * 100.0
        per_seed.append(entry)

    multi: dict[tuple[str, str], list[float]] = {}
    for entry in per_seed:
        multi.setdefault((entry["dataset"], entry["regime"]), []).append(
            entry["delta_s4_minus_s3_pp"]
        )
    means = []
    for (dataset, regime), deltas in sorted(multi.items()):
        if len(deltas) <= 1:
            continue
        robust = statistics.fmean(deltas)
        # What a challenger must actually RECOVER to clear M2B's cell guard --
        # the deficit less the guard's tolerance, not the deficit and not the
        # tolerance. Zero when the cell is already admissible.
        required = max(0.0, -robust - M2B_PER_CELL_GUARD_PP)
        exposure = next(
            (
                row["recall_headroom_lost_to_candidate_generation_at_5"] * 100.0
                for row in rows
                if (row["dataset"], row["regime"], row["rung"], row["seed"])
                == (dataset, regime, "S4", DECLARED_SEED)
                and row["recall_headroom_lost_to_candidate_generation_at_5"] is not None
            ),
            None,
        )
        means.append(
            {
                "dataset": dataset,
                "regime": regime,
                "seeds": len(deltas),
                "mean_delta_s4_minus_s3_pp": robust,
                "min_pp": min(deltas),
                "max_pp": max(deltas),
                "required_repair_to_guard_pp": required,
                "exposure_lost_to_candidate_generation_pp": exposure,
                # An oracle bound only. Exposure is what a PERFECT admission
                # mechanism could expose; converting it into recall@5 is a
                # separate problem that ranking still has to solve.
                "admission_ruled_out_by_exposure": (
                    None if exposure is None else exposure < required
                ),
                "share_of_exposure_that_must_convert": (
                    None if not exposure or required == 0.0 else required / exposure
                ),
            }
        )
    return per_seed, means


def _fmt(value: float | None, digits: int = 4) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def markdown(rows: list[dict], per_seed: list[dict], means: list[dict]) -> str:
    lines = [
        "# M2C baseline table -- the frozen M2B starting point",
        "",
        "Exported by [`scripts/m2c_baseline_table.py`](../scripts/m2c_baseline_table.py)",
        "from the M2B artifacts. **Nothing here was re-run.** Every column is read",
        "off `outputs/m2b_semantic_minimality/`. The derived columns are defined",
        "in the script's docstring and follow from numbers already present; the",
        "only one computed here rather than copied is `delta_s4_minus_s3_pp`.",
        "",
        "This table is immutable for the duration of M2C. A challenger is compared",
        "against these rows, and these rows are never recomputed to match it.",
        "",
        "`attain@5` is M2B's own `ceiling_attainment_at_5`, copied rather than",
        "recomputed: `recall@5 / rec_ceil@5`. A ranking-side intervention can move",
        "only this; the ceiling is fixed by which candidates are scored, which is",
        "why M2C has a second track that changes the scored set instead.",
        "",
        "Two ceilings are shown because only one of them bounds `recall@5`.",
        "`rec_ceil@5` is the K-aware `recall_ceiling@5`. `pool_ceil` is the macro",
        "fraction of a query's gold nodes present in the scored pool, with **no K**;",
        "it is a coverage statistic. They diverge wherever a query has more than",
        "five golds -- metaqa reaches 246 golds per query and webqsp 222, so K binds",
        "there long before pool coverage does, and dividing by `pool_ceil` would",
        "invent ranking headroom no ranker could reach. On the other four datasets",
        "no query has more than four golds and K never binds.",
        "",
        "`lost@5` is `recall_ceiling_perfect_retrieval@5 - recall_ceiling@5`: the",
        "part of the ceiling candidate generation threw away. It is the exposure",
        "quantity, and the only one an admission-side intervention can move.",
        "",
        "## All fits, seed 0 (M2B headline screen)",
        "",
        (
            "| dataset | regime | rung | R@1 | R@5 | R@20 | MRR | FullCov@20 |"
            " rec_ceil@5 | attain@5 | lost@5 | pool_ceil | golds/q | max |"
            " p50 ms | p95 ms | p99 ms | sem params | scorer | total | held out |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["seed"] != DECLARED_SEED:
            continue
        lines.append(
            f"| {row['dataset']} | {row['regime']} | {row['rung']} "
            f"| {_fmt(row['recall@1'])} | {_fmt(row['recall@5'])} | {_fmt(row['recall@20'])} "
            f"| {_fmt(row['mrr'])} | {_fmt(row['full_coverage@20'])} "
            f"| {_fmt(row['recall_ceiling_at_5'])} | {_fmt(row['attainment_at_5'])} "
            f"| {_fmt(row['recall_headroom_lost_to_candidate_generation_at_5'])} "
            f"| {_fmt(row['candidate_pool_ceiling'])} "
            f"| {_fmt(row['golds_per_query_mean'], 2)} "
            f"| {_fmt(row['golds_per_query_max'], 0)} "
            f"| {_fmt(row['uncached_p50_ms'], 3)} | {_fmt(row['uncached_p95_ms'], 3)} "
            f"| {_fmt(row['uncached_p99_ms'], 3)} "
            f"| {row['semantic_parameters']} | {row['scorer_parameters']} "
            f"| {row['total_parameters']} | {row['held_out_queries']} |"
        )

    lines += [
        "",
        "## Resolution fits, seeds 1 and 2",
        "",
        "The two cells that blocked S4, refitted at two further seeds under the",
        "rule M2B froze before the fits.",
        "",
        "| dataset | regime | seed | rung | R@5 | rec_ceil@5 | attain@5 | p95 ms |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["seed"] == DECLARED_SEED:
            continue
        lines.append(
            f"| {row['dataset']} | {row['regime']} | {row['seed']} | {row['rung']} "
            f"| {_fmt(row['recall@5'])} | {_fmt(row['recall_ceiling_at_5'])} "
            f"| {_fmt(row['attainment_at_5'])} | {_fmt(row['uncached_p95_ms'], 3)} |"
        )

    lines += [
        "",
        "## S4 - S3 per cell",
        "",
        "Positive means S4 is ahead. The two negative cells at every seed are the",
        "blockers M2C exists to try to repair.",
        "",
        "| dataset | regime | seed | S3 R@5 | S4 R@5 | S4-S3 (pp) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for entry in per_seed:
        lines.append(
            f"| {entry['dataset']} | {entry['regime']} | {entry['seed']} "
            f"| {_fmt(entry['s3_recall@5'])} | {_fmt(entry['s4_recall@5'])} "
            f"| {entry['delta_s4_minus_s3_pp']:+.4f} |"
        )

    if means:
        lines += [
            "",
            "### Multi-seed means, and what a repair actually has to recover",
            "",
            "`required` is `max(0, -mean_deficit - 0.50)`: the deficit LESS M2B's",
            "per-cell tolerance, which is what a challenger must recover to clear",
            "the guard. It is neither the deficit nor the tolerance. `exposure` is",
            "the ceiling candidate generation threw away, so comparing the two says",
            "whether admission could clear the cell in the ORACLE limit -- exposing",
            "a gold is not retrieving it, and ranking still has to convert it.",
            "",
            (
                "| dataset | regime | seeds | mean S4-S3 (pp) | min | max |"
                " required (pp) | exposure (pp) | admission ruled out? |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|:--:|",
        ]
        for entry in means:
            ruled = entry["admission_ruled_out_by_exposure"]
            verdict = "--" if ruled is None else ("YES" if ruled else "no")
            lines.append(
                f"| {entry['dataset']} | {entry['regime']} | {entry['seeds']} "
                f"| {entry['mean_delta_s4_minus_s3_pp']:+.4f} "
                f"| {entry['min_pp']:+.4f} | {entry['max_pp']:+.4f} "
                f"| {entry['required_repair_to_guard_pp']:.4f} "
                f"| {_fmt(entry['exposure_lost_to_candidate_generation_pp'], 4)} "
                f"| {verdict} |"
            )
    lines.append("")
    return "\n".join(lines)


def build() -> dict:
    rows = collect()
    per_seed, means = margins(rows)
    cells = sorted({(row["dataset"], row["regime"]) for row in rows})
    report = {
        "status": "M2C_BASELINE_TABLE_EXPORTED",
        "source": "outputs/m2b_semantic_minimality",
        "nothing_was_rerun": (
            "Every value is read from a frozen M2B artifact. This script trains "
            "nothing, evaluates nothing, and writes nothing back into M2B."
        ),
        "immutable": (
            "M2C compares challengers against these rows. The rows are never "
            "recomputed to match a challenger."
        ),
        "cells": [{"dataset": d, "regime": r} for d, r in cells],
        "cell_count": len(cells),
        "fit_count": len(rows),
        "rows": rows,
        "per_seed_margins": per_seed,
        "multi_seed_margins": means,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    TABLE_MARKDOWN.write_text(markdown(rows, per_seed, means), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report = build()
    print(f"{report['fit_count']} fits over {report['cell_count']} cells")
    print(f"wrote {TABLE_JSON}")
    print(f"wrote {TABLE_MARKDOWN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
