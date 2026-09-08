"""M2D Stage 2's gate: three seeds, one mean per blocker, two verdicts.

Committed before any Stage-2 fit exists, which is the only thing that makes it
a rule rather than a reading. Section 15b fixes what it computes and this file
cannot change it: for each blocker cell, the difference between A3-MINIMAL's
recall@5 and the SAME-SEED S3 recall@5, at seeds 0, 1 and 2, averaged. Both
cells must reach -0.50pp.

Three properties this file is built around, each guarding a specific way the
stage could be quietly won:

**All three seeds, or no verdict.** Seed 0 already exists and reads -0.565pp on
MuSiQue. A mean over whichever seeds happened to land would be a mean over a
selection, so a missing fit returns ``STAGE_2_NOT_MEASURED`` -- not a stop and
not a pass, because neither has been earned.

**Same-seed pairing, and it is checked.** A3-MINIMAL at seed 1 is compared
against S3 at seed 1. Comparing a new seed against seed 0's baseline would mix
the arm's seed effect with the baseline's, which is precisely the noise this
stage was authorised to see through.

**Nothing is refit.** Seed 0's row is the Stage-1 artifact; every S3 and S4 row
comes from M2B's immutable baseline table, which already carries all three
seeds on both blockers. Four fits is the whole cost, and it is four because
the comparisons already existed.

The per-seed sign and the sample SD are computed and reported. They do not
gate. Section 15b says why: this declaration's existing convention reports
``negative_in_every_seed`` beside its three-seed means rather than gating on
it, and adding a second constraint after seed 0 was visible would be choosing
a threshold that the number decides.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

DECLARATION = REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml"
STAGE_1_ROOT = REPO_ROOT / "outputs" / "m2d_s4_semantic_repair" / "stage1"
RESULT_ROOT = REPO_ROOT / "outputs" / "m2d_s4_semantic_repair" / "stage2"
GATE_JSON = REPO_ROOT / "outputs" / "m2d_s4_semantic_repair" / "stage2_gate.json"
GATE_DOC = REPO_ROOT / "docs" / "M2D_STAGE2_GATE.md"
BASELINE_TABLE = (
    REPO_ROOT / "outputs" / "m2c_s4_structural_conditioning" / "m2b_baseline_table.json"
)

#: One arm. Stage 1's control, A1, is not rerun: it added no semantic
#: parameters and its role was to make a seed-0 gain attributable, which is a
#: question seeds cannot reopen.
ARM = "A3_MINIMAL"

INCUMBENT = "S3"
NATIVE = "S4"

#: The three seeds the mean is taken over. Seed 0 is Stage 1's, reused.
SEEDS = (0, 1, 2)
REUSED_SEED = 0
NEW_SEEDS = (1, 2)

#: Percentage points. Section 12's guard, unchanged by section 15b, expressed
#: as a signed floor because the quantity being bounded is a difference.
BLOCKER_BOUND_PP = -0.50

#: Float representation slack, and nothing else. The filed guard reads "within
#: 0.50pp", so a mean of exactly -0.50pp meets it -- but a mean of three
#: differences computed from stored recalls lands a few ulps either side of the
#: boundary for reasons that have nothing to do with the experiment. This is
#: 1e-9 pp: eleven orders of magnitude below the smallest difference any panel
#: in this project can express, so it cannot admit a result the guard excludes.
#: It is written here rather than applied inline so that it can be seen and
#: argued with.
FLOAT_SLACK_PP = 1e-9

#: Every metric section 15b requires reported, against both same-seed rungs.
METRICS = ("recall@1", "recall@5", "recall@20", "mrr")
SELECTOR = "recall@5"

#: The identity evidence each new artifact owes. Latency is NOT here: Stage 2
#: runs no benchmark, and section 15b forbids claiming one from training seeds.
ADDED_SEMANTIC_PARAMETERS = 1536

RESOLVED = "A3_MINIMAL_BLOCKERS_RESOLVED"
CONFIRMED_STOP = "STOP_S4_DEVELOPMENT_CONFIRMED"

#: Not a verdict. Both of the above are claims over three seeds on two cells;
#: this is what the gate returns when it does not have them.
NOT_MEASURED = "STAGE_2_NOT_MEASURED"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def declaration() -> dict[str, Any]:
    return yaml.safe_load(DECLARATION.read_text(encoding="utf-8"))


def declared_cells(config: dict[str, Any]) -> list[str]:
    return list(config["stage_2"]["cells"])


def baseline_rows() -> list[dict[str, Any]]:
    return json.loads(BASELINE_TABLE.read_text(encoding="utf-8"))["rows"]


def baseline(rows: list[dict[str, Any]], cell: str, rung: str, seed: int) -> dict[str, Any]:
    dataset, regime = cell.split("/")
    for row in rows:
        if (row["dataset"], row["regime"], row["rung"], row["seed"]) == (
            dataset,
            regime,
            rung,
            seed,
        ):
            return row
    raise KeyError(f"{cell} {rung} seed {seed} is not in M2B's baseline table")


def _payload(path: Path) -> dict[str, Any]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    return envelope.get("payload", envelope)


def _accept(payload: dict[str, Any], path: Path, status: str, seeds: tuple[int, ...]) -> None:
    """Refuse anything that is not the fit this gate expects.

    Skipping an unreadable file rather than refusing it would let the gate
    return a mean over fewer seeds than it names, which is the one arithmetic
    this stage cannot get wrong.
    """

    if payload.get("status") != status:
        raise ValueError(f"{path} is not a {status} artifact")
    if payload.get("test_split_read"):
        raise ValueError(f"{path} claims to have read the test split")
    if payload.get("arm") != ARM:
        raise ValueError(f"{path} carries arm {payload.get('arm')!r}, not {ARM}")
    if payload.get("seed") not in seeds:
        raise ValueError(f"{path} carries seed {payload.get('seed')!r}, not one of {seeds}")
    unreadable = [
        section
        for section in ("metrics", "parameters")
        if not isinstance(payload.get(section), dict)
    ]
    if unreadable:
        raise ValueError(f"{path} has no {unreadable} block; the gate cannot judge it")
    missing = [metric for metric in METRICS if metric not in payload["metrics"]]
    if missing:
        raise ValueError(f"{path} is missing {missing}")


def load_results(
    root: Path | None = RESULT_ROOT, stage_1_root: Path | None = STAGE_1_ROOT
) -> dict[tuple[str, int], dict[str, Any]]:
    """Every A3-MINIMAL fit this gate judges, keyed by (cell, seed).

    Two sources, deliberately. Seed 0 is read from Stage 1's own artifacts --
    the same files its verdict was computed from, not a refit -- and seeds 1
    and 2 from Stage 2's. If seed 0 were refit here the mean would be over
    three new numbers and Stage 1's row would have been silently replaced.
    """

    results: dict[tuple[str, int], dict[str, Any]] = {}

    if stage_1_root is not None and Path(stage_1_root).is_dir():
        for path in sorted(Path(stage_1_root).glob("*.json")):
            payload = _payload(path)
            if payload.get("arm") != ARM:
                continue  # A1 is Stage 1's control and is not part of this question
            _accept(payload, path, "M2D_STAGE1_ARM_COMPLETE", (REUSED_SEED,))
            results[(payload["cell"], payload["seed"])] = payload

    if root is not None and Path(root).is_dir():
        for path in sorted(Path(root).glob("*.json")):
            payload = _payload(path)
            _accept(payload, path, "M2D_STAGE2_SEED_COMPLETE", NEW_SEEDS)
            key = (payload["cell"], payload["seed"])
            if key in results:
                raise ValueError(f"{path} is a second fit for {key}")
            results[key] = payload

    return results


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def delta_pp(arm_value: float, reference: float) -> float:
    return (arm_value - reference) * 100.0


def per_seed(
    cell: str,
    results: dict[tuple[str, int], dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """One row per seed: the arm, both same-seed rungs, and every delta."""

    table: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        fit = results.get((cell, seed))
        if fit is None:
            table[seed] = {"seed": seed, "measured": False}
            continue
        s3 = baseline(rows, cell, INCUMBENT, seed)
        s4 = baseline(rows, cell, NATIVE, seed)
        table[seed] = {
            "seed": seed,
            "measured": True,
            "reused_from_stage_1": seed == REUSED_SEED,
            "arm": {metric: fit["metrics"][metric] for metric in METRICS},
            INCUMBENT: {metric: s3[metric] for metric in METRICS},
            NATIVE: {metric: s4[metric] for metric in METRICS},
            "vs_s3_pp": {
                metric: delta_pp(fit["metrics"][metric], s3[metric]) for metric in METRICS
            },
            "vs_s4_pp": {
                metric: delta_pp(fit["metrics"][metric], s4[metric]) for metric in METRICS
            },
        }
    return table


def three_seed_summary(table: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """The primary quantity, and the companions that are reported not gated."""

    measured = [row for row in table.values() if row["measured"]]
    if len(measured) != len(SEEDS):
        return {
            "complete": False,
            "seeds_measured": sorted(row["seed"] for row in measured),
            "note": (
                "fewer than three seeds are present, so no mean is computed. A mean "
                "over a subset would be a mean over a selection."
            ),
        }

    selector = [row["vs_s3_pp"][SELECTOR] for row in measured]
    return {
        "complete": True,
        "seeds": list(SEEDS),
        "per_seed_vs_s3_pp": {row["seed"]: row["vs_s3_pp"][SELECTOR] for row in measured},
        "mean_vs_s3_pp": statistics.fmean(selector),
        "sample_sd_pp": statistics.stdev(selector),
        "signs": {row["seed"]: ("+" if row["vs_s3_pp"][SELECTOR] >= 0 else "-") for row in measured},
        "negative_in_every_seed": all(value < 0 for value in selector),
        "mean_vs_s3_pp_by_metric": {
            metric: statistics.fmean([row["vs_s3_pp"][metric] for row in measured])
            for metric in METRICS
        },
        "mean_vs_s4_pp_by_metric": {
            metric: statistics.fmean([row["vs_s4_pp"][metric] for row in measured])
            for metric in METRICS
        },
        "reported_not_gated": (
            "The per-seed signs and the sample SD are reported beside the mean and do "
            "not enter the decision. Section 15b: the declaration's existing "
            "convention reports `negative_in_every_seed` beside its three-seed means "
            "rather than gating on it, and no per-seed floor was ever filed."
        ),
    }


def identity(
    cell: str, results: dict[tuple[str, int], dict[str, Any]]
) -> dict[str, Any]:
    """What a training seed CAN attest to: the arm it actually built.

    Not latency. Stage 2 runs no benchmark and section 15b forbids claiming a
    new p95 from these fits, so this checks the two things that would make a
    Stage-2 fit a different model from the one Stage 1 measured.
    """

    seen: dict[int, dict[str, Any]] = {}
    for seed in NEW_SEEDS:
        fit = results.get((cell, seed))
        if fit is None:
            continue
        parameters = fit["parameters"]
        # Both live where the Stage-1 runner already writes them, so a Stage-2
        # artifact is checked on exactly the fields Stage 1's were, and no new
        # field had to be invented for the gate to have something to read.
        systems = fit.get("systems") or {}
        seen[seed] = {
            "semantic": parameters.get("semantic"),
            "scorer": parameters.get("scorer"),
            "total": parameters.get("total"),
            "added_semantic_parameters": parameters.get("added_semantic_parameters"),
            "added_is_the_declared_1536": (
                parameters.get("added_semantic_parameters") == ADDED_SEMANTIC_PARAMETERS
            ),
            "semantic_difference_precomputed": systems.get(
                "cached_or_precomputed_semantic_difference"
            ),
            "p95_path_identity": systems.get("what_is_timed"),
        }
    return {
        "per_seed": seen,
        # None, not False, when there is nothing to check. "Identity fails" and
        # "no fit has been produced yet" are different states and a report that
        # printed the first for the second would allege a defect in a model
        # that does not exist.
        "holds": (
            all(
                row["added_is_the_declared_1536"]
                and row["semantic_difference_precomputed"] is False
                for row in seen.values()
            )
            if seen
            else None
        ),
        "seeds_checked": sorted(seen),
        "no_latency_claim_is_made_here": True,
    }


def evaluate(
    results: dict[tuple[str, int], dict[str, Any]],
    config: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    cells = declared_cells(config)

    blockers: dict[str, dict[str, Any]] = {}
    for cell in cells:
        table = per_seed(cell, results, rows)
        summary = three_seed_summary(table)
        blockers[cell] = {
            "cell": cell,
            "per_seed": table,
            "three_seed": summary,
            "meets_bound": bool(
                summary["complete"]
                and summary["mean_vs_s3_pp"] >= BLOCKER_BOUND_PP - FLOAT_SLACK_PP
            ),
            "identity": identity(cell, results),
        }

    expected = {(cell, seed) for cell in cells for seed in SEEDS}
    absent = sorted(f"{cell}/seed{seed}" for cell, seed in expected - set(results))
    new_expected = {(cell, seed) for cell in cells for seed in NEW_SEEDS}

    if absent:
        verdict = NOT_MEASURED
        why = (
            f"{len(absent)} of {len(expected)} same-seed rows are not present, so no "
            "three-seed mean exists. Neither verdict has been earned: a confirmed stop "
            "claims three seeds were run and did not reach the guard, and a resolution "
            "claims the same three did."
        )
    elif all(row["meets_bound"] for row in blockers.values()):
        verdict = RESOLVED
        why = (
            "Both blockers' three-seed mean A3_MINIMAL - S3 recall@5 reaches "
            f"{BLOCKER_BOUND_PP:+.2f}pp."
        )
    else:
        short = sorted(cell for cell, row in blockers.items() if not row["meets_bound"])
        verdict = CONFIRMED_STOP
        why = (
            f"{', '.join(short)} does not reach {BLOCKER_BOUND_PP:+.2f}pp on the "
            "three-seed mean. The guard is section 12's and applies to both blockers."
        )

    return {
        "status": "M2D_STAGE2_GATE_EVALUATED",
        "phase": config["phase"],
        "stage": "stage_2",
        "gate_source": "scripts/m2d_stage2_gate.py",
        "gate_filed_before_any_stage_2_fit": True,
        "authorised_by": config["stage_2"]["authorised_by"],
        "arm": ARM,
        "cells": cells,
        "seeds": list(SEEDS),
        "comparison_basis": (
            "A3_MINIMAL at seed n against S3 at seed n, both blockers, seeds 0, 1 and "
            "2. Seed 0 is Stage 1's own artifact and every S3 and S4 row is M2B's; "
            "nothing already fit was refit to produce a comparison row."
        ),
        "bound_pp": BLOCKER_BOUND_PP,
        "aggregate": "arithmetic mean of the three same-seed differences",
        "blockers": blockers,
        "fits_expected": len(new_expected),
        "fits_measured": len(new_expected & set(results)),
        "fits_absent": sorted(
            f"{cell}/seed{seed}" for cell, seed in new_expected - set(results)
        ),
        "rows_expected": len(expected),
        "rows_absent": absent,
        "verdict": verdict,
        "why": why,
        "then": _then(verdict, config),
        "stop": "STOP_FOR_REVIEW",
    }


def _then(verdict: str, config: dict[str, Any]) -> str:
    verdicts = config["stage_2"]["verdicts"]
    if verdict == RESOLVED:
        return verdicts["on_pass_then"]
    if verdict == CONFIRMED_STOP:
        return verdicts["on_fail_then"]
    return "Nothing. Run the declared fits, or report that they could not be run."


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _pp(value: float) -> str:
    return f"{value:+.3f}"


def render(gate: dict[str, Any]) -> str:
    lines = [
        "# M2D Stage 2 — gate",
        "",
        (
            f"Applied by `{gate['gate_source']}`, committed before any Stage-2 fit "
            f"existed. {gate['comparison_basis']}"
        ),
        "",
        f"**Verdict: {gate['verdict']}**",
        "",
        gate["why"],
        "",
        f"Authorised by: {gate['authorised_by']}.",
        "",
        (
            f"Bound: {gate['bound_pp']:+.2f}pp on the {gate['aggregate']}, for "
            f"**both** blockers. {gate['fits_measured']} of {gate['fits_expected']} new "
            f"fits present; {gate['rows_expected'] - len(gate['rows_absent'])} of "
            f"{gate['rows_expected']} same-seed rows."
        ),
        "",
    ]

    for cell, row in gate["blockers"].items():
        lines += [
            f"## {cell}",
            "",
            "| seed | A3 R@1 | A3 R@5 | A3 R@20 | A3 MRR | A3-S3 R@5 pp | A3-S4 R@5 pp |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for seed in gate["seeds"]:
            entry = row["per_seed"][seed]
            if not entry["measured"]:
                lines.append(f"| {seed} | — | — | — | — | — | — |")
                continue
            arm = entry["arm"]
            lines.append(
                f"| {seed}{' *' if entry['reused_from_stage_1'] else ''} "
                f"| {arm['recall@1']:.6f} | {arm['recall@5']:.6f} "
                f"| {arm['recall@20']:.6f} | {arm['mrr']:.6f} "
                f"| {_pp(entry['vs_s3_pp'][SELECTOR])} "
                f"| {_pp(entry['vs_s4_pp'][SELECTOR])} |"
            )
        summary = row["three_seed"]
        lines.append("")
        if summary["complete"]:
            lines += [
                (
                    f"Three-seed mean A3-S3 recall@5: "
                    f"**{_pp(summary['mean_vs_s3_pp'])}pp** "
                    f"(sample SD {summary['sample_sd_pp']:.3f}pp, signs "
                    f"{''.join(summary['signs'][seed] for seed in gate['seeds'])}). "
                    f"Meets {gate['bound_pp']:+.2f}pp: "
                    f"**{'yes' if row['meets_bound'] else 'no'}**."
                ),
                "",
                "| metric | mean A3-S3 pp | mean A3-S4 pp |",
                "| --- | ---: | ---: |",
            ]
            for metric in METRICS:
                lines.append(
                    f"| {metric} | {_pp(summary['mean_vs_s3_pp_by_metric'][metric])} "
                    f"| {_pp(summary['mean_vs_s4_pp_by_metric'][metric])} |"
                )
        else:
            lines.append(summary["note"])
        holds = row["identity"]["holds"]
        lines += [
            "",
            (
                "Identity: no new fit to check yet."
                if holds is None
                else f"Identity holds: {str(holds).lower()} "
                f"(seeds {row['identity']['seeds_checked']})."
            ),
            "",
        ]

    lines += [
        "## What this gate does not say",
        "",
        (
            "The per-seed signs and the sample SD above are reported and do not gate. "
            "No latency result is claimed: Stage 2 ran no benchmark, and Stage 1's p95 "
            "figures remain Stage 1's."
        ),
        "",
        f"**Then:** {gate['then']}",
        "",
        f"**{gate['stop']}**",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULT_ROOT)
    parser.add_argument("--stage-1-results", type=Path, default=STAGE_1_ROOT)
    parser.add_argument("--write", action="store_true", help="persist the verdict")
    args = parser.parse_args(argv)

    gate = evaluate(
        load_results(args.results, args.stage_1_results), declaration(), baseline_rows()
    )
    document = render(gate)
    print(document)

    if args.write:
        GATE_JSON.parent.mkdir(parents=True, exist_ok=True)
        GATE_JSON.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        GATE_DOC.write_text(document, encoding="utf-8")
        print(f"\nwrote {GATE_JSON} and {GATE_DOC}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
