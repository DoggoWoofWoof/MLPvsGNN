"""M2D Stage 2's compute record: four fits, priced from four that already ran.

Filed before any Stage-2 job is submitted, and derived from measurement rather
than from prediction. Stage 1 bought A3-MINIMAL on both of these cells at seed
0 and its artifacts recorded, per cell, how long the fit took, how long scoring
the held-out panel took, and what a single uncached forward pass costs. Stage 2
runs the same arm on the same cells at two more seeds, so the honest basis is
those numbers, not a model of them.

That is the whole difference from Stage 1's record. Stage 1 had to predict:
it scaled M2B's measured S4 seconds by a forward multiplier measured on the
host, because no M2D arm had ever been fit. Stage 2 does not have to, and a
record that predicted anyway -- reusing the multiplier chain when the thing it
was estimating has since been measured -- would be choosing a model over the
observation it was built to approximate.

Two consequences worth stating rather than leaving implicit.

**The safety factor is small on purpose.** Section 6 says no large multiplier
without justification. The largest source of error in Stage 1's prediction was
the multiplier; this record has no multiplier, and the same code will run the
same panel on the same container shape. What is left is seed-to-seed variation
in training time, which Stage 1's own artifacts bound: A1 and A3-MINIMAL differ
by 2.0% on SQuAD and 0.8% on MuSiQue at fixed seed, on models of different
widths. The factor here covers that and the container's own variance, and it
is stated rather than padded.

**The utilisation divisor is Stage 1's, unchanged.** 0.4 -- the same divisor
Stage 1's launch gate was priced at. Its line items cover training, scoring and
benchmarking and still not the image pull, the dataset load, the cell master
load or the arm-store build, and those costs do not shrink because the workload
did.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mp_retrieval.compute_budget import (
    WorkUnit,
    container_rate_usd_per_hour,
    expected_spend_usd,
    feasibility,
)
from scripts.m2d_stage1_compute_record import (
    CPU_CORES,
    GPU,
    MEMORY_MB,
    TIMEOUT_SECONDS,
    UTILISATION,
    container_overhead_usd,
)

DECLARATION = REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "m2d_s4_semantic_repair"
STAGE_1_ROOT = OUTPUT_ROOT / "stage1"
RECORD_JSON = OUTPUT_ROOT / "stage2_compute_record.json"
RECORD_MARKDOWN = REPO_ROOT / "docs" / "M2D_STAGE2_COMPUTE_RECORD.md"

ARM = "A3_MINIMAL"

#: Section 6's "no large safety multiplier without justification", answered.
#: Stage 1's record used a container safety factor over a PREDICTED workload;
#: this one prices a measured one, so the only thing left to cover is
#: seed-to-seed variation in training time and container variance. 1.25 is
#: above the 2.0% arm-to-arm spread Stage 1 measured at fixed seed, by a wide
#: margin, and it is not the 2x a predicted workload would deserve.
CONTAINER_SAFETY = 1.25


def _ceiling(spend: float) -> float:
    """At least twice the expected spend, and at least the worst case.

    The same rule Stage 0 and Stage 1 used, on a finer ladder. Their steps jump
    0.5, 1.0, 5.0, and a workload this small would land on $5.00 -- six times
    the prediction, which is not a cap so much as a formality. A $2.00 step is
    added here rather than in the shared helper, because widening that ladder
    would silently re-ceiling two records that are already filed and whose
    numbers are historical.
    """

    doubled = spend * 2.0
    for step in (0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0):
        if doubled <= step:
            return step
    return float(int(doubled) + 1)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True
    ).stdout.strip()


def declaration() -> dict[str, Any]:
    return yaml.safe_load(DECLARATION.read_text(encoding="utf-8"))


def measured_stage_1(cells: list[str]) -> dict[str, dict[str, Any]]:
    """What Stage 1 actually spent on this arm, per cell.

    Read from the fetched artifacts rather than from Stage 1's record, which
    holds its own predictions. A prediction that has since been tested is not
    the number to price the next run at.
    """

    found: dict[str, dict[str, Any]] = {}
    for path in sorted(STAGE_1_ROOT.glob("*.json")):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = envelope.get("payload", envelope)
        if payload.get("arm") != ARM or payload.get("cell") not in cells:
            continue
        found[payload["cell"]] = payload

    missing = sorted(set(cells) - set(found))
    if missing:
        raise SystemExit(
            f"no measured Stage-1 {ARM} fit for {missing}. This record prices the "
            "next four fits from the ones that already ran; without them there is "
            "nothing measured to price against, and a prediction would have to be "
            "substituted for an observation."
        )
    return found


def cell_workload(cell: str, fit: dict[str, Any], seed: int) -> dict[str, Any]:
    """One container: one cell, one seed, one arm, priced line by line.

    Every second below was measured in Stage 1's container on this same cell,
    with the same code, the same panel and the same box.
    """

    latency = fit["uncached_inference"]
    # The harness's own loop, so the pass count is the harness's rather than a
    # guess: warmup, then `repeats` timed passes over the panel, then one
    # attribution pass.
    passes = int(latency["warmup_queries"]) + int(latency["queries"]) * (
        int(latency["repeats"]) + 1
    )
    benchmark_seconds = passes * float(latency["total_model_ms"]["p50"]) / 1000.0

    lines = [
        {
            "line": "native S4 re-score and its latency benchmark",
            "arm": "S4",
            "trains": False,
            "seconds": float(fit["native_s4_rescore"]["rescore_seconds"]),
            "basis": (
                f"measured in Stage 1's container on this cell: "
                f"{fit['held_out_queries']} held-out queries plus the native "
                "benchmark, from the same reused M2B checkpoint"
            ),
        },
        {
            "line": f"{ARM} fit",
            "arm": ARM,
            "trains": True,
            "seconds": float(fit["systems"]["train_time_seconds"]),
            "basis": (
                f"measured: this arm's own Stage-1 training time on this cell's "
                f"{fit['train_queries']} training queries"
            ),
        },
        {
            "line": f"{ARM} scoring",
            "arm": ARM,
            "trains": False,
            "seconds": float(fit["batched_inference"]["inference_seconds"]),
            "basis": (
                f"measured: the batched pass over this cell's "
                f"{fit['held_out_queries']} held-out queries"
            ),
        },
        {
            "line": f"{ARM} latency benchmark",
            "arm": ARM,
            "trains": False,
            "seconds": benchmark_seconds,
            "basis": (
                f"{passes} single-query passes at this arm's own measured p50 of "
                f"{latency['total_model_ms']['p50']:.4f} ms"
            ),
        },
    ]
    seconds = sum(line["seconds"] for line in lines)
    return {
        "cell": cell,
        "seed": seed,
        "arm": ARM,
        "train_queries": int(fit["train_queries"]),
        "held_out_queries": int(fit["held_out_queries"]),
        "measured_from": (
            f"outputs/m2d_s4_semantic_repair/stage1/"
            f"{cell.replace('/', '_')}_{ARM}.json"
        ),
        "lines": lines,
        "fits": 1,
        "benchmarks": 2,
        "latency_benchmark_passes": passes,
        "seconds": seconds,
        "seconds_at_container_safety": seconds * CONTAINER_SAFETY,
        "peak_train_vram_mb_measured": float(
            fit["systems"]["peak_train_vram_mb"]
        ),
    }


def build() -> dict[str, Any]:
    config = declaration()
    stage_2 = config["stage_2"]
    cells = list(stage_2["cells"])
    seeds = [int(seed) for seed in stage_2["seeds"]]

    if tuple(stage_2["arms"]) != (ARM,):
        raise SystemExit(
            f"the declaration names arms {stage_2['arms']}; this record prices {ARM} "
            "alone, which is what makes it four fits rather than eight"
        )

    fits = measured_stage_1(cells)
    workload = [cell_workload(cell, fits[cell], seed) for cell in cells for seed in seeds]

    units = [
        WorkUnit(name=f"{item['cell']} seed {item['seed']}", seconds=item["seconds"])
        for item in workload
    ]
    verdict = feasibility(units, TIMEOUT_SECONDS, safety=CONTAINER_SAFETY)
    rate = container_rate_usd_per_hour(gpu=GPU, cpu_cores=CPU_CORES, memory_mb=MEMORY_MB)
    compute_spend = expected_spend_usd(
        units, usd_per_container_hour=rate, training_fraction=UTILISATION
    )
    safe_spend = expected_spend_usd(
        [
            WorkUnit(name=unit.name, seconds=unit.seconds * CONTAINER_SAFETY)
            for unit in units
        ],
        usd_per_container_hour=rate,
        training_fraction=UTILISATION,
    )
    per_container, overhead_source = container_overhead_usd()
    overhead_spend = per_container * len(workload)
    spend = compute_spend + overhead_spend
    ceiling = _ceiling(max(spend, (safe_spend + overhead_spend) / 2.0))

    return {
        "status": "M2D_STAGE2_COMPUTE_RECORD",
        "phase": "M2D",
        "stage": "stage_2",
        "filed_before_any_job_was_submitted": True,
        "source_commit": _git("rev-parse", "HEAD"),
        "declaration": "configs/m2d_s4_semantic_repair.yaml",
        "authorised_by": stage_2["authorised_by"],
        "authorises": (
            f"{len(workload)} GPU jobs, one per cell and seed, each fitting {ARM} "
            f"once and re-scoring M2B's native S4 on the same panel -- "
            f"{len(workload)} new fits in total"
        ),
        "authorises_no_gpu": False,
        "trains_nothing": False,
        "new_fits": len(workload),
        "why_this_is_not_stage_1s_number": (
            "Stage 1's record predicted: it scaled M2B's measured S4 seconds by a "
            "forward multiplier measured on the host, because no M2D arm had ever "
            "been fit. That prediction has since been tested. Every second below is "
            "read from the Stage-1 artifacts for these two cells and this arm, so the "
            "basis is an observation rather than a model of one -- and the workload is "
            "half the size per container, because A1 is not rerun."
        ),
        "container": {
            "gpu": GPU,
            "cpu_cores": CPU_CORES,
            "memory_mb": MEMORY_MB,
            "timeout_seconds": TIMEOUT_SECONDS,
            "usd_per_hour": rate,
            "why_gpu": (
                "Stage 2 trains. Four fits of a 206,785-parameter scorer over panels "
                "of up to 20,850 training queries is a backward pass per step, and it "
                "is the shape the seconds below were measured on."
            ),
            "why_the_shape_is_unchanged": (
                "Not a choice. Pricing Stage 1's measured seconds against a different "
                "box would quote one machine's work at another machine's rate, and the "
                "same-seed comparison depends on the fits being produced the same way."
            ),
        },
        "measured_inputs": {
            "source": "the fetched Stage-1 A3_MINIMAL artifacts for these two cells",
            "nothing_is_predicted": (
                "No forward multiplier, no scaling of another rung's seconds, no "
                "host-side kernel timing. The four line items per container are the "
                "four Stage 1 measured."
            ),
            "arm_to_arm_training_spread_at_fixed_seed_pct": {
                cell: round(
                    abs(
                        fits[cell]["systems"]["train_time_seconds"]
                        - _sibling_train_seconds(cell)
                    )
                    / _sibling_train_seconds(cell)
                    * 100.0,
                    2,
                )
                for cell in cells
                if _sibling_train_seconds(cell)
            },
            "why_that_spread_is_quoted": (
                "It is the only direct evidence available for how much a fit's "
                "training time moves when the model changes but the cell does not, "
                "and it is what the container safety factor has to cover."
            ),
        },
        "workload": {
            "jobs": len(workload),
            "fits": len(workload),
            "arms": [ARM],
            "seeds": seeds,
            "cells": workload,
            "train_query_total": sum(item["train_queries"] for item in workload),
            "held_out_query_total": sum(item["held_out_queries"] for item in workload),
            "nothing_already_fit_is_refit": (
                "Seed 0 is Stage 1's artifact and is not refit. S3 and S4 are M2B's at "
                "all three seeds and are not refit. Native S4 is loaded from M2B's "
                "checkpoint and re-scored, which is a forward pass and is priced as one."
            ),
        },
        "prediction": {
            "measured_work_seconds": sum(unit.seconds for unit in units),
            "largest_single_job_seconds": max(item["seconds"] for item in workload),
            "largest_job": verdict.largest.name if verdict.largest else None,
            "feasible_within_timeout": bool(verdict),
            "feasibility_reason": verdict.reason,
            "container_safety_factor": CONTAINER_SAFETY,
            "why_the_safety_factor_is_small": (
                "Section 6 forbids a large multiplier without justification. Stage 1's "
                "2x covered a predicted workload; this one is measured, on the same "
                "code, panel and box, so the only residual is seed-to-seed training "
                "variation and container variance."
            ),
            "utilisation_assumed": UTILISATION,
            "utilisation_is_stage_1s": True,
            "compute_spend_usd": compute_spend,
            "container_overhead_usd": overhead_spend,
            "container_overhead_source": overhead_source,
            "expected_spend_usd": spend,
            "spend_usd_at_container_safety": safe_spend + overhead_spend,
            "hard_ceiling_usd": ceiling,
            "peak_train_vram_mb_measured": max(
                item["peak_train_vram_mb_measured"] for item in workload
            ),
            "storage": (
                "Four artifacts and four checkpoints on the existing result volume, "
                "under seed-bearing paths. No new volume, no new dataset copy, and "
                "nothing that could overwrite a Stage-1 result."
            ),
        },
        "abort_criteria": [
            (
                "any job exceeding its predicted walltime above by more than the "
                f"stated safety factor of {CONTAINER_SAFETY}"
            ),
            (
                "a container acquiring an accelerator other than the one priced here, "
                "which has no measured rate in this project"
            ),
            (
                "a cell whose panel size differs from the Stage-1 figure above, which "
                "would mean the same-seed comparison is not against the same panel"
            ),
            (
                "a native S4 re-score that does not reproduce M2B's filed recall@5 "
                "within the cell's stated bound, which the runner refuses on"
            ),
            "cumulative spend reaching the hard ceiling",
        ],
        "what_this_record_does_not_authorise": [
            "a fourth seed",
            "A1, or any arm other than A3_MINIMAL",
            "any cell other than squad_clean/R1 and musique_clean/R1",
            "the full 14-cell M2D screen",
            "any refit of seed 0, S3 or S4",
            "reading the test split",
        ],
    }


def _sibling_train_seconds(cell: str) -> float | None:
    """A1's measured training seconds on the same cell and seed, or None.

    The arm-to-arm spread at a fixed seed is the closest measured proxy this
    project has for how much a fit's wall time moves under a change that is not
    the data, and it is what the safety factor is sized against.
    """

    for path in sorted(STAGE_1_ROOT.glob("*.json")):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = envelope.get("payload", envelope)
        if payload.get("cell") == cell and payload.get("arm") == "A1":
            return float(payload["systems"]["train_time_seconds"])
    return None


def declaration_block(record: dict[str, Any]) -> dict[str, Any]:
    """The copy the declaration carries, generated rather than typed."""

    container = record["container"]
    prediction = record["prediction"]
    return {
        "document": str(RECORD_MARKDOWN.relative_to(REPO_ROOT)).replace("\\", "/"),
        "machine_readable": str(RECORD_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        "derived_by": "scripts/m2d_stage2_compute_record.py",
        "filed_before_launch": True,
        "jobs": record["workload"]["jobs"],
        "fits": record["workload"]["fits"],
        "gpu": container["gpu"],
        "cpu": container["cpu_cores"],
        "memory_mb": container["memory_mb"],
        "timeout_seconds": container["timeout_seconds"],
        "expected_spend_usd": round(prediction["expected_spend_usd"], 2),
        "cost_ceiling_usd": prediction["hard_ceiling_usd"],
        "why_gpu": container["why_gpu"],
        "numbers_are_derived_not_typed": (
            "Every second is read from the fetched Stage-1 A3_MINIMAL artifacts for "
            "these two cells -- training time, scoring time, the native re-score and "
            "the arm's own measured p50 -- and the pricing is this project's own "
            "compute_budget functions. Nothing is scaled from another rung. The "
            "document is regenerated, never edited."
        ),
    }


def render(record: dict[str, Any]) -> str:
    container = record["container"]
    prediction = record["prediction"]
    lines = [
        "# M2D Stage 2 — compute record",
        "",
        (
            f"Filed at `{record['source_commit'][:12]}`, before any Stage-2 job was "
            f"submitted. {record['authorises']}."
        ),
        "",
        f"Authorised by: {record['authorised_by']}.",
        "",
        "## Why this is not Stage 1's number",
        "",
        record["why_this_is_not_stage_1s_number"],
        "",
        "## Container",
        "",
        "| | |",
        "| --- | --- |",
        f"| GPU | {container['gpu']} |",
        f"| Cores | {container['cpu_cores']} |",
        f"| Memory | {container['memory_mb']} MB |",
        f"| Timeout | {container['timeout_seconds']} s |",
        f"| Rate | ${container['usd_per_hour']:.4f}/h |",
        "",
        container["why_gpu"],
        "",
        "## Workload",
        "",
        "| cell | seed | fit s | scoring s | native re-score s | benchmark s | total s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in record["workload"]["cells"]:
        by_line = {line["line"]: line["seconds"] for line in item["lines"]}
        lines.append(
            f"| {item['cell']} | {item['seed']} "
            f"| {by_line[f'{ARM} fit']:.2f} "
            f"| {by_line[f'{ARM} scoring']:.2f} "
            f"| {by_line['native S4 re-score and its latency benchmark']:.2f} "
            f"| {by_line[f'{ARM} latency benchmark']:.2f} "
            f"| {item['seconds']:.2f} |"
        )

    lines += [
        "",
        record["workload"]["nothing_already_fit_is_refit"],
        "",
        record["measured_inputs"]["nothing_is_predicted"],
        "",
        "## Prediction",
        "",
        "| | |",
        "| --- | --- |",
        f"| New fits | {record['new_fits']} |",
        f"| Jobs | {record['workload']['jobs']} |",
        f"| GPU seconds | {prediction['measured_work_seconds']:.1f} |",
        (
            f"| Largest job | {prediction['largest_job']} at "
            f"{prediction['largest_single_job_seconds']:.1f} s |"
        ),
        f"| Walltime within timeout | {str(prediction['feasible_within_timeout']).lower()} |",
        f"| Safety factor | {prediction['container_safety_factor']} |",
        f"| Utilisation assumed | {prediction['utilisation_assumed']} |",
        f"| Compute | ${prediction['compute_spend_usd']:.4f} |",
        f"| Container overhead | ${prediction['container_overhead_usd']:.4f} |",
        f"| **Expected spend** | **${prediction['expected_spend_usd']:.2f}** |",
        f"| At the safety factor | ${prediction['spend_usd_at_container_safety']:.2f} |",
        f"| **Hard ceiling** | **${prediction['hard_ceiling_usd']:.2f}** |",
        f"| Peak train VRAM measured | {prediction['peak_train_vram_mb_measured']:.0f} MB |",
        "",
        prediction["why_the_safety_factor_is_small"],
        "",
        f"Storage: {prediction['storage']}",
        "",
        f"Container overhead source: {prediction['container_overhead_source']}",
        "",
        "## Abort criteria",
        "",
    ]
    lines += [f"- {item}" for item in record["abort_criteria"]]
    lines += ["", "## What this record does not authorise", ""]
    lines += [f"- {item}" for item in record["what_this_record_does_not_authorise"]]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare against the filed record")
    args = parser.parse_args(argv)

    record = build()
    if args.check:
        if not RECORD_JSON.is_file():
            raise SystemExit(f"{RECORD_JSON} does not exist; generate the record first")
        filed = json.loads(RECORD_JSON.read_text(encoding="utf-8"))
        drifted = {
            key
            for key in record
            if key != "source_commit" and filed.get(key) != record[key]
        }
        if drifted:
            raise SystemExit(f"the filed record no longer matches its inputs: {sorted(drifted)}")
        print("the filed record matches its inputs")
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RECORD_JSON.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    RECORD_MARKDOWN.write_text(render(record), encoding="utf-8")
    print(f"wrote {RECORD_JSON}")
    print(f"wrote {RECORD_MARKDOWN}")
    print(json.dumps(declaration_block(record), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
