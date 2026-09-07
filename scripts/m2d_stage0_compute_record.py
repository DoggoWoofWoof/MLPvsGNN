#!/usr/bin/env python
"""M2D Stage-0 compute record, derived before the launch it authorises.

Read-only. Launches nothing, fits nothing, and authorises no GPU: the shape
below has no accelerator and the record refuses to price one. The convention is
M2C's -- predicted jobs, hardware, walltime, spend, a hard ceiling and abort
criteria filed BEFORE any job is submitted, so the launch is judged against a
number that existed first.

Every input is either read from an artifact or measured here with its
provenance stated:

``panel``
    each cell's fit portion, derived from the immutable M2B baseline table as
    ``split_queries - held_out_queries``. Not a fraction applied to a total --
    the table records both, and the runner's own ``holdout_split`` produced
    them, so subtracting is the arithmetic the runner already did.

``mean candidate pool``
    from the sealed M2C Stage-0 candidate contracts, which record the exact
    ``candidate_rows / queries`` for the cells M2C ran. Two of M2D's four cells
    are among them and take their measured value. The other two take the
    largest pool measured anywhere in that set, which brackets them rather than
    flattering them.

``scoring kernel``
    measured on THIS host, now, by building both sealed rungs at the frozen
    1536 width and the real structural width and timing the forward pass the
    probe actually performs. A host estimate stated as such, the convention
    ``configs/m0a_probe.yaml#compute.estimate`` established. It is measured at a
    pool larger than every cell's bracket, so it over-prices all four.

``pricing``
    ``mp_retrieval.compute_budget``, this project's own rates, for the CPU-only
    shape the container actually holds. The utilisation divisor charges for the
    image pull and the embedding load as well as for the arithmetic.

The one number that is not derived is the ceiling, which follows the track's
convention: the worst-case bracket rounded up to a stated round figure.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.run_m1a_feature_screen as _m1a
import scripts.run_m2b_semantic_minimality as _m2b
from mp_retrieval.compute_budget import (
    WorkUnit,
    container_rate_usd_per_hour,
    expected_spend_usd,
    feasibility,
)
from mp_retrieval.m1a_screen import HEAD_WIDTH as _HEAD_WIDTH
from scripts.run_m2d_stage0_probe import _family

DECLARATION = REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml"
BASELINE_JSON = (
    REPO_ROOT / "outputs" / "m2c_s4_structural_conditioning" / "m2b_baseline_table.json"
)
M2C_CONTRACTS = REPO_ROOT / "outputs" / "m2c_s4_structural_conditioning" / "stage0"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "m2d_s4_semantic_repair"
RECORD_JSON = OUTPUT_ROOT / "stage0_compute_record.json"
RECORD_MARKDOWN = REPO_ROOT / "docs" / "M2D_STAGE0_COMPUTE_RECORD.md"

# ---------------------------------------------------------------------------
# The container shape. CPU only, and the refusal is the point.
# ---------------------------------------------------------------------------
CPU_CORES = 4
MEMORY_MB = 16384
GPU = None
TIMEOUT_SECONDS = 3600

#: Fraction of billed time that does the measured work. The kernel timing below
#: covers the two forward passes and nothing else -- not the image pull, not the
#: embedding load, not the cell master -- so the spend is inflated rather than
#: quoted at face value. M2C's divisor, for the same reason and the same shape.
UTILISATION = 0.4

#: Larger than every cell's bracketed mean pool, so the measured per-query cost
#: over-prices all four rather than fitting the cheapest.
KERNEL_POOL = 400

#: The frozen payload width, and the structural block width every M2B fit used.
#: The structural width is DERIVED below from the recorded parameter counts
#: rather than typed, because a wrong value here would silently misprice the
#: dominant term.
PAYLOAD_WIDTH = 1536

#: The scorer's hidden width, imported from the model this track fits rather
#: than typed: it is the multiplier on every semantic column.
HEAD_WIDTH = _HEAD_WIDTH

REPEATS = 51
WARMUP = 3

#: The container is not this host. ``feasibility`` applies this to the measured
#: unit cost before comparing it with the timeout, and the spend ceiling below
#: is required to cover the same inflation. Stated rather than absorbed into the
#: measurement, so the estimate stays a measurement and the margin stays a
#: choice. It is ``compute_budget``'s own default, for its own reason.
CONTAINER_SAFETY = 1.5


def _git(*args: str) -> str:
    finished = subprocess.run(
        ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False
    )
    return finished.stdout.strip() if finished.returncode == 0 else ""


def declared_cells() -> list[str]:
    config = yaml.safe_load(DECLARATION.read_text(encoding="utf-8"))
    cells = config["stage_0"]["cells"]
    named = [*cells["failure_cells"], cells["passage_control"], cells["kb_control"]]
    if len(set(named)) != len(named):
        raise ValueError(f"the declaration names a cell twice: {named}")
    return named


def baseline_table() -> dict[str, Any]:
    return json.loads(BASELINE_JSON.read_text(encoding="utf-8"))


def baseline_rows(table: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in table["rows"]:
        if row["seed"] == 0 and row["rung"] == "S4":
            rows[(row["dataset"], row["regime"])] = row
    return rows


def structural_width(table: dict[str, Any]) -> int:
    """How many precomputed columns the scorer sees, derived from the fits.

    ``total = semantic + scorer`` and the scorer is
    ``Linear(width + semantic_columns, HEAD_WIDTH) + Linear(HEAD_WIDTH, 1)``, so
    the recorded parameter counts pin the width exactly. Deriving it means a
    change to the structural block cannot leave a stale constant here
    mispricing the dominant term.
    """

    widths: set[int] = set()
    columns = {"S2": 3, "S3": 5, "S4": 258}
    for row in table["rows"]:
        scorer = row["total_parameters"] - row["semantic_parameters"]
        # scorer = (input * H + H) + (H + 1)
        input_columns, remainder = divmod(scorer - HEAD_WIDTH - HEAD_WIDTH - 1, HEAD_WIDTH)
        if remainder:
            raise ValueError(
                f"{row['rung']}'s scorer holds {scorer} parameters, which is not a "
                f"Linear(n, {HEAD_WIDTH}) + Linear({HEAD_WIDTH}, 1) for any integer n"
            )
        widths.add(input_columns - columns[row["rung"]])
    if len(widths) != 1:
        raise ValueError(f"the fits disagree about the structural width: {sorted(widths)}")
    return widths.pop()


def measured_pools() -> tuple[dict[str, float], float]:
    """Mean candidate pool per cell, from M2C's sealed candidate contracts."""

    pools: dict[str, float] = {}
    for path in sorted(M2C_CONTRACTS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        contract = payload["candidate_contract"]
        pools[payload["cell"]] = contract["candidate_rows"] / contract["queries"]
    if not pools:
        raise FileNotFoundError(f"no sealed candidate contracts under {M2C_CONTRACTS}")
    return pools, max(pools.values())


def measure_kernel(*, width: int, pool: int) -> dict[str, float]:
    """Time the forward pass the probe performs, per query, on this host.

    The median is the priced estimate, and the fastest and slowest samples are
    recorded beside it so the spread is visible rather than hidden. On this
    workstation the slowest sample runs several times the median; that spread
    is interference from whatever else the machine is doing, not the kernel
    getting slower, and a verdict resting on the maximum of a few dozen samples
    would be a verdict about the host's other work.

    So the two are used for different things. The walltime verdict takes the
    median and inflates it by the stated CONTAINER_SAFETY, the margin for the
    container not being this host. The spend ceiling is required to cover the
    slowest sample as well: a cap the observed worst case would breach is not
    a cap.
    """

    torch.set_num_threads(CPU_CORES)
    nodes = torch.randn(pool, PAYLOAD_WIDTH)
    query = torch.randn(1, PAYLOAD_WIDTH)
    batch_index = torch.zeros(pool, dtype=torch.long)
    structural = torch.randn(pool, width)

    timings: dict[str, float] = {}
    for rung in ("S3", "S4"):
        model = _m1a.build_m1a_model(
            precomputed_width=width,
            semantic_rung=rung,
            dropout=0.2,
            temperature=0.07,
            embedding_dim=PAYLOAD_WIDTH,
            semantic_head=_m2b.build_semantic_head(rung, PAYLOAD_WIDTH),
        ).eval()
        with torch.no_grad():
            for _ in range(WARMUP):
                model.forward_explicit(nodes, query, batch_index, structural)
            samples = []
            for _ in range(REPEATS):
                tick = time.perf_counter()
                model.forward_explicit(nodes, query, batch_index, structural)
                samples.append((time.perf_counter() - tick) * 1000.0)
        timings[rung] = statistics.median(samples)
        timings[f"{rung}_fastest_observed"] = min(samples)
        timings[f"{rung}_slowest_observed"] = max(samples)
    timings["both"] = timings["S3"] + timings["S4"]
    timings["both_slowest_observed"] = (
        timings["S3_slowest_observed"] + timings["S4_slowest_observed"]
    )
    return timings


def build_record() -> dict[str, Any]:
    cells = declared_cells()
    table = baseline_table()
    rows = baseline_rows(table)
    width = structural_width(table)
    pools, bracket = measured_pools()
    kernel = measure_kernel(width=width, pool=KERNEL_POOL)

    if bracket > KERNEL_POOL:
        raise ValueError(
            f"the kernel was measured at a pool of {KERNEL_POOL} and the largest "
            f"bracketed cell holds {bracket:.1f}; the estimate would understate the cost"
        )

    workload = []
    for cell in cells:
        dataset, regime = cell.split("/")
        row = rows[(dataset, regime)]
        panel = int(row["split_queries"]) - int(row["held_out_queries"])
        measured_pool = pools.get(cell)
        workload.append(
            {
                "cell": cell,
                "panel_queries": panel,
                "split_queries": int(row["split_queries"]),
                "holdout_left_unexamined": int(row["held_out_queries"]),
                "mean_pool": measured_pool if measured_pool is not None else bracket,
                "mean_pool_source": (
                    f"measured, outputs/m2c_s4_structural_conditioning/stage0 ({cell})"
                    if measured_pool is not None
                    else (
                        "not measured for this cell; takes the largest pool measured "
                        f"anywhere in that set ({bracket:.1f}), which brackets it"
                    )
                ),
                "seconds": panel * kernel["both"] / 1000.0,
                "seconds_at_slowest_observed": (
                    panel * kernel["both_slowest_observed"] / 1000.0
                ),
            }
        )

    units = [WorkUnit(name=item["cell"], seconds=item["seconds"]) for item in workload]
    verdict = feasibility(units, TIMEOUT_SECONDS, safety=CONTAINER_SAFETY)
    rate = container_rate_usd_per_hour(gpu=GPU, cpu_cores=CPU_CORES, memory_mb=MEMORY_MB)
    spend = expected_spend_usd(units, usd_per_container_hour=rate, training_fraction=UTILISATION)
    slowest_spend = expected_spend_usd(
        [
            WorkUnit(name=item["cell"], seconds=item["seconds_at_slowest_observed"])
            for item in workload
        ],
        usd_per_container_hour=rate,
        training_fraction=UTILISATION,
    )
    ceiling = _ceiling(max(spend, slowest_spend / 2.0))

    by_family: dict[str, int] = defaultdict(int)
    for item in workload:
        dataset = item["cell"].split("/")[0]
        by_family[_family(dataset)] += item["panel_queries"]

    return {
        "status": "M2D_STAGE0_COMPUTE_RECORD",
        "phase": "M2D",
        "stage": "stage_0",
        "filed_before_any_job_was_submitted": True,
        "source_commit": _git("rev-parse", "HEAD"),
        "declaration": "configs/m2d_s4_semantic_repair.yaml",
        "authorises": (
            "Four CPU-only jobs, one per declared Stage-0 cell, each producing the "
            "complementarity, fixed-fusion and rescue diagnostics for that cell"
        ),
        "authorises_no_gpu": True,
        "trains_nothing": True,
        "container": {
            "gpu": GPU,
            "cpu_cores": CPU_CORES,
            "memory_mb": MEMORY_MB,
            "timeout_seconds": TIMEOUT_SECONDS,
            "usd_per_hour": rate,
            "why_cpu_only": (
                "The probe runs two forward passes per query over a few hundred "
                "candidates and does no backward pass. An accelerator would spend "
                "money on transfer, not on arithmetic."
            ),
        },
        "measured_inputs": {
            "structural_columns": width,
            "structural_columns_source": (
                "derived from the recorded parameter counts in the M2B baseline table, "
                f"which pin it exactly given a Linear(n, {HEAD_WIDTH}) + Linear({HEAD_WIDTH}, 1) scorer"
            ),
            "payload_width": PAYLOAD_WIDTH,
            "kernel_pool_candidates": KERNEL_POOL,
            "kernel_ms_per_query": kernel,
            "kernel_source": (
                "Measured on this host, now, on the forward pass the probe performs, "
                f"over {REPEATS} timed calls per rung. The priced figure is the "
                "median, inflated by the stated container safety factor for the "
                "walltime verdict; the fastest and slowest samples are recorded "
                "beside it. A host estimate, not a container measurement."
            ),
            "kernel_repeats": REPEATS,
            "largest_bracketed_pool": bracket,
            "measured_pools": pools,
        },
        "workload": {
            "jobs": len(workload),
            "cells": workload,
            "panel_total": sum(item["panel_queries"] for item in workload),
            "panel_by_family": dict(by_family),
            "no_panel_cap": (
                "Every cell runs its whole fit portion. musique_clean holds 3,190 "
                "queries and is the smallest; capping it to save seconds would spend "
                "the precision the +0.25pp gate needs, on the one cell that can "
                "least afford it."
            ),
        },
        "prediction": {
            "measured_work_seconds": sum(unit.seconds for unit in units),
            "largest_single_job_seconds": max(item["seconds"] for item in workload),
            "largest_job": verdict.largest.name if verdict.largest else None,
            "feasible_within_timeout": bool(verdict),
            "feasibility_reason": verdict.reason,
            "container_safety_factor": CONTAINER_SAFETY,
            "largest_single_job_seconds_at_slowest_observed": max(
                item["seconds_at_slowest_observed"] for item in workload
            ),
            "utilisation_assumed": UTILISATION,
            "expected_spend_usd": spend,
            "spend_usd_at_slowest_observed_kernel": slowest_spend,
            "hard_ceiling_usd": ceiling,
        },
        "abort_criteria": [
            (
                "any job exceeding its predicted walltime above by more than the "
                "stated container safety factor"
            ),
            "a container acquiring a GPU, which this record does not price",
            (
                "a cell whose panel size differs from the figure above, which would "
                "mean the sealed master is not the one M2B fitted"
            ),
            "cumulative spend reaching the hard ceiling",
        ],
        "what_this_record_does_not_authorise": [
            "any Stage-1 fit",
            "seeds 1 and 2",
            "the full 14-cell screen",
            "any GPU hour",
        ],
    }


def _ceiling(spend: float) -> float:
    """The worst-case bracket rounded up to a stated round figure."""

    doubled = spend * 2.0
    for step in (0.5, 1.0, 5.0, 10.0, 50.0, 100.0):
        if doubled <= step:
            return step
    return float(int(doubled) + 1)


def render(record: dict[str, Any]) -> str:
    container = record["container"]
    measured = record["measured_inputs"]
    prediction = record["prediction"]
    kernel = measured["kernel_ms_per_query"]

    lines = [
        "# M2D Stage-0 compute record",
        "",
        (
            f"Filed at `{record['source_commit'][:12]}`, before any job was submitted. "
            "CPU only; this record prices no accelerator and authorises none."
        ),
        "",
        "## What is authorised",
        "",
        record["authorises"] + ".",
        "",
        (
            "Nothing is trained, no weight is fitted and no test split is opened. "
            "Every number the jobs produce comes from four rankings over an "
            "already-frozen candidate pool."
        ),
        "",
        "## Container",
        "",
        "| cores | memory | GPU | timeout | rate |",
        "|---:|---:|---|---:|---:|",
        (
            f"| {container['cpu_cores']} | {container['memory_mb'] // 1024} GiB "
            f"| {container['gpu'] or 'none'} | {container['timeout_seconds']} s "
            f"| ${container['usd_per_hour']:.4f}/h |"
        ),
        "",
        container["why_cpu_only"],
        "",
        "## Measured inputs",
        "",
        (
            f"- Structural columns: **{measured['structural_columns']}**, "
            f"{measured['structural_columns_source']}."
        ),
        (
            f"- Scoring kernel at a pool of {measured['kernel_pool_candidates']}: "
            f"**{kernel['both']:.2f} ms/query** for both rungs together "
            f"(S3 {kernel['S3']:.2f}, S4 {kernel['S4']:.2f}; "
            f"{kernel['S3_fastest_observed'] + kernel['S4_fastest_observed']:.2f} "
            f"to {kernel['both_slowest_observed']:.2f} across samples). "
            f"{measured['kernel_source']}"
        ),
        (
            f"- Largest bracketed mean pool: {measured['largest_bracketed_pool']:.1f} "
            "candidates, so the kernel measurement over-prices every cell rather "
            "than fitting the cheapest."
        ),
        "",
        "## Workload",
        "",
        "| cell | panel | mean pool | predicted seconds |",
        "|---|---:|---:|---:|",
    ]
    for item in record["workload"]["cells"]:
        lines.append(
            f"| {item['cell']} | {item['panel_queries']:,} | "
            f"{item['mean_pool']:.1f} | {item['seconds']:.0f} |"
        )
    lines += [
        "",
        (
            f"{record['workload']['panel_total']:,} queries across "
            f"{record['workload']['jobs']} jobs. {record['workload']['no_panel_cap']}"
        ),
        "",
        "## Prediction",
        "",
        f"- Measured work: **{prediction['measured_work_seconds']:.0f} s** total.",
        (
            f"- Largest single job: **{prediction['largest_single_job_seconds']:.0f} s** "
            f"({prediction['largest_job']}), against a "
            f"{container['timeout_seconds']} s timeout."
        ),
        (
            f"- Feasible within the timeout at "
            f"{prediction['container_safety_factor']:.1f}x safety: "
            f"**{prediction['feasible_within_timeout']}** "
            f"— {prediction['feasibility_reason']}"
        ),
        (
            "- At the slowest kernel sample this host produced, the largest job "
            "would be "
            f"**{prediction['largest_single_job_seconds_at_slowest_observed']:.0f} s** "
            "and the whole workload "
            f"${prediction['spend_usd_at_slowest_observed_kernel']:.2f}. That "
            "figure prices the machine's other work, so it caps the bill rather "
            "than settling the verdict."
        ),
        (
            f"- Expected spend at {prediction['utilisation_assumed']:.0%} utilisation: "
            f"**${prediction['expected_spend_usd']:.2f}**."
        ),
        (
            f"- Hard ceiling: **${prediction['hard_ceiling_usd']:.2f}**, which "
            "covers the slowest-sample figure as well."
        ),
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
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    record = build_record()
    markdown = render(record)
    if args.print_only:
        print(markdown)
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RECORD_JSON.write_text(json.dumps(record, indent=2), encoding="utf-8")
    RECORD_MARKDOWN.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {RECORD_JSON.relative_to(REPO_ROOT)}")
    print(f"wrote {RECORD_MARKDOWN.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
