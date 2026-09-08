#!/usr/bin/env python
"""M2D Stage-1 compute record, derived before the launch it authorises.

Read-only. Launches nothing and fits nothing. Same convention as Stage 0's and
M2C's -- predicted jobs, hardware, walltime, spend, a hard ceiling and abort
criteria filed BEFORE any job is submitted -- with one substantive difference:
Stage 1 trains, so this record authorises GPU hours and Stage 0's refusal to
hold an accelerator is inverted rather than inherited.

Section 6 says to recompute mechanically from the actual selected cells and
current launcher pricing, and not to carry M2B's $0.92 forward blindly. The
basis genuinely differs, in both directions:

* M2B priced 28 fits across 14 cells by scaling M2's measured S3 training
  seconds by a *declared multiplier* for a rung nobody had timed yet. Stage 1
  prices 8 fits across 4 cells against M2B's own measured **S4** training
  seconds for those exact cells -- the thing the arms are one column away from
  being. The estimate is one step closer to the object.
* Stage 1 buys work M2B did not. Each cell re-scores native S4 from M2B's
  checkpoint, and times it in the same container, so three latency benchmarks
  are paid for per cell rather than two. That is section 10's requirement that
  the added column's cost be a same-clock measurement, and it is a line item
  here rather than a rounding error absorbed into a multiplier.

Every input is either read from a sealed artifact or measured here with its
provenance stated:

``per-cell training seconds``
    ``outputs/m2b_semantic_minimality/headline/<dataset>.json``, the cell's own
    filed ``systems.train_time_seconds`` for S4 at seed 0. Not an average
    across the matrix and not a fraction of a total: it is what this cell's S4
    fit actually took, on the shape below.

``per-cell inference milliseconds``
    the same artifact's filed ``uncached_inference_p50_ms`` for S4, which is
    the per-query forward cost the scoring passes and the latency benchmarks
    are multiples of.

``panel sizes``
    the immutable M2B baseline table's ``train_queries`` and
    ``held_out_queries``. The runner's own ``holdout_split`` produced them.

``arm multipliers``
    measured on THIS host, now, by building native S4 and both arms at the
    frozen 1536 width and the live structural width and timing the forward
    pass, INTERLEAVED so host load drifts hit numerator and denominator alike.
    A ratio rather than an absolute, because the absolutes it is applied to
    were measured in the container and a wall-clock figure from this
    workstation would be the wrong machine's. A ratio does not cancel the
    machine either -- see ``measure_arm_multipliers`` -- and the direction of
    that error is stated there rather than assumed away. A host estimate,
    stated as such, the convention
    ``configs/m0a_probe.yaml#compute.estimate`` established.

``pricing``
    ``mp_retrieval.compute_budget``, this project's own rates, for the shape
    the container actually holds -- which is M2B's shape, because Stage 1 fits
    the same models on the same cells and a cheaper box would make the
    training seconds above inapplicable.

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
from mp_retrieval.m2b_inference_timing import DEFAULT_REPEATS, DEFAULT_WARMUP_QUERIES
from mp_retrieval.m2d_stage1_arms import build_arm_head

DECLARATION = REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml"
BASELINE_JSON = (
    REPO_ROOT / "outputs" / "m2c_s4_structural_conditioning" / "m2b_baseline_table.json"
)
M2B_HEADLINE = REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "headline"
M2_MEASURED_COST = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "measured_cost.json"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "m2d_s4_semantic_repair"
RECORD_JSON = OUTPUT_ROOT / "stage1_compute_record.json"
RECORD_MARKDOWN = REPO_ROOT / "docs" / "M2D_STAGE1_COMPUTE_RECORD.md"

# ---------------------------------------------------------------------------
# The container shape. M2B's, because Stage 1 fits M2B's models on M2B's cells.
# ---------------------------------------------------------------------------
#: Read from M1A's infra block, which is where M2B took it from, rather than
#: retyped. The per-cell training seconds this record scales were measured on
#: this shape; quoting them against a different box would be pricing one
#: machine's work at another machine's rate.
_INFRA = yaml.safe_load(
    (REPO_ROOT / "configs" / "m1a_feature_screen.yaml").read_text(encoding="utf-8")
)["modal"]
GPU = _INFRA["gpu"]
CPU_CORES = int(_INFRA["cpu"])
MEMORY_MB = int(_INFRA["memory_mb"])

#: Not M1A's 24 hours. The largest predicted job below is minutes, and a
#: timeout is an abort criterion as well as a limit: a Stage-1 cell still
#: running after an hour is not a slow cell, it is a cell doing something this
#: record did not predict, and the cheapest response is to stop it. Asserted
#: against the prediction rather than assumed to be roomy.
TIMEOUT_SECONDS = 3600

#: Fraction of billed time that does the measured work. The line items below
#: cover training, scoring and benchmarking; they do not cover the image pull,
#: the dataset load, the cell master load or the arm-store build. M2C's and
#: Stage 0's divisor, for the same reason.
UTILISATION = 0.4

#: The frozen payload width every track fit runs at.
PAYLOAD_WIDTH = 1536

#: The scorer's hidden width, imported from the model this track fits rather
#: than typed: it is the multiplier on every semantic column.
HEAD_WIDTH = _HEAD_WIDTH

#: How many semantic columns each rung contributes, used to derive the
#: structural width from the filed parameter counts.
SEMANTIC_COLUMNS = {"S2": 3, "S3": 5, "S4": 258}

#: The pool the arm-multiplier kernel is timed at. Larger than every cell's
#: measured mean pool, checked below rather than asserted here, so the ratio is
#: taken where the per-query work is at least as large as any cell's.
KERNEL_POOL = 400
REPEATS = 31
WARMUP = 3

#: The runner's latency settings, imported from the harness that spends them.
#: One benchmark costs ``warmup + queries * repeats`` timed passes plus one
#: attribution pass over the queries; that arithmetic is the harness's, so it
#: is derived here rather than transcribed as a total.
LATENCY_QUERIES = 200
LATENCY_REPEATS = DEFAULT_REPEATS
LATENCY_WARMUP = DEFAULT_WARMUP_QUERIES

#: The arms, and native S4 which is re-scored rather than fitted.
ARMS = ("A1", "A3_MINIMAL")
NATIVE = "S4"

#: The container is not this host. ``feasibility`` applies this to the measured
#: unit cost before comparing it with the timeout, and the spend ceiling is
#: required to cover the same inflation. ``compute_budget``'s own default.
CONTAINER_SAFETY = 1.5


def _git(*args: str) -> str:
    finished = subprocess.run(
        ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False
    )
    return finished.stdout.strip() if finished.returncode == 0 else ""


def declared_cells() -> list[str]:
    """The four cells section 5 names, read from the declaration and only there."""

    config = yaml.safe_load(DECLARATION.read_text(encoding="utf-8"))
    cells = config["stage_1"]["cells"]
    named = [*cells["mandatory_blockers"], *cells["controls"]]
    if len(set(named)) != len(named):
        raise ValueError(f"the declaration names a Stage-1 cell twice: {named}")
    declared_fits = int(config["stage_1"]["new_fits"])
    if declared_fits != len(named) * len(ARMS):
        raise ValueError(
            f"the declaration authorises {declared_fits} new fits and names {len(named)} "
            f"cells for {len(ARMS)} arms, which is {len(named) * len(ARMS)}. A record "
            "that priced a different number of fits than the declaration authorises "
            "would authorise neither."
        )
    return named


def baseline_table() -> dict[str, Any]:
    return json.loads(BASELINE_JSON.read_text(encoding="utf-8"))


def structural_width(table: dict[str, Any]) -> int:
    """How many precomputed columns the scorer sees, derived from the fits.

    ``total = semantic + scorer`` and the scorer is
    ``Linear(width + semantic_columns, HEAD_WIDTH) + Linear(HEAD_WIDTH, 1)``, so
    the recorded parameter counts pin the width exactly. Stage 0's derivation,
    unchanged: a constant here could silently misprice the dominant term.
    """

    widths: set[int] = set()
    for row in table["rows"]:
        scorer = row["total_parameters"] - row["semantic_parameters"]
        columns, remainder = divmod(scorer - HEAD_WIDTH - HEAD_WIDTH - 1, HEAD_WIDTH)
        if remainder:
            raise ValueError(
                f"{row['rung']}'s scorer holds {scorer} parameters, which is not a "
                f"Linear(n, {HEAD_WIDTH}) + Linear({HEAD_WIDTH}, 1) for any integer n"
            )
        widths.add(columns - SEMANTIC_COLUMNS[row["rung"]])
    if len(widths) != 1:
        raise ValueError(f"the fits disagree about the structural width: {sorted(widths)}")
    return widths.pop()


def filed_cell(cell: str) -> dict[str, Any]:
    """One cell's filed S4 facts, from M2B's headline artifact.

    The headline rather than the baseline table for the systems numbers,
    because the table carries the metrics and the parameter counts but not the
    training seconds this record is built on.
    """

    dataset, regime = cell.split("/")
    path = M2B_HEADLINE / f"{dataset}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    block = payload["cells"][regime]
    rung = block["rungs"][NATIVE]
    if int(rung["seed"]) != 0:
        raise ValueError(f"{cell}: the filed {NATIVE} row is seed {rung['seed']}, not 0")
    contract = payload["candidate_contract"]
    return {
        "cell": cell,
        "train_queries": int(block["train_queries"]),
        "held_out_queries": int(block["held_out_queries"]),
        "precomputed_width": int(block["precomputed_width"]),
        "s4_train_seconds": float(rung["systems"]["train_time_seconds"]),
        "s4_uncached_p50_ms": float(rung["systems"]["uncached_inference_p50_ms"]),
        "s4_uncached_p95_ms": float(rung["systems"]["uncached_inference_p95_ms"]),
        "s4_peak_train_vram_mb": float(rung["systems"]["peak_train_vram_mb"]),
        "mean_pool": contract["candidate_rows"] / contract["queries"],
        "source": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
    }


def measure_arm_multipliers(*, width: int, pool: int) -> dict[str, Any]:
    """Each arm's forward cost relative to native S4's, measured on this host.

    A ratio, not an absolute. The absolutes this record scales -- M2B's per-cell
    training seconds and per-query inference milliseconds -- were measured in
    the container on an A10G; a wall-clock figure from this workstation would
    be the wrong machine's.

    A ratio does not fully cancel the machine, and saying so is the point. The
    numerator and denominator are different KINDS of work: native S4's cost is
    two 1536x64 projections, a compute-bound GEMM, while what each arm adds is
    an elementwise pass over an n x 1536 temporary, which is memory-bound. A
    GPU closes the gap between those two far more than a CPU does, so a ratio
    measured here should be expected to OVERSTATE the ratio in the container.
    That is the safe direction for a compute record and it is why the figure is
    used rather than discarded: it brackets the arms rather than flattering
    them. It is not evidence about what the arms will cost on an A10G, and
    section 10's latency verdict comes from the container's own measurement,
    never from here.

    The median is the priced figure. The spread across samples is recorded
    beside it so a ratio that is really host interference is visible rather
    than hidden.
    """

    torch.set_num_threads(CPU_CORES)
    nodes = torch.randn(pool, PAYLOAD_WIDTH)
    query = torch.randn(1, PAYLOAD_WIDTH)
    batch_index = torch.zeros(pool, dtype=torch.long)
    structural = torch.randn(pool, width)

    models = {}
    for arm in (NATIVE, *ARMS):
        head = (
            _m2b.build_semantic_head(NATIVE, PAYLOAD_WIDTH)
            if arm == NATIVE
            else build_arm_head(arm, PAYLOAD_WIDTH)
        )
        models[arm] = _m1a.build_m1a_model(
            precomputed_width=width,
            semantic_rung=arm,
            dropout=0.2,
            temperature=0.07,
            embedding_dim=PAYLOAD_WIDTH,
            semantic_head=head,
        ).eval()

    samples: dict[str, list[float]] = {arm: [] for arm in models}
    with torch.no_grad():
        for _ in range(WARMUP):
            for model in models.values():
                model.forward_explicit(nodes, query, batch_index, structural)
        # Interleaved, one pass per model per round. Timed in three separate
        # blocks the ratio measured whatever the machine was doing between
        # them: across three trials it read 1.10x to 2.66x for the same arm and
        # even reversed which arm was dearer. Round-robin puts every model
        # under the same load in the same second, which is the only way a
        # ratio taken on a shared workstation means anything.
        for _ in range(REPEATS):
            for arm, model in models.items():
                tick = time.perf_counter()
                model.forward_explicit(nodes, query, batch_index, structural)
                samples[arm].append((time.perf_counter() - tick) * 1000.0)

    native_median = statistics.median(samples[NATIVE])
    multipliers = {}
    for arm in ARMS:
        median = statistics.median(samples[arm])
        # The arm's worst sample against the PRICED native figure, not against
        # native's best. Dividing one tail by the other compounds two
        # independent interference events into a ratio no execution could
        # produce -- it read 15x here -- and a bracket that wide stops
        # bracketing anything.
        slowest = max(samples[arm]) / native_median
        multipliers[arm] = {
            "median": median / native_median,
            # A ratio below 1 means the arm measured faster than the model it
            # is a superset of, which is host noise rather than a saving. It is
            # not allowed to lower the bill.
            "priced": max(1.0, median / native_median),
            "at_slowest_observed": slowest,
            "median_ms": median,
        }
    return {
        "native_median_ms": native_median,
        "arms": multipliers,
        "repeats": REPEATS,
        "pool": pool,
        "measured_on": "this host, CPU, interleaved round-robin across the three models",
        "native_ms_quartiles": [
            statistics.quantiles(samples[NATIVE], n=4)[0],
            native_median,
            statistics.quantiles(samples[NATIVE], n=4)[2],
        ],
        "why_a_ratio": (
            "The absolutes this scales were measured on an A10G in M2B's containers, so "
            "a wall-clock figure from this workstation would be the wrong machine's. A "
            "ratio does not fully cancel the machine either, and the direction of that "
            "error is stated rather than assumed away: native S4's cost is a "
            "compute-bound GEMM (two 1536x64 projections) while what each arm adds is a "
            "memory-bound elementwise pass over an n x 1536 temporary, and a GPU closes "
            "that gap far more than a CPU does. A ratio measured here should therefore "
            "be expected to OVERSTATE the ratio in the container, which is the safe "
            "direction for a prediction: it brackets the arms rather than flattering "
            "them. It is not evidence about what the arms cost on an A10G -- section "
            "10's latency verdict comes from the container's own measurement."
        ),
    }


def benchmark_passes(held_out: int) -> int:
    """Single-query forward passes one latency benchmark costs.

    Derived from the harness's own loop -- warmup, then ``repeats`` timed passes
    over the panel, then one hooked attribution pass -- rather than quoted as a
    total, so a change to the harness changes this number instead of silently
    invalidating it.
    """

    queries = min(LATENCY_QUERIES, held_out)
    return min(LATENCY_WARMUP, queries) + queries * LATENCY_REPEATS + queries


def cell_workload(filed: dict[str, Any], multipliers: dict[str, Any]) -> dict[str, Any]:
    """Every second one Stage-1 cell is predicted to spend, as line items.

    Enumerated rather than totalled because the difference from M2B's basis is
    in the lines, not in the sum: the native re-score and its benchmark are
    work M2B never bought, and burying them in a multiplier would make section
    6's "recompute mechanically" unauditable.
    """

    per_query_s = filed["s4_uncached_p50_ms"] / 1000.0
    held_out = filed["held_out_queries"]
    passes = benchmark_passes(held_out)

    lines: list[dict[str, Any]] = [
        {
            "line": "native S4 re-score",
            "arm": NATIVE,
            "trains": False,
            "seconds": held_out * per_query_s,
            "basis": (
                f"{held_out} held-out queries at the cell's filed S4 p50 of "
                f"{filed['s4_uncached_p50_ms']:.4f} ms"
            ),
        },
        {
            "line": "native S4 latency benchmark",
            "arm": NATIVE,
            "trains": False,
            "seconds": passes * per_query_s,
            "basis": (
                f"{passes} single-query passes, the harness's warmup + "
                f"{LATENCY_REPEATS} timed passes + one attribution pass over "
                f"{min(LATENCY_QUERIES, held_out)} queries"
            ),
        },
    ]
    for arm in ARMS:
        factor = multipliers["arms"][arm]["priced"]
        lines.extend(
            [
                {
                    "line": f"{arm} fit",
                    "arm": arm,
                    "trains": True,
                    "seconds": filed["s4_train_seconds"] * factor,
                    "basis": (
                        f"this cell's filed S4 training seconds "
                        f"({filed['s4_train_seconds']:.1f}) at the measured forward "
                        f"multiplier {factor:.3f}"
                    ),
                },
                {
                    "line": f"{arm} scoring",
                    "arm": arm,
                    "trains": False,
                    "seconds": held_out * per_query_s * factor,
                    "basis": f"{held_out} held-out queries at {factor:.3f}x S4's p50",
                },
                {
                    "line": f"{arm} latency benchmark",
                    "arm": arm,
                    "trains": False,
                    "seconds": passes * per_query_s * factor,
                    "basis": f"{passes} single-query passes at {factor:.3f}x S4's p50",
                },
            ]
        )

    seconds = sum(line["seconds"] for line in lines)
    slowest = sum(
        line["seconds"]
        * (
            multipliers["arms"][line["arm"]]["at_slowest_observed"]
            / multipliers["arms"][line["arm"]]["priced"]
            if line["arm"] in ARMS
            else 1.0
        )
        for line in lines
    )
    return {
        **filed,
        "lines": lines,
        "fits": len(ARMS),
        "benchmarks": len(ARMS) + 1,
        "latency_benchmark_passes": passes,
        "seconds": seconds,
        "seconds_at_slowest_observed_multiplier": slowest,
    }


def container_overhead_usd() -> tuple[float, str]:
    """Per-container overhead, from M2's measured smoke rather than assumed."""

    measured = json.loads(M2_MEASURED_COST.read_text(encoding="utf-8"))
    overhead = measured["container_overhead"]
    return float(overhead["usd_per_container"]), (
        f"outputs/m2_qls_v2_freeze/measured_cost.json, measured over "
        f"{overhead['measured_over_containers']} containers"
    )


def build_record() -> dict[str, Any]:
    cells = declared_cells()
    table = baseline_table()
    width = structural_width(table)

    filed = [filed_cell(cell) for cell in cells]
    for item in filed:
        if item["precomputed_width"] != width:
            raise ValueError(
                f"{item['cell']} filed precomputed_width={item['precomputed_width']} and "
                f"the table derives {width}; the arms would build at a width M2B did not "
                "fit and the S4 checkpoint would not load"
            )
    largest_pool = max(item["mean_pool"] for item in filed)
    if largest_pool > KERNEL_POOL:
        raise ValueError(
            f"the multiplier kernel was timed at a pool of {KERNEL_POOL} and the largest "
            f"cell's mean pool is {largest_pool:.1f}; the ratio would be taken where the "
            "per-query work is smaller than the cells'"
        )

    multipliers = measure_arm_multipliers(width=width, pool=KERNEL_POOL)
    workload = [cell_workload(item, multipliers) for item in filed]

    units = [WorkUnit(name=item["cell"], seconds=item["seconds"]) for item in workload]
    verdict = feasibility(units, TIMEOUT_SECONDS, safety=CONTAINER_SAFETY)
    rate = container_rate_usd_per_hour(gpu=GPU, cpu_cores=CPU_CORES, memory_mb=MEMORY_MB)
    compute_spend = expected_spend_usd(
        units, usd_per_container_hour=rate, training_fraction=UTILISATION
    )
    slowest_spend = expected_spend_usd(
        [
            WorkUnit(name=item["cell"], seconds=item["seconds_at_slowest_observed_multiplier"])
            for item in workload
        ],
        usd_per_container_hour=rate,
        training_fraction=UTILISATION,
    )
    per_container, overhead_source = container_overhead_usd()
    overhead_spend = per_container * len(workload)
    spend = compute_spend + overhead_spend
    # Stage 0's convention: _ceiling doubles before rounding up, so the
    # worst case is halved going in. The result is at least twice the expected
    # spend and at least the observed worst case -- a cap the worst case would
    # breach is not a cap, and a cap ten times the prediction is not one either.
    ceiling = _ceiling(max(spend, (slowest_spend + overhead_spend) / 2.0))

    return {
        "status": "M2D_STAGE1_COMPUTE_RECORD",
        "phase": "M2D",
        "stage": "stage_1",
        "filed_before_any_job_was_submitted": True,
        "source_commit": _git("rev-parse", "HEAD"),
        "declaration": "configs/m2d_s4_semantic_repair.yaml",
        "authorises": (
            f"{len(workload)} GPU jobs, one per declared Stage-1 cell, each fitting "
            f"{len(ARMS)} arms at seed 0 and re-scoring M2B's native S4 on the same "
            f"panel -- {len(workload) * len(ARMS)} new fits in total"
        ),
        "authorises_no_gpu": False,
        "trains_nothing": False,
        "new_fits": len(workload) * len(ARMS),
        "why_this_is_not_m2bs_number": (
            "Section 6 forbids carrying $0.92 forward. The basis differs in both "
            "directions: M2B scaled M2's measured S3 seconds by a multiplier declared "
            "for an untimed rung, while this scales M2B's own measured S4 seconds for "
            "these exact cells; and Stage 1 buys a native S4 re-score and a third "
            "latency benchmark per cell that M2B never paid for, because section 10 "
            "requires the added column's cost to be a same-container measurement."
        ),
        "container": {
            "gpu": GPU,
            "cpu_cores": CPU_CORES,
            "memory_mb": MEMORY_MB,
            "timeout_seconds": TIMEOUT_SECONDS,
            "usd_per_hour": rate,
            "why_gpu": (
                "Stage 1 trains. Eight fits of a 205k-parameter scorer over panels of "
                "up to 31,310 queries is a backward pass per step, which is what an "
                "accelerator is for, and it is the shape M2B's training seconds -- the "
                "basis of every line below -- were measured on."
            ),
            "why_the_shape_is_m2bs": (
                "Not a choice. The per-cell seconds this record scales were measured on "
                "this shape; pricing them against a smaller box would quote one "
                "machine's work at another machine's rate."
            ),
            "why_the_timeout_is_not_m1as": (
                "M1A allows 24 hours. The largest job predicted below is minutes, and a "
                "cell still running after an hour is doing something this record did "
                "not predict. The timeout is an abort criterion, not headroom."
            ),
        },
        "measured_inputs": {
            "structural_columns": width,
            "structural_columns_source": (
                "derived from the recorded parameter counts in the M2B baseline table, "
                f"which pin it exactly given a Linear(n, {HEAD_WIDTH}) + "
                f"Linear({HEAD_WIDTH}, 1) scorer"
            ),
            "payload_width": PAYLOAD_WIDTH,
            "arm_forward_multipliers": multipliers,
            "largest_cell_mean_pool": largest_pool,
            "kernel_pool_candidates": KERNEL_POOL,
            "latency_settings": {
                "queries": LATENCY_QUERIES,
                "repeats": LATENCY_REPEATS,
                "warmup": LATENCY_WARMUP,
                "passes_per_benchmark_source": (
                    "derived from mp_retrieval.m2b_inference_timing's own loop: warmup, "
                    "repeats timed passes over the panel, then one attribution pass"
                ),
            },
        },
        "workload": {
            "jobs": len(workload),
            "fits": len(workload) * len(ARMS),
            "arms": list(ARMS),
            "seeds": [0],
            "cells": workload,
            "train_query_total": sum(item["train_queries"] for item in workload),
            "held_out_query_total": sum(item["held_out_queries"] for item in workload),
            "nothing_already_fit_is_refit": (
                "Native S4 is loaded from M2B's checkpoint and re-scored, never "
                "retrained, and S3 is not opened at all. The re-score is priced as a "
                "forward pass because that is what it is."
            ),
        },
        "prediction": {
            "measured_work_seconds": sum(unit.seconds for unit in units),
            "largest_single_job_seconds": max(item["seconds"] for item in workload),
            "largest_job": verdict.largest.name if verdict.largest else None,
            "feasible_within_timeout": bool(verdict),
            "feasibility_reason": verdict.reason,
            "container_safety_factor": CONTAINER_SAFETY,
            "utilisation_assumed": UTILISATION,
            "compute_spend_usd": compute_spend,
            "container_overhead_usd": overhead_spend,
            "container_overhead_source": overhead_source,
            "expected_spend_usd": spend,
            "spend_usd_at_slowest_observed_multiplier": slowest_spend + overhead_spend,
            "hard_ceiling_usd": ceiling,
            "peak_train_vram_mb_filed": max(
                item["s4_peak_train_vram_mb"] for item in workload
            ),
        },
        "abort_criteria": [
            (
                "any job exceeding its predicted walltime above by more than the stated "
                "container safety factor"
            ),
            (
                "a container acquiring an accelerator other than the one priced here, "
                "which has no measured rate in this project"
            ),
            (
                "a cell whose panel size differs from the figure above, which would mean "
                "the sealed master is not the one M2B fitted"
            ),
            (
                "a native S4 re-score that does not reproduce M2B's filed recall@5 "
                "within the cell's stated bound, which the runner refuses on"
            ),
            "cumulative spend reaching the hard ceiling",
        ],
        "what_this_record_does_not_authorise": [
            "seeds 1 and 2",
            "the full 14-cell M2D screen",
            "any arm other than A1 and A3_MINIMAL",
            "any refit of S3 or S4",
            "reading the test split",
        ],
    }


def _ceiling(spend: float) -> float:
    """The worst-case bracket rounded up to a stated round figure."""

    doubled = spend * 2.0
    for step in (0.5, 1.0, 5.0, 10.0, 50.0, 100.0):
        if doubled <= step:
            return step
    return float(int(doubled) + 1)


def declaration_block(record: dict[str, Any]) -> dict[str, Any]:
    """The copy of this record the declaration carries, generated not typed.

    The launcher reads its container shape from the declaration rather than
    from this file, because the launcher's decorators are evaluated inside the
    container and the image carries no ``outputs/`` tree. Two copies of one
    shape can disagree, so one generates the other and a declaration test holds
    them equal.
    """

    container = record["container"]
    prediction = record["prediction"]
    return {
        "document": str(RECORD_MARKDOWN.relative_to(REPO_ROOT)).replace("\\", "/"),
        "machine_readable": str(RECORD_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        "derived_by": "scripts/m2d_stage1_compute_record.py",
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
            "Panel sizes and training seconds come from M2B's filed headline artifacts "
            "for these exact cells, the structural width is derived from the recorded "
            "parameter counts, the arm multipliers are measured on the host and stated "
            "as ratios, and the pricing is this project's own compute_budget functions. "
            "The document is regenerated, never edited."
        ),
    }


def render(record: dict[str, Any]) -> str:
    container = record["container"]
    measured = record["measured_inputs"]
    prediction = record["prediction"]
    multipliers = measured["arm_forward_multipliers"]

    lines = [
        "# M2D Stage 1 compute record",
        "",
        f"Status: `{record['status']}`  ",
        f"Source commit: `{record['source_commit']}`  ",
        f"Declaration: `{record['declaration']}`",
        "",
        "Generated by `scripts/m2d_stage1_compute_record.py`. Do not edit: regenerate.",
        "",
        "## What this authorises",
        "",
        record["authorises"] + ".",
        "",
        record["why_this_is_not_m2bs_number"],
        "",
        "## Container",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| GPU | {container['gpu']} |",
        f"| CPU cores | {container['cpu_cores']} |",
        f"| Memory | {container['memory_mb']} MB |",
        f"| Timeout | {container['timeout_seconds']} s |",
        f"| Rate | ${container['usd_per_hour']:.4f}/h |",
        "",
        container["why_gpu"],
        "",
        container["why_the_shape_is_m2bs"],
        "",
        container["why_the_timeout_is_not_m1as"],
        "",
        "## Measured inputs",
        "",
        (
            f"* Structural columns: **{measured['structural_columns']}** "
            f"({measured['structural_columns_source']})."
        ),
        f"* Payload width: **{measured['payload_width']}**.",
        (
            f"* Native S4 forward on this host: "
            f"**{multipliers['native_median_ms']:.3f} ms** per query at a pool of "
            f"{multipliers['pool']}, median of {multipliers['repeats']}."
        ),
        "",
        "| arm | forward multiplier vs S4 | priced | at slowest observed |",
        "| --- | --- | --- | --- |",
    ]
    for arm in ARMS:
        item = multipliers["arms"][arm]
        lines.append(
            f"| {arm} | {item['median']:.3f} | {item['priced']:.3f} | "
            f"{item['at_slowest_observed']:.3f} |"
        )
    lines += [
        "",
        multipliers["why_a_ratio"],
        "",
        "## Workload",
        "",
        "| cell | train | held out | S4 train s | S4 p50 ms | predicted s |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in record["workload"]["cells"]:
        lines.append(
            f"| {item['cell']} | {item['train_queries']:,} | {item['held_out_queries']:,} | "
            f"{item['s4_train_seconds']:.1f} | {item['s4_uncached_p50_ms']:.3f} | "
            f"{item['seconds']:.1f} |"
        )
    lines += [
        "",
        record["workload"]["nothing_already_fit_is_refit"],
        "",
        "### Line items, per cell",
        "",
        "| cell | line | trains | seconds | basis |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in record["workload"]["cells"]:
        for line in item["lines"]:
            lines.append(
                f"| {item['cell']} | {line['line']} | "
                f"{'yes' if line['trains'] else 'no'} | {line['seconds']:.1f} | "
                f"{line['basis']} |"
            )
    lines += [
        "",
        "## Prediction",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| Measured work | {prediction['measured_work_seconds']:.0f} s |",
        (
            f"| Largest job | {prediction['largest_job']} at "
            f"{prediction['largest_single_job_seconds']:.0f} s |"
        ),
        f"| Feasible within timeout | {prediction['feasible_within_timeout']} |",
        f"| Utilisation assumed | {prediction['utilisation_assumed']} |",
        f"| Compute spend | ${prediction['compute_spend_usd']:.4f} |",
        f"| Container overhead | ${prediction['container_overhead_usd']:.4f} |",
        f"| **Expected spend** | **${prediction['expected_spend_usd']:.2f}** |",
        (
            f"| At slowest observed multiplier | "
            f"${prediction['spend_usd_at_slowest_observed_multiplier']:.2f} |"
        ),
        f"| **Hard ceiling** | **${prediction['hard_ceiling_usd']:.2f}** |",
        "",
        f"Container overhead source: {prediction['container_overhead_source']}.",
        "",
        "## Abort criteria",
        "",
    ]
    lines += [f"* {item}" for item in record["abort_criteria"]]
    lines += ["", "## What this record does not authorise", ""]
    lines += [f"* {item}" for item in record["what_this_record_does_not_authorise"]]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-declaration-block",
        action="store_true",
        help="emit the YAML block the declaration must carry, and write nothing",
    )
    args = parser.parse_args(argv)

    if args.print_declaration_block:
        # The FILED record, not a fresh one. The arm multipliers are measured
        # on the host and move a little between runs, so rebuilding here would
        # print a block describing a measurement that was never filed -- and
        # the declaration test that holds the two equal would then fail against
        # a number nobody chose. The block describes what is on disk.
        if not RECORD_JSON.is_file():
            raise SystemExit(
                f"{RECORD_JSON} does not exist. Generate the record first; the "
                "declaration block describes a filed record, not a fresh measurement."
            )
        filed = json.loads(RECORD_JSON.read_text(encoding="utf-8"))
        print(yaml.safe_dump(declaration_block(filed), sort_keys=False, width=100))
        return 0

    record = build_record()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RECORD_JSON.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    RECORD_MARKDOWN.write_text(render(record), encoding="utf-8")
    print(f"wrote {RECORD_JSON}")
    print(f"wrote {RECORD_MARKDOWN}")
    print(
        f"expected ${record['prediction']['expected_spend_usd']:.2f}, "
        f"ceiling ${record['prediction']['hard_ceiling_usd']:.2f}, "
        f"largest job {record['prediction']['largest_single_job_seconds']:.0f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
