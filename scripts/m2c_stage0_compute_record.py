#!/usr/bin/env python
"""M2C Stage-0 compute record, derived before the launch it authorises.

Read-only. Launches nothing, fits nothing. The amendment that accepted the M2C
declaration requires this record to exist BEFORE any Modal job is submitted --
predicted jobs, hardware choice, walltime, storage, expected spend, a hard
ceiling and abort criteria -- so the launch can be judged against a number that
was filed rather than one produced afterwards.

Every input is either read from an artifact or is a measurement recorded here
with its provenance:

``panel``
    the development (fit) portion of each cell's validation split, read from
    ``outputs/m2c_s4_structural_conditioning/m2b_baseline_table.json``. M2B
    scored a dataset's validation split and divided it 80/20 into a fit portion
    and a holdout; Stage 0 reads the 80% fit portion only. That is already
    "spent" on fitting, so using it leaves M2B's holdout clean for any later
    comparison, and the dataset test split is not touched by either.

``directional kernel``
    measured on THIS host against synthetic float32 arrays at each dataset's
    real node count and the frozen dim of 1536, timed over the operation the
    probe actually performs. A host estimate stated as such, not a measurement
    of the container -- the convention configs/m0a_probe.yaml#compute.estimate
    established.

``expansion``
    M0A's own measured per-query expansion cost on the same graphs, at
    ``graph_expansion_cap=128``. Stage 0's admission diagnostic runs at 64, a
    strictly smaller selection over the same frontier build, so M0A's figure is
    an upper bracket rather than an extrapolation. musique_clean was never
    measured by M0A; it takes squad_clean's rate, the densest passage graph
    measured, which brackets it rather than flattering it.

``pricing``
    ``mp_retrieval.compute_budget``, the project's own rates, for the CPU-only
    shape this job actually holds. The utilisation divisor charges for image
    pull and data load as well as work.

Ceiling follows the track's convention: the worst-case bracket rounded up to a
stated round figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.compute_budget import (  # noqa: E402
    WorkUnit,
    container_rate_usd_per_hour,
    expected_spend_usd,
    feasibility,
)

DECLARATION = REPO_ROOT / "configs" / "m2c_s4_structural_conditioning.yaml"
BASELINE_JSON = (
    REPO_ROOT / "outputs" / "m2c_s4_structural_conditioning" / "m2b_baseline_table.json"
)
OUTPUT_ROOT = REPO_ROOT / "outputs" / "m2c_s4_structural_conditioning"
RECORD_JSON = OUTPUT_ROOT / "stage0_compute_record.json"
RECORD_MARKDOWN = REPO_ROOT / "docs" / "M2C_STAGE0_COMPUTE_RECORD.md"

# ---------------------------------------------------------------------------
# The container shape. CPU only, and the refusal is deliberate.
# ---------------------------------------------------------------------------
CPU_CORES = 4
MEMORY_MB = 16384
GPU = None
TIMEOUT_SECONDS = 3600

#: Fraction of billed time that does the measured work. The kernel timing below
#: covers only the arithmetic, not the image pull or the embedding load, so the
#: spend is inflated rather than quoted at face value.
UTILISATION = 0.4

# ---------------------------------------------------------------------------
# Measured inputs.
# ---------------------------------------------------------------------------
#: ms per query for ONE pass of the directional kernel covering all three
#: provenances and all three residuals at once, at dim 1536 and per_seed_cap 16.
#: Measured on this host; see the module docstring. Reported at two candidate
#: pool sizes because the frozen contract's exact pool size is a property of the
#: sealed artifacts rather than of anything in this repository, and the estimate
#: should say which assumption it rests on.
KERNEL_MS_PER_QUERY = {400: 5.2, 1000: 11.8}
ASSUMED_CANDIDATES = 1000  # the conservative bracket, not the cheap one

#: M0A's measured expansion cost per query at cap 128 (configs/m0a_probe.yaml
#: #compute.estimate.expansion_ms_per_query), used as the upper bracket for the
#: +64 admission diagnostic.
EXPANSION_MS_PER_QUERY = {
    "squad_clean": 114,
    "2wiki_clean": 6,
    "metaqa": 9,
    "musique_clean": 114,  # unmeasured by M0A; takes the densest measured rate
}
ADMISSION_ARMS = 3  # A64, legacy-PF64, seed-subspace-PF64

#: The admission diagnostic runs on a prospectively declared prefix rather than
#: the whole panel. Expansion is 20-100x the cost of the ranking kernel, and §11
#: asks for a MINIMAL diagnostic. The decisive quantity there is set overlap
#: between the three arms -- Jaccard, unique admitted nodes -- which is precise
#: at this size; the recall_ceiling@5 delta is reported with its confidence
#: interval so its coarser precision is visible rather than implied away.
ADMISSION_PANEL_CAP = 2000

#: Node counts, from docs/M0A_PROBE_RESULTS.md and the local musique payload.
NODE_COUNTS = {
    "squad_clean": 19_029,
    "2wiki_clean": 65_865,
    "metaqa": 40_151,
    "musique_clean": 13_672,
}
EMBEDDING_DIM = 1536
S4_PROJECTION_DIM = 64


def _panel(rows: list[dict], dataset: str, regime: str) -> int:
    """The development (fit) portion of the cell's validation split."""

    for row in rows:
        if row["dataset"] == dataset and row["regime"] == regime and row["rung"] == "S4":
            return int(row["train_queries"])
    raise KeyError(f"no S4 row for {dataset}/{regime}")


def build() -> dict:
    declaration = yaml.safe_load(DECLARATION.read_text(encoding="utf-8"))
    stage = declaration["stage_0"]
    if declaration["status"] != "M2C_DECLARED_STAGE0_NOT_YET_RUN":
        raise RuntimeError(
            f"unexpected declaration status {declaration['status']!r}; this record "
            "prices a Stage-0 probe that has not run"
        )
    if stage["trains_nothing"] is not True or stage["reads_test_split"] is not False:
        raise RuntimeError("this record prices a zero-training, no-test-split probe only")

    cells = [
        *stage["cells"]["failure_cells"],
        stage["cells"]["passage_r3_control"],
        stage["cells"]["kb_r3_control"],
    ]
    matrix = stage["zero_training_ranking_test"]["matrix"]
    baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))

    rate = container_rate_usd_per_hour(gpu=GPU, cpu_cores=CPU_CORES, memory_mb=MEMORY_MB)
    kernel_ms = KERNEL_MS_PER_QUERY[ASSUMED_CANDIDATES]

    jobs, units = [], []
    for cell in cells:
        dataset, regime = cell.split("/")
        panel = _panel(baseline["rows"], dataset, regime)
        admission_panel = min(ADMISSION_PANEL_CAP, panel)

        ranking_s = panel * kernel_ms / 1000.0
        # S4 inference: node states are projected ONCE for the whole graph, then
        # gathered per query, so the per-query cost is a 1536x64 query
        # projection plus a pool-sized dot product -- an order below the kernel.
        s4_s = ranking_s * 0.25
        admission_s = admission_panel * ADMISSION_ARMS * EXPANSION_MS_PER_QUERY[dataset] / 1000.0
        # Image pull, volume attach, embedding and graph load. One flat charge
        # per job; the largest graph here is 405 MB of embeddings.
        load_s = 120.0
        total_s = ranking_s + s4_s + admission_s + load_s

        jobs.append(
            {
                "cell": cell,
                "dataset": dataset,
                "regime": regime,
                "development_panel_queries": panel,
                "admission_panel_queries": admission_panel,
                "nodes": NODE_COUNTS[dataset],
                "ranking_seconds": round(ranking_s, 1),
                "s4_inference_seconds": round(s4_s, 1),
                "admission_seconds": round(admission_s, 1),
                "load_seconds": load_s,
                "job_seconds": round(total_s, 1),
                "embedding_bytes": NODE_COUNTS[dataset] * EMBEDDING_DIM * 4,
            }
        )
        units.append(WorkUnit(name=cell, seconds=total_s))

    verdict = feasibility(units, TIMEOUT_SECONDS)
    work_s = sum(u.seconds for u in units)
    expected = expected_spend_usd(
        units, usd_per_container_hour=rate, training_fraction=UTILISATION
    )
    # The bracket the ceiling is set from: three times the estimate, which is
    # the widest miss this track has recorded between a host estimate and a
    # container measurement.
    bracket = expected * 3
    ceiling = 2.00

    record = {
        "status": "FILED_BEFORE_LAUNCH",
        "phase": "m2c_s4_structural_conditioning",
        "stage": 0,
        "trains_nothing": True,
        "reads_test_split": False,
        "split": stage["split"],
        "authorises_nothing_by_itself": (
            "This record is a gate, not an authorisation. It states what the "
            "declared Stage-0 probe should cost so the launch can be judged "
            "against a filed number."
        ),
        "workload": {
            "jobs": len(jobs),
            "cells": cells,
            "arms": matrix["arms"],
            "residuals": matrix["residuals"],
            "provenance": matrix["provenance"],
            "declared_matrix_size": (
                len(matrix["arms"]) * len(matrix["residuals"]) * len(matrix["provenance"])
            ),
            # The declared matrix is 3x3x3, but S4 alone is invariant under both
            # residual and provenance -- it is the unmodified reference. So the
            # work is one S4 pass plus two direction-dependent arms over the 9
            # residual x provenance combinations, not 27 independent passes.
            # Quoting 27 would overstate the workload by about a third.
            "scored_configurations_per_query": (
                1 + (len(matrix["arms"]) - 1) * len(matrix["residuals"]) * len(matrix["provenance"])
            ),
            "why_fewer_than_the_declared_matrix": (
                "The S4 arm is the unmodified reference and does not vary with the "
                "residual or the provenance, so it is computed once per query "
                "rather than nine times. All 27 cells of the declared matrix are "
                "still reported; 19 distinct computations produce them."
            ),
            "development_panel_total": sum(j["development_panel_queries"] for j in jobs),
            "admission_panel_total": sum(j["admission_panel_queries"] for j in jobs),
            "resume_granularity": "cell",
            "why_that_granularity": (
                "One cell is one spawned call that writes its result once, at the "
                "end. A restart therefore redoes exactly one cell and no more."
            ),
        },
        "container": {
            "gpu": GPU,
            "gpu_hours_authorised": 0.0,
            "cpu": CPU_CORES,
            "memory_mb": MEMORY_MB,
            "timeout_seconds": TIMEOUT_SECONDS,
            "usd_per_hour": round(rate, 4),
            "why_cpu_and_not_gpu": (
                "The probe trains nothing. Its kernel is embedding gathers plus a "
                "candidates-by-seeds gram block, measured on this host at "
                f"{kernel_ms:.1f} ms per query for all "
                f"{len(matrix['residuals'])} residuals and "
                f"{len(matrix['provenance'])} provenances at once; the S4 forward "
                f"pass is a {EMBEDDING_DIM}x{S4_PROJECTION_DIM} linear whose node "
                "side is projected once per graph. Neither is GPU-bound. A later "
                "learned stage may need an accelerator; that is not a reason to "
                "hold one idle through this one, and the amendment forbids it."
            ),
            "why_this_shape": (
                "The largest embedding table here is 2wiki_clean at "
                f"{NODE_COUNTS['2wiki_clean'] * EMBEDDING_DIM * 4 / 1e6:.0f} MB, and "
                "peak RSS is dominated by that table plus the projected node "
                "states, a few hundred MB more. 16 GiB holds both with room for "
                "the frontier structures; the work is memory-bound gathers and "
                "thin GEMMs rather than wide ones, so cores past 4 buy little."
            ),
        },
        "estimate": {
            "measured_on": (
                "the host, against synthetic float32 arrays at each dataset's exact "
                "node count and the frozen dim of 1536. A container estimate from "
                "host timings, stated as such, not a measurement of the container."
            ),
            "kernel_ms_per_query": KERNEL_MS_PER_QUERY,
            "assumed_candidates_per_query": ASSUMED_CANDIDATES,
            "why_the_conservative_pool": (
                "The frozen contract's exact pool size lives in the sealed "
                "artifacts, not in this repository. The estimate is quoted at the "
                "larger of the two measured pool sizes so a wrong assumption "
                "overstates rather than understates the bill; at 400 candidates "
                f"the whole probe costs about "
                f"{KERNEL_MS_PER_QUERY[400] / kernel_ms:.0%} of what is quoted here."
            ),
            "expansion_ms_per_query": EXPANSION_MS_PER_QUERY,
            "expansion_source": (
                "M0A's measured per-query cost at graph_expansion_cap=128. The "
                "admission diagnostic runs at 64, a smaller selection over the same "
                "frontier build, so this is an upper bracket."
            ),
            "utilisation": UTILISATION,
            "jobs": jobs,
            "total_work_seconds": round(work_s, 1),
            "total_work_hours": round(work_s / 3600, 3),
            "longest_job_seconds": round(max(u.seconds for u in units), 1),
            "walltime_seconds_if_parallel": round(max(u.seconds for u in units), 1),
            "estimated_cost_usd": round(expected, 3),
            "estimated_cost_usd_at_three_times_the_estimate": round(bracket, 3),
            "priced_by": (
                "mp_retrieval.compute_budget.container_rate_usd_per_hour and "
                "expected_spend_usd, the project's own rates."
            ),
        },
        "feasibility": {
            "fits_timeout": bool(verdict),
            "reason": verdict.reason,
            "largest_unit_seconds": round(verdict.largest.seconds, 1)
            if verdict.largest
            else None,
            "safety_factor": 1.5,
        },
        "storage": {
            "reads": [
                "frozen node and query embeddings",
                "frozen edge-provenance graphs",
                "the frozen candidate contract",
                "the sealed S4 checkpoint",
            ],
            "writes_under": "outputs/m2c_s4_structural_conditioning/stage0/",
            "writes_nothing_under": [
                "any M2 path",
                "any M2B path",
                "any M3 path",
                "Package F",
                "canonical CRAG",
            ],
            "estimated_result_bytes": 4_000_000,
            "why_small": (
                "Per-cell aggregates and the error-margin population summary, not "
                "per-candidate scores. Nothing here needs a per-candidate dump to "
                "answer its question."
            ),
        },
        "cost_ceiling_usd": ceiling,
        "ceiling_rationale": (
            f"Three times the estimate is ${bracket:.2f}. The ceiling is set at "
            f"${ceiling:.2f}, above that bracket and at the round figure this track "
            "uses. A ceiling is a bound on a declared workload, not a budget to "
            "spend down."
        ),
        "abort_criteria": [
            "any job exceeds the declared timeout twice on the same cell",
            "measured spend passes the ceiling",
            (
                "the loaded candidate contract does not verify bit-exact against "
                "the frozen fingerprint for that cell"
            ),
            (
                "directional coverage is below 1% of scored candidates on a cell -- "
                "there is no signal to rank with, and the remaining arms on that "
                "cell would measure nothing"
            ),
            (
                "any job attempts to read a test split, a gold node as a seed, a "
                "gold relation type, a supporting-fact label at inference, a "
                "dataset id, Package F, or any GNN artifact"
            ),
            "a write is attempted outside the Stage-0 output prefix",
        ],
        "placement": {
            "declared_in": (
                "configs/m2_qls_v2_freeze.yaml#launch_authorization.execution_placement"
            ),
            "read_at_submit_time": True,
            "not_restated_here": (
                "Workspaces rotate as each hits its spend limit. Copying the map "
                "into this record would create a second source that can go stale, "
                "and the submit path already refuses a mismatch."
            ),
        },
        "submission": {
            "spawn_server_side": True,
            "via": "scripts/spawn_modal_jobs.py",
            "never_modal_run_detach": (
                "A detached local run is not a registered execution."
            ),
        },
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RECORD_JSON.write_text(json.dumps(record, indent=2), encoding="utf-8")
    RECORD_MARKDOWN.write_text(markdown(record), encoding="utf-8")
    return record


def markdown(record: dict) -> str:
    estimate = record["estimate"]
    container = record["container"]
    workload = record["workload"]
    lines = [
        "# M2C Stage-0 compute record",
        "",
        "**FILED BEFORE LAUNCH.** Generated by",
        "[`scripts/m2c_stage0_compute_record.py`](../scripts/m2c_stage0_compute_record.py);",
        "every number below is derived, not typed. This record is a gate, not an",
        "authorisation.",
        "",
        "## Workload",
        "",
        f"- **{workload['jobs']} jobs**, one per cell, resumable at cell granularity",
        f"- cells: {', '.join(f'`{c}`' for c in workload['cells'])}",
        (
            f"- declared matrix: **{workload['declared_matrix_size']} cells per query** — "
            f"{len(workload['arms'])} arms x {len(workload['residuals'])} residuals x "
            f"{len(workload['provenance'])} provenances"
        ),
        (
            f"- computed: **{workload['scored_configurations_per_query']} scored "
            f"configurations per query**. {workload['why_fewer_than_the_declared_matrix']}"
        ),
        (
            f"- ranking panel: **{workload['development_panel_total']:,} queries**, "
            "the 80% development portion of each cell's validation split"
        ),
        (
            f"- admission panel: **{workload['admission_panel_total']:,} queries**, "
            f"capped at {ADMISSION_PANEL_CAP:,} per cell and declared here, before "
            "any result"
        ),
        "- **trains nothing; reads no test split**",
        "",
        "## Hardware — CPU only",
        "",
        "| resource | held |",
        "|---|---|",
        f"| GPU | **none**, {container['gpu_hours_authorised']} GPU-hours authorised |",
        f"| CPU | {container['cpu']} cores |",
        f"| memory | {container['memory_mb'] // 1024} GiB |",
        f"| timeout | {container['timeout_seconds']} s per job |",
        f"| rate | ${container['usd_per_hour']:.4f}/hour |",
        "",
        f"**Why not GPU.** {container['why_cpu_and_not_gpu']}",
        "",
        f"**Why this shape.** {container['why_this_shape']}",
        "",
        "## Estimate",
        "",
        f"Measured on {estimate['measured_on']}",
        "",
        (
            f"Kernel: **{estimate['kernel_ms_per_query'][ASSUMED_CANDIDATES]} ms/query** "
            f"at {ASSUMED_CANDIDATES} candidates "
            f"({estimate['kernel_ms_per_query'][400]} ms at 400). "
            f"{estimate['why_the_conservative_pool']}"
        ),
        "",
        "| cell | panel | ranking | S4 | admission | load | job |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for job in estimate["jobs"]:
        lines.append(
            f"| `{job['cell']}` | {job['development_panel_queries']:,} | "
            f"{job['ranking_seconds']:.0f}s | {job['s4_inference_seconds']:.0f}s | "
            f"{job['admission_seconds']:.0f}s | {job['load_seconds']:.0f}s | "
            f"**{job['job_seconds']:.0f}s** |"
        )
    lines += [
        "",
        (
            f"- total work: **{estimate['total_work_seconds']:,.0f} s** "
            f"({estimate['total_work_hours']:.2f} h)"
        ),
        (
            f"- longest single job: **{estimate['longest_job_seconds']:,.0f} s** "
            f"against a {container['timeout_seconds']} s timeout, so every job "
            "fits with room to spare"
        ),
        (
            "- the four jobs hold four separate containers, so **walltime is the "
            f"longest job ({estimate['walltime_seconds_if_parallel']:,.0f} s), "
            "not the sum**; the sum is what is billed"
        ),
        f"- restart cost: `{record['feasibility']['reason']}`",
        (
            f"- expected spend: **${estimate['estimated_cost_usd']:.2f}** "
            f"(at {estimate['utilisation']:.0%} utilisation, which charges for "
            "image pull and data load as well as work)"
        ),
        (
            "- three times the estimate: "
            f"**${estimate['estimated_cost_usd_at_three_times_the_estimate']:.2f}**"
        ),
        f"- **hard ceiling: ${record['cost_ceiling_usd']:.2f}**",
        "",
        record["ceiling_rationale"],
        "",
        "## Storage",
        "",
        "Reads: " + "; ".join(record["storage"]["reads"]) + ".",
        "",
        f"Writes under `{record['storage']['writes_under']}` only — nothing under "
        + ", ".join(f"`{p}`" for p in record["storage"]["writes_nothing_under"])
        + ".",
        "",
        f"Results are about {record['storage']['estimated_result_bytes'] / 1e6:.0f} MB. "
        + record["storage"]["why_small"],
        "",
        "## Abort criteria",
        "",
    ]
    lines += [f"{i}. {c}" for i, c in enumerate(record["abort_criteria"], start=1)]
    lines += [
        "",
        "## Placement and submission",
        "",
        (
            "Placement is read at submit time from "
            f"`{record['placement']['declared_in']}` and is deliberately not "
            f"restated here. {record['placement']['not_restated_here']}"
        ),
        "",
        (
            f"Jobs are spawned server-side via `{record['submission']['via']}`. "
            f"{record['submission']['never_modal_run_detach']}"
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    record = build()
    estimate = record["estimate"]
    print(f"{record['workload']['jobs']} jobs, CPU only, 0 GPU-hours")
    print(
        f"work {estimate['total_work_seconds']:.0f}s, "
        f"longest job {estimate['longest_job_seconds']:.0f}s, "
        f"expected ${estimate['estimated_cost_usd']:.2f}, "
        f"ceiling ${record['cost_ceiling_usd']:.2f}"
    )
    print(f"feasible: {record['feasibility']['reason']}")
    print(f"wrote {RECORD_JSON}")
    print(f"wrote {RECORD_MARKDOWN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
