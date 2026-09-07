#!/usr/bin/env python
"""Submit registered package jobs as persistent Modal calls that outlive the client.

``modal run --detach`` keeps only the last triggered function alive once the
launching process is killed, so a multi-hour package launched from an
interactive session dies with that session. Deploying the app and spawning each
job creates server-side calls that survive client disconnection.

This changes only how work is submitted. The same registered runners execute the
same frozen protocol, they remain idempotent, and they still write every result
to the shared volume, so a resumed package reuses completed cells rather than
recomputing them.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from modal.runner import deploy_app

from mp_retrieval.compute_budget import (
    PHASE_CONFIRMATION_SECONDS_PER_SEED,
    TRAINING_FRACTION_OF_BILLED_TIME,
    WorkUnit,
    container_rate_usd_per_hour,
    expected_spend_usd,
    feasibility,
    phase_confirmation_units,
    substrate_family_units,
)

PACKAGES: dict[str, tuple[str, dict[str, str]]] = {
    "edge-provenance": (
        "scripts.modal_edge_provenance",
        {"prepare": "prepare_dataset", "train": "run_family"},
    ),
    "candidate-budget": ("scripts.modal_candidate_budget", {"train": "run_budget"}),
    "phase-screen": ("scripts.modal_phase_screen", {"train": "run_regime"}),
    "candidate-headroom": ("scripts.modal_candidate_headroom", {"train": "run_headroom"}),
    "online-systems": ("scripts.modal_online_systems", {"train": "run_dataset"}),
    "phase-confirmation": ("scripts.modal_phase_confirmation", {"train": "run_cell"}),
    "graph-substrate": (
        "scripts.modal_graph_substrate_audit",
        {"train": "run_substrate"},
    ),
    "cache-equivalence": (
        "scripts.modal_cache_equivalence",
        {"train": "run_equivalence"},
    ),
    "graph-context": (
        "scripts.modal_graph_context_pilot",
        {"train": "run_context_pilot"},
    ),
    "m0a-probe": ("scripts.modal_m0a_probe", {"train": "run_probe"}),
    "m0a1-overlap": ("scripts.modal_m0a1_overlap", {"train": "run_overlap"}),
    "m0b-webqsp-probe": ("scripts.modal_m0b_webqsp_probe", {"train": "run_probe"}),
    "m0b-regime-map": (
        "scripts.modal_m0b_regime_map",
        {"smoke": "run_regime_map_smoke", "headline": "run_regime_map_headline"},
    ),
    "m0c-bounded-r3": (
        "scripts.modal_m0c_bounded_r3",
        {"smoke": "run_bounded_r3_smoke", "headline": "run_bounded_r3_headline"},
    ),
    "m1a-feature-screen": (
        "scripts.modal_m1a_feature_screen",
        {"smoke": "run_feature_screen_smoke", "headline": "run_feature_screen_headline"},
    ),
    "m1b-targeted-resolution": (
        "scripts.modal_m1b_targeted_resolution",
        {"smoke": "run_targeted_resolution_smoke", "headline": "run_targeted_resolution_headline"},
    ),
    # Three stages rather than two: M2's declared smoke is a pair. The R2
    # primary cannot exercise a nonzero NODE_ROLE column (it is zero there by
    # construction), so the R3 secondary is a separate submission rather than a
    # flag on the first.
    "m2-qls-v2-freeze": (
        "scripts.modal_m2_qls_v2_freeze",
        {
            "smoke": "run_m2_smoke",
            "smoke_build": "run_m2_smoke_build",
            "smoke_fit": "run_m2_smoke_fit",
            "secondary_smoke": "run_m2_secondary_smoke",
            "build": "run_m2_feature_build",
            "headline": "run_m2_headline",
        },
    ),
    # One stage, one spawned call per declared cell, and nothing fitted. The
    # probe reads M2's sealed cell masters and M2B's sealed S4 checkpoint and
    # writes only under M2C's own prefix, so there is no build stage to run and
    # no historical artifact for a stage to overwrite.
    "m2c-stage0-probe": ("scripts.modal_m2c_stage0_probe", {"probe": "run_stage0"}),
    # Three stages, and no build stage at all: M2B rebuilds no features. Every
    # cell master it reads was persisted by M2's build stage and is admitted on
    # its feature build contract. One spawned job per dataset runs every regime
    # and every semantic rung inside one container, so the rungs share a clock
    # -- the tie-break orders on uncached inference p95, and comparing that
    # across containers would compare containers. The resolution keeps that
    # property: one container per resolution cell runs both new seeds.
    "m2b-semantic-minimality": (
        "scripts.modal_m2b_semantic_minimality",
        {
            "smoke": "run_m2b_smoke",
            "headline": "run_m2b_headline",
            # Amendment 3. Runs only under the resolution status, and only for
            # the two cells that status names.
            "resolution": "run_m2b_resolution",
        },
    ),
}

# Stage B replaced the pre-launch guess with measurements, so this is now
# per-query-per-ARM rather than per-query: the cost of a job scales with how many
# arms it runs, and Stage C runs three where Stage B ran six.
#
# Measured on Modal, p95 total per query per arm, validation split:
#
#     arm         2wiki      hotpotqa
#     CAND         5.3 ms      84.2 ms
#     SEED_H1      4.8 ms     173.5 ms
#     TARGET_H1    7.9 ms     221.8 ms
#     PATH_H2     12.7 ms     302.2 ms   killed at B
#     BRIDGE_H2    9.6 ms     306.8 ms   killed at B
#     SEED_H2     81.2 ms   2,635.6 ms   killed at B
#
# 1.5 s per query per arm clears every surviving arm by about sevenfold and the
# six-arm mean by about threefold. Only SEED_H2 on hotpotqa exceeds it, and
# SEED_H2 is killed; if it is ever revived this number has to be re-derived.
#
# The load term is the fixed cost before the first query -- container start, the
# dataset, the CSR operators, and one Numba compilation. It shows up as a single
# outlier: CAND runs first and its max is 6.7 s and 9.3 s against medians of 2.5
# and 73 ms. Measured end to end at 33 s (2wiki) and 278 s (hotpotqa) for a whole
# 25-query six-arm split, so 600 s stays a ceiling rather than an estimate.
GRAPH_CONTEXT_SECONDS_PER_QUERY_PER_ARM = 1.5
GRAPH_CONTEXT_LOAD_SECONDS = 600.0

# Stage D0b does not fit the model above, and applying it anyway would be worse
# than having no model: `query_cap` is 0 there because the stage runs whole
# splits, so the estimate would collapse to the load term and the gate would wave
# through a job of any size. The two failures point in opposite directions and
# both are wrong, so D0b gets its own term.
#
# Stage C measured build + kernel per query on 2wiki at p99 4.01 ms (CAND) and
# 9.84 ms (TARGET_H1). D0b also computes the seed-support quantities and the
# induced degree per query, so the per-query figure below is roughly threefold
# that 13.85 ms rather than equal to it. Training is four linear models of at
# most twenty parameters and is bounded separately; it is small but not zero,
# because each epoch re-scores the whole validation split.
GRAPH_CONTEXT_D0B_SECONDS_PER_QUERY = 0.040
GRAPH_CONTEXT_D0B_TRAINING_SECONDS = 300.0

# D1 builds one context per arm rather than both at once, so its per-query cost
# is the sum of the two arms' p95 context latencies as D0b measured them on this
# dataset and image: CAND 9.8 ms + TARGET_H1 24.5 ms. p95 rather than the mean
# because this feeds a gate.
#
# Training is measured, not modelled: the frozen sa_mlp confirmation fits 2wiki
# in 21.4 s per seed on the same GPU shape. Two arms, doubled for the per-epoch
# rescoring of the epoch-selection holdout that D1 adds and the confirmation did
# not have at this size.
GRAPH_CONTEXT_D1_SECONDS_PER_QUERY = 0.035
GRAPH_CONTEXT_D1_TRAINING_SECONDS = 120.0

# D0c builds the same two contexts as D0b over the same queries and adds a
# per-candidate degree pass, so its per-query term is D0b's plus a margin for
# that pass. Training is six linear arms for ten epochs where D0b ran four for
# three -- five times the arm-epochs -- and each epoch now rescores a tail as
# well, so D0b's 300 s bound is carried up rather than scaled: D0b's measured
# training was 15.4 s, which is not the quantity this term is protecting
# against.
GRAPH_CONTEXT_D0C_SECONDS_PER_QUERY = 0.045
GRAPH_CONTEXT_D0C_TRAINING_SECONDS = 400.0


def _graph_context_d0c_seconds(module: Any, job: dict[str, Any]) -> float:
    """D0b's shape at D0c's arm count. Over-counts by the test split, as D0b does."""
    protocol_queries = int(job["settings"]["expected_queries"])
    return (
        GRAPH_CONTEXT_LOAD_SECONDS
        + protocol_queries * GRAPH_CONTEXT_D0C_SECONDS_PER_QUERY
        + GRAPH_CONTEXT_D0C_TRAINING_SECONDS
    )


def _graph_context_d1_seconds(module: Any, job: dict[str, Any]) -> float:
    """Both arms' whole-split feature build, plus the load ceiling, plus training.

    The 600 s load term is doubly a ceiling here: D1 is the first stage in this
    line to read the embeddings, which is 474 MB this launcher has never timed.
    """
    cap = int(job["query_cap"])
    protocol_queries = int(job["settings"]["expected_queries"])
    # Over-counts by the test split, which D1 refuses to open. The safe
    # direction for a gate.
    queries = protocol_queries if cap <= 0 else min(cap, protocol_queries)
    return (
        GRAPH_CONTEXT_LOAD_SECONDS
        + queries * GRAPH_CONTEXT_D1_SECONDS_PER_QUERY
        + GRAPH_CONTEXT_D1_TRAINING_SECONDS
    )


def _graph_context_d0b_seconds(module: Any, job: dict[str, Any]) -> float:
    """Whole-split feature build, plus the load ceiling, plus linear training.

    `query_cap` 0 means every query in both development splits, so the count
    comes from the registered protocol rather than from the cap.
    """
    cap = int(job["query_cap"])
    protocol_queries = int(job["settings"]["expected_queries"])
    # The registered count is every split. D0b opens only train and validation,
    # so this deliberately over-counts by the size of the test split -- the safe
    # direction for a gate, and cheaper than teaching the launcher a split
    # arithmetic it would then have to keep in step with the loader.
    queries = protocol_queries if cap <= 0 else min(cap, protocol_queries)
    return (
        GRAPH_CONTEXT_LOAD_SECONDS
        + queries * GRAPH_CONTEXT_D0B_SECONDS_PER_QUERY
        + GRAPH_CONTEXT_D0B_TRAINING_SECONDS
    )


def _expand(module: Any, package: str, stage: str, datasets: list[str]) -> list[dict[str, Any]]:
    jobs = module._jobs(datasets)
    if package == "edge-provenance" and stage == "train":
        return [
            {**job, "family": family}
            for job in jobs
            for family in module.CONFIG["trained_families"]
        ]
    return jobs


# ---------------------------------------------------------------------------
# Resuming from measured state
# ---------------------------------------------------------------------------

# Cells whose action the matrix reports as one of these are submitted. `skip`
# is a finished cell; `diagnose` is a cell whose recorded contract disagrees
# with the cell it was launched for, which is a stop rather than a relaunch.
SUBMITTED_ACTIONS = ("resume", "launch")


def _cell_key(job: dict[str, Any]) -> str:
    """The key ``migration_provenance.classify_cell`` builds for the same cell."""

    return f"{job['dataset']}/{job['axis']}/rate_{float(job['rate']):.2f}"


def filter_by_matrix(
    jobs: list[dict[str, Any]], matrix: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Submit the cells the measured state says still need work, and no others.

    A resume plan is a claim about what exists, so it has to come from a
    measurement of the results root rather than from a remembered cell number.
    Three rules, in the order they can go wrong:

    *   A cell the matrix does not mention means the matrix and the job
        expansion disagree about what the sweep *is* -- a stale matrix, or one
        built for a different config. Refused, because the missing cells are
        exactly the ones that would then never run.
    *   An ``INVALID`` cell is a stop. Its recorded result disagrees with the
        cell it was launched for, and relaunching would overwrite the evidence
        of whatever produced it.
    *   Everything skipped is named in the returned plan. A launcher that
        quietly submits 10 of 96 jobs looks identical to one that submitted
        everything and found little to do.
    """

    by_key = {row["key"]: row for row in matrix["cells"]}
    unknown = sorted(_cell_key(job) for job in jobs if _cell_key(job) not in by_key)
    if unknown:
        raise SystemExit(
            f"{len(unknown)} requested cell(s) absent from the integrity matrix, "
            f"starting with {unknown[0]!r}. The matrix does not describe this "
            "sweep; rebuild it against the results root these jobs write to."
        )

    invalid = sorted(
        key
        for key in (_cell_key(job) for job in jobs)
        if by_key[key]["action"] == "diagnose"
    )
    if invalid:
        raise SystemExit(
            f"{len(invalid)} cell(s) are INVALID and must be diagnosed before any "
            f"launch: {', '.join(invalid)}. "
            + "; ".join(f"{key}: {by_key[key]['detail']}" for key in invalid)
        )

    submitted = [job for job in jobs if by_key[_cell_key(job)]["action"] in SUBMITTED_ACTIONS]
    plan = {
        "matrix_results_root": matrix.get("results_root"),
        "requested": len(jobs),
        "submitted": len(submitted),
        "resume": sorted(
            key
            for key in (_cell_key(job) for job in jobs)
            if by_key[key]["action"] == "resume"
        ),
        "launch": sorted(
            key
            for key in (_cell_key(job) for job in jobs)
            if by_key[key]["action"] == "launch"
        ),
        "skipped_complete": sorted(
            key
            for key in (_cell_key(job) for job in jobs)
            if by_key[key]["action"] == "skip"
        ),
    }
    return submitted, plan


# ---------------------------------------------------------------------------
# Refusing a launch that cannot finish
# ---------------------------------------------------------------------------

# Validation queries behind the substrate audit's measured per-family cost. Only
# hotpotqa_clean is listed because the other five audits are complete and were
# never timed; a dataset absent here is reported as ungated rather than given an
# invented query count.
SUBSTRATE_VALIDATION_QUERIES = {"hotpotqa_clean": 19_570}
SUBSTRATE_EXPANSION_CAP = 512

# Model seeds per E2 cell, from configs/phase_confirmation.yaml.
PHASE_CONFIRMATION_SEEDS = 5

# How much of a package's billed time does the work its measurement covers. A
# confirmation container spends the rest pulling images and loading data with
# the A10G attached and idle; the substrate audit's per-query cost already
# includes its own loading, so inflating it again would double-count. An
# unlisted package gets the lower figure, which overstates rather than
# understates a bill.
UTILISATION = {
    "phase-confirmation": TRAINING_FRACTION_OF_BILLED_TIME,
    "graph-substrate": 1.0,
    # The figure M2C's own compute record was derived at. Held here so the
    # number this gate reports and the number the filed record predicted are
    # the same number, and a divergence between them is a real divergence
    # rather than two different utilisation assumptions.
    "m2c-stage0-probe": 0.4,
}


def _collapse_without_resumption(
    units: list[WorkUnit], module: Any, expected: str, label: str
) -> tuple[list[WorkUnit], str]:
    """Cost a job as one unit unless the runner declares it resumes inside one.

    This is the distinction the substrate audit failure actually turned on, and
    getting it wrong makes the gate miss the very run it exists to catch. One
    hotpotqa family is 3.31 h, which fits a six-hour ceiling with room to spare
    -- so per-family units would have waved that launch through. What made it
    fatal was that nothing was carried across a restart: with no resumption the
    piece a restart redoes is the *whole* four-family audit, 13.2 h, and no
    number of retries against a six-hour window ever finishes it.

    A runner therefore has to declare where it checkpoints, and silence is
    costed as no checkpointing at all. That is the safe direction: an
    undeclared package is treated as redoing everything, which can only refuse
    a launch that would have been admitted, never admit one that should not be.
    """

    if getattr(module, "RESUME_GRANULARITY", None) == expected:
        return units, f"unit = one {expected}"
    total = sum(unit.seconds for unit in units)
    return (
        [WorkUnit(f"{label} (no {expected}-level resumption)", total)],
        (
            f"unit = the whole {label}, because the runner does not declare "
            f"{expected}-level resumption and a restart redoes all of it"
        ),
    )


#: Where M2's committed estimate keeps its per-dataset serial chain: feature
#: build plus every new fit on that dataset, in minutes.
M2_CHAIN_KEY = "per_dataset_serial_chain_minutes_conservative"


def _m2_chain_seconds(module: Any) -> dict[str, float]:
    """Per-dataset chain seconds, read from the artifact the declaration names.

    Read from the manifest rather than restated here so a re-run of
    scripts/m2_compute_estimate.py cannot leave this gate quoting a number no
    artifact supports -- the same binding the declaration's own gates use.
    """

    manifest = Path(module.HOST_REPO_ROOT) / module.CONFIG["compute"]["manifest"]
    estimate = json.loads(manifest.read_text(encoding="utf-8"))
    chains = estimate["wall_clock_with_per_dataset_parallelism"][M2_CHAIN_KEY]
    return {dataset: float(minutes) * 60.0 for dataset, minutes in chains.items()}


def measured_units(
    package: str, module: Any, jobs: list[dict[str, Any]]
) -> tuple[list[WorkUnit] | None, str]:
    """Indivisible work units for a launch, or None when nothing was measured.

    The gate refuses only on evidence. Where no per-unit cost was ever measured
    this returns None and the launch proceeds ungated: inventing a number would
    produce a confident verdict with nothing behind it, and a gate that blocks
    on guesses is one that gets bypassed and then protects nothing.
    """

    if package == "phase-confirmation":
        unknown = sorted(
            {job["dataset"] for job in jobs} - set(PHASE_CONFIRMATION_SECONDS_PER_SEED)
        )
        if unknown:
            return None, f"no measured per-seed cost for {', '.join(unknown)}"
        # Every cell is costed at its full seed count, which overstates a
        # partly-finished cell's total but leaves the unit -- the only thing the
        # ceiling is compared against -- exact.
        units, granularity = _collapse_without_resumption(
            phase_confirmation_units(
                [(job["dataset"], PHASE_CONFIRMATION_SEEDS) for job in jobs]
            ),
            module,
            "seed",
            "sweep",
        )
        return units, f"{len(jobs)} cell(s) at {PHASE_CONFIRMATION_SEEDS} seeds each; {granularity}"

    if package == "graph-substrate":
        unknown = sorted({job["dataset"] for job in jobs} - set(SUBSTRATE_VALIDATION_QUERIES))
        if unknown:
            return None, f"no measured validation query count for {', '.join(unknown)}"
        families = list(module.CONFIG["graphs"])
        units: list[WorkUnit] = []
        for job in jobs:
            units.extend(
                substrate_family_units(
                    queries=SUBSTRATE_VALIDATION_QUERIES[job["dataset"]],
                    families=families,
                    expansion_cap=SUBSTRATE_EXPANSION_CAP,
                )
            )
        units, granularity = _collapse_without_resumption(units, module, "family", "audit")
        return units, f"{len(jobs)} dataset(s) x {len(families)} families; {granularity}"

    if package == "graph-context":
        units = [
            WorkUnit(
                name=f"{job['dataset']}:{job['stage']}",
                seconds=(
                    _graph_context_d0b_seconds(module, job)
                    if job["stage"] == "stage_d0b"
                    else _graph_context_d0c_seconds(module, job)
                    if job["stage"] == "stage_d0c"
                    else _graph_context_d1_seconds(module, job)
                    # D2 builds one context for both arms where D1 built two,
                    # and builds the cheaper of the two, so D1's model is a
                    # ceiling. D3 shares that one build across four arms but
                    # trains only two of them, so it is bounded by D2 in turn;
                    # D4 and D5 share the same build again while training one
                    # arm each, and D6 trains two on it -- still D3's shape, so
                    # D1's model remains the ceiling for all of them. D7 trains
                    # four arms on that same single build and refits neither of
                    # D6's, so it is bounded the same way; D8 trains one on it
                    # and refits neither D6's base nor D7's support arm, which
                    # is cheaper again, and D9 is cheaper still because it can
                    # decide at its own gate not to train at all. It is a
                    # ceiling and not a per-stage forecast, which is why every
                    # one of these stages reports the same figure.
                    if job["stage"] in {
                        "stage_d1", "stage_d2", "stage_d3",
                        "stage_d4", "stage_d5", "stage_d6", "stage_d7",
                        "stage_d8", "stage_d9", "stage_d10",
                    }
                    else GRAPH_CONTEXT_LOAD_SECONDS
                    + int(job["query_cap"])
                    * len(job.get("arms") or module.CONFIG["arms"])
                    * GRAPH_CONTEXT_SECONDS_PER_QUERY_PER_ARM
                ),
            )
            for job in jobs
        ]
        # One split, committed once, so a restart redoes the whole dataset.
        units, granularity = _collapse_without_resumption(units, module, "split", "pilot")
        # The fitted stages are priced per whole split and ignore both the query
        # cap and the pilot's context arms, so reporting those numbers here would
        # describe a calculation that did not happen -- and a gate report is read
        # later, by someone checking whether the ceiling was set against the
        # right quantity.
        fitted = {
            "stage_d0b", "stage_d0c", "stage_d1", "stage_d2", "stage_d3",
            "stage_d4", "stage_d5", "stage_d6", "stage_d7", "stage_d8",
            "stage_d9", "stage_d10",
        }
        if {job["stage"] for job in jobs} <= fitted:
            queries = sorted({int(job["settings"]["expected_queries"]) for job in jobs})
            return units, (
                f"{len(jobs)} dataset(s) at {queries} protocol quer(ies), "
                f"whole splits; {granularity}"
            )
        caps = sorted({int(job["query_cap"]) for job in jobs})
        arms = sorted({len(job.get("arms") or module.CONFIG["arms"]) for job in jobs})
        return units, f"{len(jobs)} dataset(s) at {caps} quer(ies) x {arms} arm(s); {granularity}"

    if package == "m0a-probe":
        estimate = module.CONFIG["compute"]["estimate"]["job_seconds"]
        unknown = sorted({job["dataset"] for job in jobs} - set(estimate))
        if unknown:
            return None, f"no estimated job cost for {', '.join(unknown)}"
        # One dataset is one job that writes its result once at the end, so a
        # restart redoes the whole dataset. The declaration's per-job estimate
        # is therefore already the indivisible unit.
        units = [
            WorkUnit(name=job["dataset"], seconds=float(estimate[job["dataset"]]))
            for job in jobs
        ]
        units, granularity = _collapse_without_resumption(units, module, "dataset", "probe")
        return units, (
            f"{len(jobs)} dataset(s) at the declared pre-launch estimate, "
            f"which is a host timing and not a container measurement; {granularity}"
        )

    if package == "m0a1-overlap":
        estimate = module.CONFIG["compute"]["estimate"]["job_seconds"]
        unknown = sorted({job["dataset"] for job in jobs} - set(estimate))
        if unknown:
            return None, f"no estimated job cost for {', '.join(unknown)}"
        # One dataset is one job that writes its result once at the end, so a
        # restart redoes the whole dataset. The declaration's per-job estimate
        # is anchored to M0A execution 2's measured per-query latencies, not a
        # fresh host timing, and is already the indivisible unit.
        units = [
            WorkUnit(name=job["dataset"], seconds=float(estimate[job["dataset"]]))
            for job in jobs
        ]
        units, granularity = _collapse_without_resumption(units, module, "dataset", "overlap")
        return units, (
            f"{len(jobs)} dataset(s) at the declared pre-launch estimate, "
            f"anchored to M0A execution 2's measured per-query latencies; {granularity}"
        )

    if package == "m2c-stage0-probe":
        # Not the declaration's estimate but the filed compute record's, which
        # is the artifact the amendment made a precondition of this launch. The
        # record derives its per-job seconds from a host benchmark of the actual
        # kernel, M0A's measured expansion rates and this project's own pricing
        # functions, so gating against anything else would gate against a
        # second, softer number.
        estimate = {
            job["dataset"]: float(job["job_seconds"])
            for job in module.compute_record()["estimate"]["jobs"]
        }
        unknown = sorted({job["dataset"] for job in jobs} - set(estimate))
        if unknown:
            return None, f"the filed compute record prices no job for {', '.join(unknown)}"
        units = [WorkUnit(name=job["dataset"], seconds=estimate[job["dataset"]]) for job in jobs]
        units, granularity = _collapse_without_resumption(units, module, "cell", "probe")
        return units, (
            f"{len(jobs)} cell(s) at the filed Stage-0 compute record, which is a host "
            f"timing and not a container measurement; {granularity}"
        )

    if package == "m2-qls-v2-freeze":
        try:
            chains = _m2_chain_seconds(module)
        except (OSError, KeyError, ValueError) as error:
            return None, f"no readable M2 compute estimate ({error})"
        unknown = sorted({job["dataset"] for job in jobs} - set(chains))
        if unknown:
            return None, f"no estimated serial chain for {', '.join(unknown)}"
        # One dataset at one stage is one spawned call that writes its result
        # once at the end, so a restart redoes that whole call.
        units = [
            WorkUnit(name=job["dataset"], seconds=chains[job["dataset"]]) for job in jobs
        ]
        units, granularity = _collapse_without_resumption(units, module, "dataset", "screen")
        return units, (
            f"{len(jobs)} dataset(s) at the declared per-dataset serial chain -- feature "
            "build plus every new fit, so it overstates either stage submitted alone, and "
            "the spend figure prices the CPU build stage at the GPU rate for the same "
            f"reason; {granularity}"
        )

    return None, f"no measured cost model for {package}"


def gate_launch(package: str, module: Any, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """Refuse a launch whose largest indivisible unit exceeds the timeout.

    This is the check that was missing when the substrate audit was submitted
    with a six-hour ceiling: it billed a full window per attempt, restarted from
    zero each time, and produced nothing at all. Total cost above the ceiling is
    fine and common -- what cannot work is a unit larger than its window.
    """

    # The window the container will actually get. A launcher may resolve a
    # per-stage timeout at import time, and gating against the shared default
    # when the function was decorated with a larger one would refuse a launch
    # that fits -- the gate has to read the same number Modal will enforce.
    timeout = float(getattr(module, "TIMEOUT_SECONDS", module.MODAL_CONFIG["timeout_seconds"]))
    units, note = measured_units(package, module, jobs)
    if units is None:
        return {"gated": False, "why": note, "timeout_seconds": timeout}

    verdict = feasibility(units, timeout_seconds=timeout)
    shape = module.MODAL_CONFIG
    try:
        rate = container_rate_usd_per_hour(
            # The shape the module *resolved*, not the raw config key. A launcher
            # that serves both CPU and GPU stages from one entrypoint decides its
            # accelerator at import, the same way it decides its timeout, and
            # pricing every stage as a GPU stage would over-report a CPU job's
            # spend by threefold and make its declaration look breached.
            gpu=getattr(module, "GPU", shape.get("gpu")),
            cpu_cores=shape.get("cpu", 0),
            memory_mb=shape.get("memory_mb", 0),
        )
    except ValueError as error:
        # Reported as unknown, never as zero, and never as a refusal: what a run
        # costs is the operator's business, while whether it can finish is this
        # function's. Only the second is grounds for stopping a launch.
        rate = None
        rate_note = str(error)
    cap = shape.get("max_containers")
    report = {
        "gated": True,
        "basis": note,
        "timeout_seconds": timeout,
        "units": len(units),
        "largest_unit_hours": round(verdict.largest.seconds / 3600, 3) if verdict.largest else 0.0,
        "total_hours": round(verdict.total_seconds / 3600, 2),
        "container_usd_per_hour": round(rate, 3) if rate is not None else None,
        "max_burn_usd_per_hour": round(rate * cap, 2) if (rate is not None and cap) else None,
        "expected_spend_usd": round(
            expected_spend_usd(
                units,
                usd_per_container_hour=rate,
                training_fraction=UTILISATION.get(package, TRAINING_FRACTION_OF_BILLED_TIME),
            ),
            2,
        ) if rate is not None else None,
        "spend_unknown_because": None if rate is not None else rate_note,
        "verdict": verdict.reason,
    }
    if not verdict:
        raise SystemExit(
            f"REFUSED: {verdict.reason}\n"
            f"  package {package} | ceiling {timeout/3600:.1f} h | {note}"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", choices=sorted(PACKAGES))
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--stage", default="train")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--integrity-matrix",
        type=Path,
        default=None,
        help=(
            "matrix JSON from `migration_provenance.py matrix`; submits only the "
            "cells it reports as resume or launch, and refuses on an INVALID one"
        ),
    )
    args = parser.parse_args()

    module_name, stages = PACKAGES[args.package]
    if args.stage not in stages:
        raise SystemExit(f"{args.package} has no {args.stage!r} stage; pick {sorted(stages)}")
    module = importlib.import_module(module_name)

    requested = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = set(requested) - set(module.CONFIG["datasets"])
    if unknown:
        raise SystemExit(f"Unregistered {args.package} datasets: {sorted(unknown)}")

    # A launcher whose datasets are not all on one workspace declares which one
    # each runs on, and checks it here -- before the app is deployed, before
    # anything is spawned, and before a container opens the wrong volume.
    # Absent on every package whose datasets share a workspace, which is most.
    placement: dict[str, Any] | None = None
    check_placement = getattr(module, "check_execution_placement", None)
    if check_placement is not None:
        placement = check_placement(requested)

    jobs = _expand(module, args.package, args.stage, requested)
    plan: dict[str, Any] | None = None
    if args.integrity_matrix is not None:
        matrix = json.loads(args.integrity_matrix.read_text(encoding="utf-8"))
        jobs, plan = filter_by_matrix(jobs, matrix)
        if not jobs:
            print(json.dumps({"package": args.package, "spawned": 0, "plan": plan}, indent=2))
            return 0

    function = getattr(module, stages[args.stage])
    # Before anything is deployed or spawned, because the point is to refuse a
    # run that would bill a full window and produce nothing.
    budget = gate_launch(args.package, module, jobs)
    if args.dry_run:
        print(json.dumps({
            "package": args.package,
            "app": module.app.name,
            "function": stages[args.stage],
            "datasets": requested,
            "jobs": len(jobs),
            "placement": placement,
            "plan": plan,
            "budget": budget,
        }, indent=2))
        return 0

    deploy_app(module.app, name=module.app.name)
    handles = [function.spawn(job) for job in jobs]
    print(json.dumps({
        "package": args.package,
        "app": module.app.name,
        "function": stages[args.stage],
        "datasets": requested,
        "spawned": len(handles),
        "placement": placement,
        "plan": plan,
        "budget": budget,
        "call_ids": [handle.object_id for handle in handles],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
