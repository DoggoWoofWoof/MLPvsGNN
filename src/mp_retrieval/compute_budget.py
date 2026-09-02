"""Refuse to spend compute on work that provably cannot finish.

Two runs were lost before this existed. The second is the instructive one: the
substrate audit was launched with a six-hour ceiling against four graph families
costing 3.31 h each. Every individual family fits. The job does not, and because
nothing was carried across a restart, each attempt would recompute from zero and
die at the same place -- a loop that bills indefinitely and produces nothing.

That is the rule this module encodes, and it is not "estimate < timeout":

    a run makes progress only if the timeout exceeds its largest INDIVISIBLE
    unit -- the piece of work a restart must redo from the beginning.

Total cost may exceed the ceiling as much as it likes, provided completed units
survive a restart. What must never happen is a unit larger than the window it
runs in, because then no amount of retrying advances anything.

The constants are measured on the staged data, not guessed, and each carries the
date and machine it came from. A model whose numbers are invented is worse than
no model: it produces a confident verdict with nothing behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Measured 2026-09-02 on the staged hotpotqa graph (507,494 nodes, 4,559,172
# undirected edges) after the filter-before-dedupe traversal fix, on this
# workstation. Modal's 8-CPU containers are not identical hardware, so a safety
# factor is applied at the call site rather than baked in here.
SUBSTRATE_SECONDS_PER_QUERY = 0.411
SUBSTRATE_EXPANSION_SECONDS_PER_QUERY = 7.546

# Per-seed training seconds, from the fetched phase-confirmation results. The
# spread is three orders of magnitude, which is why a single average would be
# useless for deciding whether a cell fits a window.
PHASE_CONFIRMATION_SECONDS_PER_SEED = {
    "metaqa": 396.0,
    "hotpotqa_clean": 320.0,
    "musique_clean": 40.0,
    "2wiki_clean": 31.0,
    "squad_clean": 22.0,
    "webqsp": 3.0,
}

# Observed on darkphoenix: $25.50 over ~22 A10G container-hours, of which only
# ~40% was training. Used for reporting an expected spend, never for a refusal --
# a budget is the operator's call, feasibility is not.
USD_PER_A10G_CONTAINER_HOUR = 1.18
TRAINING_FRACTION_OF_BILLED_TIME = 0.40


@dataclass(frozen=True)
class WorkUnit:
    """One indivisible piece of work.

    ``seconds`` is what a restart must redo from the start, so it is the unit
    that has to fit the timeout -- not the whole job, and not an average.
    """

    name: str
    seconds: float

    def __post_init__(self) -> None:
        if self.seconds < 0:
            raise ValueError(f"work unit {self.name!r} has negative duration")


@dataclass(frozen=True)
class Verdict:
    feasible: bool
    reason: str
    largest: WorkUnit | None = None
    total_seconds: float = 0.0
    oversized: tuple[WorkUnit, ...] = field(default_factory=tuple)

    def __bool__(self) -> bool:
        return self.feasible


def feasibility(units: list[WorkUnit], timeout_seconds: float, *, safety: float = 1.5) -> Verdict:
    """Whether a run can make progress, given what a restart has to redo.

    ``safety`` scales the measured unit cost before comparing. The constants come
    from one machine and the containers are not that machine, so a unit that only
    just fits is treated as not fitting: being wrong in that direction costs a
    whole billed window and yields nothing.
    """
    if timeout_seconds <= 0:
        return Verdict(False, "timeout must be positive")
    if not units:
        return Verdict(True, "nothing to do", total_seconds=0.0)
    if safety < 1.0:
        raise ValueError("safety factor below 1.0 would understate measured cost")

    total = sum(unit.seconds for unit in units)
    largest = max(units, key=lambda unit: unit.seconds)
    oversized = tuple(u for u in units if u.seconds * safety > timeout_seconds)

    if oversized:
        worst = max(oversized, key=lambda unit: unit.seconds)
        return Verdict(
            False,
            f"{len(oversized)} indivisible unit(s) exceed the {timeout_seconds/3600:.1f} h "
            f"ceiling at {safety:g}x safety; largest is {worst.name} at "
            f"{worst.seconds/3600:.2f} h. Raise the timeout or split the unit -- "
            f"retrying cannot help, because each attempt restarts it from zero.",
            largest=largest,
            total_seconds=total,
            oversized=oversized,
        )

    note = "fits in one window" if total * safety <= timeout_seconds else (
        f"total {total/3600:.2f} h exceeds the window, but the largest unit "
        f"({largest.name}, {largest.seconds/3600:.2f} h) fits, so completed units "
        f"survive a restart and the run advances"
    )
    return Verdict(True, note, largest=largest, total_seconds=total)


def substrate_family_units(
    *,
    queries: int,
    families: list[str],
    expansion_cap: int,
    seconds_per_query: float = SUBSTRATE_SECONDS_PER_QUERY,
    expansion_seconds_per_query: float = SUBSTRATE_EXPANSION_SECONDS_PER_QUERY,
) -> list[WorkUnit]:
    """One unit per graph family, because resumption is family-granular.

    The expansion measurement is capped, so it does not scale with the split:
    every query pays ``seconds_per_query`` and only the first ``expansion_cap``
    pay the expansion term. Treating expansion as uncapped overstates a family by
    more than an order of magnitude and would refuse a run that is fine.
    """
    if queries < 0 or expansion_cap < 0:
        raise ValueError("query counts cannot be negative")
    per_family = queries * seconds_per_query + min(queries, expansion_cap) * expansion_seconds_per_query
    return [WorkUnit(f"family:{name}", per_family) for name in families]


def phase_confirmation_units(
    cells: list[tuple[str, int]],
    *,
    seconds_per_seed: dict[str, float] | None = None,
) -> list[WorkUnit]:
    """One unit per *seed*, because a cell checkpoints after each seed.

    ``cells`` is ``(dataset, seeds_remaining)``. A partly-trained cell resumes at
    its seed boundary, so the thing that must fit the window is a single seed,
    not the whole cell.
    """
    table = seconds_per_seed or PHASE_CONFIRMATION_SECONDS_PER_SEED
    units: list[WorkUnit] = []
    for dataset, remaining in cells:
        if dataset not in table:
            raise KeyError(f"no measured per-seed cost for {dataset!r}")
        for index in range(remaining):
            units.append(WorkUnit(f"{dataset}:seed{index}", table[dataset]))
    return units


def expected_spend_usd(
    units: list[WorkUnit],
    *,
    usd_per_container_hour: float = USD_PER_A10G_CONTAINER_HOUR,
    training_fraction: float = TRAINING_FRACTION_OF_BILLED_TIME,
) -> float:
    """Billed cost of the work, inflated for the time the GPU is not training.

    Measured training seconds understate the bill: data loading and feature
    building are billed at the same rate with the accelerator attached and idle.
    Reporting the training figure alone is how a $30 job gets described as $12.
    """
    if not 0 < training_fraction <= 1:
        raise ValueError("training fraction must lie in (0, 1]")
    billed_hours = sum(u.seconds for u in units) / 3600 / training_fraction
    return billed_hours * usd_per_container_hour
