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
#
# Both rates were then divided by the speedup of the mat-vec traversal and the
# incremental expansion walk -- 3.8x and 3.6x, measured at this graph's shape
# against the pre-patch functions loaded from git, values identical. The
# container-measured anchors are kept and scaled rather than replaced by the
# workstation's absolute timings, which are not the container's.
# The rates as first measured, kept named rather than inlined: several tests
# describe the launch that was refused on 2026-09-02, and making the audit
# faster afterwards must not retroactively make that launch look feasible.
HISTORICAL_SECONDS_PER_QUERY = 0.411
HISTORICAL_EXPANSION_SECONDS_PER_QUERY = 7.546

SUBSTRATE_SECONDS_PER_QUERY = HISTORICAL_SECONDS_PER_QUERY / 3.8
SUBSTRATE_EXPANSION_SECONDS_PER_QUERY = HISTORICAL_EXPANSION_SECONDS_PER_QUERY / 3.6

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

# Modal bills the accelerator, the cores and the memory as three separate lines,
# and the phase-confirmation invoice decomposes cleanly enough to recover all
# three rates. That run held an A10G with 16 cores and 48 GiB and was billed
# $25.50: A10G $12.52, CPU $8.61, memory $4.37. Dividing the CPU and memory
# lines by the A10G line and by the quantities held gives the ratios below;
# anchoring them on the A10G hourly rate gives absolute figures.
#
# The distinction matters because a single blended "container hour" is wrong by
# a factor of five in the direction that discourages cheap work: the substrate
# audit holds no GPU at all, and costing it at the A10G rate reports $39 for a
# job that bills about $8.
USD_PER_A10G_HOUR = 1.10
USD_PER_CPU_CORE_HOUR = 0.0473
USD_PER_GIB_HOUR = 0.0080

# Fraction of billed time a phase-confirmation container spends training. The
# rest is image pull, data loading and feature building, all billed at the same
# rate with the accelerator attached and idle. Quoting training seconds alone is
# how a $25 job gets described as $10.
TRAINING_FRACTION_OF_BILLED_TIME = 0.40


def container_rate_usd_per_hour(
    *, gpu: str | None = None, cpu_cores: float = 0.0, memory_mb: float = 0.0
) -> float:
    """Hourly cost of one container of a given shape.

    Only the A10G is priced, because it is the only accelerator this project has
    ever been billed for and it is the only one an invoice pins. An unrecognised
    GPU raises rather than silently costing zero, which would understate a spend
    in exactly the situation where the number is most wanted.
    """
    if cpu_cores < 0 or memory_mb < 0:
        raise ValueError("container shape cannot be negative")
    if not gpu and not cpu_cores and not memory_mb:
        # Every Modal container holds cores and memory, so an empty shape means
        # the caller could not find one -- not that the job is free. Returning
        # 0.0 here would put a confident "$0.00" in a launch record, which is a
        # worse answer than admitting the rate is unknown.
        raise ValueError("no container shape given; cannot price a container that holds nothing")
    rate = cpu_cores * USD_PER_CPU_CORE_HOUR + (memory_mb / 1024.0) * USD_PER_GIB_HOUR
    if gpu:
        if gpu.upper() != "A10G":
            raise ValueError(f"no measured hourly rate for accelerator {gpu!r}")
        rate += USD_PER_A10G_HOUR
    return rate


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
    usd_per_container_hour: float | None = None,
    training_fraction: float = TRAINING_FRACTION_OF_BILLED_TIME,
) -> float:
    """Billed cost of the work, inflated for time the container is not working.

    ``usd_per_container_hour`` should come from `container_rate_usd_per_hour`
    for the shape the package actually runs on. It defaults to the A10G
    phase-confirmation shape only because that is the run an invoice pins;
    passing a CPU-only shape's rate is what keeps a GPU-less job from being
    quoted at five times its cost.

    ``training_fraction`` is how much of the billed time does the measured work.
    A job whose measurement already covers its whole runtime -- the substrate
    audit's per-query cost includes its loading -- should pass 1.0 rather than
    be inflated twice.
    """
    if not 0 < training_fraction <= 1:
        raise ValueError("training fraction must lie in (0, 1]")
    if usd_per_container_hour is None:
        usd_per_container_hour = container_rate_usd_per_hour(
            gpu="A10G", cpu_cores=16, memory_mb=49152
        )
    if usd_per_container_hour < 0:
        raise ValueError("container rate cannot be negative")
    billed_hours = sum(u.seconds for u in units) / 3600 / training_fraction
    return billed_hours * usd_per_container_hour
