"""M2D Stage 0: zero-training complementarity, fixed fusion, rescue analysis.

Pure arrays in, pure dictionaries out. Nothing here loads a checkpoint, opens a
dataset or touches a graph -- the runner does that and hands the results in --
so every rule below can be tested on inputs whose answer is known, off Modal
and in milliseconds. The alternative, burying these definitions in a container
script, is how a diagnostic ends up unfalsifiable: the only way to check it
would be to run the thing it is supposed to check.

Three things are computed, and all three are frozen by
``configs/m2d_s4_semantic_repair.yaml`` before any of them ran:

* **Complementarity** (section 4). Not "is S4 worse", which is already known,
  but "is S4 wrong where S3 is right AND right where S3 is wrong". A model that
  only loses is dominated and there is nothing to fuse; two models that trade
  errors disagree productively.
* **Fixed rank fusion** (section 5), at the RRF constant this project froze in
  ``configs/candidate_budget.yaml`` long before M2D existed. Reused, never
  chosen. No weight is searched, so each arm is falsifiable exactly as it
  stands.
* **Error-conditioned rescue** (section 6). On the queries S4 gets wrong, which
  existing signal already ranks a relevant candidate highly? That table is what
  a repair is chosen from, and section 6 forbids choosing one before it exists.

One convention is worth stating because it is a choice and not an inevitability.
A candidate in the scored pool but absent from a source's top-200 list is
UNRANKED by that source, and contributes exactly zero to a fusion -- which is
what ``linear_control.rank_feature_rows`` already does for the same two sources.
The alternative, giving it a rank just past the end of the list, would order the
unranked candidates among themselves by node id and quietly inject the tie-break
into the fusion.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .m2c_structural_offset import RRF_CONSTANT, rank_positions

#: The cutoffs every rung in this track reports at. Same tuple as the M2B
#: baseline table and the M2C probe; a fourth would make a row here
#: incomparable with the numbers the declaration froze.
KS = (1, 5, 20)

#: Section 6's cutoffs for the rescue table. The same three, deliberately:
#: rank 1 is where the blockers are worst and rank 20 is where S4's passage
#: deltas are nearly flat, so the spread across them is itself the finding.
RESCUE_CUTOFFS = (1, 5, 20)

__all__ = [
    "KS",
    "RESCUE_CUTOFFS",
    "RRF_CONSTANT",
    "complementarity",
    "full_coverage_contribution",
    "fuse_ranks",
    "mean_metrics",
    "metrics",
    "rescue_row",
    "rescue_table",
    "source_contribution",
    "unranked_last",
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metrics(ranks: np.ndarray, relevant: np.ndarray) -> dict[str, float]:
    """recall@k and MRR for one query, from ranks over its scored pool.

    Identical in form to the M2C probe's, deliberately: a second definition of
    recall in the same track would make these rows incomparable with the ones
    the declaration froze, and the difference would be invisible in the output.
    A test holds the two together.
    """

    ranks = np.asarray(ranks).reshape(-1)
    relevant = np.asarray(relevant, dtype=bool).reshape(-1)
    if ranks.shape != relevant.shape:
        raise ValueError(f"ranks {ranks.shape} and relevance {relevant.shape} differ")
    if not relevant.any():
        return {**{f"recall@{k}": 0.0 for k in KS}, "mrr": 0.0, "scored": 0.0}
    relevant_ranks = ranks[relevant]
    best = int(relevant_ranks.min())
    total = int(relevant.sum())
    return {
        **{f"recall@{k}": float((relevant_ranks <= k).sum()) / total for k in KS},
        "mrr": 1.0 / best,
        "scored": 1.0,
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {**{f"recall@{k}": 0.0 for k in KS}, "mrr": 0.0, "queries": 0}
    return {
        **{f"recall@{k}": float(np.mean([row[f"recall@{k}"] for row in rows])) for k in KS},
        "mrr": float(np.mean([row["mrr"] for row in rows])),
        "queries": len(rows),
    }


# ---------------------------------------------------------------------------
# Turning a stored top-N list into something rankable over the pool
# ---------------------------------------------------------------------------


def source_contribution(
    pool: np.ndarray, ranked_ids: np.ndarray, *, constant: int = RRF_CONSTANT
) -> np.ndarray:
    """``1 / (constant + rank)`` for pool members this source ranked, else 0.

    ``ranked_ids`` is the source's own ordered list -- Dense's or SPLADE's
    stored top-200 -- so a candidate's rank is its position in THAT list, not
    its position among the pool. A pool member the source never returned gets
    zero, the same treatment ``linear_control.rank_feature_rows`` gives it.
    """

    pool = np.asarray(pool).reshape(-1)
    ranked_ids = np.asarray(ranked_ids).reshape(-1)
    if constant < 0:
        raise ValueError("the RRF constant must be non-negative")
    if np.unique(ranked_ids).size != ranked_ids.size:
        raise ValueError("a source ranking must not repeat a candidate")

    position = {int(node): index for index, node in enumerate(ranked_ids.tolist())}
    out = np.zeros(pool.size, dtype=np.float64)
    for index, node in enumerate(pool.tolist()):
        rank = position.get(int(node))
        if rank is not None:
            out[index] = 1.0 / (float(constant) + rank + 1.0)
    return out


def unranked_last(contribution: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Rank a source over the pool, with everything it never returned last.

    Used when a retrieval source is treated as a ranker in its own right rather
    than as a fusion input. Candidates the source did not return all share a
    contribution of zero and are ordered among themselves by node id, which is
    the frozen tie-break -- acceptable HERE because they are genuinely tied for
    that source, and not acceptable inside a fusion, where the tie would be
    laundered into the fused score.
    """

    return rank_positions(np.asarray(contribution, dtype=np.float64), pool)


def fuse_ranks(contributions: list[np.ndarray], pool: np.ndarray) -> np.ndarray:
    """Equal-weight RRF over per-source contributions, then rank the sum.

    Equal weights and the inherited constant: nothing is fitted, so the arm can
    fail. Summing contributions rather than ranks is what lets a source that
    covers only part of the pool take part without its absences being scored.
    """

    if not contributions:
        raise ValueError("at least one source is required")
    pool = np.asarray(pool).reshape(-1)
    fused = np.zeros(pool.size, dtype=np.float64)
    for contribution in contributions:
        contribution = np.asarray(contribution, dtype=np.float64).reshape(-1)
        if contribution.shape != fused.shape:
            raise ValueError("every source must cover the same candidate set")
        if np.any(contribution < 0.0):
            raise ValueError("a reciprocal-rank contribution cannot be negative")
        fused += contribution
    return rank_positions(fused, pool)


def full_coverage_contribution(
    ranks: np.ndarray, *, constant: int = RRF_CONSTANT
) -> np.ndarray:
    """``1 / (constant + rank)`` for a ranker that covers the whole pool.

    S4 and S3 score every candidate, so they have no absences and their
    contribution is a pure function of their ranks.
    """

    ranks = np.asarray(ranks, dtype=np.float64).reshape(-1)
    if np.any(ranks < 1):
        raise ValueError("ranks are 1-based")
    return 1.0 / (float(constant) + ranks)


# ---------------------------------------------------------------------------
# Section 4: complementarity
# ---------------------------------------------------------------------------


def _top1_node(ranks: np.ndarray, pool: np.ndarray) -> int:
    return int(pool[int(np.argmin(ranks))])


def _best_relevant(ranks: np.ndarray, relevant: np.ndarray, pool: np.ndarray) -> tuple[int, int]:
    """The highest-ranked relevant candidate: its node id and its rank."""

    relevant_positions = np.flatnonzero(relevant)
    best = relevant_positions[int(np.argmin(ranks[relevant_positions]))]
    return int(pool[best]), int(ranks[best])


def complementarity(
    s4_ranks: np.ndarray,
    s3_ranks: np.ndarray,
    relevant: np.ndarray,
    pool: np.ndarray,
) -> dict[str, Any]:
    """One query's contribution to the section 4 counts.

    Returns ``None`` for the cross-rank fields when the query has no relevant
    candidate in the pool: there is no "top relevant candidate" to locate under
    the other model, and a zero there would be a measurement rather than an
    absence. The caller counts those separately.
    """

    s4_ranks = np.asarray(s4_ranks).reshape(-1)
    s3_ranks = np.asarray(s3_ranks).reshape(-1)
    relevant = np.asarray(relevant, dtype=bool).reshape(-1)
    pool = np.asarray(pool).reshape(-1)
    if not (s4_ranks.shape == s3_ranks.shape == relevant.shape == pool.shape):
        raise ValueError("every input must describe the same candidate pool")

    s4_top1 = _top1_node(s4_ranks, pool)
    s3_top1 = _top1_node(s3_ranks, pool)
    s4_right = bool(relevant[int(np.argmin(s4_ranks))])
    s3_right = bool(relevant[int(np.argmin(s3_ranks))])

    row: dict[str, Any] = {
        "s4_right_at_1": s4_right,
        "s3_right_at_1": s3_right,
        "s4_wrong_s3_right": (not s4_right) and s3_right,
        "s3_wrong_s4_right": (not s3_right) and s4_right,
        "both_right_at_1": s4_right and s3_right,
        "neither_right_at_1": (not s4_right) and (not s3_right),
        "same_top1_node": s4_top1 == s3_top1,
        "has_relevant_in_pool": bool(relevant.any()),
        "overlap": {
            k: int(np.intersect1d(pool[s4_ranks <= k], pool[s3_ranks <= k]).size)
            for k in KS
        },
    }

    if not relevant.any():
        row.update(
            {
                "same_best_relevant_node": None,
                "s3_best_relevant_rank_under_s4": None,
                "s4_best_relevant_rank_under_s3": None,
                "both_found_same_relevant_ordered_differently": None,
            }
        )
        return row

    s4_best_node, s4_best_rank = _best_relevant(s4_ranks, relevant, pool)
    s3_best_node, s3_best_rank = _best_relevant(s3_ranks, relevant, pool)
    position = {int(node): index for index, node in enumerate(pool.tolist())}
    row.update(
        {
            "same_best_relevant_node": s4_best_node == s3_best_node,
            # "rank of S3's top relevant candidate under S4, and the reverse":
            # where the OTHER model put the candidate this model found first.
            "s3_best_relevant_rank_under_s4": int(s4_ranks[position[s3_best_node]]),
            "s4_best_relevant_rank_under_s3": int(s3_ranks[position[s4_best_node]]),
            "both_found_same_relevant_ordered_differently": bool(
                s4_best_node == s3_best_node and s4_best_rank != s3_best_rank
            ),
        }
    )
    return row


# ---------------------------------------------------------------------------
# Section 6: error-conditioned rescue
# ---------------------------------------------------------------------------


def rescue_row(
    s4_ranks: np.ndarray,
    others: dict[str, np.ndarray],
    relevant: np.ndarray,
    *,
    cutoffs: tuple[int, ...] = RESCUE_CUTOFFS,
) -> dict[str, Any] | None:
    """Who already ranks a relevant candidate at k, on a query S4 gets wrong.

    ``None`` means the query is not in section 6's population, and the two
    reasons are distinguished by the caller because they mean opposite things:
    a query whose gold was never a candidate is an ADMISSION failure and is not
    evidence about ranking at all, while a query S4 already gets right is not
    an error.

    A query can have several rescuers, so the classification is not a partition
    -- the counts are per rescuer, plus an "any" and an exclusive "none".
    Forcing a single label would need a precedence order nobody declared.
    """

    s4_ranks = np.asarray(s4_ranks).reshape(-1)
    relevant = np.asarray(relevant, dtype=bool).reshape(-1)
    if not relevant.any():
        return None
    if bool(relevant[int(np.argmin(s4_ranks))]):
        return None

    best_relevant_rank = {
        name: int(np.asarray(ranks).reshape(-1)[relevant].min()) for name, ranks in others.items()
    }
    row: dict[str, Any] = {
        "s4_best_relevant_rank": int(s4_ranks[relevant].min()),
        "best_relevant_rank": best_relevant_rank,
        "rescued_at": {},
    }
    for cutoff in cutoffs:
        rescuers = sorted(
            name for name, rank in best_relevant_rank.items() if rank <= cutoff
        )
        row["rescued_at"][cutoff] = {
            "rescuers": rescuers,
            "any": bool(rescuers),
            "none": not rescuers,
        }
    return row


def rescue_table(
    rows: list[dict[str, Any]],
    *,
    names: tuple[str, ...],
    excluded: dict[str, int],
    cutoffs: tuple[int, ...] = RESCUE_CUTOFFS,
) -> dict[str, Any]:
    """Aggregate rescue rows, with the measurability fraction attached.

    Section 0 of the declaration requires that every conditioned statistic
    travel with the fraction of the population it was measurable on, because
    M2C's margins looked clean on 6.5% of the errors. Here the population is
    fully measurable by construction -- all four systems rank the entire pool --
    and saying so, with the excluded counts beside it, is the honest form of
    that requirement rather than an exemption from it.
    """

    population = len(rows)
    total_errors = population + int(excluded.get("no_relevant_in_pool", 0))
    table: dict[str, Any] = {
        "population": population,
        "excluded": dict(excluded),
        "measurable": population,
        "fraction_of_the_error_population_measurable": (
            1.0 if population else 0.0
        ),
        "why_fully_measurable": (
            "every system ranks the whole scored pool, so a relevant candidate that "
            "is in the pool has a rank under all of them. The unmeasurable case here "
            "is admission, not ranking, and it is excluded and counted."
        ),
        "queries_whose_gold_was_never_a_candidate": int(
            excluded.get("no_relevant_in_pool", 0)
        ),
        "share_of_all_top1_failures_that_are_admission_failures": (
            float(excluded.get("no_relevant_in_pool", 0)) / total_errors
            if total_errors
            else 0.0
        ),
        "by_cutoff": {},
    }
    for cutoff in cutoffs:
        counts = {name: 0 for name in names}
        any_rescued = 0
        for row in rows:
            rescuers = row["rescued_at"][cutoff]["rescuers"]
            for name in rescuers:
                counts[name] += 1
            any_rescued += int(bool(rescuers))
        table["by_cutoff"][cutoff] = {
            "rescued_by": counts,
            "rescued_by_any": any_rescued,
            "rescued_by_none": population - any_rescued,
            "share_rescued_by": {
                name: (count / population if population else 0.0)
                for name, count in counts.items()
            },
            "share_rescued_by_none": (
                (population - any_rescued) / population if population else 0.0
            ),
        }
    return table
