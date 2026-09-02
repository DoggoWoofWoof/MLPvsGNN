# Compute ledger

One row per experiment. Pilot cost, projected cost, actual cost, and the reason
it advanced or was killed. Opened 2026-09-02, after roughly $28 was spent on two
runs that a 100-query pilot would have rejected.

The governing rule:

> **Expand compute only when the current scale has produced evidence that a
> larger run can change a scientific decision.**

Not "the experiment was planned", not "the code already exists", not "72 cells
are already done".

## The ladder

Every experiment climbs this. A candidate advances only if the stage below it
justifies spending more.

```text
A  implementation validation   unit / formula / equivalence / synthetic edge cases
B  microscopic smoke           10-50 real queries, exact outputs and invariants
C  runtime + signal pilot      100-500 real validation queries; correctness,
                               metric direction, p50/p95/p99, memory, full-cost estimate
D  minimal scientific pilot    1 dataset x 1 seed
E  transfer pilot              2 representative datasets x 1 seed
F  robustness pilot            promising candidates only, 2-3 seeds
G  full validation             non-dominated survivors only
H  freeze
I  final confirmation          5 seeds, once
```

Development uses **one seed** unless variance is itself the question. Five seeds
are for confirmation, never exploration. No Cartesian sweep of
`S0-S4 x R0-R5b x datasets x seeds` -- add one feature family, pilot it, keep or
kill it, and only survivors continue.

A feature advances only if it delivers a defensible benefit: a meaningful
validation gain, the same effectiveness at lower cost, clearer generalisation,
or it replaces something more expensive. Strictly Pareto-dominated means killed
immediately, with the result kept in this ledger and no further evaluation.

Structural primitives are benchmarked for p50/p95/p99, RSS, storage and
asymptotic work at 100-500 queries **before** any model-scale evaluation. A
feature that cannot be served is not a feature.

## Pre-launch declaration

No launch without all of these stated first. If they cannot be stated, there is
no launch.

```text
scientific question      number of jobs           estimated wall time
exact hypothesis         estimated GPU-hours      estimated storage
datasets / queries       estimated CPU-hours      stopping rule
models / seeds                                    advancement criterion
```

## Rows

| experiment | pilot | projected | actual | outcome |
|---|---|---|---|---|
| Packages A, B, C, D, E1 | none | — | — | COMPLETE and frozen before this policy |
| E2 phase-confirmation | none | 960 seed-units | 891 seed-units, ~$25.50 + ~$27 relaunch | **PAUSED at 82/96.** Superseded in priority; locked to v1's features, architecture and an unvalidated graph substrate. 69 seed-units / ~12 GPU-h / ~$27 to resume. |
| Phase -1 substrate audit (first launch) | none | assumed to fit 6 h | ~$8, zero families written | **KILLED.** 6 h ceiling below one 3.31 h family with no resumption; every attempt restarted from zero. Cause of the feasibility gate. |
| Phase -1 traversal optimisation | profile at true shape, 25 queries | — | 0 GPU, local | **ADVANCED.** 74% of runtime in one call; 3.6x overall, values identical vs the pre-patch code at full scale. Frozen as `graph-substrate-implementation-v2`. |
| Phase -1 substrate audit (optimised) | the profile above | 3.54 CPU-h, $2.25 | in flight | Gate declared before launch. Decision rests on CORE only. |

## What each earlier failure would have cost to catch

Every one of these was findable below the scale it was found at, which is the
whole argument for the ladder.

| defect | found at | findable at |
|---|---|---|
| `uint8` accumulator wrapping at 256 accumulated frontier edges into one node | full-scale benchmark | Stage A: 4 nodes with parallel edges, or 256 distinct sources into one target |
| `result.json` presence read as completion | after reporting 86/96 to the user, twice | Stage B, one cell mid-run |
| substrate audit could not finish in its ceiling | ~$8 and zero output | Stage C, a 100-query timing pilot |
| global BFS as 74% of runtime | 3.31 h/family in production | Stage C, the same pilot |
