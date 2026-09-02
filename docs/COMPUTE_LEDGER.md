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

### QLS-v2 P1 -- which restored global context, and at what radius?

Pre-launch declaration, Stages A-C only. Filed 2026-09-02, amended the same day
after the interpretation checks below. The first version of this row froze H=2
on the target side; the audit's own node counts show that context is 93% of the
hotpotqa graph, so the arm set and the budget both changed.

**Cost screen, settled for $0 from Phase -1 data.** Context nodes per query and
worst-case share of the corpus graph:

```text
arm                      nodes/query (min-max)     worst share of graph
CAND        G[Cq]              316 -    361                --
SEED-H1     Cq u N1(Sq)        394 -    580               2.9%
TARGET-H1   Cq u N1(Cq)       1994 -   7160              29.8%
SEED-H2     Cq u N2(Sq)       2147 - 329095              64.8%
SEED-H3     Cq u N3(Sq)       7872 - 505742              99.7%   killed
TARGET-H2   Cq u N2(Cq)       9102 - 472846              93.2%   killed
TARGET-H3   Cq u N3(Cq)      12842 - 607121             100.0%   killed
```

The three killed arms are not context; they are the corpus. They fail the
ladder's serving criterion ("a feature that cannot be served is not a feature")
on already-measured numbers, so no pilot compute is spent on them. If a later
result makes a corpus-scale context worth revisiting, it needs its own row.

```text
scientific question   Which restored global context, at which radius, recovers
                      the structure candidate induction deletes -- and does the
                      recovery land on the candidates that lost it?

exact hypothesis      TARGET-H1 moves features most, because it is the arm that
                      directly repairs the measured radius-1 deletion (median
                      rho_1 0.077-0.250). SEED-H1 moves them least, since
                      Sq is a subset of Cq and it adds only 35-235 nodes.
                      SEED-H2 moves seed-relative features on the datasets
                      where induction destroyed the most 2-hop reach (2wiki
                      13.6%, hotpotqa 27.5% retained) -- but that is also where
                      it is 60-65% of the graph, so it is expected to advance
                      on information and fail on cost.

datasets / queries    A: synthetic only.  B: 25 real queries, hotpotqa + 2wiki.
                      C: 300 validation queries x 6 datasets x 4 arms.
models / seeds        none -- feature-only, no training.
number of jobs        A,B local. C: 6 CPU jobs, substrate container shape.
estimated GPU-hours   0
estimated CPU-hours   <= 2.5   (raised from 1.0: four arms, and SEED-H2 on
                                hotpotqa touches 329k nodes/query)
estimated wall time   <= 40 min, jobs concurrent
estimated storage     < 200 MB
estimated cost        <= $2.00 at the $0.634/h substrate shape

measured per arm      feature movement vs QLS-v1 on G[Cq]
                      isolate / constant-feature reduction
                      fraction of low-degree candidates whose features changed
                      context nodes/query, context edges/query
                      p50 / p95 / p99 feature-build latency
                      peak RSS and structural workspace

                      feature movement broken down by the induced degree the
                      candidate had before: formerly isolated / low / normal

stopping rule         Kill an arm if the isolated fraction does not fall, or if
                      features do not move on the multi-hop datasets. Either
                      means the arm carries no information G[Cq] lacked, and no
                      training run can recover it.

advancement criterion An arm advances to D only if movement is concentrated on
                      the candidates Phase -1 identified as truncated --
                      formerly isolated or low induced degree -- AND its
                      projected serving cost is plausible. Movement spread
                      evenly over already-well-connected candidates is noise.
                      If a larger radius adds substantial information at modest
                      incremental cost, both radii carry to D; if it mostly
                      adds context and cost, it dies here.
```

Constraint carried from the frozen protocol: no GNN anywhere in QLS-v2, no GNN
teacher, no GNN-derived feature selection. This pilot compares QLS features to
QLS features. Frozen candidate pools are untouched -- the scored set is exactly
`Cq` in every arm, and context nodes are never admitted to it.

No training until this resolves. E2 stays paused, F stays sealed, no workspace
migration.
