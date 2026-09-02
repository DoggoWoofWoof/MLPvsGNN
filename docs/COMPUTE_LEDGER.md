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

#### Second amendment, filed 2026-09-02 before Stage B launched

Three changes. The declaration above is left standing rather than rewritten, so
what was believed at filing time and what replaced it are both readable.

**The arm set gained two enclosing arms.** The four above are radius rules; a
graph-retrieval context in the literature is more often a *path* set, and the
declaration had no arm of that kind. `PATH_H2` admits a node lying on a
two-step seed-to-candidate path, `BRIDGE_H2` a node lying on a two-step
candidate-to-candidate path, both requiring two DISTINCT candidate endpoints.
Neither uses a gold id, an answer label, a relevance label, any test
information, or any GNN signal. All-pairs is not implemented and is not
proposed. Measured on the real 2wiki CSR with hub pools, they cost:

```text
arm                       nodes/query (median)     share of graph
PATH_H2     seed-anchored               156               0.24%
BRIDGE_H2   candidate-anchored        6,818              10.35%
```

`PATH_H2` is essentially free and `BRIDGE_H2` buys 0.914 median retention for
a tenth of the graph, against `TARGET_H1`'s 1.000 for six tenths. That is the
comparison the frontier exists to make, so both carry into Stage B.

**The primary measure is withdrawn, before Stage B rather than after.** The
declaration made "feature movement vs QLS-v1 on G[Cq]" the signal, and its
stopping rule kills an arm whose features do not move. Both are vacuous above
`SEED_H1`, provably and on any graph. Columns 0-3 of the frozen QLS-v1
descriptor are one-hot over {0, 1, 2, >=3-or-unreachable}; reaching bucket 2
requires a path `seed -> x -> candidate` with a single intermediate, and every
such `x` is an out-neighbour of a seed, which is exactly what `SEED_H1` admits.
The raw seed-incidence column is pinned for the same reason -- `Sq` is inside
`Cq`, so every seed-candidate edge already lies in `G[Cq]`. The remaining six
columns are each divided by a per-query maximum over the whole local node
space, so they move under a wider context whether or not a candidate's own
topology did.

Verified over 480 arm comparisons on 120 random graphs, symmetric and directed:
zero bucket movements and zero raw-count changes above `SEED_H1`, with the
normalised seed-incidence column differing on 17 of them, which is the
rescaling confound and nothing else. On the real 2wiki CSR all six arms
returned byte-identical buckets, `SEED_H1` at 166 context nodes against
`TARGET_H1` at 40,653.

The replacement is seed distance recomputed through the arm's own context,
uncapped to four hops and divided by nothing, with `distance_improved` -- the
fraction of candidates strictly closer to a seed than `G[Cq]` made them -- as
primary, stratified by prior induced degree exactly as the original measure
was. The lattice makes it one-sided: a wider context can only bring a candidate
closer. On the same 2wiki pools it separates arms the bucket collapses to a
constant 0.0129 across, and it lands where Phase -1 found the loss --
`TARGET_H1` improves 26.4% of degree-1 candidates against 3.5% of ordinary
ones. Both descriptor measures are still computed and still reported.

This re-specifies a measurement before it is taken. No arm, no threshold, no
dataset and no spend moves because of it.

**The budget is restated on a measured ceiling.** Stage B stays at the filed
scope -- 25 validation queries, 2wiki and hotpotqa, the two datasets where
induction destroyed the most two-hop reach -- now at six arms rather than four.

```text
                        declared          now
stage B datasets        2wiki, hotpot     unchanged
stage B queries         25                unchanged
arms                    4                 6
CPU-hours (ceiling)     <= 2.5            <= 0.9   stage B
cost (ceiling)          <= $2.00          <= $1.41 stage B
GPU-hours               0                 unchanged
```

The per-query figure behind it is a ceiling, not an estimate. Context
construction is 0.4-232 ms per query per arm at true graph scale and the
seed-distance probe 22-73 ms p95, both negligible; the QLS kernel is the only
unmeasured part, because the Modal image compiles it with Numba and the local
pure-Python fallback does not represent it. So the kernel is costed at the one
figure that is measured -- that same local fallback, 38 s per query for all six
arms on 2wiki -- rounded to 40. Any Numba speedup at all makes the real number
smaller, and 25 queries fit the declared 3600 s timeout even with none. Stage B
exists in part to replace it with a measured throughput, which is what gates
Stage C's own cost.

Stage C is not re-priced here. Its cost depends on which arms survive B and on
B's measured throughput, and pricing it before either would be inventing a
number. It gets its own line when B returns.

E2 stays paused, F stays sealed, no workspace migration, nothing trained.

#### Stage C declaration, filed 2026-09-02 after B returned

Stage B ran as declared: 25 validation queries on 2wiki and hotpotqa, six arms,
no query skipped for want of a retrieval seed. Both jobs returned, in 33 s and
278 s.

**Measured throughput, which is what B was for.** Total per query per arm, p95,
on the Modal image with the Numba kernel:

```text
arm           2wiki      hotpotqa      verdict at B
CAND           5.3 ms      84.2 ms     control, on the frontier only by being free
SEED_H1        4.8 ms     173.5 ms     survives on a half-millisecond margin
TARGET_H1      7.9 ms     221.8 ms     survives
PATH_H2       12.7 ms     302.2 ms     killed -- a subset of SEED_H1 costing 6x more
BRIDGE_H2      9.6 ms     306.8 ms     killed -- a subset of TARGET_H1 costing more
SEED_H2       81.2 ms   2,635.6 ms     killed -- less recovery at half the corpus
```

The pre-launch ceiling of 40 s per query was wrong by three orders of magnitude
in the safe direction: it costed the QLS kernel at the local pure-Python
fallback because no measured figure existed. The cost model is now per query per
**arm**, at 1.5 s, which clears every surviving arm about sevenfold. Only
SEED_H2 exceeds it and SEED_H2 is dead; reviving it means re-deriving this.

**What B decided.** Three arms killed, each dominated on both declared axes on
both datasets, and two of them killed structurally rather than narrowly: PATH_H2
is a subset of SEED_H1 and BRIDGE_H2 a subset of TARGET_H1, each costing more to
build because identifying a path node takes a second matrix step that a radius
rule does not take. No quantity of further queries reverses a subset relation.
That the literature-style enclosing contexts are dominated by a plain radius-1
in-neighbourhood is a result rather than a formality, and it is the reason they
were run rather than argued about.

```text
scientific question   Do B's survivors hold at 300 queries across all six
                      datasets, and does the repair keep landing on the
                      candidates Phase -1 identified as starved?

arms                  CAND, SEED_H1, TARGET_H1. CAND stays first: it defines the
                      baseline every reach number is scored against.
datasets / queries    six datasets x 300 validation queries x 3 arms
models / seeds        none -- feature-only, no training, no GPU
number of jobs        6 CPU jobs, substrate container shape

estimated CPU-hours   <= 3.25   ceiling; measured throughput puts the real
                                figure near 0.7
estimated cost        <= $5.15  ceiling at $0.634/h; near $1 in practice
estimated wall time   <= 35 min, jobs concurrent, largest unit 0.54 h against a
                                1 h timeout
estimated storage     < 5 MB

stopping rule         Kill an arm that is dominated on (p95 construction
                      latency, median retention) on every one of the six
                      datasets. Ties kill nothing, an unmeasurable retention
                      decides nothing, and one dataset is not a frontier. The
                      rule is executable and tested, not prose.

advancement criterion An arm advances only if its recovery holds across all six
                      datasets AND its seed-distance improvement stays
                      concentrated on formerly isolated and low-degree
                      candidates. On B, TARGET_H1 improved 97.7% and 99.7% of
                      formerly isolated candidates against 49.1% and 29.2% of
                      ordinary ones; a survivor that flattens out across strata
                      on the remaining four datasets is rescaling, not repair.
```

Nothing trained, no GPU, no gold id, no test split, no candidate pool touched
and no GNN anywhere. E2 stays paused at 82/96 with its 69 seed-units unspent, F
stays sealed, and the workspace does not change.



---

## Graph-context pilot, closed out 2026-09-02: declared against actual

Both stages returned. The declarations above stand as filed; this records what
they actually cost, because a ledger that only ever records estimates is not a
ledger.

```text
                       declared ceiling        actual              over/under
Stage B  CPU-hours     <= 2.5                  0.086               3.5% of it
Stage B  cost          <= $2.00                $0.05               2.7% of it
Stage B  wall time     <= 40 min               4.6 min             concurrent
Stage C  CPU-hours     <= 3.25                 0.256               7.9% of it
Stage C  cost          <= $5.15                $0.16               3.1% of it
Stage C  wall time     <= 35 min               5.8 min             concurrent
                       ----------------------------------------------------
pilot total            <= 5.75 h / $7.15       0.34 h / $0.22      6.0% / 3.0%
```

Eight containers, no GPU, 1,232 container-seconds, at the $0.634/h substrate
shape. Storage under 5 MB as declared.

Two containers beyond those eight ran for about two seconds each and produced
nothing: the graph-context image shipped without `torch-geometric`, which the
runner needs transitively through the candidate-contract validator, and fifteen
sibling images already pinned. That is roughly $0.0004 of compute and the only
wasted spend in the pilot. `tests/test_spawn_registry.py` now asserts every
registered image installs the same pin, so the same failure cannot recur
silently.

The Stage C ceiling was 12.7x its actual on CPU-hours and 32x on cost. The
CPU-hour ceiling was deliberate headroom over a measured throughput; the cost
ceiling was not re-derived after the throughput measurement replaced the
placeholder rate, and it should have been. Ceilings are for approval, not for
prediction, but a 32x cost ceiling approves more than it needs to.

### Outcome

`TARGET_H1 = Cq u N1_in(Cq)` is the recommended context: median radius-1
retention 1.0000 and isolated fraction 0.0000 on all six datasets, 2,012-7,520
median context nodes, total p95 8.7-170.1 ms per query. `SEED_H1` survives the
frontier rule as an ablation. `PATH_H2`, `BRIDGE_H2` and `SEED_H2` are killed.
The advancement criterion filed above -- improvement concentrated on formerly
isolated and low-degree candidates -- was met on 6/6 datasets.

Full results, including the qualification that the context is bounded in
absolute node count and *not* in share of the graph on the three small corpora,
are in [`GRAPH_CONTEXT_PILOT_RESULTS.md`](GRAPH_CONTEXT_PILOT_RESULTS.md).

Nothing trained, no GPU, no gold id, no test split, no candidate pool touched
and no GNN anywhere. E2 stays paused at 82/96 with its 69 seed-units unspent, F
stays sealed, and the workspace did not change.

---

## Stage D0 declaration, filed 2026-09-02 after the pilot closed

Stage C established that `TARGET_H1` performs its intended repair. It did not
establish that the repair is useful, and retention 1.0000 is a consequence of
including a candidate's global one-hop neighbourhood rather than evidence about
ranking. D0 is the cheap gate before any GPU is spent: if the restored
information does not separate relevant candidates from irrelevant ones, training
a new GNN on it is pointless and the right result is the negative one.

```text
scientific question   Do structural quantities computed under TARGET_H1
                      discriminate relevant candidates from irrelevant ones
                      better than the same quantities under CAND?

arms                  CAND, TARGET_H1. CAND first: it defines the baseline.
datasets / queries    six datasets x 300 validation queries x 2 arms -- the
                      Stage-C sample, not a fresh draw
features              seed_distance, distinct_seed_support (negative control),
                      two_hop_seed_support, bridge_support
statistic             per-query rank AUC, ties at one half, reported beside the
                      tie fraction; unscorable queries counted, never imputed
models / seeds        none -- no parameters, no training, no GPU
labels                validation relevance, for SCORING ONLY. Read after the
                      pool and seeds; never passed to context construction. The
                      runner refuses the test split.
number of jobs        6 CPU jobs, the graph-context container shape

estimated CPU-hours   <= 1.2   ceiling. D0 runs 2 arms where C ran 3 and skips
                              the PPR descriptor kernel entirely, so it should
                              cost less than C's measured 0.26; the ceiling is
                              4.7x that, not a prediction.
estimated cost        <= $0.80 at $0.634/h. Derived from the CPU-hour ceiling
                              rather than guessed -- the Stage C ceiling was
                              32x its actual because it was not.
estimated wall time   <= 20 min, jobs concurrent
estimated storage     < 2 MB

stopping rule         None. Every job is minutes and the comparison needs all
                      six datasets to mean anything.

advancement criterion Stage D1 is declared only if TARGET_H1's discrimination
                      exceeds CAND's, AND does so on the formerly isolated and
                      low-degree strata where the repair lands. A gain that
                      appears only on already-well-connected candidates is not
                      the repair paying off. If neither holds: DO NOT SCALE,
                      and record that substantial graph structure was removed
                      historically and restoring it in this form did not improve
                      ranking.
```

Nothing trained, no GPU, no test split, no candidate pool touched and no GNN
anywhere. E2 stays paused at 82/96, F stays sealed, the workspace does not
change, and no strong-GNN run is authorised by this declaration.

## Stage D0, closed out 2026-09-02: declared against actual

```text
                       declared ceiling        actual              over/under
Stage D0 CPU-hours     <= 1.2                  <= 0.75 (BOUND)     <= 63% of it
Stage D0 cost          <= $0.80                <= $0.48 (BOUND)    <= 60% of it
Stage D0 wall time     <= 20 min               ~11 min             concurrent
```

Six CPU containers, no GPU, nothing trained. The actual is a **bound, not a
measurement**: the launcher recorded no container elapsed time, so the figure is
derived from observed wall-clock returns at <= ~450 s per container.
`scripts/modal_graph_context_pilot.py` now returns `elapsed_seconds`, so D1
reports a number. A ledger that can only bound its own spend is a ledger with a
hole in it, and this is where it was found.

### Outcome, against the criterion as filed

The filed criterion was: *"Stage D1 is declared only if `TARGET_H1`'s
discrimination exceeds `CAND`'s, AND does so on the formerly isolated and
low-degree strata where the repair lands."*

**It is half met, and the half that fails is recorded rather than reinterpreted.**

| clause | verdict |
|---|---|
| discrimination exceeds `CAND`'s overall | **NO.** `bridge_support` improves on 5/6 datasets by 0.010-0.066; `two_hop_seed_support` degrades on 5/6 by up to 0.274; `seed_distance` does not move. No overall dominance. |
| on the formerly isolated stratum | **YES, decisively.** `CAND` scores exactly 0.5000 there by identity -- all three support quantities are constant-zero for a candidate with no induced neighbour. `TARGET_H1` reaches 0.9513 / 0.9054 / 0.8749 mean AUC on the three datasets where the stratum is measurable. |
| on the low-degree stratum | **NO.** `bridge_support` at degree 2-4 improves on 2 of 6 datasets and degrades on 4, worst on squad (0.9611 -> 0.8423). |

The clause exists to rule out a gain confined to already-well-connected
candidates -- "a gain that appears only on already-well-connected candidates is
not the repair paying off". The observed result is the opposite failure: the
gain is confined to the *most* starved candidates and does not extend to the
merely sparse ones. The rule's stated purpose is satisfied; its literal
conjunction is not. That is a judgement, it is being flagged as one, and it is
the reason D1 is declared rather than launched on the strength of a rule that
did not anticipate this shape.

The user-level kill rule -- *do not scale if `TARGET_H1 <= CAND` within noise
**and** the restored features show no useful relevance separation* -- is a
conjunction whose second clause is decisively false, so it does not fire.

Full report: [`GRAPH_CONTEXT_D0_RESULTS.md`](GRAPH_CONTEXT_D0_RESULTS.md).

---

## Stage D1 declaration, filed 2026-09-02 after D0 returned

The first stage in this line that trains anything. It is deliberately the
smallest experiment that can answer its question: one dataset, one seed, two
contexts, no new architecture, no GNN, no test split.

```text
scientific question   Does a ranker with the frozen QLS-v1 architecture and
                      feature schema rank better when its features are computed
                      over TARGET_H1 than over the historical CAND context?

arms                  CAND, TARGET_H1. Identical in every respect except the
                      node space the frozen feature kernel runs on.
datasets              2wiki_clean only. Chosen before D0 returned, on Phase -1
                      grounds: hop-2 retention 13.6%, median rho1 11.1%,
                      boundary cut 83.9%, and no hotpotqa-scale serving tail.
                      D0 subsequently found it is also the dataset with the
                      largest addressable population -- 20.0% of its validation
                      queries have a gold candidate that was isolated in G[Cq],
                      against 1.0% on squad and 0.0% on metaqa.
splits                train for fitting, validation for reporting. The test
                      split is not read. The runner must refuse it.
models / seeds        exactly one seed, one model per arm. Two training runs.
candidates            bit-identical Cq in both arms, the frozen contract hash
                      checked at load as in every stage since B.
features              the shipped frozen QLS-v1 local descriptor kernel,
                      unchanged, called through graph_context.local_descriptors
                      so that neither arm can move because a second
                      implementation disagrees with the first.
metrics               R@1, R@5, R@20, MRR on validation, per arm, plus the
                      feature-build p50/p95/p99 both arms actually paid.
number of jobs        2 GPU jobs (one per arm), or 1 job running both arms.

projected feature     MEASURED per query on the Modal image at Stage C, 2wiki,
build                 300 validation queries, build + kernel:
                        CAND       p50 2.99 ms  p95 3.71 ms  p99 4.01 ms
                        TARGET_H1  p50 7.21 ms  p95 8.70 ms  p99 9.84 ms
                      CAND's mean over the same 300 queries is 43.3 ms and
                      its max 12,076 ms: the first call compiles the Numba
                      kernel. Percentiles are the honest summary here and the
                      mean is not; the one-off compile is paid once per
                      container, not once per query.
                      2wiki has 15,000 queries across all splits (frozen
                      candidate contract) and 5,389,449 candidate rows, so
                      359.3 candidates per query on average against the 316-364
                      median the timings were taken at -- the extrapolation is
                      to the same shape, not a different one. Costing every
                      query in both arms, which strictly over-counts because
                      test is never built:
                        CAND       <= 60 s at p99
                        TARGET_H1  <= 148 s at p99
                        both arms  <= 3.5 CPU-min = 0.06 CPU-h = $0.04

projected training    2 seed-units. E2 measured 69 seed-units at ~12 GPU-h
                      including container overhead, so 0.174 GPU-h per unit
                      -> 0.35 GPU-h; its pre-launch pure-compute rate was
                      0.041 h per unit -> 0.08 GPU-h. The larger is used.

estimated GPU-hours   <= 1.0  ceiling, dominated by fixed cost rather than by
                              training: two containers pulling the image and
                              loading 2wiki cost more than 0.35 h of fitting.
                              This is a ceiling for approval, not a prediction.
estimated cost        <= $2.29 = $2.25 GPU at the measured $2.241/h, plus $0.04
                              CPU at $0.634/h. Derived from the ceilings above.
estimated wall time   <= 40 min
estimated storage     < 50 MB

stopping rule         If either arm's feature build exceeds 10x its projected
                      p99, stop and re-derive before training. Nothing else:
                      two training runs are not worth a mid-flight gate.

advancement criterion Validation R@k and MRR for TARGET_H1 above CAND. If
                      TARGET_H1 <= CAND: DO NOT SCALE, and record the negative
                      result -- substantial graph structure was removed
                      historically, and restoring it in this form did not
                      improve ranking. Only on a positive result may the
                      two-dataset Stage E declaration (hotpotqa_clean for
                      severe truncation, musique_clean as the semantic-dominant
                      control, one seed each, both chosen before outcomes) be
                      PREPARED. Preparing is not launching.
```

**A confound in the frozen schema, declared before the run rather than
discovered after it.** Six of the ten frozen QLS-v1 descriptor columns are
normalised by a per-query maximum taken over the whole local node space, and
that space is what differs between the arms. A candidate whose own topology did
not change is rescaled simply because the context grew. So a positive D1 cannot,
on its own, be attributed to restored structure rather than to a moved
normaliser.

Holding the schema fixed is still the right control -- the normaliser is part of
the frozen system, and changing it would make the arms differ in two ways at
once -- so the run goes ahead as specified. What is added is a confound-free
channel to report beside the metrics: columns 0-3 are a one-hot seed-distance
bucket over `{0, 1, 2, >=3-or-unreachable}`, unnormalised, and they *do* separate
these two arms even though Stage C proved no context wider than `SEED_H1` moves
them. An isolated candidate sits at `>=3` under `CAND` and can sit at 1 or 2
under `TARGET_H1`. That column group is where restored structure shows up
without rescaling, and it will be reported per arm.

D1 trains no GNN, distils nothing, and reads no GNN output. E2 stays paused at
82/96 with its 69 seed-units unspent, F stays sealed, and the workspace does not
change. No strong-GNN run and no R0-R5 frontier is authorised by this
declaration.
