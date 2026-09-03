# M0B regime map: results, Steps 4-8

**Status:** `STEPS_4_THROUGH_7_COMPLETE`. This document is the record Step 8
(stop) points to. It reports what Steps 4-7 of
[`M0B_REGIME_MAP_PROTOCOL.md`](M0B_REGIME_MAP_PROTOCOL.md) measured and
proposes; it declares and launches nothing further.

**Source:** `outputs/m0b_regime_map/headline/*.json` (gitignored, not
committed; every number below traces to one of those six files). Six
datasets, 100 queries each, validation split, deterministic prefix,
`structural_only` mainline family, `graph_expansion_cap=64`. Launched via
`scripts/spawn_modal_jobs.py m0b-regime-map --stage headline`, all six
returned `M0B_REGIME_MAP_COMPLETE`.

## Step 4: what ran

| dataset | queries | num_nodes | candidate_rows (frozen contract) |
| --- | ---: | ---: | ---: |
| `squad_clean` | 100 | 19,029 | 41,524,403 |
| `2wiki_clean` | 100 | 65,865 | 5,389,449 |
| `musique_clean` | 100 | 13,672 | 6,616,805 |
| `hotpotqa_clean` | 100 | 507,494 | 34,086,252 |
| `metaqa` | 100 | 40,151 | 151,817,043 |
| `webqsp` | 100 | 781,485 | 541,514 |

Every job's `candidate_contract.status` reads
`BIT_EXACT_FROZEN_CANDIDATE_EQUIVALENCE` and every `expected_contract_sha256`
matches its `observed_contract_sha256` -- the frozen Cq each job scored is
verified bit-identical to the filed candidate pool, not assumed from the
dataset name.

## Step 5: invariant verification

**All seven hard invariants hold on all six datasets, all 600 queries.** No
`RuntimeError` was raised by any job (the runner raises on
`scored_r1_equals_scored_r2` and would raise on any of the other six):

| dataset | oracle_R1=R2 | scored_R1=R2 | U2⊇Cq | A64∩Cq=∅ | \|A64\|≤64 | Cq_struct=Cq∪A64 | scored_R1⊆R3 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `squad_clean` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `2wiki_clean` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `musique_clean` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `hotpotqa_clean` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `metaqa` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `webqsp` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

`NODE_ROLE` partition exhaustiveness/mutual-exclusivity is not a field in
this table because it is a construction guarantee of the `node_roles()`
helper itself (independently unit-tested), not a post-hoc numeric check on
aggregate output; nothing in this run contradicts it, and no per-query set
data is retained in the aggregate JSON to re-derive it from.

### Containment -- two separate figures, as filed, never conflated

**All-admitted-node containment** (`fraction(A64 inside U2)`, measured, not
assumed):

| dataset | admitted total | admitted in U2 | containment_rate |
| --- | ---: | ---: | ---: |
| `squad_clean` | 4,591 | 4,591 | **1.0000** |
| `2wiki_clean` | 1,638 | 1,638 | **1.0000** |
| `musique_clean` | 3,266 | 3,266 | **1.0000** |
| `hotpotqa_clean` | 4,167 | 4,152 | **0.9964** |
| `metaqa` | 2,298 | 2,298 | **1.0000** |
| `webqsp` | 1,636 | 1,636 | **1.0000** |

**This is the resolution of the directionality risk this protocol named in
advance, not a new problem.** `hotpotqa_clean`'s `graph.pt` is the one
dataset among the six whose base graph is not bidirectionally closed, so
`A64 ⊆ U2` was never a mathematical identity there the way it is for the
other five -- the protocol repeatedly flagged this as measured-not-assumed
and explicitly declined to close it on the 5-query smoke sample's 1.0
result. At 100 queries, the gap is real, small, and exactly where predicted:
15 of 4,167 admitted-instance events (0.36%) land outside U2. This does not
contradict any hard invariant (containment was deliberately never coded as
one) and does not trigger a stop -- it is the named risk resolving into a
specific, small, measured number instead of an open question.

**Recovered-gold containment** (`fraction(recovered gold inside U2)`,
reported separately, never merged with the figure above):

| dataset | recovered gold instances | already in R2 (U2) | beyond R2 (A64-only) | beyond U2, still not recovered |
| --- | ---: | ---: | ---: | ---: |
| `squad_clean` | 0 | n/a | n/a | 0 |
| `2wiki_clean` | 41 | 1.0000 | 0.0000 | 5 |
| `musique_clean` | 0 | n/a | n/a | 3 |
| `hotpotqa_clean` | 9 | 0.8889 | **0.1111** | 0 |
| `metaqa` | 43 | 1.0000 | 0.0000 | 0 |
| `webqsp` | 78 | 1.0000 | 0.0000 | **86** |

Two findings worth naming plainly:

- **`hotpotqa_clean` is the only dataset where structural admission (A64)
  recovered gold that R2's TARGET_H1 context did not.** One of its nine
  recovered-gold instances (11%) was beyond U2 and only reachable through
  the capped structural-neighbour rule. On the other five datasets, every
  recovered-gold instance was already inside U2 -- A64 added zero
  incremental recall-relevant reach beyond plain one-hop context, on this
  100-query sample.
- **`webqsp` has 86 gold instances that are beyond U2 and still not
  recovered even by A64** -- by far the largest such figure of the six, and
  consistent with `webqsp` also having the weakest candidate eligibility
  below. For `webqsp`, neither R2 nor R3's bounded structural reach closes
  most of the gap; this is a graph-availability ceiling A64 at cap=64 does
  not lift.

### Cold-start/steady-state split, confirmed at headline scale (100 queries, not 5)

| dataset | cold_start_compile_ms | R1 2nd-largest raw | R2 max raw | R3 max raw |
| --- | ---: | ---: | ---: | ---: |
| `squad_clean` | 10,916.0 | 13.47 | 180.50 | 171.71 |
| `2wiki_clean` | 11,797.8 | 2.98 | 6.01 | 89.80 |
| `musique_clean` | 11,206.7 | 1.29 | 15.21 | 14.06 |
| `hotpotqa_clean` | 12,636.3 | 76.57 | 235.26 | 1,621.10 |
| `metaqa` | 8,353.8 | 2.28 | 28.09 | 52.35 |
| `webqsp` | 13,660.5 | 76.80 | 68.15 | 393.49 |

At 6x the query count of the smoke sample, the pattern holds exactly: no R2
or R3 raw array anywhere approaches cold-start magnitude, on any dataset.
The one-time Numba parallel-JIT compile is isolated to R1's first call, and
only there, confirmed across 600 real queries now instead of 30.

### Peak RSS, confirmed at headline scale

| dataset | peak RSS (GB) |
| --- | ---: |
| `squad_clean` | 4.79 |
| `2wiki_clean` | 3.70 |
| `musique_clean` | 3.64 |
| `hotpotqa_clean` | 5.78 |
| `metaqa` | **7.69** |
| `webqsp` | 4.71 |

`metaqa` stays the real peak-RSS worst case at 20x the sample size (the
smoke-stage correction to `why_webqsp_stays_the_named_risk` in
`configs/m0b_regime_map.yaml` holds, not an artifact of a 5-query sample).
All six remain far under the 16GB container ceiling (highest at 48%).

## Step 6: the dataset x regime map

### 6a. Bottleneck attribution per dataset

Candidate eligibility is read from R1's headroom block
(`any_gold_at_pool`/`all_gold_at_pool`/`missing_gold_fraction_micro`); graph
availability is read from the containment/recovery tables above.

| dataset | all_gold_at_pool | missing_gold_frac (micro) | queries w/ zero gold in pool | recovered by R2/R3 | still missing after R3 | primary bottleneck |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `squad_clean` | 1.0000 | 0.0000 | 0/100 | n/a | 0 | **none** -- already at ceiling |
| `musique_clean` | 0.9600 | 0.0200 | 0/100 | 0 | 3 | graph availability, but the residual is tiny |
| `2wiki_clean` | 0.6100 | 0.2112 | 0/100 | 41 | 5 | **candidate eligibility**, mostly closed by R2 |
| `hotpotqa_clean` | 0.9000 | 0.0500 | 0/100 | 9 (1 via A64 only) | 0 | candidate eligibility; graph fully closes the residual |
| `metaqa` | 0.6000 | 0.3525 | **30/100** | 43 | 0 | **candidate eligibility**, severe -- graph closes what's missing |
| `webqsp` | 0.3100 | 0.5470 | **36/100** | 78 | **86** | **both** -- worst candidate eligibility AND a graph ceiling A64 doesn't lift |

Reading down that last column is the map the protocol asked for: three
distinct regimes of failure, not one universal story.
`squad_clean`/`musique_clean` are essentially solved at the candidate stage
already. `2wiki_clean`/`hotpotqa_clean`/`metaqa` are candidate-eligibility
bound, but their gap is one that graph context (R2, mostly; R3 marginally
for `hotpotqa_clean`) already recovers most or all of on this sample.
`webqsp` alone is bound at both levels simultaneously -- generalizing any
one dataset's story to "the" M0B story would have been wrong regardless of
which dataset was picked.

### 6b. Feature-family computability per regime

This is a computability/availability map, built from the frozen
[feature catalog](GRAPH_CONTEXT_FEATURE_CATALOG.md)'s family definitions
plus this run's real structural measurements (node roles, context-size
growth). **It is not an accuracy map** -- no per-column value or occupancy
rate was computed by this runner (that would be new scope beyond what
`configs/m0b_regime_map.yaml` declares), and nothing here ranks a family.

| family | R1 (`G[Cq]`, no context) | R2 (`TARGET_H1(Cq)`) | R3 (`TARGET_H1(Cq_struct)`) |
| --- | --- | --- | --- |
| RETRIEVAL | computable, identical to R2/R3 on Cq | computable, identical to R1/R3 on Cq | computable on Cq; **undefined on the A64-admitted nodes** -- they were never retrieval-scored |
| SEED | computable, identical across regimes | computable, identical across regimes | computable on Cq; undefined on A64-admitted nodes, same reason as RETRIEVAL |
| GEOMETRY | computable but starved -- no context nodes to bucket distance against | computable over U2's real context | computable over U3; richness gap vs. R2 tracks the context-growth table (6c) exactly |
| SUPPORT | starved, same reason as GEOMETRY | computable over U2's edges | computable over U3's edges; same dataset-dependent richness gap |
| PATH | starved | computable | computable; same richness gap |
| DIFFUSION | starved | computable | computable; same richness gap |
| TOPOLOGY | starved | computable | computable; same richness gap |
| PROVENANCE | n/a (graph-axis, not a per-candidate column) | n/a | **this entire run is fresh PROVENANCE evidence** -- `structural_only` measured on six datasets, not the prior three |
| NODE ROLE | only `RETRIEVAL_CANDIDATE` exists (0 `STRUCTURAL_SCORED_CANDIDATE` on all six) | same as R1 | **`STRUCTURAL_SCORED_CANDIDATE` is real and nonzero on all six datasets** -- the member the catalog listed as `NOT_TESTED` now has genuine presence data, though still zero accuracy evidence |

GEOMETRY/SUPPORT/PATH/DIFFUSION/TOPOLOGY are "starved" in R1 specifically
because R1's graph is `G[Cq]`, the candidate-induced subgraph with no
restored context -- these families' catalogued definitions all require
graph reachability beyond the scored set itself to produce a non-degenerate
distribution, which is exactly what R2/R3 add and R1 does not.

### 6c. Context growth R2 -> R3, the shared driver behind 6b's richness gap

| dataset | R2 context (median) | R3 context (median) | growth |
| --- | ---: | ---: | ---: |
| `squad_clean` | 6,195.5 | 7,125.0 | 1.15x |
| `2wiki_clean` | 2,037.5 | 40,196.0 | 19.73x |
| `musique_clean` | 2,171.0 | 3,418.5 | 1.57x |
| `hotpotqa_clean` | 2,973.5 | 167,313.5 | **56.27x** |
| `metaqa` | 5,960.0 | 10,461.5 | 1.76x |
| `webqsp` | 1,864.0 | 16,885.0 | 9.06x |

`2wiki_clean` is the dataset every family in the frozen catalog was actually
fitted on, and its own R2→R3 growth here is 19.73x -- meaning even on the
one dataset with real accuracy evidence for these families, that evidence
(D3-D10) was measured under R2-scale context, never R3-scale. Whether
GEOMETRY/SUPPORT/PATH/DIFFUSION/TOPOLOGY's D-ladder verdicts hold under a
~20x larger reachable graph is unmeasured by both the D-ladder and this run
-- M0B measures structural availability, not per-feature accuracy, by
design.

## Step 7: proposed minimum trained-screen matrix

**Proposal only -- `PROPOSED_NOT_DECLARED`, matching the discipline of
[`ZERO_TRAINING_MATRIX_PROPOSAL.md`](ZERO_TRAINING_MATRIX_PROPOSAL.md).**
Nothing below is launched by this document. Zero-training AUC/correlation
figures, where referenced, mean the D-ladder's existing univariate-screen
methodology (e.g. NODE ROLE's D0 `bridge_support` AUC of 0.9513) applied to
decide screening *priority*, never read as a feature-usefulness verdict --
per the standing instruction, that distinction is load-bearing and is
preserved here.

Priority is set by where Step 6 found real signal or a real gap, not by
uniform coverage:

| priority | dataset | family | why |
| --- | --- | --- | --- |
| 1 | `hotpotqa_clean` | NODE ROLE | the only dataset where `STRUCTURAL_SCORED_CANDIDATE` demonstrably recovered gold R2 alone missed (6b, 1/9 recovered-gold instances) |
| 1 | `webqsp` | GEOMETRY, SUPPORT, PATH, DIFFUSION, TOPOLOGY (R3) | worst candidate eligibility and a real 86-instance graph ceiling (6a) -- if any structural family screens well anywhere, this is the dataset with the most headroom for it to matter |
| 2 | `2wiki_clean`, `hotpotqa_clean` | GEOMETRY, SUPPORT, PATH, DIFFUSION, TOPOLOGY (R3 vs. R2) | largest R2→R3 context growth (6c) after `webqsp`; the D-ladder's 2Wiki verdicts for these families were never measured at R3 scale |
| 3 | `metaqa` | NODE ROLE, RETRIEVAL | severe candidate-eligibility gap (30/100 zero-gold queries) that graph context already closes -- worth confirming whether RETRIEVAL/SEED alone (no structural family) already explains the R2 recovery, before spending a screen on structural families here |
| deprioritized | `squad_clean` | all nine | at ceiling already (all_gold_at_pool=1.0, recall_ceiling@20=1.0) -- no family can improve what is not missing |
| deprioritized | `musique_clean` | all nine | residual gap is 3 gold instances total across the whole 100-query sample -- too small to screen informatively at this sample size |
| not proposed | any | PROVENANCE | not a per-candidate column; already measured as a graph-axis choice by this run itself and `EDGE_PROVENANCE_RESULTS.md`, not a screening target |

This is a priority ordering for a future declaration to pick up, not a
commitment that any cell above will be screened. Screening itself --
running the D-ladder's univariate AUC/correlation methodology against real
per-column feature values on these six datasets -- is new work with its own
cost and its own declaration, out of scope for M0B.

## Step 8: stop

M0B Steps 1-7 are complete. No further Modal launch, no new candidate-
admission rule, no per-dataset cap tuning, no learned offset, no
`TARGET_H1` change, no QLS/GNN training, and no screening run happens on the
strength of this document. The next action on any of Step 7's proposed
cells requires its own declaration, following this protocol's own
discipline.
