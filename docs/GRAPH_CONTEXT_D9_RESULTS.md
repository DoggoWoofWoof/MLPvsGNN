# Stage D9: the systems gate fired first, and the representation gate would have let it through

D9 was built with two kill switches. One stops the stage if the new
construction costs more than the feature build it augments; the other stops it
if the new block is representation-equivalent to the historical one, because
reporting "no difference" from a run that could not have found one is not a
result. Both thresholds were filed in the declaration before either was
computed.

The first one fired. The second, computed on validation before the abort and
recorded in full, would not have.

    branch-diversity construction   92.4 s   >   historical feature build   74.2 s
    -> STOPPED AT MICROBENCHMARK, no fitted arm

    frozen mechanistic gate, applied afterwards to the recorded comparison:
    hop 1  representation-equivalent        0.094% of within-query pairs reordered
    hop 2  DIVERGES  0.370%  |  hop 3  DIVERGES  0.845%   (threshold 0.0025)
    -> would NOT have fired; the stage would have trained one arm

**PATH REPLACEMENT EFFECTIVENESS UNTESTED — ABORTED BY PRE-REGISTERED SYSTEMS
GATE.** The representation was informative; what went unmeasured is whether it
ranks better. That distinction is the result, and the rest of this document is
about not letting it blur.

2Wiki, seed 0, `G[Cq]`, `CAND`. 13,500 queries opened, 3,000 reported;
4,851,276 candidate rows built, 1,079,071 of them validation. No arm was
fitted, so there is no ladder, no increment and no accuracy claim anywhere
below.

## Which gate fired, and on what

The construction is timed in three parts, because the declaration says shared
graph preparation is not charged to a new feature:

| part | seconds | what it is |
|---|---:|---|
| branch kernel | **9.84** | the new signal: three bounded passes over deduplicated edges |
| shared induced-edge extraction | **79.24** | `context_nodes` + `induced_edges` + `searchsorted`, work the historical build already does |
| D8 support, recomputed | **1.19** | the overlap diagnostic only; not a D9 feature |
| **total** | **92.4** | what the filed abort rule compares |
| historical local feature build | 74.2 | what it is compared against |

The kernel alone is 13.25% of the build it would sit beside. The total is
124.49% of it. The abort rule as filed and as implemented — identically to D8's
— compares the **total**, so it fired on the re-extraction, which the same
declaration says is not the feature's cost.

Those two clauses are in tension, and this run is where the tension became
decisive rather than academic. **The rule was not changed after seeing that.**
D8 passed the same rule at 42.0 s against a 51.4 s build; this container ran
the identical historical build in 74.2 s where D8's ran it in 51.4 s, so the
re-extraction scaled with it and crossed. A rule whose outcome depends on which
machine the container landed on is a rule that needs fixing before it is used
again — but fixing it *after* it produced an unwelcome answer, in the same
stage, would be exactly the result-driven rewriting this project forbids. It is
recorded here and proposed as the next experiment's first correction.

Per-query, the kernel is not the expensive part:

| | p50 | p95 | p99 | mean |
|---|---:|---:|---:|---:|
| branch kernel (ms/query) | 0.620 | 0.877 | 1.022 | 0.729 |
| shared extraction (ms/query) | 5.991 | 6.878 | 7.957 | 5.870 |
| historical build (ms/query) | 3.830 | 8.432 | 12.984 | 5.361 |

Temporary workspace 6368 bytes. Peak process RSS 6,250,299,392 bytes. Three
fixed graph passes, one per hop, no convergence loop.

## The historical formula, read from the kernel

VERIFIED FROM CODE — `src/mp_retrieval/structural_features.py:426-443`, not
inferred from the column names, and reproduced against `qls_local_features` on
eight constructed graphs before the replacement was defined.

    w_0 = the retrieval-seed indicator
    for each induced edge (source, target) of G[Cq]:
        w_h[target] += w_{h-1}[source]
    column_h = log1p(w_h) / max over the query's local node space
    the state carried to the next hop is the RAW count, not the column

| property | value |
|---|---|
| quantity | directed **walks** of length exactly h from any seed to the candidate — a walk count, not a path count and not a seed count |
| lengths | **exact**, not cumulative |
| direction | strictly source → target; the family is directed |
| visited set | none, so vertices and edges may repeat |
| **reciprocal** pair `s<->d` | gives `d` a walk at hop 1, the **seed** a walk at hop 2, and `d` another at hop 3 — one edge presented as three lengths of evidence |
| **parallel** edges | multiply counts |
| **self-loop** | contributes at every hop: a walk idles in place and manufactures hop-2 and hop-3 evidence without moving |
| same predecessor twice | counted separately — a **hub** with fan-out k gives its sink k walks from a single seed |
| normalisation | `log1p`, then divide by the per-query maximum, per hop |
| second normalisation | `candidate_readout` divides columns 4-9 again by the per-candidate maximum; under `CAND` the node space *is* the pool, so that division is by exactly 1.0 |
| dtype | float32, exact for integers below 2**24 |

The audit turned up an asymmetry D7's family decomposition could not show:
`seed_connections` credits **both** endpoints of a seed-incident edge and is
undirected in effect, while the path family follows edge direction only. The
two historical families D7 compared were not measuring the same kind of
adjacency.

Computation path: `run_graph_context_d1.build_local_features` →
`graph_context.qls_local_features` → `structural_features._local_feature_chunk`
→ `graph_context.candidate_readout`.

## The replacement, defined before implementation

Three quantities, kept separate on purpose:

    REACH_h(d)   which distinct SEEDS reach d in exactly h hops
    BRANCH_h(d)  through how many distinct immediate PREDECESSORS that arrives
    WALK_h(d)    how many enumerated length-h walks arrive -- the historical count

Only BRANCH is injected. "Distinct seeds within k hops" is multi-hop seed
support: it is D8's SUPPORT feature at a larger radius, and running it would
have answered a question D8 already answered. REACH is computed anyway, from
the same pass, so the separation can be measured rather than asserted.

    M_0(v) = {v} if v in Sq else empty
    M_h(v) = union over distinct predecessors p != v of M_{h-1}(p)
    branch_h(d) = |{p : (p,d) a distinct induced edge, p != d, M_{h-1}(p) non-empty}|
    branch_diversity_h(d) = branch_h(d) / (branch_h(d) + 1)

Self-loops are excluded from the count and from the propagation alike — a node
is not an independent branch of evidence into itself. Parallel edges are
collapsed once, before any hop.

The saturating scalar is bounded in [0,1] by construction, so it needs no
per-query maximum and a candidate's value never moves because another candidate
entered the pool. It is **strictly monotone** in the branch count, deliberately:
any ordering divergence from the historical column is then attributable to the
counting rule and not to the transform. It is concave, so fifty branches read
0.980 against three at 0.750 rather than sixteen times larger.

`branch/indegree` was considered and rejected: it is a purity measure that
saturates at 1.0 for every degree-1 candidate whose one predecessor carries
evidence, and 2Wiki's gold nodes concentrate in the isolated and degree_1
strata. The measured mean distinct in-degree here is 1.290, which is what that
rejection was about.

FORBIDDEN and not implemented: simple-path enumeration, DFS over all paths,
disjoint-path max-flow, motif enumeration, any path-list materialisation. Three
fixed passes, seed bitsets, no iteration to convergence.

## What the two signals actually measure

Validation feature construction: 1,079,071 candidate rows over 3,000 queries.
The comparison is on the **post-transform values actually presented to the
learner**, not on the raw counts.

| | hop 1 | hop 2 | hop 3 |
|---|---:|---:|---:|
| rows nonzero, historical | 50,003 | 93,090 | 134,554 |
| rows nonzero, replacement | 50,003 | 93,090 | 134,554 |
| nonzero agreement | 1.000000 | 1.000000 | 1.000000 |
| pooled Spearman | 0.999669 | 0.999151 | 0.998383 |
| Pearson | 0.958677 | 0.947219 | 0.943991 |
| **Spearman, supported rows only** | **0.409761** | **0.650787** | **0.710165** |
| rows differing numerically | 45,734 | 92,094 | 134,249 |
| fraction differing numerically | 4.238% | 8.535% | 12.441% |
| total-variation distance of the value distributions | 0.042 | 0.076 | 0.106 |
| **within-query pairs reordered** | **0.094%** | **0.370%** | **0.845%** |
| strictly reversed pairs | 209 | 83,743 | 395,695 |
| pairs the walk count separates and the replacement ties | 168,254 | 579,989 | 1,176,596 |
| pairs the replacement separates and the walk count ties | 13,307 | 55,395 | 68,410 |
| maximum walk count | 14 | 484 | 7,880 |
| maximum branch count | 7 | 113 | 255 |
| maximum reach | 7 | 8 | 9 |

194,238,384 within-query candidate pairs per column, counted exactly from the
per-query joint contingency table rather than sampled.

Read the last three rows first. At hop 3 the historical column ranges over
walk counts up to 7,880 while the branch count reaches 255 and the number of
distinct seeds reaching the candidate never exceeds 9. Walk multiplicity is
mostly multiplicity: the same evidence, recounted along every route it can
idle, backtrack or fan out through.

Then read the pooled Spearman against the supported-only Spearman. Pooled, the
two columns look nearly interchangeable (0.998-0.9997). Restricted to the rows
where either column is nonzero, the rank correlation collapses to 0.410 at hop
1 and 0.710 at hop 3. The pooled figure is an artifact of the shared zero mass
— 1.03 million of 1.08 million rows are zero in both columns — and **it was
recorded in advance that it would be**.

## The gate that was never reached

The gate is a three-condition AND, filed with its thresholds in
`configs/graph_context_pilot.yaml` before any of it was computed:

    abs(pooled Spearman) >= 0.995   AND
    fraction of within-query pairs reordered < 0.0025   AND
    nonzero-mask agreement > 0.995
    ... on EVERY mapped column -> STOP BEFORE TRAINING

| column | Spearman | ordering | nonzero | equivalent? |
|---|---|---|---|---|
| `paths_length_1` vs `branch_diversity_1` | pass | pass | pass | **yes** |
| `paths_length_2` vs `branch_diversity_2` | pass | **fail** | pass | no |
| `paths_length_3` vs `branch_diversity_3` | pass | **fail** | pass | no |

The gate does not fire. Hops 2 and 3 diverge materially; hop 1 does not. The
same evaluation over all 13,500 opened queries and 4,851,276 rows gives the
same answer (0.092%, 0.373%, 0.853%).

This is arithmetic applied to the comparison the stage measured on Modal, not a
second measurement: the abort returned at step 4 of the declared execution
order, so the runner never reached step 5 and the result file
carries no `frozen_gate` block. The comparison it is computed from is the one
the stage recorded.

Two of the three conditions were declared weak **before** the measurement, and
both behaved exactly as declared. `branch_h > 0` implies `walk_h > 0` by
construction, so nonzero agreement came back at 1.000000 in all three columns —
no discriminating power whatever. The pooled Spearman is inflated by the shared
zero mass and passed in all three columns too. The ordering condition is the
one with teeth, and it is the only one that separated the columns. Recording
that in advance is what stops a 0.9997 from being read as evidence of
equivalence now.

## REACH is not BRANCH, and the difference is measurable

The correction this stage was built around, checked rather than assumed:

| | hop 1 | hop 2 | hop 3 |
|---|---:|---:|---:|
| Pearson, REACH vs BRANCH | 1.000000 | 0.725978 | 0.637605 |
| rows where they differ | 0 | 36,786 | 74,356 |
| mean REACH | 0.056899 | 0.146211 | 0.261152 |
| mean BRANCH | 0.056899 | 0.158729 | 0.286512 |

At one hop they are the same number, and necessarily so: a hop-1 predecessor
carrying seed evidence *is* a seed. From hop 2 they come apart on 36,786 and
then 74,356 rows, and the ceilings separate hard — reach never exceeds 9 while
branch reaches 255. Injecting "distinct seeds within k hops" would have handed
the learner the flat column, not the structural one.

## Overlap with D8's distinct support

D7 measured strong sub-additivity — the individual family R@5 deltas sum to
+1.96 against a joint +0.89 — so a replacement built from propagated seed masks
might have been another encoding of support. Diagnostic only. D9 does **not train** distinct support and path
diversity together, and did not here.

| | Pearson | Spearman |
|---|---:|---:|
| D8 distinct support vs `branch_diversity_1` | 0.931212 | 0.999810 |
| D8 distinct support vs `branch_diversity_2` | 0.558870 | 0.527883 |
| D8 distinct support vs `branch_diversity_3` | 0.594940 | 0.611398 |

At hop 1 branch diversity is very nearly D8's feature — which is what the
concern predicted, and which is consistent with hop 1 being the one column the
mechanistic gate calls representation-equivalent. At hops 2 and 3 the overlap
falls away. The part of the path family that is not already support is the part
beyond one hop.

## Injection discipline and the tensor invariants

The three scalars are appended after `candidate_readout`, exactly as D5's
graded retrieval prior and D8's support fraction are. `normalised_columns`
stays `(4,5,6,7,8,9)` and was **not** extended to them; routing them through the
per-query maximum would restore precisely the query-relative rescaling the
replacement exists to remove.

The gate is elementwise equality against `branch/(branch+1)` on every candidate
row, and it held on all 4,851,276. The corroborating statistic: 13,498 opened
queries carry any diversity at all, and in **all 13,498** the column maximum is
below 1.0, in all three columns. A query-relative column cannot do that — every
occupied query would report exactly 1.0.

`D9_PATH_DIVERSITY_13` was verified equal to `D7_PATHS_13` in columns 0-3 and
10-12, equal to the injected scalars in columns 5-7, zero in columns 4, 8 and 9
in both arms, and differing from `D7_PATHS_13` in the three path columns and
nowhere else. Maximum absolute difference across every check: 0.0.

One real limit, measured rather than absorbed into a tolerance. The frozen
13-column block is float16. The 138 distinct branch-saturation values present
in the data collapse to 78 when stored, and branch counts above **51** are no
longer distinguishable from their successors. With branch counts reaching 255
at hop 3, the top of the range is compressed. The historical columns are
`log1p` ratios in the same dtype and pay the same cost, so this does not
advantage either arm — but it is a property of the frozen block that any future
structural feature inherits.

## Reuse instead of a third run

`D6_BASE_13` and `D7_PATHS_13` were fitted at this exact architecture, seed,
split and budget, and are reused rather than refitted. 33 conditions were
checked between `stage_d9` and `stage_d7` — dataset, seed, data fingerprint,
context, model, node count, candidate contract, feature schema, local
dimension, parameter accounting and the arm rows themselves — and 10 more
recompute D7's content-sensitive statistics from D9's own build: candidate rows
(4,851,276), residual nonzero entries, each of the three replaced path columns
separately (224,317 / 419,756 / 605,423 nonzero rows), and the prior's ranking
counts. All 43 matched.

Checking the three path columns separately is deliberate: a drift confined to
one column D9 replaces would otherwise hide inside a six-column total.

Since no arm was fitted, none of this was used to support an effectiveness
claim. It is recorded because a substrate that had drifted would have made the
mechanistic comparison meaningless too.

## Cost

229.0 s of container time on one A10G at $1.734/h: **0.064 GPU-h, $0.11**,
against 0.08 GPU-h and $0.40 authorised. The declaration projected 0.035 GPU-h
and $0.06 if the gate passed and 0 if it fired; this run fired a gate and still
cost more than the passing projection, because that projection was priced from
D8's container and this one ran the same historical build 1.44 times slower.

No GPU training was performed. The A10G was allocated and idle for the feature
work, which is the shape of every stage in this series and is why the
authorisation is a ceiling rather than a forecast.

Accounted work: 74.2 s historical feature build, 92.4 s branch-diversity
construction, 12.3 s graded retrieval prior. The remainder is data loading and
container startup.

## What this does not establish

**Nothing about effectiveness.** No arm was fitted. There is no ladder, no
increment, no `V - H`, and no statement anywhere above about whether branch
diversity ranks better or worse than walk multiplicity. The three readings
filed in advance — that the replacement wins, matches on Pareto grounds, or
loses — remain untested, and D7's PATHS classification is untouched.

**Nothing about whether the abort was the right call.** The rule fired as
written. Whether the quantity it compares is the right quantity is a separate
question, raised by this run and not settled by it.

**The mechanistic comparison is not a proxy for the effectiveness one.** It
says the two blocks are far enough apart at hops 2 and 3 for a training run to
be able to distinguish them. It does not say a training run would find a
difference that matters, and D8 is the standing reminder: there, the two
signals were nearly rank-equivalent *and* the corrected representation still
came out behind on all five metrics.

**Hop 1 is a partial exception in the other direction.** On that column the two
representations *are* equivalent under the filed rule, so even a fitted D9 arm
could not have attributed anything at hop 1 to the correction.

One dataset, one seed, development evidence on `G[Cq]`, and a gate that stopped
it early. Nothing here is an evaluation.

## Verdict

**PATH REPLACEMENT EFFECTIVENESS UNTESTED — ABORTED BY PRE-REGISTERED SYSTEMS
GATE**

The stage stopped at the microbenchmark under a rule filed before launch, and
spent no fitted run. That is the registered outcome and it is reported as
registered.

The runner's own constant for this branch is
`PATH REPLACEMENT UNINFORMATIVE ON 2WIKI`, and that string is still what
`stage_d9.json` says, because the result file records what the runner decided
at the time it decided it and is not edited afterwards. The label is withdrawn from the conclusion
line anyway: read plainly it says the representation carried no signal, and
that is the opposite of what was measured. Hops 2 and 3 diverge substantially
from the historical walk column, and the mechanistic gate would have passed.
Uninformative describes the systems race, not the signal. This is a wording
correction; no measurement in this document changed.

But the reason matters and is recorded rather than smoothed over. The abort
compared 92.4 s of construction against a 74.2 s build, and 79.24 s of that
construction was re-extracting induced edges that the historical build already
computes and that the same declaration says is not charged to a new feature.
The feature's own kernel is 9.84 s — 13.25% of the build, at 0.877 ms p95 per
query, in 6368 bytes of workspace, over three fixed passes.

And the gate that was supposed to decide whether this comparison was worth
running at all would have said yes. Hops 2 and 3 reorder 0.370% and 0.845% of
within-query pairs against a 0.0025 threshold; the walk column reaches 7,880
where the branch count reaches 255; on supported rows the two disagree far more
than the pooled correlation suggests.

So D9 ends with the substantive question open and a systems rule in need of
repair — and with the correction that motivated it confirmed on real data:
REACH and BRANCH differ on 74,356 rows at hop 3, and "distinct seeds within k
hops" would have been the wrong signal to inject.
