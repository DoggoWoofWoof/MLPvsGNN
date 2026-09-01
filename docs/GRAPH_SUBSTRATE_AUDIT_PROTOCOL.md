# Phase −1 — Graph Substrate Validity Audit

**Status:** `DIAGNOSTIC_SPEC_FROZEN_MEASUREMENTS_NOT_RUN`
**Date:** 2026-09-02
**Blocks:** the Phase 0–2 freeze in
[`QLS_V2_DEVELOPMENT_PROTOCOL.md`](QLS_V2_DEVELOPMENT_PROTOCOL.md)
**Changes nothing frozen:** Packages A–E and all 38 tags stand exactly as they are.

> **The QLS-v2 Phase 0–2 freeze is PAUSED.** Before deciding *which* query-local
> features matter, we have to establish *which graph they are defined over*. That
> question is logically prior and has never been asked in this project.

---

## 0. Why this phase exists

Every frozen Paper-1 result was computed on the **candidate-induced** graph
`G_q = G[C_q]`. An edge survives only if **both** endpoints were retrieved by
Dense or SPLADE. If the real graph contains

```
seed ── bridge ── gold
```

and retrieval returns `{seed, gold}` but not `bridge`, then the induced graph is

```
seed        gold          (no edge; the relationship is gone)
```

Neither a message-passing layer nor a QLS hop feature can recover that, because
neither is ever shown the bridge. If this happens often, the frozen experiment
answered

> *Can learned propagation help after semantic retrieval has already deleted most
> of the topology?*

rather than

> *Does learned message passing help retrieval ranking?*

Those are different questions, and only measurement can tell us which one we
answered.

**The candidate-induced object is still a graph.** It is not "not a graph"
because it is disconnected. The question is strictly *how much* of the original
neighbourhood structure it retains for propagation to aggregate over — a
quantity, measured continuously (§6.1), not a verdict.

---

## 1. What the code actually does — VERIFIED, not assumed

Traced in implementation, not documentation, as required.

### 1.1 The graph is a strict vertex-induced subgraph

[`complete_data.py:71-105`](../src/mp_retrieval/complete_data.py:71),
`CompleteRetrievalDataset.induced_subgraph`:

```python
starts   = rowptr[candidates]                     # expand each candidate's CSR row
degrees  = rowptr[candidates + 1] - starts        # its TRUE global out-degree
neighbors = col[neighbor_positions]               # all real global neighbours
...
keep[in_range] = sorted_candidates[positions[in_range]] == neighbors[in_range]
```

The last line is the whole finding. A neighbour is kept **only if it is itself a
candidate**. Confirmed:

```
Gq = G[Cq]          strict vertex-induced subgraph
                    non-candidate bridge nodes are deleted outright
                    edges are returned in LOCAL candidate indices
```

`tests/test_graph_substrate.py::test_induced_view_matches_the_frozen_induced_subgraph`
asserts the audit's own reimplementation agrees edge-for-edge with this function,
so Phase −1 measures the graph the models actually received and not a
reconstruction of it.

**A convenient consequence:** `degrees` above is each candidate's true global
degree, computed and then discarded. It is exactly the denominator of the
retention statistic in §3, so retention costs essentially nothing to measure.

### 1.2 The frozen GNN is ONE layer — and this is a second, separate restriction

[`operator_models.py:198`](../src/mp_retrieval/operator_models.py:198) builds
`layers` convolutions. Every frozen Paper-1 config sets:

```
configs/confirmation.yaml            layers: 1
configs/phase_confirmation.yaml      layers: 1
configs/edge_provenance.yaml         layers: 1
configs/candidate_budget.yaml        layers: 1
configs/online_systems.yaml          layers: 1
configs/phase_screen.yaml            layers: 1
configs/sa_mlp_confirmation.yaml     layers: 1
configs/six_dataset_study.yaml       layers: 1
configs/operator_screen.yaml         layers: 1
```

So the comparator's receptive field is **one hop of the induced graph**.

Meanwhile QLS-v1's features reach much further on the *same* substrate:

| QLS-v1 feature | Reach | Evidence |
|---|---:|---|
| `distance_0..3+` | 3 hops | `for hop in range(1, 4)` ([`structural_features.py:375`](../src/mp_retrieval/structural_features.py:375)) |
| `paths_length_1/2/3` | 3 hops | three propagation passes |
| `personalized_pagerank` | **8 hops** | `iterations: 8` (`configs/sa_mlp_screen.yaml`) |

> **Therefore QLS-v1 already has a strictly deeper receptive field than the GNN
> it is being compared against.** The frozen comparison is not "fixed summary
> versus learned propagation at equal reach"; it is "a 3-to-8-hop fixed summary
> versus 1-hop learned propagation".

This is a finding of this audit, it is not a defect in any frozen result, and it
does not invalidate anything. But it **changes the design of the fix** — see §7.2.

The depth sweep that would settle it already exists and was never run:
`configs/phase_diagram.yaml` declares `message_passing_layers: [0, 1, 2, 4, 8]`
under `status: deferred_until_operator_screen_gate`. **No depth evidence exists
anywhere in this repository.**

External evidence does exist and it motivates measuring depth (§9.2, verified
against primary sources): GCN, GraphSAGE and PinSage all use **two** layers in
their reported experiments, and GraphSAGE measures the first-to-second hop gain
at 10–15% accuracy on uncut graphs.

**That is a reason to measure depth, not to presume a correct one.** There is no
universal correct GNN depth — deeper is not automatically better, because of
oversmoothing and neighbourhood explosion, and GCN's own Appendix B reports
degradation past two or three layers. This document therefore does **not** claim
one layer is below a field standard. It claims only what the code shows:

> The historical seed-aware GNN has a **one-hop learned receptive field**, while
> QLS-v1 computes **multi-hop fixed summaries** on the same candidate-induced
> substrate. Learned-versus-fixed aggregation has therefore never been evaluated
> under matched structural reach.

The right descriptor for the frozen comparator is **a shallow one-hop historical
baseline**, not a substandard one.

### 1.3 Still to verify when the data is reachable

Not yet confirmed, and listed so they are not silently assumed:

```
directed vs undirected storage orientation per dataset
duplicate-edge handling in the frozen graph.pt
self-loop presence in the stored global graph
edge-provenance flattening point (native vs kNN)
```

The audit code excludes self-loops from every connectivity statistic and
symmetrizes for reachability, matching `candidate_headroom.symmetric_csr`, which
already records whether symmetrizing changed the stored graph.

---

## 2. Candidate-induced connectivity

Per query, and separately for `native` / `kNN` / `combined` edge provenance:

```
|Cq|                              components
non-self |Eq|                     largest-component fraction
isolated-node fraction            second-largest-component fraction
degree 0 / 1 / >=2 fractions
mean / median degree
```

Reported per dataset as `mean, p10, p25, median, p75, p90, p95, max`.

Implemented: `graph_substrate.connectivity_summary`.

**Interpretation fixed in advance.** If a dataset shows something like
`400 candidates, 330 isolated, LCC = 25`, then the one-layer GNN has been
operating as an MLP on 82% of its candidates, and that must be stated plainly in
the paper.

The provenance split matters because Package B already showed a graph can look
connected through kNN edges while its relational graph is fragmented. A result
of the form `combined LCC = 85%, native LCC = 18%` would mean apparent
connectivity is mostly semantic-similarity connectivity.

---

## 3. Global-neighbourhood retention

For every candidate `v`, excluding self-loops:

```
retention(v) = degree_{G[Cq]}(v) / degree_G(v)
```

Report `mean, median, p10/p25/p50/p75/p90`, and the fractions with retention
`= 0`, `< 10%`, `< 25%`.

Plus the boundary-cut ratio:

```
boundary(Cq) = |E(Cq, V \ Cq)| / |E(Cq, V)|
```

Implemented: `graph_substrate.retention_summary`.

A candidate with global degree 40 and induced degree 1 has lost 97.5% of what a
conventional GNN would aggregate. A high boundary ratio means retrieval is
cutting *through* neighbourhoods rather than selecting graph-coherent regions —
the condition GraphSAINT-style sampling is designed to avoid.

---

## 4. Effective receptive field — TWO notions, never merged

Symmetrised connectivity answers *"are these nodes related at all"*. It does not
answer *"can this candidate actually receive that node's signal"*. Both are
reported; they are different questions and the audit keeps them apart.

```
WEAK / SYMMETRISED                    ACTUAL MESSAGE FLOW
------------------                    -------------------
undirected view                       exact stored orientation
                                      + the operator's aggregation convention

used for:                             used for:
  components                            GNN R1 / R2 / R3
  largest component                     effective incoming neighbourhood
  retention, boundary cut               seed signal propagation
  gold path preservation                operator message load
  bridge loss
```

### 4.1 Why the distinction is not pedantic

A symmetrised view can show

```
seed -- bridge -- candidate
```

while the stored orientation is

```
seed <- bridge <- candidate
```

Every connectivity statistic calls that path intact. But messages travel
`source_to_target`, so the candidate aggregates nothing from the seed at **any**
depth, and a seed-aware GNN cannot propagate the seed indicator down it. Reporting
only symmetric reach would overstate what the operator can use — and would do so
in exactly the direction that flatters the substrate.

`tests/test_graph_substrate_message_flow.py` encodes this case directly, both as
a unit test and end-to-end through the runner.

### 4.2 Operator edge semantics — VERIFIED, not assumed

Measured by running each frozen factory on a three-node graph with one directed
edge, a duplicated edge and an isolated node:

| family | flow | inserts self-loops | coalesces duplicates | duplicate-sensitive | aggregation |
|---|---|---|---|---|---|
| `gcn` | source→target | **yes** | no | **yes** | sum, symmetric degree normalisation |
| `gat` | source→target | **yes** | no | **yes** | attention-weighted sum |
| `gin` | source→target | no | no | **yes** | sum, `(1+ε)·x` root |
| `sage` | source→target | no | no | no | mean, separate root linear |

Three consequences, all of which the audit reports:

1. **Edge multiplicity is a real message.** Package B established that the sealed
   graph is a multigraph (`baseline_a_simple` is the *deduplicated* projection of
   `sealed_a`). For `gcn`, `gat` and `gin` a repeated edge is a genuinely
   repeated message: GCN's degree normalisation counts it, GAT's softmax ranges
   over it, GIN's sum adds it twice. Only `sage`'s mean is invariant.
2. **The frozen selections are all duplicate-sensitive.** 2wiki `gat`, musique
   `gcn`, webqsp `gat`, hotpotqa `gin`, squad `gcn`, metaqa `gat` — **not one
   dataset uses `sage`.**
3. **`gcn` and `gat` add their own self-loop**, so a stored self-loop becomes a
   doubled one for four of the six datasets.

The audit therefore reports **unique non-self edges** and **messages actually
consumed by the operator** as separate numbers, per query and per graph.

### 4.3 An isolated candidate is scored, not dropped

All four families still emit a representation for a candidate whose receptive
field is empty — through an inserted self-loop (`gcn`, `gat`) or a root term
(`gin`, `sage`). So `R1 = 0` does not mean "unscored"; it means **scored as a
plain MLP**, with the topology input contributing nothing. That is the precise
sense in which a fragmented substrate silently degrades a GNN into an MLP, and
it is why `R1_zero_fraction` is a headline number rather than a footnote.

### 4.4 What is reported

```
R1 / R2 / R3       median, mean, zero-fraction     on BOTH notions
seed reach         fraction of candidates a seed signal reaches, per hop,
                   on the induced substrate (symmetric AND message-flow)
                   and on the global graph
message load       unique_non_self_edges
                   stored_non_self_messages
                   duplicate_messages and their fraction
                   stored_self_loops
                   operator_inserted_self_loops
                   messages_consumed_by_operator
```

---

## 5. Seed connectivity, induced versus global

For all candidates and separately for golds inside the pool, compute reachability
from `S_q` at `@1 / @2 / @3` on **both** substrates using the *same* function
(`graph_substrate.hop_distances`), so the difference is attributable to the
substrate and not to two different traversal implementations.

| Metric | Candidate-induced | Global |
|---|---:|---:|
| reachable@1 | … | … |
| reachable@2 | … | … |
| reachable@3 | … | … |

The gap is structural information destroyed by candidate induction.

---

## 6. Path preservation and bridge loss

For golds present in the pool, classify every seed→gold relationship:

```
connected globally + connected induced
connected globally + DISCONNECTED induced      <- the harm
distance increased by induction                <- a distinct, milder harm
already disconnected globally
```

```
P_h = P( d_{Gq}(Sq, g) <= h  |  d_G(Sq, g) <= h )        path preservation
bridge_loss@h = 1 - P_h
```

Implemented: `graph_substrate.path_preservation`, `graph_substrate.bridge_loss`.
Distance inflation is reported separately from disconnection because they are
different failures with different fixes.

Report `bridge_loss@2` and `bridge_loss@3`.

Split by `native` / `kNN` / `combined`, because a bridge lost from the native
graph and a bridge lost from the kNN graph have different causes and different
repairs.

### 6.1 There is no adequacy threshold, and none will be invented

The audit does **not** emit a verdict of the form *"the substrate is adequate"*
or *"the substrate is inadequate"*. No number in this document is compared
against a cutoff, and no cutoff is defined anywhere in the code, because any such
line would be arbitrary: nothing in the retrieval literature establishes a
retention fraction or a bridge-loss rate below which message passing stops being
worthwhile, and inventing one here would convert a measurement into an assertion.

What is produced instead is a **continuous characterisation** along nine axes,
each reported as a distribution rather than a pass/fail bit:

```
isolation                     R1_zero_fraction, and its full distribution
component structure           count, largest and second-largest fractions
neighbourhood retention       per-candidate distribution, not just the mean
boundary cut                  fraction of incident edges leaving the pool
effective receptive field     R1/R2/R3 on message flow AND symmetrised
seed reachability             per hop, induced versus global
gold path preservation        P_h for golds inside the pool
distance inflation            reported separately from disconnection
bridge loss                   1 - P_h at h = 2 and h = 3
```

The decision these feed is **not** "is the graph good enough". It is "which graph
basis should QLS-v2 structural features be defined over", and that is settled by
comparing the candidate-induced numbers against the global ones on the same
axes — a relative reading, which needs no absolute threshold. Where the audit's
own text pre-registers an expectation, it says what a value would *mean*, never
what a value would *decide*.

---

## 7. The graph-basis control

### 7.1 Two bases, same scoring set

```
CANDIDATE  (historical)      score Cq ; structural context = G[Cq]
GLOBAL     (new control)     score Cq ; structural context = the real graph
```

For QLS, GLOBAL means propagating seed bitmasks, distances, path statistics and
bounded diffusion **through non-candidate intermediate nodes**, then reading
features only for `d ∈ Cq`. The v2 bitmask design makes this a one-line change of
substrate: propagate over `G` from `S_q` for exactly `H` rounds, then read
`mask[v]` for `v ∈ Cq`. Same feature definitions, better substrate.

**Fairness rule, non-negotiable:** both methods receive equivalent graph-context
privilege. Never compare `QLS-CAND` against `GNN-GLOBAL`.

Required comparisons:

```
QLS-CAND   vs GNN-CAND          effect of learned propagation, old substrate
QLS-GLOBAL vs GNN-GLOBAL        effect of learned propagation, restored substrate
QLS-GLOBAL  - QLS-CAND          effect of substrate on the fixed summary
GNN-GLOBAL  - GNN-CAND          effect of substrate on learned propagation
```

### 7.2 The 2×2 is not sufficient on its own — depth must be matched too

This follows directly from §1.2 and is an addition to the proposed design, not a
substitution for it.

A one-layer `GNN-GLOBAL` sees **one hop** of the global graph. `QLS-GLOBAL` sees
**three** (and eight through PPR). Restoring the substrate would then hand QLS far
more new information than the GNN, and the 2×2 would read as *"global context
helps the fixed summary more than it helps message passing"* when the true cause
was that the GNN never had the depth to reach the restored context.

That is the same class of error as comparing `QLS-CAND` to `GNN-GLOBAL`, and it
is ruled out for the same reason.

**Therefore the design is `{CAND, GLOBAL} × {QLS, GNN} × H`, with the hop budget
matched across methods within each cell.**

But matching `H = L` is **necessary and not sufficient**, and the protocol must
say so plainly: a hand-engineered 3-hop statistic and three learned layers are
not computationally or representationally identical. Matching the hop budget
controls *topological reach*. It does not equalise capacity, and no result from
this grid may be described as though it did. Two separate comparisons are
therefore pre-registered.

#### 7.2.1 Mechanism-controlled grid — matched reach

```
graph basis = CAND                  graph basis = GLOBAL-CONTEXT
  QLS-H1  vs  GNN-L1                  QLS-H1  vs  GNN-L1
  QLS-H2  vs  GNN-L2                  QLS-H2  vs  GNN-L2
  QLS-H3  vs  GNN-L3                  QLS-H3  vs  GNN-L3
```

Answers: *at matched topological reach, do fixed statistics or learned
aggregation make better use of the structure that is available?*

#### 7.2.2 Strong-baseline comparison — the GNN at its best legitimate depth

Separately, let the GNN select `L ∈ {1, 2, 3}` on **validation only**, under a
frozen search and selection rule, and compare the final QLS configuration
against that winner.

This exists so the eventual QLS claim cannot rest on an artificially shallow
comparator. A reviewer who asks *"you beat a one-layer GNN, but a two-layer GNN
was better"* must have an answer in the paper. If the goal is to dominate message
passing, the thing to beat is its strongest legitimate version, not its
historical one.

**Leakage rule:** GNN outcomes never select QLS features, and test results never
select any depth.

`configs/phase_diagram.yaml` already specifies `message_passing_layers:
[0, 1, 2, 4, 8]`; Phase −1 is the reason to finally run it. Both controls are
registered in `configs/graph_substrate_audit.yaml` under `preregistered_controls`
with `status: DECLARED_NOT_RUN`.

### 7.3 Three GNN receptive-field standards

```
GNN-CAND     message passing on G[Cq]                     (historical baseline)
GNN-LOCAL    candidates + their true global H-hop context  (GraphSAGE/SEAL-style)
GNN-FULL     the complete global graph                     (where feasible)
```

`GNN-LOCAL` is the literature-standard construction and is the primary new arm.
Full-batch whole-graph propagation is **not** required for a fair comparison and
should not be forced on large datasets — neighbour-sampled or `H`-hop
neighbourhood construction is the norm precisely because full propagation is
expensive.

The verified reference constructions (§9.1) give it concrete, defensible
settings rather than invented ones:

```
GraphSAGE   uniform draw from the true adjacency, K = 2, S1 = 25, S2 = 10
PinSage     top-T by random-walk visit count on the real graph, T = 50,
            two layers, importance-pooled by those same weights
SEAL        the h-hop enclosing subgraph around the target
```

Each is closed under the graph's own neighbourhood operator, which is exactly the
property `GNN-CAND` lacks. PinSage is the closest analogue to what Phase 1 needs,
because it also has to bound a neighbourhood in a graph too large to propagate
over in full — and it bounds it **structurally**, by proximity on the graph,
rather than semantically.

### 7.4 GLOBAL-CONTEXT is not defined yet — the expansion audit defines it

`GNN-LOCAL` above names a *shape*. It does not yet name a size, and it must not
be frozen into a concrete construction before Phase −1 measures how big that
construction actually is. Two different neighbourhoods are both defensible
readings of "restore the global context", they answer different questions, and
they do **not** grow at the same rate:

```
U_seed(H)    = Cq  ∪  N_H(Sq)      the seeds' H-hop neighbourhood
U_target(H)  = Cq  ∪  N_H(Cq)      the pool's own H-hop neighbourhood
```

**`U_seed`** restores the `seed → bridge → candidate` path that vertex-induction
severs. It is the minimal repair for the specific failure §6 measures: a bridge
node that is not itself a candidate, and therefore not in `G[C_q]`, but that sits
between a seed and a gold.

**`U_target`** is the *computational* neighbourhood an `H`-layer GNN genuinely
requires. Every scored candidate aggregates over its own neighbours, those
neighbours aggregate over theirs, and so on `H` times. This is the standard
GraphSAGE/PinSage/SEAL object, and it is the one that explodes: `|S_q| ≤ 10` by
the seed contract, but `|C_q|` is an order of magnitude larger and much more
likely to contain a hub.

The audit measures both, for `H ∈ {1, 2, 3}`, on **both** connectivity notions
(§4), and on each provenance graph separately — `dataset_default`,
`structural_only` (native), `knn_only`, and `baseline_a_simple` (native ∪ kNN) —
because native and embedding-kNN edges have very different degree profiles and a
merged figure would hide that.

Reported per dataset, per graph, per `H`:

```
nodes            median, p90, p95, max
edges            median, p90, p95, max
density
expansion factor over |Cq|   (median, p95, max)
```

**Why this gates the definition.** The choice between exact `H`-hop construction
and neighbour sampling is an empirical consequence of these curves, not a
preference:

```
U_target(H) stays small          ->  build the exact H-hop neighbourhood
U_target(H) explodes             ->  GraphSAGE-style fixed-size sampling, or
                                     PinSage-style top-T random-walk selection,
                                     with the fan-out chosen from the measured
                                     degree profile
```

Either way the **scoring set remains exactly `C_q`** (§7.6), and expansion nodes
enter only as context. The measurement admits nothing to any pool; it reports
sizes. §8 is where admission is considered, and it is oracle-only.

### 7.5 Feasibility, estimated from artifacts that are already frozen

The graph sizes below come from the completed candidate-headroom diagnostic
(`outputs/candidate_headroom/*.json`, status
`CANDIDATE_HEADROOM_DIAGNOSTIC_COMPLETE`), so no new measurement was needed to
produce them.

| dataset | nodes | directed edges | mean degree | global CSR | R3 branching bound |
|---|---:|---:|---:|---:|---:|
| musique_clean | 13,672 | 280,108 | 20.5 | 2.4 MB | 100% of graph |
| squad_clean | 19,029 | 2,857,316 | 150.2 | 23.0 MB | 100% of graph |
| metaqa | 40,151 | 585,728 | 14.6 | 5.0 MB | 77.3% |
| 2wiki_clean | 65,865 | 855,146 | 13.0 | 7.4 MB | 33.2% |
| hotpotqa_clean | 507,494 | 16,223,058 | 32.0 | 133.8 MB | 64.4% |
| webqsp | 781,485 | 13,379,166 | 17.1 | 113.3 MB | 6.4% |

The last column is `min(N, 10 · d̄ʰ)` at `h = 3` — a loose branching upper bound
that ignores overlap and degree skew, given only to bracket the question. The
real frontier is what §4 measures, and it will be smaller.

**Storage is never the obstacle.** Every global CSR fits in RAM with room to
spare; the largest is 134 MB. Anything claiming a global basis is impossible on
memory grounds is wrong.

**GLOBAL QLS — feasible.** Bounded traversal from `|S_q| ≤ 10` seeds at
`Θ(H·|E_q|)` with a frontier cap, over a resident CSR. This is the same shape as
the v1 feature extractor, pointed at a different adjacency.

**GNN-LOCAL — feasible, and it is the literature standard.** `h`-hop
neighbourhood construction with GraphSAGE-style fixed-size sampling or
PinSage-style top-`T` random-walk selection (§9.1). This is the primary new arm.

**GNN-FULL — depends entirely on which comparator, and this is a code finding,
not an estimate:**

```
MessagePassingOperator          project_nodes(nodes) -> convs -> score against query
                                propagation does NOT depend on the query, so the
                                whole graph can be propagated ONCE per forward
                                pass and every query scored against the shared
                                node states.  Cheap even at hotpotqa scale.

SeedAwareMessagePassingOperator node_projection(cat([nodes, seed_indicator]))
                                the seed indicator is per query, so propagation
                                IS query-conditioned and must be repeated for
                                every query.
```

The headline frozen contrast is `sa_mlp_minus_seed_aware_gnn`, so it is the
**query-conditioned** variant that matters, and that is the expensive one. At one
layer and hidden width 64, full-graph propagation per query costs:

```
musique_clean     0.4 TMAC/epoch      2wiki_clean     0.8 TMAC/epoch
webqsp            1.4 TMAC/epoch      metaqa         15.3 TMAC/epoch
squad_clean      23.8 TMAC/epoch      hotpotqa_clean 101.6 TMAC/epoch
```

Sparse scatter-gather runs far below dense peak, so hotpotqa at 101.6 TMAC per
epoch per seed is not a five-seed protocol. **`GNN-FULL` should therefore be run
only where it is cheap, and `GNN-LOCAL` is the arm that must carry the
comparison.** That is not a compromise forced by budget — it is what GraphSAGE,
PinSage and SEAL all do, for the same reason.

### 7.6 The candidate ceiling must not move

GLOBAL still scores exactly `C_q`. Therefore the candidate ceiling must be
**bit-identical** to the historical pool, and this is verified by hash, not by
inspection:

```
candidate_contract_sha256      must match the frozen value
candidate_id_order_sha256      must match the frozen value
```

Any effectiveness difference is then attributable to restored graph context and
to nothing else.

---

## 8. Graph-expansion headroom — a SEPARATE question

Restoring graph *context* cannot recover a gold that is not in `C_q`. If
`g ∉ C_q`, neither method can return it, under any substrate. **Global context
does not fix retrieval headroom.** Only changing the scoring universe does:

```
GLOBAL context + fixed Cq   -> reranking experiment      ceiling UNCHANGED
expanded scoring universe   -> candidate generation      ceiling MAY IMPROVE
```

These must not be conflated.

Oracle-only diagnostic. **No model is trained on an expanded pool, in this phase
or in any phase gated by it.** The expansion is constructed, measured, and
discarded; the frozen pools are untouched and their hashes unchanged.

```
C0 = Dense u SPLADE                                   (the frozen pool)
C1 = C0 u {native <=1-hop from seeds}
C2 = C0 u {native <=2-hop from seeds}
C3 = C0 u {native <=3-hop from seeds}
```

and the same ladder on `knn_only` and on `native u kNN`, reported as three
separate curves rather than merged — the two provenance families have different
degree profiles, so a combined number would hide which one is paying the cost.

For every (dataset, provenance, hop) cell:

```
POOL SIZE          median, p95, max
CEILING            R@1, R@5, R@20
COVERAGE           AnyGold, AllGold, GoldFraction, FullCov ceiling
GAIN               additional golds recovered
COST               added nodes per recovered gold
RATE               ceiling gain per 1,000 added nodes
```

The last two are the Pareto quantities: coverage gain against pool explosion. A
`+3 hop` expansion that takes 400 candidates to 150,000 is not a system. Reported
per 1,000 added nodes so the ladders are comparable across datasets whose pools
differ by two orders of magnitude.

The summary figure is the trade-off curve itself, one panel per dataset:

```
x = median pool size (and a second series at p95)
y = R@5 ceiling
three curves: native, kNN, native u kNN
four points per curve: C0, C1, C2, C3
```

`C0` is the frozen operating point, so every curve starts at a number that
already exists in the sealed results and moves right and up from there. A curve
that goes far right for very little rise is the answer *"no"*, stated
quantitatively.

**Much of this already exists.** `candidate_headroom.missing_gold_reachability`
already buckets every missing gold by shortest hop distance from the frozen seeds
with `max_hops=3`, using `symmetric_csr` and `_expand`
([`candidate_headroom.py:284-413`](../src/mp_retrieval/candidate_headroom.py:284)),
and `docs/CANDIDATE_HEADROOM_PROTOCOL.md` is already written. Phase −1 extends
that from *"where are the missing golds"* to *"what would it cost to admit
them"*; it does not rebuild it.

### 8.1 The "where" half is already answered — and it is stark

The diagnostic has **completed on all six datasets**
(`outputs/candidate_headroom/*.json`). Test split, seeds =
`dense_top5 ∪ splade_top5`, undirected view, `max_hops = 3`:

| dataset | queries scanned | with a missing gold | missing golds | ≤1 hop | ≤2 hops | ≤3 hops | beyond / unreachable |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wiki_clean | 1,500 | 673 | 852 | 84.0% | 97.9% | 99.6% | 0.35% |
| hotpotqa_clean | 9,786 | 1,358 | 1,379 | 70.8% | 98.7% | 100.0% | 0.00% |
| metaqa | 39,093 | 29,079 | 260,233 | 20.7% | 58.6% | 100.0% | 0.04% |
| musique_clean | 1,995 | 286 | 313 | 26.5% | 69.0% | 92.0% | 7.99% |
| squad_clean | 13,033 | 67 | 67 | 11.9% | 49.3% | 92.5% | 7.46% |
| webqsp | 159 | 98 | 537 | 24.4% | 64.4% | 96.6% | 3.35% |

**Between 92.0% and 100% of every gold the retriever missed lies within three
hops of a node the retriever found.** On 2wiki and hotpotqa the majority is a
*single* hop away.

This is not itself a substrate measurement — it concerns golds *outside* the
pool, whereas §6's bridge loss concerns golds *inside* it whose connecting path
left. But it establishes the precondition under which vertex induction is most
destructive: the evidence is **locally concentrated around the retrieved set** in
the real graph. A construction that keeps an edge only when both endpoints were
independently retrieved is discarding exactly the region where these numbers say
the signal lives.

It also sharpens why §8's cost question is the one that matters. The graph
clearly knows where the missing evidence is. Whether admitting it is affordable —
added nodes per recovered gold — is unmeasured, and is what separates a real
system from a coverage number.

For context, the frozen test-split ceilings these sit against:

| dataset | pool coverage | ceiling@5 | headroom lost to candidate generation@5 |
|---|---:|---:|---:|
| squad_clean | 99.5% | 99.5% | 0.5% |
| musique_clean | 94.1% | 94.1% | 5.9% |
| hotpotqa_clean | 93.0% | 93.0% | 7.0% |
| 2wiki_clean | 79.7% | 79.7% | 20.3% |
| webqsp | 49.1% | 45.0% | 44.0% |
| metaqa | 33.5% | 32.6% | 49.9% |

This experiment likely belongs to Paper 2. It is run here as a diagnostic only,
to decide whether it is worth pursuing at all.

---

## 9. Literature positioning — VERIFIED 2026-09-02

Each claim below was checked against its primary source in this session. Two
findings came out of it, and the second was not anticipated when the audit was
proposed.

### 9.1 Every reference substrate is closed under the graph's own neighbourhood operator

| Method | Substrate it propagates over | Source |
|---|---|---|
| GCN | the normalised adjacency itself, `H⁽ˡ⁺¹⁾ = σ(D̃^(−1/2) Ã D̃^(−1/2) H⁽ˡ⁾ W⁽ˡ⁾)` with `Ã = A + I` | Kipf & Welling, ICLR 2017, [1609.02907](https://arxiv.org/abs/1609.02907) |
| GraphSAGE | a fixed-size **uniform draw from `{u : (u,v) ∈ E}`** — the true adjacency, subsampled for cost, never reselected | Hamilton, Ying & Leskovec, NeurIPS 2017, [1706.02216](https://arxiv.org/abs/1706.02216) |
| GraphSAINT | subgraphs of the training graph sampled to hold a fixed number of **well-connected** nodes at every layer, with an explicit **normalisation that removes the sampling bias** and variance-reduction samplers | Zeng et al., ICLR 2020, [1907.04931](https://arxiv.org/abs/1907.04931) |
| PinSage | the top-`T` nodes by L1-normalised random-walk visit count (an approximation to Personalized PageRank), importance-pooled by those same weights, `T = 50`, walks run on the real graph | Ying et al., KDD 2018, [1806.01973](https://arxiv.org/abs/1806.01973) |
| SEAL | the `h`-hop **enclosing subgraph** around each target link; the γ-decaying heuristic theory proves a broad class of link heuristics is well approximated from such local subgraphs | Zhang & Chen, NeurIPS 2018, [1802.09691](https://arxiv.org/abs/1802.09691) |
| GNN-RAG | a **dense** KG subgraph, followed by extraction of the shortest paths joining question entities to answer candidates, verbalised for the LLM | Mavromatis & Karypis, 2024, [2405.20139](https://arxiv.org/abs/2405.20139) |

PinSage is the sharpest case and it is worth stating carefully, because at first
glance it looks like a counterexample. PinSage **does** select a neighbourhood by
a scoring function rather than taking all neighbours — but the score is a
*structural* one, random-walk visit count computed **on the graph itself**, and
the selected set is by construction reachable from the target. Candidate
induction selects by a *semantic* score computed **off the graph**, by a
retriever that never consults an edge. The two operations are not the same
species: one is a structural approximation of the neighbourhood, the other is
independent of it.

So the framing to defend is not *"are subgraphs legitimate."* They plainly are,
and five of these six methods are subgraph methods. It is:

> **Is a subgraph induced on an off-graph semantic top-K a faithful propagation
> substrate?** Every reference above answers a different question, because every
> one of them derives its substrate from the graph's own neighbourhood operator —
> full adjacency, uniformly sampled adjacency, connectivity-aware sampling with
> bias correction, random-walk proximity, `h`-hop closure, or an explicitly dense
> subgraph. None of them induces on a set chosen without reference to the edges.

§§2–6 measure how far the candidate-induced substrate falls from that standard.
Nothing in §§1–8 depends on this section; the measurements stand on the code
trace in §1. This section decides only how the result is positioned.

### 9.2 Reference implementations commonly use two hops — a reason to measure depth

This was not part of the original motivation and is the more immediately
actionable finding.

```
GCN        experiments are 2-layer; Appendix B reports 2-3 layers best,
           degrading beyond that
GraphSAGE  K = 2 with S1 = 25, S2 = 10; K = 2 beats K = 1 by 10-15% accuracy,
           and K > 2 gives 0-5% for a prohibitive runtime cost
PinSage    two convolution layers at neighbourhood size T = 50
```

Three independent reference implementations use **two hops**, and GraphSAGE
quantifies the first-to-second hop gain at 10–15% — on graphs that were never cut.
The frozen Paper-1 comparator is **one** layer (§1.2) on a graph that was.

**This does not establish that one layer is wrong.** Depth is empirical: GCN's
Appendix B reports 2–3 layers best and degradation beyond, GraphSAINT studies 2-
and 4-layer variants, and oversmoothing and neighbourhood explosion are real
costs. Nothing here licenses assuming three layers is the correct GNN.

What it does establish is that **depth is a live variable that was never varied
here**, and that the frozen comparison mixes two differences at once — one hop
versus multi-hop reach, and learned versus fixed aggregation. That is why §7.2.1
requires a matched-reach grid and §7.2.2 requires a validation-selected strong
baseline.

It is **not** a reason to retract anything. Depth was a fixed, declared,
identically-applied protocol constant across every Paper-1 cell, so every
comparison it entered remains internally valid. It bounds the claim's scope, not
its correctness — see §10.

---

## 10. What this does NOT do to the frozen results

**Nothing.** Packages A–E, every sealed artifact and all 38 tags are unchanged
and remain valid.

Their scope is relabelled precisely, not retracted:

> The frozen Paper-1 results characterise the **candidate-induced reranking
> regime**: a retriever returns top-K, a graph is induced among those K, and a
> reranker scores them. That is a realistic and deployable systems configuration.

To the extent Phase −1 measures large structural loss, the correct statement is
that those experiments measured a **graph-starved reranking setting**, with the
degree of starvation reported as the measured quantity rather than as a label —
not that they were wrong. No frozen result, protocol or tag is rewritten.

---

## 11. Gating

```
FROZEN NOW      the audit metric definitions in this document
                the code in src/mp_retrieval/graph_substrate.py
                the fairness rules in 7.1 and 7.2
                the ceiling-invariance check in 7.6

BLOCKED         QLS-v2 Phase 0-2 freeze of the STRUCTURAL feature formulas
                (support, distance, path diversity, diffusion) -- their values
                depend on the substrate this phase selects

MAY PROCEED     the semantic frontier S0-S3, which is graph-independent
                Phase 0 instrumentation that does not fix a graph basis
                the Pareto selection procedure and tolerance review

PROHIBITED      training any global-context model before this diagnostic is
                reviewed; opening Package F; touching E2
```

**The QLS-v2 structural feature contract must not be frozen as candidate-induced
until Phase −1 reports.** The semantic-frontier work committed in `280348d` is
unaffected, because `cosine`, `dot`, `mean_abs_diff`, `semantic_product` and
`semantic_difference` do not reference the graph at all.

---

## 12. Execution status

**The metric specifications are frozen and the code is written and tested. No
measurement has been run.**

### What is built

| Artifact | Purpose |
|---|---|
| [`src/mp_retrieval/graph_substrate.py`](../src/mp_retrieval/graph_substrate.py) | the diagnostics: induced view retaining the global degrees `induced_subgraph` discards, connectivity, retention and boundary cut, receptive field R1/R2/R3 on both the symmetrised and the message-flow view, verified operator edge semantics and message load, multi-source hop distances usable on either substrate, path preservation, bridge loss, `U_seed`/`U_target` expansion sizes |
| [`tests/test_graph_substrate.py`](../tests/test_graph_substrate.py) | 12 unit tests, including an equivalence test against the shipped `induced_subgraph` and a direct encoding of the bridge-deletion counterexample |
| [`tests/test_graph_substrate_message_flow.py`](../tests/test_graph_substrate_message_flow.py) | 16 tests pinning directed message flow, the operator-semantics table against the real PyG layers, duplicate-edge message load, and `U_seed` vs `U_target` |
| [`scripts/run_graph_substrate_audit.py`](../scripts/run_graph_substrate_audit.py) | the runner: validates the frozen candidate contract first, then audits the dataset graph and each provenance family, checkpointing after every split |
| [`tests/test_graph_substrate_runner.py`](../tests/test_graph_substrate_runner.py) | 19 end-to-end tests over a toy corpus whose only seed→gold path leaves the pool; every asserted number is derived from that fixture by hand |
| [`configs/graph_substrate_audit.yaml`](../configs/graph_substrate_audit.yaml) | the registered protocol: substrate definition, measurement list, aggregation rule, graphs, gating |
| [`scripts/modal_graph_substrate_audit.py`](../scripts/modal_graph_substrate_audit.py) | CPU-only Modal driver, registered in `spawn_modal_jobs.py` as `graph-substrate` so submission is a server-side spawn, never `modal run --detach` |

### Two aggregation levels, never mixed

`node_level` pools individual candidates — the level at which retention ρ₁(v) is
defined. `query_level` averages a per-query summary — the level at which
connectivity, receptive field and path preservation are defined. Node-level
pooling uses a deterministic prefix of the split order, capped at 4,000 queries,
and the output records how many queries it drew from, so the distribution is
reproducible rather than sampled.

### Provenance comes from Package B, not from a rebuild

`structural_only` (native), `knn_only` and `baseline_a_simple` (the combined
Paper-1 graph) were already reconstructed in the frozen node coordinate system by
[`edge_provenance.py`](../src/mp_retrieval/edge_provenance.py) and persisted per
dataset under `edge_provenance_graphs/<dataset>/<fingerprint>/`. The audit reads
those sidecars and reports each family separately. It does not re-derive them,
and it records each family's `undirected_edge_key_sha256` so the graph a row was
measured on is identifiable after the fact.

### The audit cannot silently overwrite itself

`completed_audit` refuses to reuse a stored result whose dataset, data
fingerprint, graph list, split list, hop budget, or read-only flags differ from
the current request. Changing what is measured forces a new run rather than
returning a stale answer under a new question.

**Verified 2026-09-02:** the literature positioning (§9) is checked against
primary sources. It produced a second finding — reference implementations
commonly use two hops, and GraphSAGE measures the first-to-second hop gain at
10–15% on uncut graphs — which motivates **measuring** depth rather than
presuming one. The frozen comparator is described throughout as a shallow
one-hop historical baseline, never as substandard.

**Blocker:** the frozen graphs live on the Modal Volume — `storage/` does not
exist in the local checkout — and the workspace is still over its spend limit
(`ap-cj2qvLjN99Vcr4ki5r22sU` reports 0 tasks as of 2026-09-02). Phase −1 is
CPU-only and read-only, but it still needs the Volume. It runs the moment compute
is unblocked, and it is cheap: no training, no GPU, one pass per query.

### Ordering when compute returns: concurrent, not serialised

E2 and Phase −1 are **scientifically independent** and run at the same time:

```
E2 (phase confirmation)   GPU     resumes at unit 49/96, 392 units left
                                  frozen protocol, tagged, no design decision
                                  depends on Phase -1

Phase -1 (this audit)     CPU     read-only, one pass per query, no training
                                  writes to its own output prefix
                                  blocks the QLS-v2 structural freeze
```

There is no shared resource that forces a queue: E2 needs GPUs, Phase −1 needs
none; E2 writes under `phase_confirmation_cache/`, Phase −1 under
`outputs/graph_substrate_audit/`; neither reads the other's output. Nothing in
Phase −1 can alter an E2 cell, and no E2 result feeds a Phase −1 measurement.

Serialising them would be a real cost, not a neutral choice. Phase −1 is cheap
and blocks **all** v2 topology work, while E2 is a long sweep that blocks
nothing downstream of it. Making the cheap blocking measurement wait behind the
expensive non-blocking one would idle the v2 design for the duration of a
training sweep for no scientific reason. **Launch both.**

(This supersedes an earlier "resume E2 first" note in this document.)
