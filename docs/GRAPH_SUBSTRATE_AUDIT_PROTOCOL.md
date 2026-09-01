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
because it is disconnected. The question is strictly whether it retains enough of
the original neighbourhood structure for propagation to have anything useful to
aggregate.

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

External evidence does exist, and it points the same way (§9.2, verified against
primary sources): GCN, GraphSAGE and PinSage all use **two** layers in their
reported experiments, and GraphSAGE measures the first-to-second hop gain at
10–15% accuracy on uncut graphs. One layer is below the field-standard depth
before the substrate question is even raised.

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

## 4. Effective receptive field

Distinct non-self nodes reachable within `h` layers inside `G[C_q]`:

```
R_h(v) = |{u : d_{Gq}(u,v) <= h, u != v}|
```

Report median and mean `R1, R2, R3`, and the fractions with `R1 = 0`, `R2 = 0`.

Implemented: `graph_substrate.receptive_field_sizes`.

**`R1 = 0` is the number that matters most for the frozen comparison**, because
the comparator is one layer deep (§1.2). Every candidate with `R1 = 0` was
scored by a GNN that had literally nothing to message-pass, i.e. by an MLP.

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

Report `bridge_loss@2` and `bridge_loss@3`, split by `native` / `kNN` /
`combined`.

**Pre-registered reading:** if only ~20% of global ≤3-hop seed→gold
relationships survive induction, that is a major experimental confound and must
be reported as one.

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
matched across methods within each cell.** At minimum the GLOBAL arm must run the
GNN at `H = 3` layers to match the QLS hop budget, and report `H = 1` alongside it
so the frozen configuration stays visible and comparable.

`configs/phase_diagram.yaml` already specifies `message_passing_layers:
[0, 1, 2, 4, 8]`; Phase −1 is the reason to finally run it.

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

### 7.4 The candidate ceiling must not move

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

Oracle-only diagnostic, no model, no training:

```
C0 = Dense u SPLADE
C1 = C0 u {native 1-hop from seeds}
C2 = C0 u {native <=2-hop from seeds}
C3 = C0 u {native <=3-hop from seeds}
```

plus kNN-only and combined expansions. For each report:

```
median / p95 / max pool size        AnyGold / AllGold / GoldFraction
R@1 / R@5 / R@20 ceilings           FullCov ceiling
additional golds recovered          ADDED NODES PER RECOVERED GOLD
```

The last is the Pareto quantity: coverage gain against pool explosion. A `+3 hop`
expansion that takes 400 candidates to 150,000 is not a system.

**Much of this already exists.** `candidate_headroom.missing_gold_reachability`
already buckets every missing gold by shortest hop distance from the frozen seeds
with `max_hops=3`, using `symmetric_csr` and `_expand`
([`candidate_headroom.py:284-413`](../src/mp_retrieval/candidate_headroom.py:284)),
and `docs/CANDIDATE_HEADROOM_PROTOCOL.md` is already written. Phase −1 extends
that from *"where are the missing golds"* to *"what would it cost to admit
them"*; it does not rebuild it.

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

### 9.2 The canonical depth is 2, and the frozen comparator is 1

This was not part of the original motivation and is the more immediately
actionable finding.

```
GCN        experiments are 2-layer; Appendix B reports 2-3 layers best,
           degrading beyond that
GraphSAGE  K = 2 with S1 = 25, S2 = 10; K = 2 beats K = 1 by 10-15% accuracy,
           and K > 2 gives 0-5% for a prohibitive runtime cost
PinSage    two convolution layers at neighbourhood size T = 50
```

Three independent reference implementations converge on **two hops**, and
GraphSAGE quantifies the first-to-second hop gain at 10–15% — on graphs that were
never cut. The frozen Paper-1 comparator is **one** layer (§1.2) on a graph that
was. Those two facts compound: a one-layer GNN on a candidate-induced graph is
below the field-standard depth *and* propagating over a damaged substrate, and
the frozen results cannot separate the two causes.

This is external support for §7.2's requirement that any GLOBAL arm vary depth.
It is **not** a reason to retract anything: depth was a fixed, declared,
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

If Phase −1 finds severe graph destruction, the correct statement is that those
experiments measured a **graph-starved reranking setting** — not that they were
wrong. No frozen result, protocol or tag is rewritten.

---

## 11. Gating

```
FROZEN NOW      the audit metric definitions in this document
                the code in src/mp_retrieval/graph_substrate.py
                the fairness rules in 7.1 and 7.2
                the ceiling-invariance check in 7.4

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
| [`src/mp_retrieval/graph_substrate.py`](../src/mp_retrieval/graph_substrate.py) | the diagnostics: induced view retaining the global degrees `induced_subgraph` discards, connectivity, retention and boundary cut, receptive field R1/R2/R3, multi-source hop distances usable on either substrate, path preservation, bridge loss |
| [`tests/test_graph_substrate.py`](../tests/test_graph_substrate.py) | 12 unit tests, including an equivalence test against the shipped `induced_subgraph` and a direct encoding of the bridge-deletion counterexample |
| [`scripts/run_graph_substrate_audit.py`](../scripts/run_graph_substrate_audit.py) | the runner: validates the frozen candidate contract first, then audits the dataset graph and each provenance family, checkpointing after every split |
| [`tests/test_graph_substrate_runner.py`](../tests/test_graph_substrate_runner.py) | 15 end-to-end tests over a toy corpus whose only seed→gold path leaves the pool; every asserted number is derived from that fixture by hand |
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
primary sources. It produced a second finding — the canonical GNN depth in the
reference implementations is two layers, and GraphSAGE measures the
first-to-second hop gain at 10–15% on uncut graphs — which strengthens §7.2's
depth-matching requirement with external evidence rather than internal argument
alone.

**Blocker:** the frozen graphs live on the Modal Volume — `storage/` does not
exist in the local checkout — and the workspace is still over its spend limit
(`ap-cj2qvLjN99Vcr4ki5r22sU` reports 0 tasks as of 2026-09-02). Phase −1 is
CPU-only and read-only, but it still needs the Volume. It runs the moment compute
is unblocked, and it is cheap: no training, no GPU, one pass per query.

Ordering when compute returns: **resume E2 first** (it is a frozen, tagged,
half-finished experiment with 392 units left), then Phase −1, which has no
deadline and blocks only unstarted work.
