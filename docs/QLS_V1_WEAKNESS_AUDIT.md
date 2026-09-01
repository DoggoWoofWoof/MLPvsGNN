# QLS-v1 Weakness Audit (read-only diagnosis)

**Status:** `DIAGNOSTIC_ONLY_NO_FROZEN_RESULT_MODIFIED`
**Date:** 2026-09-02
**Method under audit:** QLS-MLP v1 (internal frozen code key `sa_mlp`)
**Supersedes:** the W1/W2/W3 numbering of the 2026-09-02 initial draft; the
weaknesses are unchanged in substance but are now split into six named defects
(W1–W6) so that each maps to exactly one proposed representational change.

## 0. What this document is, and what it is not

This audit re-reads the **already frozen** QLS-v1 implementation and the
**already frozen** Package A/B/C/D/E1 results to state, precisely and with
evidence, where QLS-v1 loses information and where it costs too much.

It is a diagnosis. It is explicitly **not** a rewrite, retune, deletion or
reinterpretation of any frozen result; not a selection of a favourable subset;
and not permission to train anything.

Package E2 is **running and untouched**. Package F is **unopened**.

### The thesis this audit now serves

The research objective has changed, and it changes what a "weakness" means here.
The target claim is no longer *"a clever MLP beats a GNN"*. It is:

> For retrieval ranking, most useful graph information reduces to a small set of
> query-conditioned structural statistics. Once those statistics are exposed
> explicitly, a tiny feed-forward ranker is sufficient; recursive learned message
> passing is unnecessary overhead.

Under that thesis, a QLS-v1 weakness is an **information loss in the feature
map** or an **unbounded cost in the feature computation** — not a shortage of
model capacity. Every defect below is one or the other. None of them is
"the model is too small", and §3 gives the frozen evidence that capacity is in
fact not the binding constraint.

---

## 1. The QLS-v1 feature contract, as implemented

Source of truth: [`src/mp_retrieval/structural_features.py`](../src/mp_retrieval/structural_features.py)
and [`configs/sa_mlp_screen.yaml`](../configs/sa_mlp_screen.yaml).

**7 static (query-independent) features** — [`structural_features.py:35`](../src/mp_retrieval/structural_features.py:35):

```
log_out_degree, log_in_degree, log_total_degree, pagerank,
hub_degree_percentile, coreness, clustering_wedge_estimate
```

**10 query-local (seed-conditioned) features** — [`structural_features.py:44`](../src/mp_retrieval/structural_features.py:44):

```
distance_0, distance_1, distance_2, distance_3_plus_or_unreachable,
seed_connections,
paths_length_1, paths_length_2, paths_length_3,
personalized_pagerank,
common_out_neighbors_with_seed_neighborhood
```

**The seed set is small and label-free.** `retrieval_seeds` is
`dense_top_k: 5` ∪ `splade_top_k: 5`, `union: stable_unique`,
`labels_used: false` ([`configs/sa_mlp_screen.yaml`](../configs/sa_mlp_screen.yaml)).
Therefore **5 ≤ |S_q| ≤ 10 for every query in every dataset.** This bound is
load-bearing for the v2 computation design and is stated here because it was not
previously recorded: per-seed statistics that would be prohibitive for large seed
sets are cheap at |S_q| ≤ 10, and a per-seed bitmask fits in a single 16-bit
word with no multi-word fallback path.

**Embedding width is 768**, projected to 64, and **both projections are
bias-free** — `node_projection` and `query_projection` are declared
`nn.Linear(embedding_dim, hidden_dim, bias=False)`
([`operator_models.py:34-35`](../src/mp_retrieval/operator_models.py:34)), and
`parameter_matched_head_width` accounts them as
`2 * embedding_dim * projection_dim`
([`:310`](../src/mp_retrieval/operator_models.py:310)). The semantic path alone
is therefore

```
2 x 768 x 64 = 98,304 trainable parameters = 46.0% of the 213,506-parameter model
```

spent before a single structural feature is consumed. *(An earlier revision of
this document gave 98,432 by including biases the layers do not have; 98,304 is
the figure, and the percentage is unchanged.)*

This is the **largest single parameter mass in QLS-v1** and it is not among the
six defects below, because it is not a defect — it is an unexamined assumption.
Whether a `768 → 64` projection is the necessary way to compare a query and a
candidate is exactly the question the QLS-v2 semantic frontier S0–S3 asks, at
`2 × 768 = 1,536` parameters. The ratio is the identity `2DP / 2D = P`, so the
compression is exactly **64×** by construction. See
[`QLS_V2_FEATURE_CATALOG.md`](QLS_V2_FEATURE_CATALOG.md) Group F.

---

## 2. The six defects

### W1 — Minimum seed distance, aggressively collapsed and bucketed

**Mechanism.** A multi-source BFS runs from all seeds jointly; each candidate
keeps `bucket = distance[node]`, clamped by
`if bucket < 0 or bucket >= 3: bucket = 3`
([`structural_features.py:384`](../src/mp_retrieval/structural_features.py:384)).
Two losses follow:

- the **distribution** of distances over seeds is discarded — only the minimum
  survives;
- distances 3, 4, 5, … and **unreachable** become one value.

**The counterexample this must resolve.** Candidate A is adjacent to seed 1 and
disconnected from every other seed. Candidate B is adjacent to seed 1 *and*
two hops from seeds 2, 3, 4. Both currently produce `min_distance = 1` and are
otherwise near-identical, yet B has four independent lines of evidence and A has
one.

**Information lost:** the seed-distance distribution; the reachable/unreachable
distinction beyond 2 hops.

### W2 — Seed support counts edges, not distinct supporting seeds

**Mechanism.** `seed_connections` increments per *edge* incident to a seed
([`:397`](../src/mp_retrieval/structural_features.py:397),
[`:400`](../src/mp_retrieval/structural_features.py:400)), both directions summed
into one scalar, then `log1p` and divided by the per-query max.

Three edges to a single seed and one edge to each of three seeds produce the
**identical** value. Multi-seed *agreement* — the signal distinguishing a genuine
bridge across retrieved evidence from a node merely adjacent to one popular
seed — is not representable at all.

**Information lost:** independent seed support; support concentration.

This is, on the evidence in §4, likely the single most damaging omission in v1.

### W3 — Path features are walk counts, not independent evidence paths

**Mechanism.** `paths_length_1/2/3` are computed by iterated propagation
`following[dst] += current[src]`
([`:434`](../src/mp_retrieval/structural_features.py:434)). These are **walks**:
revisits are counted, cycles and hubs inflate them without bound, and there is no
per-seed decomposition, no notion of redundancy, disjointness or branching.

A single high-degree intermediate node can dominate the count while contributing
exactly one line of evidence.

**Information lost:** path diversity; predecessor/branch structure; per-seed path
origin.

### W4 — Graph provenance is flattened

**Mechanism.** `topology_edges` is one flat array. Native/relational edges and
kNN similarity edges are indistinguishable to every local feature. A candidate
supported by three real relation edges and one supported by three embedding
neighbours receive identical structural evidence.

**Information lost:** edge provenance.

Note the GNN comparator consumes the same untyped edge set, so this is not an
asymmetry in v1's favour or against it — it is a shared limitation of the current
contract, and §6 records why v2 does **not** simply fix it in the headline model.

### W5 — Max-normalization compresses the useful mid-range

**Correction to a common description of v1, stated explicitly because the
initial draft of this audit had to correct it too:** v1's features are **already**
per-query normalized (`log1p(x) / query_max`, e.g.
[`:410`](../src/mp_retrieval/structural_features.py:410),
[`:442`](../src/mp_retrieval/structural_features.py:442)), and
`hub_degree_percentile` is already a percentile. **"QLS-v1 uses raw absolute
counts" is false and must not be repeated.**

The actual defect is narrower: max-normalization is **outlier-dominated**. Given
candidate degrees `2, 3, 4, 5, 6, 800`, max-normalization yields
`0.0025, 0.00375, 0.005, 0.00625, 0.0075, 1.0` — every candidate that ranking
actually has to separate is compressed into a near-indistinguishable band, and
the compression worsens as the candidate pool grows.

**Information lost:** resolution among mid-ranked candidates, precisely where
ranking decisions are made.

### W6 — The cold-query cost is a heavy tail in one stage

**Mechanism.** Per query the local block performs roughly 3 BFS edge passes + 1
seed-connection pass + 1 common-neighbour pass + 3 path-propagation passes + 8
PPR iterations ≈ **16 full passes over the induced edge list**. BFS frontier
expansion and path-count propagation are near-identical operations over the same
array, executed separately. PPR runs a fixed 8 iterations regardless of induced
graph size or convergence.

**Cost defect:** an unbounded per-query worst case, concentrated in
`query_local_summary`, not distributed across the pipeline.

---

## 3. Why the fix is representational, not architectural

The strongest evidence for the new thesis is already frozen, in Package A3.

**A3 is a bias-free 19-parameter linear scorer** over two source-rank, seven
static-graph and ten query-local structural features. It loads **neither graph
adjacency nor node/query embeddings**
([`P0_LINEAR_RANK_STRUCTURE_RESULTS.md`](P0_LINEAR_RANK_STRUCTURE_RESULTS.md)).

R@5, frozen:

| Dataset | Selected RRF | A3 linear (19 params) | QLS-MLP (213K) | Seed-aware GNN (213K) |
|---|---:|---:|---:|---:|
| 2wiki_clean    | 68.48 | 68.57 | 68.40 | 69.85 |
| musique_clean  | 69.24 | 68.51 | 80.28 | 81.24 |
| webqsp         | 10.20 | 22.21 | 33.37 | 33.09 |
| hotpotqa_clean | 72.24 | 74.58 | 77.13 | 77.66 |
| squad_clean    | 89.31 | 89.50 | 89.23 | 89.33 |
| metaqa         | 13.75 | 24.48 | 30.11 | 30.13 |

A3 recovers **51.8% (WebQSP), 47.9% (HotpotQA), 65.6% (MetaQA)** of the
selected-RRF → QLS gap — with 19 parameters, no embeddings and no adjacency. On
2wiki and squad it matches QLS outright.

Two conclusions follow, and they point in the same direction:

1. **Roughly half the structural value is already linear in v1's features.**
   The marginal return on model capacity is therefore low; the marginal return on
   *better features* is where the remaining gap lives.
2. **The residual gap is largest exactly where v1's features are most collapsed.**
   A3 trails QLS by 11.77 on musique and 11.16 on webqsp — datasets where seed
   geometry and multi-seed support (W1, W2) carry the most information and where
   v1 destroys the most of it.

This is the empirical basis for the v2 strategy: **find the minimum sufficient
feature set first; choose the smallest learner that can combine it second.**

Corollary for the parameter budget: 46% of v1's parameters are the 768→64 q/x
projection (§1). If scalar retrieval features suffice in place of projected
embeddings, that mass disappears entirely rather than being re-spent.

---

## 4. Evidence per defect

### W1–W4: relational information is real, and v1 extracts less of it

Package B (edge provenance, FROZEN, 24/24) varies **only** the edge family under
a fixed candidate contract. Recall@5, points, mean over 5 seeds:

| dataset | QLS `symbolic_b` − `knn_only` | GNN `symbolic_b` − `knn_only` | **extraction gap** |
|---|---:|---:|---:|
| 2wiki_clean    | +2.02 | +4.05 | **+2.04** |
| webqsp         | +2.66 | +5.37 | **+2.71** |
| hotpotqa_clean | +4.15 | +4.60 | +0.45 |
| squad_clean    | −0.02 | +0.02 | +0.03 |
| musique_clean  | +0.75 | +0.53 | −0.21 |
| metaqa         | +6.76 | +6.62 | −0.14 |

Both models gain from relational edges. On **2wiki and webqsp the GNN converts
the same edges into 2.0–2.7 more recall points than v1 does** — that difference
is the measurable size of v1's representational loss. On metaqa and musique v1
extracts the signal as well or better, so this is **not** a universal deficit.

The complementary result is sharper. GNN − QLS by family (Holm-adjusted across
datasets within family, `*` = p < 0.05):

| dataset | symbolic_b | knn_only |
|---|---:|---:|
| 2wiki_clean    | +1.51* (0.026) | −0.52 (0.217) |
| webqsp         | +1.53  (0.417) | −1.18 (0.538) |
| hotpotqa_clean | +0.24* (0.038) | −0.21 (0.139) |
| musique_clean  | +0.55  (0.417) | +0.77 (0.217) |
| squad_clean    | +0.04  (0.555) | +0.00 (1.000) |
| metaqa         | −0.14* (0.008) | +0.00 (1.000) |

> **On similarity-only graphs the GNN never significantly beats QLS-v1 in any
> dataset** (every `knn_only` p ≥ 0.139; point estimate negative in 4/6).
> **Learned message passing does not obtain its advantage from kNN topology.**

This is a preserved narrower finding and it must not be generalized away. It also
motivates a Pareto experiment that costs nothing to try: if *dropping* kNN edges
improves accuracy, latency and memory simultaneously, that is a win requiring no
new feature at all (see [`QLS_V2_DESIGN.md`](QLS_V2_DESIGN.md) §7).

**Caveat, stated plainly:** the family-to-family contrast carries **no p-value**.
The frozen paired test is QLS vs GNN *within* a family, not family vs family. The
extraction-gap column is a point estimate; its significance is unestablished by
any frozen artifact and requires a development-set measurement (§7).

**Quantity confound rebutted:** `full_union_c` has the most edges on every
dataset yet is *worse* than `symbolic_b` for QLS on 2wiki (67.16 vs 68.13),
hotpotqa (76.46 vs 77.69) and metaqa (30.02 vs 30.44). More edges is not the
mechanism; relational edges are.

### W5: the compression worsens as the pool grows

Package C structural context, per query, median:

| dataset | budget 50 → 400 | candidates | directed edges | density | components |
|---|---|---:|---:|---:|---:|
| 2wiki_clean    | | 50 → 364 | 60 → 662 | 0.0245 → 0.0050 | 34 → 196 |
| hotpotqa_clean | | 50 → 353 | 104 → 848 | 0.0424 → 0.0069 | 25 → 154 |
| metaqa         | | 50 → 379 | 78 → 658 | 0.0318 → 0.0047 | 30 → 203 |
| webqsp         | | 50 → 347 | 318 → 2337 | 0.1298 → 0.0194 | 9.5 → 68 |

Density falls 4–6× on every dataset while absolute component count grows 5–7×.
Under W1 a growing share of candidates lands in the single
`distance_3_plus_or_unreachable` bucket with near-zero paths and near-zero PPR — a
**degenerate signature shared by many candidates at once** — while under W5 the
survivors are compressed by whichever hub holds the per-query max.

GNN − QLS on Recall@5 by budget, `*` = Holm-significant:

| dataset | 50 | 100 | 200 | 400 |
|---|---:|---:|---:|---:|
| 2wiki_clean    | +0.16 | +0.63 | +1.02* | +1.28* |
| hotpotqa_clean | +0.22 | +0.31 | +0.44* | +0.70* |
| metaqa         | +0.09* | +0.10* | +0.11* | +0.02 |
| musique / webqsp / squad | ns | ns | ns | ns |

> **Preserved narrower finding: the context-related separation is primarily
> 2wiki and hotpotqa, and is not universal.** On metaqa it is significant at
> small budgets and vanishes at 400 — the opposite direction. On webqsp the
> point estimate is negative throughout.

Attainment falls with budget for *both* models everywhere, so decline itself is
not QLS-specific; the differential is (2wiki: QLS −4.0 pts vs GNN −2.7;
hotpotqa: −4.5 vs −3.9; webqsp: reversed).

### W6: the tail is one stage, and the median is already competitive

Package D (FROZEN, 6/6) reports QLS slower in 10/12 cells **by mean**. That
report stands unmodified. Decomposing the *same* frozen artifacts by percentile
(batch 1, ms/query):

| dataset | QLS p50 | GNN p50 | QLS p95 | GNN p95 | QLS mean | GNN mean |
|---|---:|---:|---:|---:|---:|---:|
| 2wiki_clean    | **2.816** | 3.102 | 7.394 | 4.236 | 3.767 | 3.243 |
| webqsp         | **2.582** | 2.782 | 6.924 | 3.674 | 3.489 | 2.916 |
| metaqa         | **2.872** | 3.197 | 7.447 | 4.428 | 3.765 | 3.329 |
| musique_clean  | **3.009** | 3.017 | 7.558 | 4.151 | 3.919 | 3.136 |
| hotpotqa_clean | 2.953 | 2.604 | 7.519 | 4.012 | 3.998 | 2.886 |
| squad_clean    | 3.794 | 3.638 | 8.166 | 5.408 | 4.391 | 3.840 |

**QLS-v1's median is already better than the GNN's in 4/6 datasets.** The deficit
is entirely tail. Stage p95/p50 ratios:

| stage | ratio across the six datasets |
|---|---|
| `fusion_and_seed_ms` | 1.18 – 1.38 |
| `topology_induction_ms` | 1.26 – 1.84 |
| `gather_transfer_forward_topk_ms` | 1.15 – 1.46 |
| **`query_local_summary_ms`** | **6.68 – 8.69** |

`query_local_summary_ms` has p50 0.53–0.72 ms but p95 **4.60–4.79 ms** — and that
p95 is nearly identical across all six datasets despite induced graphs differing
by more than an order of magnitude in edge count. **A near-constant tail across
wildly different graph sizes is the signature of an unbounded per-query worst
case, not of dataset scale.**

---

## 5. Relationship to the frozen reports

[`ONLINE_SYSTEMS_RESULTS.md`](ONLINE_SYSTEMS_RESULTS.md) reports the mean-latency
comparison and is correct as written. §4's percentile decomposition uses the
**same frozen measurement artifacts** — it is a refinement, not a correction and
not a different measurement. Both are simultaneously true because the
distribution is right-skewed.

The frozen report is left exactly as tagged. Any paper text quoting the mean must
also quote the percentile split, because the mean alone implies a uniform
overhead the data does not show.

---

## 6. Dimensions on which QLS-v1 already matches or beats the GNN

A v2 that regresses these has not dominated anything.

- **GPU memory** (incremental peak, batch 1): QLS 4.42–5.32 MB on every dataset;
  GNN 4.75–67.22 MB and it scales with graph size (hotpotqa 67.22, squad 28.73).
  v1 already dominates by **>12×** on hotpotqa.
- **Median cold latency**: better in 4/6 datasets.
- **Similarity-only graphs**: the GNN never significantly beats v1 anywhere.
- **Effectiveness on 3/6 datasets**: no significant GNN advantage at any budget on
  musique, webqsp or squad; on metaqa `symbolic_b` the GNN is significantly worse.
- **Parameters**: 213,506 vs 213,568 on 2wiki/metaqa/webqsp (marginally under);
  209,351 vs 209,280 on musique/squad (marginally **over**).

v1 is behind on: **throughput** (GNN wins 6/6: 250–347 vs 228–287 q/s),
**p95/p99** (6/6), **mean latency** (10/12 cells), and **R@5 on 2wiki and
hotpotqa at large candidate budgets**.

---

## 7. What this audit could not establish

Not asserted above; requires new development-set (validation-only) analysis:

1. **Query-level correlates of the v1−GNN gap** against inference-safe covariates
   (connected-seed fraction, path redundancy, diffusion concentration, hub
   exposure, component size).
2. **The degenerate-signature fraction** by candidate budget, and the gap
   restricted to those candidates — the direct test of W1/W5's mechanism.
3. **Significance of the relational extraction gap** (§4). No frozen test covers
   the family-vs-family contrast.
4. **Which of the ~16 edge passes produces the tail.** §4 localizes it to the
   stage, not to the operation.
5. **Whether the seven global static features carry any signal.** Detailed
   below, because it directly shapes the v2 frontier.

### 7.1 The seven global static features were never isolated

A3 consumes all nineteen inputs — two rank features, the **seven graph-wide
static features**, and the ten query-local features
([`P0_LINEAR_RANK_STRUCTURE_PROTOCOL.md`](P0_LINEAR_RANK_STRUCTURE_PROTOCOL.md)
lines 25–35) — but its nineteen learned coefficients were **not persisted**.
`outputs/p0_linear_rank_structure/*.json` records metrics, timing and contract
hashes; there is no weight vector in any artifact. The individual contribution of
`log_out_degree`, `log_in_degree`, `log_total_degree`, `pagerank`,
`hub_degree_percentile`, `coreness` and `clustering_wedge_estimate` has therefore
**never been measured**, and this audit cannot say whether they carry signal.

What *can* be said, and is the strongest available evidence: A2's
`structural_summary` — the source of the only demonstrated fixed-structure wins,
**+5.20 R@5 on webqsp and +4.41 on metaqa** — is constructed from the **ten
query-local features only** (`fixed_structural_scores` takes a `[N, 10]` array;
`structural_controls.py:120-141` combines distance, path/connectivity and PPR).
**Zero static features participate.** Every structural gain this project has
actually demonstrated came from query-local structure.

That is why QLS-v2 places the seven static features outside the primary frontier
R0–R5 and keeps only one of them in the audit catalog. It is a prediction with
partial support, recorded as such, not a settled result — see
[`QLS_V2_FEATURE_CATALOG.md`](QLS_V2_FEATURE_CATALOG.md) §3 prediction 10.

---

## 8. Package F allocation — decided 2026-09-02

Recorded here because this audit is the document that establishes what the six
development datasets can and cannot support.

**Package F is reserved exclusively for the final QLS-v2 confirmation.** It is
not opened, not inspected, and **QLS-v1 will not be run on it.**

The reason follows from this audit's own method. Every finding above reads frozen
numbers from the six development datasets — the defect evidence in §4, the
Package A decomposition, the musique semantic finding, and the seed-noise
estimates that set the v2 Pareto tolerance. That is legitimate diagnostic use,
but it means any v2 result on those six can be challenged as shaped by what we
already looked at. **F has never been opened, so a v2 result there is the only
number in this project immune to that objection.**

QLS-v1's role is now **diagnostic and historical evidence from the six existing
datasets**. The frozen v1 and GNN results remain evaluation baselines on Packages
A–E exactly as they stand, and nothing in this document alters them.

Governed by [`QLS_V2_DEVELOPMENT_PROTOCOL.md`](QLS_V2_DEVELOPMENT_PROTOCOL.md)
§4 Phase 8 and §9.

---

## 9. A naming hazard that must be handled before submission

The internal frozen code key is `sa_mlp`. A published method named **SA-MLP**
(arXiv 2210.09609) learns from a **GNN teacher via distillation** — precisely the
approach QLS-v2 forbids. The collision is purely nominal, but it invites exactly
the misreading the new thesis must avoid ("the MLP merely compressed a GNN").

- The code key `sa_mlp` **must not be renamed**: doing so would invalidate frozen
  hashes, configs and results.
- The publication name remains **QLS-MLP**, and the paper must explicitly
  distinguish it from SA-MLP-style distillation, stating that QLS uses no GNN
  teacher, no GNN-derived supervision and no learned propagation at any stage.

---

## 10. Summary

| ID | Defect | Kind | Scope in frozen evidence |
|---|---|---|---|
| W1 | Min seed distance, collapsed and bucketed | information loss | all datasets; costliest where seed geometry matters |
| W2 | Seed support counts edges, not distinct seeds | information loss | all datasets; likely the largest single omission |
| W3 | Path features are walks, not independent evidence | information loss | all datasets; also a cost driver |
| W4 | Graph provenance flattened | information loss | shared with the GNN comparator |
| W5 | Max-normalization compresses the mid-range | information loss | worsens with candidate budget |
| W6 | Unbounded tail in `query_local_summary` | cost | all six; p95/p50 = 6.7–8.7× |

**Preserved narrower findings, not to be generalized:**
context-related GNN separation is primarily 2wiki + hotpotqa; and learned message
passing does **not** obtain its advantage from similarity-only topology.

Remedies, the candidate feature contract, the tiny learner and the staged program
are in [`QLS_V2_FEATURE_CATALOG.md`](QLS_V2_FEATURE_CATALOG.md),
[`QLS_V2_DESIGN.md`](QLS_V2_DESIGN.md),
[`QLS_V2_DEVELOPMENT_PROTOCOL.md`](QLS_V2_DEVELOPMENT_PROTOCOL.md) and
[`QLS_V2_SYSTEMS_PLAN.md`](QLS_V2_SYSTEMS_PLAN.md).
