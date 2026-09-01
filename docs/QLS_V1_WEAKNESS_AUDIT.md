# QLS-v1 Weakness Audit (read-only diagnosis)

**Status:** `DIAGNOSTIC_ONLY_NO_FROZEN_RESULT_MODIFIED`
**Date:** 2026-09-02
**Method under audit:** QLS-MLP v1 (internal frozen code key `sa_mlp`)

## 0. What this document is, and what it is not

This audit re-reads the **already frozen** QLS-v1 implementation and the
**already frozen** Package A/B/C/D/E1 results in order to state, precisely and
with evidence, where QLS-v1 is weaker than the parameter-matched seed-aware GNN.

It is a diagnosis. It is explicitly **not**:

- a rewrite, retune, deletion, or reinterpretation of any frozen result;
- a selection of a favourable subset of results;
- a design document (that is `QLS_V2_DESIGN.md`);
- permission to train anything.

Every frozen number cited below is reproduced from the sealed artifacts as they
stand. Where this audit's decomposition is **finer** than what a frozen report
states, that is recorded as a *refinement of the same data*, and the frozen
report is left untouched. Section 5 states the case where this happens.

Package E2 (five-seed phase confirmation) is **running and untouched** by this
document. Package F is **unopened**; no statement here is derived from it.

### Reading discipline this audit obeys

QLS-v1 is not "worse than the GNN." Across the six-dataset frozen confirmation
the two are close, and QLS-v1 wins outright on several systems axes (§6). The
weaknesses below are **specific and localized**, and stating them narrowly is
what makes them actionable. A vague "QLS is weaker on graphs" would license an
unfalsifiable redesign; the scoped claims below can each be wrong.

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

PPR configuration is `damping: 0.85, iterations: 8`
([`configs/sa_mlp_screen.yaml`](../configs/sa_mlp_screen.yaml)).

**Correction to a common description of v1.** Features 4–9 are *already*
per-query max-normalized (`log1p(x) / query_max`, e.g.
[`structural_features.py:410`](../src/mp_retrieval/structural_features.py:410),
[`:442`](../src/mp_retrieval/structural_features.py:442)), and
`hub_degree_percentile` is already a percentile. So "QLS-v1 uses raw absolute
counts" is **false**. The real defect in v1's normalization is different and
narrower: max-normalization is *outlier-dominated*. One hub candidate with a
large count compresses every other candidate toward zero, destroying resolution
exactly among the mid-ranked candidates where ranking decisions are made. This
distinction matters, because it changes the v2 fix from "normalize" (already
done) to "use rank/percentile statistics that a single outlier cannot compress".

---

## 2. Weakness W1 — QLS-v1 under-expresses genuine relational topology

### 2.1 Mechanism (code-level, verified)

Four independent collapses occur inside
[`_local_feature_chunk`](../src/mp_retrieval/structural_features.py:340):

**W1a — Distance is the minimum over seeds, then bucketed to 4 values.**
A multi-source BFS runs from *all* seeds jointly; each candidate keeps
`bucket = distance[node]`, clamped by
`if bucket < 0 or bucket >= 3: bucket = 3`
([`:384`](../src/mp_retrieval/structural_features.py:384)). Consequences:

- the *distribution* of distances over seeds is discarded — a candidate at
  distance 1 from one seed and 5 from four others is indistinguishable from a
  candidate at distance 1 from all five;
- distances 3, 4, 5, … and **unreachable** are one and the same value.

**W1b — Seed support is an edge count, not a seed count.**
`seed_connections` increments per *edge* incident to a seed
([`:397`](../src/mp_retrieval/structural_features.py:397),
[`:400`](../src/mp_retrieval/structural_features.py:400)), both directions
summed into one scalar. Three edges to a single seed and one edge to each of
three seeds produce the identical feature value. Multi-seed *agreement* — the
signal that distinguishes a genuine multi-hop bridge from a locally popular
node — is therefore not representable in v1 at all.

**W1c — Path features are walk counts, aggregated over seeds.**
`paths_length_1/2/3` are computed by iterated propagation
`following[dst] += current[src]`
([`:434`](../src/mp_retrieval/structural_features.py:434)). These are **walks**,
so revisits are counted; there is no notion of path *redundancy*, *disjointness*
or *branching*, and no per-seed decomposition.

**W1d — Edge provenance is invisible.**
`topology_edges` is a single flat edge array. Native/relational edges and kNN
similarity edges are indistinguishable to every local feature. A candidate
supported by three real relation edges and one supported by three embedding
neighbours receive identical structural evidence.

The GNN is subject to W1d as well (it also consumes one untyped edge set — see
the fairness note in §7), but it is **not** subject to W1a–W1c, because it
re-derives its own per-neighbour aggregation instead of consuming a fixed
scalar summary.

### 2.2 Evidence

Package B (edge provenance, FROZEN, 24/24 conditions) varies **only** the edge
family under a fixed candidate contract. This isolates "ability to exploit
relational structure" from "general ranking ability". Recall@5, points,
mean over 5 seeds:

| dataset | QLS `symbolic_b` − `knn_only` | GNN `symbolic_b` − `knn_only` | **GNN extraction advantage** |
|---|---:|---:|---:|
| 2wiki_clean    | +2.02 | +4.05 | **+2.04** |
| webqsp         | +2.66 | +5.37 | **+2.71** |
| hotpotqa_clean | +4.15 | +4.60 | +0.45 |
| squad_clean    | −0.02 | +0.02 | +0.03 |
| musique_clean  | +0.75 | +0.53 | −0.21 |
| metaqa         | +6.76 | +6.62 | −0.14 |

Both models gain from relational edges. On **2wiki and webqsp the GNN converts
those same edges into 2.0–2.7 more recall points than QLS-v1 does.** That
difference is the measurable size of W1 on those datasets. On metaqa and
musique QLS-v1 extracts the relational signal *as well as or better than* the
GNN, so W1 is **not** a universal deficit.

The complementary observation is sharper still. GNN − QLS by family
(Holm-adjusted across datasets within family, `*` = p < 0.05):

| dataset | symbolic_b | knn_only |
|---|---:|---:|
| 2wiki_clean    | +1.51* (0.026) | −0.52 (0.217) |
| webqsp         | +1.53  (0.417) | −1.18 (0.538) |
| hotpotqa_clean | +0.24* (0.038) | −0.21 (0.139) |
| musique_clean  | +0.55  (0.417) | +0.77 (0.217) |
| squad_clean    | +0.04  (0.555) | +0.00 (1.000) |
| metaqa         | −0.14* (0.008) | +0.00 (1.000) |

**On a similarity-only graph the GNN never significantly beats QLS-v1 in any
dataset** (every `knn_only` p ≥ 0.139; the point estimate is negative in 4/6).
The GNN's advantage appears only once relational edges are present. This is the
strongest available statement of W1: QLS-v1's deficit is specifically a
*relational-topology-exploitation* deficit, not a general ranking deficit.

One caveat stated plainly: the family-to-family contrast (`symbolic_b` vs
`knn_only`) carries **no p-value**. The frozen paired test is QLS vs GNN *within*
a family; it is not a test of one family against another. The extraction-advantage
column is therefore a point estimate, and W1's magnitude on 2wiki/webqsp is
**not** significance-tested by any frozen artifact. Establishing it requires a
new development-set measurement (§8).

The quantity confound is separately rebutted by the frozen data: `full_union_c`
has the most edges of any family on every dataset, yet is *worse* than
`symbolic_b` for QLS on 2wiki (67.16 vs 68.13), hotpotqa (76.46 vs 77.69) and
metaqa (30.02 vs 30.44). More edges is not the mechanism; relational edges are.

---

## 3. Weakness W2 — QLS-v1 degrades as candidate context grows

### 3.1 Mechanism (hypothesis, mechanism-consistent, not yet directly measured)

As the candidate budget grows, the induced graph gets **larger but sparser**.
From Package C's structural context (per query, median):

| dataset | budget | candidates | directed edges | density | components |
|---|---:|---:|---:|---:|---:|
| 2wiki_clean | 50 → 400 | 50 → 364 | 60 → 662 | 0.0245 → 0.0050 | 34 → 196 |
| hotpotqa_clean | 50 → 400 | 50 → 353 | 104 → 848 | 0.0424 → 0.0069 | 25 → 154 |
| metaqa | 50 → 400 | 50 → 379 | 78 → 658 | 0.0318 → 0.0047 | 30 → 203 |
| webqsp | 50 → 400 | 50 → 347 | 318 → 2337 | 0.1298 → 0.0194 | 9.5 → 68 |

Density falls 4–6× on every dataset while the absolute number of connected
components grows 5–7×. Under W1a, a growing share of candidates therefore lands
in the single `distance_3_plus_or_unreachable` bucket with near-zero path counts
and near-zero PPR — a **degenerate structural signature** shared by many
candidates at once. For those candidates v1's local block is close to constant,
so ranking among them falls back to static and semantic features. The GNN
retains its own per-node transform and does not collapse the same way.

This is a mechanism *consistent with* the observed pattern. The direct
measurement — the per-query fraction of candidates whose local feature vector is
degenerate, and the recall difference restricted to those candidates — has
**not been computed**, and is specified as development work in §8 rather than
asserted here.

### 3.2 Evidence

Package C (candidate budgets 50/100/200/400, FROZEN, 24/24). GNN − QLS on
Recall@5, points, `*` = Holm-significant:

| dataset | 50 | 100 | 200 | 400 |
|---|---:|---:|---:|---:|
| 2wiki_clean    | +0.16 | +0.63 | +1.02* | +1.28* |
| hotpotqa_clean | +0.22 | +0.31 | +0.44* | +0.70* |
| musique_clean  | ns | ns | ns | ns |
| metaqa         | +0.09* | +0.10* | +0.11* | +0.02 |
| webqsp         | negative throughout (ns) | | | |
| squad_clean    | ≈0 (ns) | | | |

The GNN's advantage grows **monotonically** with budget on 2wiki (+0.16 →
+1.28) and hotpotqa (+0.22 → +0.70), crossing into Holm significance at budget
200 and staying there at 400.

**W2 is real but confined to 2wiki and hotpotqa.** On musique, webqsp and squad
there is no growing gap; on metaqa the gap is significant at small budgets and
vanishes at 400 — the opposite direction. Claiming a general
"QLS degrades with context" would misstate the frozen evidence.

Attainment (recall ÷ candidate ceiling) falls with budget for *both* models on
every dataset, so decline itself is not QLS-specific. What is QLS-specific is
the *differential*: on 2wiki attainment@5 falls 0.900 → 0.860 for QLS
(−4.0 pts) versus 0.903 → 0.876 for the GNN (−2.7 pts); on hotpotqa
0.874 → 0.829 (−4.5) versus 0.876 → 0.837 (−3.9). On webqsp the differential
runs the other way.

---

## 4. Weakness W3 — QLS-v1 is inefficient for cold queries

### 4.1 Mechanism

Per query, v1's local block performs roughly: 3 BFS edge passes + 1 seed-connection
pass + 1 common-neighbour pass + 3 path-propagation passes + 8 PPR iterations
≈ **16 full passes over the induced edge list**. The BFS frontier expansion and
the path-count propagation are near-identical operations over the same edge
array, executed separately. PPR runs a fixed 8 iterations regardless of induced
graph size or convergence.

### 4.2 Evidence — and a refinement of the frozen report

Package D (uncached online systems, FROZEN, 6/6) reports QLS slower than the
GNN in 10/12 cells **by mean latency**. That report stands and is not modified.

Decomposing the *same* frozen artifacts by percentile shows the deficit is not
uniform overhead. Batch 1, ms/query:

| dataset | QLS p50 | GNN p50 | QLS p95 | GNN p95 | QLS mean | GNN mean |
|---|---:|---:|---:|---:|---:|---:|
| 2wiki_clean    | **2.816** | 3.102 | 7.394 | 4.236 | 3.767 | 3.243 |
| webqsp         | **2.582** | 2.782 | 6.924 | 3.674 | 3.489 | 2.916 |
| metaqa         | **2.872** | 3.197 | 7.447 | 4.428 | 3.765 | 3.329 |
| musique_clean  | **3.009** | 3.017 | 7.558 | 4.151 | 3.919 | 3.136 |
| hotpotqa_clean | 2.953 | 2.604 | 7.519 | 4.012 | 3.998 | 2.886 |
| squad_clean    | 3.794 | 3.638 | 8.166 | 5.408 | 4.391 | 3.840 |

**QLS-v1's median is already better than the GNN's in 4/6 datasets.** The
deficit is entirely a tail: p95 is 1.5–1.9× worse in all six.

The tail is localized to a single stage. QLS stage p95/p50 ratios:

| stage | ratio range across the six datasets |
|---|---|
| `fusion_and_seed_ms` | 1.18 – 1.38 |
| `topology_induction_ms` | 1.26 – 1.84 |
| `gather_transfer_forward_topk_ms` | 1.15 – 1.46 |
| **`query_local_summary_ms`** | **6.68 – 8.69** |

`query_local_summary_ms` has p50 0.53–0.72 ms but p95 **4.60–4.79 ms** — and
that p95 is nearly identical across all six datasets despite their induced
graphs differing by more than an order of magnitude in edge count. This is the
signature of an unbounded per-query worst case in the local-feature computation,
not of dataset scale.

This changes what W3's fix must be: **bound the tail of one stage**, not cheapen
everything. Consequently the phrase "QLS is slower" overstates and mis-locates
the problem; the accurate statement is "QLS-v1 has a heavy tail in
`query_local_summary_ms`, and its median is competitive."

---

## 5. Relationship to the frozen reports

`docs/ONLINE_SYSTEMS_RESULTS.md` reports the mean-latency comparison and is
correct as written. §4.2 above is a **percentile decomposition of the same
frozen measurement artifacts**, not a different measurement and not a
correction. Both descriptions are true simultaneously: the mean is worse and the
median is better, because the distribution is right-skewed.

Per the standing rule that historical protocol and result definitions are not
rewritten, `ONLINE_SYSTEMS_RESULTS.md` is left exactly as frozen. Any future
paper text that quotes the mean must also quote the percentile split, because
reporting the mean alone would imply a uniform overhead that the data does not
show.

---

## 6. Dimensions on which QLS-v1 already matches or beats the GNN

Recording these is a requirement, not a courtesy: a v2 that regresses here has
not dominated anything.

- **GPU memory (incremental peak, batch 1).** QLS 4.42–5.32 MB on every dataset.
  GNN 4.75–67.22 MB, and it scales with graph size (hotpotqa 67.22, squad 28.73,
  webqsp 8.14). QLS-v1 already dominates, by more than 12× on hotpotqa.
- **Median cold latency.** Better than the GNN in 4/6 datasets (§4.2).
- **Similarity-only graphs.** The GNN never significantly beats QLS-v1 on
  `knn_only` in any dataset (§2.2).
- **Effectiveness on 3/6 datasets.** No significant GNN advantage at any
  candidate budget on musique, webqsp or squad; on metaqa `symbolic_b` the GNN
  is significantly *worse*.
- **Parameters.** QLS-v1 213,506 vs GNN 213,568 on 2wiki/metaqa/webqsp — v1 is
  already marginally under budget there (it is marginally over on
  musique/squad: 209,351 vs 209,280).

QLS-v1 is behind on: **throughput** (GNN wins all six: 250–347 vs 228–287
queries/s), **p95/p99 latency** (all six), **mean latency** (10/12 cells), and
**Recall@5 on 2wiki and hotpotqa at large candidate budgets**.

---

## 7. Fairness constraints that any fix inherits

- **Relation typing is symmetric or absent.** Both models currently consume one
  untyped edge set. If v2 receives relation-type or provenance channels, the GNN
  comparator must receive the equivalent information (a typed/relational GNN),
  or v2 must not receive it. Giving QLS richer topology and comparing against an
  untyped GNN would not be evidence.
- **Direction is used only where it exists.** Direction-derived features may be
  computed only on datasets whose source graph genuinely carries direction.
  Fabricating direction on undirected datasets is prohibited; those datasets must
  receive an explicit null/undirected treatment, recorded per dataset.
- **Parameter budget.** v2 trainable parameters ≤ the selected GNN's count on
  the same dataset: 213,568 (2wiki, metaqa, webqsp), 213,440 (hotpotqa),
  209,280 (musique, squad).
- **No online learned message passing at inference.** Non-negotiable; it is the
  claim being tested.
- **The candidate contract does not change.** Every weakness above is a *ranking*
  weakness measured under a fixed candidate pool. Expanding candidates would
  change the ceiling and invalidate comparison with A–E.

---

## 8. What this audit could not establish (deliberately not computed)

These require a new read-only development-set analysis and are **not** asserted
above:

1. **Query-level correlates of the QLS−GNN gap.** Whether the gap concentrates
   in queries with low connected-seed fraction, high path redundancy, diffuse PPR
   or high hub exposure. Needs a per-query join of the frozen query-metric arrays
   against inference-safe structural covariates.
2. **The degenerate-signature fraction** (§3.1) and the gap restricted to those
   candidates. This is the direct test of W2's proposed mechanism.
3. **Significance of the relational extraction advantage** (§2.2). No frozen
   test covers the family-vs-family contrast.
4. **Which of the ~16 edge passes produces the `query_local_summary_ms` tail.**
   §4.2 localizes the tail to the stage, not to the operation inside it.

All four are development-set measurements. None may be run against the frozen
six-dataset test metrics, and none may be used to select hyperparameters.

---

## 9. Summary

| ID | Weakness | Scope established by frozen evidence | Magnitude |
|---|---|---|---|
| W1 | Under-expresses relational topology (min-distance collapse, edge-count seed support, walk-based paths, provenance-blind) | 2wiki, webqsp; absent on metaqa/musique/squad | +2.0 to +2.7 pts of unextracted relational signal |
| W2 | Degrades as candidate context grows | 2wiki, hotpotqa only | GNN advantage +0.16 → +1.28 and +0.22 → +0.70 across budgets 50 → 400 |
| W3 | Cold-query inefficiency | All six datasets | p95 1.5–1.9× worse; localized to `query_local_summary_ms` (p95/p50 = 6.7–8.7×); median already better in 4/6 |

Proposed remedies, their costs, their fairness implications and the ablations
that would isolate each are in [`QLS_V2_DESIGN.md`](QLS_V2_DESIGN.md). The
selection discipline that prevents these frozen results from being used as a
tuning target is in
[`QLS_V2_DEVELOPMENT_PROTOCOL.md`](QLS_V2_DEVELOPMENT_PROTOCOL.md).
