# QLS-v2 Candidate Feature Catalog

**Status:** `CANDIDATE_CONTRACT_NOT_FROZEN_NOT_IMPLEMENTED`
**Date:** 2026-09-02
**Depends on:** [`QLS_V1_WEAKNESS_AUDIT.md`](QLS_V1_WEAKNESS_AUDIT.md)

> **This is a superset / audit catalog, not the default model.** Every feature
> below is *eligible* for testing. None is admitted by default. The model that
> actually gets built is the much smaller **primary staged frontier R0–R5**
> defined in
> [`QLS_V2_DEVELOPMENT_PROTOCOL.md`](QLS_V2_DEVELOPMENT_PROTOCOL.md) §1, which
> draws **12–15 structural dimensions** from this catalog and leaves the rest
> unadmitted. The catalog exists so that (a) every information loss identified in
> the v1 audit has a named, costed candidate answer, and (b) anything we do *not*
> use was declined explicitly rather than never considered.

The scientific deliverable is the *minimum sufficient* subset. Two frontiers run
against this catalog:

| Frontier | Varies | Rungs | Final target |
|---|---|---|---|
| **Structural** R0–R5 | query-local graph statistics | 6 | **12–15 dims** |
| **Semantic** S0–S3 | how `q` and `d` are compared | 4 | **0–1,536 params** |

They are crossed, not sequential — see the protocol. The expected outcome is that
**R4 × S2 or R4 × S3** wins and that roughly half of this catalog is never
admitted.

## 0. Notation and standing facts

| Symbol | Meaning | Range in this study |
|---|---|---|
| `q` | query | — |
| `d` | candidate node | — |
| `S_q` | retrieval seed set, `dense_top5 ∪ splade_top5`, label-free | **5 ≤ \|S_q\| ≤ 10** |
| `G_q = (V_q, E_q)` | frozen candidate-induced directed graph | 50–400 nodes |
| `N_q` | candidate pool size | 50, 100, 200, 400 |
| `s ⇝ d` | `d` reachable from seed `s` within the hop bound | bound `H = 3` |
| `rank_D(d)`, `rank_P(d)` | Dense / SPLADE rank | 1…200, or **missing** |

Three facts constrain every design decision below and are established in the
audit:

1. **`|S_q| ≤ 10`.** Per-seed statistics are cheap. A per-node seed bitmask fits
   in **one 16-bit word**; there is no multi-word fallback path to build.
2. **Dense and SPLADE rankings are top-200 arrays**, while candidate pools reach
   400. A candidate may therefore have **no rank** from one or both retrievers.
   Every rank-derived feature below specifies its missing-value encoding
   explicitly; a silent 0 or 1 would be a lie about a different condition.
3. **Embedding width is 768 and the v1 projections are bias-free.**
   `operator_models.py:34-35` declares both `node_projection` and
   `query_projection` as `nn.Linear(embedding_dim, hidden_dim, bias=False)`, and
   `parameter_matched_head_width` accounts them as
   `2 * embedding_dim * projection_dim`. With `projection_dim: 64` frozen in both
   `configs/sa_mlp_screen.yaml` and `configs/phase_confirmation.yaml`, the v1
   semantic path is exactly

   ```
   2 x 768 x 64 = 98,304 parameters = 46.0% of the 213,506-parameter model
   ```

   Group F replaces that dense projection with **two 768-vectors = 1,536
   parameters**. The saving is not an estimate; it is the identity
   `2DP / 2D = P`, so the ratio is exactly `projection_dim = 64x`.

**Universality rule.** No feature may use dataset identity, a dataset-specific
constant, or a tuned threshold. Every normalization is within-query. This is what
makes the leave-one-dataset-out transfer test in the protocol meaningful.

**Cost legend.** `build` = query-time construction cost.
`cacheable: static` = precomputable offline per node, query-independent.
`cacheable: no` = depends on `S_q`, must be computed per query.

---

## Group A — Retrieval prior (9 features)

Query-conditioned but **graph-free**. These exist to answer a question the paper
must answer honestly: how much of the result is topology at all? Group A alone is
rung R0 of the frontier.

| # | Name | Formula | Range |
|---|---|---|---|
| A1 | `dense_rank_pct` | `1 − rank_D(d)/200`, or `0` if missing | [0,1] |
| A2 | `splade_rank_pct` | `1 − rank_P(d)/200`, or `0` if missing | [0,1] |
| A3 | `rrf` | `1/(k+rank_D) + 1/(k+rank_P)`, `k=60`, missing term contributes 0 | [0, 2/(k+1)] |
| A4 | `dense_only` | `1` if `rank_D` present and `rank_P` missing | {0,1} |
| A5 | `splade_only` | `1` if `rank_P` present and `rank_D` missing | {0,1} |
| A6 | `both_retrievers` | `1` if both present | {0,1} |
| A7 | `rank_disagreement` | `\|dense_rank_pct − splade_rank_pct\|` | [0,1] |
| A8 | `best_rank_pct` | `max(dense_rank_pct, splade_rank_pct)` | [0,1] |
| A9 | `is_seed` | `1` if `d ∈ S_q` | {0,1} |

**Rationale.** A1–A2 give each retriever's opinion on a bounded, pool-size-free
scale. A3 is the fusion baseline the linear skip will reuse. A4–A6 make the
*missing-rank* condition explicit rather than encoding it as "bad rank" — the
distinction between "SPLADE ranked it 190th" and "SPLADE never retrieved it" is
real and v1 could not express it. A7 is the disagreement signal: candidates the
two retrievers disagree about are exactly where a reranker has room to act. A9
matters because seeds are themselves candidates and are structurally degenerate
(distance 0 to themselves).

**Information recovered vs v1:** v1 consumed two source-rank scalars in A3's
role; A4–A7 are new.

**Cost:** array lookups and arithmetic, `O(N_q)`. **build ≈ 0.01 ms.**
**Storage:** none beyond the existing frozen ranking arrays.
**Cacheable:** no (query-conditioned), but trivially cheap.

**Expected failure mode:** on datasets where one retriever is near-useless
(webqsp dense R@5 = 10.20), A1/A4 may be noise and could dilute. The frontier
will show this; do not pre-remove.

---

## Group B — Seed geometry (7 features)

Targets **W1**. Replaces the 4-way minimum-distance one-hot.

Let `dist(s,d)` be the hop distance from seed `s` to `d` in `G_q`, `∞` if
unreachable within `H=3`. Let `R = {s ∈ S_q : dist(s,d) ≤ 3}`.

| # | Name | Formula |
|---|---|---|
| B1 | `min_seed_distance` | `min_s dist(s,d)`, encoded as discrete `{0,1,2,3,4=unreachable}` |
| B2 | `reachable_seed_fraction` | `\|R\| / \|S_q\|` |
| B3 | `frac_seeds_at_1` | `\|{s : dist=1}\| / \|S_q\|` |
| B4 | `frac_seeds_within_2` | `\|{s : dist≤2}\| / \|S_q\|` |
| B5 | `frac_seeds_within_3` | `\|{s : dist≤3}\| / \|S_q\|` |
| B6 | `mean_reachable_distance` | `(1/\|R\|) Σ_{s∈R} dist(s,d)`, `0` if `R=∅` (paired with B2) |
| B7 | `unreachable_seed_fraction` | `1 − B2` |

**Rationale — the counterexample from the audit.** Candidate A (adjacent to seed
1, disconnected from the rest) and candidate B (adjacent to seed 1, two hops from
seeds 2/3/4) both yield `min_distance = 1` in v1. Here they separate cleanly:

```
             B1   B2    B3    B4    B7
candidate A   1   0.2   0.2   0.2   0.8
candidate B   1   0.8   0.2   0.8   0.2
```

**Information recovered vs v1:** the seed-distance *distribution*; the
reachable/unreachable distinction beyond 2 hops (v1 merges 3, 4, 5, ∞).

**Redundancy warning, stated in advance:** B2 and B7 are exactly complementary
(`B7 = 1 − B2`), and B5 ≈ B2 under `H=3`. This is deliberate — the frontier's
backward elimination is expected to remove two or three of B2/B5/B7. Listing them
separately lets the elimination make that call on evidence instead of assuming
which encoding a linear head prefers.

**Cost:** all seven derive from **one** bounded multi-source traversal (§ Group
C's bitset pass). No additional traversal. **build ≈ 0.15 ms** shared with C.
**Storage:** none. **Cacheable:** no.

**Expected failure mode:** on very dense graphs (webqsp, density 0.130 at budget
50) nearly everything is reachable within 3 hops, so B2–B5 saturate near 1 and
carry little signal. Expect B-group value to be **highest on sparse, fragmented
graphs** (2wiki, metaqa: 196 and 203 components at budget 400) and lowest on
webqsp — a testable prediction, registered here.

---

## Group C — Independent support and path diversity (9 features)

Targets **W2** and **W3**. This group carries the audit's strongest hypothesis.

Let `M_h(d)` be the bitmask of seeds reaching `d` within `h` hops.

| # | Name | Formula |
|---|---|---|
| C1 | `distinct_seeds_at_1` | `popcount(M_1(d)) / \|S_q\|` |
| C2 | `distinct_seeds_within_2` | `popcount(M_2(d)) / \|S_q\|` |
| C3 | `distinct_seeds_within_3` | `popcount(M_3(d)) / \|S_q\|` |
| C4 | `support_concentration` | `popcount(M_1(d)) / (#seed-incident edges to d)`, `0` if no edges |
| C5a | `rank_weighted_support_at_1` | `W(M_1(d)) / W(S_q)` — see the frozen weighting rule below |
| C5b | `rank_weighted_support_within_2` | `W(M_2(d)) / W(S_q)` |
| C5c | `rank_weighted_support_within_3` | `W(M_3(d)) / W(S_q)` |
| C6 | `unique_predecessor_count` | `\|{v : v→d, v on a shortest path from some seed}\| / N_q` |
| C7 | `shortest_path_multiplicity` | `log1p(# shortest paths from S_q to d)`, then within-query percentile |

### The rank-weighting rule — FROZEN before any result exists

Support from the rank-1 seed and support from the rank-5 seed are not the same
evidence, and C1–C3 cannot tell them apart. C5a–C5c fix that. The rule is fixed
**now**, on the argument below, and must not be revisited after seeing an R2
number.

For a seed `s ∈ S_q`, let

```
r(s) = min over retrievers that returned s of that retriever's rank
```

The seed contract (`dense_top_k: 5`, `splade_top_k: 5`) guarantees every seed is
in some retriever's top 5, so **`r(s) ∈ {1,2,3,4,5}`** always. Define

```
w(s)  = 1 / r(s)                          <-- pure reciprocal rank, NO constant
W(M)  = sum of w(s) over seeds s in mask M
C5x   = W(M_h(d)) / W(S_q)   in [0,1]
```

**Why `1/r` and not RRF's `1/(60+r)`.** The RRF constant exists to damp top-rank
dominance when fusing *full* rankings over thousands of documents. Here we weight
at most ten seeds, all inside the top five. At `k=60` the weights span
`1/61 … 1/65` — a ratio of **1.066** — so C5 would be a near-exact copy of C1–C3
and the rung would measure nothing. Pure reciprocal rank spans `1 … 1/5`, a ratio
of **5.0**, and introduces **no free constant at all**, which is the stronger
property for a universality claim. The alternative `1/(1+r)` was considered and
rejected because it adds a tunable constant to buy a smaller spread.

Normalizing by `W(S_q)` makes the feature invariant to `|S_q|`, to pool size, and
to how many seeds each retriever contributed.

**Cost: zero.** `W(M)` is not summed per candidate. Because `|S_q| ≤ 10`, the
whole function `M ↦ W(M)` is tabulated once per query in `2^|S_q| ≤ 1024` single
additions via `W[m] = W[m & (m-1)] + w[lowest_set_bit(m)]`, after which C5a–C5c
are **three array lookups per candidate**. See
[`QLS_V2_SYSTEMS_PLAN.md`](QLS_V2_SYSTEMS_PLAN.md) §3.

**Rationale.** This is the direct fix for the clearest bug in v1. Currently

```
seed1 ──┬─ x          versus        seed1 ─ x
        ├─ x                        seed2 ─ x
        └─ x                        seed3 ─ x
```

are **identical**. Under C1 they are `1/|S_q|` and `3/|S_q|`. C4 measures the
same contrast as a ratio: 1/3 versus 1.0.

C5a–C5c weight support by *how confident the retriever was in the supporting
seed*. A candidate reached only from the rank-5 seed and one reached only from
the rank-1 seed are identical under C1–C3; under C5 they differ by 5x. v1 could
not express this at all. Splitting the weighting across the same three hop
budgets as C1–C3 is deliberate: the R1 → R2 transition in the protocol then
isolates exactly one question — *does retrieval confidence in the supporting seed
carry information beyond the count of supporting seeds?* — and answers it with a
clean marginal-efficiency delta rather than a confound.

C6/C7 replace v1's walk counts. They ask "how many independent ways does the
retrieved evidence reach this candidate" rather than "how many walks exist",
which is the quantity that hubs and cycles inflate without bound.

**Information recovered vs v1:** independent seed support (entirely absent in
v1); support concentration; retriever-confidence-weighted support; branch
structure; shortest-path multiplicity in place of walk counts.

**Cost:** C1–C5c come free from the same bitset traversal as Group B — the mask
*is* the feature, and C5 adds only a per-query lookup table. C6/C7 need one
predecessor-counting pass during the same BFS.
**build ≈ 0.15 ms total for B + C**, versus v1's ~8 non-PPR edge passes.
**Storage:** one 16-bit word per induced node (≤ 800 bytes per query at budget
400). **Cacheable:** no.

**Expected failure mode:** C4 is undefined for candidates with no seed-incident
edges and is set to 0, which collides with "one seed, many edges" (also low). If
the frontier shows C4 unhelpful, this collision is the first thing to check
before concluding support concentration is uninformative.

> **Recommended to test first.** If any single feature justifies this project,
> the audit's evidence points to **C1/C2 (distinct seed support)**. It fixes the
> most clear-cut information loss, it is free given the traversal, and it is
> the feature v1 most conspicuously lacks.

---

## Group D — Fixed-depth diffusion (6 features)

Targets **W6** (cost) while preserving the diffusion signal. Replaces iterative
PPR.

Let `P` be the row-normalized adjacency of `G_q`, `s` the seed indicator vector.
Define `z_h = P z_{h-1}`, `z_0 = s`, and truncated PPR
`π̃ = (1−α) Σ_{h=0..H} α^h z_h` with fixed `H ∈ {1,2,3}`.

| # | Name | Formula |
|---|---|---|
| D1 | `diffusion_h1` | `log1p(z_1[d])` |
| D2 | `diffusion_h2` | `log1p(z_2[d])` |
| D3 | `diffusion_h3` | `log1p(z_3[d])` |
| D4 | `diffusion_pct_h1` | within-query percentile of `z_1[d]` |
| D5 | `diffusion_pct_h2` | within-query percentile of `z_2[d]` |
| D6 | `diffusion_pct_h3` | within-query percentile of `z_3[d]` |

**Rationale.** v1 runs 8 PPR power iterations regardless of graph size or
convergence — a fixed cost per query with an unbounded worst case in dense
queries, and the audit localizes the entire latency tail to this stage's
neighbourhood. Fixed-depth diffusion has **deterministic bounded work**: exactly
`H` sparse matrix-vector products, no convergence loop, no data-dependent
termination.

This follows the SGC/SIGN principle — use **fixed** graph operators rather than
recursive learned propagation — with the difference that our operator is
query-conditioned (seeded at `S_q`) rather than a global filter.

Both magnitude (D1–D3) and percentile (D4–D6) are listed because they fail
differently: magnitude preserves absolute mass, percentile is outlier-robust
(**W5**). The frontier decides; carrying both is the point.

**Information recovered vs v1:** per-hop decomposition (v1 collapses diffusion to
one scalar); outlier-robust percentile encoding.

**Cost:** `H` sparse mat-vecs, `O(H·|E_q|)`, fully deterministic. At `H=3` this
is 3 edge passes versus v1's 8. **build ≈ 0.10–0.25 ms, bounded.**
**Storage:** none. **Cacheable:** no.

**Expected failure mode:** truncation at `H=3` may lose signal on graphs whose
useful diffusion mass sits further out. The audit's evidence argues against this
worry — v1's own path features stop at length 3 and its distance bucket stops at
3 — but the `H` comparison in the systems plan tests it directly rather than
assuming.

---

## Group E — Cheap candidate topology (4 features)

Targets **W5**. Query-conditioned position within the induced graph, replacing
v1's globally-normalized static features.

| # | Name | Formula |
|---|---|---|
| E1 | `induced_degree_pct` | within-query percentile of `deg_{G_q}(d)` |
| E2 | `component_size_pct` | within-query percentile of `\|component(d)\|` |
| E3 | `seed_component_fraction` | fraction of `S_q` in `d`'s connected component |
| E4 | `global_hub_pct` | `hub_degree_percentile` from v1 (global graph) |

**Rationale.** The audit shows the induced graph fragments as budget grows
(components 34 → 196 on 2wiki; density falls 4–6× on every dataset). E2/E3 are
exactly the features that separate candidates which v1's collapsed encoding makes
identical, and their value should **increase with candidate budget** — a
registered, testable prediction.

E1 is the direct W5 fix: percentile rather than max-normalized degree. E4 is
retained from v1 unchanged as the one genuinely global signal, and is the only
feature in the catalog that is offline-cacheable.

**Information recovered vs v1:** induced-graph position (v1 has only global
degree); component structure (absent in v1); outlier-robust encoding.

**Cost:** one union-find over `E_q` (near-linear) plus two sorts.
**build ≈ 0.05 ms.** **Storage:** E4 is one float per node, offline.
**Cacheable:** E4 **static**; E1–E3 no.

**Expected failure mode:** on dense graphs a single giant component makes E2/E3
constant. Expect E-group value to track the same sparsity gradient as Group B.

---

## Group F — Semantic micro-branch (5 features, 0 or 1,536 parameters)

Targets the **largest single parameter mass in QLS-v1** and the finding that
motivates this entire group: on musique, structure contributes **nothing** and
the whole QLS lift is semantic (§3, prediction 6). A structure-only v2 would lose
~11 points there. Semantics are therefore **not optional**, and the question is
not *whether* to compare `q` and `d` but *how cheaply*.

The v1 answer is a learned `768 → 64` projection on both sides: **98,304
parameters**, 46% of the model, spent before a single structural feature is read.
This group asks whether that is necessary or merely conventional.

Let `q, d ∈ R^768` be the frozen query and candidate embeddings, `q̂ = q/‖q‖`,
`d̂ = d/‖d‖`. All five features are **scalars**.

| # | Name | Formula | Params | Rung |
|---|---|---|---:|---|
| F1 | `cosine_qd` | `⟨q̂, d̂⟩` | 0 | S1 |
| F2 | `dot_qd_pct` | within-query percentile of `⟨q, d⟩` | 0 | S2 |
| F3 | `mean_abs_diff` | `(1/768) Σ_i \|q_i − d_i\|` | 0 | S2 |
| F4 | `semantic_product` | `Σ_i w_i q_i d_i`, `w ∈ R^768` | **768** | S3 |
| F5 | `semantic_difference` | `Σ_i v_i \|q_i − d_i\|`, `v ∈ R^768` | **768** | S3 |

### The semantic frontier

| Rung | Semantic input | Learned semantic params | vs v1 |
|---|---|---:|---:|
| **S0** | none — Group A ranks and RRF only | 0 | −100% |
| **S1** | + F1 | 0 | −100% |
| **S2** | + F1, F2, F3 | 0 | −100% |
| **S3** | + F1–F5 | **1,536** | **−64.0x** |
| *v1* | learned `768→64` on `q` and `d` | 98,304 | — |

S0 is the honest floor: it asks how much of the result is semantic *at all*, the
same way R0 asks how much is topological at all. S3 is the interesting rung.

### F4/F5 in detail

`F4` is a **diagonal bilinear form**, `qᵀ diag(w) d`. It is the rank-768-diagonal
restriction of the general bilinear similarity `qᵀ W d` that the v1 projection
approximates at rank 64. `F5` is a **learned weighted L1**, which no projection in
v1 could express: `|q−d|` is not a bilinear function of `q` and `d`, so this is
new capacity, not only cheaper capacity.

**Initialization is not free choice.** Set `w_i = 1/768` and `v_i = 0`. At
initialization `F4` is then exactly the mean elementwise product (a monotone
function of `⟨q,d⟩`) and `F5` is identically zero and inert. **S3 therefore
starts as S2 plus one redundant channel and can only depart from it by learning.**
`v = 0` still trains: `∂F5/∂v_i = |q_i − d_i| ≠ 0`.

**Normalization.** `F4` and `F5` are standardized *within query*:
`F' = (F − μ_q)/(σ_q + ε)`, `ε = 1e−6`, over the candidate pool. This is
differentiable (unlike the percentile encoding used elsewhere, which would block
the gradient to `w` and `v`), dataset-independent, and pool-size invariant.

A consequence worth stating: standardization makes the **global scale of `w` and
`v` unidentifiable**. Only their *shape* is learned. F4/F5 therefore learn a
**per-dimension importance profile over the embedding space** — which of the 768
directions matter for retrieval relatedness — and nothing else. That is a
sharper and more interpretable object than a 64-dimensional projection, and it is
directly inspectable in the paper.

**Information recovered vs v1:** none lost that we can name — the projection is
rank-64 and this is rank-768-diagonal, so neither strictly contains the other.
The projection can mix dimensions; the diagonal cannot. The diagonal can weight
all 768 dimensions; the rank-64 projection cannot preserve more than 64
directions. **Which restriction costs more is exactly the S3-vs-v1 experiment.**
F5 adds an absolute-difference channel that v1 has no way to represent.

**Cost — parameters, compute, and the bandwidth that does not shrink.**

| | v1 projection | S3 (F4+F5) | ratio |
|---|---:|---:|---:|
| parameters | 98,304 | 1,536 | **64.0x** |
| MACs per query @ `N_q=400` | 19,709,952 | 614,400 | **32.1x** |
| embedding bytes read @ fp16 | 614,400 | 614,400 | **1.0x** |

The third row is the honest one. **S1–S3 all still read 768 floats per
candidate.** Compressing the projection removes parameters and arithmetic; it
removes **no embedding bandwidth whatsoever**. Only S0 does that. If uncached
post-retrieval latency turns out to be bandwidth-bound rather than compute-bound,
S3 will be no faster than v1 and its entire advantage is parameters, training
memory and training time — which are real, but are different axes. Phase 0
measures which regime holds before any claim is made.

**Storage:** the frozen 768-d embeddings, already present. **Cacheable:** the
per-node terms of F4 are not cacheable (they depend on `q`), but `w` and `v` are
query-independent weights of 3 KB each at fp32.

**Expected failure mode:** on webqsp, where dense retrieval R@5 is 10.20, the
embedding geometry is poor and every F-feature may be weak regardless of
parameterization — so the S-frontier should be judged on musique and hotpotqa,
where semantics demonstrably carry the result, and S0 should not be dismissed on
webqsp evidence alone.

---

## 1. Summary and cost model

| Group | Features | Targets | Query-time build | Cacheable |
|---|---:|---|---|---|
| A Retrieval prior | 9 | — (R0 baseline) | ~0.01 ms | no |
| B Seed geometry | 7 | W1 | shared with C | no |
| C Support & path diversity | 9 | W2, W3 | ~0.15 ms (B+C) | no |
| D Fixed-depth diffusion | 6 | W6, W5 | shared with B+C (fused) | no |
| E Candidate topology | 4 | W5 | ~0.05 ms | E4 static |
| F Semantic micro-branch | 5 | v1 projection mass | ~0.05 ms (compute-bound part) | `w`, `v` static |
| **Total catalog** | **40** | | **~0.3–0.5 ms projected** | |
| *of which admitted by the primary frontier* | *12–15 structural + 1–5 semantic* | | | |

Against v1's `query_local_summary` p50 of 0.53–0.72 ms and **p95 of 4.60–4.79
ms**, the projected build is a reduction at the median and — because every
operation above is bounded — the elimination of the tail. These are estimates
from operation counts, **not measurements**; Phase 0 of the protocol measures
them before any of it is believed.

## 2. What is deliberately excluded

Not in the catalog, and not to be added unless the simpler frontier demonstrably
fails:

```
exact disjoint-path / max-flow computation
triangle enumeration, motif census
expensive clustering coefficients
full relation-type embeddings
long path enumeration (H > 3)
exact or iterative-to-convergence PPR
learned structural embeddings
high-dimensional adjacency embeddings
```

Each threatens the exact advantage the thesis is trying to establish. The
governing principle:

> Do not imitate everything a GNN could theoretically represent. Identify what
> retrieval actually requires.

**Also excluded from the primary frontier: the seven v1 global static
features.** `log_out_degree`, `log_in_degree`, `log_total_degree`, `pagerank`,
`hub_degree_percentile`, `coreness`, `clustering_wedge_estimate` remain in the
audit catalog (E4 keeps one of them) but are **not rungs of R0–R5**. See
prediction 10 for the evidence and for the honest limit on that evidence.

**Also excluded from the headline model: edge-provenance typing (W4).** Package B
motivates it, but provenance channels create a comparator-fairness complication
that would need resolving in advance. It stays a *graph ablation*
(`structural-only` vs `kNN-only` vs `combined`), not a feature — see
[`QLS_V2_DESIGN.md`](QLS_V2_DESIGN.md) §7.

## 3. Registered predictions

Recorded before measurement so they can be wrong:

1. **C1/C2 (distinct seed support) will be the highest-value single addition**,
   and will help most on musique and webqsp where A3's linear model trails QLS by
   11.77 and 11.16 points.
2. **Group B and Group E value will correlate with graph sparsity** — largest on
   2wiki and metaqa, smallest on webqsp.
3. **Group E value will grow with candidate budget**, mirroring the 2wiki and
   hotpotqa budget effect.
4. **Fixed-depth diffusion at `H=2` or `H=3` will be accuracy-neutral versus
   iterative PPR** while removing the latency tail.
5. **The final structure vector will contain 12–15 dimensions, not 20+**, with
   B2/B5/B7 and D1–D3 or D4–D6 partially eliminated as redundant.
6. **Structural features will not close the gap on musique.** The frozen Package
   A decomposition shows musique's entire QLS lift is semantic: the seed-only
   MLP (embeddings, no structural features) reaches 80.08 versus QLS's 80.28,
   while A3 (structure, no embeddings) reaches 68.51 versus RRF's 69.24 —
   structure contributes **−0.73**. Expect every structural rung R1–R5 to be
   approximately flat on musique, and expect a scalar-only variant to lose
   ~11 points there. This is a property of the dataset, not a failure of the
   catalog, and it must be reported rather than averaged away.
   See [`QLS_V2_DESIGN.md`](QLS_V2_DESIGN.md) §10.
7. **S3 will match or beat the v1 projection on validation R@5 at 64x fewer
   semantic parameters.** The rank-64 projection's ability to mix dimensions is
   predicted to be worth less than the diagonal's ability to weight all 768,
   plus F5's absolute-difference channel that v1 cannot represent at all. If this
   is wrong — if the mixing matters — S2 vs S3 vs v1 will show it, and the honest
   report is that the projection was load-bearing.
8. **S3 will not reduce uncached p95 latency relative to S1.** Both read the same
   768 floats per candidate; only the arithmetic shrinks. We predict the semantic
   path is **bandwidth-bound, not compute-bound**, so the S-frontier's win shows
   up in parameters, peak training memory and training wall-time, and *not* in
   inference latency. Phase 0 tests this directly. If S3 does cut p95, the
   compute-bound hypothesis was right and the systems claim gets stronger than
   predicted here.
9. **R5 will be unnecessary.** Path diversity (C6/C7) and induced-component
   structure are predicted to add **less than the 0.20 R@5 admission threshold**
   on the six-dataset validation mean over R4, and so to be dropped by the
   frontier, leaving a 12-dimensional structure vector. The supporting evidence is that A2's
   `structural_summary` — which produced the only demonstrated fixed-structure
   wins, +5.20 R@5 on webqsp and +4.41 on metaqa — is built from the **ten
   query-local features only** (`structural_controls.py:120-141`: distance,
   path/connectivity, PPR), with **zero** static or component features. Every
   structural gain the project has actually demonstrated came from the R1–R4
   tier. R5 is included so the claim is tested rather than assumed.
10. **Global static features (v1's seven) will not re-enter.** They are catalog
    entries E4 and the audit tier only, deliberately outside R0–R5. Note this
    prediction is **weakly grounded**: A3 consumed all seven but its learned
    weights were not persisted (`outputs/p0_linear_rank_structure/*.json` stores
    metrics, not the 19 coefficients), so their individual contribution has never
    been isolated. This is recorded as an open question, not as settled evidence.
