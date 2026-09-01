# QLS-v2 Candidate Feature Catalog

**Status:** `CANDIDATE_CONTRACT_NOT_FROZEN_NOT_IMPLEMENTED`
**Date:** 2026-09-02
**Depends on:** [`QLS_V1_WEAKNESS_AUDIT.md`](QLS_V1_WEAKNESS_AUDIT.md)

These are **candidates for testing, not the final feature list.** The scientific
deliverable of this project is the *minimum sufficient* subset, discovered by the
staged frontier in
[`QLS_V2_DEVELOPMENT_PROTOCOL.md`](QLS_V2_DEVELOPMENT_PROTOCOL.md). The expected
outcome is that 15–20 of the 33 survive.

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
3. **Embedding width is 768.** The v1 q/x projection is 98,432 parameters, 46% of
   the model. The features below are **scalars**, so a scalar-only ranker deletes
   that mass rather than re-spending it.

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

## Group C — Independent support and path diversity (7 features)

Targets **W2** and **W3**. This group carries the audit's strongest hypothesis.

Let `M_h(d)` be the bitmask of seeds reaching `d` within `h` hops.

| # | Name | Formula |
|---|---|---|
| C1 | `distinct_seeds_at_1` | `popcount(M_1(d)) / \|S_q\|` |
| C2 | `distinct_seeds_within_2` | `popcount(M_2(d)) / \|S_q\|` |
| C3 | `distinct_seeds_within_3` | `popcount(M_3(d)) / \|S_q\|` |
| C4 | `support_concentration` | `popcount(M_1(d)) / (#seed-incident edges to d)`, `0` if no edges |
| C5 | `rank_weighted_support` | `Σ_{s ⇝ d} 1/(k + rank(s))`, `k=60`, normalized by `Σ_{s∈S_q} 1/(k+rank(s))` |
| C6 | `unique_predecessor_count` | `\|{v : v→d, v on a shortest path from some seed}\| / N_q` |
| C7 | `shortest_path_multiplicity` | `log1p(# shortest paths from S_q to d)`, then within-query percentile |

**Rationale.** This is the direct fix for the clearest bug in v1. Currently

```
seed1 ──┬─ x          versus        seed1 ─ x
        ├─ x                        seed2 ─ x
        └─ x                        seed3 ─ x
```

are **identical**. Under C1 they are `1/|S_q|` and `3/|S_q|`. C4 measures the
same contrast as a ratio: 1/3 versus 1.0.

C5 weights support by *how confident the retriever was in the supporting seed*.
A candidate supported by the dense-rank-1 seed is better evidenced than one
supported by the rank-5 seed, and v1 could not express this at all. The
normalization makes it pool-size and seed-count invariant.

C6/C7 replace v1's walk counts. They ask "how many independent ways does the
retrieved evidence reach this candidate" rather than "how many walks exist",
which is the quantity that hubs and cycles inflate without bound.

**Information recovered vs v1:** independent seed support (entirely absent in
v1); support concentration; retriever-confidence-weighted support; branch
structure; shortest-path multiplicity in place of walk counts.

**Cost:** C1–C5 come free from the same bitset traversal as Group B — the mask
*is* the feature. C6/C7 need one predecessor-counting pass during the same BFS.
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

## 1. Summary and cost model

| Group | Features | Targets | Query-time build | Cacheable |
|---|---:|---|---|---|
| A Retrieval prior | 9 | — (R0 baseline) | ~0.01 ms | no |
| B Seed geometry | 7 | W1 | shared with C | no |
| C Support & path diversity | 7 | W2, W3 | ~0.15 ms (B+C) | no |
| D Fixed-depth diffusion | 6 | W6, W5 | ~0.10–0.25 ms | no |
| E Candidate topology | 4 | W5 | ~0.05 ms | E4 static |
| **Total** | **33** | | **~0.3–0.5 ms projected** | |

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
5. **The final set will contain 15–20 features**, with B2/B5/B7 and D1–D3 or
   D4–D6 partially eliminated as redundant.
6. **Structural features will not close the gap on musique.** The frozen Package
   A decomposition shows musique's entire QLS lift is semantic: the seed-only
   MLP (embeddings, no structural features) reaches 80.08 versus QLS's 80.28,
   while A3 (structure, no embeddings) reaches 68.51 versus RRF's 69.24 —
   structure contributes **−0.73**. Expect every structural rung R1–R5 to be
   approximately flat on musique, and expect a scalar-only variant to lose
   ~11 points there. This is a property of the dataset, not a failure of the
   catalog, and it must be reported rather than averaged away.
   See [`QLS_V2_DESIGN.md`](QLS_V2_DESIGN.md) §10.
