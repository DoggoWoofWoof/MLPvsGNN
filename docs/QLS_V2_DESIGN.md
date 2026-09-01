# QLS-v2 Design

**Status:** `PROPOSED_DESIGN_NOT_FROZEN_NOT_TRAINED`
**Date:** 2026-09-02
**Depends on:** [`QLS_V1_WEAKNESS_AUDIT.md`](QLS_V1_WEAKNESS_AUDIT.md),
[`QLS_V2_FEATURE_CATALOG.md`](QLS_V2_FEATURE_CATALOG.md)
**Gated by:** [`QLS_V2_DEVELOPMENT_PROTOCOL.md`](QLS_V2_DEVELOPMENT_PROTOCOL.md)

**Supersedes the 2026-09-02 initial draft**, which proposed a ~210K-parameter
model with a learned gate and provenance-typed channels. Three of its proposals
are now explicitly withdrawn (§9).

---

## 0. The thesis

> Retrieval does not require a learned neighborhood propagation mechanism. It
> requires a **compact description of how each candidate relates to the query's
> retrieved evidence.**

Formally, for candidate `d`, seed set `S_q` and graph `G`, we want
`φ(q, d, S_q, G)` to contain the relevant topology, and then

```
score(q,d) = f_θ( retrieval_features, φ )
```

where `f_θ` is an extremely small MLP.

**The scientific challenge is therefore: find the smallest sufficient structural
feature set.** That is the contribution. The architecture is downstream of it and
is deliberately the least interesting part of this design.

---

## 1. The absolute constraint: no GNN anywhere in the method

QLS-v2 must not use a GNN in **any** part of the method:

```
no GNN teacher
no GNN distillation
no GNN hidden representations
no GNN-generated labels
no GNN-generated residual targets
no learned message passing during training
no learned message passing during inference
```

The existing frozen GNN results remain **evaluation baselines only**. The new
model is developed independently of them.

**Why this is a strengthening, not a limitation.** If any GNN-derived supervision
entered the method, a reviewer could reasonably say the MLP merely *compressed* a
GNN — which would collapse the thesis into a distillation result. Distillation
methods (GLNN, TINED, and the similarly-named **SA-MLP**, arXiv 2210.09609) learn
from a GNN teacher. We cite them as related work and state a **stricter**
objective: no GNN teacher, no GNN-generated supervision, no learned propagation
at training or inference.

> **Naming hazard.** Our internal frozen code key `sa_mlp` collides nominally
> with the published SA-MLP distillation method. The code key **must not be
> renamed** (frozen hashes and configs depend on it); the publication name is
> **QLS-MLP**, and the paper must explicitly disambiguate. See audit §8.

**Non-negotiables inherited unchanged:**

| Constraint | Value |
|---|---|
| Candidate contract | Unchanged from Packages A–E |
| Frozen A–E test metrics | Never a tuning target |
| Dataset identity as a feature | Prohibited (universality) |
| Direction features | Only where the source graph genuinely carries direction |
| CRAG | Strictly read-only |

---

## 2. Why the fix is representational, not architectural

The frozen A3 result is the load-bearing evidence, and it is already ours.

A3 is a **19-parameter bias-free linear scorer** with no embeddings and no
adjacency. It recovers **51.8% (WebQSP), 47.9% (HotpotQA), 65.6% (MetaQA)** of
the selected-RRF → QLS R@5 gap, and matches QLS outright on 2wiki and squad.

Two consequences:

1. **Roughly half the structural value is already linear in v1's features.** The
   marginal return on capacity is low; the marginal return on better features is
   where the remaining gap lives.
2. **A3's largest shortfalls (musique −11.77, webqsp −11.16) are on datasets
   where v1's features are most collapsed** — precisely the W1/W2 losses.

Add the parameter arithmetic: 46% of v1's 213,506 parameters are the 768→64 q/x
projection. If **scalar** retrieval features suffice in place of projected
embeddings, that mass disappears rather than being re-spent.

So the strategy is: **feature frontier first, smallest sufficient learner
second.**

---

## 3. The candidate feature contract

33 candidate scalars in five groups, defined with formulas, costs, cacheability
and failure modes in [`QLS_V2_FEATURE_CATALOG.md`](QLS_V2_FEATURE_CATALOG.md):

| Group | n | Targets | Note |
|---|---:|---|---|
| A Retrieval prior | 9 | — | graph-free; rung R0 |
| B Seed geometry | 7 | W1 | distance *distribution*, not the minimum |
| C Independent support & path diversity | 7 | W2, W3 | **highest-expected-value group** |
| D Fixed-depth diffusion | 6 | W6, W5 | bounded work, no convergence loop |
| E Cheap candidate topology | 4 | W5 | induced-graph position |

**Every feature must correspond to a specific information loss identified in
QLS-v1.** That constraint is what keeps this from becoming a fifty-feature
kitchen sink, and it is why the catalog excludes motif censuses, exact
disjoint-path computation and learned structural embeddings.

Expected survivors: **15–20 features.** The elimination is a result, not
housekeeping.

---

## 4. Bounded seed-bitset computation

The central implementation idea. Because `|S_q| ≤ 10` (audit §1), assign each
seed one bit:

```
seed0 = 0000000001
seed1 = 0000000010
seed2 = 0000000100
...
```

Each candidate holds a bitmask of which seeds reach it. Propagate by OR for a
fixed depth:

```
hop 0 : mask[s] = bit(s) for s in S_q
  |
hop 1 : mask1[v] |= mask0[u]  for each edge u->v
  |
hop 2 : mask2[v] |= mask1[u]
  |
hop 3 : mask3[v] |= mask2[u]
```

Then `popcount(mask_h[d])` is the number of seeds reaching `d` within `h` hops,
and from **one bounded traversal** we derive simultaneously:

```
minimum distance          (first h where popcount > 0)
unique supporting seeds   (popcount)
hop-specific seed support (popcount per h)
reachable seed fraction   (popcount / |S_q|)
branch / predecessor stats (counted during the same passes)
```

That is Groups B and C — 14 features — from **3 edge passes**, versus v1's ~8
non-PPR passes producing a strictly weaker summary.

**This is the design's best property: v2 becomes cheaper and more informative at
the same time.** The information v1 destroys (which seeds, not how many edges) is
exactly the information the bitmask carries for free.

**Sizing.** One 16-bit word per induced node covers `|S_q| ≤ 10` with room to
spare; at budget 400 that is ≤ 800 bytes per query. A `ceil(S/64)`-word
generalization is specified for portability but **is not needed by any dataset in
this study**, and will not be implemented until a dataset requires it.

---

## 5. Bounded diffusion replaces iterative PPR

v1 runs 8 PPR power iterations per query regardless of graph size or convergence.
Replace with fixed-depth diffusion `z_h = P z_{h-1}`, `z_0 = s`, exposing
`z_1, z_2, z_3`; or truncated PPR `π̃ = (1−α) Σ_{h=0..H} α^h z_h` with fixed
`H = 2` or `3`.

**Deterministic cost. No convergence loop. No pathological tail.**

The systems plan compares `v1 iterative PPR / H=1 / H=2 / H=3 / H=3 truncated`
on effectiveness and p50/p95/p99. This follows the SGC/SIGN principle — fixed
graph operators instead of recursive learned propagation — with our operator
query-conditioned (seeded at `S_q`) rather than a global filter.

---

## 6. Correct normalization (W5)

**Do not repeat the false claim that v1 uses raw features.** It already
normalizes. The defect is specifically **max**-normalization, which is
outlier-dominated:

```
degrees:  2       3        4      5        6        800
max-norm: 0.0025  0.00375  0.005  0.00625  0.0075   1.0
```

Every candidate ranking must actually separate is compressed into an
indistinguishable band. Replacements, by feature kind:

| Kind | Encoding |
|---|---|
| Counts | `log1p(raw)` **and** within-query percentile |
| Ranks | `rank / pool_size` |
| Distances | discrete `{0,1,2,3,>3}` — used directly, not scaled |
| Support | `support_count / \|S_q\|` — naturally bounded |
| Diffusion | percentile **and** log mass |

No dataset identity, no dataset-specific normalization constants.

---

## 7. Provenance stays an ablation, not a feature

Package B shows relational topology beats similarity-only topology, and that the
GNN's advantage exists **only** when relational edges are present. That motivates
typed features — but explicit provenance channels create a comparator-fairness
complication that is not worth paying for yet.

**Decision: no provenance typing in the headline architecture.** Instead run it
as a *graph* ablation using adjacency information available to all methods:

```
structural / native graph only
kNN only
combined
```

If simply **removing kNN edges** improves accuracy, latency and memory
simultaneously, that is a Pareto win requiring **no new feature at all** — and
the frozen Package B evidence (the GNN never significantly beats QLS on
`knn_only` anywhere) suggests it is worth checking first. Typed provenance
features may be promoted later only if demonstrably necessary *and* the fairness
protocol is resolved in advance.

---

## 8. The tiny learner

Architecture philosophy from LINKX: represent topology and retrieval signal as
**separate modalities** and combine them once, rather than repeatedly mixing them
through propagation. Ours can be far smaller because we never feed an adjacency
row — only ~33 scalars.

```
        Retrieval features (9)          Structural features (~24)
                 |                                |
             Linear w                        Linear w
               GELU                            GELU
                 |                                |
                 hr                               hs
                 |                                |
                 +----------------+---------------+
                                  |
                        [ hr , hs , hr (*) hs ]
                                  |
                              Linear w
                                GELU
                              Linear 1
                                  |
                                  +  <---- linear RRF/structure skip
                                  |
                                SCORE
```

**Linear skip (from LINKX and from our own A3 result).** Do not force the MLP to
relearn the strong simple baseline:

```
score = score_linear + Δ_nonlinear
```

where `score_linear` is a linear scorer over RRF and the strongest simple
structural features, and the MLP learns only the interactions the linear ranker
misses. A3 proves the linear part is already worth ~half the structural gap; this
makes that explicit and separately measurable, letting the paper decompose

```
retrieval signal  +  explicit structural signal  +  nonlinear interaction
```

instead of conflating all three.

**Parameter count, exactly** (9 retrieval + 24 structural + 33-term linear skip):

| width `w` | branches | interaction | output | skip | **total** | vs GNN 213,568 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 560 | 784 | 17 | 34 | **1,395** | **153× fewer** |
| 24 | 840 | 1,752 | 25 | 34 | **2,651** | **81× fewer** |
| 32 | 1,120 | 3,104 | 33 | 34 | **4,291** | **50× fewer** |

Thousands of parameters, not hundreds of thousands — and that is *before*
counting the GNN's edge-index storage, layer activations and autograd through
graph operations, none of which v2 has at all.

**Learner ladder, simplest first.** Escalate only if validation demands it:

```
linear
tiny single MLP
two-branch residual MLP          <- primary candidate
two-branch + explicit crosses
```

**No gating, attention, or transformer.** This reverses the initial draft's
proposal (§9). A gate adds parameters, interpretability cost, and an opportunity
for dataset-specific routing that would undermine the universality claim. Add one
only if validation shows a clear regime problem the residual model cannot handle.

**The final architecture should be the smallest point on the Pareto frontier, not
the architecture with every clever mechanism.**

---

## 9. Withdrawn from the initial draft

| Withdrawn | Reason |
|---|---|
| **Gated residual interaction head** | Adds parameters and dataset-specific routing risk; try the two-branch residual first. Reinstate only on validation evidence. |
| **Provenance-typed channels in the headline model** (old C6) | Comparator-fairness complication; demoted to a graph ablation (§7). |
| **Exact/greedy disjoint-path counting** (old C3.disjoint) | Expensive and unbounded — exactly the property W6 requires removing. Replaced by bounded predecessor/branch statistics. |
| **~210K parameter budget as the target** | The budget is now "as small as suffices" (~1.4K–4.3K), not "≤ the GNN". |
| **Exact PPR retained as the default** | Replaced by bounded fixed-depth diffusion (§5). |

---

## 10. Scalar-only is a hypothesis to test, not an assumption

Before designing anything larger, explicitly compare:

```
scalar retrieval + structural features only
            versus
scalar features + projected semantic embeddings (768 -> 64)
```

Prefer scalar-only **if it is within the effectiveness frontier**, because it
improves parameters, training memory, training time, inference FLOPs and
portability simultaneously. If scalar-only + topology reaches GNN-level results,
the paper can state something considerably stronger than a latency win:

> Graph-aware retrieval ranking does not require high-dimensional candidate
> embedding transformations inside the ranker at all.

If it does **not** hold, that is reported, and the projected-embedding variant
becomes the method with its cost stated honestly.

---

## 11. Literature positioning

The related work now frames us cleanly. **These citations were supplied by the
project lead and have not been independently verified against the source papers;
every specific claim must be checked before it appears in a submission.**

| Work | Lesson for us |
|---|---|
| SGC | Learned graph convolution often collapses to fixed propagation + a simple learner |
| SIGN | Multi-scale fixed graph operators can be precomputed and combined cheaply |
| BUDDY | Explicit low-order structural statistics/sketches capture what message passing struggles to infer, and precompute well — the closest conceptual precedent |
| LINKX | Topology and features as separate modalities, combined late, without recursive aggregation |
| Graph-MLP | Structural supervision can train a pure MLP with no adjacency at inference |
| GSSC | Sparsified structural supervision gives MLPs graph awareness without message passing |
| ES-MLP | Edges are not equally useful; separating relevant structure speeds inference |
| GLNN / TINED / **SA-MLP** | **Contrast case** — these distill from a GNN teacher, which we forbid |

Our distinction from the distillation line is the sharpest one available: they
learn *from* a GNN; we never use one.

---

## 12. Out of scope

- Any change to the candidate contract, pools or hashes.
- Any learned message passing, or candidate-to-candidate propagation, at any stage.
- Separate "fast" and "accurate" models — there is **one** method; a cost/accuracy
  split would evade the Pareto claim rather than establish it.
- A utility predictor (deferred).
- Package F. Unopened; no design decision here may reference it.
- Any modification to Package E2, which continues untouched as the QLS-v1
  diagnosis.
