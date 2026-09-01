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

40 candidate scalars in six groups, defined with formulas, costs, cacheability
and failure modes in [`QLS_V2_FEATURE_CATALOG.md`](QLS_V2_FEATURE_CATALOG.md):

| Group | n | Targets | Note |
|---|---:|---|---|
| A Retrieval prior | 9 | — | graph-free; rung R0 |
| B Seed geometry | 7 | W1 | distance *distribution*, not the minimum |
| C Independent support & path diversity | 9 | W2, W3 | **highest-expected-value group** |
| D Fixed-depth diffusion | 6 | W6, W5 | bounded work, no convergence loop |
| E Cheap candidate topology | 4 | W5 | induced-graph position |
| F Semantic micro-branch | 5 | v1's 98,304-parameter projection | 0 or 1,536 parameters |

> **That catalog is a superset and an audit record, not the model.** The model is
> built from the two staged frontiers below, which together admit **12–15
> structural dimensions and 0–5 semantic scalars** — not 40.

| Frontier | Question it answers | Rungs | Target |
|---|---|---|---|
| **Structural R0–R5** | how much query-local graph structure does ranking need? | 6 | **12–15 dims** |
| **Semantic S0–S3** | how cheaply can `q` and `d` be compared? | 4 | **0–1,536 params** |

They are **crossed, not sequential** — see
[`QLS_V2_DEVELOPMENT_PROTOCOL.md`](QLS_V2_DEVELOPMENT_PROTOCOL.md) §4 Phase 2.
Both semantic branches (scalar/light-semantic S0–S2, and compact-embedding-
interaction S3) are carried in parallel through early development; neither is
assumed to win, and musique is the reason neither can be dropped on principle.

**Every feature must correspond to a specific information loss identified in
QLS-v1.** That constraint is what keeps this from becoming a fifty-feature
kitchen sink, and it is why the catalog excludes motif censuses, exact
disjoint-path computation and learned structural embeddings.

Expected survivors: **12–15 structural dimensions**, with R5 predicted to be
dropped entirely. The elimination is a result, not housekeeping.

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
rank-weighted support     (one table lookup -- see below)
diffusion h1 / h2 / h3    (fused into the same edge iteration)
```

**The diffusion is fused into the same passes.** Mask propagation and diffusion
are both scatter-accumulate operations reading index `u` and writing index `v`
from the previous generation. They share one loop over `E_q`, so the boolean OR
and the float multiply-accumulate touch the same cache line. That collapses
`3 + 3` passes into **3**.

**Rank-weighted support costs nothing.** Because `|S_q| ≤ 10`, the whole function
`mask ↦ Σ_{s ∈ mask} w(s)` is tabulated once per query in `2^|S_q| ≤ 1024` single
additions via `W[m] = W[m & (m-1)] + w[ctz(m)]`. Every candidate then reads its
rank-weighted support with **one array lookup**, not a loop over seeds. This is
only possible because the seed contract bounds `|S_q|` at 10 — at 20 seeds the
table would be a megabyte and the trick would be worthless.

That is Groups B and C — 16 features — plus Group D, from **3 edge passes**,
versus v1's ~16 producing a strictly weaker summary.

**This is the design's best property: v2 becomes cheaper and more informative at
the same time.** The information v1 destroys (which seeds, not how many edges) is
exactly the information the bitmask carries for free.

**Exact complexity**, stated because boundedness is a design objective and not a
hoped-for side effect (see
[`QLS_V2_SYSTEMS_PLAN.md`](QLS_V2_SYSTEMS_PLAN.md) §3 for the full algorithm):

```
time    Theta(H*|E_q| + N_q + 2^|S_q|)  =  Theta(3|E_q| + N_q + 1024)
memory  <= 13.0 KB per query at N_q = 400, INDEPENDENT of |E_q|
```

Worst case, best case and average case are the same expression — no convergence
loop, no early exit, no data-dependent iteration count. At `N_q = 400` the
worst case is `3 x 159,600 = 478,800` edge operations against v1's `2,553,600`,
a **5.33x** reduction by counting alone.

**Sizing.** One 16-bit word per induced node covers `|S_q| ≤ 10` with room to
spare. A `ceil(S/64)`-word generalization is specified for portability but **is
not needed by any dataset in this study**, and will not be implemented until a
dataset requires it.

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

### 8.0 The universal architecture target

> **A tiny semantic/retrieval branch and a tiny structural branch, combined by
> one simple cross interaction, feeding a residual ranking MLP with a linear
> retrieval skip. Expected scale: low thousands of parameters.**

The claim being made is not "retrieval needs structure and not semantics." It is:

> Retrieval ranking needs a **compact semantic relation representation** and a
> **compact query-local structural representation**, and **neither requires
> learned message passing.**

musique is why the first half is stated as strongly as the second. There, the
seed-only MLP (embeddings, no structural features) reaches 80.08 against QLS's
80.28, while A3 (structure, no embeddings) reaches 68.51 against RRF's 69.24 —
structure contributes **−0.73**. A structure-only v2 would lose ~11 points on
that dataset. Semantics are load-bearing. §10 develops this.

What musique does **not** show is that the semantic representation must be a
98,304-parameter learned projection. That is the question §8.2 asks.

**Forbidden in the headline model,** unless a later validation phase explicitly
demonstrates that a simpler candidate fails: GNN teacher, distillation, learned
message passing, learned gating, attention, transformer blocks, relation-typed
channels. Each is a mechanism a reviewer can point at to say the MLP is merely
imitating a GNN. The whole value of this design is that there is nothing to point
at.

### 8.1 Structure

Architecture philosophy from LINKX: represent topology and retrieval/semantic
signal as **separate modalities** and combine them once, rather than repeatedly
mixing them through propagation. Ours can be far smaller because we never feed an
adjacency row — only scalars.

```
   Retrieval prior (9)                     Query-local structure (12-15)
   + semantic scalars (0-5)                        |
              |                                    |
          Linear w                             Linear w
            GELU                                 GELU
              |                                    |
              hr                                   hs
              |                                    |
              +------------------+-----------------+
                                 |
                       [ hr , hs , hr (*) hs ]        <- one cross, not a gate
                                 |
                             Linear w
                               GELU
                             Linear 1
                                 |
                                 +  <---- linear retrieval/structure skip
                                 |
                               SCORE
```

At **S3** the semantic scalars `semantic_product` and `semantic_difference` are
produced by two 768-vectors upstream of this diagram — 1,536 parameters, no
hidden layer, no projection. At S0–S2 there are no semantic parameters at all.

**Linear skip (from LINKX and from our own A3 result).** Do not force the MLP to
relearn the strong simple baseline:

```
score = score_linear + delta_nonlinear
```

A3 proves the linear part is already worth roughly half the structural gap; this
makes that explicit and separately measurable, letting the paper decompose

```
retrieval signal  +  semantic signal  +  explicit structural signal  +  interaction
```

instead of conflating them.

### 8.2 Exact parameter counts

`n_r = 9 + n_sem` retrieval-and-semantic inputs, `n_s` structural inputs:

```
branches      (n_r*w + w) + (n_s*w + w)
interaction    3w*w + w
output         w + 1
linear skip    n_r + n_s + 1
diagonal       1,536 at S3, else 0
```

Headline configurations, against the frozen `seed_aware_gnn` at **213,568**
parameters and QLS-v1 at **213,506**:

| config | `w` | branches | interaction | out | skip | diagonal | **total** | vs GNN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **S1 × R4** (12 struct) | 16 | 384 | 784 | 17 | 23 | 0 | **1,208** | **176.8×** |
| | 32 | 768 | 3,104 | 33 | 23 | 0 | **3,928** | 54.4× |
| **S2 × R4** | 16 | 416 | 784 | 17 | 25 | 0 | **1,242** | **172.0×** |
| | 32 | 832 | 3,104 | 33 | 25 | 0 | **3,994** | 53.5× |
| **S3 × R4** | 16 | 448 | 784 | 17 | 27 | 1,536 | **2,812** | **75.9×** |
| | 32 | 896 | 3,104 | 33 | 27 | 1,536 | **5,596** | 38.2× |
| **S3 × R5** (15 struct) | 16 | 496 | 784 | 17 | 30 | 1,536 | **2,863** | 74.6× |
| | 32 | 992 | 3,104 | 33 | 30 | 1,536 | **5,695** | 37.5× |

**The entire grid spans 1,208 to 5,695 parameters — 37× to 177× fewer than the
GNN, and always in the low thousands.** (R0 is omitted: with no structural
features the architecture degenerates to a single-branch MLP and the two-branch
count does not apply.)

And that is *before* counting what the GNN also carries and v2 does not at all:
edge-index storage, per-layer activations over the induced graph, and autograd
through graph operations — which is why the GNN's peak incremental GPU memory
reaches 67.22 MB on hotpotqa against v1's 5.32 MB.

### 8.3 The semantic branch is where the parameters actually are

At S3 the diagonal pair is **1,536 of 2,812** parameters — 55% of the whole
model. Everything structural, both branches, the interaction and the head
together cost 1,276.

That inverts v1's balance and is the clearest statement of the design:

| | v1 | v2 @ S3 × R4, `w=16` |
|---|---:|---:|
| semantic path | 98,304 (46.0%) | 1,536 (54.6%) |
| everything else | 115,202 (54.0%) | 1,276 (45.4%) |
| **total** | **213,506** | **2,812** |

The semantic comparison remains the dominant cost even after a 64× compression,
because comparing two 768-dimensional vectors is genuinely the expensive part of
retrieval ranking. The structural half — the part this paper is about — is
essentially free, which is precisely the point: **the graph information that
matters costs on the order of a thousand parameters and three passes over the
edge list.**

### 8.4 Learner ladder, simplest first

Escalate only if validation demands it:

```
linear
tiny single MLP
two-branch residual MLP          <- primary candidate
two-branch + explicit crosses
```

**No gating, attention, or transformer.** This reverses the initial draft's
proposal (§9). A gate adds parameters, interpretability cost, and an opportunity
for dataset-specific routing that would undermine the universality claim.

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

## 10. The semantic question, and why scalar-only is a branch rather than the plan

The claim is **not** "structure replaces semantics." It is:

> Retrieval ranking needs a compact semantic relation representation and a
> compact query-local structural representation. **Neither requires learned
> message passing**, and the semantic one does not require a large learned
> projection either.

Four positions are compared, not two:

```
S0-S2   scalar retrieval + structural features, ZERO semantic parameters
S3      + diagonal bilinear and weighted-L1 scalars      1,536 parameters
v1      + learned 768 -> 64 projection on q and d       98,304 parameters
GNN     the frozen comparator                          213,568 parameters
```

Prefer the cheapest rung **that is within the effectiveness frontier**, because
each step down improves parameters, training memory, training time, inference
FLOPs and portability simultaneously.

**The 64× is exact, not estimated.** The projection costs
`2 × embedding_dim × projection_dim`; the diagonal pair costs
`2 × embedding_dim`. The ratio is the identity `2DP / 2D = P = 64`. Both v1
projections are declared `bias=False`
([`operator_models.py:34-35`](../src/mp_retrieval/operator_models.py:34)), so
`2 × 768 × 64 = 98,304` is the exact figure — 46.0% of the 213,506-parameter
model.

**What we expect S3 to give up, and what it gains.** The rank-64 projection can
*mix* embedding dimensions; the diagonal cannot. The diagonal weights all 768
dimensions; the rank-64 projection cannot preserve more than 64 directions.
Neither strictly contains the other, so this is a real empirical question rather
than a compression with a known answer. `semantic_difference` adds an
absolute-difference channel that no bilinear projection can express at all.

**And a limit we state before measuring.** S1, S2, S3 and v1 all read the same
768 floats per candidate. Compressing the projection removes parameters and
arithmetic; it removes **no embedding bandwidth**. If the semantic path is
bandwidth-bound, S3's win is parameters, training memory and training time — not
inference latency. Registered as prediction 8 in the catalog and gated by Phase 0
in [`QLS_V2_SYSTEMS_PLAN.md`](QLS_V2_SYSTEMS_PLAN.md) §2.1.

If S0–S2 + topology reaches GNN-level results, the paper can state something
considerably stronger than a latency win:

> Graph-aware retrieval ranking does not require high-dimensional candidate
> embedding transformations inside the ranker at all.

The frozen data says that stronger claim will not hold everywhere. Which is
exactly why S3 exists as a parallel branch.

### The Package A decomposition says this will not hold everywhere

Package A already contains the control that isolates the two contributions.
**A3** has structural features but *no embeddings and no adjacency*;
**seed-only MLP** has embeddings and a seed indicator but *no structural
features*. R@5, frozen:

| dataset | RRF | A3 (structure, no emb.) | seed-only (emb., no structure) | QLS-MLP | structure lift | **embedding lift** |
|---|---:|---:|---:|---:|---:|---:|
| 2wiki_clean    | 68.48 | 68.57 | 65.83 | 68.40 | +0.09 | −2.65 |
| musique_clean  | 69.24 | 68.51 | **80.08** | 80.28 | −0.73 | **+10.84** |
| webqsp         | 10.20 | 22.21 | 29.26 | 33.37 | +12.01 | **+19.06** |
| hotpotqa_clean | 72.24 | 74.58 | 73.43 | 77.13 | +2.34 | +1.19 |
| squad_clean    | 89.31 | 89.50 | 89.31 | 89.23 | +0.19 | 0.00 |
| metaqa         | 13.75 | 24.48 | 23.25 | 30.11 | +10.73 | +9.50 |

> **On musique, structure contributes nothing (−0.73) and the entire QLS lift is
> the embeddings**: the seed-only MLP reaches 80.08 of QLS's 80.28 with no
> structural features at all. A scalar-only ranker would plausibly land near RRF
> (69.24) — an ~11-point loss on that dataset.

On webqsp both contribute and they are partly complementary (QLS 33.37 exceeds
either control alone). On 2wiki and squad neither contributes. Only on metaqa and
hotpotqa is structure the larger share.

**Consequences, adopted now rather than discovered in Phase 2:**

1. **Scalar-only is not the default.** It is rung R0's extension and a Pareto
   candidate, not the presumed answer. The projected-embedding variant is carried
   through the frontier in parallel, not as a fallback.
2. **Expect the R0 rung to be weak on musique**, and do not read that as a
   failure of the feature catalog — it is a property of the dataset.
3. **The honest form of the claim is conditional**: scalar-only suffices *where
   structure is what carries the signal* (metaqa, hotpotqa, webqsp-in-part), and
   does not where semantics carries it (musique). If that is the outcome, it is
   reported that way. **Do not hide dimensions on which the final method fails.**
4. This is also the strongest argument for the **two-branch** architecture (§8):
   the two modalities demonstrably dominate on different datasets, which is
   exactly the situation separate branches with late combination are for.

If a scalar-only model nonetheless matches the embedding variant everywhere, that
is a substantially stronger result than expected and should be reported as such —
but the design does not assume it.

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
