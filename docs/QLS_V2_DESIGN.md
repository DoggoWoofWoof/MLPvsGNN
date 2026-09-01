# QLS-v2 Design

**Status:** `PROPOSED_DESIGN_NOT_FROZEN_NOT_TRAINED`
**Date:** 2026-09-02
**Depends on:** [`QLS_V1_WEAKNESS_AUDIT.md`](QLS_V1_WEAKNESS_AUDIT.md)
**Gated by:** [`QLS_V2_DEVELOPMENT_PROTOCOL.md`](QLS_V2_DEVELOPMENT_PROTOCOL.md)

## 0. Objective and non-negotiables

**Objective.** Design a QLS-v2 that Pareto-dominates the parameter-matched
seed-aware GNN in retrieval effectiveness *and* systems cost, while performing
**no learned message passing at inference**.

**Non-negotiables carried from the audit (§7 there):**

| Constraint | Value |
|---|---|
| Learned message passing at inference | Prohibited |
| Candidate contract | Unchanged from Packages A–E |
| Trainable parameters | ≤ selected GNN on the same dataset |
| Relation typing | Symmetric between QLS-v2 and the GNN comparator, or absent from both |
| Direction features | Only on datasets whose source graph genuinely carries direction |
| Frozen A–E test metrics | Never a tuning target |

**Parameter budget, exactly.** Selected-GNN trainable counts: 213,568
(2wiki, metaqa, webqsp), 213,440 (hotpotqa), 209,280 (musique, squad). QLS-v1 is
213,506 / 209,351. On **musique and squad v1 is already 71 parameters over the
GNN**, so v2 cannot simply add features there — every added input dimension must
be paid for. This is a design constraint from the outset, not a late correction
(§4).

**What "dominate" requires.** Every axis below must be met; a v2 that wins
Recall@5 and loses p95 has not dominated. Targets (not claims):

- Recall@5 > GNN; Recall@20, MRR, FullCoverage@20 ≥ GNN
- p50, p95, p99 uncached latency < GNN; cached operator latency < GNN
- throughput > GNN; peak GPU memory ≤ GNN (v1 already wins this — see audit §6)
- parameters ≤ GNN

---

## 1. Design overview

Three feature families address W1, one addresses W2, and the architecture change
addresses the interaction that neither v1 nor a plain concatenation can express.
W3 is addressed almost entirely by the systems plan
([`QLS_V2_SYSTEMS_PLAN.md`](QLS_V2_SYSTEMS_PLAN.md)), because the audit showed
W3 is a tail-bounding problem rather than a feature-count problem.

| Change | Targets | Adds dims | Nature |
|---|---|---:|---|
| C1 Multi-scale seed geometry | W1a | ~6 | replaces min-distance one-hot |
| C2 Multi-seed support | W1b | ~4 | new |
| C3 Path structure & redundancy | W1c | ~5 | replaces walk counts |
| C4 Diffusion distribution | W1a/W2 | ~4 | replaces PPR scalar |
| C5 Candidate-local structure | W2 | ~4 | new |
| C6 Provenance-aware split | W1d | ×2 on C1–C4 subset | conditional, fairness-gated |
| C7 Rank/percentile transforms | W2 normalization | 0 (replaces) | transform |
| C8 Gated residual interaction head | W1+W2 | 0 features | architecture |
| C9 Compact subset selection | W3, parameters | negative | pruning |

C6 is **conditional** and may only ship together with a typed GNN comparator
(§5). C9 exists because C1–C5 cannot all be afforded within the parameter budget
on musique/squad.

---

## 2. Feature changes

Each entry gives: the change, why it addresses the weakness, its computational
cost, its fairness implication, and the ablation that isolates it. Development
selection rules and the confirmation rule are in §6–§7 and in the protocol
document.

### C1 — Multi-scale seed geometry (replaces `distance_0..3+`)

**Change.** Replace the 4-way one-hot of the *minimum* seed distance with a
summary of the *distribution* of per-seed distances: fraction of seeds at hop 1,
at hop 2, at hop 3; minimum; mean over reachable seeds; and fraction of seeds
from which the candidate is unreachable. Unreachability becomes an explicit
proportion rather than a bucket shared with distance-3.

**Why it addresses W1a.** The audit showed a candidate at distance 1 from one
seed and 5 from four others is currently identical to one at distance 1 from all
five. The distribution separates them, and separates "far" from "disconnected" —
the collapse that §3.1 of the audit identifies as growing with candidate budget.

**Cost.** Requires per-seed distances rather than a single multi-source BFS.
Naively this is O(seeds) BFS runs. Mitigation: a **bitset-parallel BFS** — one
traversal carrying an `S`-bit reachability mask per node (S = seed count, ≤ 64 in
one machine word for every dataset in this study), so per-seed hop membership is
recovered from three bitmask snapshots at hops 1/2/3 at ~1× the current BFS cost.
Net expected change: neutral to slightly cheaper than v1's three separate BFS
passes.

**Fairness.** Uses only seed identity and the same edge set the GNN sees. No new
information class. Symmetric.

**Ablation.** `v2 − C1` (restore v1's min-distance one-hot), holding everything
else fixed.

### C2 — Multi-seed support (replaces `seed_connections`)

**Change.** Split the single edge-count scalar into: number of *distinct seeds*
adjacent to the candidate; number of edges to seeds; the ratio of the two
(concentration); and the max share attributable to any single seed.

**Why it addresses W1b.** The audit's Candidate A/B case — three edges to one
seed vs one edge to each of three seeds — becomes representable. Distinct-seed
support is the signal that distinguishes a genuine multi-hop bridge from a node
that is merely adjacent to one popular seed.

**Cost.** One pass over seed-incident edges maintaining a per-candidate seed
bitmask (same word-width trick as C1). Popcount at the end. Effectively free
given C1's bitset traversal — they share the mask.

**Fairness.** Seed identity only; symmetric.

**Ablation.** `v2 − C2`.

### C3 — Path structure and redundancy (replaces `paths_length_1/2/3`)

**Change.** Keep length-1/2/3 counts, and add: distinct-seed path origin count
per length; a branching factor (paths ÷ distinct intermediate nodes); and a
**bounded disjoint-path approximation** — greedy vertex-disjoint path count
capped at 3, computed on the ≤3-hop induced subgraph.

**Why it addresses W1c.** v1 counts walks, so a single high-degree intermediate
inflates the count without indicating redundant connectivity. Disjointness and
branching distinguish "genuinely multiply-connected to the seed set" from
"connected once through a hub".

**Cost.** The highest-risk item in this design. The greedy disjoint-path count is
bounded by capping at 3 paths, restricting to ≤3 hops, and skipping the
computation entirely when the candidate has < 2 distinct seed-paths (which the
C2 counters already establish). Expected: touched for a minority of candidates.
**This is the change most likely to be cut by the systems plan**, and the plan
must profile it before it is admitted.

**Fairness.** Derived from the same untyped edge set; symmetric.

**Ablation.** `v2 − C3`, and separately `v2 − C3.disjoint` (keep branching, drop
the disjoint-path approximation) so that the expensive half can be dropped on
its own evidence.

### C4 — Diffusion distribution (replaces the PPR scalar)

**Change.** Retain PPR mass, but represent it as: within-query percentile rank;
log-ratio to the query's median candidate; the entropy of the candidate's
seed-wise PPR contributions; and the number of seeds contributing above a
threshold share.

**Why it addresses W1a and W2.** v1 keeps a single max-normalized scalar, which
the audit identified as outlier-compressed: one hub squashes every other
candidate toward zero, and that compression worsens as the candidate pool grows
(W2). A percentile is invariant to a single outlier. Seed-wise entropy adds the
concentration information that the aggregate mass discards.

**Cost.** Seed-wise contributions require either S-vector PPR (cost ×S) or a
**sketched approximation**: run PPR on a small random projection of the seed
indicator (k ≈ 4–8 columns) and estimate concentration from the sketch. The
systems plan must compare exact-PPR, truncated-PPR, push-based approximate PPR
and fixed-depth diffusion before this is fixed. Percentile/ratio transforms
themselves are a sort, essentially free relative to the diffusion.

**Fairness.** Same edge set, same seeds; symmetric.

**Ablation.** `v2 − C4`, plus a diffusion-variant sub-study owned by the systems
plan (accuracy-neutral variants chosen on cost alone).

### C5 — Candidate-local structure

**Change.** Add: degree percentile *within the induced candidate graph* (not the
global graph, which v1 already has); local clustering within the induced
subgraph; the size of the candidate's connected component relative to the query's
largest component; and an indicator for isolation.

**Why it addresses W2.** The audit shows the induced graph fragments as budget
grows (components 34 → 196 on 2wiki; density falls 4–6× on every dataset).
Component-relative features are exactly the signal that distinguishes candidates
that all look identical under v1's collapsed encoding, and their value should
*increase* with budget — the regime where W2 bites.

**Cost.** One union-find over the induced edge list (near-linear), plus a
triangle estimate reusing the existing wedge machinery. Small.

**Fairness.** Structure of the shared induced graph; symmetric.

**Ablation.** `v2 − C5`. Additionally, C5's effect must be reported **per
candidate budget**, since its motivation is budget-dependent.

### C6 — Provenance-aware structural channels (CONDITIONAL)

**Change.** Compute a subset of C1–C4 separately over native/relational edges and
over kNN similarity edges, instead of over the flattened union.

**Why it addresses W1d.** Package B is unambiguous that relational and
similarity edges are not interchangeable: QLS gains +0.75 to +6.76 points from
relational edges, and the GNN's advantage over QLS exists *only* when relational
edges are present. v1 cannot tell the two apart.

**Cost.** Roughly doubles the traversal work for the duplicated channels. Should
be applied to the cheapest subset only (C1 distances, C2 support), not to C3/C4.

**Fairness — the binding constraint.** This is the one change that risks an
unfair comparison. Provenance is a form of relation typing. **C6 may only be
admitted if the GNN comparator is simultaneously given the same provenance
channel** (e.g. an edge-type embedding or a relational/typed GNN variant), and
the comparison is then v2-with-C6 against typed-GNN. If a typed GNN comparator
is not run, C6 is **excluded from the final method**. This is recorded here so
the decision cannot be made later on the basis of which option looks better.

**Ablation.** `v2 − C6` is mandatory and must be reported whether or not C6 is
admitted, so readers can see the method's standing without any typing
information.

### C7 — Rank and percentile transforms (replaces max-normalization)

**Change.** Replace `log1p(x)/query_max` with within-query percentile rank and
log-ratio-to-median for count-like features; keep a log-magnitude channel where
absolute scale is meaningful.

**Why.** The audit corrected a common misconception: v1's features are already
per-query normalized, so "raw → relative" is not the fix. The actual defect is
that max-normalization is outlier-dominated — a single hub compresses the
mid-range where ranking decisions happen — and this worsens as the pool grows.
Percentiles are invariant to that outlier.

**Cost.** A sort per feature per query (n log n on ≤ 400 candidates). Negligible
next to the traversals.

**Fairness.** A monotone reparameterization of information v1 already has;
symmetric.

**Ablation.** `v2 − C7` (restore max-normalization).

---

## 3. C8 — Gated residual interaction head (architecture)

**Change.** Keep the v1 backbone (`q`, `x`, `q*x`, `|q−x|`, cosine, dot) and add a
gated residual block over a small set of **explicit crosses** between semantic
and structural channels:

```
semantic_strength x ppr_percentile
retriever_rank    x distinct_seed_support
min_distance      x path_diversity
native_support    x dense/SPLADE_disagreement
```

The gate is a learned sigmoid over the structural block that modulates the
semantic residual, so the model can learn *when* to trust structure rather than
adding a fixed structural contribution.

**Why it addresses W1 and W2.** Both weaknesses are conditional: relational
structure helps on 2wiki/webqsp/hotpotqa and not on squad; structural degeneracy
grows with budget. A purely additive concatenation must commit to one global
weighting. A gate lets the model down-weight structure exactly where the audit
shows it carries no signal (squad: every family within 0.1 pt), which is also
where v1 has nothing to lose.

**Cost.** Inference-time cost is a handful of elementwise products and one small
gate MLP — no traversal, no message passing. This is the cheapest change in the
design. It is not message passing: the crosses are between a candidate's own
features and the query's, never between candidates.

**Fairness.** No new input information; it is a function of features already
admitted. Symmetric by construction.

**Ablation.** `v2 − C8` (plain concatenation head with the same features), and
`v1 + C8` (v1 features with the gated head) — the second isolates how much of any
gain is architectural rather than informational, which is the honest decomposition
and the one most likely to be asked at review.

---

## 4. Parameter budget accounting

Adding ~27 feature dimensions at `projection_dim: 64` costs ≈ 1,700 parameters in
the input projection, plus ≈ 4–8K for the C8 gate depending on width. Against a
budget that is already 71 parameters *over* on musique/squad, v2 must therefore
**pay for new features by removing old ones and/or narrowing a layer**.

Planned payment, in priority order:

1. Drop v1 features that C1–C4 strictly subsume (`distance_0..3+`,
   `seed_connections`, raw `paths_length_*`, the PPR scalar): −9 dims.
2. C9 compact-subset pruning on development evidence (§6): target −5 to −10 more.
3. If still over: reduce `hidden_dim` for the gate block only, never for the
   shared trunk, and record the reduction.

**Rule:** the parameter count is checked *before* any confirmation run, and a v2
that exceeds its dataset's GNN count is not eligible for confirmation. Being
under budget is not a contribution and is not claimed as one.

---

## 5. Fairness summary

| Change | New information class? | Comparator requirement |
|---|---|---|
| C1, C2, C3, C4, C5, C7 | No — re-encodings of the shared untyped graph + seeds | None |
| C8 | No — function of admitted features | None |
| **C6** | **Yes — edge provenance/typing** | **Typed/relational GNN comparator, or C6 is excluded** |
| Direction-derived variants of C1/C3 | Only on genuinely directed sources | Per-dataset direction declaration; undirected datasets get an explicit null treatment |

The direction declaration must be written down per dataset **before** any
direction-derived feature is computed, sourced from
[`DATASET_GRAPH_PROVENANCE.md`](DATASET_GRAPH_PROVENANCE.md), and it may not be
revised after seeing a result.

---

## 6. Development-only selection rule (summary)

Full rule, including splits and the no-peeking mechanics, is in
[`QLS_V2_DEVELOPMENT_PROTOCOL.md`](QLS_V2_DEVELOPMENT_PROTOCOL.md). In summary:

- Every choice among C1–C9 — admit, drop, or configure — is made **solely** on
  development/validation data.
- The selection statistic is fixed in advance: **validation Recall@5, mean over
  development seeds**, with ties broken toward the cheaper variant, then toward
  fewer parameters.
- C9's compact subset is chosen by a pre-registered procedure (greedy backward
  elimination while validation Recall@5 stays within a pre-declared tolerance of
  the full set), not by inspecting which features look important after the fact.
- The frozen six-dataset test metrics from Packages A–E are **not** consulted at
  any point during development.

## 7. Frozen confirmation rule (summary)

- The complete v2 specification — features, architecture, hyperparameters, the
  C6 decision, and the per-dataset direction declaration — is committed and
  tagged **before** any confirmation run.
- Confirmation is evaluated **once**, on untouched confirmation data, over the
  registered seed set.
- Every ablation listed in §2–§3 is reported, including those that do not favour
  v2, and every Pareto axis in §0 is reported including those v2 loses.
- If v2 fails to dominate, that is the reported result. The method is not
  revised against confirmation outcomes.

---

## 8. Explicitly out of scope for v2

- Any change to the candidate contract, candidate pools, or candidate hashes.
- Any learned message passing, attention over candidates, or candidate-to-candidate
  propagation at inference.
- A separate "fast" and "accurate" model. There is **one** final method; a
  cost/accuracy split would evade the Pareto claim rather than establish it.
- A utility predictor (deferred; the audit's §8 measurements come first).
- Package F. Unopened, and no design decision here may reference it.
