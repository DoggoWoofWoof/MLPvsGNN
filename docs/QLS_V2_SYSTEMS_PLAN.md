# QLS-v2 Systems Plan

**Status:** `PROPOSED_PLAN_NOT_FROZEN_NOT_EXECUTED`
**Date:** 2026-09-02
**Depends on:** [`QLS_V1_WEAKNESS_AUDIT.md`](QLS_V1_WEAKNESS_AUDIT.md) §4,
[`QLS_V2_DESIGN.md`](QLS_V2_DESIGN.md)
**Gated by:** [`QLS_V2_DEVELOPMENT_PROTOCOL.md`](QLS_V2_DEVELOPMENT_PROTOCOL.md)

## 0. The systems problem, stated correctly

The frozen Package D report says QLS-v1 is slower than the GNN in 10/12 cells by
**mean** latency. Decomposing the same frozen artifacts by percentile (audit §4.2)
shows this is not a uniform overhead:

- QLS-v1's **median is already better than the GNN's in 4/6 datasets**;
- p95 is 1.5–1.9× worse in **all six**;
- the excess is localized to one stage, `query_local_summary_ms`, whose p95/p50
  ratio is **6.68–8.69** while every other stage sits at **1.15–1.84**;
- that stage's p95 is **4.60–4.79 ms on all six datasets**, nearly constant
  despite induced graphs differing by more than an order of magnitude in edge
  count.

A near-constant tail across wildly different graph sizes is the signature of an
**unbounded per-query worst case**, not of dataset scale. So the objective is not
"make QLS cheaper". It is:

> **Bound the tail of `query_local_summary_ms` without losing the median, while
> absorbing the added cost of the v2 features.**

Everything in this plan follows from that.

---

## 1. Quantified target and headroom

If `query_local_summary_ms` were bounded to the p95/p50 ratio of the *other*
stages (~1.5×), and no other stage regressed, projected total p95 would be:

| dataset | QLS-v1 p95 | stage p95 → bounded | projected p95 | GNN p95 | margin |
|---|---:|---:|---:|---:|---:|
| 2wiki_clean    | 7.394 | 4.629 → 0.799 | **3.564** | 4.236 | +0.672 |
| hotpotqa_clean | 7.519 | 4.628 → 0.847 | **3.738** | 4.012 | +0.275 |
| metaqa         | 7.447 | 4.600 → 0.821 | **3.668** | 4.428 | +0.760 |
| musique_clean  | 7.558 | 4.690 → 0.899 | **3.767** | 4.151 | +0.384 |
| squad_clean    | 8.166 | 4.791 → 1.076 | **4.450** | 5.408 | +0.958 |
| webqsp         | 6.924 | 4.648 → 0.820 | **3.096** | 3.674 | +0.578 |

**This is an arithmetic projection, not a measurement.** It assumes the tail can
be fully bounded, that no other stage regresses, and — importantly — that the v2
features add nothing. None of those is guaranteed; C3's disjoint-path
approximation in particular could reintroduce a tail. The table's purpose is to
establish that the p95 deficit is *plausibly closable* and to set the budget:

> **v2's entire feature-addition budget at p95 is the margin column: 0.275 ms in
> the worst case (hotpotqa).** If the new features cost more than that at p95,
> v2 does not dominate on latency no matter how well the tail is bounded.

That is a tight budget, and it is the reason the design's cost vetoes exist.
Throughput must clear a similar bar: the GNN currently wins on all six
(250–347 vs 228–287 queries/s).

---

## 2. Stage 1 — Feature-by-feature profiling (prerequisite, measurement D4)

Nothing is optimized before it is measured. The audit localizes the tail to a
*stage*; it does not identify which operation inside that stage produces it.

**Instrument** `_local_feature_chunk`
([`structural_features.py:340`](../src/mp_retrieval/structural_features.py:340))
per operation: the three BFS passes, the seed-connection pass, the
common-neighbour pass, the three path-propagation passes, and the 8 PPR
iterations — approximately 16 full passes over the induced edge list.

**Report per operation:** p50/p95/p99 wall time, and the correlation of each
operation's time with induced node count, edge count, seed count, and component
count.

**Decides:** whether the tail is (a) a small subset of queries with large induced
graphs, (b) PPR's fixed 8 iterations on the densest queries, (c) frontier blowup
in BFS, or (d) allocation/memory behaviour rather than graph work. Each implies a
different fix, and picking the fix before the measurement would be guessing.

This runs on `validation` only, per the development protocol.

---

## 3. Stage 2 — One fused multi-source traversal

**Change.** Replace the separate BFS passes, the seed-connection pass and the
path-propagation passes with **one** multi-source traversal that emits, in a
single sweep over the edge list per hop:

- per-seed hop membership (via an `S`-bit reachability mask per node — S ≤ 64 for
  every dataset here, so one machine word);
- distance distribution statistics (design C1);
- distinct-seed adjacency and edge counts (design C2);
- short-path counts and branching (design C3 cheap half).

**Why it addresses W3.** BFS frontier expansion and path-count propagation are
near-identical operations over the same array, currently executed separately
(`following[dst] += current[src]`,
[`structural_features.py:434`](../src/mp_retrieval/structural_features.py:434)).
Fusing them cuts edge passes from ~8 (excluding PPR) to ~3, and the bitmask
carries the per-seed information that v1 discards — so C1 and C2 arrive at
**lower** cost than v1's collapsed versions, not higher.

**Cost.** Expected 2–3× reduction in the non-PPR portion. Memory: one 64-bit word
per induced node, negligible against the 4.4–5.3 MB incremental GPU footprint.

**Risk.** Datasets with > 64 seeds would need multi-word masks. This must be
checked per dataset before implementation; if any dataset exceeds it, the mask
degrades to chunked passes and the cost claim must be re-derived for that dataset
rather than assumed.

**Fairness.** Pure implementation change. No information difference.

**Ablation.** Not an accuracy ablation — verified by an equivalence test that the
fused traversal reproduces v1's features bit-for-bit where they overlap.

---

## 4. Stage 3 — Diffusion variant study

**Change.** Compare four diffusion implementations at equal feature semantics:

| Variant | Description | Expected tail behaviour |
|---|---|---|
| V1 exact | current: 8 power iterations, damping 0.85 | unbounded in dense queries |
| V2 truncated | early termination on L1 residual < ε | bounded by ε, data-dependent |
| V3 push-based | Andersen-style local push with residual threshold | **work-bounded by construction** |
| V4 fixed-depth | 2–3 hop diffusion, no convergence | hard bound, lowest fidelity |

**Why it addresses W3.** V1's fixed 8 iterations pay full cost on every query
regardless of size or convergence. A push-based method's work is bounded by the
residual threshold rather than by graph size, which is precisely the property a
bounded tail requires.

**Selection rule — and it is a cost rule, not an accuracy rule.** Variants are
first screened for **accuracy neutrality** on validation (within 0.10 Recall@5
points of exact). Among neutral variants, the one with the lowest p95 wins. If no
variant is accuracy-neutral, the choice escalates to the ladder in the
development protocol §4 and is made on validation Recall@5 under the standard
admission threshold. This ordering is fixed now so that a slow-but-slightly-better
variant cannot be justified after the fact.

**Cost.** V3 is expected to dominate; V4 is the fallback if V3's constant factors
disappoint.

**Fairness.** Approximation quality affects only QLS. If an approximation is
adopted, the confirmation reports the exact-PPR variant's accuracy too, so no
reader can suspect the approximation was chosen for an accuracy artifact.

**Ablation.** All four variants reported with accuracy and p50/p95, on validation.

---

## 5. Stage 4 — QLS-Compact subset (design C9)

**Change.** Greedy backward elimination over the v2 feature set, removing the
feature whose removal costs least validation Recall@5, stopping at a cumulative
**0.30 point** tolerance (declared in the development protocol §5.5).

**Why it addresses W3 and the parameter budget.** Two independent pressures make
pruning mandatory rather than optional:

1. **Latency.** The p95 margin is 0.275 ms in the worst dataset (§1).
2. **Parameters.** QLS-v1 is already **71 parameters over** the GNN budget on
   musique and squad (209,351 vs 209,280). Every added input dimension must be
   paid for; v2 cannot simply accumulate features.

**Cost.** Pruning reduces both. The risk is that pruning removes the feature that
fixes W1 on 2wiki/webqsp — which is why elimination is scored on validation
Recall@5 across **all six** datasets, not on aggregate throughput.

**Fairness.** No implication.

**Ablation.** The elimination trace itself is the ablation and is reported in
full, including features removed and their individual costs.

**One model only.** There is no separate "QLS-Fast" and "QLS-Accurate". Shipping
two models would let each Pareto axis be claimed by a different system, which is
not dominance. The compact subset *is* the method.

---

## 6. Stage 5 — Cached-path and break-even accounting

The three latency regimes established in Package D remain distinct and must not
be conflated:

```
cached operator latency  !=  uncached / on-demand post-retrieval latency
                         !=  raw-query end-to-end latency
```

**Cache break-even** stays as defined: `build_ms <= repeats * (uncached_ms - cached_ms)`,
compute-only, storage excluded. v2 changes both `build_ms` (the fused traversal is
cheaper) and `uncached_ms`, so break-even must be **recomputed**, not inherited
from v1's report. Static per-node features (the 7 static features) remain
precomputable; all query-local features remain on-demand by definition.

**Reported for v2:** build cost, cached operator latency, uncached p50/p95/p99,
throughput, peak GPU memory, and the recomputed break-even repeat count — for
both v2 and the GNN comparator, on all six datasets.

---

## 7. Full Pareto reporting table

The confirmation must fill this table completely. Cells where v2 loses are
reported as losses.

| Axis | QLS-v1 (frozen) | GNN (frozen) | v2 target |
|---|---|---|---|
| Recall@5 | reference | GNN advantage significant in 2wiki, hotpotqa (large budgets), metaqa | **>** GNN |
| Recall@20 / MRR / FullCov@20 | reference | ~parity | **≥** GNN |
| Cached operator latency | reference | reference | **<** GNN |
| Uncached p50 | 2.582–3.794 ms | 2.604–3.638 ms | **<** GNN (v1 already wins 4/6) |
| Uncached p95 | 6.924–8.166 ms | 3.674–5.408 ms | **<** GNN (v1 loses 6/6) |
| Uncached p99 | 7.520–10.741 ms | 4.225–6.402 ms | **<** GNN (v1 loses 6/6) |
| Throughput | 228–287 q/s | 250–347 q/s | **>** GNN (v1 loses 6/6) |
| Peak incremental GPU memory | 4.42–5.32 MB | 4.75–67.22 MB | **≤** GNN (v1 already wins 6/6, by >12× on hotpotqa) |
| Trainable parameters | 213,506 / 209,351 | 213,568 / 213,440 / 209,280 | **≤** GNN per dataset |
| Learned message passing at inference | none | yes | none |

**Axes v1 already wins** (GPU memory; median latency in 4/6; similarity-only
effectiveness) are reported as such. A v2 that regresses them has not dominated,
and the plan treats them as constraints, not as slack.

---

## 8. Order of execution and gates

1. **D4 profiling** (§2) — no optimization before measurement.
2. **Fused traversal** (§3) with a bit-exactness equivalence test.
3. **Diffusion variant study** (§4) — accuracy-neutral screen, then cost.
4. **Feature ladder** — owned by the development protocol §4.
5. **QLS-Compact elimination** (§5).
6. **Break-even recomputation** (§6).
7. **Freeze**, then confirmation.

**Gate:** if §2 shows the tail is *not* dominated by graph traversal work — for
example if it is allocation behaviour — then §3 and §4 are re-planned before
implementation rather than executed on the assumption that they are the fix.

**Standing constraints.** Package E2 continues untouched to completion; no
profiling or development run may contend for its Modal capacity in a way that
perturbs it. Package F stays unopened. All frozen A–E artifacts, including
`ONLINE_SYSTEMS_RESULTS.md` and its mean-latency framing, remain exactly as
frozen; §0's percentile decomposition is a refinement of the same data and is
reported alongside the mean, never in place of it.
