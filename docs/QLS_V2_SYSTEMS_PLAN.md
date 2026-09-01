# QLS-v2 Systems Plan

**Status:** `PROPOSED_PLAN_NOT_FROZEN_NOT_EXECUTED`
**Date:** 2026-09-02
**Depends on:** [`QLS_V1_WEAKNESS_AUDIT.md`](QLS_V1_WEAKNESS_AUDIT.md) §4,
[`QLS_V2_FEATURE_CATALOG.md`](QLS_V2_FEATURE_CATALOG.md)
**Gated by:** [`QLS_V2_DEVELOPMENT_PROTOCOL.md`](QLS_V2_DEVELOPMENT_PROTOCOL.md)

---

## 0. The systems problem, stated correctly

Package D reports QLS-v1 slower than the GNN in 10/12 cells by **mean** latency.
Decomposing the same frozen artifacts by percentile:

- QLS-v1's **median is already better than the GNN's in 4/6 datasets**;
- p95 is 1.5–1.9× worse in **all six**;
- the excess is localized to `query_local_summary_ms`, whose p95/p50 ratio is
  **6.68–8.69** while every other stage sits at **1.15–1.84**;
- that stage's p95 is **4.60–4.79 ms on all six datasets** — nearly constant
  despite induced graphs differing by >10× in edge count.

A near-constant tail across wildly different graph sizes is the signature of an
**unbounded per-query worst case**, not of dataset scale.

> The objective is not "make QLS cheaper". It is: **remove the unbounded
> component**, and let the already-competitive median stand.

This reframes the entire plan. We are not optimizing an architecture; we are
deleting a convergence loop and fusing redundant traversals.

---

## 1. Quantified target and headroom

If `query_local_summary_ms` were bounded to the p95/p50 ratio of the other stages
(~1.5×), and no other stage regressed:

| dataset | QLS-v1 p95 | stage p95 → bounded | projected p95 | GNN p95 | margin |
|---|---:|---:|---:|---:|---:|
| 2wiki_clean    | 7.394 | 4.629 → 0.799 | **3.564** | 4.236 | +0.672 |
| hotpotqa_clean | 7.519 | 4.628 → 0.847 | **3.738** | 4.012 | +0.275 |
| metaqa         | 7.447 | 4.600 → 0.821 | **3.668** | 4.428 | +0.760 |
| musique_clean  | 7.558 | 4.690 → 0.899 | **3.767** | 4.151 | +0.384 |
| squad_clean    | 8.166 | 4.791 → 1.076 | **4.450** | 5.408 | +0.958 |
| webqsp         | 6.924 | 4.648 → 0.820 | **3.096** | 3.674 | +0.578 |

**This is an arithmetic projection, not a measurement.** It assumes the tail is
fully bounded and no other stage regresses. Its purpose is to set the budget:

> **v2's entire feature-computation budget at p95 is the margin column: 0.275 ms
> in the worst case (hotpotqa).**

Two things make that budget more comfortable than it looks:

1. The catalog's projected build (~0.3–0.5 ms total) **replaces** v1's local
   block rather than adding to it — v1's ~16 edge passes become ~6.
2. The tiny learner removes the 768→64 projection from the forward path, which
   Package D bills under `gather_transfer_forward_topk_ms` (p50 0.82–0.93 ms).

Throughput must clear a similar bar: the GNN currently wins 6/6
(250–347 vs 228–287 q/s).

---

## 2. Phase 0 — Feature-by-feature profiling (prerequisite)

Nothing is optimized before it is measured. The audit localizes the tail to a
*stage*, not to an operation.

**Instrument** [`_local_feature_chunk`](../src/mp_retrieval/structural_features.py:340)
per operation: 3 BFS passes, the seed-connection pass, the common-neighbour pass,
3 path-propagation passes, 8 PPR iterations (~16 full edge passes).

**Report per operation:** p50/p95/p99, plus correlation of each operation's time
with induced node count, edge count, seed count and component count.

**Decides** whether the tail is (a) a few queries with large induced graphs,
(b) PPR's fixed iterations on the densest queries, (c) BFS frontier blowup, or
(d) allocation behaviour rather than graph work. Each implies a different fix.

**Gate:** if the tail is *not* dominated by traversal work — e.g. if it is
allocation — §3 and §4 are re-planned before implementation rather than executed
on the assumption that they are the fix.

Every catalog feature also gets a measured `build p50/p95/p99`, storage, CPU RSS
and GPU memory. Validation split only.

---

## 3. One fused bounded traversal

Replace the separate BFS passes, the seed-connection pass and the
path-propagation passes with **one** bounded multi-source traversal carrying a
per-node seed bitmask (`|S_q| ≤ 10`, one 16-bit word):

```
hop 0 : mask[s] = bit(s)
hop 1 : mask1[v] |= mask0[u]    for each edge u->v
hop 2 : mask2[v] |= mask1[u]
hop 3 : mask3[v] |= mask2[u]
```

Emitted from the same sweep: per-seed hop membership, distance distribution
(Group B), distinct-seed support and concentration (C1–C5), predecessor and
branch statistics (C6–C7).

**Why this is the key change.** BFS frontier expansion and path-count propagation
are near-identical operations over the same edge array, currently executed
separately ([`:434`](../src/mp_retrieval/structural_features.py:434)). Fusing cuts
non-PPR edge passes from ~8 to **3**, and the bitmask carries the per-seed
information v1 destroys. **v2 becomes cheaper and more informative at once** —
Groups B and C (14 features) cost less than v1's 5 collapsed ones.

**Cost:** ≤ 800 bytes per query at budget 400. Negligible against the 4.4–5.3 MB
incremental GPU footprint.

**Verification, not ablation:** an equivalence test asserting the fused traversal
reproduces v1's features bit-for-bit where they overlap. A performance change
that silently alters a feature is a correctness bug, not a speedup.

---

## 4. Bounded diffusion replaces iterative PPR

Compare, at equal feature semantics:

| Variant | Work | Tail behaviour |
|---|---|---|
| v1 iterative PPR (8 iters, α=0.85) | fixed iterations, dense-query cost | unbounded worst case |
| `H=1` diffusion | 1 sparse mat-vec | hard bound |
| `H=2` diffusion | 2 | hard bound |
| `H=3` diffusion | 3 | hard bound |
| `H=3` truncated PPR | 3 + weighting | hard bound |

Record validation effectiveness, p50/p95/p99 and CPU RSS for each.

**The specific objective is eliminating the `query_local_summary` heavy tail**,
and fixed-depth variants achieve it by construction: no convergence loop, no
data-dependent termination.

**Selection is cost-first among accuracy-neutral variants** (within 0.10 R@5 of
exact on validation), then lowest p95. Fixed now so a slow-but-marginally-better
variant cannot be justified afterwards.

If an approximation is adopted, the confirmation still reports the exact-PPR
variant's accuracy, so no reader can suspect the approximation was chosen for an
accuracy artifact.

---

## 5. A BUDDY-style sketch backend (second implementation)

A competing structural backend, evaluated on the same Pareto axes:

```
OFFLINE                          QUERY TIME
global graph                     retrieve seeds
   |                                |
per-node precompute:             look up seed signatures
  1-hop signature                   |
  2-hop signature                intersect / aggregate
  structural sketch                 |
                                 features        <- no BFS at all
```

Candidate sketch mechanisms: bitset neighbourhood signatures, MinHash where
useful, compact hop signatures, sparse diffusion sketches, landmark signatures.

**Why it might win.** It moves nearly all graph work offline; query time becomes
lookup plus aggregation, with no traversal and therefore no traversal tail.

**Why it might not.** Sketches are lossy, and node-level signatures over the
*global* graph answer a slightly different question than exact computation over
the *query-induced* graph — which is the object every v1 feature is defined on.
Storage also becomes a real cost where the exact backend has none.

**Do not assume sketches are superior.** They are a speed/accuracy Pareto
candidate against exact bounded computation, and both are carried to Phase 3's
comparison. Storage cost is reported as a first-class axis for this backend.

---

## 6. Cached-path and break-even accounting

The three latency regimes stay distinct and must never be conflated:

```
cached operator latency  !=  uncached / on-demand post-retrieval latency
                         !=  raw-query end-to-end latency
```

**Break-even** remains `build_ms <= repeats * (uncached_ms - cached_ms)`,
compute-only, storage excluded — and must be **recomputed** for v2, not inherited
from v1, since v2 changes both terms. The sketch backend additionally requires a
storage-inclusive variant, because its whole design trades storage for time.

Only feature E4 (`global_hub_pct`) is offline-cacheable in the exact backend;
every other catalog feature is query-conditioned by definition.

---

## 7. Training efficiency — measured, not inferred

A GNN's training cost includes candidate features, `edge_index`, layer
activations, neighbour aggregation and autograd through graph operations. v2's
learner consumes ~33 scalars and has none of those. Feature extraction is
CPU-side or a bounded non-learned operator, with **no gradients**.

Training memory should therefore fall dramatically — but **do not infer this from
parameter count.** Measure, for every final candidate:

```
trainable parameters     training wall time      GPU-hours
peak GPU VRAM            peak CPU RSS            samples/sec
energy, if feasible
```

---

## 8. Pareto reporting

A candidate is **dominated** if another achieves `>= R@5, >= R@20, >= MRR` and
`<= p95, <= memory, <= parameters, <= training time`. Dominated points are kept
for audit, excluded from final candidates.

Required plots:

```
R@5 vs uncached p95              <- primary systems plot
R@5 vs trainable parameters
R@5 vs peak training memory
R@5 vs training wall-clock
R@5 vs feature-build p95
ceiling-normalized attainment vs p95
```

Plus the explanatory statistic per feature group, **never used as the sole
selection criterion**:

```
Efficiency(F_i) = ΔR@5 / Δp95_ms
```

Its purpose is to make decisions legible. If distinct seed support buys +2.0 R@5
for +0.02 ms and PPR buys +0.15 for +1.3 ms, the correct scientific decision —
remove PPR — is obvious, and *that* is how a simpler model ends up beating a GNN.

### Full axis table

| Axis | QLS-v1 (frozen) | GNN (frozen) | v2 target |
|---|---|---|---|
| R@5 | reference | significant advantage on 2wiki, hotpotqa (large budgets), metaqa | **>** GNN |
| R@20 / MRR / FullCov@20 | reference | ~parity | **≥** GNN |
| Cached operator latency | reference | reference | **<** GNN |
| Uncached p50 | 2.582–3.794 ms | 2.604–3.638 ms | **<** GNN (v1 wins 4/6) |
| Uncached p95 | 6.924–8.166 ms | 3.674–5.408 ms | **<** GNN (v1 loses 6/6) |
| Uncached p99 | 7.520–10.741 ms | 4.225–6.402 ms | **<** GNN (v1 loses 6/6) |
| Throughput | 228–287 q/s | 250–347 q/s | **>** GNN (v1 loses 6/6) |
| Peak incremental GPU memory | 4.42–5.32 MB | 4.75–67.22 MB | **≤** GNN (v1 wins 6/6, >12× on hotpotqa) |
| Training VRAM / wall-clock | measure | measure | **<** GNN |
| Trainable parameters | 213,506 / 209,351 | 213,568 / 213,440 / 209,280 | **~1.4K–4.3K** |
| Message passing at inference | none | yes | none |

**Axes v1 already wins are constraints, not slack.** A v2 that regresses GPU
memory or median latency has not dominated.

---

## 9. The kNN-removal experiment — cheapest possible Pareto win

Package B shows the GNN never significantly beats QLS on `knn_only` in any
dataset, and that `full_union_c` (most edges) is *worse* than `symbolic_b` on
2wiki, hotpotqa and metaqa. So test, as a graph ablation with **no new feature**:

```
structural / native graph only    vs    kNN only    vs    combined
```

If removing kNN edges improves accuracy **and** latency **and** memory
simultaneously — plausible, since it shrinks `|E_q|` and every bounded traversal
above is `O(|E_q|)` — that is a Pareto win obtained by deleting code. It should
be tried early for that reason.

---

## 10. Order of execution

```
Phase 0   profile per operation and per feature; D1-D4 diagnostics
Phase 3a  fused bounded traversal + bit-exactness test
Phase 3b  diffusion variant study (accuracy-neutral screen, then cost)
Phase 3c  sketch backend vs exact bounded backend
   |      (feature frontier, Phases 1-2, runs against these backends)
Phase 6   LODO transfer
Phase 7   freeze
Phase 8   confirmation
```

**Standing constraints.** Package E2 continues untouched to completion; no
profiling or development run may contend for its Modal capacity in a way that
perturbs it. Package F stays unopened. All frozen A–E artifacts, including
[`ONLINE_SYSTEMS_RESULTS.md`](ONLINE_SYSTEMS_RESULTS.md) and its mean-latency
framing, remain exactly as frozen; §0's percentile decomposition is a refinement
of the same data and is reported alongside the mean, never in place of it.
