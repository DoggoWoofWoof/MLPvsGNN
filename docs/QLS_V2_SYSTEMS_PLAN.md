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

### 2.1 The semantic path must be profiled separately — compute vs bandwidth

The structural tail is not the only thing Phase 0 has to settle. The semantic
frontier (catalog Group F) rests on a claim that is **not yet measured**: that
compressing v1's `768 → 64` projection into two 768-vectors buys latency as well
as parameters. Per query at `N_q = 400`:

| | v1 projection | S3 (F4+F5) | ratio |
|---|---:|---:|---:|
| parameters | 98,304 | 1,536 | **64.0x** |
| MACs | 19,709,952 | 614,400 | **32.1x** |
| **embedding bytes read (fp16)** | **614,400** | **614,400** | **1.0x** |

Arithmetic shrinks 32x. **Bytes moved do not shrink at all** — S1, S2, S3 and v1
all read the same 768 floats per candidate. Only S0 avoids that traffic.

**Phase 0 therefore measures, on the validation split:** wall-time of the
semantic path alone at S0/S1/S2/S3/v1, arithmetic intensity (MACs per byte), and
whether the stage scales with `N_q` (bandwidth-bound) or with parameter count
(compute-bound).

**This gates a claim.** If the semantic path is bandwidth-bound — which
[`QLS_V2_FEATURE_CATALOG.md`](QLS_V2_FEATURE_CATALOG.md) §3 prediction 8 states
we expect — then S3's advantage is **parameters, peak training memory and
training wall-time only**, and the systems plan must not report it as an
inference-latency win. Writing that limitation down before measuring is the point
of putting it here.

---

## 3. One fused bounded traversal

### 3.0 Boundedness is a design objective, not a side effect

Every operator in the query-local backend must satisfy:

> **B1.** Worst-case work is a fixed function of `|E_q|` and `N_q` with **no
> data-dependent iteration count**, no convergence test, and no early exit.
> **B2.** Worst-case memory is a fixed function of `N_q` and is **independent of
> `|E_q|`** — density must not buy allocation.
> **B3.** Worst case, best case and average case are the **same expression**.

B3 is the one that matters for the paper. A method whose mean is fast and whose
p99 is not has not solved the problem the audit identified; v1's
`query_local_summary` has p50 0.53–0.72 ms and **p95 4.60–4.79 ms**, and it is
the tail that makes graph-aware reranking hard to deploy. **We therefore report
worst-case complexity as the headline number and measured p99 alongside p50, not
mean runtime.** Any candidate operator that cannot state a worst case is
rejected before it is measured.

### 3.1 The algorithm

One bounded multi-source traversal carries a per-node seed bitmask **and** the
diffusion state over the same edge iteration. The seed contract
(`dense_top_k: 5`, `splade_top_k: 5`, `union: stable_unique`) guarantees
`5 ≤ |S_q| ≤ 10`, so the mask is **one 16-bit word** and the rank-weight domain
is at most `2^10 = 1024`.

```
FUSED-BOUNDED-QUERY-LOCAL(G_q = (V_q, E_q), S_q, H = 3, alpha)

  S <- |S_q|                     # 5 <= S <= 10   (frozen contract)
  n <- |V_q| <= N_q <= 400

  # -- rank-weight table: 2^S - 1 additions, once per query --------------
  for i in 0..S-1: w[i] <- 1 / r(s_i)          # r(s) in {1..5}
  W[0] <- 0
  for m in 1 .. 2^S - 1:
      W[m] <- W[m & (m-1)] + w[ctz(m)]         # 1 add, 1 blsr, 1 tzcnt
  Wtot <- W[2^S - 1]

  # -- initialise --------------------------------------------------------
  mask[0..H][v] <- 0                            # (H+1) x n  uint16
  for i in 0..S-1: mask[0][s_i] <- 1 << i
  first[v] <- H+1 ; for i: first[s_i] <- 0      # uint8
  z[v] <- 0 ; for i: z[s_i] <- 1/S              # float32
  acc[v] <- 0 ; preds[v] <- 0

  # -- exactly H passes over E_q ----------------------------------------
  for h in 1..H:
      cur <- mask[h-1] ; nxt <- copy(cur) ; zn <- zeros(n)
      for (u,v) in E_q:                         # ONE traversal, both signals
          new <- cur[u] & ~cur[v]
          if new: nxt[v] |= new
                  preds[v] += 1
                  if first[v] > h: first[v] <- h
          zn[v] += z[u] / outdeg(u)             # same edge, same cache line
      mask[h] <- nxt ; z <- zn ; acc += alpha^h * z

  # -- per-candidate read-out: O(1) each --------------------------------
  for d in candidates:
      support_h     <- popcount(mask[h][d]) / S           # h = 1,2,3
      rank_weighted <- W[mask[h][d]] / Wtot               # h = 1,2,3
      min_distance  <- first[d]
      unreachable   <- 1 - popcount(mask[H][d]) / S
      mean_reach_d  <- sum_h h*popcount(mask[h][d] & ~mask[h-1][d])
                       / popcount(mask[H][d])
      diffusion_h   <- z_h[d] ;  truncated_ppr <- acc[d]
      predecessors  <- preds[d]
```

### 3.2 Exact complexity

**Time.** The loop structure contains no conditional iteration count.

```
table build      2^S - 1 additions              <=     1,023
edge passes      H * |E_q|                       =  3 * |E_q|
read-out         O(n), 3 popcounts + 3 lookups
-----------------------------------------------------------------
TOTAL            Theta(H*|E_q| + n + 2^S)  =  Theta(3|E_q| + n + 1024)
```

and by B3 this is simultaneously the worst, best and average case.

With `|E_q| <= N_q(N_q - 1)`, the **worst case at `N_q = 400` is
`3 x 159,600 = 478,800` edge operations**, against v1's 16 passes
(3 BFS + 1 seed-connection + 1 common-neighbour + 3 path + 8 PPR) =
**2,553,600** — a **5.33x** worst-case reduction. This is a counting argument
over the loop structure, not a measurement; Phase 0 measures the constant.

**Memory,** at `N_q = 400`, and note that no term depends on `|E_q|`:

| Buffer | Type | Bytes @ N_q=400 |
|---|---|---:|
| `mask[0..3]` | `(H+1) x n` uint16 | 3,200 |
| `first` | `n` uint8 | 400 |
| `preds` | `n` uint16 | 800 |
| `z`, `zn`, `acc` | `3n` float32 | 4,800 |
| `W` | `2^S` float32 | <= 4,096 |
| **total** | | **<= 13,296 B (13.0 KB)** |

Against the 4.4–5.3 MB incremental GPU footprint this is **~0.25%**, and it is a
hard bound: a maximally dense query costs exactly the same 13 KB as a maximally
sparse one. That satisfies **B2**.

### 3.3 What this single pass buys

Every structural feature in primary frontier rungs **R1 through R4 — all twelve
dimensions — is a read-out of this one traversal.** Nothing in R1–R4 requires a
second pass over `E_q`:

| Rung | Features | Source in the pass |
|---|---|---|
| R1 | support @1 / ≤2 / ≤3 | `popcount(mask[h][d])` |
| R2 | rank-weighted @1 / ≤2 / ≤3 | `W[mask[h][d]]` |
| R3 | min distance, mean reachable distance, unreachable fraction | `first`, `mask[h] & ~mask[h-1]` |
| R4 | diffusion h1 / h2 / h3 | `z` at each `h`, `acc` |

R5 is the exception and this is a second reason to expect it to be dropped: its
component features need a union-find, i.e. **an additional near-linear pass** that
R1–R4 do not.

**Why this is the key change.** BFS frontier expansion, path-count propagation
and diffusion are near-identical scatter-accumulate operations over the same edge
array, currently executed separately
([`:434`](../src/mp_retrieval/structural_features.py:434)). Fusing them cuts edge
passes from ~16 to **3** while the bitmask carries the per-seed information v1
destroys. **v2 becomes cheaper and strictly more informative at once.**

**Verification, not ablation:** an equivalence test asserting the fused traversal
reproduces v1's features bit-for-bit where they overlap. A performance change
that silently alters a feature is a correctness bug, not a speedup.

---

## 4. Bounded diffusion replaces iterative PPR — as a Pareto experiment

This is run as a **direct Pareto comparison**, not as an assumption.

| Variant | Work | Worst case | Tail |
|---|---|---|---|
| v1 iterative PPR (8 iters, α=0.85) | 8 mat-vecs | `8\|E_q\|` | data-dependent constant, unbounded in `\|E_q\|` |
| `H=1` diffusion | 1 mat-vec | `\|E_q\|` | hard bound |
| `H=2` diffusion | 2 | `2\|E_q\|` | hard bound |
| `H=3` diffusion | 3 | `3\|E_q\|` | hard bound |
| `H=3` truncated PPR | 3 + weighting | `3\|E_q\|` | hard bound |

**Primary plot: validation R@5 (y) against `query_local_summary` p95 in ms (x),
one point per variant, six datasets shown separately and as a mean.** This plot
is the deliverable of the experiment and its axes are fixed here so they cannot
be chosen later to flatter a result.

Secondary, reported in the same table for every variant: p50, p99, peak CPU RSS,
and mean. **p99 is mandatory** — the entire motivation is tail behaviour, and a
p50-only comparison would hide exactly the failure we are trying to remove.

**Decision rule, frozen now.** If any bounded variant is **Pareto-superior** to
v1 iterative PPR — no worse on validation R@5 within the frontier tolerance τ of
§6 *and* strictly better on p95 — then **iterative PPR is removed from the v2
candidate set entirely**, not kept as an option. If no bounded variant is
Pareto-superior, iterative PPR stays and we report that the bounded
approximation did not pay, which is a publishable negative result about the
`H`-truncation hypothesis and must not be quietly dropped.

Among bounded variants that are mutually non-dominated, the §6 lexicographic
rule applies unchanged: smallest `H` wins ties, because smaller `H` is strictly
cheaper and strictly more bounded.

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

### Mandatory marginal-efficiency report, one row per frontier transition

For **every** transition R0→R1, R1→R2, R2→R3, R3→R4, R4→R5 and every semantic
transition S0→S1, S1→S2, S2→S3, the following is reported in full. Not a
summary statistic — the whole row, on the validation split, six datasets plus
mean:

```
effectiveness   dR@1   dR@5   dR@20   dMRR
latency         dp50_ms   dp95_ms   dp99_ms          (uncached, end-to-end)
memory          dCPU_RSS_MB      dtraining_peak_VRAM_MB
size            dtrainable_parameters
```

Deltas are against the immediately preceding rung, with the same seeds, the same
splits and the same learner width. A rung that is not reported in this form is
not admitted, whatever its R@5.

Alongside each row, **as an explanatory statistic only and never as the selection
objective**:

```
Efficiency(rung) = dR@5 / dp95_ms
```

Its purpose is to make decisions legible, not to make them. If distinct seed
support buys +2.0 R@5 for +0.02 ms and PPR buys +0.15 for +1.3 ms, the correct
scientific decision — remove PPR — is obvious. But `dR@5 / dp95` is a ratio of
two noisy quantities and is unstable when `dp95 → 0`; selection is done by the
lexicographic rule in §6 of the development protocol, which never divides.

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
