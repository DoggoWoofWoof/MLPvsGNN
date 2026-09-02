# Stage D0b: does `bridge_support` survive contact with the other features?

> **RETRACTION (D0c, D1).** The stratified `TARGET_H1` - `CAND` split reported
> below does **not** survive convergence and does **not** appear at the ranker.
> At ten epochs it inverts -- isolated R@1 goes +3.06 to -1.80, `ordinary` -0.51
> to +3.28 -- and at QLS-v1 every stratum is null. All four arms here were still
> climbing at the three-epoch budget, so they were compared at their rates of
> ascent. **Section 4 and the second headline paragraph are withdrawn as
> evidence.** The `bridge_support` kill in section 3 is unaffected: it is a
> within-arm comparison at a shared budget, and D0c confirms the local block it
> was redundant with is load-bearing. See
> [GRAPH_CONTEXT_D0C_D1_RESULTS.md](GRAPH_CONTEXT_D0C_D1_RESULTS.md).

D0 scored each structural quantity on its own and found one striking number:
on candidates isolated inside `G[Cq]`, `bridge_support` computed over
`TARGET_H1` reaches 0.9513 mean rank AUC on 2wiki, against an exact 0.5000
under the historical context. A univariate AUC is not a ranking, though, and a
feature that is informative alone can be worth nothing beside features a ranker
already has. D0b is the cheap multivariate test that decides whether
`bridge_support` earns a place in QLS-v2: four linear arms, one CPU container,
no GNN, no architecture search, no test split.

**`bridge_support` does not earn its place.** Added to an arm that already has
the ten frozen local columns over `TARGET_H1`, it moves overall validation R@1
by +0.53 points, R@5 by +0.02, and R@20 by **-0.10**. On the isolated stratum
it was built for it changes R@5 and R@20 by **exactly zero**. The ranker does
use it -- it carries a weight of +1.173, the fourth largest in the arm -- but
what it carries is already there. That is not surprising in hindsight: the
bridge is computed from the restored context, and so are the ten columns.

**The interesting result is the one D0b was not asked for.** `CAND` and
`TARGET_H1` are near-identical overall (validation R@5 67.82 vs 67.83) and
sharply different once split by how connected the query's hardest gold was:

| `TARGET_H1` - `CAND` | n | R@1 | R@5 | R@20 | MRR |
|---|---:|---:|---:|---:|---:|
| isolated | 834 | **+3.06** | **+1.11** | -0.24 | **+4.44** |
| degree_1 | 750 | -1.70 | -0.57 | +0.47 | -2.26 |
| low_degree | 1218 | -2.50 | -0.35 | +1.31 | -3.18 |
| ordinary | 198 | -0.51 | -0.25 | +2.02 | -1.17 |
| overall | 3000 | -0.63 | +0.01 | +0.72 | -0.70 |

Validation, points. MEASURED.

A flat overall number is hiding a large positive on the 27.8% of queries whose
gold had no in-pool structure to describe, and a large negative on the queries
that already had some. This is the sharpened problem statement holding up under
a ranker rather than under an AUC: restoring context supplies evidence where
there was none, and dilutes evidence where there was some.

| | |
|---|---|
| **Decision** | **KILL `bridge_support`.** It is redundant with the ten frozen columns over the same context. |
| **Also** | D1 stays worth running, on the strength of the stratified split -- not on the overall number, which is null. |
| **Not** | a measurement of QLS-v1. A 19-parameter linear map is not a 205k-parameter ranker, and this is a prior for D1, not a substitute. |
| **Cost** | 412.1 s, $0.073, against a declared ceiling of $0.70. |

---

## 1. What ran

2wiki_clean, whole train (10,500) and whole validation (3,000) splits, seed 0,
learning rate 0.05 carried from A3's frozen selection rather than re-chosen.
Four arms over a shared nine-column base of 2 rank and 7 static features:

| arm | features | params | what it is |
|---|---|---:|---|
| `RETRIEVAL_ONLY` | base | 9 | a floor, not a candidate design |
| `CAND` | base + 10 local over `G[Cq]` | 19 | reproduces A3's feature set |
| `TARGET_H1` | base + 10 local over `Cq u N1_in(Cq)` | 19 | the D1 contrast, linearly |
| `TARGET_H1_BRIDGE` | + `bridge_support` | 20 | the question |

Three epochs, final epoch reported. No epoch was selected, because D0b has no
test split to report on and selecting on validation would have spent the
reporting split twice. This turned out not to matter: **epoch 3 is the argmax
of validation R@5 in all four arms**, so the reported numbers are also the ones
A3's selection rule would have chosen. VERIFIED FROM THE RECORDED TRAJECTORY.

## 2. The `CAND` arm is A3

The launcher could not run its intended alignment check -- the active workspace
holds this dataset topology-only and has no sealed
`derived/fixed_structural_features_v1/` to compare against, so `cand_arm_alignment`
is recorded as `checked: false` with the reason. The static block was rebuilt
from `graph.pt` through the shipped `build_static_features` at the frozen
parameters instead, in 2.6 s.

The end-to-end evidence is stronger than the cache check would have been:

| | A3 seed 0 (sealed caches, GPU) | D0b `CAND` (rebuilt static, CPU) |
|---|---|---|
| validation R@5 | 0.6781666666666667 | 0.6781666666666667 |
| 19 weights | — | max abs difference **3.53e-05** |

MEASURED. A3's frozen five-seed validation R@5 is 0.6777333 +/- 0.0003699, and
its seed 0 is the value above. The residual on the weights is consistent with
float32 accumulation order and the float16 storage of the local block; this is
functional identity, not bit-identity, and is not claimed as more.

That single comparison corroborates three separate things at once: the rebuilt
static block matches the sealed one, candidate-readout normalisation left `CAND`
unchanged, and the D0b harness is the A3 learner rather than a lookalike.

## 3. `bridge_support`, against the gate it was given

The kill gate was: *kill `bridge_support` if it fails even this simple
multivariate test*. `TARGET_H1_BRIDGE - TARGET_H1`, validation, points:

| + `bridge_support` | n | R@1 | R@5 | R@20 | MRR |
|---|---:|---:|---:|---:|---:|
| isolated | 834 | +0.63 | **0.00** | **0.00** | +0.85 |
| degree_1 | 750 | +0.23 | 0.00 | -0.03 | +0.29 |
| low_degree | 1218 | +0.78 | +0.02 | -0.23 | +0.80 |
| ordinary | 198 | -0.25 | +0.25 | 0.00 | -0.17 |
| overall | 3000 | +0.53 | +0.02 | -0.10 | +0.62 |

MEASURED. The largest effect anywhere is +0.85 MRR on 834 queries; the sign
flips on `ordinary`; R@5 and R@20 are unchanged on the stratum where D0
measured 0.9513 AUC. At one seed, none of this is distinguishable from noise,
and half a point of R@1 is about five queries in 834.

**Killed.** §5 had reserved a separate small QLS-v2 probe for this feature. That
probe is no longer needed: D0b *is* that probe, and it came back negative. The
D0 finding stands as a mechanistic result about what the restored context
contains -- it is simply not incremental information once the ten frozen columns
are computed over that same context.

## 4. What `TARGET_H1` does, and what it costs

Overall the two contexts are a wash. Underneath, `TARGET_H1` trades top-of-list
precision for depth almost everywhere -- R@1 and MRR down, R@20 up -- and does
the opposite on the isolated stratum, where it gains on R@1, R@5 and MRR
together and loses only 0.24 on R@20.

This is the shape the sharpened problem statement predicted. It is also a
warning about the arm: two thirds of the validation split gets *worse*
top-of-list ranking under indiscriminate two-hop reach. Whether a 205k-parameter
ranker can keep the isolated-stratum gain while suppressing the dilution is
exactly the question D1 asks, and D0b cannot answer it -- a linear map has no
capacity to condition a structural feature on how connected the candidate is.
HYPOTHESIS, NOT TESTED.

### The addressable population is larger than D0 reported

D0 measured queries-with-an-isolated-gold at **60 of 300 (20.0%)** on 2wiki,
from a deterministic prefix of the validation split. D0b measured the whole
split: **834 of 3,000 (27.8%)**. Same definition -- at least one in-pool gold
with no in-pool neighbour -- different population. The prefix under-counted by
7.8 points, which moves D1's prior in the favourable direction. MEASURED; D0's
figure is not wrong, it is a prefix.

## 5. `RETRIEVAL_ONLY` leads on R@1, and that is not a finding

| arm | R@1 | R@5 | R@20 | MRR |
|---|---:|---:|---:|---:|
| `RETRIEVAL_ONLY` | **39.22** | 67.25 | 71.54 | **93.68** |
| `CAND` | 33.09 | 67.82 | 75.23 | 85.90 |
| `TARGET_H1` | 32.47 | **67.83** | **75.95** | 85.20 |
| `TARGET_H1_BRIDGE` | 33.00 | 67.85 | 75.85 | 85.82 |

Nine features with no query-local structure at all beat every structural arm at
R@1 by more than six points, and lose to all of them at R@20 by nearly four.
**This must not be read as "the local block hurts."** All four arms are still
climbing steeply at the fixed three-epoch budget -- `RETRIEVAL_ONLY` goes
32.2 -> 37.0 -> 39.2 on R@1 with its mean loss falling 5.22 -> 4.00 -> 3.11 --
so the arms are being compared mid-ascent, at rates of ascent that differ with
their parameter count. The learning rate is A3's, selected for the 19-feature
configuration and applied unchanged to a 9-feature one. A converged comparison
was not run and is not what D0b was for. INFERENCE, low confidence.

The one thing the row does establish is that the structural columns buy depth:
+3.7 to +4.4 points of R@20 over the retrieval floor, consistently across all
three structural arms.

## 6. Cost, measured

| | |
|---|---:|
| feature build, both contexts, 13,500 queries / 4,851,276 rows | 367.9 s |
| static rebuild from `graph.pt` | 2.6 s |
| training, four linear arms | 15.4 s |
| dataset load and container overhead | ~26 s |
| **total elapsed** | **412.1 s** |
| container cost at $0.634/h | **$0.073** |
| declared ceiling | $0.70 |

Per-query context latency, which is what D1's projection is built from:

| arm | p50 | p95 | p99 | mean |
|---|---:|---:|---:|---:|
| `CAND` | 5.36 ms | 9.78 ms | 12.61 ms | 6.48 ms |
| `TARGET_H1` | 19.03 ms | 24.47 ms | 29.88 ms | 19.47 ms |

MEASURED. `TARGET_H1` costs 3.0x `CAND` at the mean and 2.5x at p95. `CAND`'s
7.78 s maximum is the first call's Numba compilation, not a query.

## 7. What this does and does not decide

- **Decides:** `bridge_support` is not a QLS-v2 feature. Killed on the gate it
  was given.
- **Decides:** the D0b harness reproduces A3, so the rebuilt static block is
  usable where the sealed one is absent.
- **Does not decide:** whether `TARGET_H1` helps QLS-v1. D1 is still the
  experiment that answers that, and D0b's stratified split is its prior, not its
  result.
- **Does not decide:** anything about a second dataset. 2wiki has the highest
  isolated-gold prevalence of the six; this is its best case, not a typical one.
