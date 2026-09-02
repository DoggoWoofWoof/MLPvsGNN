# Stage D3: nearly all of QLS-v1's local block is one bit

D2 found that zeroing QLS-v1's ten query-local columns costs 20.82 points of
R@5. It could not say what that number was made of, because all ten columns are
seed-anchored and `sa_mlp` has no other seed channel: zeroing them removes seed
identity, seed geometry and graph-derived structure in one move.

D3 separates them on one ladder, at one operating point, with all four arms at
213,506 parameters and only the surviving local columns differing.

| arm | kept | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---|---:|---:|---:|---:|---:|
| `ZERO_LOCAL` | nothing | 26.55 | 47.07 | 61.45 | 70.36 | 29.30 |
| `SEED_ID_ONLY` | `distance_0` | **30.15** | 65.27 | 72.22 | **80.73** | 42.77 |
| `DISTANCE_ONLY` | `distance_0..3` | 29.63 | 66.76 | 77.36 | 80.08 | 51.53 |
| `FULL_LOCAL` | all ten | 29.03 | **67.88** | **77.53** | 79.28 | **51.93** |

2wiki_clean, validation, seed 0, points. MEASURED.

| increment | measures | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---|---:|---:|---:|---:|---:|
| `delta_seed` | which candidates were seeds | **+3.60** | **+18.20** | **+10.77** | **+10.37** | **+13.47** |
| `delta_distance` | seed-distance geometry | -0.52 | +1.49 | **+5.13** | -0.65 | **+8.77** |
| `delta_remaining_structure` | support, paths, PPR, common neighbours | -0.60 | +1.12 | +0.17 | -0.80 | +0.40 |

The three telescope to D2's gap exactly, on every metric, by construction.

**Verdict: SEED IDENTITY DOMINATES.**

Share of D2's gap: on R@5, **87.4%** seed / 7.2% distance / 5.4% remaining. On
R@20, 67.0% / 31.9% / 1.0%. On FullCov@20, 59.5% / 38.7% / 1.8%. On R@1 and MRR
the seed share *exceeds* 100% -- one bit beats the full block outright.

---

## 1. One bit is better than ten columns at the top of the list

`SEED_ID_ONLY` is a model whose entire query-local input is `I[d in Sq]`. It
beats the complete QLS-v1 block on **R@1 (30.15 vs 29.03)** and on **MRR (80.73
vs 79.28)**, and it reaches 96.1% of `FULL_LOCAL`'s R@5 from one column instead of
ten.

That is the single most consequential number here. The nine remaining columns,
taken together, are not merely a small gain on top-of-list ranking -- they are a
**net loss** of 1.12 R@1 and 1.45 MRR against the bare indicator.

## 2. Distance geometry earns its place, but only on depth

`delta_distance` is the one structural increment with real value, and its value
is entirely in coverage: **+5.13 R@20** and **+8.77 FullCov@20**, against -0.52
R@1 and -0.65 MRR. Adding the three non-zero distance buckets does not sharpen
the head of the ranking; it finds more golds further down.

The stratification says why, and it is a clean mechanistic check rather than a
post-hoc story:

| `delta_distance` | n | R@1 | R@5 | R@20 | MRR |
|---|---:|---:|---:|---:|---:|
| isolated | 834 | +0.24 | -0.12 | **+1.02** | +0.19 |
| degree_1 | 750 | -0.77 | +0.17 | +5.27 | -1.01 |
| low_degree | 1218 | -0.92 | +2.91 | +7.14 | -1.09 |
| ordinary | 198 | -0.25 | +4.55 | **+9.60** | -0.06 |

Distance geometry is worth almost nothing where the induced graph is empty and
most where it is richest -- the exact opposite of `delta_seed`, below. On an
isolated candidate every bucket collapses to `distance_3_plus_or_unreachable`,
so there is no geometry to read. `boundaries_fitted_here: false`; these are the
same four strata, unchanged since D1.

## 3. The rest of the structural machinery is lower priority, not dead

`seed_connections`, `paths_length_1/2/3`, `personalized_pagerank` and
`common_out_neighbors_with_seed_neighborhood` -- six columns, the whole
support/path/diffusion apparatus -- are worth **+1.12 R@5, +0.17 R@20 and +0.40
FullCov@20**, and are **negative** on R@1 (-0.60) and MRR (-0.80).

Per stratum they are worth something only on the best-connected queries:

| `delta_remaining_structure` | n | R@1 | R@5 | R@20 | MRR |
|---|---:|---:|---:|---:|---:|
| isolated | 834 | -0.99 | -0.27 | +0.12 | -1.62 |
| degree_1 | 750 | -0.73 | +1.47 | -0.20 | -0.82 |
| low_degree | 1218 | -0.51 | +1.64 | +0.29 | -0.62 |
| ordinary | 198 | **+1.01** | **+2.53** | **+1.01** | **+1.59** |

`ordinary` is 198 queries, 6.6% of validation. Everywhere else this block is
neutral to harmful.

**These six columns are not eliminated by this result.** +1.12 R@5 is small
beside seed identity's +18.20, but it is not negligible in absolute terms: it
sits at roughly the scale of the QLS-versus-GNN differences measured on some
datasets, so a result of that size cannot be discarded on relative grounds. The
correct status is **lower priority, and each family must justify its own cost**,
not dead.

The sign pattern is itself a lead worth keeping: positive on R@5 (+1.12) and
near-zero on R@20 (+0.17) while negative on R@1 (-0.60) and MRR (-0.80). That
shape suggests these columns may be reorganising the ranking specifically around
the R@5 operating region rather than improving it globally. Testing that means
decomposing the six individually or by family, which this stage deliberately did
not do -- it moved all six together.

## 4. Seed identity: largest exactly where the graph is thinnest

| `delta_seed` | n | R@1 | R@5 | R@20 | MRR |
|---|---:|---:|---:|---:|---:|
| isolated | 834 | +6.26 | **+23.23** | +13.25 | +15.78 |
| degree_1 | 750 | +5.20 | +21.40 | +13.00 | +13.25 |
| low_degree | 1218 | +2.07 | +14.47 | +8.72 | +6.93 |
| ordinary | 198 | **-4.29** | +7.83 | +4.55 | -2.23 |

Monotone on all four measures, and on `ordinary` the bare indicator actually
*hurts* R@1 and MRR. Where the candidate graph carries usable structure, knowing
which candidates were retrieved matters least.

## 5. The measurement replaced an inference, and confirmed it

The D2 write-up sized this split by comparing against the frozen `seed_only`
model across two splits at two head widths, and labelled it INFERENCE: roughly
+18.8 for the bit and about +2.57 for everything else. D3 measures the same
quantities within one stage: **+18.20** and **+2.61**. The inference was close,
but it is no longer what the conclusion rests on.

## 6. What made the decomposition legitimate

- **`distance_0` is exactly `I[d in Sq]`,** not a proxy. It is set from `is_seed`
  before any BFS hop and never overwritten, `candidate_readout` leaves columns
  0-3 untouched, and `_seed_arrays` reads the same `retrieval_seed_local` the
  historical `_seed_indicator` does. VERIFIED FROM CODE, and asserted at full
  scale inside this run: **4,851,276 candidate rows compared, 0 mismatches**,
  105,675 of them seeds.
- **The distance group is a complete one-hot** on all 4,851,276 rows, so
  `DISTANCE_ONLY` preserves geometry exactly and `SEED_ID_ONLY` preserves
  membership exactly.
- **The arm keeping four buckets is named `DISTANCE_ONLY`,** not
  seed-membership-only. Keeping the group exposes geometry -- seed, one hop, two
  hops, three-or-unreachable -- and naming it after membership would have
  mislabelled the quantity the stage exists to separate.
- **Masked, not removed.** All four arms are 213,506 parameters at head width 61.
- **`ZERO_LOCAL` and `FULL_LOCAL` were not retrained.** They are D2's
  `NO_QUERY_LOCAL` and `FULL_CAND`, reused behind an 18-condition guard covering
  dataset, fingerprint, context, model, node count, candidate contract,
  candidate rows, static-feature source, all three split sizes and all five
  training values. All 18 matched.

## 7. What is not established

- **Not a converged measurement.** Every arm runs the frozen QLS-v1 budget of
  three epochs and every arm is still descending at the boundary
  (`SEED_ID_ONLY` 2.658 -> 2.206 -> 1.895, `DISTANCE_ONLY` 2.431 -> 2.004 ->
  1.694). That is deliberate: D3 decomposes D2 at D2's operating point, and
  giving one arm more epochs than another would destroy the comparison. It is
  not a claim about converged contributions.
- **Not a feature-level attribution inside the last group.** `delta_remaining_
  structure` moves six columns together, so this stage says nothing about any one
  of them. A per-family decomposition is the way to settle whether the R@5-only
  shape is real; it is deferred, not refused.
- **One dataset, one seed.** 2Wiki, seed 0. `TARGET_H1` stays closed; both
  contexts are `G[Cq]`.
- **Not a statement about retrieval quality.** The seed set is dense top-5 union
  SPLADE top-5. D3 says the ranker leans on *which* candidates were retrieved,
  not that this is the best possible seed signal.

## 8. What it decides

- **QLS-v1's query-local block is, to first order, a retrieval prior.** 87.4% of
  its R@5 value is one binary column, and on precision measures that column is
  better alone than the full block.
- **The broad structural frontier is not justified by this evidence, but the
  remaining columns are not eliminated either.** Everything beyond seed identity
  and distance buckets is worth about a point of R@5 and costs R@1 and MRR, so
  `REMAINING STRUCTURE HAS LARGE VALUE` is refused -- yet a point of R@5 is not
  nothing at the scale this project operates at. They are lower priority and
  must justify their cost individually, not written off.
- **Seed identity plus distance recovers almost all of the block**: 94.6% of the
  R@5 gap, 99.0% of R@20, 98.2% of FullCov@20. QLS-v2 should start from
  retrieval prior + seed identity + minimal seed geometry and add one family at
  a time, not from a feature catalogue.
- **Depth and precision want different things.** Seed identity alone maximises
  R@1 and MRR; distance geometry buys R@20 and FullCov@20 while costing both.
  Any QLS-v2 objective has to choose, or model the trade-off explicitly.

| | |
|---|---|
| **Cost** | 90.4 s of accounted compute for two new arms (51.0 s shared feature build, 2.0 s static, 18.9 s + 17.8 s training, 0.7 s inference). Two runs, not four. Container total not captured by the spawn path; wall clock under ten minutes, so under $0.29 at $1.734/h against $0.90 authorised. |
| **Contract** | `gnn_trained: false`, `message_passing: false`, `test_split_read: false`, `epoch_selected_on_validation: false`, `architecture_changed: false`, `scored_nodes: exactly Cq in every arm`. |
