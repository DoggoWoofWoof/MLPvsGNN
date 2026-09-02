# Stage D7: the six columns are one family's worth of information, told four times

D6 measured QLS-v1's six residual structural columns as a block and found they
still pay once graded retrieval evidence and seed-distance geometry are both
present: +0.89 R@5, +0.24 R@20, +0.53 FullCov@20, against -0.63 R@1 and -0.70
MRR. D7 splits that block along the four kinds of graph information it contains
and asks which part bought the depth and which part sold the head.

The six under test, as the frozen historical columns define them:

    SUPPORT         col 4     seed_connections
    PATHS           cols 5-7  paths_length_1   paths_length_2   paths_length_3
    DIFFUSION       col 8     personalized_pagerank
    NEIGHBOURHOOD   col 9     common_out_neighbors_with_seed_neighborhood

**Four new runs, not six.** `D6_BASE_13` is already the exact matched 13-column
control at head width 61 and 213,689 parameters, fitted at this operating point,
so it and `D6_FULL_13` are reused as a causal control and a joint reference
rather than refitted. Only the four family arms were trained.

2Wiki validation, seed 0, `G[Cq]`, three epochs, all six arms at **213,689**
parameters and `local_dim` 13.

## The ladder

| arm | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---|---|---|---|---|
| `D6_BASE_13` — retrieval prior + seed geometry (reused) | 39.39 | 71.38 | 77.70 | 93.88 | 51.90 |
| `D7_SUPPORT_13` | 39.37 | 72.02 | 77.90 | 93.96 | 52.30 |
| `D7_PATHS_13` | 39.27 | 72.20 | 77.83 | 93.73 | 52.20 |
| `D7_DIFFUSION_13` | 39.27 | 71.82 | 77.87 | 93.82 | 52.23 |
| `D7_NEIGHBOURHOOD_13` | 39.17 | 71.45 | 77.78 | 93.69 | 52.07 |
| `D6_FULL_13` — all six (reused) | 38.77 | 72.28 | 77.94 | 93.18 | 52.43 |

## The four family increments

Each arm minus `D6_BASE_13`.

| family | columns | R@1 | R@5 | R@20 | MRR | FullCov@20 | verdict |
|---|---|---|---|---|---|---|---|
| SUPPORT | `seed_connections` | -0.03 | **+0.64** | +0.20 | **+0.09** | +0.40 | **PROMISING** |
| PATHS | `paths_length_1`, `paths_length_2`, `paths_length_3` | -0.12 | **+0.82** | +0.13 | -0.15 | +0.30 | **PROMISING** |
| DIFFUSION | `personalized_pagerank` | -0.13 | +0.43 | +0.17 | -0.06 | +0.33 | **NEGLIGIBLE** |
| NEIGHBOURHOOD | `common_out_neighbors_with_seed_neighborhood` | -0.22 | +0.07 | +0.08 | -0.19 | +0.17 | **NEGLIGIBLE** |

Verdicts follow the thresholds registered before any arm was fitted: qualify on
≥ +0.50 pp R@5 or on a clear R@20/FullCov gain without materially sacrificing
R@5, then fail to TRADEOFF if the head-ranking regression is comparable to or
larger than the benefit. No family landed in TRADEOFF.

DIFFUSION missed the line by 0.07 points. It is reported as NEGLIGIBLE because
that is the rule that was written down, and the margin is stated here rather
than used to move it.

## The result: strong sub-additivity

| | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---|---|---|---|---|
| joint, `D6_FULL_13 - D6_BASE_13` (from D6) | -0.63 | +0.89 | +0.24 | -0.70 | +0.53 |
| sum of the four family increments | -0.48 | **+1.96** | +0.58 | -0.30 | +1.20 |
| **additivity residual** | **-0.14** | **-1.07** | **-0.34** | **-0.39** | **-0.67** |

Descriptive only, and negative on every metric. More than half the R@5 the four
families produce separately disappears when they are combined. On the reading
registered in advance, a large negative residual means redundancy or
interference — and -1.07 against a +1.96 sum is not a small correction, it is
the dominant fact about this block.

The plainest statement of it: **the six columns carry roughly one family's worth
of distinct information, told four times.** Two of the four tellings are good
enough to qualify on their own, and stacking all four buys less than either of
them does alone plus a little.

This is not a statistical interaction test. One dataset, one seed.

## Where the head-ranking loss came from: nowhere in particular

D6's block cost -0.63 R@1 and -0.70 MRR. The four families together account for
-0.48 and -0.30. The rest — and for MRR that is more than half of it — is in the
additivity residual, which is to say it appears only when the columns are
combined and belongs to no family.

What can be attributed:

- **NEIGHBOURHOOD is the largest single contributor to both losses** (-0.22 R@1,
  -0.19 MRR) and produces the least effectiveness of any family (+0.07 R@5,
  inside the near-zero band). It is the one clean delete in this stage: it costs
  the most head ranking per point of depth it returns.
- **SUPPORT is head-neutral.** -0.03 R@1 and *+0.09 MRR* — the only family that
  improves a head metric at all, while returning +0.64 R@5 and +0.40 FullCov@20.
- PATHS and DIFFUSION each cost a little (-0.12/-0.15 and -0.13/-0.06), well
  under what they return on depth.

So the ideal outcome sketched before the run happened only partly. One family
does give depth with essentially no head loss, and one family does give almost
nothing while costing the most. But the bulk of the MRR regression is emergent
rather than localised, and deleting any single family would not recover it.

## Both qualifying families individually clear half the joint

`dominant_family` fired for PATHS: +0.82 R@5 exceeds both the +0.50 bar and half
the joint +0.89. But SUPPORT's +0.64 clears the same half-joint bar (+0.445), so
two families do, and PATHS leads SUPPORT by 0.18 points at one dataset and one
seed. This is a two-family result, not a single winner, and the sub-additivity is
the reason both can be true at once.

SUPPORT is also the arm whose held-out score was still rising at the budget
boundary — it peaked at epoch 3 where every other arm here peaked at epoch 2 —
so its +0.64 is if anything the more likely of the two to be understated. D7 was
not given extra epochs to find out; the frozen three-epoch budget is the point of
comparison, not an obstacle to it.

## By stratum

Counts unchanged from D5 and D6 and not refitted here: isolated 834, degree_1
750, low_degree 1218, ordinary 198 (`boundaries_fitted_here` false).

R@5 increment by stratum:

| family | isolated | degree_1 | low_degree | ordinary |
|---|---|---|---|---|
| SUPPORT | +0.99 | +1.03 | -0.02 | +1.77 |
| PATHS | +0.33 | +0.80 | +0.96 | +2.02 |
| DIFFUSION | +0.09 | +0.30 | +0.53 | +1.77 |
| NEIGHBOURHOOD | -0.06 | +0.30 | +0.02 | +0.00 |

The two qualifying families are not concentrated in the same place. SUPPORT does
its work on the sparse end — isolated and degree_1, where it is worth about a
point each — and does nothing at low_degree. PATHS is the opposite, rising with
connectivity to +2.02 on `ordinary`. Every family is largest on `ordinary`, which
is 198 queries, so that column carries the least weight and the most noise.

The one sharp head-ranking cost in the stratified view is SUPPORT on `ordinary`:
-2.53 R@1 and -2.48 MRR against +1.77 R@5, on those 198 queries. It does not
survive into the pooled figure, where SUPPORT is head-neutral.

## The match

Both directions were checked for every arm: that it carries its own family at
`D6_FULL_13`'s exact values, and that it carries nothing else.

- Every tensor invariant held at `max_abs_diff 0.0` over 4,851,276 candidate
  rows: distance geometry equal to `D6_BASE_13`'s, prior columns equal to the
  injected prior and to `D6_BASE_13`'s, family columns equal to `D6_FULL_13`'s.
- Each arm differs from `D6_BASE_13` in exactly its own family's columns and
  nowhere else — `seed_connections` alone; `paths_length_1`, `paths_length_2`,
  `paths_length_3`; `personalized_pagerank` alone; the common-neighbour column
  alone — with every other residual column exactly zero.
- No residual column is empty, so no invariant was satisfied vacuously:
  occupancy runs from 224,317 rows (`seed_connections`, `paths_length_1`) to
  1,106,161 (`personalized_pagerank`).
- The prior is still D4's, `max_abs_diff 0.0`.
- `candidate_readout`'s `NORMALISED_COLUMNS` remains (4,5,6,7,8,9), not extended;
  the PPR and path definitions are untouched.
- All six arms at 213,689 parameters, head width 61, `local_dim` 13.

## Reuse instead of a fifth and sixth run

`D6_BASE_13` is an **EXACT MATCHED CAUSAL CONTROL** here, not the descriptive
reference D5's `Z2R` was for D6 — it carries the identical architecture, width
and parameter count, so `arm - D6_BASE_13` is a clean information comparison.
That is what makes reuse legitimate rather than convenient.

Because D6's arms were fitted in a different container, the reuse was proved
rather than assumed. D7 rebuilt D6's two blocks locally — masking an array, not
fitting a model — and recomputed every content-sensitive statistic D6 recorded
from its own build. 31 reuse conditions and 7 reproduction conditions, all
matched:

| recomputed | D7 | D6 |
|---|---|---|
| candidate rows | 4,851,276 | 4,851,276 |
| residual columns' nonzero entries | 2,999,730 | 2,999,730 |
| prior rows ranked by both retrievers | 548,724 | 548,724 |
| prior rows ranked by neither | 0 | 0 |
| prior vs D4 `max_abs_diff` | 0.0 | 0.0 |
| seed-identity proof | identical | identical |
| gold stratum counts | identical | identical |

A failure in any of these would have stopped the stage. It would not have been
refitted around.

## Cost

Four fitted arms on one shared build: 50.5 s feature build, 2.0 s static, 14.6 s
assembling and proving the prior, then 18.5 + 17.5 + 17.5 + 17.6 s of training
and 0.34 s of inference each — **139.6 s accounted**, about 0.039 GPU-hours and
$0.07 at $1.734/h, under $0.25 billed with container overhead. The declaration
filed before launch projected ~132 s against an authorisation ceiling of 0.15
GPU-hours and $0.60. The gate's 0.2 h / $0.87 is D1's stage-independent
whole-split ceiling and was never a D7 forecast.

## What this does not establish

Not a pairwise decomposition. The sub-additivity says the families overlap; it
does not say *which* pairs overlap, and nothing here licenses a combinatorial
sweep to find out.

Not a statistical claim. +0.82 against +0.64 is a difference of two point
estimates at one dataset and one seed, and the -1.07 residual is descriptive.

**Not a feature recommendation.** This is the interpretation rule that matters
most here. A family surviving is evidence that its *kind* of information is
useful, never that its QLS-v1 implementation should be carried forward:

- SUPPORT surviving does not mean keep `seed_connections`. That column is a raw
  edge count, a known QLS-v1 representation weakness. What it motivates is
  distinct supporting seeds, support fraction or retrieval-weighted support over
  bounded bitsets.
- PATHS surviving does not mean keep the walk counts, which inflate around hubs
  and cycles. It motivates path diversity, unique predecessor count or
  independent seed support by hop.
- DIFFUSION did not survive, so nothing here argues to retain iterative PPR —
  and it would not have argued for it either way; bounded diffusion would have
  been the question.
- NEIGHBOURHOOD did not survive as this statistic. That does not close cheap
  bounded topology as an idea, only this common-neighbour count.

Not an architecture result: the 13-column block is a diagnostic, three columns
wider and 183 parameters larger than the frozen model, and is not the QLS-v2
input.

One dataset, one seed, development evidence on `G[Cq]`. No GNN, no message
passing, no test split read.

## Verdicts

    SUPPORT          PROMISING
    PATHS            PROMISING
    DIFFUSION        NEGLIGIBLE
    NEIGHBOURHOOD    NEGLIGIBLE
