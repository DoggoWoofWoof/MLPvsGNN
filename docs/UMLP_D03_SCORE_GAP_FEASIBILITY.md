# UMLP-D0.3 — score-gap feasibility of the twin's missed 2wiki secondary golds

Declaration: `configs/umlp_d03_score_gap_feasibility.yaml` (16d637a, filed before any number). Script: `scripts/umlp_d03_score_gap.py`. Tests: `tests/test_umlp_d03_score_gap.py` (7 pass).

**Reading: D03_A_CAPACITY_NOT_THE_LIMIT, with the D03_C_LOCAL_PROBE_NOT_GLOBAL flag raised.**

This phase is analysis only:
- no model was fitted, and v2.2A is unaltered;
- V2_HELD_CONFIRMATION was not read;
- no MetaQA or SQuAD number was read;
- v2.2B is not started (as declared).

## Question

The frozen twin misses some 2wiki secondary golds from its top 5. How large a score correction would bring each one in? D0.3 measures that correction three ways:
- against the twin's own within-query score scale;
- against the most v2.2A was allowed to move (τ = 1);
- against what v2.2A actually learned.

## In short

- **The bound was not the limit.**
  - The median missed gold needs a lift of 0.35 within-query standard deviations to enter the top 5.
  - 88.9% [87.3, 90.4] of the 1,424 missed golds could get in on their own offset alone (required lift < τ).
  - 99.4% fit inside the most favourable correction the bound permits (< 2τ).
  - Only 0.6% need 2τ or more. The declared D03_B threshold was 25%.
- **v2.2A had room but did not aim.**
  - The offset it gave these golds has median +0.19, and even its 95th percentile is only 0.73.
  - Against the nearest displacer, the pair moved the right way in 52.7% [50.0, 55.3] of cases, barely better than a coin flip.
  - The median movement was 0.03 against a median requirement of 0.36.
  - 64.3% moved the wrong way or far too little. The D03_A threshold was > 50%.
- **The GNN-recovered golds were the easiest to fix.**
  - G_miss_mp is the D0.2-positive population. Its golds need a median lift of 0.25, and 95.6% sit within τ.
  - The globally trained offset still improved hit@1 and cut full coverage. This raises the D03_C flag.
- **v2.2A did move the golds; its net coverage loss is a balance.**
  - Of 6,290 queries, full coverage@5 was gained on 223 and lost on 317 (net −94, the filed −0.0149).
  - 196 of the gains are in the 1,293 analysed queries.
  - With pair directions this close to 50/50, the pattern is consistent with an undirected perturbation. It is not evidence of targeted repair.

## Setup

- **Population:** 2wiki V2_GATE, 6,290 queries. Held rows are dropped before compilation.
- **Analysed queries:** those where the twin's (`u_mlp_v2_mix__H128__s0`) rank 1 is a gold and at least one other in-pool gold ranks below 5. There are 1,293 such queries.
- **Unit:** the missed secondary gold (D0.1's `G_miss`). There are 1,424.
  - They sit at twin rank median 9 (q10 6, q90 21).
  - Every one has at least one non-gold displacer at ranks 2–5, so no unit is excluded from the pair readouts.
  - Pairing each with the rank 2–5 non-golds (`N_disp`) gives 4,570 (gold, displacer) pairs.
- **Units of measurement:**
  - ŝ is the twin score z-scored within the query over the full pool. It is exactly v2.2A's `s0_hat`, the units in which τ = 1 is defined.
  - ε = 1e-6.
  - The raw twin-score standard deviation per analysed query has median 2.75, so one ŝ unit ≈ 2.7 raw units.
- **Quantities:**
  - required lift = max(0, ŝ(rank-5 candidate) − ŝ(g) + ε);
  - pairwise requirement r(g, n) = max(0, ŝ(n) − ŝ(g) + ε);
  - learned pair movement m = Δ(g) − Δ(n), with Δ = v2.2A's offset.
- **Statistics:** shares carry query-level percentile bootstrap intervals (1000 resamples, `default_rng(0)`, 95%).
- **Compute:** one forward pass of four frozen checkpoints (twin s0–s2, v2.2A s0). It took 565.7 s on the local CPU with 6 threads.

## Integrity (all held)

- Every per-query metric of every model in the pass equals its stored array on all 6,290 queries (0 mismatches). The stored arrays are the v2 eval for twin s0–s2 and the v2.2A eval for v2.2A s0.
- v2.2A's internal `s0_hat` equals the separately loaded twin s0's within-query z-score. The maximum absolute difference is 0.0.
- Every gold's twin rank equals the D0.1 sidecar's (0 mismatches). The declaration asked this for s0; the check was extended to s1–s2 as a hard stop too.
- D0.1's `groups_for_seed`, run on its own sidecar, yields the same `G_miss`, `G_miss_mp`, `G_miss_both` and `N_disp` as sets of (query, local), for all three seeds.
- v2.2A's per-query mean |Δ| equals the stored eval exactly.
- Every pinned input matched its sha256, and the six frozen configs were byte-unchanged before and after.

## 1. How far below the top 5 the missed golds sit

Twin s0, in ŝ units:

| stratum | missed golds | queries | required lift median (q25 to q75) | q10 / q90 / q95 | < τ (one-sided) | < 2τ (two-sided) | ≥ 2τ | rank-5 is a gold |
|---|---|---|---|---|---|---|---|---|
| all | 1,424 | 1,293 | 0.35 (0.15 to 0.65) | 0.05 / 1.04 / 1.33 | 0.889 [0.873, 0.904] | 0.994 [0.990, 0.998] | 0.006 [0.002, 0.010] | 0.071 [0.058, 0.085] |
| gold count = 2 | 664 | 664 | 0.35 (0.13 to 0.62) | 0.05 / 0.96 / 1.11 | 0.910 [0.887, 0.929] | 0.996 [0.990, 1.000] | 0.004 [0.000, 0.011] | 0.000 |
| gold count ≥ 3 | 760 | 629 | 0.36 (0.16 to 0.68) | 0.06 / 1.13 / 1.48 | 0.871 [0.847, 0.894] | 0.993 [0.987, 0.999] | 0.007 [0.001, 0.013] | 0.133 [0.106, 0.159] |
| first STRUCT support h1 | 1,340 | 1,209 | 0.35 (0.14 to 0.64) | 0.05 / 1.03 / 1.33 | 0.892 [0.875, 0.909] | 0.994 [0.990, 0.998] | 0.006 [0.002, 0.010] | 0.075 [0.061, 0.092] |
| h2 | 56 | 56 | 0.55 (0.22 to 0.88) | 0.11 / 1.12 / 1.26 | 0.821 [0.714, 0.911] | 1.000 | 0.000 | 0.000 |
| h3 | 17 | 17 | 0.40 (0.24 to 0.62) | 0.18 / 0.76 / 1.00 | 0.941 [0.824, 1.000] | 1.000 | 0.000 | 0.000 |
| none | 11 | 11 | 0.60 (0.51 to 0.98) | 0.38 / 1.47 / 1.68 | 0.727 [0.455, 1.000] | 1.000 | 0.000 | 0.000 |
| **G_miss_mp** (GNN top 5 on ≥ 2 of 3 seeds) | 911 | 865 | **0.25** (0.11 to 0.48) | 0.04 / 0.74 / 0.98 | **0.956** [0.944, 0.969] | 0.997 [0.992, 1.000] | 0.003 [0.000, 0.008] | 0.070 [0.054, 0.088] |
| G_miss_both (GNN also misses) | 513 | 471 | 0.60 (0.29 to 0.92) | 0.12 / 1.39 / 1.62 | 0.770 [0.733, 0.807] | 0.990 [0.981, 0.998] | 0.010 [0.002, 0.019] | 0.072 [0.051, 0.098] |

How to read the table:
- The typical missed gold sits a third of a within-query standard deviation below the rank-5 candidate. Even the 95th percentile (1.33) is inside the two-sided bound of 2.
- When the rank-5 candidate is itself a gold (7.1%), lifting g alone swaps one gold for another and does not raise coverage. With exactly two golds this cannot happen, because the rank-5 candidate must then be a non-gold.
- The GNN-recovered golds sit closest to the boundary. The golds the GNN also misses sit about twice as far.

## 2. Against each displacer

The requirement r(g, n) is measured against each non-gold at twin ranks 2–5:
- the **nearest** displacer is the lowest-ranked one, usually the rank-5 candidate;
- the **hardest** is the highest-ranked one.

| stratum | displacer | pairs | r median (q25 to q75) | q10 / q90 / q95 | r < τ | r < 2τ |
|---|---|---|---|---|---|---|
| all | nearest | 1,424 | 0.36 (0.15 to 0.65) | 0.05 / 1.04 / 1.34 | 0.888 [0.871, 0.903] | 0.994 [0.990, 0.998] |
| all | hardest | 1,424 | 0.78 (0.48 to 1.12) | 0.29 / 1.58 / 1.86 | 0.672 [0.648, 0.697] | 0.963 [0.953, 0.972] |
| all | all pairs | 4,570 | 0.56 (0.28 to 0.91) | 0.14 / 1.33 / 1.61 | 0.793 [0.775, 0.810] | 0.981 [0.976, 0.987] |
| G_miss_mp | nearest | 911 | 0.26 (0.12 to 0.49) | 0.04 / 0.75 / 0.98 | 0.956 [0.944, 0.969] | 0.997 [0.992, 1.000] |
| G_miss_mp | all pairs | 2,920 | 0.45 (0.23 to 0.74) | 0.10 / 1.07 / 1.28 | 0.878 [0.861, 0.895] | 0.993 [0.988, 0.997] |
| G_miss_both | nearest | 513 | 0.60 (0.31 to 0.94) | 0.12 / 1.40 / 1.62 | 0.766 [0.729, 0.803] | 0.990 [0.981, 0.998] |
| G_miss_both | all pairs | 1,650 | 0.81 (0.49 to 1.21) | 0.24 / 1.64 / 1.89 | 0.642 [0.604, 0.678] | 0.961 [0.947, 0.974] |

Even full reversal against the hardest displacer is inside the two-sided bound for 96.3% of the missed golds. The remaining strata are in `outputs/umlp_d03/tables.md`.

## 3. What v2.2A actually moved

### Pair movement m = Δ(g) − Δ(n) against the requirement r

The categories, as declared:
- **right way:** m > 0;
- **wrong way:** m < 0;
- **reversed:** v2.2A serves g above n;
- **far too small:** 0 < m < r/2 without reversal;
- **bad:** m ≤ 0, or far too small.

No pair had m = 0.

| stratum | displacer | right way | wrong way | reversed | far too small | bad | m median (q25 to q75) | median m − r | median m / r |
|---|---|---|---|---|---|---|---|---|---|
| all | nearest | 0.527 [0.500, 0.553] | 0.473 [0.447, 0.500] | 0.258 [0.236, 0.281] | 0.170 [0.149, 0.191] | **0.643** [0.618, 0.669] | 0.03 (−0.28 to 0.37) | −0.37 | 0.09 |
| all | hardest | 0.419 [0.393, 0.447] | 0.581 [0.553, 0.607] | 0.089 [0.074, 0.105] | 0.228 [0.206, 0.251] | 0.808 [0.786, 0.829] | −0.09 (−0.40 to 0.25) | −0.88 | −0.11 |
| all | all pairs | 0.479 [0.460, 0.499] | 0.521 [0.501, 0.540] | 0.164 [0.150, 0.178] | 0.207 [0.195, 0.222] | 0.728 [0.711, 0.746] | −0.02 (−0.34 to 0.31) | | |
| gold count = 2 | nearest | 0.533 [0.497, 0.568] | 0.467 [0.432, 0.503] | 0.267 [0.232, 0.301] | 0.166 [0.139, 0.193] | 0.632 [0.596, 0.667] | 0.04 (−0.25 to 0.38) | −0.33 | 0.13 |
| gold count ≥ 3 | nearest | 0.521 [0.478, 0.559] | 0.479 [0.441, 0.521] | 0.251 [0.218, 0.285] | 0.174 [0.147, 0.202] | 0.653 [0.614, 0.688] | 0.03 (−0.32 to 0.36) | −0.40 | 0.07 |
| G_miss_mp | nearest | 0.556 [0.525, 0.590] | 0.444 [0.410, 0.475] | 0.317 [0.288, 0.350] | 0.149 [0.126, 0.171] | 0.593 [0.559, 0.625] | 0.06 (−0.23 to 0.38) | −0.24 | 0.20 |
| G_miss_both | nearest | 0.474 [0.429, 0.519] | 0.526 [0.481, 0.571] | 0.154 [0.122, 0.189] | 0.207 [0.174, 0.242] | 0.733 [0.691, 0.772] | −0.02 (−0.37 to 0.33) | −0.63 | −0.02 |

The learned movement is centred near zero:
- Against the nearest displacer, it is barely better than a coin flip.
- Over all pairs, it is slightly worse than one.
- Against the hardest displacer, it is clearly worse: the offset lifted the rank-2 non-golds more than the missed gold.

The right-way interval clears one half only for G_miss_mp (0.556 [0.525, 0.590]) and, just, for h1 (0.532 [0.505, 0.559]). Even for G_miss_mp, the median movement covers only 20% of what is required.

### The gold's own offset, and whether it entered the top 5

| stratum | Δ(g) median (q25 to q75) | Δ(g) q10 / q90 / q95 | \|Δ(g)\| ≥ 0.9 | Δ(nearest n) median | entered the top 5 | entered, if < τ | entered, if ≥ τ | v2.2A rank median |
|---|---|---|---|---|---|---|---|---|
| all | 0.19 (−0.11 to 0.44) | −0.36 / 0.64 / 0.73 | 0.011 [0.006, 0.017] | 0.15 (−0.11 to 0.39) | 0.203 [0.182, 0.224] | 0.228 [0.206, 0.252] | 0.000 | 9 |
| G_miss_mp | 0.22 (−0.06 to 0.45) | −0.30 / 0.64 / 0.73 | 0.005 [0.001, 0.011] | 0.15 (−0.12 to 0.39) | 0.251 [0.224, 0.280] | 0.263 [0.234, 0.291] | 0.000 | 7 |
| G_miss_both | 0.12 (−0.21 to 0.43) | −0.47 / 0.65 / 0.73 | 0.021 [0.010, 0.035] | 0.15 (−0.10 to 0.38) | 0.117 [0.089, 0.146] | 0.152 [0.121, 0.186] | 0.000 | 12 |
| h2 | −0.11 (−0.43 to 0.26) | −0.76 / 0.42 / 0.48 | 0.018 [0.000, 0.054] | −0.04 (−0.39 to 0.29) | 0.054 [0.000, 0.125] | 0.065 [0.000, 0.130] | 0.000 | 14 |
| h3 | −0.38 (−0.58 to 0.00) | −0.70 / 0.17 / 0.24 | 0.000 | −0.27 (−0.47 to 0.27) | 0.118 [0.000, 0.294] | 0.125 [0.000, 0.312] | 0.000 | 10 |

- **The gold rose, but so did its displacer.** The typical missed gold was raised by 0.19, and its nearest displacer by 0.15.
- **The offset never approached its bound on these golds.** Its 95th percentile is 0.73; only 1.1% of missed golds and 0.4% of nearest displacers reached |Δ| ≥ 0.9.
- **The bound was reached elsewhere in the pool.**
  - In almost every query some candidate got close to it: the per-query max |Δ| has median 0.963 and q10 0.899.
  - Yet only 2.4% of all candidates reach |Δ| ≥ 0.9, and mean |Δ| over the pool is 0.40.
- **On average, the missed golds stayed where they were.** Their mean rank is 11.47 under both the twin and v2.2A.
- **No gold needing ≥ τ entered.**
- **Golds first reached at two or three hops were pushed down** (h2 and h3 median Δ(g) −0.11 and −0.38). These strata are small (56 and 17 golds) and the observation is descriptive.

## 4. Whole-query full coverage@5

Can all of an analysed query's golds be placed in the top 5?

A query is reachable by any reranker only if every gold is in the pool and there are at most 5. Reachability then needs every gold to clear the (6 − G_in)-th highest non-gold.

| stratum | analysed queries | reachable by reranking | unreachable | required median (q25 to q75) | q10 / q90 / q95 | < τ | < 2τ | v2.2A reaches FC@5 |
|---|---|---|---|---|---|---|---|---|
| all | 1,293 | 1,183 | 110 | 0.38 (0.16 to 0.69) | 0.06 / 1.09 / 1.37 | 0.877 [0.856, 0.894] | 0.992 [0.987, 0.996] | 0.152 [0.132, 0.172] |
| gold count = 2 | 664 | 664 | 0 | 0.35 (0.13 to 0.62) | 0.05 / 0.96 / 1.11 | 0.910 [0.887, 0.929] | 0.996 [0.990, 1.000] | 0.190 [0.161, 0.220] |
| gold count ≥ 3 | 629 | 519 | 110 | 0.42 (0.20 to 0.80) | 0.08 / 1.27 / 1.60 | 0.834 [0.801, 0.867] | 0.987 [0.975, 0.996] | 0.111 [0.086, 0.137] |

All 110 unreachable queries are gold count ≥ 3; in each, a gold is outside the pool or there are more than five golds. Among the reachable queries:
- 87.7% need less than τ (every gold lifted, nothing else moved);
- 99.2% need less than 2τ (golds lifted, non-golds pushed down).

v2.2A reached full coverage on 15.2% of all 1,293 analysed queries.

## 5. Gold flows, twin s0 → v2.2A (descriptive)

| population | queries | golds | in twin top 5 | in v2.2A top 5 | entered | left | twin rank-1 gold left the top 5 | twin rank-1 gold lost top-1 | FC@5 gained | FC@5 lost |
|---|---|---|---|---|---|---|---|---|---|---|
| analysed queries | 1,293 | 3,726 | 2,302 | 2,506 | 289 | 85 | 0 | 155 | 196 | 0 |
| queries with ≥ 2 in-pool golds | 6,020 | 14,361 | 12,570 | 12,434 | 343 | 479 | 1 | 626 | 223 | 317 |
| all queries | 6,290 | 14,629 | 12,826 | 12,690 | 344 | 480 | 1 | 629 | 223 | 317 |

- The net is 223 − 317 = −94 queries, and −94 / 6,290 = −0.0149. This is v2.2A's filed full_coverage@5 delta.
- In the multi-gold queries outside the analysed set, v2.2A brought 54 golds in and pushed 394 out.
- The twin's rank-1 gold, which v2.2A's objective protected, left the top 5 once. The other 479 of the 480 golds that left sat at twin ranks 2–5.

These populations are defined by the twin's own outcome. The analysed queries are the ones the twin failed (FC@5 = 0), so any movement can only gain coverage there. Queries the twin covered can only lose it. The flows therefore are not a test of targeting; the pair-direction readout in §3 is, and it is close to 50/50.

## 6. The other twin seeds (capacity only, descriptive)

No v2.2A exists for these seeds, and no reading uses them.

| twin | missed golds | queries | required lift median (q25 to q75) | < τ | < 2τ | ≥ 2τ | G_miss_mp < τ | G_miss_both < τ |
|---|---|---|---|---|---|---|---|---|
| s0 | 1,424 | 1,293 | 0.35 (0.15 to 0.65) | 0.889 [0.873, 0.904] | 0.994 [0.990, 0.998] | 0.006 [0.002, 0.010] | 0.956 | 0.770 |
| s1 | 1,364 | 1,249 | 0.29 (0.13 to 0.60) | 0.884 [0.867, 0.900] | 0.993 [0.989, 0.997] | 0.007 [0.003, 0.011] | 0.954 | 0.776 |
| s2 | 1,487 | 1,355 | 0.35 (0.14 to 0.74) | 0.852 [0.833, 0.870] | 0.984 [0.977, 0.990] | 0.016 [0.010, 0.023] | 0.933 | 0.704 |

The capacity picture does not depend on which twin seed carried v2.2A.

## Reading — the declared rules, applied

All values are point estimates on the s0 missed golds (1,424).

| reading | quantity | value | threshold | holds |
|---|---|---|---|---|
| D03_B_BOUND_LIMITED | share with required lift ≥ 2τ | 0.0056 | ≥ 0.25 | no |
| D03_A_CAPACITY_NOT_THE_LIMIT | share with required lift < τ | 0.8890 | ≥ 2/3 | yes |
|  | share wrong-way or far too small against the nearest displacer | 0.6433 | > 0.5 | yes |
| D03_C_LOCAL_PROBE_NOT_GLOBAL (flag) | G_miss_mp share with required lift < τ | 0.9561 | ≥ 2/3 | yes |
|  | filed v2.2A gate, hit@1 lower bound | +0.0038 | > 0 | yes |
|  | filed v2.2A gate, full_coverage@5 upper bound | −0.0078 | < 0 | yes |

The readings are evaluated in the declared order: D03_B, else D03_A, else D03_NEITHER. D03_B does not hold, and both D03_A conditions do, so **the reading is D03_A_CAPACITY_NOT_THE_LIMIT.** The D03_C flag is evaluated independently, and it holds.

As declared:
- **D03_A:** the v2.2A failure is optimisation, objective or generalisation, not insufficient offset amplitude. v2.2B is not started automatically.
- **D03_C:** the targeted pairwise separability found by D0.2 did not translate into a globally valid full-pool correction. This is evidence against architecture-shopping on that post-hoc subset.

## What this shows and does not show

**What it shows**, for the frozen twin s0 and the one v2.2A fit:
- The correction needed to bring the missed secondary golds into the top 5 is small. The median is 0.35 within-query standard deviations, and it is almost always inside v2.2A's bound.
- v2.2A's learned offset was not aimed at those pairs:
  - its movement against the displacers is centred near zero;
  - it lifts the displacers about as much as the golds;
  - its median is a tenth of what was required.
- The amplitude was available and went unused where the top 5 is decided.

**Why the D03_C flag bites:** the golds D0.2 found linearly separable (G_miss_mp) were the easiest of all to fix, at 95.6% within τ. Yet even on them, the trained global correction moved the right way only 55.6% of the time. Its gains in the queries the twin failed were outweighed by losses in the queries the twin had covered.

**What it does not show:**
- It is not evidence that no correction can work.
- It rests on one v2.2A fit (seed 0), one τ, one linear form, and the 2wiki V2_GATE half only. The held half and the MetaQA and SQuAD numbers were not read.
- The small strata (h2, h3 and none: 56, 17 and 11 missed golds) have wide intervals; what they show is descriptive.

**For the registered question**, nothing changes. D0.3 is a diagnostic. The coverage share of the 2wiki message-passing gap remains unrecovered by the matched non-message-passing scorer. D0.3 adds that the bound on the tried correction is not the reason.

## Next

**STOP_FOR_REVIEW.** v2.2B is not started, as both the declaration and the ruling say.

Any follow-up needs its own declaration first. It could be a change on the objective or generalisation side, or closing the coverage branch.

## Files

- Declaration: `configs/umlp_d03_score_gap_feasibility.yaml`. Its run record is `run_record_umlp_d03_2026_09_29` and its status is RUN.
- Sidecar: `outputs/umlp_d03/gaps_2wiki_gate.npz`, sha256 `c45985a74478447e627a1fd117e8cb80d3f038919916f021610ac62b314c63e9`. It holds 184,275 kept candidates: every in-pool gold plus each model's top 20.
- Record: `outputs/umlp_d03/record.json`, sha256 `3dc98d1dd586411f81f306fd65aa3840b15df637e5108a0ad8dca91d94ca08b3`.
- Every stratum: `outputs/umlp_d03/tables.md`.
- Scoring-pass integrity: `outputs/umlp_d03/score_meta.json`.
- `outputs/` is untracked; every file is pinned by the sha256 above.
