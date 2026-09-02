# Stage D5: the retrieval prior and seed geometry compose

D4 found the graded retrieval prior larger than any structural increment on
precision and smaller than distance geometry on both coverage measures, and read
that as complementary: the prior owning the head of the ranking, geometry owning
the depth. D5 tests the composition directly by putting D3's distance group and
D4's prior in the same block.

| | arm | carries | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---|---|---:|---:|---:|---:|---:|
| `Z1` | `SEED_ID_ONLY` | the bare bit | 30.15 | 65.27 | 72.22 | 80.73 | 42.77 |
| `Z2` | `DISTANCE_ONLY` | distance buckets 0-3 | 29.63 | 66.76 | 77.36 | 80.08 | 51.53 |
| `Z3` | `FULL_LOCAL` | all ten columns | 29.03 | 67.88 | 77.53 | 79.28 | 51.93 |
| `Z1R` | `SEED_PLUS_GRADED_RETRIEVAL` | bit + prior | 39.32 | 69.39 | 73.78 | 93.76 | 45.20 |
| `Z2R` | `DISTANCE_PLUS_GRADED_RETRIEVAL` | geometry + prior | **39.56** | **71.44** | **77.81** | **94.18** | **52.13** |

2wiki_clean, validation, seed 0, points. All five arms 213,506 parameters. Four
rows reused; **one new training run**, `Z2R`.

**Verdict: RETRIEVAL + GEOMETRY COMPOSE STRONGLY.**

`Z2R` is the best of the five arms on every one of the five metrics, and both
conditionals are positive on every one of the five:

| increment | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| **`delta_prior_given_geometry`** = `Z2R - Z2` | +9.93 | +4.68 | +0.45 | +14.10 | +0.60 |
| **`delta_geometry_given_prior`** = `Z2R - Z1R` | +0.23 | +2.05 | +4.03 | +0.42 | +6.93 |
| **`interaction`** = `(Z2R - Z1R) - (Z2 - Z1)` | +0.75 | +0.56 | -1.10 | +1.06 | -1.83 |

Against the earlier increments, on the same ladder at the same operating
point:

| increment | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| `delta_seed_d3` | +3.60 | +18.20 | +10.77 | +10.37 | +13.47 |
| `delta_distance_d3` | -0.52 | +1.49 | +5.13 | -0.65 | +8.77 |
| `delta_remaining_structure_d3` | -0.60 | +1.12 | +0.17 | -0.80 | +0.40 |
| `delta_retrieval_quality_d4` | +9.18 | +4.13 | +1.55 | +13.03 | +2.43 |

Each of the two D5 increments is the conditional version of one of the last two
rows. The prior's R@5 value grows slightly when geometry is present (+4.68
against D4's +4.13) and so does geometry's (+2.05 against D3's +1.49), while
geometry's coverage value shrinks (+4.03 against +5.13 R@20, +6.93 against +8.77
FullCov@20).

Neither signal is redundant given the other. The prior keeps essentially all of
its precision value once geometry is present (+9.93 R@1 against D4's +9.18,
+14.10 MRR against +13.03), and geometry keeps its depth value once the prior is
present (+4.03 R@20 and +6.93 FullCov@20).

---

## 1. Seven columns beat all ten, on every metric

D4's sharpest finding was a trade: `Z1R` beat the entire frozen ten-column block
on precision and lost to it on both coverage measures. Adding the four distance
columns closes that gap and reverses it.

| | `Z2R` | `Z3` `FULL_LOCAL` | |
|---|---:|---:|---:|
| recall@1 | **39.56** | 29.03 | +10.53 |
| recall@5 | **71.44** | 67.88 | +3.56 |
| recall@20 | **77.81** | 77.53 | +0.28 |
| mrr | **94.18** | 79.28 | +14.90 |
| full_coverage@20 | **52.13** | 51.93 | +0.20 |

Four distance buckets plus three retrieval columns beat all ten historical
columns on every measure reported. That is the concrete form of the D4-A branch:
the minimal QLS-v2 query-local base is the graded retrieval prior plus seed
identity plus seed-distance geometry, and `distance_0` *is* the seed bit, so the
seven columns already contain it.

The R@20 and FullCov@20 margins are small (+0.28, +0.20) and should be read as
"no longer behind" rather than "ahead". R@20 is also close to the pool's own
limit: `candidate_ceiling` is 79.33 and `Z2R` reaches 77.81, so 98.1% of the
achievable recall@20 is taken and there is little room left for any signal to
show a large R@20 increment. `delta_prior_given_geometry`'s +0.45 on R@20 is
partly that ceiling, not only a property of the prior.

## 2. The interaction has two signs, and they say different things

`interaction` is positive on R@1 (+0.75), R@5 (+0.56) and MRR (+1.06), and
negative on R@20 (-1.10) and FullCov@20 (-1.83).

On precision the two signals are more than additive. D3 measured
`delta_distance` as *negative* on R@1 (-0.52) and MRR (-0.65): adding distance
geometry to the bare bit made the head of the ranking slightly worse. With the
prior present the same geometry is positive on both (+0.23, +0.42). Geometry
stops costing precision once the model has direct retrieval evidence to rank
with.

On depth they partly overlap. Geometry is worth +5.13 R@20 and +8.77 FullCov@20
over the bare bit, but +4.03 and +6.93 over the bit plus the prior, so roughly a
fifth of its coverage value is information the prior already carried.

This is descriptive. One dataset, one seed, one operating point, no repeated
measurements: these are the signs of five differences, not an estimated
interaction term, and nothing here supports a claim about their variability.

## 3. The two increments have opposite stratum signatures

| `delta_prior_given_geometry` | n | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|---:|
| isolated | 834 | +9.17 | +4.86 | +0.78 | +13.56 | +0.96 |
| degree_1 | 750 | +9.93 | +5.53 | +0.17 | +15.06 | +0.27 |
| low_degree | 1218 | +10.20 | +4.35 | +0.51 | +13.64 | +0.74 |
| ordinary | 198 | +11.36 | +2.78 | -0.25 | +15.50 | -0.51 |

| `delta_geometry_given_prior` | n | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|---:|
| isolated | 834 | +0.27 | +0.42 | +0.63 | +0.51 | -0.24 |
| degree_1 | 750 | +0.10 | +1.63 | +3.93 | +0.23 | +6.27 |
| low_degree | 1218 | +0.04 | +3.35 | +5.99 | +0.15 | +11.25 |
| ordinary | 198 | +1.77 | +2.53 | +6.69 | +2.40 | +13.13 |

The prior's precision gain is flat across strata -- nine to eleven points of R@1
and thirteen to sixteen of MRR everywhere -- and its coverage gain is near zero
everywhere, turning slightly negative on the most connected stratum. Geometry's
gain is the mirror image: near zero on R@1 and MRR, and rising monotonically
with connectivity on R@20 (+0.63 to +6.69) and FullCov@20 (-0.24 to +13.13).

The strata are a graph property, the degree of the hardest in-pool gold. A
signal indifferent to it is text evidence; a signal that scales with it is using
the induced structure. Each increment behaves as its own account of itself
predicts, which is the strongest evidence in this stage that the two are
carrying different information rather than two views of the same thing.

`boundaries_fitted_here: false`; the same four strata, unchanged since D1, and
identical query counts to D4.

## 4. Why `Z2R - Z1R` is a geometry comparison

`Z2R` places the prior at columns 4-6. VERIFIED FROM CODE,
`candidate_readout` divides columns 4-9 by their own per-query maximum and
leaves 0-3 untouched, so a prior written into those slots *before* that
normalisation would not be the prior D4 measured, and this stage's headline
difference would confound adding geometry with changing the retrieval transform.

The runner injects after `build_local_features` returns, which is after
`candidate_readout` has run, and then proves the result rather than asserting
it:

| invariant | how | outcome |
|---|---|---|
| prior == D4's | D4's own `build_graded_retrieval_block`, called from D5 | `max_abs_diff` **0.0**, elementwise identical, 4,851,276 rows |
| geometry == D3's | D3's own `masked_local` over `DISTANCE_COLUMNS` | elementwise identical, complete one-hot, slots 7-9 exactly zero |

So `Z2R` is exactly D3's geometry plus exactly D4's prior and nothing else. The
prior is unchanged in every respect that could have moved it: `K = 60`, the same
`(K + 1) / (K + rank + 1)` transform, exactly `0.0` for a missing retriever,
agreement 1.0 iff both retrievers ranked the candidate, no second normalisation,
no percentile transform, no seed-only masking.

**A measured caveat about that fix, recorded because it is true.** On this data
the pre-normalisation layout would have produced the same values anyway. Each
prior column's per-query maximum is exactly 1.0 -- rank 0 is always in `Cq`, and
agreement reaches 1 whenever the two lists overlap -- so `candidate_readout`
would have divided by one. MEASURED: max absolute difference 0.0 over the prior
columns, with per-query column maxima `[1. 1. 1.]`; a probe column with maximum
0.5 was rescaled to 1.0 in the same call, so the mechanism is real and only its
effect here is nil. That is a coincidence of the reciprocal-rank transform, not
a guarantee: a different `K`, a percentile transform, or a pool that did not
contain rank 0 would each break it. The ordering and the proof are kept because
depending on the coincidence would be fragile, and the fix is not credited with
averting a corruption it did not avert.

## 5. What made the five-row table legitimate

- **One new run.** `Z1`, `Z2` and `Z3` are D3's rows, `Z1R` is D4's. 54 guarded
  conditions were checked against *both* source files -- dataset, fingerprint,
  context, model, node count, candidate contract, local feature schema,
  candidate rows, static-feature source, architecture, epoch-budget policy, the
  seed-identity proof's column, row count, seed rows and mismatches, all three
  split sizes and all five training values, on each source -- and all 54 matched.
- **The chain is checked, not assumed.** D4 reused D3's `SEED_ID_ONLY`, so the
  guard also requires D3 and D4 to report the same numbers for it. A
  disagreement would mean the table was assembled across a break somewhere
  behind this stage.
- **`Z3` is a two-hop row and is labelled as one.** D3 did not train
  `FULL_LOCAL` either; it reused D2's `FULL_CAND`. Its `measured_in` says
  `stage_d2`, not the file D5 read it out of, and the guard follows it back
  there.
- **`distance_0` is still exactly `I[d in Sq]`** -- 4,851,276 candidate rows
  compared elementwise against the historical `_seed_indicator`, 105,675 seed
  rows, 0 mismatches, re-checked here.
- **Zeroed slots, not a wider block.** The prior occupies three columns that are
  exactly zero in `DISTANCE_ONLY`, so `local_dim` stays 10, head width stays 61,
  and all five arms are 213,506 parameters. The increments measure information,
  not capacity.
- **Identical initialisation and batch order.** `seed_everything(0)` precedes
  `_build_model` in every stage, and `_fit` shuffles with
  `random.Random(seed + epoch * 1_000_003)`, independent of global RNG state.

## 6. What is not established

- **Not a converged measurement.** `Z2R` runs the frozen three-epoch budget and
  is still improving at the boundary: loss 1.846 -> 1.528 -> 1.415, held-out
  tail R@5 0.6938 -> 0.7064 -> 0.7088. Deliberate, since four of the five rows
  are earlier stages' and this arm must not be given a convergence advantage
  over its own controls.
- **Not a statistical interaction claim.** The interaction row is five
  differences of differences at one dataset, one seed, one operating point.
  Signs, not estimates.
- **Not a verdict on the remaining six structural columns.** D3's
  `delta_remaining_structure` was +1.12 R@5 measured against a much weaker
  baseline than this one. Whether any of it survives the prior is exactly what
  `Z3R` would test, and this stage does not answer it. That number cannot be
  used as the justification for further structural work in the meantime.
- **Not new information about the world.** `Cq` was built by RRF over the same
  two ranked lists, so the prior reports retrieval evidence the ranker
  previously could not see. That makes it free at serving time, not novel.
- **Not a feature-level attribution.** Three retrieval columns and four distance
  columns each move as a group.
- **One dataset, one seed.** 2Wiki, seed 0, context `G[Cq]`. `TARGET_H1` stays
  closed.

## 7. What it decides

- **QLS-v2's minimal query-local base is settled at seven columns:** the graded
  retrieval prior, seed identity, and the seed-distance buckets. It beats the
  full historical block on every metric at identical parameter count.
- **The head/depth trade-off D4 reported is not a permanent property of the
  method.** It was a consequence of having only one of the two signals. With
  both, one arm holds the best value on all five metrics, so QLS-v2's objective
  does not have to choose an end of it.
- **Geometry is not redundant given retrieval**, and this now rests on a
  conditional measurement rather than on D3's unconditional one.
- **Do not open a wide feature grid on the strength of this.** Two families
  composing is not evidence that six more will. `Z3R` -- `FULL_LOCAL` plus the
  exact prior -- is the one question this result raises. It is not launched from
  here and needs its own declaration.

| | |
|---|---|
| **Cost** | 76.6 s of accounted compute for one new arm (49.8 s shared feature build, 1.9 s static, 6.4 s assembling and proving the prior, 18.2 s training, 0.3 s inference). One run. Container total not captured by the spawn path; the result was on the volume within minutes of the spawn returning, so under $0.15 at $1.734/h against $0.45 authorised. The gate reported 0.2 h / $0.87, byte-identical to D3's and D4's figures: VERIFIED FROM CODE, every fitted graph-context stage is priced with D1's whole-split model, which the source documents as a ceiling the later stages are bounded by, so it is not a D5-specific projection. |
| **Contract** | `gnn_trained: false`, `message_passing: false`, `test_split_read: false`, `epoch_selected_on_validation: false`, `architecture_changed: false`, `scored_nodes: exactly Cq in every arm`. |
