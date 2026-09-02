# Stage D6: the six residual columns survive the retrieval prior, at a price

D3 measured QLS-v1's six non-distance structural columns at +1.12 R@5 against a
baseline with no retrieval evidence at all. D5 then showed the graded retrieval
prior is far stronger than any structural increment, which raised an obvious
possibility: that the six were compensating for the missing prior rather than
adding independent graph knowledge. D6 re-measures them conditionally.

The six under test are the whole historical block minus the distance buckets:

    seed_connections   paths_length_1   paths_length_2   paths_length_3
    personalized_pagerank   common_out_neighbors_with_seed_neighborhood

| arm | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| `D6_BASE_13` | **39.39** | 71.38 | 77.70 | **93.88** | 51.90 |
| `D6_FULL_13` | 38.77 | **72.28** | **77.94** | 93.18 | **52.43** |
| **the increment** | -0.63 | **+0.89** | +0.24 | -0.70 | +0.53 |

2wiki_clean, validation, seed 0, points. **Two new runs**, both 213,689
parameters at head width 61 and `local_dim` 13. The arms differ in exactly the
six columns under test and in nothing else.

**Verdict: REMAINING STRUCTURE MODEST / FAMILY TEST WARRANTED.**

The compensation hypothesis is wrong. Set the conditional increment beside D3's
unconditional one:

| | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| **the increment** (given retrieval + geometry) | -0.63 | +0.89 | +0.24 | -0.70 | +0.53 |
| D3 unconditional (given the bare bit only) | -0.60 | +1.12 | +0.17 | -0.80 | +0.40 |

The whole shape survives: negative on R@1 and MRR, its largest positive on R@5,
small positives on both coverage measures. R@5 shrinks by 0.23 points, while
R@20 and FullCov@20 both grow slightly. Whatever these six columns carry, the
graded retrieval prior did not already contain it -- which is a stronger result
for structure than D5 alone would have suggested, and a smaller one than the
+1.12 was ever going to justify on its own.

---

## 1. It is a trade, not a gain

`D6_FULL_13` is not the better arm. It wins R@5, R@20 and FullCov@20 and loses
R@1 (-0.63) and MRR (-0.70). Adding all six columns to a model that already has
graded retrieval evidence and seed geometry makes the head of the ranking worse.

This is the same sign pattern D3 saw, and D5 explains why it should be expected:
the retrieval prior owns precision, and structural features that push a
candidate up the list on graph evidence displace candidates the retrievers
ranked highly. What D5 showed for the distance buckets -- that they stop costing
precision once the prior is present -- does **not** extend to the other six.
They still cost it.

So "modest" here means modest and mixed, not small and free. Any QLS-v2 that
adopts these columns wholesale would be trading roughly 0.6 R@1 and 0.7 MRR for
roughly 0.9 R@5 and 0.5 FullCov@20, and nothing in this stage says that trade is
worth making.

## 2. The gain is concentrated where the graph is

| stratum | n | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|---:|
| isolated | 834 | -0.30 | +0.36 | -0.12 | -0.28 | -0.24 |
| degree_1 | 750 | -0.37 | +1.23 | +0.20 | -0.62 | +0.53 |
| low_degree | 1218 | -1.19 | +0.90 | +0.39 | -1.30 | +0.82 |
| ordinary | 198 | +0.51 | +1.77 | +1.01 | +0.95 | +2.02 |

`ordinary` -- the most connected stratum, and the smallest at 198 queries -- is
the only one positive on all five metrics, and it holds the largest value on
every one of them. `isolated` is negative on four of five. Coverage tracks
connectivity monotonically (-0.24, +0.53, +0.82, +2.02).

That is what features built from support counts, path counts, PPR and common
neighbours should look like: they need a neighbourhood to measure. It also means
the headline +0.89 R@5 is an average over strata where the effect ranges from
+0.36 to +1.77, and that the stratum carrying the strongest signal is 6.6% of
the validation set. Read the direction, not the third digit.

`low_degree` is the awkward row: the second-largest R@5 gain (+0.90) alongside
the worst R@1 (-1.19) and MRR (-1.30). The trade in section 1 is mostly being
paid there.

`boundaries_fitted_here: false`; the same four strata, unchanged since D1, and
identical query counts to D5.

## 3. Why two arms, and what that cost

`FULL_LOCAL` already occupies all ten local slots. There is no exact
representation of the ten historical features plus the three-column retrieval
prior inside ten dimensions: the prior would have to overwrite columns 4-6 --
`seed_connections`, `paths_length_1`, `paths_length_2` -- which are three of the
six features this stage exists to measure. Comparing a 13-column treatment
against D5's reused 10-column `Z2R` would instead attribute the dimensionality
and parameterisation difference to graph structure. At an effect near one R@5
point, neither is acceptable.

So D6 trains both of its arms in one widened representation:

| | `D6_BASE_13` | `D6_FULL_13` |
|---|---|---|
| cols 0-3 | exact D3 `DISTANCE_ONLY` geometry | (same) |
| cols 4-9 | **exactly zero** | **the six historical columns** |
| cols 10-12 | the exact D4/D5 retrieval prior | (same) |

| invariant | outcome |
|---|---|
| `D6_BASE_13[:,0:4] == D3 DISTANCE_ONLY[:,0:4]` | `max_abs_diff` 0.0, identical, 4,851,276 rows |
| `D6_FULL_13[:,0:10] ==` historical `FULL_LOCAL[:,0:10]` | `max_abs_diff` 0.0, identical |
| both arms `[:,10:13] ==` the D4 prior | `max_abs_diff` 0.0, identical |
| `D6_BASE_13[:,4:10] == 0` | exactly zero |
| the arms differ in | columns `[4, 5, 6, 7, 8, 9]` and no others |

`candidate_readout.NORMALISED_COLUMNS` stays `(4, 5, 6, 7, 8, 9)` and is not
extended to the appended columns, so the prior keeps the exact A3 values D4 and
D5 measured and this does not quietly become a second normalisation experiment.
The runner refuses if that tuple is not the frozen one.

**The parameter cost, disclosed rather than absorbed.**

| | input_dim | head width | parameters |
|---|---:|---:|---:|
| historical (`local_dim` 10) | 275 | 61 | 213506 |
| D6, both arms (`local_dim` 13) | 278 | 61 | **213689** |
| difference | +3 | 0 | **+183** (+0.0857%) |

The head width is taken from the historical ten-column solve and held fixed.
`build_explicit_feature_mlp` would otherwise re-solve it for the wider input and
return head 60 at 213,409 parameters -- a *narrower* head on a wider input,
changing capacity and information together. Both D6 arms carry exactly the
widened architecture, so the increment is six columns of information and not
capacity. **This is a diagnostic model used to isolate information value. It is
not the final architecture** and nothing here proposes it as one.

## 4. Widening the input changed almost nothing

`D6_BASE_13` carries the same information as D5's `Z2R` at a different width, so
the gap between them measures the widening rather than any feature.

| | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| `D6_BASE_13` | 39.39 | 71.38 | 77.70 | 93.88 | 51.90 |
| D5 `Z2R` | 39.56 | 71.44 | 77.81 | 94.18 | 52.13 |
| difference | -0.17 | -0.06 | -0.11 | -0.30 | -0.23 |

Every difference is under a third of a point, and all five are slightly
negative. Cross-stage reading between D5 and D6 is therefore safe at this
resolution, and D5's ladder is quoted below on that basis. It is quoted as a
**DESCRIPTIVE REFERENCE, NOT CAUSAL CONTROL**: those arms have `local_dim` 10
and 213,506 parameters, and are the control for nothing in this stage. Both D6
increments come from two arms trained here.

| D5 arm (descriptive) | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| `SEED_ID_ONLY` | 30.15 | 65.27 | 72.22 | 80.73 | 42.77 |
| `DISTANCE_ONLY` | 29.63 | 66.76 | 77.36 | 80.08 | 51.53 |
| `FULL_LOCAL` | 29.03 | 67.88 | 77.53 | 79.28 | 51.93 |
| `SEED_PLUS_GRADED_RETRIEVAL` | 39.32 | 69.39 | 73.78 | 93.76 | 45.20 |
| `DISTANCE_PLUS_GRADED_RETRIEVAL` | 39.56 | 71.44 | 77.81 | 94.18 | 52.13 |

For scale, the increments measured across this sequence:

| increment | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| `delta_distance_d3` | -0.52 | +1.49 | +5.13 | -0.65 | +8.77 |
| `delta_retrieval_quality_d4` | +9.18 | +4.13 | +1.55 | +13.03 | +2.43 |
| `delta_prior_given_geometry_d5` | +9.93 | +4.68 | +0.45 | +14.10 | +0.60 |
| `delta_geometry_given_prior_d5` | +0.23 | +2.05 | +4.03 | +0.42 | +6.93 |
| **D6, six residual columns** | -0.63 | +0.89 | +0.24 | -0.70 | +0.53 |

The six columns are the smallest positive R@5 contribution in the sequence, at
less than half the conditional value of the four distance buckets and under a
fifth of the prior's.

## 5. What is not established

- **Not a decomposition.** Six columns from four different families -- support
  counts, path counts, diffusion, common-neighbour topology -- moved as one
  group. Nothing here says which of them produced +0.89 R@5, or whether one
  family produced all of it while another cost the R@1.
- **Not a statistical claim.** One dataset, one seed, one operating point, no
  repeated measurements. +0.89 against D3's +1.12 is a difference of two point
  estimates and the gap between them is not resolvable at this resolution.
- **Not a converged measurement.** Both arms ran the frozen three-epoch budget.
  VERIFIED FROM CODE, `_fit` restores the best held-out epoch, and both arms
  peaked at epoch 2 (`D6_BASE_13` 0.7090 then 0.7081; `D6_FULL_13` 0.7181 then
  0.7155), so neither was favoured by selection and the budget was not the
  binding constraint here -- unlike D4 and D5, which were still rising at the
  boundary.
- **Not an architecture result.** The 13-column block is a diagnostic. It is
  three columns wider and 183 parameters larger than the frozen model, and is
  not proposed as QLS-v2's input.
- **One dataset, one seed.** 2Wiki, seed 0, context `G[Cq]`. `TARGET_H1` stays
  closed.

## 6. What it decides

- **The pre-registered D6-B branch.** +0.89 R@5 sits inside the +0.5 to +1.5
  band, with a real coverage benefit and a real precision cost. Remaining
  structure is live and must justify itself family by family.
- **Do not add all six to QLS-v2.** The group as a whole is a trade, and its
  gain is concentrated in the most connected 6.6% of queries.
- **Do not open the old 30-35 feature frontier either.** Nothing here supports
  going wider; the question is whether any *existing* family earns its place.
- **The next decomposition is four grouped families**, against the strong
  retrieval + geometry base: support (`seed_connections`), paths
  (`paths_length_1..3`), diffusion (`personalized_pagerank`), and
  common-neighbour topology
  (`common_out_neighbors_with_seed_neighborhood`). One family at a time.
- **The global-context branch stays closed** whatever the outcome, since
  `TARGET_H1` already failed its controlled effectiveness test.

| | |
|---|---|
| **Cost** | 96.5 s of accounted compute for two new arms (49.5 s shared feature build, 1.9 s static, 9.5 s assembling and proving the prior, 18.0 s + 16.9 s training, 0.3 s + 0.3 s inference). Two runs, as declared. Against the declared ceilings of 0.20 GPU-hours and $0.90, and the measured expectation of ~95 s / $0.05 filed before launch. Container total not captured by the spawn path; under $0.15 billed at $1.734/h. The gate reported 0.2 h / $0.87 again, which is D1's stage-independent whole-split ceiling and not a D6 forecast. |
| **Contract** | `gnn_trained: false`, `message_passing: false`, `test_split_read: false`, `epoch_selected_on_validation: false`, `architecture_changed_between_arms: false`, `scored_nodes: exactly Cq in every arm`. 19 reference conditions checked against D5, all matched. |
