# Stage D2: the query-local block is load-bearing, and most of it is one bit

D2 asked one question: **what is the incremental value of QLS-v1's ten
query-local structural columns, under the historical candidate-induced graph?**
Two arms, identical in every other respect -- same architecture, same 213,506
parameters, same seed, same three epochs, same shared feature build. The only
difference is that `NO_QUERY_LOCAL` sees those ten columns as zeros.

**The block is worth 20.8 points of R@5.** That is not a marginal ablation, and
it settles the question D0c left open.

| arm | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| `FULL_CAND` | **29.03** | **67.88** | **77.53** | **79.28** | **51.93** |
| `NO_QUERY_LOCAL` | 26.55 | 47.07 | 61.45 | 70.36 | 29.30 |
| **delta** | **-2.48** | **-20.82** | **-16.07** | **-8.92** | **-22.63** |

2wiki_clean, validation, seed 0, points. MEASURED.

**Verdict: STRUCTURAL BLOCK MATTERS.** This is reading `D2-A`, fixed in advance.

**But the headline number is not a structural finding, and reporting it as one
would be wrong.** All ten columns are seed-anchored, and `sa_mlp` has no other
seed input. Zeroing them does not remove "graph structure" from the model -- it
removes the model's only way of knowing *which candidates the retriever
returned*. The decomposition in section 3 puts roughly nine tenths of the 20.8
points on that single bit.

---

## 1. The control reproduced D1 exactly

`FULL_CAND` is not merely an arm. It is the same configuration as D1's `CAND`,
and it had to reproduce it or the stage would be measuring a different world.

| | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| D1 `CAND` | 29.033 | 67.883 | 77.525 | 79.2816 | 51.933 |
| D2 `FULL_CAND` | 29.033 | 67.883 | 77.525 | 79.2816 | 51.933 |

Bit-identical on all five measures, and the per-epoch train loss matches to
every recorded digit (2.320238 / 1.944303 / 1.699931). This is the third
independent container to produce these numbers. MEASURED.

## 2. Where the block is worth most

`NO_QUERY_LOCAL - FULL_CAND` against the strata frozen before D1, points:

| stratum | n | R@1 | R@5 | R@20 | MRR |
|---|---:|---:|---:|---:|---:|
| isolated | 834 | -5.52 | -22.84 | -14.39 | -14.35 |
| degree_1 | 750 | -3.70 | -23.03 | -18.07 | -11.42 |
| low_degree | 1218 | -0.64 | -19.03 | -16.15 | -5.23 |
| ordinary | 198 | **+3.54** | -14.90 | -15.15 | **+0.69** |
| overall | 3000 | -2.48 | -20.82 | -16.07 | -8.92 |

The block's value **falls as candidate-induced evidence rises**, on R@1, R@5 and
MRR alike. On `ordinary` -- the best-connected stratum -- removing it *improves*
R@1 by 3.54 and leaves MRR unchanged. The strict `non_increasing` test fails on
each of those three because `degree_1` is marginally more damaged than
`isolated`, but the shape across the four strata is not in doubt on R@5, which
spans 22.8 down to 14.9.

`boundaries_fitted_here: false`. These are the same four strata, unchanged.

Two cautions. The `ordinary` stratum is 198 queries. And the D0b lesson applies
to any stratified reading taken at a non-converged budget -- see section 4.

## 3. The decomposition: one bit does most of the work

Every one of the ten columns is anchored on the retrieval seed set:
`distance_0..3_plus_or_unreachable` is distance *from the seeds*,
`seed_connections`, `paths_length_1/2/3`, `personalized_pagerank` and
`common_out_neighbors_with_seed_neighborhood` all measure a candidate's relation
*to the seeds*. VERIFIED FROM CODE, `src/mp_retrieval/linear_control.py:29-40`.

And `sa_mlp` gets nothing else that could carry that signal. It receives static
and local features and no others -- no reciprocal-rank retrieval features, no
separate seed indicator. VERIFIED FROM CODE,
`scripts/run_sa_mlp_confirmation.py:198-205`.

So `NO_QUERY_LOCAL` is a model that **cannot tell which nodes the retriever
returned**. It sees candidate embeddings, query embeddings, their interaction,
and seven query-independent global graph statistics. Nothing else.

Set the frozen ladder beside it, on R@5:

| model | what it has | R@5 |
|---|---|---:|
| `NO_QUERY_LOCAL` | interactions + 7 static, **no seed information at all** | 47.07 |
| `seed_only` | interactions + **one binary seed-membership column** | 65.83 |
| `sa_mlp` = `FULL_CAND` | interactions + 7 static + all 10 seed-anchored columns | 68.40 |

INFERENCE, and the weakest step in this document: the middle row is the test
split at five seeds and head width 65, the top row is validation at seed 0 and
head width 61. Read as a size, not as a measurement.

Read that way, of the 20.8 points the block is worth:

- **roughly +18.8** is bare seed membership, recoverable from a single bit;
- **+2.57** is everything else the structural features add on top of that bit,
  and that figure *is* measured -- five paired seeds, positive at every one,
  Holm-corrected p = 5.0e-4 -- though it also carries the seven static columns
  and drops the seed indicator.

**This reconciles D2 with D0c.** At the linear level, `CAND - RETRIEVAL_ONLY`
was mild: -2.21 R@1, +0.96 R@5. The linear model keeps two reciprocal-rank
features when the local block is removed, so it still knows what was retrieved.
The nonlinear model does not. The two stages are not in tension; they are
ablating different amounts of seed information, and that difference is visible
in the code rather than inferred from the numbers.

## 4. What is not established

- **Not a converged measurement.** Both arms select epoch 3 of 3 -- the
  boundary -- and both are still descending: loss 2.320 -> 1.944 -> 1.700 for
  `FULL_CAND`, 4.858 -> 2.980 -> 2.248 for `NO_QUERY_LOCAL`, with the ablated
  arm still 32% higher at the end and falling faster. Three epochs is the
  *frozen QLS-v1 budget*, and running it is what makes `FULL_CAND` reproduce
  `CAND`; it is not a claim of convergence. So 20.8 points is the block's value
  **at QLS-v1's operating point**, and it is plausibly an overestimate of its
  value to a converged model. D0b was retracted for reading strata off arms
  mid-ascent, and the same caution applies to section 2.
- **Not a decomposition.** D2 moved all ten columns together. The split in
  section 3 is INFERENCE across two splits and two stages, not a measurement.
- **Not evidence about restored context.** Both arms run `CAND`. `TARGET_H1`
  stays closed.
- **One dataset, one seed.** 2Wiki, seed 0.

## 5. What it decides

- **The ten query-local columns are load-bearing in QLS-v1.** Removing them
  costs more than a fifth of R@5. `STRUCTURAL BLOCK MATTERS`.
- **The dominant term is seed membership, not graph structure.** The honest
  target for QLS-v2 is the ~2.6-point band above `seed_only`, not the 20.8-point
  block. Any claim that better statistics on `G[Cq]` are worth twenty points is
  not supported by this stage.
- **The value concentrates where the graph is thin.** Largest on `isolated` and
  `degree_1`, and on `ordinary` the block costs R@1.

| | |
|---|---|
| **Cost** | 88.1 s of accounted compute (49.7 s feature build shared by both arms, 2.1 s static, 18.5 s + 17.1 s training, 0.70 s inference). Container total was not captured by the spawn path; wall clock from launch to `COMPLETE` was under ten minutes, so under $0.29 at $1.734/h against $0.90 authorised. |
| **Contract** | `gnn_trained: false`, `message_passing: false`, `test_split_read: false`, `epoch_selected_on_validation: false`, `architecture_changed: false`, candidate contract `BIT_EXACT_FROZEN_CANDIDATE_EQUIVALENCE`. |
