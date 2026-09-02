# Stage D4: the graded retrieval prior is worth more than seed identity

D3 reduced QLS-v1's ten-column query-local block to one bit and found that bit
carried 87.4% of its R@5 value. The bit is coarse: it says whether a candidate
was in the dense-top-5 union SPLADE-top-5 set, and nothing about rank, retriever
agreement, or the other ~350 candidates in the pool.

D4 keeps that bit and adds graded evidence on top of it.

| arm | carries | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---|---:|---:|---:|---:|---:|
| `SEED_ID_ONLY` | `I[d in Sq]` | 30.15 | 65.27 | 72.22 | 80.73 | 42.77 |
| `SEED_PLUS_GRADED_RETRIEVAL` | that bit + the A3 prior | **39.32** | **69.39** | **73.78** | **93.76** | **45.20** |
| **`delta_retrieval_quality`** | | **+9.18** | **+4.13** | **+1.55** | **+13.03** | **+2.43** |

2wiki_clean, validation, seed 0, points. MEASURED. Both arms 213,506 parameters.

**Verdict: GRADED RETRIEVAL PRIOR DOMINATES NEXT PRIORITY.**

Read against D3's three increments, on the same ladder, at the same operating
point:

| increment | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| `delta_seed` (D3) | +3.60 | **+18.20** | **+10.77** | +10.37 | **+13.47** |
| `delta_distance` (D3) | -0.52 | +1.49 | +5.13 | -0.65 | +8.77 |
| `delta_remaining_structure` (D3) | -0.60 | +1.12 | +0.17 | -0.80 | +0.40 |
| `delta_retrieval_quality` (D4) | **+9.18** | +4.13 | +1.55 | **+13.03** | +2.43 |

On R@5 the graded prior is **3.67x** the remaining structural block and **2.77x**
distance geometry -- far outside the 0.5-1.5 band that would have meant "both
live". On R@1 it is **2.55x seed identity itself**, and on MRR **1.26x**. With
seed identity it is one of only **two** increments measured across D2-D4 that is
positive on all five metrics; both structural increments beyond the bit are
negative on R@1 and MRR.

---

## 1. Three columns beat ten, on precision

`SEED_PLUS_GRADED_RETRIEVAL` carries four non-zero columns and no graph-derived
structure at all. Against D3's full ten-column QLS-v1 block:

| | Z1R | `FULL_LOCAL` | |
|---|---:|---:|---:|
| R@1 | **39.32** | 29.03 | **+10.29** |
| R@5 | **69.39** | 67.88 | **+1.51** |
| MRR | **93.76** | 79.28 | **+14.48** |
| R@20 | 73.78 | **77.53** | -3.75 |
| FullCov@20 | 45.20 | **51.93** | -6.73 |

One bit plus dense rank, SPLADE rank and their agreement beat the entire frozen
query-local block on every precision measure, and lose on both coverage
measures. That split is the stage's most useful finding, and section 2 is what
it means.

The conditional metrics say the same thing more directly. `conditional_hit@1` --
the share of queries whose best in-pool gold is ranked first -- goes from 67.53%
to **88.50%**, and `conditional_hit@20` reaches **100.00%**: with the graded
prior, every validation query with an in-pool gold has one in its top twenty.

A note on scale, since R@1 39.32 beside MRR 93.76 looks contradictory. VERIFIED
FROM CODE (`rank_fusion.ranking_metrics`): `recall@k` is `|gold and top-k| /
|gold|` over *all* golds, while `mrr` is the reciprocal rank of the *first*
gold. `full_coverage@1` is 0.00 in both arms, so every validation query has at
least two golds and R@1 is bounded near 0.5. The two numbers are consistent.

## 2. What it does not buy is coverage

`delta_retrieval_quality` is smallest on exactly the two measures D3's distance
geometry was largest on: +1.55 R@20 against `delta_distance`'s +5.13, and +2.43
FullCov@20 against its +8.77.

This is not a coincidence of magnitudes. Both retrievers rank a candidate by
text similarity to the query; a second gold that no retriever placed highly is
not made findable by telling the model where the retrievers placed it. Finding
it is what seed-distance geometry was for. The two signals are complementary,
and the numbers say so on the metrics that separate them.

So D4 does not close the structural question. It reorders it: the retrieval
prior owns the head of the ranking, seed geometry owns the depth.

## 3. The gain is flat across strata, and that is the mechanism

| `delta_retrieval_quality` | n | R@1 | R@5 | R@20 | MRR |
|---|---:|---:|---:|---:|---:|
| isolated | 834 | +9.14 | +4.32 | +1.17 | +13.24 |
| degree_1 | 750 | +9.07 | +4.07 | +1.50 | +13.82 |
| low_degree | 1218 | +9.24 | +3.92 | +1.66 | +12.40 |
| ordinary | 198 | +9.34 | +4.80 | +2.65 | +13.04 |

Nine points of R@1 and thirteen of MRR on every stratum, within a third of a
point end to end. Neither structural increment behaves like this: `delta_seed`
falls monotonically with connectivity (+23.23 to +7.83 R@5) and
`delta_distance` rises (+1.02 to +9.60 R@20).

That is the signature of a signal that has nothing to do with the graph. The
strata are a graph property -- the degree of the hardest in-pool gold -- and the
retrieval prior is indifferent to it, because it is text evidence. The one
exception is R@20, which does climb with connectivity (+1.17 to +2.65), the
small part of the effect that interacts with the induced structure.

`boundaries_fitted_here: false`; the same four strata, unchanged since D1.

## 4. The columns are A3's, not new ones

VERIFIED FROM CODE, and re-verified inside this run:

| | |
|---|---|
| feature names | `dense_reciprocal_rank`, `splade_reciprocal_rank` |
| source | `mp_retrieval.linear_control.rank_feature_rows`, called as D0B calls it |
| formula | `(K + 1) / (K + zero_based_rank + 1)` |
| constant | `K = 60`, A3's frozen fusion constant |
| missing retriever | exactly `0.0` |
| normalisation | universal: rank 0 is exactly `1.0`, rank 199 is `0.234615`, and halving the ranked list moves no value |
| seeds special-cased | **no** -- the value depends only on rank position |

The universality claim is checked rather than asserted: the runner halves the
list and requires every value unchanged, and confirms that all 200 distinct
ranks survive the local block's float16 storage. A raw rank would fail the
first check.

Only `retriever_agreement` is new. It is 1.0 iff the candidate is ranked by both
retrievers -- a deterministic function of the two A3 columns, each strictly
positive exactly when that retriever ranked the candidate -- and it is recorded
as new rather than presented as an A3 feature.

**Definition B, the full candidate retrieval prior.** Graded evidence covers
every candidate, not just seeds: 4,851,276 candidate rows, **0 ranked by
neither retriever**, 548,724 (11.31%) ranked by both, and 2,151,276 by each
retriever alone. This is possible because `Cq` *is* the top-200 dense union
SPLADE union while `Sq` is only the top-5 union -- 359.35 candidates per query
against 7.83 seeds. Most of the arm's information is therefore about candidates
the seed bit says nothing about, which is why it is named
`SEED_PLUS_GRADED_RETRIEVAL` and not "seed quality".

## 5. What made the comparison legitimate

- **The bit is preserved in both arms**, bit-identical, taken from the same
  `build_local_features` call rather than recomputed. Asserted in the run:
  `seed_bit_identical_across_arms`, 105,675 seed rows, and the control verified
  to carry nothing else. Replacing the bit would have confounded *is graded
  evidence better than the bit?* with *what happens when the bit is removed?*
- **`distance_0` is still exactly `I[d in Sq]`** -- 4,851,276 candidate rows
  compared against the historical `_seed_indicator`, 0 mismatches, checked again
  here rather than inherited from D3.
- **Zeroed slots, not a wider block.** The three retrieval columns occupy
  positions that are exactly zero in `SEED_ID_ONLY`, and none of them is one
  `candidate_readout` rescales, so the stored values are A3's transform
  unmodified and both arms are 213,506 parameters at head width 61.
- **`SEED_ID_ONLY` was not retrained.** It is D3's own row, reused behind a
  25-condition guard covering dataset, fingerprint, context, model, node count,
  candidate contract, local feature schema, candidate rows, static-feature
  source, architecture, epoch-budget policy, that D3's reused row is the
  bare-bit arm and was trained in D3, that the seed-identity proof agrees on
  column, row count, seed rows and mismatches, all three split sizes and all
  five training values. All 25 matched.
- **Identical initialisation and batch order.** `seed_everything(0)` precedes
  `_build_model` in both stages, and `_fit` shuffles with
  `random.Random(seed + epoch * 1_000_003)`, independent of global RNG state.
  The two arms differ in their feature values and in nothing else.

## 6. What is not established

- **Not a converged measurement.** The new arm runs the frozen three-epoch
  QLS-v1 budget and is still descending at the boundary (loss 1.961 -> 1.699 ->
  1.598, held-out tail R@5 0.6840 -> 0.6869 -> 0.6905). That is deliberate: D4
  reuses a D3 arm as its control and must not give itself a convergence
  advantage over it.
- **Not a feature-level attribution.** Three columns move together. The
  agreement column's individual contribution is not separated from the two rank
  columns.
- **Not new information about the world.** `Cq` was itself built by RRF over
  these two ranked lists, so the prior tells the ranker the retrieval evidence
  its own candidate pool was constructed from and which it previously could not
  see. That makes it free at serving time, not novel.
- **Not a claim that structure is unnecessary.** Z1R is worse than `FULL_LOCAL`
  on R@20 and FullCov@20, and D3's remaining six structural columns are still
  lower-priority-but-not-eliminated.
- **One dataset, one seed.** 2Wiki, seed 0. `TARGET_H1` stays closed; the
  context is `G[Cq]`.

## 7. What it decides

- **QLS-v2 starts from the graded retrieval prior, seed identity and minimal
  seed geometry** -- the pre-registered D4-A branch. The prior is the largest
  single lever measured in this sequence on R@1 and MRR, and it costs three
  columns of already-computed retrieval output.
- **Do not jump to a structural sweep.** Coverage is the one place the prior is
  weak, and it is exactly where seed geometry is strong. The developing minimum
  is retrieval prior + seed identity + seed geometry, added one family at a
  time.
- **The remaining six structural statistics stay where D3 left them:**
  lower-priority, to be justified individually, not eliminated. Nothing here
  tests them.
- **Precision and depth still want different things.** The prior maximises R@1
  and MRR; distance geometry buys R@20 and FullCov@20. QLS-v2's objective has to
  model that trade-off rather than pick one end of it.

| | |
|---|---|
| **Cost** | 74.4 s of accounted compute for one new arm (50.7 s shared feature build, 2.1 s static, 2.8 s assembling the prior, 18.5 s training, 0.3 s inference). One run, not two. Container total not captured by the spawn path; the result was on the volume within three minutes of the spawn returning, so under $0.15 at $1.734/h against $0.45 authorised. |
| **Contract** | `gnn_trained: false`, `message_passing: false`, `test_split_read: false`, `epoch_selected_on_validation: false`, `architecture_changed: false`, `seed_bit_replaced: false`, `scored_nodes: exactly Cq in every arm`. |
