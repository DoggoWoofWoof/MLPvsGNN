# U2v: a learned cascade, stage 0 (how deep in a model's own order do the golds sit?)

Declared 30 October 2026, before any U2v number. This file declares stage 0 only, with its rule, and names what each
outcome leads to. Code: `outputs/mp_unified/zu2v.py` (self-test passed before this commit). No training, no new fit, no
GPU, no change to any pool.

## 1. Why this, now

- **The user's idea (30 October):** do not only try fixed rules for the pool. Let the MLP or the GNN do both jobs: choose a
  better, smaller pool and rank the pool it chose, perhaps trained in alternation (EM), and then take what it learned
  back into a better fixed, more general rule set. This relaxes my own "no learning until the pools are final" for
  declared stages. It is P1h's named next form ([P1H_FEWER_ROWS.md](P1H_FEWER_ROWS.md) section 5, NO_CUT: a learned
  cascade, a cheap scorer's per-question cut).
- **What the diagnostics left open** ([U2T_STORED_ARRAY_DIAGNOSTIC.md](U2T_STORED_ARRAY_DIAGNOSTIC.md)): the shared
  construction costs about 3,300 rows at level 0.95 because of musique's tail (12.4% of its questions need 1,000 or more
  walk nodes); the walk's orders are not the lever (the best of 14 per question still needs 2,392 rows); no stored
  measure, and no first-stage signal beyond a weak one, predicts a question's depth; a budget that knew each question's
  depth would cost 111 rows. What none of those used is a trained model's own score, which sees the row's features.
- **The question stage 0 asks:** in the order a trained model gives the pool's rows, how far down is a question's last
  gold? If the golds sit in the model's top 200 where the walk needs 3,000, a model could cut the pool for the heavy
  stage. If they sit as deep as in the walk's order, a cascade of this kind has nothing to cut.
- **What a cascade would and would not do.** Reach is set by the walk pool, which a cascade does not build. A cut
  reduces the rows the heavy stage scores; the walk and the cheap features of every pool row are still computed, so the
  cold build is unchanged. Rows below are rows kept for the heavy stage.

## 2. What is read

- **Models:** the step-1 lean MLP fits that are on this laptop, candidate `p@swa` (lean_gpu's `models.pt`, SWA of the
  pick:none variant): `L-musique` (trained on metaqa, squad, hotpotqa, 2wiki; reads musique zero-shot) and `L-metaqa`
  (trained on squad, musique, hotpotqa, 2wiki; reads metaqa zero-shot). webqsp is zero-shot for both. squad, hotpotqa
  and 2wiki are read in-domain by both (their eval carve holds questions the fit never trained on, but the dataset is
  seen); those three are flagged in-domain wherever they appear.
- **Pools:** the step-1 feature cache's pools (P1c2's, at each question's frozen size), carve `s1eval` of each of the six
  datasets, from `outputs/step1/cache`. Not PKO's pools; not the current zrckpt or zg6r.
- **Five orders of a question's pool rows:** the model's score; the first stage's rrf (`base_z`); the pool's stored row
  order (`pool`; not asserted to be the walk's order); and the cache's two references, gnn0 and twin0. Ties follow
  `lean_gpu.top_hit` (descending score, NaN last, ties by pool position).
- **Per question:** need(order) is the 1-based rank of its last gold row among the pool's rows in that order. A question
  with a gold (`gold_total > 0`) is *reached* when the pool holds all of `gold_total` gold rows; an unreached question
  fails at every K. `ALL(order, K)` is the share of the questions with a gold that are reached and have need(order) <= K;
  `rows(K)` is their mean of min(pool size, K). Grid: K in {50, 100, 200, 300, 500, 1000, 2000}.
- **A per-question cut from the model's own scores:** keep the shortest prefix of its sorted scores whose softmax mass
  (float64, the listwise loss's own logits) reaches tau, at least 5 rows and at most the pool. Grid: tau in {0.90, 0.95,
  0.99, 0.999}. `ALL(mass, tau)` and its mean kept rows are read the same way.

## 3. The rule (fixed now)

Reads use U2s's weights (metaqa 15,745, squad 17,729, musique 7,018, hotpotqa 13,335, 2wiki 18,504) for the weighted mean
over the five datasets; webqsp is read beside and decides nothing.

- **K\* of an order** is the smallest K on the grid at which, on each of the five datasets, `ALL(order, K) >= reach - 0.005`
  (reach is the pool's own share of reached questions). `none` when no K up to 2000 does.
- **rows\* of an order** is the weighted mean of `rows(K*)` over the five datasets; **pool** is the weighted mean of the
  pools' mean sizes.
- **Verdict, per fit.**

  | | |
  | --- | --- |
  | CASCADE_PROMISING | the model's K\* <= 500, its rows\* <= 0.5 x pool, and its K\* is below the smaller of rrf's and the pool order's K\* (a `none` counts as 4,000) |
  | CASCADE_PARTIAL | the model's K\* <= 1,000 and below that same smaller K\* |
  | NO_CASCADE | otherwise |

- **Stage verdict:** the worse of the two fits' verdicts. The zero-shot reads that matter most (musique under L-musique,
  metaqa under L-metaqa, and webqsp) are printed beside with their own K, but the verdict does not wait on them
  separately; the mass rule's tau\* (smallest tau meeting the same condition) and its weighted rows are printed beside as
  well, and decide nothing.
- **Nothing is adopted here.** K\* and tau\* are read on `s1eval`, the program's eval carve, as a measurement. A later
  stage that uses a cut chooses it on `s1sel` with the pools it will run on.

## 4. What each outcome leads to (fixed now)

| stage 0 | the next form |
| --- | --- |
| CASCADE_PROMISING | Stage 1, declared in its own file before any fit: a selector trained for recall (a listwise loss weighted to the last gold, with the universal cut of section 3), then the ranker on the selector's pools with the golds forced in at training and not at reading; run on Kaggle's T4 on PKO's pools after 1 November, read on the twelve reads under the floors, zero-shot wiki graphs held out. Then stage 2, one or two EM rounds, if stage 1 gains. |
| CASCADE_PARTIAL | The mass rule's rows decide: if its weighted rows are below the fixed-K rows, a stage on the per-question cut; otherwise the same stage 1 with a smaller aim. |
| NO_CASCADE | A cascade of this form is closed: a ranker's order does not hold the golds higher than the walk's. The level ruling (about 2,000 rows at about 0.92, or about 3,300 at 0.95, or each dataset's own cell by one procedure, pick O's 0.96 at 1,904.6 rows) goes back to the user. Next form: a selector trained for recall alone, in its own file. |

## 5. How it runs

    python outputs/mp_unified/zu2v.py selftest
    python outputs/mp_unified/zu2v.py run --fit L-musique --out outputs/u2v/stage0
    python outputs/mp_unified/zu2v.py run --fit L-metaqa  --out outputs/u2v/stage0
    python outputs/mp_unified/zu2v.py report --out outputs/u2v/stage0

CPU, four threads, one carve at a time (the largest cache is musique's, 1.25 GB). The records written are
`<fit>__<dataset>__s1eval.json` (committed) and `.npz` per-question arrays (not committed).

**Self-test, passed before this commit** (about 26 s): on 8 toy questions of 0 to 200 rows with NaN and tied scores, the
need, gold-row count, top-5 count and the mass cuts at every tau equal a plain rewrite without torch or numpy tricks, and
`lean_gpu.top_hit` gives the same top-5 counts; on the real `L-musique` model and the real squad `s1eval` cache (11,873
questions, 594,063 rows) the rank-based top-5 count equals `lean_gpu.top_hit`'s for every question, ALL is non-decreasing
in K for every order and bounded by the reach, no need exceeds the pool, and the mass cut grows with tau.

## 6. Not shown, and not claimed

- Anything on the current models (zrckpt, zg6r), on PKO's pools, or on a GNN's order. The step-1 MLP and P1c2's pools are
  what is on this laptop.
- That a trained selector would do better than the model's ranking order: this reads a ranker's order, not a selector's.
- Any gain in R@5 or latency. This is a measure of where the golds sit.
- Anything on the three in-domain datasets as a zero-shot claim.
