# Step 4g: where the golds that step 4e's pools miss are

Declared 9 October 2026, before any number. This is a diagnosis: it trains nothing and files no pool. Its job is to
choose the next pool method from evidence, not from guesses.

## Why

A gold outside a question's pool cannot be ranked by any model. Step 4e's chosen pools (A3, k = 2) miss at least one
gold on 4.0% (metaqa), 0.8% (squad), 5.0% (musique), 2.3% (hotpotqa), 5.4% (2wiki) and 9.1% (webqsp) of the
questions on the choice carves. Step 4f showed that a single ranked cut cannot shrink these pools without losing golds.
The user's direction (9 Oct): the pools must get better, as a contribution of its own, in parallel with the models.
To do that, we need to know where the missing golds are.

## What is measured

For every dataset, on the s1sel carve (the five training datasets) and the s1eval carve (all six):

1. **Rebuild step 4e's pools exactly.** Each pool is U_q plus the first 2 x B_q nodes of the A3 walk outside U_q,
   computed by step 4e's own code (`run_carves`, `arm_orders`).
   - **Identity gate:** the carve's ALL must equal step 4e's filed `coverage.json` value for A3 at k = 2 exactly, or
     the stage refuses.
2. **For every gold outside its question's pool** (a *miss*), record:
   - its 1-based rank in the dense and in the SPLADE top-1000 list (0 if absent);
   - its 1-based rank in the A3 walk order (computed to 3 x the carve's largest B_q, step 4e's KMAX; 0 if absent),
     and that rank divided by B_q;
   - whether exact-match linking from the question names it;
   - its degree in the regime union;
   - **its hop distance from the pool**: a breadth-first search from all pool nodes over the regime's edge families,
     as the walk reads them, up to 3 hops;
   - which families carry a first hop into it (from a pool node);
   - its hop distance, up to 3, from the question's golds that are in the pool (if any);
   - the question's kind, from the query record (`hop`, else `type`).
3. **Bucket.** Each miss goes in the first bucket that fits:

| bucket | the miss is | the remedy it points to |
| --- | --- | --- |
| F | in the dense or SPLADE top-1000 list, but outside U_q | take more of the first-stage lists into U_q |
| W | reached by the walk, but ranked below 2 x B_q | rank the walk better, or a larger k |
| H1 | one hop from the pool | a second expansion round from the pool |
| H2 | two hops from the pool | the same, two rounds or a learned frontier |
| H3 | three hops from the pool | deeper expansion; likely too costly |
| X | further than 3 hops, or unreachable | the graph cannot bring it; only the first stage can |

## What follows

The next pool method (step 4h, declared in its own file) targets the bucket or buckets that hold most misses on most
datasets. It is graded on ALL and on pool size against step 4e, on all six. Nothing here is a result about the models.

## Running it

    python scripts/step4g_pool_misses.py miss DATASET [--threads 5] [--host]    -> outputs/step4g/misses_<dataset>.json
    python scripts/step4g_pool_misses.py report                                 -> outputs/step4g/report.json, report.md
    python scripts/step4g_pool_misses.py --selftest

On the host, the miss stage runs through `outputs/host_ops/pylib_run.py` (numba), with `--host` reading the verified
mirror, as step 4f's host jobs did.

## Results

Run 9 October 2026 (filed 10 October). squad, hotpotqa and 2wiki ran on the laptop; metaqa, musique and webqsp ran on
the host, one job per dataset. Every carve passed its identity gate. Records: outputs/step4g/misses_<dataset>.json,
report.json, report.md.

Shares of the misses by bucket (s1sel / s1eval):

| dataset | questions missing a gold | F | W | H1 | H2 | H3 | X | within 2 hops of a found gold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metaqa | 60 / 1,497; 464 / 9,785 | 0 / 0 | 0.09 / 0.11 | **0.89 / 0.88** | 0.02 / 0.01 | 0 | 0 | 0.54 / 0.53 |
| musique | 77 / 1,534; 153 / 2,417 | 0.01 / 0 | 0.19 / 0.19 | **0.78 / 0.79** | 0.01 / 0.02 | 0 | 0 | 0.71 / 0.59 |
| webqsp | -; 137 / 1,503 | - / 0.00 | - / 0.07 | - / **0.88** | - / 0.06 | 0 | 0 | - / 0.92 |
| 2wiki | 80 / 1,496; 692 / 12,576 | 0.16 / 0.23 | 0.10 / 0.13 | **0.42 / 0.30** | 0.24 / 0.24 | 0.06 / 0.08 | 0.02 / 0.01 | 0.35 / 0.39 |
| hotpotqa | 34 / 1,508; 158 / 7,405 | **0.68 / 0.69** | 0.02 / 0.07 | 0.23 / 0.18 | 0.05 / 0.04 | 0.02 / 0.02 | 0 | 0.55 / 0.51 |
| squad | 12 / 1,498; 94 / 11,873 | **0.75 / 0.89** | 0 | 0 | 0 | 0 | 0.25 / 0.11 | 0 |

What this says:

- **One hop is the main gap.** On metaqa, musique and webqsp, 78-89% of the missing golds sit one hop from the
  pool. On 2wiki it is the largest bucket too (30-42%), with 24% more at two hops. The first hop runs almost entirely
  over the `structural` family: KB triples on metaqa and webqsp, hyperlinks on 2wiki and hotpotqa. musique's
  title mentions carry about half of its first hops; ner and knn carry the rest.
- **The walk's cut is the second gap** (W: 7-19% on metaqa, musique, webqsp and 2wiki). These golds are reached but
  ranked below 2 x B_q, at a median of about 2.5 x B_q.
- **On hotpotqa and squad the first stage misses.** These golds are in the dense or SPLADE top 1000, but at a
  median dense rank of about 250-600 (hotpotqa, 2wiki) or about 350 (squad). squad's graph cannot reach them: its
  misses have degree 0 or are unreachable.
- **The missing golds sit next to found golds.** 51-92% of them are within two hops of a gold already in the pool
  (squad excepted). The question's own found evidence points at the rest of the chain.
- **Hyperlinks carry the reach on 2wiki and hotpotqa.** Every first hop into a missing gold there is a hyperlink.
  So U1's universal links matter for the pools as well as for the models.

**Step 4h targets H1 and W:**
- a second expansion round from the pool's highest-ranked nodes, over the regime's families, rather than from all of
  the pool (which would multiply its size);
- graded on ALL and pool size against step 4e, on all six.

The F bucket (hotpotqa, squad) is a first-stage question. It is noted for a later step, not mixed into 4h.
