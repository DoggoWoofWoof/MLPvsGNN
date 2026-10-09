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

(Filed after the run.)
