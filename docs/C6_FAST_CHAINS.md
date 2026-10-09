# C6: the MLP's chain match in an exact serving form

Declared 10 October 2026, about 05:05, before any timed number. It follows C5 (docs/C5_TYPED_KB.md). C5 found the MLP
slower cold than the six GNN on webqsp (0.84×). There, building the chain entries took 83 ms per question and the chain
term 16 ms of its 26 ms forward.

## What C6 changes

Two serving forms, in `outputs/mp_unified/chains_fast.py` (new):

1. **`fast_entries`** replaces rmatch.question_entries (zsp's mass cap) and zrc.contrib_entries (zrc's contribution
   cap) and gives the **same entries bit for bit**. It changes three steps:
   - **The walk** is numba, chain by chain. Each chain's out-steps are summed per (step, node) in order of appearance,
     the order np.bincount adds them, in a dense scratch over (the pool's steps, pool rows). They are then sorted by
     (step, node), the order np.unique gives.
   - **The live-chain pass** renormalises masses in entry order and sums each chain's step scores in step order. The
     exp of the key stays in numpy, as zrc computes it.
   - **The k_row cap** works row by row: every key above the k_row-th largest, then the ties at it in chain order.
     That is exactly lexsort's (row, −key, chain) order, in linear time instead of a global sort.

   The step scores (zrc.step_of) and the typed graph (rmatch.typed_graph) are unchanged. Restricting the relation
   cosines to the pool's relations would change their last bits (20 of 33 development questions), so the full product
   stays.

   The selftest compares both caps on every question of the first two look chunks of metaqa and webqsp (44) and on 300
   random toy graphs: equal bit for bit.
2. **`FastChain`**: ChainMatch.chain_feats for one question over the relations on its pool only.
   - The step scores of other relations never reach an entry.
   - The weight-only product rel_centered @ cm_b.T (7,058 × 1,536 by 8 on webqsp) is taken once at index time.
   - The per-entry sum is numba.

   It is the same function on the same weights, in float32, checked under C3's rule.

Nothing else changes. The six GNN keeps C4/C5's fastest exact form (galloping compile, FusedGNN). The GNN has no chain
stage.

## What is timed

`outputs/mp_unified/c6_chains.py run` (new; frozen once run), under C5's protocol:
- laptop, pinned, one thread, session idle;
- the first 200 s1eval questions of metaqa and webqsp;
- order rotating question by question.

| path | what it is |
| --- | --- |
| zrc6, zsp6 | C5's zrc5 / zsp5 with fast_entries for the chains and FastZ6 (FastZ5 with FastChain) for the forward |
| zrc5, zsp5, gnn5 | C5's paths unchanged, the references, timed in the same run |

## Checks (untimed, every question)

- C5's gating checks on its three paths.
- zrc6's and zsp6's entries equal zrc5's and zsp5's **bit for bit**. Those equal the filed builds.
- FastZ6 is within TOL = 1e-4 of the fit's own forward on the same rows and entries, with the same top 5.

A question failing any check is kept out of the summaries. More than 2% failing stops a dataset.

**Development check** (30 questions each, not timed): 0 failing on metaqa and on webqsp. The entries are bit for bit;
FastZ6 is within 8.1e-5.

## What is reported

`outputs/c6/{metaqa,webqsp}.json` and `outputs/c6/report.{json,md}`, all cold, batch 1:
- p50, p95 and p99 per path;
- gnn5 / zrc6 (the headline) and gnn5 / zsp6;
- gnn5 / zrc5 again in this run;
- the speed-ups zrc5 / zrc6 and zsp5 / zsp6;
- the chains and forward stage p50s, and every stage's p50;
- the warm forward ratio, labelled warm.

There are no thresholds. If the MLP stays slower cold on webqsp, the paper states Claim 3 on five datasets and gives
webqsp's number as measured.
