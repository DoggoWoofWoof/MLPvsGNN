# C5: the cold cost table on the two typed KB graphs

Declared 10 October 2026, about 04:50, before any timed number. It follows C4 (docs/C4_GALLOP_EDGES.md). C1–C4 timed the
four untyped datasets only; C1 refused the typed graphs. Claim 3 (the MLP is faster cold) has no number on metaqa and
webqsp until C5.

## Why the typed graphs differ

On metaqa and webqsp both MLP fits read **chain entries**: rmatch.ChainMatch's gated chain match over the relation
chains that the question's seeds start on its pool's typed structural edges. The step-1 caches hold these entries built
offline:
- zrc reads zrc.build's entries (contribution cap);
- zsp reads rmatch.build's (mass cap).

A question served cold has no such cache, so it must build its own entries, and the MLP pays for that stage. The six
GNN reads the same relations as edge slots in its pack, so it pays only for the slots.

The pools are also larger: about 2,000 rows a question, against 100–2,150 on the untyped graphs. The filed builds hold
about 7,000 entries a question on metaqa and 88,000 on webqsp (part 0's means).

## What is timed

`outputs/mp_unified/c5_typed.py run` (new; frozen once run), under C3/C4's protocol:
- laptop, pinned, one thread, session idle;
- the first 200 s1eval questions;
- each path's whole chain on its own, order rotating question by question.

All three paths use C4's galloping typed edges (the fastest exact form of the compile for both families).

| path | stages |
| --- | --- |
| zrc5 (the MLP) | pool, read, compile (c2_fast.shared_mlp); edges, lean (c3_fast.lean_rows3); **chains**; forward (FastZ5) |
| zsp5 (the GNN track's base) | as zrc5 with rmatch's entries, plus links (c3_fast.links_fast) |
| gnn5 (the six GNN) | pool, read, compile (the full fast compile); pack with relation slots (fast_pack5); forward (c3_fast.FusedGNN) |

**chains** has four steps:
1. Each structural pair's first K_REL relations (universal_v2_models.relation_slots, as the look stores e_rel).
2. rmatch.row_triples and the seed buckets.
3. zrc.contrib_entries on zrc.step_of (zrc5) or rmatch.question_entries (zsp5).
4. The entries cast to the builds' stored dtypes (int16 rows and steps, int8 bucket, float16 mass).

**FastZ5** is c3_fast.FastZ plus ChainMatch's chain term, added to the base score before zprop's head. That is the
same function as zrm.ZRM / zprop.ZProp on the same weights.

Each fast path also times a warm repeat of its forward on its own inputs, reported beside and labelled warm. The
headline stays cold.

## Checks (untimed, every question)

| check | rule |
| --- | --- |
| pool, seeds and buckets | equal the look's |
| the edges and their e_rel | equal the look's |
| each MLP path's chain entries | equal its fit's filed build (zrc/cache, rmatch/cache, part 0) **bit for bit** |
| FastZ5 | within TOL = 1e-4 of the fit's own forward (lean_gpu's batch plus rmatch.chain_batch) on the same rows and entries, same top 5 |
| gnn5 | same top 5 as the look's stored gnn0 scores |

Reported, not gating:
- the fit's forward on the step-1 cache's rows and the filed entries, against FastZ5's scores;
- the float16 rows and codes against the step-1 cache's. The cache was built at the host's BLAS thread count, so a few
  compiled columns differ in their last bits.

A question failing any gating check is kept out of the summaries. More than 2% failing stops a dataset.

**Development check** (30 questions each, not timed): 0 failing on metaqa and on webqsp.
- Chain entries: bit for bit on all 60.
- FastZ5 against the reference forward: at most 8.1e-5.
- gnn5 against the look's scores: at most 8.1e-6, same top 5 on all 60.

## What is reported

`outputs/c5/{metaqa,webqsp}.json` and `outputs/c5/report.{json,md}`, all per question, cold, batch 1:
- p50, p95 and p99 per path;
- gnn5 / zrc5 (the headline), gnn5 / zsp5 and zsp5 / zrc5, with bootstrap 95% CIs;
- every stage's p50, the chains stage included;
- the warm forward ratio gnn5 / zrc5, labelled warm.

There are no thresholds. If the MLP is not faster cold on a KB graph, that is the result, and its stage table says
which stage costs it.

## Result (filed 10 October 2026, about 04:55)

Run on the laptop under the protocol above (outputs/c5/{metaqa,webqsp}.json, report.{json,md}).

**Every gating check passed on every question: 400 of 400.**
- Pool, seeds, edges and e_rel equal the look's.
- Chain entries equal the filed builds bit for bit.
- FastZ5 is within 8.8e-5 of the fit's forward, with the same top 5.
- gnn5 is within 1.5e-5 of the look's scores, with the same top 5.
- On the cache's own rows and the filed entries, the fit's forward has the same top 5 as FastZ5 on 400 of 400.

**Cold, batch 1, p50 ms per question** (bootstrap 95% CI on the ratios):

| dataset | pool rows | MLP zrc5 | zsp5 | six GNN gnn5 | **gnn5 / zrc5** | gnn5 / zsp5 | warm forward gnn5 / zrc5 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| metaqa | 2,038 | 42.0 | 44.1 | 98.1 | **2.34 [2.32, 2.36]** | 2.22 [2.20, 2.25] | 5.91 [5.83, 6.00] |
| webqsp | 2,122 | 152.6 | 145.3 | 127.8 | **0.84 [0.78, 0.89]** | 0.88 [0.81, 0.93] | 2.34 [2.29, 2.40] |

Stage p50 ms:

| stage | metaqa zrc5 | metaqa gnn5 | webqsp zrc5 | webqsp gnn5 |
| --- | ---: | ---: | ---: | ---: |
| pool + read | 9.2 | 9.2 | 14.3 | 14.4 |
| compile | 9.0 | 31.3 | 16.3 | 47.5 |
| edges + lean | 7.9 | – | 9.3 | – |
| **chains** | **5.6** | – | **83.1** | – |
| pack | – | 1.5 | – | 1.9 |
| forward | 9.6 | 55.8 | 26.4 | 63.4 |

Chain entries per question (p50): 6,408 on metaqa, 87,299 on webqsp.

### What it says

- On **metaqa** the MLP is 2.3× faster cold, in line with the untyped graphs (2.0–3.4×, C4).
- On **webqsp** the MLP is **slower cold** than the six GNN, at 0.84×. The gap comes from the chain match:
  - building the chain entries takes 83 ms, more than half the MLP's time;
  - the chain term then makes its forward 26 ms, against 10 ms on metaqa.

  Without both, the MLP's other stages total about 40 ms against the GNN's 128 ms.
- Claim 3 therefore does **not** hold on webqsp in the fits' current serving form. The chain entries are a per-question
  walk (rmatch.walk, then a lexsort over every (chain, row) pair before the k_row cap) written for offline builds, not
  for serving.

Next, C6 (its own declaration): an exact serving form of the chain build and the chain term. It must give the same
entries bit for bit and the same scores under C3's rule. If it cannot close the gap, the paper states Claim 3 for five
datasets and gives webqsp's number as measured.
