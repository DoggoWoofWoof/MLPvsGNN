# C4: the pool's typed edges by a galloping merge, for every model

Declared 10 October 2026, about 04:05, before any timed number. It follows C3 (docs/C3_FASTEST_FORMS.md).

## Why

C3's per-stage table puts the compile at 41% of the MLP's cold time on the small graphs. A development profile on
2wiki (60 questions, not filed) splits that compile's edge stage:

| piece | p50 |
| --- | ---: |
| fast_features' typed-edge scan | 1.6–3.4 ms |
| everything else in the compile | under 1 ms |

The cost comes from hubs. For a 104-node pool, the scan walks about 84,600 stored entries to keep 256 pairs, probing
the 6M-entry global lookup once per entry.

`outputs/mp_unified/edges_fast.py` (tracked, a design look until now) has `typed_edges_gallop`:
- a stored row longer than a threshold is **merged** with the ascending pool, by galloping lower bounds;
- a shorter row is scanned as before.

The structural store's rows are sorted by neighbour id, so the merge gives the **same arrays in the same order**. Its
selftest compares every output on 12,000 random cases with hubs, self-loops and repeated pairs.

## What C4 changes

The serving form of one kernel, in **both** families' compiles: the MLP's `compile_mlp` and the six GNN's full fast
compile. The threshold is a row longer than **4 × the pool size**. It was chosen on the development profile:

| threshold | 2wiki p50 | musique p50 |
| --- | ---: | ---: |
| 4n | 0.17 ms | 2.8 ms |
| n | 0.21 ms | 4.0 ms |
| 256 | 0.15 ms | 9.7 ms |

Nothing else changes. The weighted families (ner, knn) keep their scan: their merge needs an index-time copy of every
row.

**The rule is C3's, and it holds exactly.** Every compiled scalar, every family's edges and every score equal C3's
form **bit for bit**. The development check passed this on 30 2wiki questions, with 0 failing.

## What is timed

`outputs/mp_unified/c4_fast.py run` (new; frozen once run), under C3's protocol:
- laptop, pinned, one thread, session idle;
- the first 200 s1eval questions;
- the four untyped datasets;
- each path's whole chain on its own, order rotating question by question.

Six paths:

| path | what it is |
| --- | --- |
| zrc4, zsp4, gnn4 | C3's fast paths with the galloping typed edges |
| zrc3, zsp3, gnn3 | C3's fast paths unchanged, the references, timed in the same run |

## Checks (untimed, every question)

- Each new path's compiled scalars and edges equal its C3 form's bit for bit.
- Its scores are exactly equal to its C3 form's.
- The six GNN's scores (gnn4) match the look's stored scores, with the same top 5.
- Every path's pool equals the look's.

A question failing any check is kept out of the summaries. More than 2% failing stops a dataset.

## What is reported

`outputs/c4/<dataset>.json` and `outputs/c4/report.{json,md}`, all per question, cold, batch 1:
- p50, p95 and p99 per path;
- gnn4 / zrc4 (the headline) and gnn4 / zsp4;
- gnn3 / zrc3, read again in this run;
- the speed-ups zrc3 / zrc4, zsp3 / zsp4 and gnn3 / gnn4;
- the compile p50 per path.

There are no thresholds. The forward and the warm numbers are C3's and are not re-timed.
