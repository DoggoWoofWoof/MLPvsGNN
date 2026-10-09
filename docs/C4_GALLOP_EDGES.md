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

## Result (filed 10 October 2026, about 04:20)

Run on the laptop under the protocol above (outputs/c4/{2wiki,hotpotqa,squad,musique}.json, report.{json,md}).

**Every check passed on every question: 800 of 800.**
- For each family, the compiled scalars, edges and scores equal C3's form bit for bit.
- The six GNN's scores are within 4.8e-6 of the look's stored scores, with the same top 5.

**Cold, batch 1, p50 ms per question** (bootstrap 95% CI on the ratios):

| dataset | MLP zrc4 | zsp4 | six GNN gnn4 | **gnn4 / zrc4** | gnn4 / zsp4 | MLP speed-up zrc3 / zrc4 | GNN speed-up gnn3 / gnn4 |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 2wiki | 4.5 | 4.8 | 9.2 | **2.05 [2.00, 2.11]** | 1.90 [1.86, 1.95] | 1.38 [1.31, 1.44] | 1.25 [1.21, 1.29] |
| hotpotqa | 3.9 | 4.2 | 7.7 | **2.00 [1.96, 2.06]** | 1.86 [1.82, 1.91] | 1.34 [1.27, 1.46] | 1.29 [1.24, 1.35] |
| squad | 2.5 | 2.8 | 5.6 | **2.27 [2.22, 2.34]** | 2.01 [1.95, 2.06] | 0.98 [0.97, 1.00] | 0.99 [0.97, 1.02] |
| musique | 51.5 | 58.2 | 176.7 | **3.43 [3.39, 3.47]** | 3.03 [2.99, 3.06] | 1.00 [0.99, 1.01] | 0.99 [0.99, 1.01] |

Compile p50 ms, C3 form → C4 form:

| dataset | MLP | six GNN |
| --- | --- | --- |
| 2wiki | 2.59 → 1.14 | 4.53 → 2.71 |
| hotpotqa | 2.03 → 0.92 | 3.81 → 2.22 |
| musique | 16.66 → 16.71 | 58.99 → 59.02 |
| squad | 0.58 → 0.61 | 1.54 → 1.57 |

### What it says

- On **2wiki and hotpotqa** the hub rows made the typed-edge scan the largest part of the compile, and the merge
  removes most of it. Both families gain:
  - the MLP is 1.34–1.38× faster;
  - the six GNN is 1.25–1.29× faster;
  - the MLP's cold lead over the six GNN moves from 1.85–1.92× (C3's forms, same run) to **2.00–2.05×**.
- On **squad and musique** the merge changes nothing; every CI includes 1.00. squad's graph has no hub rows, and
  musique's pools of about 2,150 nodes mostly read short rows.
- The best exact cold forms now stand at:

  | | 2wiki, hotpotqa | squad | musique |
  | --- | --- | --- | --- |
  | MLP over the six GNN | 2.0× | 2.3× | 3.4× |
  | MLP's cold time | 3.9–4.5 ms | 2.5 ms | 51.5 ms |

  The GNN track's base (zsp) costs 1.06–1.13× the MLP.

On the small graphs the MLP's cold time is now mostly the pool, the embedding read and its forward, so another exact
serving form has little left to take. A larger cold lead needs an MLP that reads cheaper inputs. That is a model
change, read for accuracy and cost together on all six datasets (C3's "Next").
