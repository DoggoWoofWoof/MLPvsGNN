# P1a: our pools against the state-of-the-art pools, on the full corpora (a diagnostic)

Declared 13 October 2026, committed at 02:37 (54d6d12), before any number of the stage (the first job started at 02:38). P1's first stage in
[PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md). Nothing here is adopted, and no model trains or reads.

## 1. Question

On all six full corpora, does our pool construction hold at least as many golds as the state-of-the-art pool
constructions do on the same corpus, at every pool size? And how far is it from the KB-QA line's exposure (the assigned
topic entity) on metaqa and webqsp?

Step 4e compared its arms only with today's union pool kept whole inside every pool, so no arm's own order was ever cut
at a matched size. Step 4f cut only our own walk.

## 2. What is compared

Every walk uses the same graph and the same kernel: the dataset's regime union graph and step 4e's weighted push walk
(its alpha and eps). Only the restart differs. Each order is cut at up to 7,000 rows (step 4f's maximum), and nothing is
excluded.

| order | what | whose |
| --- | --- | --- |
| A3 | step 4e's chosen restart: seeds and linked names at half the mass, the RRF-weighted dense and SPLADE top-1000 lists at the other half, HippoRAG's specificity on both (step 4f's O1) | **ours** |
| A0 | today's seeds only | ours, before step 4e |
| A1 | HippoRAG 2's restart rule: seeds at half the mass, the RRF-weighted lists at the other half | SOTA (HippoRAG 2) |
| A2 | A1 with HippoRAG's node specificity | SOTA (HippoRAG) |
| LINKED | uniform over our exact-match linked nodes only (step 4e's linking, PullNet's rule) | SOTA (the KB-QA line, from our linking) |
| ORACLE | uniform over the **assigned** topic entities (metaqa, webqsp only) | the KB-QA line's exposure: a diagnostic column, never our input |
| DENSE, SPLADE | the first-stage top-1000 lists | first stage |
| RRF | the equal-RRF order of the two lists | first stage |

- **What differs from the published systems.** HippoRAG 2's own graph is an LLM-built knowledge graph with phrase nodes,
  and its restart weights passages by dense similarity. Here the graph is ours and the weights are RRF's (step 4e). The
  KB-QA line uses PageRank-Nibble over the KB. Here the walk runs on the regime's union graph, which on webqsp holds the
  KNN and NER families beside the KB's triples. These are LLM-free reproductions of the restart rules on our graph, not
  the systems as released.
- **squad has no graph.** Every walk order is its RRF order, as in steps 4e and 4f. LINKED and ORACLE are empty there.
- **No entity, no entry.** A question with no linked node (LINKED) or no assigned topic entity (ORACLE) gets an empty
  order: those systems cannot enter the graph without one. The record gives each order's share of such questions.
- **Sets, each at its own size:**
  - I_q, the frozen pools today's models read;
  - U_q, step 4d's union pools;
  - P4e, step 4e's chosen pools (base and seeds plus the expansion, checked against its manifest).
- **Matched size.** Each order is also cut at every question's own |I_q| and |P4e|. That answers: at the size the models
  read today, which construction holds the most golds?

**Carves:** s1eval on all six datasets. s1sel on the five training datasets is computed beside it, for a later choice,
and is not part of the call. **Measures:** ALL (every gold in the pool) and gold recall, at N = 25 to 7,000 (step 4f's
grid), and at the matched sizes.

## 3. The call, fixed before any number

For each dataset's s1eval carve:
- At every N of the grid, and on both measures, A3 is compared with the best of the LLM-free SOTA orders (A1, A2, LINKED,
  DENSE, SPLADE, RRF) at that N.
- **AT_OR_ABOVE:** A3 is never more than 0.002 below the best (step 4e's tie).
- **BELOW:** anything else. The record names every N and measure where A3 is below, the order that leads there, and the
  largest gap.

Reported beside each call, deciding nothing:
- A3, the best SOTA order, and the set itself at the matched sizes of I_q and P4e.
- **The KB line (metaqa, webqsp):** ORACLE's ALL and recall at ReaRev's m (500 on MetaQA, 2,000 on WebQSP), beside
  LINKED's and A3's at the same m, and the smallest N of the grid at which A3's ALL reaches ORACLE's at m (or "beyond
  7,000").

## 4. What follows

- **A dataset BELOW:** P1b, one pool change for all six datasets aimed at the measured gap, declared in its own file
  before its numbers.
- **All six AT_OR_ABOVE:** the construction stands. P1b then asks whether the models should read a larger or
  better-cut pool. The matched-size rows show that headroom, and P1b weighs it against the look cost (about quadratic in
  the pool, step 4e).
- **The KB line** goes to P3's ledger as the linking gap, and sets P1b's KB target.

## 5. How it runs

- **Code:** `scripts/p1a_pool_scoreboard.py` (its self-test passes). Development check: the first 300 squad, 300 metaqa
  and 150 webqsp s1eval questions, written to the scratchpad and filed nowhere. It caught one bug before this file: step
  4e's pool files hold base and seeds apart from the expansion, so P4e is their union, checked against the manifest's
  mean size.
- **One host job per dataset** (step 4f's pattern: the host mirror through `--host`, numba through
  `outputs/host_ops/pylib_run.py`). Each writes `outputs/p1a/scoreboard_<dataset>.json`. Then `verdict` writes
  `outputs/p1a/verdict.{json,md}`. It refuses parts from different scripts or freezes, and parts run with `--limit`.
- **Resources:** step 4f's sizes, 5 CPUs and 12 GB a job (squad: 1 CPU, 4 GB). Each job writes a few hundred KB, so it
  runs in the feeder's light lane.
- **ETA:** step 4f's walk took 15-861 s per carve. P1a walks four full restarts and two small ones, so about three times
  that. 2wiki and hotpotqa take about 45-60 minutes; the other four datasets take minutes. The call should land within
  about an hour of the start.
- The encoder stays frozen. No model trains or reads, nothing is adopted, and test splits are never read.

## 6. Results (filed 13 October, 03:10)

**The call on s1eval: BELOW on five datasets.** squad is AT_OR_ABOVE only because it has no graph (every order there is
its RRF order). Records: `outputs/p1a/scoreboard_<dataset>.json`, `outputs/p1a/verdict.{json,md}`.

| dataset | call | the order that leads | where A3 is below (ALL, A3 against the leader) |
| --- | --- | --- | --- |
| metaqa | BELOW | LINKED | every N: 0.237 against 0.546 at N = 25, 0.800 against 0.959 at 2,000 |
| squad | AT_OR_ABOVE | (all equal) | nowhere |
| musique | BELOW | A1 | N <= 300 (0.548 against 0.575 at N = 100) and N = 4,000 (0.003) |
| hotpotqa | BELOW | A1 | N <= 750: 0.512 against 0.830 at N = 25, 0.868 against 0.973 at 100 |
| 2wiki | BELOW | A1 | N <= 300: 0.578 against 0.838 at N = 50; A3 leads at 500 to 2,000 (0.962 against 0.951 at 1,000) |
| webqsp | BELOW | A1 | every N: 0.005 to 0.043 |

s1sel (beside, not part of the call) says the same: BELOW on the same five datasets.

**At the sizes the models read** (each order cut at each question's own |I_q|; ALL):

| dataset | rows (mean) | today's pools I_q | A3 | the best SOTA order |
| --- | ---: | ---: | ---: | ---: |
| metaqa | 2,018 | 0.895 | 0.800 | 0.961 (LINKED) |
| squad | 50 | 0.980 | 0.981 | 0.981 (A1) |
| musique | 2,092 | 0.803 | 0.884 | 0.883 (A2) |
| hotpotqa | 94 | 0.968 | 0.857 | 0.973 (A1) |
| 2wiki | 106 | 0.906 | 0.772 | 0.911 (A1) |
| webqsp | 2,120 | 0.688 | 0.759 | 0.790 (A1) |

At the same size and look cost, a walk's own top N holds more golds than today's pools on every dataset with a graph:
+0.066 on metaqa (LINKED), +0.082 on musique (A3), +0.102 on webqsp (A1), and +0.005 on hotpotqa and 2wiki (A1).

**Why A3 is below, from the columns:**
- **The linked names dilute the head on passages.** A3 restarts from the seeds and the linked names at equal weight. On
  hotpotqa, the walk from the linked names alone holds 0.617 at N = 100, the walk from the seeds alone (A0) 0.947.
- **The lists waste half the restart on the KB.** metaqa's nodes carry names only. Its dense and SPLADE lists hold every
  gold of 0.079 and 0.202 of the questions at N = 1,000 (ALL). A3 and A1 put half their restart there, LINKED none. On
  webqsp, the seeds alone (A0, ours before step 4e) lead A1 at N = 100 to 1,500 (0.723 against 0.697 at 1,000).

**The KB-QA line (diagnostic).**
- metaqa, m = 500: the assigned topic entities' walk (ORACLE) 0.903, our linked entities' (LINKED) 0.868, A3 0.670. A3
  does not reach ORACLE's 0.903 by N = 7,000.
- webqsp, m = 2,000: ORACLE 0.816, LINKED 0.656, A3 0.752. A3 reaches ORACLE's 0.816 at N = 4,000.

Our own linking gets metaqa within 0.036 of the assigned entities at ReaRev's m. On webqsp, our exact-match linking is
the weak part (LINKED 0.656 against ORACLE 0.816).

**What follows (section 4):** P1b, one pool change for all six, aimed at the measured gap
([P1B_RESTART_MIXTURE.md](P1B_RESTART_MIXTURE.md)): one restart whose source weights a question's own label-free
features set, fit by EM. The KB line goes to P3's ledger as the linking gap.
