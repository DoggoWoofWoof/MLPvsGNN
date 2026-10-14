# P4a: KIND_T's pool, built cold, beside C12's models

Declared 14 October 2026 at 11:31, before any timed number. It follows C12 (docs/C12_SHARED_COMPILE.md) and P1c2
(docs/P1C2_POOLS_KIND_T.md), and it is the first stage of P4 (docs/PROGRAM_2026_10_13.md).

## Why

- **P1c2 made KIND_T the final pool.** That followed the rule of 13 October: links and pools are parameter-free, and
  hold the most golds in the fewest nodes. KIND_T's build walks the graph before any model starts. A passage question
  takes up to four push walks over a whole union, and an entity question one or two.
- **C12 timed the models on today's pool.** Today's pool is built by C10's FastPool in 0.07-0.33 ms (C12's p50s).
  KIND_T's build has never been timed cold.
- **Claim 3 needs that number.** Every model of both families reads the same pool, so the build is shared. It adds
  the same time to the MLP and to the GNN, so it moves every MLP/GNN ratio toward 1.
- **The user's speed rules:**
  - "mlp should be at minimum 2-3x faster than gnn across all families";
  - "we are prioritizing cold latencies rest all the stuff should remain same including the links, pools edges etc";
  - "i dont want to give any unfair advantage to mlp make both of them the fastest".
- **P4a measures the build and changes nothing.** The pools stay P1c2's bit for bit (checked on every question).

## What is timed

`scripts/p4a_pool_cold.py` (new; frozen once run). Each question builds KIND_T's pool in its fastest exact form, one
stage at a time:

| stage | what it does |
| --- | --- |
| pool | today's pool by C10's FastPool, the stage C12 times: \|I_q\| and the seeds |
| bs | the base (FastPool's own base rule) and the seeds; B_q = \|I_q\| - \|base and seeds\| |
| link | step 4e's links (squad has none) |
| kind | P1d's anchors and ROUTE's condition, P1e's kind |
| restarts | P1d's A1 restart, the uniform restart over L*, P1e's steer |
| walks | each walk KIND_T reads, timed on its own: w_kst, w_a1st, w_a1all, w_ktst, w_ktall |
| order | ROUTEst (P1d's backfill) or RRF5T (P1d's rrf over A1st, A1all, KTst, KTall and R0) |
| fill | the order's first B_q nodes outside base and seeds (step 4c's fill); the pool is base, seeds and fill |

**Four exact shortcuts.** P1c2's own path computes P1d's five sources and P1e's four steered walks on every question,
up to eight walks, in batches of 1,000. P4a computes only what KIND_T's pool reads:
1. **Only the walks KIND_T reads.**
   - An entity question on the route reads K(st), then A1(st).
   - An entity question off the route reads A1(st).
   - A passage question reads A1(st), A1(all), KT(st) and KT(all). KT is not walked when L* is empty, because P1e's
     restart is then empty.
2. **B_q = 0 ends the question.** |I_q| = |base and seeds| leaves KIND_T no fill: its pool is base and seeds. No
   link, kind or walk is needed. This is all 200 of squad's questions.
3. **ROUTEst's backfill only when it is reached.** ROUTEst is K(st)'s order, then A1(st)'s other nodes. When K(st)
   holds at least B_q nodes outside base and seeds, the fill is K(st)'s and A1(st) is not walked.
4. **Buffers kept across questions.** The walks run step 4e's and P1e's kernels (`_wpush`, `_tpush` and step 4's
   `_top`), one walk per call. Their buffers are sized once at index time and reset through the touched list, as
   the batch kernels reset them between the questions of a batch.

**Index time** (recorded, not timed): the stores and both unions, the name index and the name-only mask, the question
texts, the walk buffers and FastPool's stamps. Every page of the unions, the stamps and the buffers is read once
before the JIT question, and the name index is walked, so the timed pass does not fault them in. The index is built
once per corpus, like C12's.

## Protocol

C12's:
- the laptop, pinned to logical processor 2, EcoQoS off, above-normal priority (own process only);
- one thread; the session idle during the timed passes;
- the CPU load (2 s) and the free memory recorded before each timed pass;
- the first 200 s1eval questions of each dataset. These are C12's questions: the look's ids, which are P1c2's file's
  first 200, checked by the carve's digest;
- one untimed question first, for the JIT: the look's 201st question runs the timed path, then every walk kernel on
  both unions, steered and not;
- gc off, with a collection every 25 questions, between questions;
- cold, batch 1;
- timing and checks in separate passes;
- page faults and the working set recorded around the timed pass.

The datasets run in the order squad, metaqa, musique, webqsp, hotpotqa, 2wiki: smallest graph first, so the largest
setups come last. The background watcher of P1c2's host chain is stopped during the runs and restarted after them.

## Checks (untimed, in a separate pass after the timed one)

Per question:
- **Today's pool.** FastPool's pool is prepare's, at P1c2's filed size and sha256. Its seeds are prepare's.
- **Base and seeds.** They equal P1c2's filed ones and m3b_compile.base_rows' own. B_q is the filed one.
- **The kind and ROUTE's condition.** Each equals P1c2's filed flag, the flag of P1c2's own path (below) and the
  timed pass's. They are computed on every question, B_q = 0 included.
- **The fill, two ways.**
  - It equals P1c2's filed fill.
  - It equals the fill of P1c2's own path, recomputed here on the same questions: P1d's sources, P1e's steered walks,
    KIND_T's order and step 4c's fill, imported unchanged and run in one batch.
- **The pool's size** is P1c2's filed one.
- **The re-run.** Its fill and pool equal the timed pass's.

A question failing any check is kept out of the summaries. More than 2% failing stops a dataset, and a stopped
dataset is left out of the call.

## What is reported

`outputs/p4a/<dataset>.json` and `outputs/p4a/report.{json,md}`, all cold, batch 1, one thread:
- **KIND_T's build.** p50, p95 and p99. Each stage's share of the build and its p50 where it ran. How many questions
  had B_q = 0, how many were entity questions, on the route, or took the shortcut. The walks run and the nodes they
  touched.
- **Today's pool stage** (FastPool) beside C12's, as a consistency check.
- **End to end, per model**, at C12's fastest forms (outputs/c12/report.json's best):
  - C12's cold total on today's pool;
  - the same with KIND_T's build in place of today's pool stage.
  - **The caveat, written before the run.** The model stages are C12's, timed on today's pools. On these 200
    questions KIND_T's pools are the same size as today's, because P1c2 filed no short question. They hold other
    nodes, though, so other edges. The models on KIND_T's pools are timed in the confirmation run, on P1c2's fits.
- **C12's ratios**, on today's pool and with KIND_T's build, each with the 2,000-resample bootstrap 95% CI (C2's
  ratio_ci, default_rng(0)):
  - universal: gnn0 / twin0;
  - lean: zsp / zrc;
  - across the families: the fastest GNN over twin0 and over zrc, and gnn0 over zrc. The fastest GNN (gnn0 or zsp)
    is picked on each pool's end-to-end p50.
- **KIND_T's build as a share** of each model's end to end (the median over questions), and its p50 over zrc's
  end-to-end p50.
- **The questions** are paired with C12's by position. P4a's |I_q| must equal C12's pool size on every question, or
  the report stops. Only questions kept by both are summarized.

## The call (fixed before any number)

- **WALKS_DOMINATE.** On at least one dataset that is not stopped, two things hold:
  - KIND_T's build p50 is at least 25% of zrc's end-to-end p50 with KIND_T's build;
  - the five walk stages take at least half of the build's time, summed over the questions.

  **Next:** P4b, faster exact forms of the walks (the same pools, bit for bit), declared in its own file.
- **POOL_MINOR** otherwise. **Next:** P4c, C13's exact kernels for the universal compile's prototypes and depth basis.

Either way, Claim 3's wording stays the user's call.

## What to expect (written before the run)

The 200 questions, from P1c2's filed arrays:

| dataset | B_q | entity questions | on the route |
| --- | --- | --- | --- |
| squad | 0 on all 200 | 0 | 0 |
| metaqa | 779-1,992 | 200 | 159 |
| musique | 1,098-1,986 | 0 | 87 |
| webqsp | 1,859-1,981 | 188 | 55 |
| hotpotqa | 5-101 | 0 | 2 |
| 2wiki | 8-126 | 0 | 1 |

- **squad:** no walk. The build is today's pool and the base, about C12's pool stage.
- **metaqa:** K(st) on a 43,234-node graph, often alone (the shortcut). A1(st) off the route.
- **musique:** two to four walks on 117,534 nodes.
- **webqsp:** mostly one A1(st) walk on 2.59 million nodes.
- **hotpotqa and 2wiki:** two to four walks on 5.2 and 6.0 million nodes.
  - P1c2's host stage took about half a CPU-second per question for up to eight walks.
  - KIND_T's build is therefore likely 0.1-0.3 s per question, against C12's model totals of 2.4-6 ms.
  - If so, every MLP/GNN ratio there falls close to 1 with KIND_T's build, and the call is WALKS_DOMINATE.

## What this does not do

- No training, no fit, no gold read, no test split read.
- No change to any pool, link, edge, model or the encoder. KIND_T's pools are P1c2's, bit for bit, on every kept
  question.
- No approximation: the walks keep step 4e's alpha and eps and P1e's steer, the same orders and the same fills.
- Nothing is chosen per dataset: one path for all six.

## Files

    scripts/p4a_pool_cold.py               the walks one question at a time, the timed path, the checks, run, report
                                           and selftest (new)
    outputs/p4a/<dataset>.json             one dataset's run
    outputs/p4a/report.{json,md}           the six datasets

    python scripts/p4a_pool_cold.py --selftest
    python scripts/p4a_pool_cold.py check --dataset metaqa --queries 20
    python scripts/p4a_pool_cold.py run --dataset metaqa
    python scripts/p4a_pool_cold.py report

## Selftest

14 October, 11:18, and again at 11:30 on the committed script, on toy graphs (three families; the structural family
in components of eight nodes):
- **The walks.** 240 single-question walks on buffers kept across calls equal P1b's walk_mass and P1e's twalk, order
  for order, on both unions, steered and not, empty restarts included. The buffers come back all zero.
- **KIND_T's lazy fill.** On 240 toy questions it equals P1c2's own path (P1d's sources_batch, P1e's steered_batch,
  P1c2's kind_t and step 4c's fill) in every branch:
  - an entity question on the route, with and without the shortcut;
  - an entity question off the route;
  - a passage question with L* empty and not;
  - B_q = 0.
- **The rest.** The stage clock adds up. The end to end swaps today's pool stage for the build. The call's rule holds
  on both sides.

## Development check

14 October, 11:19-11:28, on the real data: `check --dataset D --queries Q`. Each runs the setup, the JIT question, the
timed path untimed, P1c2's own path and the re-run, then every check above. It logs which branches ran and no number.

| dataset | questions | branches run | failing |
| --- | ---: | --- | ---: |
| squad | 20 | B_q = 0 | 0 |
| metaqa | 20 | entity off the route; entity on the route, shortcut | 0 |
| musique | 20 | passage; passage on the route | 0 |
| webqsp | 20 | entity off the route; entity on the route, shortcut; passage | 0 |
| 2wiki | 10 | passage | 0 |

hotpotqa was not checked here; its timed run checks all 200 questions like the others. An entity question on the
route without the shortcut (ROUTEst's backfill) did not come up on the real data, only in the selftest. Any of the
200 questions that takes it is checked in the timed run's check pass.

## Results (filed 14 October, 11:56)

**The call: WALKS_DOMINATE** on metaqa, musique, webqsp, hotpotqa and 2wiki. **Next:** P4b, faster exact forms of the
walks, declared in its own file.

- **Every check held.** 200 of 200 questions were kept on all six datasets. The checks were today's pool, the seeds,
  base and seeds, B_q, the kind, ROUTE's condition, the fill (P1c2's filed one and P1c2's own path's), the pool's
  size and the re-run. KIND_T's pools as timed here are P1c2's, bit for bit.
- **The protocol held.**
  - Every run was pinned.
  - The session was idle during all six timed passes. 2wiki's ended at 11:49:23; the next command of this session
    came at 11:49:49, during 2wiki's untimed check pass.
  - The CPU load before the timed passes was 8.8-18.1%, from the laptop's own background processes. The free memory
    was 2.8-5.6 GB.
- **The expectation held.** The build took 0.20 s on 2wiki and 0.29 s on hotpotqa, against "likely 0.1-0.3 s".

**KIND_T's build, cold, batch 1, one thread (ms):**

| dataset | KIND_T p50 | p95 | p99 | walks' share | walks run (Kst, A1st, A1all, KTst, KTall) | CPU load before | free GB before |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| squad | 0.19 | 0.35 | 0.45 | 0.00 | 0, 0, 0, 0, 0 | 8.8 | 5.6 |
| metaqa | 10.44 | 26.02 | 30.20 | 0.88 | 159, 42, 0, 0, 0 | 14.6 | 5.48 |
| musique | 75.98 | 110.94 | 116.45 | 0.92 | 0, 200, 200, 154, 154 | 11.5 | 5.22 |
| webqsp | 61.32 | 102.60 | 145.58 | 0.96 | 55, 146, 12, 0, 0 | 9.6 | 4.4 |
| hotpotqa | 286.24 | 378.71 | 404.12 | 0.97 | 0, 200, 200, 178, 178 | 9.0 | 3.78 |
| 2wiki | 198.91 | 254.81 | 267.44 | 0.97 | 0, 200, 200, 200, 200 | 18.1 | 2.82 |

- **The walks are 88-97% of the build** wherever a walk runs.
- **squad runs no walk** (B_q = 0 on all 200). Its build is today's pool and the base: 0.19 ms.
- **The walks are bound by memory latency, not arithmetic.** On hotpotqa and 2wiki an A1 walk touches 267,000-327,000
  nodes (p50) and reads 0.8-1.0 million union entries, at 67-142 ns per entry (p50). That is the time of random reads
  of each neighbour's residual, flag and degree, and of the reset of every touched node after the walk. These figures
  come from the per-question walk counts in `outputs/p4a/<dataset>.json`.

**End to end, cold p50 ms, today's pool -> KIND_T's build** (C12's model stages, with the caveat as declared):

| dataset | twin0 | gnn0 | zrc | zsp | build / zrc's end to end |
| --- | --- | --- | --- | --- | ---: |
| squad | 2.67 -> 2.81 | 4.24 -> 4.38 | 1.65 -> 1.77 | 1.73 -> 1.87 | 0.11 |
| metaqa | 49.47 -> 59.15 | 77.45 -> 86.86 | 26.42 -> 36.19 | 28.07 -> 37.92 | 0.29 |
| musique | 75.66 -> 150.79 | 143.26 -> 218.77 | 35.87 -> 111.02 | 42.19 -> 116.42 | 0.68 |
| webqsp | 69.87 -> 130.32 | 107.26 -> 169.67 | 82.65 -> 149.11 | 79.62 -> 146.04 | 0.41 |
| hotpotqa | 3.91 -> 290.01 | 5.88 -> 291.62 | 2.36 -> 288.57 | 2.48 -> 288.62 | 0.99 |
| 2wiki | 4.11 -> 203.59 | 6.00 -> 205.12 | 2.52 -> 201.56 | 2.61 -> 201.67 | 0.99 |

**Ratios, today's pool -> KIND_T's build** (2,000-resample bootstrap 95% CIs; the fastest GNN is zsp on every
dataset, on both pools):

| dataset | universal (gnn0 / twin0) | lean (zsp / zrc) | fastest GNN / zrc | gnn0 / zrc |
| --- | --- | --- | --- | --- |
| squad | 1.59 [1.55, 1.69] -> 1.56 [1.52, 1.65] | 1.05 [1.03, 1.09] -> 1.05 [1.03, 1.09] | 1.05 [1.03, 1.09] -> 1.05 [1.03, 1.09] | 2.57 [2.50, 2.72] -> 2.47 [2.38, 2.61] |
| metaqa | 1.57 [1.55, 1.57] -> 1.47 [1.45, 1.48] | 1.06 [1.06, 1.07] -> 1.05 [1.04, 1.06] | 1.06 [1.06, 1.07] -> 1.05 [1.04, 1.06] | 2.93 [2.91, 2.95] -> 2.40 [2.36, 2.44] |
| musique | 1.89 [1.87, 1.91] -> 1.45 [1.43, 1.47] | 1.18 [1.16, 1.19] -> 1.05 [1.05, 1.06] | 1.18 [1.16, 1.19] -> 1.05 [1.05, 1.06] | 3.99 [3.94, 4.03] -> 1.97 [1.93, 2.00] |
| webqsp | 1.54 [1.51, 1.55] -> 1.30 [1.27, 1.32] | 0.96 [0.93, 0.99] -> 0.98 [0.95, 0.99] | 0.96 [0.93, 0.99] -> 0.98 [0.95, 0.99] | 1.30 [1.24, 1.34] -> 1.14 [1.11, 1.18] |
| hotpotqa | 1.50 [1.44, 1.56] -> 1.01 [1.00, 1.01] | 1.05 [1.01, 1.09] -> 1.00 [1.00, 1.00] | 1.05 [1.01, 1.09] -> 1.00 [1.00, 1.00] | 2.49 [2.38, 2.57] -> 1.01 [1.01, 1.01] |
| 2wiki | 1.46 [1.43, 1.50] -> 1.01 [1.01, 1.01] | 1.04 [1.01, 1.07] -> 1.00 [1.00, 1.00] | 1.04 [1.01, 1.07] -> 1.00 [1.00, 1.00] | 2.38 [2.33, 2.45] -> 1.02 [1.01, 1.02] |


**What this means**
- **The build is shared by every model.** On hotpotqa and 2wiki it is 99% of every model's cold end to end. There
  every MLP/GNN ratio is 1.00-1.02 end to end, whatever the models do.
- **On the other datasets the build is a smaller share.** It is 0.35-0.68 of the end to end on musique, where gnn0/zrc
  falls from 3.99 to 1.97. On metaqa that ratio falls from 2.93 to 2.40, and on webqsp from 1.30 to 1.14. squad is
  unchanged: it runs no walk.
- **Claim 3 on the model stages is unchanged.** Those are C12's numbers, with the pool given. Whether Claim 3 is stated
  on the model stages with the build reported beside them, or end to end, stays the user's call.
- **P4b cannot close the end-to-end gap alone.** It makes the walks faster and keeps them exact. Even several-fold
  faster walks would leave the build far above the models' 1.7-6 ms on hotpotqa and 2wiki. The end-to-end ratios
  there stay near 1 unless the build falls by about two orders of magnitude.

Files: `outputs/p4a/<dataset>.json` (per question: each stage's time, the kind, the route, the shortcut, each walk's
counts, and the checks) and `outputs/p4a/report.{json,md}`.
