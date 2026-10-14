# P4b: KIND_T's walks in a faster exact form, timed beside P4a's

Declared 14 October 2026 at 16:24, before any timed number of this stage. It follows P4a's verdict
(docs/P4A_POOL_COLD.md, section "Results", filed 056918b): **WALKS_DOMINATE**, next P4b.

## Why

- **P4a timed KIND_T's build cold.** The walks are 88-97% of the build wherever a walk runs. On hotpotqa and 2wiki the
  build is 99% of every model's cold end to end, so every MLP/GNN ratio there is 1.00-1.02 end to end.
- **The walks are bound by memory latency, not arithmetic** (P4a's per-question counts). Each visit reads a neighbour's
  residual, flag and degree from three arrays spread over the whole graph, and every touched node is reset after the
  walk.
- **P4b changes only where the walk's state lives in memory.** The walks keep the same floating-point operations in
  the same order, so the orders and counts are the same, bit for bit. Nothing is cut short or approximated.
- **The user's speed rules:**
  - "we are prioritizing cold latencies rest all the stuff should remain same including the links, pools edges etc";
  - "all these models should be the best models wrt to the recall coverage etc, we are not removing stuff to make it
    faster we are making faster keeping the wins";
  - "i dont want to give any unfair advantage to mlp make both of them the fastest".
- **The build is shared.** Both families read the same pool, so a faster build helps the MLP and the GNN alike.

## The form

`scripts/p4b_fast_walks.py`, class `FastWalker`: P4a's `Walker` interface (`walk`, `twalk`) on a different memory
layout. Per union, at index time:

| part | what it is |
| --- | --- |
| hot record | 16 bytes per node, read on every visit: the residual, the walk key and the union degree (P4a reads them from three arrays) |
| cold record | 16 bytes per node, read when the node is popped: the start of its neighbour run, its first two families' lengths, its slot in the processed list |
| merged lists | the three families' neighbour lists merged per node: family 0's run, then family 1's, then family 2's, each in its own order |
| walk key | generation x 4 + 2 x processed + queued: a node is fresh in a walk when its key is older than the walk's generation, so no reset pass follows the walk |
| processed list | the nodes whose mass went above 0, in order, with the mass beside them: the top reads them, not every touched node |
| queue | a ring of n + 1 slots with a conditional wrap: P4a's FIFO order without a division |
| steer bitmap | one bit per node: a neighbour's extra steer weight is read only when its bit is set |
| numbering | every node renumbered, both unions alike, so a walk's nodes sit closer in memory |
| prefetches | reads issued ahead of use; they change no value |

**The declared form** (`FORM` in the script):
- **Numbering:** breadth-first over the union of all families (each node's runs merged in family order), roots in
  node order.
- **Records:** hot and cold apart.
- **Prefetch distances:**
  - the next neighbours' hot records 16 entries ahead in a run;
  - the hot and cold records of the node 12 places ahead in the queue;
  - the run's first entry, and its 16th, of the node 6 places ahead;
  - the hot records of the first 16 neighbours of the node 2 places ahead.

**What stays.**
- Step 4e's alpha and eps and P1e's steer.
- The order of every operation: seeds in their order, each popped node's neighbours in family order and list order,
  the same FIFO queue.
- The same top: mass descending, then the original node id ascending (step 4's `_top`'s ties).
- Every walk of KIND_T has no excluded node, so the candidates are step 4's: every node whose mass is above 0.

**Index time** (recorded, not timed): the numbering, the records and the merged lists. They hold about 1.4 GB on
2wiki, the largest graph, beside the unions P4a's form reads.

## How the form was chosen (disclosed)

- **The tuning set was C12's 200 questions.** In development, every walk KIND_T's build made on C12's 200 questions
  (and the JIT question) was recorded with P4a's outputs, on the five datasets that walk. The walks were then replayed
  in candidate forms:
  - the numbering: none, breadth-first, and reverse Cuthill-McKee where tried;
  - split or single 32-byte records;
  - the prefetch distances.

  Every replayed walk gave P4a's order and counts exactly. The declared form had the lowest walk time per question on
  those questions.
- **So the call is made on 200 fresh questions:** the look's s1eval questions 201-400, which are P1c2's file's next
  200. No development replay read them.
- **C12's 200 are timed beside,** labelled as the tuning set. They give the end to end with C12's models, which were
  timed on those questions.

## What is timed

- **P4a's timed path, unchanged** (`p4a_pool_cold.timed_pool`): today's pool, base and seeds, the links, the kind,
  the restarts, only the walks KIND_T reads, the order and the fill, each stage timed.
- **Run once with each walker.** P4a's form is P4a's `Walker`, P4b's form is `FastWalker`.
- **Both forms on every question, back to back.** P4a's form goes first on even positions and P4b's on odd ones, so
  neither form always runs on caches the other warmed, and both see the same conditions.

## Protocol

C12's and P4a's:
- the laptop, pinned to logical processor 2, EcoQoS off, above-normal priority (own process only);
- one thread;
- the session idle during the timed passes;
- the CPU load (2 s) and the free memory recorded before each timed pass;
- one untimed question first, for the JIT. The look's 401st question runs the timed path with each form, then every
  walk kernel of each form on both unions, steered and not;
- gc off, with a collection every 25 questions, between questions;
- cold, batch 1;
- timing and checks in separate passes;
- page faults and the working set recorded around each timed pass.

**Index time,** before the JIT question:
- the bench (P4a's, on the look's first 401 questions);
- P1c2's own path on all 400 questions, untimed. It runs first, so its batch buffers are freed before the form is
  built, which keeps the peak memory down;
- the numbering and the form;
- every page of the unions, both forms' buffers, FastPool's stamps and the name index, read once.

**Order of runs.** The datasets run in P4a's order: squad, metaqa, musique, webqsp, hotpotqa, 2wiki. In each, the
fresh set's timed pass runs before C12's set's.

## Checks (untimed, after both timed passes)

Per question, for each form, P4a's ten checks:
- today's pool;
- the seeds;
- base and seeds;
- B_q;
- the kind;
- ROUTE's condition;
- the fill against P1c2's filed fill;
- the fill against P1c2's own path, recomputed here;
- the pool's size;
- the re-run.

Plus two checks between the forms:
- **forms_equal.** In the timed pass, the two forms' pools and fills are equal. So are |I_q|, B_q, the kind, the
  route, the shortcut and every walk's counts (nodes touched, entries read).
- **walks_equal.** In the check pass, both re-runs go through a recorder. They make the same walks in the same order,
  and each walk returns the same order (up to 7,000 nodes) and the same counts.

**Which questions count.**
- A question is kept when all 22 checks hold.
- More than 2% failing in either set stops the dataset (P4a's rule). A stopped dataset is left out of the call.

## What is reported

`outputs/p4b/<dataset>.json` and `outputs/p4b/report.{json,md}`.

**Per set, per form:**
- the build's p50, p95 and p99;
- the walks' p50, p95 and p99, on questions that walk;
- the ratios P4a's form / P4b's form for the build and for the walks. Each comes with a 2,000-resample paired
  bootstrap 95% CI (C2's `ratio_ci`, default_rng(0)), with the same resamples for both forms;
- each walk's p50 where it ran;
- each form's build p50 when it ran first and when it ran second;
- the CPU load, the free memory and the page faults around the timed pass;
- the index time and memory of the form.

**On C12's set** (the tuning set, so labelled):
- P4a's end-to-end table, with each form's build in place of today's pool stage. These are C12's model stages, with
  P4a's caveat: they were timed on today's pools.
- P4a's ratios (universal, lean and across the families) with each form's build.
- P4a's filed build p50 beside P4a's form re-timed here, as a consistency check.

## The call (fixed before any number)

- **NOT_EXACT:** on some question of some dataset, the two forms differ (forms_equal or walks_equal fails). P4b's form
  is not adopted, and P4a's form stays.
- **FASTER:** otherwise, if on the fresh set both of these hold:
  - no dataset that walks and is not stopped has its build ratio's CI entirely below 1;
  - at least one has it entirely above 1.

  **Next:** P4's table uses P4b's form for KIND_T's build.
- **NOT_FASTER:** otherwise. P4a's form stays.
- **INCOMPLETE:** not all six datasets have run. No call is filed.

There is one form for all six datasets, and nothing is chosen per dataset. squad runs no walk (B_q = 0 on every
question), so its two forms run the same code. It is timed and checked, and it cannot decide the call.

## What follows

- **FASTER:**
  - P4's end-to-end table takes P4b's build. The table on C12's set is filed with the tuning caveat.
  - The confirmation run times the models on KIND_T's pools with P4b's form.
- **Either way, P4a's conclusion keeps its direction.** On hotpotqa and 2wiki even a several-fold faster build stays
  far above the models' 1.7-6 ms, so the end-to-end MLP/GNN ratios there stay near 1.
- Claim 3's wording stays the user's call: the model stages with the build beside them, or end to end.
- P1c3's serving cost (docs/P1C3_KT_INPUT.md, section 5) is timed with the form this call keeps.

## What this does not do

- No training, no fit, no gold read, no test split read.
- No change to any pool, link, edge, model or the encoder.
- KIND_T's pools are P1c2's, bit for bit, on every kept question, with both forms.
- No approximation: no walk is cut short, and alpha, eps, the steer and the top are unchanged.
- Nothing is chosen per dataset: one form and one path for all six.

## Files

    scripts/p4b_fast_walks.py              the form, the declared form's walker, the timed pass with both forms, the
                                           checks, run, check, report and selftest (new; frozen once run)
    outputs/p4b/<dataset>.json             one dataset's run, both sets
    outputs/p4b/report.{json,md}           the six datasets and the call

    python scripts/p4b_fast_walks.py --selftest
    python scripts/p4b_fast_walks.py check --dataset metaqa --queries 20
    python scripts/p4b_fast_walks.py run --dataset metaqa
    python scripts/p4b_fast_walks.py report

## Selftest

14 October, 16:12, on toy graphs (P4a's: three families, the structural family in components of eight nodes):
- **The top.** On the processed list, the top equals step 4's `_top` on the touched list in 3,200 cases. The ties
  and the cut are included, with and without a renumbering.
- **The walks.** 4,800 fast walks equal P4a's `Walker`, order for order and count for count, on buffers kept across
  calls. They cover both unions, steered and not, with empty restarts. The forms tested:
  - six forms of the numbering, records and prefetches;
  - the declared form, its breadth-first numbering a permutation rooted at node 0.

  The steer bitmap comes back all zero.
- **P1c2's own path on any questions** (`references_at`) equals it question by question. These are 48 toy questions
  over six cases of kind, route, L* and fill, run in one batch and in a subset given out of order.
- **The rest:**
  - the recorder passes every walk through;
  - the two comparisons each catch five kinds of difference;
  - the timed pass alternates the first form and gives P4a's walker back;
  - the checks catch a form whose walks or counts differ;
  - the set's report;
  - the call's four outcomes.

## Development check

14 October, 16:13-16:22, on the real data: `check --dataset D --queries Q`.
- **What each check runs:**
  - the setup and both forms;
  - the JIT question;
  - the first Q questions of each set through both forms;
  - P1c2's own path;
  - the re-runs and every check.
- **What it logs:** the steps, the branches and the failures. It logs no number.

| dataset | questions per set | walks compared | branches run | failing |
| --- | ---: | ---: | --- | ---: |
| squad | 20 | 0 | B_q = 0 | 0 |
| metaqa | 20 | 41 | entity off the route; entity on the route, with and without the shortcut | 0 |
| musique | 20 | 136 | passage; passage on the route | 0 |
| webqsp | 20 | 41 | entity off the route; entity on the route, shortcut; passage | 0 |
| hotpotqa | 10 | 74 | passage | 0 |
| 2wiki | 10 | 80 | passage | 0 |

metaqa ran an entity question on the route without the shortcut (ROUTEst's backfill). That branch did not come up in
P4a's development check.

## Results (filed 14 October, 16:53)

**The call: FASTER** on metaqa, musique, webqsp, hotpotqa and 2wiki, the five datasets that walk. No dataset is
slower. **Next:** P4's table uses P4b's form for KIND_T's build.

- **Every check held.** 200 of 200 questions were kept in both sets of all six datasets, all 22 checks each.
  - Both forms gave P1c2's fills and pools, bit for bit.
  - The two forms' walks were compared one by one: 5,322 walks, every one with the same order and the same counts.
- **The protocol held.**
  - Every run was pinned.
  - The session was idle from 16:24:06, when the runs started, to 16:51:11, after the report. That covers every
    timed pass.
  - The CPU load before the timed passes was 3.8-10.7%. The free memory was 1.33-5.86 GB.
  - The laptop was on mains power (read at 16:51; P4a's records do not hold the power state).

**KIND_T's build, the fresh set (questions 201-400, deciding), cold, batch 1, one thread, p50 ms:**

| dataset | P4a's form | P4b's form | build ratio (95% CI) | walks, P4a's form | walks, P4b's form | walks ratio (95% CI) |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| squad | 0.12 | 0.12 | 0.97 (0.85-1.07), no walk | - | - | - |
| metaqa | 7.24 | 6.36 | 1.14 (1.13-1.18) | 6.36 | 5.40 | 1.18 (1.15-1.20) |
| musique | 48.93 | 34.40 | 1.42 (1.40-1.46) | 44.46 | 29.84 | 1.49 (1.46-1.53) |
| webqsp | 34.58 | 13.20 | 2.62 (2.55-2.74) | 33.00 | 11.63 | 2.84 (2.74-2.96) |
| hotpotqa | 171.05 | 47.72 | 3.58 (3.47-3.64) | 165.63 | 42.91 | 3.86 (3.74-3.93) |
| 2wiki | 188.47 | 50.30 | 3.75 (3.68-3.81) | 183.16 | 45.46 | 4.03 (3.95-4.11) |

- **The gain grows with the graph.** It is 1.18x on metaqa's walks (43,234 nodes), 2.84x on webqsp's (2.59 million)
  and about 4x on hotpotqa's and 2wiki's (5.2 and 6.0 million). On a small graph the walk's state likely fits in the
  cache already.
- **The tuning set agrees with the fresh set.** On C12's 200 questions the build ratios are 1.17, 1.46, 2.87, 3.54
  and 3.70. Each is within 0.05 of the fresh set's, except webqsp's (2.87 against 2.62). Tuning the form on those
  questions did not inflate its gain.
- **The position did not matter.** On the walking datasets, each form's build p50 when it ran first and when it ran
  second differ by at most 5% (6 ms), far below the ratios. squad's walk-free build is 0.04 ms faster in second
  place, where its question's arrays are already in the cache. The alternation gives that to both forms equally.
- **Each walk, P4a's form -> P4b's form, fresh set:**
  - hotpotqa: A1st 64.91 -> 15.32 ms, A1all 60.84 -> 14.92, KTst 18.66 -> 6.33, KTall 21.43 -> 6.58;
  - 2wiki: A1st 66.86 -> 15.09, A1all 66.04 -> 15.88, KTst 23.83 -> 7.04, KTall 25.47 -> 7.06.
- **The form's index cost:**
  - the numbering takes 0.6-3.0 s and the records and lists 0.2-6.4 s;
  - it holds 0.01-1.41 GB (1.00 GB on hotpotqa, 1.41 GB on 2wiki), beside P4a's walk buffers of up to 0.20 GB.

**End to end on C12's set** (C12's model stages, cold p50 ms; KIND_T's build in place of today's pool stage, P4a's
caveat; the tuning set). Each cell reads: today's pool / P4a's form / P4b's form.

| dataset | twin0 | gnn0 | zrc | zsp |
| --- | --- | --- | --- | --- |
| squad | 2.67 / 2.71 / 2.71 | 4.24 / 4.28 / 4.30 | 1.65 / 1.67 / 1.69 | 1.73 / 1.77 / 1.79 |
| metaqa | 49.47 / 56.34 / 55.44 | 77.45 / 84.16 / 83.13 | 26.42 / 33.20 / 32.26 | 28.07 / 34.86 / 34.01 |
| musique | 75.66 / 125.18 / 109.29 | 143.26 / 192.40 / 176.81 | 35.87 / 85.52 / 70.33 | 42.19 / 91.97 / 76.39 |
| webqsp | 69.87 / 105.93 / 83.24 | 107.26 / 145.38 / 120.46 | 82.65 / 121.74 / 96.97 | 79.62 / 118.73 / 93.41 |
| hotpotqa | 3.91 / 170.92 / 51.12 | 5.88 / 173.13 / 53.40 | 2.36 / 169.19 / 49.38 | 2.48 / 169.12 / 49.37 |
| 2wiki | 4.11 / 191.99 / 54.71 | 6.00 / 194.50 / 56.60 | 2.52 / 190.48 / 53.32 | 2.61 / 190.41 / 53.55 |

**Ratios with P4b's build** (C12's set; 95% CIs in the report). The fastest GNN is zsp on every dataset.

| dataset | universal (gnn0 / twin0) | lean (zsp / zrc) | gnn0 / zrc |
| --- | --- | --- | --- |
| squad | 1.59 | 1.06 | 2.55 |
| metaqa | 1.50 | 1.05 | 2.58 |
| musique | 1.62 | 1.09 | 2.51 |
| webqsp | 1.45 | 0.96 | 1.24 |
| hotpotqa | 1.04 | 1.00 | 1.08 |
| 2wiki | 1.03 | 1.00 | 1.06 |

**A session difference, reported beside the call.**
- P4a's form, re-timed here on C12's set, runs at 0.58-0.95 of P4a's filed build p50:
  - squad 0.12 against 0.19 ms (no walk);
  - metaqa 7.37 against 10.44;
  - musique 51.13 against 75.98;
  - webqsp 37.94 against 61.32;
  - hotpotqa 167.13 against 286.24;
  - 2wiki 188.11 against 198.91.
- The laptop ran faster in this session than in P4a's. The gap is largest on hotpotqa, whose filed build was slower
  than 2wiki's although its walks do less work. Here it is faster, as the work says.
- **The call does not depend on this.** Both forms ran on every question, back to back, under the same conditions.
- **The end-to-end tables do.** They combine C12's model stages, from C12's session, with builds timed here. The
  confirmation run times the models and the build in one session.

**What this means**
- **KIND_T's build is 2.6-3.8 times faster on the large graphs, with the same pools.** It is 13.20 ms on webqsp,
  47.72 ms on hotpotqa and 50.30 ms on 2wiki (fresh set). Nothing was removed and nothing approximated.
- **The build still dominates on hotpotqa and 2wiki.** There it is about 20 times zrc's whole end to end on today's
  pool (2.36 and 2.52 ms). The universal, lean and gnn0/zrc ratios there are 1.00-1.08 end to end, as P4a expected.
- **Elsewhere the faster build raises the cross-family ratio:**
  - musique's gnn0/zrc rises from 2.25 with P4a's form to 2.51;
  - metaqa's from 2.54 to 2.58;
  - webqsp's from 1.19 to 1.24.
- **No exact form left cuts the push walk's own work.** A question's walks are independent of each other, so they
  could run on separate threads exactly. That changes C12's one-thread protocol, and the build is shared by both
  models.
- Claim 3's wording stays the user's call: the model stages with the build reported beside them, or end to end.
