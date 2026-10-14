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

## Results

To be filed.
