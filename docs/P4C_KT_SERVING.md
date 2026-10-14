# P4c: the KT input's serving cost, timed cold beside KIND_T's build

Declared 14 October 2026 at 18:22, before any timed number of this stage. It follows P1c3's screen
(docs/P1C3_KT_INPUT.md, section 7, filed d06c384): **ADOPT_FOR_FULL_RUN**. P1c3's section 5 says the input's serving
cost is timed cold, in P4's table, before any claim uses a KT model. P4c is that timing. It uses P4b's form for every
walk (docs/P4B_FAST_WALKS.md, filed 8ae22fc: **FASTER**).

## Why

- **P1c3 gives zrc and zsp one more input.** For each pool row, it records where KIND_T's construction put that row:
  eight columns, read through a 177-parameter head. The models with it are zrck (the MLP) and zspk (the GNN). Its full
  run (docs/P1C3_FULL_RUN.md) decides whether they replace zrc and zsp.
- **KIND_T's build walks only what the pool needs** (P4a's lazy fill). The KT input reads six orders on every
  question: KIND_T, A1st, A1all, Kst, KTst and KTall. So serving it adds every walk the build skips:

  | question | the build walks | serving the KT input adds |
  | --- | --- | --- |
  | B_q = 0 (all of squad) | nothing | A1st and A1all; Kst, KTst and KTall when L* is not empty |
  | entity, on the route | Kst; A1st unless the shortcut is taken | A1all, KTst and KTall; A1st when the shortcut is taken |
  | entity, off the route | A1st | A1all; Kst, KTst and KTall when L* is not empty |
  | passage | A1st and A1all; KTst and KTall when L* is not empty | Kst when L* is not empty |

  It also adds the links, the kind and the restarts where B_q = 0 (the build stops before them), the columns
  themselves, and the head's forward.
- **The cost lands on both models of the lean pair.** zrck and zspk read the same input. But it may be large next to
  the lean models' own stages, so it has to be timed before any speed claim uses them.
- **The user's speed rules:**
  - "we are prioritizing cold latencies rest all the stuff should remain same including the links, pools edges etc";
  - "all these models should be the best models wrt to the recall coverage etc, we are not removing stuff to make it
    faster we are making faster keeping the wins";
  - "i dont want to give any unfair advantage to mlp make both of them the fastest".

  So the input is kept whole and served in the fastest exact form: P4b's walks, and the columns in one pass over each
  order.

## What is timed

Two paths per question, back to back, each timed stage by stage (`scripts/p4c_kt_serving.py`):

- **build:** KIND_T's pool alone, as the models without the KT input read it. This is P4a's timed path
  (`p4a_pool_cold.timed_pool`) on P4b's declared walker: exactly what P4b timed.
- **serve:** KIND_T's pool and the KT input, as zrck and zspk read them.

  | stage | what it does |
  | --- | --- |
  | pool, bs | today's pool, base and seeds, and B_q (P4a's) |
  | link, kind | step 4e's links, P1d's anchors and ROUTE's condition, and P1e's kind, on every question |
  | restarts | A1's restart; K's (uniform over L*) and the first stage's steer when L* is not empty |
  | walks | A1st and A1all on every question; Kst, KTst and KTall when L* is not empty (P4b's walker) |
  | order | KIND_T: ROUTEst (P1d's backfill) on an entity question, RRF5T (P1d's RRF over A1st, A1all, KTst, KTall and R0) on a passage question |
  | fill | the order's first B_q nodes outside base and seeds; the pool is base, seeds and fill |
  | columns | the pool rows' eight KT columns, rounded to float16 as P1c3 files them and the models read them |

- **head (timed beside, after serve):** the KT head's forward as zp1c3's `KTMix` runs it. That is `kt_inputs` (the
  clipped segment z-score of the score) and `kt_head` (linear, GELU, linear), added to the score. The columns are read
  as float32 from float16. It runs in torch on one thread, as C12's stages run the lean models' MLP parts. The weights
  are random, with the head's shapes, and the score is a stand-in; no value changes the cost.

**The columns' exact form.** P1c3's `columns()` places each pool row in each order by sorting the order and searching
it. The serve path computes the same values another way:
- one hash of the pool rows;
- one pass over each order, each node looking up its row;
- each place p reads log1p(p) / log1p(7000) from a table of the 7,000 values, computed as `columns()` computes them.

It equals `columns()` bit for bit (see Selftest). It refuses an order that holds a pool row twice, and an order longer
than 7,000 nodes, as `columns()` does.

**The alternation.** The build goes first on even positions, the serve path on odd ones, so neither path always runs on
caches the other warmed.

## Protocol

C12's, P4a's and P4b's:
- the laptop, pinned to logical processor 2, EcoQoS off, above-normal priority (own process only);
- one thread;
- the session idle during the timed passes;
- the CPU load (2 s) and the free memory recorded before each timed pass, and the power state at the start and end;
- one untimed question first, for the JIT: the look's 401st question runs P4a's warm-up (the build path, then every
  walk kernel on both unions, steered and not), then the serve path and the head;
- gc off, with a collection every 25 questions, between questions;
- cold, batch 1;
- timing and checks in separate passes;
- page faults and the working set recorded around each timed pass.

**Index time** (recorded, not timed):
- the bench (P4a's, on the look's first 401 questions);
- P1c3's own path on all 400 questions, untimed and first: P1d's sources, P1e's steered walks, kinds and orders,
  P1c2's fill and P1c3's `columns()`, all imported unchanged;
- P4b's declared form. P4a's walker is dropped, so only P4b's form walks; P4b showed the two give the same walks;
- the table of place values, the head's weights and the host's slice of P1c3's filed KT rows;
- every page of the unions, P4b's form, FastPool's stamps and the name index, read once.

**The sets: P4b's two.** The fresh set (questions 201-400) runs first, then C12's set (questions 1-200). C12's set is
P4b's tuning set, so it is labelled as such. It gives the end to end, because C12's models were timed on it.

**Order of runs.** P4a's: squad, metaqa, musique, webqsp, hotpotqa, 2wiki.

## Checks (untimed, after both timed passes)

Per question, 17 checks:
- **P4a's ten, on the build path's re-run** (`build_*`): today's pool, the seeds, base and seeds, B_q, the kind, ROUTE's
  condition, the fill against P1c2's filed fill, the fill against P1c2's own path, the pool's size, the re-run;
- **serve_pool:** the serve path's pool and fill are the build path's, from the timed pass;
- **serve_kind:** its kind and ROUTE's condition are P1c2's filed flags and P1c3's own path's;
- **orders_p1c3:** its six orders (KIND_T, A1st, A1all, Kst, KTst, KTall) are P1c3's own path's, node for node
  (sha256 of each order);
- **columns_p1c3:** its float32 columns are P1c3's `columns()` on P1c3's own orders, pool and fill, bit for bit;
- **kt_filed:** its float16 columns are P1c3's filed KT rows of the question, row by row by node (the host's slice);
- **serve_rerun:** the re-run's pool, fill and columns are the timed pass's;
- **head_rerun:** the re-run's head output is the timed pass's.

A question is kept when all 17 hold. More than 2% failing in either set stops the dataset (P4a's rule).

**The host's slice.** P1c3's KT rows and P1c2's caches live on the host. For the 401 questions, the host writes each
question's filed KT rows and each row's node into `outputs/p4c/filed/<D>.npz`, with a record `<D>.json`:

    python outputs/host_ops/pylib_run.py scripts/p4c_kt_serving.py slice --dataset D

Before slicing, it checks the pools file is KIND_T's, P1c3's record is the KT input's (its columns and float16), the
arrays' sha256s against P1c3's record, and the cache's questions and row counts against the KT rows'. The npz is
never committed, as P1c3's KT arrays are not.

## What is reported

`outputs/p4c/<dataset>.json` and `outputs/p4c/report.{json,md}`.

**Per set:**
- the build's, the serve path's and the head's p50, p95 and p99;
- serve / build, and serve + head / build: p50 ratios, each with a 2,000-resample paired bootstrap 95% CI (C2's
  `ratio_ci`, default_rng(0));
- the p50 of the per-question difference, serve minus build;
- each serve stage's p50 where it ran, and its share of the serve path;
- the walks each path ran, and the serve path's extra walks;
- each path's p50 when it ran first and when it ran second;
- the rows per question, B_q = 0 and entity counts;
- the CPU load, the free memory and the page faults around the timed pass;
- P4b's filed build p50, beside this session's build: a session check.

**End to end, on C12's set** (cold p50 ms):
- twin0, gnn0, zrc and zsp: C12's model stages (C12's cold total less its pool stage) plus the build timed here, as in
  P4b;
- zrck: zrc's C12 stages plus the serve path and the head;
- zspk: zsp's C12 stages plus the serve path and the head (zspk adds the head before propagation, so the rest of its
  forward is zsp's).

**Ratios on C12's set, with paired CIs:**
- universal (gnn0 / twin0);
- lean on the build (zsp / zrc);
- lean with the KT input (zspk / zrck);
- the fastest GNN (gnn0 or zspk, by p50) over zrck;
- gnn0 / zrck;
- the fastest GNN over twin0.

**Caveats, stated with every number:**
- **P4a's.** C12's model stages were timed on today's pools. KIND_T's pools have the same sizes (the same B_q), but
  not the same rows.
- **The session's.** C12's stages come from C12's session; the build, the serve path and the head are timed here.
- **The KT models' own compiled forms are not timed.** zrck and zspk are taken as zrc's and zsp's stages plus the
  head. The confirmation run times them whole, in one session.

## The call (fixed before any number)

- **INCOMPLETE:** not all six datasets have run. No call is filed.
- **STOPPED:** some dataset has more than 2% of a set's questions failing a check. The dataset is named, and no KT
  model enters a speed claim until a re-run passes.
- **TIMED:** otherwise. The serve path's cost, the head's and the KT models' end to end are filed in P4's table.

No threshold decides anything here. Whether the KT models are adopted is the full run's call (docs/P1C3_FULL_RUN.md);
P4c times what serving them costs. How Claim 3 is worded stays the user's call.

## What follows

- **TIMED, and the full run ADOPTs:** zrck and zspk become the lean pair, with this end to end and its caveats. The
  confirmation run times the compiled KT models and the build in one session.
- **TIMED, and the full run does not ADOPT:** the timing is filed for the record, and P3's ledger continues.
- **STOPPED:** diagnose the failing checks. Nothing is timed again until the cause is found.
- **Faster still, exactly:** a question's walks are independent, so they could run on separate threads with the same
  results. That changes C12's one-thread protocol, so it is the user's call, as in P4b.

## What this does not do

- No training, no fit, no gold read, no test split read.
- No change to any pool, link, edge, model or the encoder.
- KIND_T's pools are P1c2's, and the KT columns P1c3's, bit for bit on every kept question.
- No approximation: no walk is cut short; the orders, the place values and the float16 rounding are P1c3's.
- Nothing is chosen per dataset: one path for all six.

## Files

    scripts/p4c_kt_serving.py              the serve path, the columns' exact form, the head, the timed pass with
                                           both paths, the checks, the host's slice, run, check, report and selftest
                                           (new; frozen once run)
    outputs/p4c/filed/<dataset>.json       the host's slice's record (the npz beside it is never committed)
    outputs/p4c/<dataset>.json             one dataset's run, both sets
    outputs/p4c/report.{json,md}           the six datasets and the call
    outputs/p4c/first_pass/                the first runs and their report, kept beside (see Re-timing)

    python scripts/p4c_kt_serving.py --selftest
    python outputs/host_ops/pylib_run.py scripts/p4c_kt_serving.py slice --dataset D     (on the host)
    python scripts/p4c_kt_serving.py check --dataset metaqa --queries 20
    python scripts/p4c_kt_serving.py run --dataset metaqa
    python scripts/p4c_kt_serving.py report

## Selftest

14 October, 18:01, on toy graphs (P4a's: three families, the structural family in components of eight nodes):
- **The place values** equal `columns()`'s at all 7,000 places, bit for bit.
- **The columns' exact form** equals `columns()` bit for bit in 300 random cases. These include orders of up to 7,000
  nodes, empty orders, fill nodes outside the rows, and both entity values. It refuses a repeated row and an order
  that is too long.
- **The head** equals zp1c3's `KTMix.kt_inputs` and `kt_head` bit for bit, on 1 to 2,100 rows.
- **The serve path** equals P1c3's own path (the six orders, the fill, the pool and the columns) and the build path's
  pool and fill, on 160 toy questions. They cover 12 cases of kind, route, L* and B_q = 0, and the build's shortcut
  both taken and not taken.
- **The rest:**
  - the timed pass alternates the first path;
  - each check catches its failure: a changed filed row, column, head output or order;
  - the set's report, the end to end and the call's three outcomes.

## Development check

14 October, 18:02-18:18, on the real data: `check --dataset D --queries Q`.
- **What each check runs:**
  - the setup;
  - the JIT question;
  - the first Q questions of each set through both paths;
  - P1c3's own path;
  - the re-runs, and every check but kt_filed. The host's slice is not on the laptop yet.
- **What it logs:** the steps, the branches and the failures. It logs no number.

| dataset | questions per set | branches run (serve path) | failing |
| --- | ---: | --- | ---: |
| squad | 20 | B_q = 0 (no links, so no L*) | 0 |
| metaqa | 20 | entity off the route with L*; entity on the route | 0 |
| musique | 20 | passage without L*; passage with L*; passage on the route | 0 |
| webqsp | 20 | entity off the route with L*; entity on the route; passage | 0 |
| hotpotqa | 10 | passage without L*; passage with L* | 0 |
| 2wiki | 10 | passage with L* | 0 |

**With the host's slices, after the declaration** (14 October, 18:23-18:38; the slices filed in 0974bf9): the same
check on the first 10 questions of each set, now with kt_filed. On every dataset, every serve row's eight columns equal
P1c3's filed float16 columns, and all 17 checks pass with 0 failing. It logs no number.

## Re-timing (declared 14 October, 20:28, before any number was read)

The first runs went from 20:02 to 20:26, in P4a's order. Every dataset kept 200 of 200 questions in both sets, passed
all 17 checks on every question, and ran pinned on mains power. But this session was still running commands until
about 20:10, while squad, metaqa and musique were being timed. That breaks "the session idle".

The 2 s CPU loads before the timed passes (fresh set / C12's set, %):

| squad | metaqa | musique | webqsp | hotpotqa | 2wiki |
| --- | --- | --- | --- | --- | --- |
| 27.9 / 20.2 | 10.6 / 19.7 | 23.4 / 18.3 | 7.0 / 14.4 | 6.7 / 11.5 | 9.0 / 4.7 |

P4b's runs had 3.8-10.7.

**The rule.**
- All six are timed again, once, in one session, in P4a's order. The session stays idle, and its waiter is stopped.
- The second records are the filed ones, whatever their numbers. There is no third run.
- The first records and the first report are kept beside in `outputs/p4c/first_pass/`.

So far only these have been read: the loads above, the power state, the pin, the check counts, and each timed pass's
wall time in the run log. No timing number from the first runs has been read.

## Results

**Filed 14 October, 20:56: TIMED.** The record is `outputs/p4c/report.{json,md}`, from the re-timed runs (20:28-20:52).
- Every dataset kept 200 of 200 questions in both sets, and every check held on every question.
- All twelve timed passes ran pinned, on mains power, with the session idle.
- The 2 s CPU load before each timed pass was 3.4-13.3%.

**The fresh set** (questions 201-400). p50 ms per question, cold, batch 1, one thread:

| dataset | KIND_T's build | serve path | head | (serve + head) / build [95% CI] | serve minus build |
| --- | ---: | ---: | ---: | --- | ---: |
| squad | 0.15 | 21.72 | 0.53 | 152.96 [147.87, 189.51] | 21.61 |
| metaqa | 6.49 | 45.38 | 0.67 | 7.10 [6.97, 7.28] | 39.13 |
| musique | 35.45 | 39.33 | 0.74 | 1.13 [1.10, 1.18] | 3.98 |
| webqsp | 13.02 | 42.81 | 0.73 | 3.36 [3.16, 3.58] | 31.52 |
| hotpotqa | 47.40 | 55.59 | 0.57 | 1.19 [1.15, 1.20] | 7.35 |
| 2wiki | 50.88 | 58.04 | 0.58 | 1.15 [1.14, 1.17] | 7.44 |

C12's set gives the same picture: serve over build 1.12-1.15 on musique, hotpotqa and 2wiki, 3.19 on webqsp, 6.86 on
metaqa and 172 on squad.

**Where the time goes: the walks.**
- **musique, hotpotqa, 2wiki:** the build already runs four of the five walks the columns need. The serve path adds
  only Kst, 4-7 ms.
- **metaqa, webqsp:** the build runs one or two of them. The serve path adds three to five walks, 31-39 ms.
- **squad:** the build has no links on any question (B_q <= 0), so it runs no walk. The columns still need A1st and
  A1all, which take 19 of the serve path's 22 ms.
- **The rest is small:** the columns cost 0.26-0.64 ms and the head 0.53-0.74 ms.

**End to end on C12's set**, cold p50 ms. Each model is C12's model stages (C12's session, today's pools), plus the
build or the serve path and head timed here. zrck's head is an upper bound for zrckp's (161 parameters and no
z-score).

| dataset | twin0 | gnn0 | zrc | zsp | zrck (>= zrckp) | zspk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| squad | 2.73 | 4.33 | 1.72 | 1.82 | 27.92 | 28.05 |
| metaqa | 55.29 | 83.17 | 32.29 | 33.95 | 69.42 | 71.19 |
| musique | 110.64 | 177.56 | 71.21 | 77.44 | 75.78 | 81.70 |
| webqsp | 83.00 | 120.14 | 96.29 | 92.57 | 128.99 | 125.80 |
| hotpotqa | 50.84 | 52.86 | 48.95 | 49.36 | 56.45 | 56.75 |
| 2wiki | 55.40 | 56.92 | 53.75 | 53.87 | 62.26 | 62.13 |

**Ratios on C12's set** (paired bootstrap 95% CIs in the report):

| ratio | squad | metaqa | musique | webqsp | hotpotqa | 2wiki |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| zsp / zrc on KIND_T's build | 1.06 | 1.05 | 1.09 | 0.96 | 1.01 | 1.00 |
| zspk / zrck | 1.00 | 1.03 | 1.08 | 0.98 | 1.01 | 1.00 |
| gnn0 / twin0 | 1.59 | 1.50 | 1.60 | 1.45 | 1.04 | 1.03 |
| fastest GNN / zrck | 0.16 | 1.03 | 1.08 | 0.93 | 0.94 | 0.91 |

**In plain words:**
- **Where KIND_T already walks,** the KT input is cheap: +13-19% on the build and +6-16% end to end on musique,
  hotpotqa and 2wiki.
- **Where it does not, it is dear:** end to end, the MLP with the KT input takes 16x zrc's time on squad, 2.1x on
  metaqa and 1.34x on webqsp.
- **Within the lean family the ratio does not move.** Both models pay the same serve path, so zspk / zrck is 0.98-1.08.
- **zrck is slower than gnn0** on squad, webqsp, hotpotqa and 2wiki. No MLP is 2x faster than the fastest GNN with
  this input, and none was before it either (C12).

**Beside:** the first runs (`outputs/p4c/first_pass/`, under load) agree with the filed ones: every build and serve
p50 is within 3.3%.

**What follows:** P1c3p's full run ADOPTs (docs/P1C3P_FULL_RUN.md), so the lean pair is (zrckp, zspk), with this end
to end and its caveats. The confirmation run times the compiled KT models and the build in one session.

**Open, for the user or a declared stage:**
- **Threads for the walks** (P4b's question). A question's walks are independent, so they could run on separate
  threads with the same results.
- **squad.** The serve path walks on every question, yet zrckp moves squad's R@5 by at most 0.0015 on any split. Whether the columns carry
  anything when a question has no links (B_q <= 0) is a question for every dataset alike. It is not a per-dataset
  skip.
