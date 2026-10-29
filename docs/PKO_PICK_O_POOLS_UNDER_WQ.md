# PKO: U2s's pick O pools under WQ, and the chain the WQ refits read

Declared on 28 October 2026, before any PKO pool is built and before any number of PKO's pools exists. The code is
`outputs/mp_unified/zpko.py`. It imports committed modules unchanged and patches them in process. It edits no
committed file: zu2m*, zu2o, zu2p, zu2q, zu2r, zu2m_e1 and zu2s are sha-locked, and the frozen and host-stage scripts
are untouched.

## 1. Why, and what waits on it

U2s carried pick O, `own/kind_t/kt_cos:ada_f16_k0@0.85`, at 1,904.6 rows per question on its carves (U2s's declaration
is docs/U2S_WALK_MASS_BUDGET.md, its verdict outputs/u2s/verdict.json, and zu2s.py's sha256 is 0c4fb2ef…). WQP
(5352b17) gained nothing over it, so it stays the carried pick. No WQ fit has read it yet, because WQK's eleven carves
are U2d's base build.

Two declared refits wait on PKO:

1. **zrckpt.** U2n's typed-channel MLP was adopted on the T4. Its next form is a refit on U2s's pick.
2. **zg6r.** G6RW filed ADOPT_FOR_FULL_RUN (e0e2c51; docs/G6RW_G6R_UNDER_WQ.md section 7). Its fixed next form is zg6r
   fitted again on the pools WQP carried, which are these pools.

So PKO is on the critical path of both the MLP and the GNN.

PKO builds pools and the inputs read from them, and nothing else:

- nothing trains and nothing is fit;
- nothing reads a gold;
- no command computes, reads or prints a recall, R@5 or ceiling of the new pools.

## 2. The rule

The rule is U2s's, unchanged and restated as calls. Each carve is opened on U2s's graph by
`zu2d_pools.open_carve("new", …)`; its graph_detail equals U2e's and U2s's on all six datasets. The walk is U2s's
carried walk (`zu2q.carried_walk`): G 0 and beta 0.5, which is RWe@0.5 on the KBs and RWe@0.5 x LW@0.5 on the passage
graphs. For each question:

1. **The walks.** The five walks kind_t reads (`zu2q.WALKS`) run on (a)'s relational kernel (`zu2m_a.rwalk`), with
   U2p's pair weight on the passage graphs and the typed relations on the KBs.
2. **KIND_T.** It comes from P1e's orders, with the three relational walks in place of P1e's (zu2s.py:160-476). It is
   checked to be ROUTEst on entity questions and RRF5T otherwise.
3. **B.** B is the own construction's base rows plus the seeds. It is checked to equal step 4d's B and to lie inside I.
4. **The orders.** U2e's 14 orders are built on B and checked exact. kt_cos is pick O's order.
5. **The budget and the cut.** k_q is ADA's budget at f = 16 and k0 = 0 on today's fill (`zu2e.budget_k`). Then
   k = min(|kt_cos|, k_q), and P1h's cut at 0.85 is applied on |B| and k (`keep_counts`). The pool is B plus the first
   `keep` rows of kt_cos.
6. **The walk-mass counts.** These are the counts U2s files for A1all and KTall. They are used for check (a) only.
7. **The KT input.** It is P1c3's eight columns (scripts/p1c3_kt.py) on P1e's own walks, as in U2d new's build. Between
   PKO and the WQ inputs before it, only the pools change.

`rule_batch` in zpko.py makes these calls in the order zu2s.stats_stage makes them, for the own construction and kind_t
only.

## 3. Carves

These are zwq.SCREEN's eleven carves: 69,076 questions in all.

| carve | questions | | carve | questions |
|---|---:|---|---|---:|
| webqsp/fit | 1,202 | | squad/s1eval | 11,873 |
| squad/fit | 5,856 | | musique/s1eval | 2,417 |
| musique/s1fit | 4,601 | | hotpotqa/s1eval | 7,405 |
| hotpotqa/fit | 5,930 | | 2wiki/s1eval | 12,576 |
| 2wiki/fit | 5,928 | | webqsp/s1eval | 1,503 |
| metaqa/s1eval | 9,785 | | | |

webqsp's fit carve is opened as WQK opened it, through `zwqk.on_wqk_build` with `zwq.on_wq_pools`. The other ten are
opened as U2d opened them.

The chained carves are zwq.CHAINED: webqsp/fit, webqsp/s1eval and metaqa/s1eval.

## 4. The chain, per carve

| step | what it does | code it calls | root under outputs/pko |
|---|---|---|---|
| pools | Section 2's rule on shards of 1,000 questions (KB datasets) or 3,000 (passage datasets) | `rule_batch`, `p1gb_pools.pool_arrays`, `p1c3_kt.columns` | pools/parts |
| merge | Joins the shards into one pools file, its KT rows and a manifest | zpko.py | pools |
| look | `look_u2d.main --build new` | `zu2d_pools.load_filed` and `roots`, and `look_u2d.DECLARED`, patched to PKO's | look |
| cache | `lean_cache.py` | PKO's look and cache roots; the bases are copied first from outputs/step1/cache, checked against their npz_sha256 | cache |
| rmatch, zrc | Chained carves only | `zp1c2.build` with its roots patched | chains/rmatch, chains/zrc |
| links | Every carve | `zp1c2.build` | links |
| typed | Chained carves only | `zu2n.build_typed` with PKO's roots; webqsp's fit through `zu2n.extra_carves` | typed |
| KT | P1c3's columns put in the cache's row order | `zu2d_pools.to_cache_order`; writes the record `zp1c3.load_kt` reads | kt |
| record | Section 5 (c) | zpko.py | records |
| prune | With `--prune` only: section 6 | zpko.py | look |

Besides those roots, PKO writes check/, logs/, ledger.json and gate.json. The records are committed later, by the user.
Nothing else is ever committed: not the arrays, looks, caches, chains, links, typed arrays or KT arrays, not the parts,
and not the logs or the ledger.

Threads: each lane is its own process, running one carve at a time, with 2 numba threads (NUMBA_NUM_THREADS=2 and
`--threads 2`) and BLAS at 2.

## 5. Checks before any build

### (a) U2s's own arrays

`zpko.py check` runs PKO's code again on U2s's carve. Question by question, it compares every per-question field U2s
files for the own construction and kind_t:

- |B|, |I| and |I \ B|;
- the fill and the 14 order lengths;
- U2e's union sizes;
- n1, r0 and the second link's counts, taken and hubs;
- the walk-mass counts and their totals;
- and, derived from those, k_q and the kept rows.

U2s's gold arrays (pos, in_I, one_link and ptr) are never loaded.

The check runs on two carves:

- **webqsp/s1eval.** This is U2s's carve and also a WQ carve, so the check runs inside the pools pass that builds it
  (`pools … --check`).
- **hotpotqa/s1sel.** Of U2s's five s1sel carves, squad's is cheaper (9,042.7 s against 9,197.6 s), but it is
  degenerate: its own cell is rrf50+none, so I = B, the fill is 0 and the cut has nothing to cut. hotpotqa is the
  cheapest non-degenerate one.

Each check writes check/<dataset>.json, PASS or FAIL.

### (b) squad

U2s's own cell on squad is rrf50+none (I = B), so the fill is 0, k_q is 0 and the pool is B, the same as U2d new's.

`check-squad` compares PKO's squad pools with U2d new's cache part records on:

- the question count and the rows;
- the sha256 of the int32 pool sizes;
- the sha256 of the sorted pool rows, in that record's question order.

The records hold 5,856 questions and 292,994 rows for fit, and 11,873 and 594,063 for s1eval.

It also compares U2d new's manifest entry (outputs/u2d/new/pools/squad/pools.json, files.squad__<carve>): questions,
changed, short, the budget, fill and today's-inc means, graph_detail, the pool mean, the maximum and mean of |B| and of
the rows per question, and the sum of its parts' rows. The manifest's I, pool, caps and shares are never accessed.

U2d new's squad cache arrays are not on the laptop; only part_0of1/record.json is. So squad's chain is rebuilt. If the
arrays are present with matching sha256s, check-squad places them in PKO's cache root, and the cache step links them
instead of building.

### (c) Records

Each carve's record, records/<dataset>__<carve>.json, names the sha256 of every input it was built from:

- the merged pools file, its KT rows and its manifest;
- the part records of the look, the cache, rmatch, zrc, the links, typed and KT;
- U2s's verdict and record;
- the EM and score types;
- every script.

The record also re-checks the step-1 cache's question sizes with `zu2d_pools.step1_check`, using WQK's step-1 cache for
webqsp's fit. The KT step has already checked the cache's rows against the pools row by row. The gate (gate.json)
re-hashes every input of every record.

Any check that fails stops the chain before any look.

## 6. Ledger, lanes, disk guard and prune

The chain is run as `zpko.py chain [--carves D/C,…] [--lanes N] [--prune] [--retry]`. It runs in this order:

1. copy-bases;
2. checks (a) and (b), on N lanes;
3. each carve's steps, on N lanes, with one carve per lane at a time;
4. the gate, once all eleven carves are built.

**Ledger.** ledger.json holds one entry per step: OK or FAILED, with the sha256 of the files the step wrote. Those
files are records, never chunks.

- A rerun skips every entry whose files still match.
- A FAILED entry stops the chain unless `--retry` is given.
- A step that died midway may leave partial outputs that its rerun refuses. Clear those by hand, after reading its log
  under logs/.

**Disk guard.** Before a carve starts, its bytes are estimated from its rows:

- the rows are exact once its pools are merged;
- before that, they are its questions times U2s's filed rows per question at pick O on the dataset's U2s carve;
- the bytes per row for each root were measured on WQK's and U2d's builds.

The carve starts only if the estimate fits in the free disk, less 15 GiB, less what the running carves are still
expected to write. Otherwise it waits while another carve runs; when none is running, the chain stops.
`zpko.py plan` prints the estimate for each carve.

**Prune.** `prune --carve D/C`, or `--prune` in the chain, deletes only that carve's look chunks, which can be
regenerated. It keeps the look's records and its pools JSON. It runs only once every step that reads the look is filed
OK: the cache parts and the links, plus rmatch, zrc and typed on the chained carves. This is stricter than "once the
cache record exists", because the links, rmatch, zrc and typed read the look too.

Shipping to Kaggle is the user's job, outside the repo.

## 7. Beside, computed after the build and deciding nothing

For each carve and in all, the record files:

- rows per question, against U2s's 1,904.6;
- the changed and short counts;
- the budget and fill means;
- the bytes per root;
- seconds per step and per question.

No recall, R@5 or ceiling of PKO's pools is computed in this stage.

## 8. What follows, fixed now

1. zrckpt is fitted again on PKO's roots, with its bases fitted again there. This is its own declared file, written
   before its fit. That stage patches `zp1c3.DECLARED` to this file for PKO's KT records, as WQK's on_mixed_kt does.
2. zg6r is fitted again on PKO's roots, G6RW's section 7 next form, in its own declared file.
3. Then comes any refit an ADOPT path names on these roots. G6RW's has already been adopted.

## 9. Estimates and risks, before any build

**Rows and disk.** `zpko.py plan` prints this table from U2s's filed rows per question at pick O (read without its
gold arrays; exact once a carve is merged) and the bytes per row of WQK's and U2d's built carves:

| carve | questions | rows/q | rows | GiB | look GiB |
|---|---:|---:|---:|---:|---:|
| webqsp/fit | 1,202 | 4,803 | 5,773,701 | 11.2 | 1.9 |
| squad/fit | 5,856 | 50 | 293,027 | 0.3 | 0.1 |
| musique/s1fit | 4,601 | 5,522 | 25,407,562 | 33.9 | 16.3 |
| hotpotqa/fit | 5,930 | 713 | 4,228,755 | 4.7 | 1.5 |
| 2wiki/fit | 5,928 | 1,071 | 6,345,639 | 6.4 | 2.2 |
| metaqa/s1eval | 9,785 | 4,370 | 42,759,516 | 33.9 | 14.3 |
| squad/s1eval | 11,873 | 50 | 594,110 | 0.6 | 0.3 |
| musique/s1eval | 2,417 | 5,522 | 13,347,116 | 17.8 | 8.6 |
| hotpotqa/s1eval | 7,405 | 713 | 5,280,595 | 5.9 | 1.8 |
| 2wiki/s1eval | 12,576 | 1,071 | 13,462,003 | 13.6 | 4.7 |
| webqsp/s1eval | 1,503 | 4,803 | 7,219,527 | 14.0 | 2.4 |
| **all** | 69,076 | 1,805 | 124,711,551 | 142.3 | 54.2 |

The laptop's C: drive had 45.3 GB free on 28 October, so the chain cannot hold every carve at once.

- With `--prune`, a carve's look is freed once its readers are done (the looks are 54 of the 142 GiB).
- The largest carve is metaqa/s1eval at about 34 GiB. musique/s1fit is the same size.
- The guard stops the chain unless built carves are shipped and freed between runs. Shipping is the coordinator's.

**Memory.** On WQK's webqsp/fit build, a look or cache process peaked at 5 to 6.5 GiB at about 1.27M rows. So look
shards stay at about 1M rows and cache parts at about 1.25M rows. Use one lane, or two when 13 GiB is free.

**Time.** The downstream rates were measured on WQK's laptop build:

| step | µs per row |
|---|---:|
| look | 1,170 |
| cache | 115 |
| rmatch | 162 |
| zrc | 211 |
| links | 8 |

The pools pass was timed on this laptop on webqsp/s1eval's first 50 questions (`zpko.py check webqsp --limit 50
--threads 2`; nothing written; no recall, ceiling or gold computed):

| measured | value |
|---|---:|
| pools pass | 2.03 s per question (101.5 s for 50) |
| opening the carve (once per process) | 117 s |
| section 5 (a) fields on those 50 | IDENTICAL |

So at 2.03 s per question the pools pass is at most 38.9 h for all 69,076 questions on one lane. It is the upper
bound, because webqsp has the largest B and N1 tables per question, and squad's questions (50 rows, no expansion) are
cheaper. The downstream steps are 50.6 h on one lane (the looks alone are 40.5 h). `plan --sq 2.03` gives 89.5 h on
one lane in all. The lanes are separate processes, one carve each, so 3 lanes would take about 30 h if the
memory allows it.

Every shard is its own process and opens the carve again. A KB carve opens in 117 s (webqsp). A passage carve opens
far more slowly: a 20-question hotpotqa/s1sel check had read the node names at 80 s and was still opening when the
570 s guard ended it (9 min 30 s), so its opening and its rate are not measured. That is why pools shards hold 1,000
questions on webqsp and metaqa and 3,000 on the passage datasets (21 passage shards, not 59). The first thing the
chain does on hotpotqa is this check, so its opening is timed there.

**Not exercised on the laptop before the chain:**

- webqsp's fit opening;
- the look, cache, rmatch, zrc, links, typed and step-1 wrappers on real data;
- the passage path of the pools pass after the carve is open. Its types block (EM and score types, the walk's name,
  the weights) was run on U2s's hotpotqa record alone and held.

Their first real run is the chain's own step, and a failure there stops that carve. It does not stop the carves
already filed.

## 10. What PKO never does

- It never edits a committed file.
- It never loads a gold array.
- It never computes, reads or prints a recall, R@5 or ceiling.
- It never overwrites a filed part, manifest, record or KT array.
- It never deletes anything but its own look chunks, and only under section 6's condition.
- It never writes outside outputs/pko.
- It makes no git, Kaggle, Modal or host actions.
