# PKO A1: the look and cache counts from the rows, a peak record, a RAM preflight, smallest carve first

This amends `outputs/mp_unified/zpko.py`, PKO's own file (docs/PKO_PICK_O_POOLS_UNDER_WQ.md), before any further pick-O
number is built. It changes no number any step computes. Nothing here is a result. It is declared with its code; no
sha-locked module is touched (zu2m*, zu2o, zu2p, zu2q, zu2r, zu2m_e1, zu2s, look_u2d, lean_cache, zp1c2, zu2n stay as
they are).

## 1. Why

The read-only feasibility study of 30 October found four faults in the chain as committed (cea456a):

- `CHUNK_Q = 479` is squad's questions per look chunk. look_x_six cuts a chunk at 24,000 nodes
  (`chunk = max(1, int(CHUNK_NODES // mean pool))`, look_x_six.py:260), which is 479 for squad, 255 for hotpotqa, 227 for
  2wiki and 11 for webqsp, metaqa and musique on the base pools. At PKO's 4,400 to 5,500 rows per question a chunk is
  4 to 5 questions. So the chain capped every KB and musique carve at 3 to 21 look shards and cache parts of 1.8 to
  2.5M rows, not the 1M and 1.25M section 9 intends.
- No step's memory is recorded, and nothing checks it. The pools runs killed at about question 400 and 600 of 751 had
  ~4.6 GiB free on a 15.7 GiB laptop, against 5 to 9.5 GiB a lane needs.
- Nothing stops a step from starting when its memory is not there.
- The carves run in the order of `CARVES`, so the 34 GiB carves are not last.

## 2. (a) The counts from the rows

`chunk_count(nq, rows)` is look_x_six's rule on the carve's own mean pool (`rows / nq`, the mean of the pool sizes
the look reads from the merged file): `ceil(nq / max(1, 24000 // max(mean, 1)))`. A carve's look shards and cache
parts are `ceil(rows / target)`, never more than its chunks (a look shard takes whole chunks, a cache part a contiguous
slice of the look's chunk list).

Targets: look shards `LOOK_ROWS` = 1,000,000 as declared (look memory is a floor of ~3.7 GiB plus a part that grows with
the question count; more shards do not lower it, so this only sets how much a kill loses). Cache parts
`PART_ROWS_BY` by dataset, from the laptop fits of peak against rows (1.6 GiB + c per million rows; c = 2.9 webqsp,
1.1 metaqa, 4.5 2wiki, 5.8 hotpotqa, ~1.5 musique and squad): webqsp 0.5M, metaqa 1.2M, musique 0.95M, hotpotqa 0.4M,
2wiki 0.45M, squad 1.25M (one part). `--look-rows N` and `--part-rows N` override them on `chain` and `plan`.

`zpko.py plan` prints the result (rows are U2s's filed estimate until a carve is merged, then exact):

| carve (smallest first) | rows (M) | chunks | A0 look shards (rows each) | A1 look shards | A0 cache parts (rows each) | A1 cache parts |
|---|---:|---:|---:|---:|---:|---:|
| squad/fit | 0.29 | 13 | 1 | 1 | 1 | 1 |
| squad/s1eval | 0.59 | 25 | 1 | 1 | 1 | 1 |
| hotpotqa/fit | 4.23 | 180 | 5 (0.85M) | 5 | 4 (1.06M) | 11 (0.38M) |
| hotpotqa/s1eval | 5.28 | 225 | 6 (0.88M) | 6 | 5 (1.06M) | 14 (0.38M) |
| webqsp/fit | 5.77 | 301 | 3 (1.92M) | 6 (0.96M) | 3 (1.92M) | 12 (0.48M) |
| 2wiki/fit | 6.35 | 270 | 7 (0.91M) | 7 | 6 (1.06M) | 15 (0.42M) |
| webqsp/s1eval | 7.22 | 376 | 4 (1.80M) | 8 (0.90M) | 4 (1.80M) | 15 (0.48M) |
| musique/s1eval | 13.35 | 605 | 6 (2.22M) | 14 (0.95M) | 6 (2.22M) | 15 (0.89M) |
| 2wiki/s1eval | 13.46 | 572 | 14 (0.96M) | 14 | 11 (1.22M) | 30 (0.45M) |
| musique/s1fit | 25.41 | 1,151 | 10 (2.54M) | 26 (0.98M) | 10 (2.54M) | 27 (0.94M) |
| metaqa/s1eval | 42.76 | 1,957 | 21 (2.04M) | 43 (0.99M) | 21 (2.04M) | 36 (1.19M) |

The A0 columns are the chain's old counts (the cap `ceil(nq / 479)`, then the row targets). The eleven carves go from
78 to 131 look shards and from 72 to 177 cache parts. The extra processes cost start-up time only (the rows, and so the
downstream hours, are the same): ~105 more cache starts at ~30 s is ~0.9 h; each extra look shard opens the carve again
(GUESS 2 to 5 min, not measured). `--look-rows 2000000` halves the look shards if that turns out to matter.

**A carve's counts are fixed by its first file.** lean_cache and the look read every file of a carve, so a part count
cannot change after a part is filed. The chain now refuses (before queuing the carve's look) when a look record or a
cache, rmatch, zrc or links part of the carve is filed under another count, and names the files. Pass `--look-rows` and
`--part-rows` with the values the carve was started with. No carve has a look or cache file today.

## 3. (b) The peak record

`run_step` starts the step with Popen and samples its process tree every 0.5 s (psutil): the larger of the best sampled
sum of resident bytes and the largest single-process peak the OS keeps (`peak_wset` on Windows, `VmHWM` on Linux), so
a spike between two samples counts. On Windows the working set includes memory-mapped file pages (the dense embedding
memmaps), which is the number that decides whether a step fits in RAM here. Each ledger entry gains `peak_rss_bytes`
(the step's peak) and `need_bytes` (what it declared). Without psutil neither field is written, and the preflight, which
cannot measure, admits every step.

## 4. (c) The preflight, and the order

A step starts only when its declared RAM is available now (`psutil.virtual_memory().available`) less what the steps
already running still have to grow into (their declared RAM less their resident bytes). It waits, polling every 30 s,
for up to `--ram-wait` seconds (default 7,200; 0 stops at once). Then the chain stops with exit code 2 and files
nothing FAILED, so the same command resumes. `--no-preflight` starts steps without the check. Declared RAM is the
measured peak plus 1 GiB:

| step | declared (GiB) | basis |
|---|---|---|
| pools, check, look | squad 5, metaqa 5, webqsp 7, 2wiki 7, hotpotqa 7, musique 9.5 | laptop looks, 3.7 floor to 8.6 (MEASURED), plus PKO's larger pools |
| cache part | 1.6 + c x (part rows in millions) + 1 | the fits in section 2: webqsp 0.48M 4.0, metaqa 1.19M 3.9, 2wiki 0.42M 4.5, hotpotqa 0.38M 4.8, musique 0.94M 4.0 |
| merge | 0.3 + 114 B per row + 1 | derived from the code: metaqa/s1eval 5.8, musique/s1fit 4.3 |
| rmatch, zrc | 4 | GUESS (2 to 4 GiB on webqsp parts) |
| links, typed, record | 3 | GUESS |
| KT | 4 | GUESS (the node-name table) |
| check-merge, check-squad, copy-bases, gate, prune | 1, 2, 0.5, 1, 0.5 | code read |

The GUESS rows are filed as guesses; the ledger's `peak_rss_bytes` replaces them by an amendment once a carve has run.
The model predicts a cache part's peak at 2.9 to 3.8 GiB; its declared need is that plus 1.

**Order.** `chain` runs the carves smallest rows first (exact rows once merged, else the plan's estimate): squad/fit,
squad/s1eval, hotpotqa/fit, hotpotqa/s1eval, webqsp/fit, 2wiki/fit, webqsp/s1eval, musique/s1eval, 2wiki/s1eval,
musique/s1fit, metaqa/s1eval. `--as-given` keeps the order given. Section 5's checks still run first, as before.

**Small additions.** `chain --checks-filed` skips section 5's check jobs when `check/` already holds their PASS (for a run
on a base the checks were filed in); `--disk-reserve-gib` sets the 15 GiB kept free; `plan` prints each carve's chunks,
look shards, cache parts and run order, and the declared RAM by dataset. A job whose steps raise (a refused count change,
a missing manifest) now stops the chain; before, it ended its lane without a word and `run_lanes` still said it was done.

## 5. (d) What stays valid

| entry in outputs/pko/ledger.json | A1 |
|---|---|
| `copy-bases` OK | valid: same key, same four files, same shas |
| `pools:webqsp/s1eval:0/2` OK (751 questions, 3,591,601 rows, 1,315.3 s; npz sha256 7f25a31f63153b61c9838a40b8ab2eab4ab96455e6b5a33d2af17c6af60d560c) | valid: not rebuilt, not invalidated |

Why the pools entry holds, checked rather than argued:

- An entry is OK when its status is OK and its listed files exist with the listed shas (`entry_ok`). Nothing in it names
  chunk counts, part counts, the order or this file's sha. The two new fields are optional and absent from A0's entries.
- Its key, argv and outputs come from `pre_steps`, which uses only `QUESTIONS`, `shard_q`, `u2s_carve` and the roots. The
  self-test asserts all four steps' keys, shard 0/2's argv and outputs, and that they do not move with the rows now
  given to `pre_steps`; the real ledger was read under A1's code: both entries `entry_ok` True, shard 0/2's key and
  outputs are `pre_steps`'s.
- Comparing the A0 and A1 files by syntax tree: 123 top-level definitions are identical, among them `open_pko`,
  `rule_batch`, `head`, `part_arrays`, `pools_pass`, `merge_arrays`, `check_filed`, `beside`, `scripts`, `check_merge`,
  `check_squad`, `copy_bases`, `look_shard`, `cache_part`, `chain_build`, `typed_carve`, `kt_stage`, `record_carve`,
  `gate`, `prune_carve`, `entry_ok`, `shards`, `shard_q`, `QUESTIONS`, `SHARD_Q`, `SHARD_Q_PASSAGE`, `POOLS_BPR`,
  `ROWS_Q`, `LOOK_ROWS`. The changed ones are the ledger and lane code (`file_entry`, `run_step`, `do_step`,
  `run_lanes`), `step`, `pre_steps` (an optional argument), `post_steps`, `carve_job`, `check_job`, `chain`,
  `plan_table`, `main`, `selftest`, and `merge_carve` (one line, below). `CHUNK_Q` is removed.
- So the pools pass is the same code on the same inputs, and shard 1/2, built after A1, is the same computation as
  shard 0/2.

**One real consequence, handled.** Every pools part's head records the sha256 of the scripts it ran, this file among
them, and `merge_carve` refuses shards whose heads differ ("not built by one script on one input"). Editing this file
changes its sha, so shard 1/2, built under A1, would have been refused against shard 0/2 (A0, sha 8629c869...).
The merge now compares the heads with this file's entry counted as one value when it is the current file's sha or one in
`ZPKO_SAME` (A0's, declared here). A different sha for any other script, or for this file when it is neither, still
makes two shards differ. The self-test builds the shards both ways and refuses a third sha and a changed script.
Two rules follow: build no pools shard before A1 is committed, so this file's sha is fixed while shards are built; and a
later amendment appends the sha of the file it amends to `ZPKO_SAME` in the same change.

Not in the ledger, and so not affected: the killed `pools:webqsp/s1eval:1/2` run (its log under logs/ stops at 400 of 752
questions; no part file was written, and the chain rebuilds it).

## 6. Self-test

`python outputs/mp_unified/zpko.py --selftest` ends `zpko selftest: 20 checks pass`: A0's 11 and 9 for A1: the chunk and
part counts for all eleven carves, A0's keys and entries unchanged (an A0-format entry is skipped as OK, one with the new
fields is read the same, a changed file is not OK), the peak of a real 160 MiB child, the ledger fields, the preflight
(refuses, waits and starts when the memory returns, counts what a running step has yet to grow into, a lane that stops
for RAM files nothing), a job that raises stops the chain, the carve order, the counts guard, the merge across A0 and A1
shas, and the declared RAM. Eight single breaks of the A1 code in a harness (the head comparison, the chunk rule, the
counts guard, the preflight, the peak, the order, `ZPKO_SAME`, the ledger fields) each failed the self-test.

## 7. What A1 does not do

It builds nothing and launches nothing. It does not change a threshold, a rule, a number any step computes, the
carves, the pick, or what is committed (outputs/pko stays uncommitted). The laptop run is
`python outputs/mp_unified/zpko.py chain --carves webqsp/s1eval --lanes 1 --threads 2` once A1 is committed; with fewer
than 7 GiB available it waits for the memory, and stops after two hours with code 2 rather than starting a step that
will be killed. PKO on the lab host is declared separately (docs/PKOH_PKO_ON_THE_HOST.md; `outputs/mp_unified/zpkoh.py`
adds the host's wrapper and does not edit zpko.py).
