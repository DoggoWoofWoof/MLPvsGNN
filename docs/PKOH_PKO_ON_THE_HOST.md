# PKOH: PKO on the lab host

Declared before any host action. This file builds nothing, pushes nothing and files no number. It fixes where PKO's chain
runs, what is checked first, what is pushed, how the work is cut into restartable items, and the rules the checks are
judged by. Commit it, `outputs/mp_unified/zpkoh.py` and A1 (docs/PKO_A1_AMENDMENT.md, `outputs/mp_unified/zpko.py`)
before the first push; no shard is built before A1 is committed (a shard's head records zpko.py's sha, section 2 of the
A1 doc).

## 1. What this is

PKO (docs/PKO_PICK_O_POOLS_UNDER_WQ.md) is U2s's pick O built under WQ on eleven carves: 69,076 questions, 124.7M rows,
142.3 GiB (88.1 GiB without the look chunks). `zpko.py` with A1 is the one program and stays so. PKOH changes where it
runs. It changes no threshold, rule, number, carve or pick, and it does not edit zpko.py.

`outputs/mp_unified/zpkoh.py` adds only what the host needs:

| command | where | what |
|---|---|---|
| `inputs` | laptop | the input map `outputs/pkoh/inputs.json`: path, bytes, sha256, tracked or not, group, dataset scope, required or candidate; `--trace F...` cross-checks traces against it |
| `inventory` | host | read only: which mapped files exist with the mapped size (small ones by sha, code LF-folded), the package by size, free disk and RAM, and per carve READY or what it lacks. Prints no host path |
| `push-plan` | laptop | from a fetched inventory: the files the host lacks as `rx push --inputs` lines, GB and minutes at 1.2 MB/s |
| `env` | host | CPU features (AVX-512 flags), numpy, numba, llvmlite's CPU name, BLAS threadpool, thread variables |
| `check-c` | host | rebuilds the laptop's webqsp/s1eval shard 0/2 in a scratch base (`outputs/pkoh/c`) and compares it, section 3 (c) |
| `trace LABEL -- ZPKO ARGS` | host | one zpko step under an audit hook that records every file opened for reading, split into repository files and package files |
| `step -- ZPKO ARGS` | host | one zpko step with BLAS at 2 threads, numba from the pushed zip and the mirror in place of the package |
| `unit KIND CARVE [PART]` | host | one hostfeed item (section 7) |
| `carve D/C --lanes N` | host | the coarse mode: zpko's `chain` for one carve inside one rx job |
| `items PHASE` | laptop or `rx exec` | the hostfeed2 item lines (section 8) |
| `fold` | laptop | the per-unit ledger entries into `outputs/pko/ledger.json` |

Held, every one of them:

- **Two threads in every PKO process.** OpenBLAS's thread count changes the features (docs/GPU_HOST_HANDOVER.md, the
  six-dataset mirror work). rx sets `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to the job's CPUs, so
  `zpkoh.py` assigns all three, and `NUMBA_NUM_THREADS`, to 2 before numpy is imported, and passes them to every child.
  More lanes, not more threads, is how the host is used.
- **The mirror replaces the package in memory only.** `step` applies `S6.host_mode` (it checks the declared mirror root
  and the five pinned VERIFIED records, and refuses otherwise) and step 4f's `on_the_mirror` patch (`S4.load_script` and
  `S4.Opened`, so `m3b_compile.open_package` reads a config copy whose `package_root` is the mirror). No config file is
  edited, the freeze check still runs, and nothing under the frozen package is copied into the repository. The mirror
  root is written `<mirror>` below. The host's logs name it, so the host's `outputs/pko/logs` is never fetched or
  committed.
- **numba comes from the pushed zip.** `outputs/host_ops/pylib_run.py` unpacks it and puts it on `sys.path`; a child
  does not inherit that, so every child command is `python pylib_run.py zpkoh.py step -- ...`.
- No credentials on the host. No `rx gc`, no `push --prune`. No other project's job is touched. mpr priority 10.
  Pushing data is a declared step, and section 4 is that declaration.
- Fetch back only small JSON (ledger entries, manifests, trace and check records). Arrays stay on the host: they are
  regenerable, and pulling 88 GiB at 0.75 MB/s is about 35 hours.

## 2. Order

1. Commit A1, zpkoh.py and the two docs (`git add -f` under outputs/). zpko.py's sha is fixed from then on.
2. Laptop: `python outputs/mp_unified/zpkoh.py inputs` (writes `outputs/pkoh/inputs.json`, 115 KiB, 360 files with sha256).
3. `python tools/rx/rx.py push` (the tracked tree) and `rx push --inputs outputs/pkoh/inputs.json` (a declared 115 KiB).
4. CHECK, section 3: `pkoh-inv`, `pkoh-env`, `pkoh-c`, then `pkoh-open-*`. The coordinator reads the inventory and pushes
   what it names (section 4) before the dependent items can run.
5. Pools phase (49 items), then each carve's build items once its merge is done (section 8), then the gate.
6. Fetch the small JSON; `zpkoh.py fold`; the filed ledger and records stay uncommitted like A0's.

## 3. The CHECK stage

Nothing is built on the host before (a), (b) and (c) are done, and every pools, look, cache and tail item waits on
`pkoh-c` by name, so a failed check drops them (a hostfeed2 item whose dependency ends badly is dropped). Every host job
that reads a carve's inputs also re-checks them (sizes only, 360 stats) and exits with code 2 when one is missing or
another size, or the package is incomplete; it exits with code 3 below a disk floor (default 12 GiB). Neither files
FAILED.

### (a) The input map

Read off the loaders (`zpko.open_pko`, `zu2d_pools.open_carve`, `look_u2d`, `lean_cache`, `zp1c2.build`,
`zu2n.build_typed`, `S6.host_mode`), U2d's filed lists (`outputs/u2d/inputs.json`, `inputs_pins.json`, whose `new` chain
is PKO's look and cache chain), and checked on the laptop: every required glob matches, none is unmatched. Sizes
measured 30 Oct.

| group | files | MiB | not in git (MiB) | read by |
|---|---|---|---|---|
| `code` (the 22 scripts `zpko.scripts()` hashes, plus zpkoh.py) | 23 | 1 | 0.1 | all |
| `pylib` (pylib_run.py, the numba 0.67 / llvmlite 0.49 zip) | 2 | 42 | 41.6 | all |
| `mirror_verify` (manifest.json, five `verify_*.json`) | 6 | 0.2 | 0.2 | `S6.host_mode` |
| `m3b_csr` (18 stores: knn, ner, structural for six datasets) | 18 | 2,623 | 2,622 | S4's walk and kNN/EL/R links |
| `u2d_list` (universal_v2/six, six base, relations, step 1 records and bases, P1c2 and step 4d pools) | 202 | 368 | 368 | look, cache, links, KT |
| `step1` (`carves.json`) | 1 | 2 | 0 | `S4.Opened` |
| `u2s` (verdict; stats `<d>__<carve>.json/.npz`, webqsp on s1eval, the others on s1sel) | 13 | 14 | 13.7 | `open_pko`, check (a) |
| `u2m` (verdicts; `a/types/<d>.*`) | 21 | 128 | 127.3 | EM types, the carried walk |
| `u2p` (verdict; `types/<d>.*` for the four passage datasets) | 9 | 143 | 143.3 | score types |
| `u2b` (choice, the chosen graph W2_S and its build record, four passage datasets) | 9 | 902 | 902.0 | `U2C.graph_csr` |
| `u2c`, `u2e` (verdicts, stats) | 8 | 2 | 0 | U2e's graph check |
| `u2n_tau0` (candidate) | 4 | 0.2 | 0.2 | typed columns |
| `wqk_fit` (`outputs/wqk/step4d`, `p1c2` webqsp pools, `s1cache`) | 44 | 1,023 | 1,023 | the webqsp/fit carve only |
| package (`<mirror>/data/final_canonical`, 577 files, 69.30 GB) | 577 | | on the host since 30 Sept | all (2wiki 27.79 GB, hotpotqa 22.28, webqsp 9.17, metaqa 6.73, squad 2.35, musique 0.98) |

Total outside the package 5.12 GiB, 360 files, 294 of them untracked. Tracked files ride `rx push` of the tree; the
untracked ones are the push list. The map is static (read off code), so (d) below measures it.

### (b) The host inventory

`pkoh-inv` runs `python outputs/mp_unified/zpkoh.py inventory --map outputs/pkoh/inputs.json` and writes
`outputs/pkoh/inventory.json`. It touches nothing but reads. `rx exec` runs the same command for a quick look; the
coordinator reads the JSON back (a few KB) and runs `zpkoh.py push-plan --inventory F` on the laptop. A carve is READY
iff every required input of its scope (`all`, its dataset, and for webqsp/fit the WQK group) exists with the mapped size
and the package is complete by size. `--sha` hashes everything, `--sha-package` the 69 GB (minutes to tens of minutes, not measured; the
30 Sept verification already holds).

### (c) The replica: webqsp/s1eval shard 0/2 on the host, before any carve

The laptop's filed unit `pools:webqsp/s1eval:0/2` (A0, 29 Oct, `--threads 2`, section 5 (a) IDENTICAL) is the reference:

| field | value |
|---|---|
| pools npz sha256 | `7f25a31f63153b61c9838a40b8ab2eab4ab96455e6b5a33d2af17c6af60d560c` |
| KT npy sha256, shape | `469fd274b5aaf9fec324b3fddd2fcd12dcda5a4c09e52ec524b2b4b27e0775b5`, [3591601, 8] |
| rows, questions, nq, range | 3,591,601; 751; 1503; [0, 751] |
| walk, threads | `RWe@0.5`, 2 |
| laptop seconds | open 118.9, pass 1,179.9 |

`pkoh-c` rebuilds it with the same arguments (`pools webqsp --carve s1eval --shard 0/2 --threads 2 --check`) in
`outputs/pkoh/c`, so it cannot collide with a filed part, and writes `outputs/pkoh/c/check_c.json` (the verdict, both
sets of numbers, the host's seconds, the env record).

**Rule, fixed now.** PASS iff the npz sha256, the KT sha256 and shape, the row count, the questions, nq, the range, the
walk, the threads and section 5 (a)'s verdict all equal the table. There is no tolerance: `host_cpu_t8` was bit for bit
at 8 threads and this is a pools build at 2. With `--ref outputs/pko/pools/parts/webqsp__s1eval__0of2.json` (a 4 KiB push)
the whole head must also equal the laptop's apart from `seconds`, `utc` and zpko.py's own sha. FAIL stops PKOH: nothing
is built on the host, the first differing field is reported, and the next form is a declared change (a different env, or
the laptop run in A1's section 7, or PKO on Modal), not an edit. A FAIL closes only "PKO on this host and env at two
threads".

On a PASS the host's `seconds.pass` over the laptop's 1,179.9 s is the host's measured speed factor for the lane-hour
numbers in section 5.

If the laptop's unit is pushed with its ledger (the 10.5 MiB npz, the 54.8 MiB KT, the 4 KiB head, the 4 KiB check part
and `outputs/pko/ledger.json`, 65 MiB, about a minute), `pkoh-wqe-p0` finds the entry OK and builds nothing, so the
filed array is not regenerated elsewhere. That is the recommended default. Without the push the host builds it again
and `unit` compares the result with this table afterwards (exit 1 on a difference).

### (d) The open-smoke and the traces

`pkoh-open-<carve>` (eleven items, after `pkoh-c`) runs `pools <d> --carve <c> --limit 1 --threads 2` under the audit
hook with the mirror in place. It writes nothing under `outputs/pko` (zpko's `--limit` writes nothing), reads the package
and every pools input, and writes `outputs/pkoh/trace/open-<carve>.json` (the files it opened, repository-relative or
package-relative). Rule: PASS iff it exits 0. The traces then go to
`zpkoh.py inputs --trace outputs/pkoh/trace/*.json`, which lists traced untracked repository files the map does not
name (they join the push list before the carve's items run) and required mapped files no trace opened (candidates to
leave out).

The pools opener is what (d) covers. The look, cache and tail steps read the U2d `new` chain's files, covered by U2d's
filed list and by the code reading above. To measure them, `items ... --trace-carves squad/fit` adds `--trace` to that
carve's units, which then write `outputs/pkoh/trace/unit-<step>.json` for every step; squad/fit is 0.3 GiB and 0.11
lane-hours, and the small-first order runs it first anyway.

## 4. What would be pushed

Declared step. Sizes are the laptop's bytes; time is at 1.2 MB/s up (the DERP relay's rate, 29 Sept; a push over 512 MB
needs `--force`; an interrupted push resumes by bytes). The inventory decides what is actually sent.

The files the feeder history says are already on the host (it ran items that named them), so expected present and not
pushed unless the inventory says otherwise: the numba zip, `m3b_csr`, `u2d_list` (universal_v2/six, six base, relations,
step 1, P1c2, step 4d), the verify records, and the package (verified 30 Sept). If every one of them were absent the push
would be 3.18 GB, 44 minutes.

No feeder item has ever named the rest, so they are expected absent:

| scope | MB | at 1.2 MB/s | what |
|---|---|---|---|
| all | 0.2 | 0 | tau0 |
| metaqa | 7.9 | 7 s | u2s stats npz, u2m types |
| squad | 10.4 | 9 s | + u2p types, u2b graph |
| musique | 26.3 | 22 s | same |
| webqsp | 29.3 | 24 s | u2s stats npz, u2m types |
| hotpotqa | 376.0 | 5.2 min | + u2b graph 299 MB |
| 2wiki | 793.9 | 11 min | + u2b graph 620 MB |
| webqsp/fit (WQK) | 1,073.1 | 15 min | 44 files, only for that carve |
| **all** | **2,317** | **32 min** | |

So no carve can be assumed to start without a push, but the first one is small: check (c) and open-smoke on
webqsp/s1eval need 29.5 MB (25 s); metaqa/s1eval 8.1 MB; squad 10.6 MB; musique 26.5 MB. The tracked files travel with
`rx push` (66 of the 360). The inventory names the real list; `push-plan` prints it. The package, if it turned out to
be absent, is 69.3 GB, 16 hours, and would be a separate declaration.

## 5. Lanes and hours

One lane is one item: 2 CPUs and the RAM A1 declares (measured peak plus 1 GiB): pools and look 7.0 GiB for webqsp,
2wiki and hotpotqa, 9.5 for musique, 5.0 for squad and metaqa; a cache part 3.5 to 4.8; the merges and tails 1 to 4.

The host has 32 CPUs and 127.7 GB, less the 2 CPUs and 8 GB reserve: 30 CPUs (15 lanes) and 119.7 GB (17 lanes at 7 GiB,
12 at 9.5). **Recommended: 8 concurrent lanes at the start** (16 CPUs, 40 to 76 GiB declared, depending on the mix),
because the host is shared, mpr runs at priority 10, and the look and cache steps read large files through the page
cache. Raise to 12 after the first two hours if hostfeed2's paging guard and its available-memory floor have not
tripped. hostfeed2 sizes by measured peaks per arm; PKOH item names carry no seed token, so each item is its own arm and
the declared figure is what it asks for. These are recommendations, not measurements.

Work: 80 to 90 lane-hours at the laptop's speed (look 40.5, pools 30 to 39, the rest cache, rmatch, zrc, links, typed,
KT). At 8 lanes about 10 to 11 hours, at 12 about 7 to 8, times the host's factor from check (c). The critical path is
the tail of the large carves (metaqa/s1eval: 42.8M rows, 36 cache parts), not a lane. Counts (A1): 35 pools shards, 131
look shards, 177 cache parts, 11 tails.

## 6. Disk

Per carve (GiB, `zpko.py plan`; look in brackets): webqsp/fit 11.2 [1.9], squad/fit 0.3 [0.1], musique/s1fit 33.9 [16.3],
hotpotqa/fit 4.7 [1.5], 2wiki/fit 6.4 [2.2], metaqa/s1eval 33.9 [14.3], squad/s1eval 0.6 [0.3], musique/s1eval 17.8
[8.6], hotpotqa/s1eval 5.9 [1.8], 2wiki/s1eval 13.6 [4.7], webqsp/s1eval 14.0 [2.4]. Total 142.3 [54.2]. Add the inputs
(5.1 GiB, beside the package) and zpko's disk reserve (15 GiB).

- Keep the looks: 142.3 + 15 + 5.1, so about 165 GiB free to start.
- Prune each carve's looks once its readers are filed OK (`items build --prune-looks` adds `pkoh-<c>-x`, which folds the
  unit ledgers under a lock first because `prune` reads the shared ledger): small to large, the peak is the last two
  carves in flight with their looks, 118.7 GiB, plus the reserve and inputs, so about 140 GiB free. Pruned looks cost
  40.5 lane-hours to regenerate, so prune only when the disk says so.
- The handover recorded 590 GB free on 29 Sept; `auto_cleanup.log` shows 72 to 117 GB free on 5 to 7 Oct and 90 GB as
  the clean-up point. The current figure is unknown to me. `pkoh-inv` reports it (`host.free_gib`). If it is under
  about 140 GiB, archive finished carves to the HF archive with `scripts/host_archive_hf.py` (links signed on the
  laptop, no token on the host) and delete the archived arrays only after the archive is read back. That is the
  coordinator's step, not an item.

## 7. Restart design

The link flaps and the host goes away. Every unit is one rx job, idempotent, and recorded in its own file.

- **Units.** `copy-bases`; `pools D/C i/k`; `merge D/C` (with `check-merge` or `check-squad` where the carve has one);
  hotpotqa's section 5 shards `check hotpotqa i/k` and `checkmerge hotpotqa`; `look D/C i/n`; `cache D/C i/n`;
  `tail D/C` (rmatch, zrc, links, typed, KT, record; the chained carves have all six); `prune D/C`; `gate`.
- **The ledger.** zpko's `ledger.json` is rewritten whole under a thread lock, so two rx jobs would race. A unit writes
  `outputs/pko/ledger/<unit>.json` (same entry format: status, rc, seconds, peak working set, declared RAM, sha256 of each
  file). It skips a step that is OK in either its own file or the shared ledger with the files still matching. `fold`
  (laptop, after the small files are fetched) and `prune` (host) merge entries into `ledger.json` under a lock file,
  through a rename, and only entries whose files verify.
- **What a lost job leaves.** A pools shard writes its head last, so none or whole; zpko refuses to rebuild a filed part,
  so a rerun after a crash that left a head is a manual clean-up, and a crash before it just reruns (a shard is 1,000
  questions for webqsp and metaqa, 3,000 for the passage datasets, with no checkpoint inside). A look chunk is written
  to a temporary name and renamed and existing chunks are skipped, so a rerun resumes at the first missing chunk. A cache
  part writes each file by temporary name and rename and its `record.json` last. `merge` writes its manifest last. A tail
  step writes its record last.
- **Rerun policy.** No OK entry: run. OK with matching files: nothing. FAILED: refuse until `--retry`; no automatic
  deletion; the coordinator reads `outputs/pko/logs/<step>.log` on the host (never fetched) and re-adds the item with
  `--retry`. A host power loss files nothing, so the same item just runs again.
- **Section 5.** zpko's `need_checks` stays: no look, cache or tail step runs until `check/webqsp.json`,
  `check/hotpotqa.json`, `check/squad__fit.json` and `check/squad__s1eval.json` all PASS. In items this is four merge
  dependencies on every look.

## 8. The item form (hostfeed2)

`line|x|DEPS|NAME|CPUS|MEM|INPUTS|COMMAND`. The second field is ignored; DEPS are comma-separated item names that must
have ended rc 0; MEM is GiB; INPUTS are `;`-separated workspace paths that must already exist (an item whose inputs are
missing waits); the file is re-read every poll, so the laptop can append. Names are at most 30 characters (rx truncates
at 34). The shape is that of the existing numba items (`s4f2-*`) and lean_cache items (`lc1-*`).

```
line|x||pkoh-inv|1|1.0|outputs/mp_unified/zpkoh.py;outputs/pkoh/inputs.json|python outputs/mp_unified/zpkoh.py inventory --map outputs/pkoh/inputs.json
line|x|pkoh-inv,pkoh-env|pkoh-c|2|8.0|<code;pylib;inputs.json>|python outputs/host_ops/pylib_run.py outputs/mp_unified/zpkoh.py check-c --host
line|x|pkoh-c|pkoh-open-wqe|2|7.0|<same>|python outputs/host_ops/pylib_run.py outputs/mp_unified/zpkoh.py trace open-wqe --host -- pools webqsp --carve s1eval --limit 1 --threads 2
line|x|pkoh-c,pkoh-open-wqe|pkoh-wqe-p0|2|7.0|<same>|python outputs/host_ops/pylib_run.py outputs/mp_unified/zpkoh.py unit pools webqsp/s1eval 0/2 --host
line|x|pkoh-wqe-p0,pkoh-wqe-p1|pkoh-wqe-m|2|1.7|<same>|python outputs/host_ops/pylib_run.py outputs/mp_unified/zpkoh.py unit merge webqsp/s1eval --host
line|x|pkoh-mqe-m,pkoh-bases,...|pkoh-mqe-l37|2|5.0|<same>|python outputs/host_ops/pylib_run.py outputs/mp_unified/zpkoh.py unit look metaqa/s1eval 37/43 --host
line|x|pkoh-mqe-l0,...,pkoh-mqe-l42|pkoh-mqe-c12|2|4.0|<same>|python outputs/host_ops/pylib_run.py outputs/mp_unified/zpkoh.py unit cache metaqa/s1eval 12/36 --host
line|x|pkoh-mqe-c0,...|pkoh-mqe-t|2|4.0|<same>|python outputs/host_ops/pylib_run.py outputs/mp_unified/zpkoh.py unit tail metaqa/s1eval --host
```

`<code;pylib;inputs.json>` is `outputs/mp_unified/zpkoh.py;outputs/mp_unified/zpko.py;outputs/host_ops/pylib_run.py;`
`outputs/host_ops/pylib/numba067_llvmlite049_cp313_win_amd64.zip;outputs/pkoh/inputs.json`. Names: `pkoh-<dataset><carve>`
with `sq mu mq wq hp 2w` and `f` (fit), `sf` (s1fit), `e` (s1eval), then `-pI` pools shard, `-m` merge, `-lI` look
shard, `-cI` cache part, `-t` tail, `-x` prune; `pkoh-bases`, `pkoh-hpchk-I` and `-m` (hotpotqa's section 5), `pkoh-gate`.
Every item but `pkoh-inv` and `pkoh-env` waits, through its dependencies, on both.

Three phases, because look and cache counts need each carve's merged row count:

| phase | command | items | needs |
|---|---|---|---|
| check | `zpkoh.py items check` | 14 | nothing |
| pools | `zpkoh.py items pools` | 49 | nothing (small-first order from zpko's plan) |
| build | `zpkoh.py items build --carves D/C --as-given [--prune-looks]` | per carve: look + cache + tail (+ prune) | that carve's merge done on the host |
| gate | `zpkoh.py items gate` | 1 | |

The build phase runs on the host after a carve's merge item ends:
`python tools/rx/rx.py exec -- python outputs/host_ops/pylib_run.py outputs/mp_unified/zpkoh.py items build --carves squad/fit --as-given`
prints the lines (the ones starting `line|`), which the coordinator appends to `outputs/host_ops/hostfeed_items.txt`.
In all: 14 + 49 + 308 (131 look, 177 cache) + 11 tails + 1 gate = 383 items, 394 with pruning.
This file does not edit `hostfeed2.py` or `hostfeed_items.txt`.

## 9. Commands, in order (the coordinator)

```
python outputs/mp_unified/zpkoh.py inputs
python tools/rx/rx.py push
python tools/rx/rx.py push --inputs outputs/pkoh/inputs.json
python outputs/mp_unified/zpkoh.py items check --out outputs/pkoh/items_check.txt     # append its lines to hostfeed_items.txt
python outputs/mp_unified/zpkoh.py push-plan --inventory <the fetched outputs/pkoh/inventory.json>
python outputs/mp_unified/zpkoh.py items pools --out outputs/pkoh/items_pools.txt     # after the check items pass
```

## 10. What this does not do, and what is open

- It does not build, push or fetch anything, edit zpko.py, hostfeed2.py or hostfeed_items.txt, or touch the host,
  Modal or Kaggle. Nothing under `outputs/pko` or `outputs/pkoh` is committed (arrays, looks, caches, links, KT, typed,
  parts, logs, the ledger and the traces stay local, like A0's).
- The input map is read off code and U2d's lists, validated by the self-test on toy data, and not yet traced on a real
  carve. (d) is that measurement; a laptop trace of a passage carve needs 5 to 9.5 GiB and was not run (6.8 GiB free).
- The host's free disk, its present inputs and its speed factor are unknown until `pkoh-inv` and `pkoh-c` report.
- Whether to push the laptop's shard 0/2 (section 3 (c), 65 MiB) or let the host rebuild and compare it is the
  coordinator's call; the default above is to push.
- The lane count is a recommendation (section 5), not a measurement.
- The fetch of the small per-unit ledger files and trace records over `rx` is the coordinator's command.

## 11. The self-test

`python outputs/mp_unified/zpkoh.py --selftest`: 76 checks on toy data in temporary folders (no host, rx, git or
network): the input scopes and the push bytes by scope; the inventory's missing, changed, CRLF-only and size cases and
per-carve readiness; the guards (map missing, input missing, disk); the tracer and its classification of repository,
package and code files; the cross-check both ways; check (c) (a matching shard passes; a changed sha, KT, row count or
another look script fails; zpko's own sha and the seconds may differ); the mirror patch (once, in memory, by copy); the
child command with and without pylib and a trace; the item list (all 383 shapes, dependencies closed and acyclic, every
item waits on the inventory and on check (c), every look on section 5's four merges, one item per look shard and cache
part, declared RAM per dataset, short names, the open-smoke's `--host` before `--`); a unit through a fake runner (filed
in its own ledger with its peak, skipped when run again, skipped when the shared ledger holds it OK, FAILED needs
`--retry`, the shared ledger untouched, `fold` verifying files, the lock refusing a second taker and taking over a stale
one); the laptop's shard verified after a rebuild. zpko.py's own self-test still passes (20 checks).
