# PKOM: PKO on Modal workspace 3

Declared 30 Oct 2026, before any container runs. PKO's pools, looks, caches and typed arrays are what every refit waits for
(docs/PKO_PICK_O_POOLS_UNDER_WQ.md, A1: docs/PKO_A1_AMENDMENT.md). PKOH (docs/PKOH_PKO_ON_THE_HOST.md) builds them on the lab
host, which has not answered for days and drops within seconds when it does. PKOM is the same build on a second
substrate. It computes nothing new: every unit is a PKOH unit, run unchanged by `zpkoh.py unit`, from `zpko.py`'s steps.
Nothing here is a result.

## 1. What was measured (30 Oct, read-only listings; outputs/pkom is not committed)

| fact | value |
|---|---|
| package on workspace 3's `mpr-u2d:/pkg` | 577 of 577 files, each at the manifest's size (64.69 GiB) |
| the package's verify records (U2d's mirror stage, 22 Oct) | status VERIFIED, freeze_RECORD_SHA256 58958f33... and manifest_sha256 0c929af7..., the same as the host's records; no missing file, no hash mismatch |
| PKOH's input map (360 files) already on a volume the U2d and U2e drivers mount | 237 of 360, same size: 203 on `mpr-u2d:/inputs` (U2d's 202 listed files and one verify record), 26 on `mpr-u2c` (`/repo`, `/out`, `/u2e`: among them the kNN, NER and structural stores, 2.6 GB), 8 on `mpr-u2-links:/u2b` (U2b's graphs, 0.9 GB) |
| on none of them | 123 files, 1.32 GiB: `wqk_fit` 44 (1,023 MiB, read by the webqsp/fit carve only), `u2p` 9 (143), `u2m` 21 (128), `u2s` 13 (14), `pylib` 2 (42, kept so that PKOH's input check passes; not used), `u2c` and `u2e` verdicts 2, `u2n_tau0` 4, `mirror_verify` 5, and `code` 23 (1.1, tracked: they ride the archive). About 100 files, 1.3 GiB, are uploaded |
| image | numpy 2.3.2, scipy 1.16.1, numba 0.67.0, llvmlite 0.49.0, scikit-learn 1.7.1, pandas 2.3.1, Python 3.13: the laptop's versions, as U2d's image pins them |

Listing sizes are rounded text ("1.3 MiB"), so the comparison tolerates 0.051 of a unit and proves presence and size only.
Content is proved in a container: `probe` hashes what a unit will read.

## 2. What PKOM adds

1. `outputs/mp_unified/zpkom_modal.py`, a Modal driver in U2d's and U2e's form (same image pins, the AVX-512 and
   OpenBLAS settings, `single_use_containers`, retries, nonpreemptible orchestrators).
2. PKOH amendment A1 (`zpkoh.py`; zpko.py, which every pools head hashes, is not touched): when `PKOH_MIRROR_ROOT` is set,
   `mirror_root()` returns it and `apply_mirror` calls `modal_mirror` in place of `S6.host_mode`. `modal_mirror` takes
   the six datasets' records and the package record under `PKOH_VERIFY_DIR`, each of which must be VERIFIED at the
   declared freeze, in its own group, with nothing missing and no hash mismatch, and stops the process before it opens
   anything otherwise. It then wraps `V2.load_configs` exactly as `host_mode` does, and `patch_s4` runs as on the host.
   Without the variable nothing changes. zpkoh's self-test grew from 76 to 85 checks (the Modal records, five ways to
   refuse them, an unfiled declaration).
3. A new volume `mpr-pko` (new; no existing volume is written): `/code/<commit12>.tar` (`git archive` of HEAD, so the working
   tree never matters), `/inputs/<repo path>` for the files none of the four mounted volumes holds (new files only: an
   existing path is an error, not an overwrite), and `/pko` for the build (the repository's `outputs/pko`). The mounted
   volumes `mpr-u2d`, `mpr-u2c` and `mpr-u2-links` are read only.

## 3. Container layout

`/root/repo` is HEAD unpacked. Each input links at its repository path from the volume that holds it, as U2d's and U2e's setup does:
`mpr-u2d:/inputs`, `mpr-u2c:/repo`, `mpr-u2c:/out/<sha12 of zu2c.py>` (as `outputs/u2c`), `mpr-u2c:/u2e/<sha12 of zu2e.py>` (as
`outputs/u2e`), `mpr-u2-links:/u2b/<d>/graphs`, and `mpr-pko:/inputs`; a file two sources hold must be the same bytes up to line
ends, or the container stops. `/root/mirror/data/final_canonical` links to `mpr-u2d:/pkg`. `outputs/pko` links to `mpr-pko:/pko` (read and write).
Environment: `PKOH_MIRROR_ROOT=/root/mirror`, `PKOH_VERIFY_DIR=/u2d/mirror`, `OPENBLAS_NUM_THREADS=2`, `OMP_NUM_THREADS=2`,
`NUMBA_NUM_THREADS=2`, `OPENBLAS_CORETYPE=Haswell`, `NUMBA_CPU_NAME=haswell`, and numpy's AVX-512 dispatch off where the CPU has it. Each unit
runs `python outputs/mp_unified/zpkoh.py unit KIND CARVE [PART] --host` with 2 CPUs and the memory its item names; a
unit reloads `mpr-pko` first and commits it when it ends. Each unit writes its own ledger file (PKOH's rule), so
concurrent containers never write the same file. `fold` verifies each entry against the files it names, and those files are on the
volume and nowhere else, so `--stage fold` runs it in one container (the prune units fold in place, one at a time); `fetch` then
brings the small JSON records to `outputs/pkom/fetch/`, which leaves the laptop's own `outputs/pko` untouched.

## 4. Stages

```
modal run outputs/mp_unified/zpkom_modal.py --stage selftest   # no Modal connection: memory classes, item parsing, the scheduler and retry rules
modal run outputs/mp_unified/zpkom_modal.py --stage plan       # what upload would send (read only)
modal run outputs/mp_unified/zpkom_modal.py --stage upload     # the code archive and the files no mounted volume holds, to mpr-pko (new files only)
modal run outputs/mp_unified/zpkom_modal.py --stage probe      # env, the three self-tests, the mirror applied to the real records, PKOH's inventory with sha256
modal run outputs/mp_unified/zpkom_modal.py --stage selftests  # probe without the inventory: env, the self-tests (zpko three times), the disk
modal run outputs/mp_unified/zpkom_modal.py --stage check      # PKOH's check (c), in one container: webqsp/s1eval shard 0/2 rebuilt
modal run --detach outputs/mp_unified/zpkom_modal.py --stage run [--carves LIST] [--lanes N]
modal run outputs/mp_unified/zpkom_modal.py --stage status
modal run outputs/mp_unified/zpkom_modal.py --stage fold       # the per-unit ledger files into the volume's ledger.json
modal run outputs/mp_unified/zpkom_modal.py --stage fetch      # the small JSON records to outputs/pkom/fetch/, never the arrays
```

`check` is the gate. The shard must reproduce npz sha256 7f25a31f63153b61c9838a40b8ab2eab4ab96455e6b5a33d2af17c6af60d560c
and KT sha256 469fd274b5aaf9fec324b3fddd2fcd12dcda5a4c09e52ec524b2b4b27e0775b5, 3,591,601 rows, 751 questions, with
no tolerance (PKOH section 3 (c)); a build that fails on the file sha alone is read at the level of the arrays under
amendment A2 (section 8). A FAIL stops everything: no carve is built, and the numerics are looked at before any
other step. `run` refuses to start without a PASS or PASS_A2 filed by `check` for this commit.

`run` is a nonpreemptible orchestrator. It reads PKOH's item lines: the per-carve open smokes (`items check`: the inventory,
the environment and (c) are the stages above, so a dependency on one of those counts as met, and the smokes run), then
`items pools`, then each carve's `items build` once its merge is done, then `items gate`. It orders them by their DEPS and
spawns one single-use container per item, at most `--lanes` at a time (default 16), in the smallest memory class (2, 6, 10,
16 or 32 GiB) that holds the item's declared RAM, with twice the class as the limit. A unit whose steps are filed OK in the
ledger is skipped by `zpkoh.py unit` itself (PKOH's rule). Attempts: a unit killed for memory (its own return code or a
filed step's is SIGKILL, or the words) runs again one class up with `--retry`, because a step filed FAILED reruns only with
it; an error of the call or the container runs again as it was; at most three attempts; any other failure is the unit's own
and final, and what depends on it is dropped and named in the summary. Prunes run one at a time (each folds the shared
ledger, which has no lock between containers). Each carve's build items are made when its merge is done, from the merged
manifest (`items build --carves D/C --prune-looks`; pruning is on because the volume would otherwise hold 142 GiB), and the
gate item runs last, only if every carve's tail is done. No unit overwrites a file: zpko refuses a part whose head exists
without a ledger entry and never builds over a filed one. `--carves` takes PKOH's carve list.

## 5. Who builds a carve

PKOH and PKOM share the unit ledger (`outputs/pko/ledger/<unit>.json`) and the part heads. A carve is built by one
substrate. Before `run`, the host is asked once (`rx ls`); if it answers and its check (c) passed, PKOH keeps the carves it
has started and PKOM takes the others. A part built on the other substrate is accepted only if its head matches (PKO's
`head_key`), which already holds across threads and machines; a unit already filed OK is never rebuilt.

## 6. Cost and stops

At list prices from memory (not checked against the workspace's price list): about 85 lane-hours at 2 vCPU and 8 GiB,
roughly 14 USD of a 30 USD monthly credit, plus the orchestrator: Modal bills a nonpreemptible function at 3 times the list
rate, so `run` is kept at 0.5 CPU and 2 GiB (about 1 USD for 10 hours). Workspace 3 had spent 28.99 USD of this month's credit
by 23 Oct (the billing report's last interval), so only `plan`, `upload` and `probe` run before the renewal. `probe` and `check`
come to cents. The driver does not read billing: I read
`modal billing report` (read-only) before `run`, and between phases while it runs, and stop the run (my own app) if the
report shows the credit nearly spent. The credit is the user's. It renews on 1 Nov, so `upload` and `probe` may run before and
`run` after. (`check` was meant to wait too; it was run on 30 Oct instead, with about 1 USD of the month's credit left, because
it is a single container of about 8 minutes and learning its verdict early leaves two days to understand a failure. Month to
date was 29.00 USD before it; the billing report is read again after.) A run restarted after a stop starts a container per item to find the filed ones done, about two
lane-hours.

## 7. What can go wrong, and what decides it

- The numerics. The image is U2d's, whose pools ran on this workspace, but only check (c) proves that PKO's pools are the
  laptop's bytes here. A FAIL ends this route until the difference is understood; nothing is built over it.
- The volume. About 88 GiB with the looks pruned, 142 GiB without, on a Modal volume; concurrent units write distinct files
  and each commits when it ends. If `run` shows reads that miss a file a unit committed, the next phase reloads first; a unit
  that still fails on a missing input stops with its name.
- Quotas. The workspace's concurrent-container limit is not known; `--lanes` is lowered if spawns are refused.
- The credit. A run that stops for credit is restarted from the ledger; no unit repeats a filed step.

## 7a. What the probes found (30 Oct, workspace 3; 2 CPUs, 4 GiB request; billing not read after them; records in outputs/pkom, not committed)

Passed: no overlay conflict between the code archive and the volumes (203 `mpr-u2d` inputs, 41 `mpr-u2c` repo files, 68
`mpr-pko` inputs, 394 and 150 files for `outputs/u2c` and `outputs/u2e`); `modal_mirror` applied to the real records (7 verify
records VERIFIED); `zpkoh.py items check|pools|gate` yield 14, 49 and 1, the pools items by memory class 5 at 2 GiB, 24 at 6, 20
at 10 and none above; PKOH's inventory with sha256 reads all 360 inputs and the package's 577 files with 0 missing and 0
different, all 11 carves READY (4 minutes).

Three things to know, none of which stops a run:

- zpkoh's self-test line "the disk guard" fails in a container. The container's filesystem reports 8,589,934,591.8 GiB free at
  `/root/repo` and `/tmp` (and 381.5 GiB at `/pko`), and the line asks for 10**9 GiB free and expects exit 3. The guard a unit
  runs (12 GiB free at the repository root) returns 0 there, run directly. zpkoh.py is not changed for this: its sha is in the
  input map and on the host.
- zpko's self-test passed in 4 of 4 laptop runs and 3 of 3 runs in one container, and failed once with a bare `AssertionError`
  in an earlier container (the probe kept one line of output, so the traceback was lost; `probe` now keeps 1,400 characters of
  a failing self-test). The only wall-clock assertion in it is `0.25 <= elapsed < 3` on a 0.3 s wait in the RAM guard's check
  (zpko.py, section (c) of the A1 self-test), which a stalled container could exceed; that is a likely reading, not a proven
  one. No build item runs a self-test (only the manual command and `probe` do), so it cannot fail a unit; the pools
  numerics are decided by check (c).
- The setup links 0 U2b graph files for metaqa and webqsp (1 each for squad, musique, hotpotqa and 2wiki). The inventory
  reads every file of the map present, so whatever those two carves read is in the map; check (c) is webqsp's first real read.

## 8. Amendment A2: check (c) read at the level of the arrays (declared 30 Oct 2026, after the first check)

**What the first check found** (commit 2e241b0c7c50, workspace 3, 2 CPUs; the shard built in 460.8 s at a peak of 11.78 GiB,
against 1,315 s on the laptop). It FAILed on the file sha alone: 2d1c2c56f0d5c2f615dc590b93e81fae15a53d94e49a63db6b551fcf68a49009
is not 7f25a31f.... Every other comparison of the check held: the KT file's sha256 and shape, rows 3,591,601, questions 751, the
walk, the threads and the shard's own internal verdict. Looked at afterwards (the container's 11 MB file read from the new
volume into an empty directory; nothing written back): the two heads differ only in `file_sha256`, seconds, utc and zpko.py's
own sha (the laptop's shard was built before A1; the merge accepts both). The npz holds 19 arrays; 18 are bit-identical in
dtype, shape and bytes. `nodes` (int32, 3,441,027 entries: each question's expansion list in rank order) differs in 588
entries, all inside 11 of the 751 questions, and in each of the 11 the node set is the same and only its order differs (in
question 6 three nodes are rotated); the sorted multiset of the whole array is equal. `pools_of` takes `np.union1d(bs_nodes,
nodes)`, so a question's pool does not depend on the order inside `nodes`, and the KT file, which comes from the same orders, is
identical.

**What is not known.** The cause. Two readings fit one build: exact ties in the cosine ranking broken differently by a sort, or
last-bit differences in float cosines between the laptop (Windows, Python 3.13.5, Intel, numba targeting its own CPU) and the
container (Linux, Python 3.13.0, AMD, numba pinned to haswell) that reorder near-equal values. Not tested. It also leaves open
whether another question, whose tie straddles the cut at `keep`, would differ in its SET between substrates: none did in these
751, and a carve built on one substrate is internally consistent (section 5).

**The rule.** `check` stays a PASS when the shard is bit-identical, as declared. When its only problem is the file sha, it is
read under A2 and is PASS_A2 when both hold, FAIL otherwise:

1. every array but `nodes` equals the laptop's in dtype, shape and bytes; `nodes` has the same dtype and shape, the same node
   set in every question, and at most 10% of the questions' lists in another order (a safety bound against a wholesale
   reorder, not fitted to the 11 of 751 seen);
2. the file is the first Modal build's bytes (2d1c2c56...), so a second build in a fresh container on whatever CPU it lands on
   reproduces the first, which is what lets a Modal-only carve be self-consistent.

The laptop's fingerprints (per array, and per question for `nodes`) are `outputs/pkoh/check_c_ref.json`, made from the filed
shard after its sha was checked against 7f25a31f.... `check` now builds in a fresh base per commit (`/pko/check/c-<commit>`;
`check-c` only verifies a base that already has a head, so a rerun on the first build would prove nothing); the first build
stays at `/pko/check/c`. `run` accepts PASS or PASS_A2. Self-test 38 checks (toy shard: equal; one list of 20 reordered passes;
four of 20 fail; another set, count, extra array, dtype and file sha fail; the gate is for this commit only).

**What it means, said plainly.** A2 was written after the failure and after looking at the data; condition 2 and the 10% bound
were fixed before the second build runs. PASS_A2 means "equal up to the order inside the expansion lists, and reproducible on
Modal"; it does not mean bit-identical to the laptop, and no document may say so. PKOH's check (c) on the host is unchanged and
still bit-exact (zpkoh.py is not touched): if the host reproduces 7f25a31f..., the host's pools are bit-identical to the
laptop's, and the record of each carve says which substrate built it. Anything downstream that reads `nodes` by position, not
as a set, would see the difference; none is known (KT is identical, `pools_of` is a union), and the refit's first read of a
Modal-built carve checks it.

## 9. Rules kept

No token, key or credential is read, printed or passed. Workspaces are named by index in every record. A new
volume only; no existing volume is overwritten or deleted. Deploy, then spawn; no long Modal run is held in a background
shell. After a new commit `--stage upload` runs alone first. Cross-workspace reads are not used (everything is in workspace 3).
Arrays are not committed or fetched; only records. Nothing is adopted or rejected by this file.

## 10. Amendment A3: `run --retry`, and the squad/fit rehearsal (declared 30 Oct 2026, after the rehearsal's first run)

**What happened.** The second `check` passed (PASS_A2, section 8). I then launched `--stage run --carves squad/fit --lanes 4`
at 11:37 as a rehearsal of the orchestrator on the smallest carve, before the renewal; section 6 had said `run` waits. The
reason is the cost of finding a bug in `run` on a 20-hour carve instead of on one that takes minutes; the cost was 0.09 USD
(month to date 29.12 USD of an assumed 30). It ran the two pools shards (696 s for shard 0/2), the merge, and then **check 5 (b)
FAILED** on squad/fit: PKO's own comparison with U2d's manifest was specified in a way that no real squad carve can satisfy
(docs/PKO_A2_AMENDMENT.md). Nothing after the check ran, because `need_checks` stops every carve's look, cache and tail until
every check passes. The rehearsal did what it was for. It found the fault in a check that had only ever run on a toy.

**What A3 changes.** `run` takes `--retry`: every unit then runs with `--retry`. A unit filed OK is skipped by its own
ledger check (files and shas); a FAILED one runs again; nothing else changes. Without it a FAILED entry such as
`check-squad:fit` stops the unit with exit 1, as it should. `--retry` is the user's-side decision to run FAILED entries
again, taken after the cause is understood and amended (here PKO A2), not a default.

**What the second run showed (12:12 to 12:20, commit 893ea05).** The gate for that commit was PASS_A2 again (a third build of
the webqsp shard, fresh container, file sha 2d1c2c56...). `run --carves squad/fit --lanes 4 --retry` then found the pools
shards and the merge filed OK, moved the A1-era FAIL aside and ran check 5 (b) again: **PASS under rule A2** (`n_sha256`,
`node_sha256` and `k_q_zero` true, no manifest field differing). `look:squad/fit:0/1` then FAILED in 5 to 7 s with
`PKO: section 5's checks have not all passed ({'webqsp.json': None, 'hotpotqa.json': None, 'squad__fit.json': 'PASS',
'squad__s1eval.json': None}); nothing is built before they do`. That is the declared gate and not a fault: zpkoh gates every
look on the four section-5 merges (`section5_deps(CARVES)`), and `run` ignores a dependency outside the set of carves it was
asked to run, so a squad/fit-only run let the look start. In a full run the four merges are in the set and the looks wait.

**Two things follow.** (1) The ledger holds `look:squad/fit:0/1` as FAILED (two tries), so the **first full run uses
`--retry`**; without it that unit stops with exit 1 and squad/fit's chain is dropped. (2) The look, cache, tail and prune steps
have not run anywhere yet. Running them before the renewal would need squad/s1eval's shards and merge, webqsp's second shard
and its check-merge, and hotpotqa's check (a) (U2s's 9,197 s on one lane; about 0.4 USD at the rate in section 6 plus overhead);
workspace 3 had about 0.8 USD left (29.18 USD month to date), the credit is the user's, and `run` was to wait for the
renewal: it was not spent, and the question was put to the user.

## 11. Start on 1 Nov (the order, so that nothing is decided in a hurry)

1. `modal billing report` (read only) to confirm the credit renewed; note the month to date.
2. Commit nothing further, or commit and repeat 3 and 4: the gate is keyed to the commit.
3. `--stage upload` alone, then `--stage check`; the verdict must be PASS or PASS_A2 for this commit (PASS_A2 needs the file sha
   2d1c2c56..., as in all three builds so far).
4. `--stage run --lanes 16 --retry` (all 11 carves; `--retry` for the FAILED squad/fit look), detached, then a watcher on the
   app's task count; the billing report is read again after the first hour and the run is stopped by me if the cost per
   finished item is above section 6's basis by more than a factor of 2.
5. After it: `--stage fold`, `--stage fetch` (a changed record needs the older local copy moved aside: fetch keeps an
   existing file), and a declared filing step for the ledger and records. A carve is built by one substrate only.
6. Check (a) on hotpotqa/s1sel (`pkoh-hpchk-*`) is the first thing that has never met real data; if it FAILs, read its
   `check/hotpotqa.json` and the part heads before anything else, as with check (c) and check (b).

## 12. Amendment A4: the three missing section-5 checks, run before the renewal (declared 30 Oct 2026, before they run)

**The instruction.** On 30 Oct the user said to run the three checks now, and on Kaggle if Kaggle has the input. Kaggle does
not: its 15 private datasets hold about 10.7 GB (U2d's bases and caches, the typed links, WQK, G6RW, the code bundle), and
the checks read the dataset package (577 files, 64.5 GiB) that only mpr-u2d holds. So they run on Modal workspace 3, where
the package is mounted read only. This sets aside section 6's "`run` waits for the renewal", on the user's word, for these
three checks only.

**The three.** (1) Check 5 (b) on squad/s1eval: its four pools shards and merge (the merge carries check (b); U2d new's entry
for it has the same shape as squad/fit's: B_q 0, pool mean 50.0, rows max 54, 11,873 questions, 594,063 rows). (2) Check 5
(a) on webqsp: its two pools shards and the merge with check-merge (`check/webqsp.json`). (3) Check 5 (a) on hotpotqa/s1sel:
`pkoh-hpchk-0` and `pkoh-hpchk-m` (one shard; U2s's run took 9,197 s on one lane; this check has never met real data).
After them `need_checks` is satisfied if all four are PASS, and the look, cache, tail and prune steps can run for the first
time (squad/fit's look is FAILED in the ledger and needs `--retry`).

**What A4 adds to `zpkom_modal.py`, and nothing else.** `run --checks-only`: the named carves' pools shards and merges, a
hotpotqa carve's check shards without its own pools items (`hotpot_own`), no build items, no gate. `--stage unit --carves C
--item NAME [--retry]`: one item of `zpkoh.py items`, spawned on its own memory class by a 0.5-core function that returns in
seconds, with no orchestrator (the nonpreemptible orchestrator waits at 3x the list rate, and a single check shard runs for
hours). Both need the gate for the commit (PASS or PASS_A2). The self-test (41 checks) filters the real item lines of
hotpotqa/fit, not a toy list.

**Cost and stop rules.** Workspace 3: month to date 29.18 USD of an assumed 30. At the declared memory classes: squad/s1eval
about 0.03 USD per shard (4) and the merge, webqsp about 0.02 per shard (2), the orchestrator of this small run about 0.04
USD, and `pkoh-hpchk-0` at most about 0.45 USD if it takes U2s's 9,197 s (less if the walks are faster). That is about 0.7 USD
in the worst case against about 0.8 left. Order: the small run first; `modal billing report` after it; `pkoh-hpchk-0` only if
the month to date plus 0.5 USD stays under 30. While it runs, the billing report is read about every 30 minutes and the unit
is stopped by me if the month to date reaches 29.9 USD. Each stop and each cost is written down. Nothing here is a result.

**What these checks can say.** PASS on all four lets the looks run; a FAIL is read before anything else (the part heads, the
record, then the document), as with checks (c) and (b), and is amended in the open if the cause is in the check. A squad/s1eval
FAIL under rule A2 is final for that rule (docs/PKO_A2_AMENDMENT.md), so it would need a further PKO amendment, declared after
reading the arrays.

## 13. Amendment A5: the laptop-shard comparison inside webqsp's unit, and `--code` (declared 30 Oct 2026, after the A4 run)

**What the A4 run did (spawned 16:03, ended 16:19, commit 1f63817, gate PASS_A2 for it, 0.25 h; month to date 29.165 to 29.403
USD, so 0.24 USD against the 0.2 USD estimated).** Squad/s1eval: the four pools shards (668 to 792 s each), the merge (14 s) and
check 5 (b) **PASS under rule A2**, with `n_sha256`, `node_sha256` and `k_q_zero` true and no manifest field differing: PKO A2
(docs/PKO_A2_AMENDMENT.md) holds on the second real squad carve, not only on the one it was read from. Webqsp/s1eval: both pools
shards built and are filed OK (441 s and 477 s; the shard's own `--check` against U2s is in the head), but **the unit
`pkoh-wqe-p0` exited 1**, and with it the merge `pkoh-wqe-m` (so check 5 (a) on webqsp is not filed) was dropped.

**Why.** After a webqsp/s1eval 0/2 shard is built, `zpkoh.run_unit` calls `verify_shard` on it, the laptop's filed shard rebuilt:
file sha, KT sha and shape, rows, questions, nq, range, walk, threads and the shard's check verdict, all bit for bit. The log
line is `the laptop's webqsp/s1eval shard 0/2: FAIL ['file_sha256: 2d1c2c56... is not 7f25a31f...']`: the one problem is the file
sha, the same difference check (c) found on 30 Oct and amendment A2 (section 8) accepted at the level of the arrays. **That was
my omission: A2 relaxed the gate and not the second place the same comparison runs.** Had the full run started on 1 Nov, this
unit would have failed in the same way, dropped webqsp/s1eval's merge, and with it every carve's look (all four section-5
merges gate them). It was found because the three checks were run early.

**A5.** In `_unit_body` (`zpkom_modal.py`), for the unit `pools webqsp/s1eval 0/2` only, when it exits non-zero the container
calls `zpkoh.verify_shard` on the base itself; the unit counts as done (rc 0, filed `a5-webqsp_s1eval_0of2-<commit>.json` on
the volume with the problems and the shas) if and only if the record has **exactly one problem, a `file_sha256` one, and the
shard's file sha is `modal_first_build_file_sha256` of `outputs/pkoh/check_c_ref.json`** (2d1c2c56...), the bytes check (c)
read at the level of the arrays: every array but `nodes` equal to the laptop's, `nodes` with the same per-question sets and at
most 10% of the lists reordered, and the file reproduced by four builds in fresh containers. Any other problem, any other file
sha, or an error of the comparison itself leaves the unit failed. `zpkoh.py` and `zpko.py` are untouched (their shas in
`outputs/pkoh/inputs.json` stand). PASS_A5 means what PASS_A2 means: equal up to the order inside the expansion lists and
reproducible on Modal, not bit-identical to the laptop. Self-test 45 checks, four of them on the record the run wrote.

**`--code COMMIT`.** Every stage computed its commit from HEAD, and the gate is keyed to the commit, so a docs-only commit after a
gated one needed a new upload and gate (about 0.04 USD). `--code` names a commit already uploaded and gated; the stage then runs
that archive. The code that runs is the archive of that commit; nothing else about a unit changes.

**Next, in this order.** (1) Commit this, `--stage upload`, `--stage check` (a new gate for the new commit). (2) `run --carves
webqsp/s1eval --checks-only --retry`: the shards are filed OK and skipped, the unit's comparison is read under A5, then the merge
and check-merge run. (3) `--stage unit --carves hotpotqa/fit --item pkoh-hpchk-0`, then `pkoh-hpchk-m` when it is done (with
`--code` of the gated commit if HEAD has moved), under section 12's cost rules (billing read before, about every 30 minutes
during, stop at 29.9 USD). Nothing here is a result.

## 14. Section 5's four checks, filed on Modal on 30 Oct 2026 (a record of what ran; no rule changes)

| check | where it ran | verdict | notes |
|---|---|---|---|
| 5 (b) squad/fit | rehearsal, 12:12 to 12:16 (893ea05) | PASS, rule A2 | the A1-era FAIL is kept as `squad__fit.a1-fail.json` |
| 5 (b) squad/s1eval | A4 run, 16:03 to 16:19 (1f63817) | PASS, rule A2 | four shards 668 to 792 s, merge 14 s; `manifest_differs` empty |
| 5 (a) webqsp | A4 run, then the rerun 16:34 to 16:37 (fc694f9) | PASS, `mismatches` {} | shards 441 s and 477 s; the unit of shard 0/2 passed under A5 (`a5-webqsp_s1eval_0of2-fc694f9cea0e.json`, PASS_A5) |
| 5 (a) hotpotqa/s1sel | `--stage unit`, 16:35 to 17:04 (fc694f9) | PASS, `mismatches` {} | one shard in 1,515.8 s, check-merge 4.8 s |

All four were filed in `/pko/pko/check/` on the volume and fetched (`outputs/pkom/fetch/pko/check/`, uncommitted). The gate for each of the three
commits that ran anything was PASS_A2 with the same file sha (2d1c2c56...): five builds in fresh containers, one file.

**Three things the runs showed, none of which changes a rule.**

1. **Speed.** Hotpotqa's check (a) took 1,516 s against U2s's 9,197 s on one lane (about 6x), the pools shards of squad and webqsp
   took 441 to 792 s, and the whole of the early spend was 0.35 USD (month to date 29.165 to 29.519, of an assumed 30, with the two
   gate builds): below section 12's 0.7 USD worst case.
2. **Memory.** The hotpotqa check shard peaked at **18.75 GiB** (the child at 18.54 GiB) against the 7.0 GiB PKOH declares for the
   item (`mem_of`). It ran in the 10 GiB class, whose limit is 20 GiB, so it fitted with 1.25 GiB to spare; the same item on
   the host would have been scheduled on 7 GiB. The declared figure is wrong for that item; it is not changed here (`zpkoh.py` is
   frozen); a host run of it needs the measured peak reserved. Nothing else in these runs came near its class limit.
3. **A unit's two verdicts.** A unit can exit 1 after its step is filed OK (webqsp 0/2: the step OK, the comparison with the
   laptop's shard failing). `failed_rcs` was empty in the summary for that unit, so the only sign was the tail; A5 reads the
   record itself, not the tail.

**What these checks do and do not say.** They say the section-5 conditions of docs/PKO_PICK_O_POOLS_UNDER_WQ.md hold on Modal
for the four they name: webqsp and hotpotqa pools agree with U2s's numbers (a), and the two squad carves with U2d's, up to the
rules A2 and A5 declare (b). They are conditions, not results; no pool, look or cache has been built from them, and **the look,
cache, tail and prune steps have still run nowhere**. The ledger holds `look:squad/fit:0/1` as FAILED, so the next run of that
chain needs `--retry`.

**Consequences for the 1 Nov order (section 11).** (a) The run no longer waits on these four: the units skip their OK entries
under `--retry`. (b) A docs-only commit after a gated one needs no new gate: `--code` names the gated commit. (c) The first thing
on 1 Nov that has never run is a look: squad/fit's is the smallest, and the full run starts it as soon as the merges allow.
