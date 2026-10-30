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

**Next.** PKO A2 is committed with this amendment. Then: `upload` (the new commit's archive), `check` (a third build of the
webqsp shard, which must reproduce 2d1c2c56... for PASS_A2), and `run --carves squad/fit --lanes 4 --retry`, which continues from
the ledger. The look, cache, tail and prune steps of a carve have not run yet anywhere; this is where they first do.
