# KG: U2d's base set on Kaggle's T4, a second GPU line

Declared 27 October 2026. Code: `outputs/mp_unified/zkg.py` (the fits, compares, null and verdict) and
`outputs/mp_unified/zkg_session.py` (the Kaggle session). No KG fit has run. Nothing in this file reads a KG number.

## 1. Why this, now

- **The GPU stages have no card.** The lab host has been offline since 20 October. Workspace 3, which holds U2d's base
  build, renews on 1 November (GL, section 6). Workspaces 4 and 5 are spent. Workspaces 6 to 10 need the user's
  payment method.
- **Kaggle has free T4s.** The user's Kaggle account gets about 30 GPU-hours a week, with two NVIDIA T4s per session.
  A probe kernel on 26 October saw:
  - two Tesla T4s, 15,360 MiB each;
  - torch 2.11.0+cu128, CUDA 12.8, cuDNN 91900, Python 3.13.15, numpy 2.1.3;
  - 4 CPUs and 31.3 GB of RAM;
  - 3.21 TFLOPS on an fp32 matmul.
- **A new device needs its own bases and nulls.** The program of 24 October says: "Bases and nulls are refit on each
  new device" (docs/PROGRAM_2026_10_24.md, section 4). U2d's base set is that set on the L40S:
  - zrckp and zspk with seeds 0, 1 and 2;
  - zg6 with seed 0;
  - all on both splits.

KG fits the same set again on the T4, from the same build and through the same code. It decides no idea and changes no
filed verdict. It makes the T4 a device that later stages can declare. G6b, G6's next form (GL, section 7), comes first.

## 2. The fits

Fourteen fits, each the T4 twin of one of U2d's:

| KG fit (L-musique; L-hotpotqa adds `-hp`) | U2d's fit on the L40S | arm | seed |
| --- | --- | --- | --- |
| `scr-kg-base-zrckp` | `scr-u2d-base-zrckp` | zrckp | 0 |
| `scr-kg-base-zrckp-s1` | `scr-u2d-base-zrckp-s1` | zrckp | 1 |
| `scr-kg-base-zrckp-s2` | `scr-u2d-base-zrckp-s2` | zrckp | 2 |
| `scr-kg-base-zspk` | `scr-u2d-base-zspk` | zspk | 0 |
| `scr-kg-base-zspk-s1` | `scr-u2d-base-zspk-s1` | zspk | 1 |
| `scr-kg-base-zspk-s2` | `scr-u2d-base-zspk-s2` | zspk | 2 |
| `scr-kg-base-zg6` | `scr-u2d-base-zg6` | zg6 | 0 |

**The same path as U2d's.** zkg.py runs zu2d.py's train and read bodies:
- U2d's base build, through zu2d.py's roots for it;
- P1c2's card: variant p, config `2e-3:1e-4:0.1:8:2`, hidden 128, eight epochs, read as p@swa;
- P1c3's KT input;
- the seed from the fit's name;
- rmatch's train and read.

KG passes none of the flags U2d's train and read refuse. Before training, each arm is checked against U2d's arm.

**The reads.** Each fit is read on the six s1eval carves of the base build at p@swa, with `--rows-cap 65536`. That is
G6's batch (docs/G6_FREE_GNN.md, section 4). A batch holds whole questions, so the scores do not change (U2d, section
5). The cap keeps zg6's read inside the T4's 15 GiB: at the default batch, U2d's zg6 read peaked at 13.34 GB on the L40S.

**One session per fit.** Each fit trains and reads in one session, in its own process, since /kaggle/working does not
carry over between sessions.

**What each fit records.**
- screen.json and read.json carry KG's stamp, under the key `kg`:
  - declared_in, fit, build, arm, seed, split, and the fit's U2d twin;
  - the device and its environment;
  - the gate's sha256, the pools' construction and the head;
  - the KT, rmatch, zrc and link records;
  - the code's sha256s: zu2d.py's list, plus zkg.py.
- kg_timing.json holds the seconds, the peak RAM and the rows cap.

## 3. The device

**Where a fit may run.** A KG fit trains and reads only when all three hold:
- it runs in a Kaggle kernel (KAGGLE_KERNEL_RUN_TYPE is set);
- the GPU is a T4;
- the image's tag is `kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3`, the probe's.

Anything else stops before any data is read.

**The device record** is U2d's five fields: the GPU's name, the image's tag, torch, CUDA and cuDNN.
- A fit's read must run on the device the fit was trained on.
- A compare refuses fits from two devices (U2d's one-device rule).
- The kernel is pushed with its image pinned to its first version's, so every session runs on one image. If Kaggle's
  image is not the probe's, every fit stops at its first check, and KG is amended before any number.
- The two T4s of a session count as one device: the record has no GPU index.

**The environment.** Each fit runs with U2d's image environment (zu2d_modal.py's ENV): two BLAS and two OpenMP threads,
and OpenBLAS and numba set for Haswell. Two fits run at once on the kernel's 4 CPUs, one per T4.

**Never recorded:** environment variables other than KAGGLE_KERNEL_RUN_TYPE (Kaggle sets token-like ones), the account,
and any path outside the repository.

## 4. The data and the session

**The data: nine private datasets on the user's Kaggle account.**
- **Eight hold U2d's base build: what its fits read.** That is the cache, chains/rmatch, chains/zrc, kt, links, the
  pools without their parts, and gate.json: 1,332 files, 23.2 GiB.
  - They were copied from workspace 3's volume, read only, in groups of at most 3.6 GiB.
  - On the way, each file was checked against a sha256 the laptop holds: its twin's bytes, its part's record.json, or
    the pools.json or basis json that names it.
  - The look (17.8 GiB) is not copied, since only U2d's builds read it.
- **The ninth holds U2d's inputs:** the 203 files of outputs/u2d/inputs.json and inputs_pins.json (368 MiB), each
  checked against its list.

Each dataset carries a manifest.json: each file's stored name, repository path, size and sha256. Files are stored under
short names, since the Kaggle client names a resume file after each upload's full path, which must stay under Windows'
260 characters. The manifest maps each name to its path.

**The code.** The kernel has no internet.
- A tenth private dataset holds main's history as a git bundle.
- The kernel's code file, `kg_stub.py`, comes from `zkg_session.py stub`. It finds the bundle, clones it, checks out the
  commit, and stops if HEAD is not that commit.
- `stub` refuses to write anything unless the tree is clean, HEAD is main, and origin/main holds HEAD.

**The session: `zkg_session.py run`, in the kernel.** In order, it:
1. stops on a clone that differs from its commit;
2. finds the nine datasets by their manifests, each attached once;
3. checks every file's size and sha256 against its manifest, and the inputs' manifest against the two lists git holds;
4. links each file at its repository path; a file git already holds must be the same bytes up to line ends;
5. links outputs/screen/fits to /kaggle/working/fits, so the fits are the kernel's output;
6. runs `zkg.py fit` for each named fit, one queue per T4, each fit in its own process, and starts no fit after 8 hours.

**Amended 27 October, before any KG number, with G6b's declaration** (docs/G6B_GATED_BACKOFF.md, section 7):
- step 6 runs each named fit through its stage's script: `zkg.py fit` for KG's scr-kg- fits, `zg6b.py fit` for G6b's
  scr-g6b- fits;
- the session stops on a name that no stage there fits, or on a fit queued twice, before reading any data;
- `zkg.py take` leaves scr-g6b- folders to G6b's take.

session.json records the commit, the manifests' sha256s, the versions, the environment, and each fit's exit code and
seconds.

**After a session, on the laptop.**
1. `kaggle kernels output` into a new, empty folder.
2. `zkg.py take` copies each fit's records and reads into outputs/screen/fits/<name>, never models.pt. It first checks
   KG's stamps, and a file that is already there must have the same bytes.

## 5. The compares and the null

Ten compares, lean_screen's compare (p@swa on s1eval), each stamped with KG's record:
- the null: `scr-kg-null-zrckp-s1`, `-s2`, `scr-kg-null-zspk-s1`, `-s2`, and their `-hp` twins, each fit of seed 1 or 2
  against its arm's seed-0 fit;
- `scr-kg-zg6-vs-zspk` and `-hp`.

KG's floors are screen_recall's, per arm, split and read: max(0.0075, 2·sqrt((D1² + D2²)/2)), where D1 and D2 are the
seed-1 and seed-2 R@5 deltas. zkg.py's `kg_null` computes them as zu2d.py's `u2d_null` does, from KG's compares.

## 6. The rule

- **BASE_READY:** all fourteen fits are trained and read under KG's stamp on one device, a T4 on KG's image, and all
  ten compares are KG's, made from those fits on that device.
- **INCOMPLETE:** otherwise. The verdict names each missing or refused piece, and the next step is to supply it.

After BASE_READY, a later stage may declare the T4 as its device. KG's fits are then its bases and KG's nulls its floors.
G6b comes first.

## 7. Beside, deciding nothing

- Each fit's R@5 per read on the T4, beside U2d's fit of the same arm, seed and split on the L40S: the same fit on two
  devices.
- KG's floors beside U2d's.
- **G6's call, replicated on the T4.** zg6 against zspk (screen_pair over KG's two compares), re-called under KG's zspk
  null (screen_recall) and read by G6's rule. G6 stays as filed on the L40S: NOT_ADOPTED (GL, section 7). The files are
  `outputs/kg/kg-g6-zg6-pair` and `outputs/kg/kg-g6-zg6-pair-recall`.
- Seconds and memory per fit, on the T4 and on the L40S.

## 8. How it runs

**Two sessions, two queues each.** The hours below are U2d's train-plus-read seconds on the L40S.
- **Session 1, the null's seeds:**
  - GPU 0: zrckp s1, s1-hp, s2 and s2-hp (0.81 h on the L40S);
  - GPU 1: zspk s1, s1-hp, s2 and s2-hp (0.90 h).
- **Session 2, the seed-0 fits and zg6:**
  - GPU 0: zg6-hp, zrckp and zrckp-hp (1.34 h);
  - GPU 1: zg6, zspk and zspk-hp (1.06 h).

If session 1's T4 times would put a session-2 queue above 8 hours, session 2 splits in two. The split decides nothing.

G6b's two fits go first in session 1's queues, before the null's seeds (docs/G6B_GATED_BACKOFF.md, section 7). Where a
fit runs decides nothing.

**Time and cost.** U2d's fourteen fits took 14,780 s on the L40S (4.1 h). The T4 is slower. At three to five times
slower, that is 12 to 20 T4-hours in two queues, so about 6 to 10 hours of sessions, inside one week's quota. It costs
nothing.

**Commands.** In the kernel, the stub runs the session. On the laptop:

    python outputs/mp_unified/zkg_session.py stub --out DIR --gpu0 N,N --gpu1 N,N --rows-cap 65536
    kaggle kernels output <kernel> -p <new empty folder>
    python outputs/mp_unified/zkg.py take --from <folder>
    python outputs/mp_unified/zkg.py compare-all
    python outputs/mp_unified/zkg.py verdict

`stub` writes the kernel's code file and the bundle. The bundle goes up as a private dataset. The kernel is pushed with
the ten datasets attached, the GPU on and internet off.

**What is committed.**
- Committed: KG's compares (outputs/screen/scr-kg-*), and the beside pair, re-call and verdict (outputs/kg/).
- Never committed: the fits' folders, weights, reads, session.json, the logs and the datasets' manifests.

## 9. Self-tests, passed before this commit

`python outputs/mp_unified/zkg.py --selftest` runs on fake files in a temporary folder:
1. **The plan:** fourteen fits and ten compares, each with U2d's namesake.
2. **Where:** train and read refuse anything but a Kaggle T4 on KG's image, before reading data.
3. **Stamps and one device:** it refuses a fit that is not KG's, a read on another device, and fits from two devices.
4. **Take:** records and reads are copied, and models.pt never is. It refuses an unread fit, a name that is not KG's,
   and a file with other bytes. It leaves a scr-g6b- folder alone (added with G6b).
5. **The null:** the floors are checked. It refuses a compare that is not KG's, one from another device, and seeds
   compared with different seed-0 R@5s.
6. **The verdict:**
   - BASE_READY on the full fake set, with G6's call on the T4 beside;
   - INCOMPLETE for a missing compare, an unread fit, and a fit on another GPU.
7. Nothing is written under KG's real folders.

`python outputs/mp_unified/zkg_session.py --selftest` runs on a toy repository and nine toy datasets:
1. find, plan, verify and link put every file at its repository path, keep git's file, and link the fits' folder.
2. **Refusals:**
   - a changed byte, and a file of the wrong size;
   - a missing group, and a group attached twice;
   - a path outside the build, and a pools part;
   - an input that is not in U2d's lists;
   - git's file with other bytes, and a link that points elsewhere.
3. **The stub:** its code compiles, and a toy bundle clones and checks out at its commit.
4. **The environment:** each fit's environment is zu2d_modal.py's ENV.
5. **The stage scripts** (added with G6b): each name maps to its stage's script, and a name that no stage there fits,
   or a fit queued twice, stops `run` and `stub` before anything else.

The text encoder stays frozen. webqsp never trains. Test splits are never read. There is no LLM or transformer, and no
new graph.

## 10. Results

Not run yet.
