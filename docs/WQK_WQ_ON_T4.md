# WQK: WQ's base set on Kaggle's T4

Declared 28 October 2026, before any WQK fit or number. Code: `outputs/mp_unified/zwqk.py` (the laptop's build of
webqsp's fit carve; in the kernel, the overlay, the gate and the fits; on the laptop, the take, compares, floors and
verdict) and `outputs/mp_unified/zwqk_session.py` (the Kaggle session). Nothing in this file reads a WQK number.

## 1. Why this, now

- **WQ fixed this stage.** WQ's section 9, step 2: "WQ on Kaggle's T4: a KG-style base set under WQ, so that G6r, R2g
  and pool attention are decided under WQ" (docs/WQ_WEBQSP_TRAINS.md). Every later T4 screen is decided under WQ, and
  the T4 has no WQ base set yet.
- **The host is offline.** It has been down since 22:17 on 27 October. WQ's own base set and PSM wait for it. The user,
  27 October: "this is why i said the host might go on and off so keep the queue full proper utilization and keep stuff
  ready".
- **The T4 line does not need the host.**
  - WQK's new training carve, webqsp's fit carve, is built on the laptop by WQ's own steps (WQ, sections 3 and 4).
  - The other ten carves of WQ's screen are U2d's base build. KG's eight Kaggle datasets already hold it, and each of
    their files was checked by sha256 (docs/KG_KAGGLE_T4_LINE.md, section 4).
- **The order.** WQ's section 9 lists PSM first and this stage second. PSM's fits run on the host after WQ's
  BASE_READY, and nothing here reads or changes them. So WQK runs now, beside the host's queue.
- **What it decides.** Only WQK's own BASE_READY. It decides nothing about WQ's host base set, PSM or any filed verdict.

## 2. The fits

Fourteen fits on WQ's two splits. This is KG's set, under WQ:

| WQK fit (WQ-musique; WQ-hotpotqa adds `-hp`) | arm | seed | WQ's host twin |
| --- | --- | --- | --- |
| `scr-wqk-zrckp-s0` | zrckp | 0 | `scr-wq-zrckp-s0` |
| `scr-wqk-zrckp-s1` | zrckp | 1 | `scr-wq-zrckp-s1` |
| `scr-wqk-zrckp-s2` | zrckp | 2 | `scr-wq-zrckp-s2` |
| `scr-wqk-zspk-s0` | zspk | 0 | `scr-wq-zspk-s0` |
| `scr-wqk-zspk-s1` | zspk | 1 | `scr-wq-zspk-s1` |
| `scr-wqk-zspk-s2` | zspk | 2 | `scr-wq-zspk-s2` |
| `scr-wqk-zg6-s0` | zg6 | 0 | none: WQ's host set has no zg6 |

**How they train.**
- **zrckp and zspk** train through zwq.py's train, unchanged (WQ, section 5), under zwqk.py's patches:
  - WQ's roots and records are moved to outputs/wqk;
  - the fit names are WQK's.

  zrckp trains through zp1c3p's train and zspk through zp1c3's, each with the seed from its name. zwq writes its `wq`
  block.
- **zg6** trains through zg6.py's train, unchanged, under the same patches (zwq's on_wq: WQ's carves and roots).
  - For the run, its splits are WQ's and its fit names WQK's.
  - zg6 refuses `--seed`, so it trains at lean_gpu's seed, 0, as KG's zg6 did.
  - zwqk writes the same `wq` block for it.
- **Everything else is WQ's:** the splits, webqsp's fit carve, the card (config, hidden 128, eight epochs, p@swa), the
  KT input, the pools and the device kind (cuda).

**The reads.** Each fit is read by its trainer's own read, on the six s1eval carves at p@swa. Each read takes
`--rows-cap 65536`, KG's batch. A batch holds whole questions, so the scores do not change (KG, section 2).

**The KT records.** zp1c3's load_kt checks each KT record's declaration and cache root (WQ, section 5).
- webqsp's fit carve's record is P1c3's (docs/P1C3_KT_INPUT.md). It was written on WQK's cache root,
  outputs/wqk/p1c2/cache, and goes through as it is.
- The other ten carves' records are U2d's base build's:
  - declared in docs/U2D_REFIT_ON_U2E.md;
  - written on outputs/u2d/base/cache, whose files WQK's cache root links.

  For those ten records only, zwqk hands load_kt and kt_records U2d's declaration and cache root.
- Every other check runs unchanged: the arrays' and the cache parts' sha256s, the rows, the ids, the dtype and the
  shape.

**One session per fit.** Each fit trains and reads in one session, in its own process (KG, section 2).

**What each fit records.**
- screen.json holds:
  - the trainer's record and stamp;
  - WQ's `wq` block: this file, the split, seed and arm, WQK's roots, and the gate's and overlay's sha256s;
  - WQK's `wqk` block: the fit, arm, seed and split, the device and its environment, the gate's, overlay's and check's
    sha256s, the KT, rmatch, zrc and link records, and the code's sha256s.
- read.json holds the `wqk` block: the device, the gate's sha256, the KT records and the rows cap.
- wqk_timing.json holds the seconds, the peak RAM and the rows cap.

## 3. The device

The device is KG's, unchanged (KG, section 3).

**Where a fit may run.** A WQK fit trains and reads only when all three hold:
- it runs in a Kaggle kernel (KAGGLE_KERNEL_RUN_TYPE is set);
- the GPU is a T4;
- the image is `kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3`.

Anything else stops before any data is read.

**The device record** is U2d's five fields: the GPU's name, the image's tag, torch, CUDA and cuDNN.
- A fit's read must run on the device the fit was trained on.
- A compare refuses fits from two devices.

**The environment** is U2d's image environment (zkg_session.FIT_ENV): two BLAS and two OpenMP threads, and OpenBLAS and
numba set for Haswell. Two fits run at once on the kernel's 4 CPUs, one per T4.

**Never recorded:** environment variables other than KAGGLE_KERNEL_RUN_TYPE, the account, and any path outside the
repository.

## 4. The data, the build and the session

**The build of webqsp's fit carve, on the laptop.** zwqk.py's laptop half runs WQ's own steps. Every WQ root and record
is moved from outputs/wq to outputs/wqk, and the records carry this file's declaration. zwq.py is imported, not edited.
- **W0 to W8**, as WQ declares them, without `--host`:
  - the look, in three shards;
  - the bases;
  - step 1's cache;
  - step 4d's and P1c2's pools;
  - P1c2's look and cache;
  - rmatch's and zrc's chains, and the links;
  - the KT input.
- **Section 4's reproduction check**, into outputs/wqk/check. It uses step 1's own cache (its webqsp s1eval parts are on
  the laptop) and step 4b's filed eval arrays. It must be REPRODUCED.
- **Thread counts** are WQ's: `--threads 3` for W3, W4, W8 and the check's two runs.
- **Amendments.** If WQ's build (its sections 3 and 4) is amended, WQK follows the amendment and is built again before
  any WQK number is used.

**What differs from the host's build.**
- The laptop is not the host. The float32 features of webqsp's fit carve need not be the host's bytes, since OpenBLAS's
  thread count changes them.
- The ten other carves are U2d's base build, made on Modal. Its pools and KT arrays are P1c2's and P1c3's: U2d's KT
  records say REPRODUCED beside P1c3's. Its float features need not be the host's bytes.
- So WQK is its own base set. Its fits, and every later WQ screen on the T4, read the same files. A WQK fit is set beside
  a host fit only in section 7, where nothing is decided.

**The data: KG's nine private datasets, and WQK's.**
- **KG's nine, unchanged:** U2d's base build (cache, chains, KT, links, pools and gate) and U2d's inputs (KG, section 4).
- **WQK's groups: webqsp's fit carve as the laptop built it.** These are every file the gate and the fits read, each at
  its path below outputs/wqk:
  - bases.json and check/check.json;
  - W0's look;
  - W2's cache;
  - W3's and W4's pools (the manifests, and webqsp's fit file);
  - W5's look;
  - W6's cache;
  - W7's chains and links;
  - W8's KT input.

  outputs/wqk/ship.json (`zwqk.py ship`) lists exactly these files, with each one's size and sha256. It is committed
  before any session runs. The groups hold exactly ship.json's files, each in one group, and at most 3.6 GiB each.
  Files are stored under short names with a manifest, as KG's are: each file's stored name, repository path, size and
  sha256.
- **Ship's preflight.** `zwqk.py ship` writes nothing unless all of these hold:
  - the check is WQK's and REPRODUCED;
  - every piece, W0 to W8, is present;
  - zwq's gate, run on the laptop at WQK's roots, leaves only what the kernel's overlay settles: W9, and the ten
    carves' cache parts, which are U2d's;
  - U2d's records of the ten carves, held in git, agree. Each cache part names step 1's two bases by W1's sha256s. Each
    KT record is U2d's (its declaration and cache root) over those parts. Each carve's links, and the chains of
    webqsp's and metaqa's s1eval carves, have their part records.
- **The code:** main's history as a git bundle, a private dataset, as KG's.

**The session: `zwqk_session.py run`, in the kernel.** In order, it:
1. stops on a queue that names anything but WQK's fits or names a fit twice, a rows cap other than 65536, or a clone that
   differs from its commit;
2. finds KG's nine datasets and WQK's groups by their manifests, each attached once;
3. checks WQK's groups against ship.json. Together they must hold exactly its files, each in one group, at its size and
   sha256, at a path below outputs/wqk. A file outside ship.json, a ship.json file in no group, a file in two groups, or
   a group's file that is not ship.json's bytes stops the session;
4. checks every file's size and sha256 against its manifest, and the inputs against U2d's two lists, as KG's session
   does;
5. links each file at its repository path; a file git holds must be the same bytes up to line ends;
6. **W9 on the T4, the overlay:** `zwqk.py overlay` links every file of the ten other carves' folders in U2d's base build
   at WQK's roots:
   - the folders are the cache, links and KT, plus rmatch's and zrc's chains for webqsp's and metaqa's s1eval carves;
   - each file gets a symbolic link, since the datasets are read only, on another file system;
   - each folder is checked whole there (p1c2_cache_restore's check);
   - outputs/wqk/overlay.json is zwq's record, with U2d's folders as the sources.
7. **W10, the gate:** `zwqk.py gate` runs zwq.py's gate, unchanged, at WQK's roots on the kernel's files. It checks W1 to
   W9, the bases each cache part names, and the check. If it is not PASS, no fit runs.

   The overlay and the gate each run in their own process, and each record is copied to /kaggle/working/wqk as it is
   written. If either does not exit 0, the session stops before any fit;
8. links outputs/screen/fits to /kaggle/working/fits;
9. runs `zwqk.py fit --name N --rows-cap 65536` for each named fit: one queue per T4, each fit in its own process, and
   no fit started after `--start-before` hours.

session.json records the commit, the manifests' and ship.json's sha256s, the versions, the overlay's and the gate's exit
codes, and each fit's exit code and seconds.

**After a session, on the laptop.**
1. `kaggle kernels output` into a new, empty folder.
2. `zwqk.py take` copies each fit's records and reads into outputs/screen/fits/<name>, never models.pt. A fit without
   WQK's stamp, and its read on the device it was fitted on, is left where it is and named. A folder that is not one of
   WQK's fits, or a file already there with other bytes, stops the take.
3. The same take copies the session's gate.json and overlay.json to outputs/wqk/sessions/<the gate's sha256, first 12
   characters>/.

## 5. The compares and the floors

Ten compares, by lean_screen's compare (p@swa on the six s1eval carves). Each is stamped with WQK's record and the fits'
device.
- **The null, in WQ's form** (WQ, section 6): each seed-1 and seed-2 fit against its arm's seed-0 fit of the same split.
  The files are `outputs/screen/scr-wqk-<arm>-s<k>[-hp].json`, made by zwq.py's compare at WQK's names.
- **zg6 against zspk** on each split: `outputs/screen/scr-wqk-zg6-vs-zspk[-hp].json`. These are for section 7 only.

**The floors** are WQ's (WQ, section 6; screen_recall.floor_of), made by zwq.py's floors at WQK's names. The records are
outputs/wqk/floors-zrckp.json and floors-zspk.json, each with an `.md`.

## 6. The rule

- **BASE_READY:** all of these hold:
  - every gate that a fit's train, its `wq` block or its read names is a session's PASS gate, filed under
    outputs/wqk/sessions with that session's overlay, and naming the REPRODUCED check;
  - each fit's overlay is its gate's;
  - all fourteen fits exited 0 and are trained and read under WQK's stamp on one device, a T4 on KG's image. Each is
    read on the six s1eval carves with `--rows-cap 65536`;
  - all ten compares are WQK's, made from those fits on that device;
  - both floors records are OK, and each equals the floors recomputed from its compares.
- **INCOMPLETE:** otherwise. outputs/wqk/base-verdict names each missing or refused piece, and the next step is to supply
  it.

After BASE_READY, a later WQ stage may declare the T4 as its device. WQK's seed-0 fits are then its bases, and WQK's
floors its floors.

## 7. Beside, deciding nothing

- **The switch on the T4.** Each WQK fit's R@5 per read, beside KG's fit of the same arm and seed on the old split with
  the same wiki side (L-musique for WQ-musique, L-hotpotqa for WQ-hotpotqa). This is what WQ's switch does on the T4;
  WQ's section 8 shows it on the host. webqsp is expected to rise, since it now trains, and metaqa to fall, since it is
  now zero-shot.
- **WQK's floors beside KG's**, both on the T4.
- **The same fit on two devices**, once WQ's host fits are filed: each WQK fit beside its host twin (section 2's
  table), and WQK's floors beside WQ's.
- **G6's call under WQ, on the T4.** zg6 against zspk (screen_pair over the two compares), re-called under WQK's zspk
  floors (screen_recall) and read by G6's rule, as KG's beside did. The files are `outputs/wqk/wqk-g6-zg6-pair` and
  `outputs/wqk/wqk-g6-zg6-pair-recall`. G6 stays as filed: NOT_ADOPTED.
- **Seconds and memory per fit** on the T4.

## 8. How it runs

**Two sessions at once.** These are two kernels, since the account runs at most two GPU sessions, with two queues
each, as KG's were split:
- **Session 1, the null's seeds:**
  - GPU 0: zrckp s1, s1-hp, s2 and s2-hp;
  - GPU 1: zspk s1, s1-hp, s2 and s2-hp.
- **Session 2, the seed-0 fits and zg6:**
  - GPU 0: zg6-hp, zrckp-s0 and zrckp-s0-hp;
  - GPU 1: zg6, zspk-s0 and zspk-s0-hp.

Where a fit runs decides nothing.

Each session is a new version of one of KG's two kernels, which keeps KG's pinned image. A new kernel would take
Kaggle's current image, and every fit would refuse it (section 3).

**Time and cost.** KG's fourteen fits took 6.6 T4-hours. WQK's should be close to that, inside the week's quota. It
costs nothing.

**Commands.** On the laptop, the build (see zwqk.py's header for each step's command), then:

    python outputs/mp_unified/zwqk.py ship                 # outputs/wqk/ship.json: the files the gate and the fits read
    python outputs/mp_unified/zwqk_session.py stub --out DIR --gpu0 N,N --gpu1 N,N --rows-cap 65536
    kaggle kernels output <kernel> -p <new empty folder>
    python outputs/mp_unified/zwqk.py take --from <folder>
    python outputs/mp_unified/zwqk.py compare-all
    python outputs/mp_unified/zwqk.py floors
    python outputs/mp_unified/zwqk.py verdict

In the kernel, `zwqk_session.py run` runs `zwqk.py overlay`, `zwqk.py gate` and `zwqk.py fit --name N --rows-cap 65536`.

ship.json is committed and pushed before the stub is made. The stub refuses a HEAD that is not main, that origin does
not hold, or that does not track WQK's code, ship.json, bases.json, check.json and this file.

**What is committed.**
- Committed: this file, the code, and WQK's small records. Those are bases.json, check/check.json, ship.json, the
  sessions' gate.json and overlay.json, the ten compares, the floors, the verdict and the beside files.
- Never committed: the arrays, looks, caches, chains, links, KT arrays and parts; the fits' folders and weights;
  session.json, the logs and the datasets' manifests.

## 9. What follows (fixed now)

After BASE_READY, each of these gets its own declared file before any number:
1. **G6r under WQ, on the T4.** docs/G6R_RELATION_MESSAGES.md is declared on the old splits and held. Under WQ, it needs
   U2n's typed links for webqsp's fit carve, and its encoder from WQK's zspk seed-0 fits.
2. **R2g under WQ**: R2t's next form for zrckpb.
3. **Pool attention under WQ.**

## 10. Self-tests, passed before this commit

`python outputs/mp_unified/zwqk.py --selftest`:
1. **The laptop half:** every WQ root and record is moved to outputs/wqk and restored after, also on a failure. It
   refuses `--host` and anything but the build's steps.
2. **The plan:** fourteen fits and ten compares, each fit's twin WQ's name. KG's device is read from KG's own sources
   (zkg.py and zu2d.py).
3. **Where:** fit, train and read refuse anything but a Kaggle T4 on KG's image, before any data is read. Read and fit
   refuse a rows cap other than 65536.
4. **The mixed KT input**, through zp1c3's own load_kt on toy carves:
   - U2d's declaration and cache root go to the ten carves' records only, and P1c3's to webqsp's fit carve. A record of
     either kind on the other's carve is refused, and so is a read on another cache root.
   - The input nests, and refuses a second root while it is on.
   - zg6 runs at WQ's splits and WQK's names, accepts the train's and the read's flags, and gets its own constants back
     afterwards.
5. **The overlay**, on a toy base build:
   - every file is linked, and the record is in zwq's form;
   - a folder that is not whole is refused;
   - a file already there that is not a link to the base build's is refused.
6. **Ship:** exactly the files the kernel reads, with no temporary files, logs, gate or overlay. It is refused for a
   gate problem the overlay does not settle, a check that is not REPRODUCED, a missing piece, and a U2d record that
   does not hold.
7. **Take, compares, floors and verdict**, on fake fits:
   - BASE_READY on the full set;
   - INCOMPLETE for a missing compare, an unread fit, a fit on another device, a fit that names another gate, a gate
     that failed, a read without `--rows-cap 65536`, and floors that are not their compares';
   - take never copies models.pt and leaves an unread fit. It refuses a file with other bytes, a folder that is not
     WQK's, and an output with no gate.
8. Nothing is written under WQK's real folders.

`python outputs/mp_unified/zwqk_session.py --selftest` runs on a toy repository with KG's nine toy datasets and two WQK
groups:
1. find, plan, verify and link put every file at its repository path. A file git holds with the same bytes stays.
2. **Refusals:**
   - a WQK path outside ship.json, or one that climbs out of outputs/wqk;
   - a ship.json file in no group, a file in two groups, and a group's file that is not ship.json's bytes;
   - an empty group, a group attached twice, and no WQK group;
   - a changed byte.
3. **The run:**
   - the overlay, then the gate, then the fits, each in its own process, every fit with `--rows-cap 65536`;
   - an overlay or a gate that fails stops the session before any fit;
   - a clone that differs from its commit, a name that is not WQK's, a fit queued twice and another rows cap are
     refused before anything runs.
4. **The stub:** its code compiles and runs WQK's session with its arguments. It refuses a name that is not WQK's,
   another rows cap, and a HEAD that does not track WQK's code, ship.json, its records and this file.

The text encoder stays frozen. Test splits are never read. There is no LLM, no topic-entity help and no new graph.
webqsp trains under WQ's ruling only.

## 11. Results

None yet.
