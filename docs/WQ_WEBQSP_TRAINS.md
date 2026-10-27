# WQ: webqsp trains, metaqa is read zero-shot

Declared 27 October 2026, before any WQ build, fit or number. This file declares a new protocol, its build, its base
set and its seed null. The old protocol is unchanged: its splits (L-musique, L-hotpotqa), fits, floors and verdicts
stand for those splits and are not rewritten. A WQ verdict is a WQ verdict only.

## 1. Why

- **The ruling.** The user, 27 October: "lets switch up we train on webqsp and zeroshot metaqa, because i think webqsp
  gives a good starting point, similarly i think we can zeroshot the other wiki graphs etc?" This reverses "webqsp never
  trains" for this protocol only. Every old stage keeps its declared rule.
- **What it changes.** The old splits train on metaqa's fit carve and read webqsp zero-shot. webqsp has the larger
  relation set of the two Freebase graphs: metaqa's relations are film relations, which webqsp's also cover. So WQ
  learns typed relations from webqsp and reads metaqa zero-shot. That read is near-domain (the same Freebase film
  relations), and any claim made on it says so.
- **The wiki side** stays leave-one-out, as in the old protocol: each split reads one wiki graph zero-shot (musique or
  hotpotqa), and squad and 2wiki always train.

## 2. The protocol

- **Two splits.**

  | split | trains on (carve) | zero-shot |
  |---|---|---|
  | WQ-musique | webqsp (fit), squad (fit), hotpotqa (fit), 2wiki (fit) | musique, metaqa |
  | WQ-hotpotqa | webqsp (fit), squad (fit), musique (s1fit), 2wiki (fit) | hotpotqa, metaqa |

  The basis is 2wiki's for both (lean_gpu.basis_of, unchanged). metaqa's fit carve trains in neither split and is not
  built into WQ's roots.
- **webqsp's training carve** is M3B's fit carve of webqsp (m3b_compile.carves_for, unchanged). Its source is
  sorted(train)[1::2], 1,549 questions. The fit carve holds 1,239 of them (M3B's select carve holds the rest) and 1,202
  after the 37 with no gold in the graph are excluded. Its ids hash to
  `fecf94bbfdff25925abd771421d4ad7276889d81b8449891783506bd087bdb90` (look_x_six's 4 October record of this carve, which
  checked M3B's rule against stage 2's filed carves). The build must give exactly these 1,202 questions or it stops.
- **The reads** are the six s1eval carves at p@swa, unchanged. webqsp's s1eval carve is M3B's eval population,
  sorted(train)[::2], 1,503 questions scored. It shares no question with the training carve. M3B's select carve of
  webqsp is held: the base set never reads it, and it is the selection carve a later WQ stage may name.
- **Unchanged:** the pools (P1c2's: P1e's KIND_T at each question's frozen size), the inputs (P1c2's cache, chains and
  links, and P1c3's KT input), the models, settings, seeds and batches, the device, the encoder (frozen). No test split
  is read. No LLM, no topic-entity help and no per-dataset rule enters.

## 3. The build

All of it is written under `outputs/wq`; nothing outside that folder is written. The committed builders run unchanged.
A new driver, `outputs/mp_unified/zwq.py`, runs them in-process with WQ's roots and webqsp's fit carve in place of
P1c2's roots and carve lists, and puts every patched name back afterwards, also on a failure (zp1c2.on_p1c2's form).

- **W0. Step 1's look** of webqsp's fit carve: `look_x_six.py --dataset webqsp --carve fit --host --full --shard i/3
  --out outputs/wq/look/webqsp/fit`, unchanged. The 4 October look of this carve was not `--full`, which the cache
  needs.
- **W1. The bases.** Step 1's basis_2wiki and basis_hotpotqa, each checked against its step-1 record's sha256 and
  linked into `outputs/wq/s1cache` and `outputs/wq/p1c2/cache` (a hard link, else a copy, checked again after).
- **W2. Step 1's cache** of the carve: `lean_cache.py --dataset webqsp --carve fit --part i/2 --host --look-root
  outputs/wq/look --out-root outputs/wq/s1cache`, unchanged.
- **W3. Step 4d's union pools**: step4d_union_pools.pools_stage on webqsp, with four patches:
  - step 4c's carves_of gives webqsp's fit carve only;
  - the carves record gains webqsp's fit entry (name fit, 1,202 questions, the sha256 above), which the population
    must hash to;
  - step 4d's POOLS and MANIFEST are `outputs/wq/step4d/pools`;
  - step 4c's STEP1_CACHE is `outputs/wq/s1cache`.

  step 4c's step1_rows returns nothing when no step-1 record exists. So zwq refuses the stage's output unless the
  manifest entry's `step1_cache_rows` is present and equal.
- **W4. P1c2's pools**: p1c2_pools.pools_stage on webqsp, with carves_for giving the fit carve, POOLS at
  `outputs/wq/p1c2/pools`, step 4d's POOLS and MANIFEST at W3's, and STEP1_CACHE at W2's. The same refusal on
  `step1_cache_rows`.
- **W5. P1c2's look**: look_p1c2.main with its POOLS at `outputs/wq/p1c2/pools` and LOOK at `outputs/wq/p1c2/look`;
  `--full`, three shards. zwq imports look_p1c2 before anything else in that process, so look_x_six sets the thread
  variables before numpy loads, as it does when run alone.
- **W6. P1c2's cache**: `lean_cache.py --dataset webqsp --carve fit --part i/2 --host --look-root outputs/wq/p1c2/look
  --out-root outputs/wq/p1c2/cache`, unchanged.
- **W7. Chains and links**: zp1c2.build with zp1c2's LOOK, CACHE, CH_RM, CH_ZRC and LINKS at
  `outputs/wq/p1c2/{look,cache,chains/rmatch,chains/zrc,links}`. rmatch's chains first, then zrc's (against rmatch's),
  and the links.
  - zrc keeps, per row and seed bucket, 64 chain entries, and on webqsp that cap binds (zrc.py's header). webqsp's fit
    carve is built under the same rule as its s1eval carve, so the MLP trains and reads on the same kind of entries.
  - zrc.py's identity with zrm held because metaqa was the only typed graph a fit trained on. It does not carry over to
    WQ, and nothing here uses it: WQ's fits are trained, not forked.
- **W8. The KT input**: p1c3_kt.feats_stage on webqsp's fit carve, with p1c3_kt's KT at `outputs/wq/p1c3/kt` and
  p1c2_pools' POOLS and CACHE at WQ's. Its record names the cache root `outputs/wq/p1c2/cache`.
- **W9. The overlay.** The other ten carves of WQ's screen are linked from P1c2's and P1c3's roots into WQ's: squad
  fit, musique s1fit, hotpotqa fit, 2wiki fit, and the six s1eval carves.
  - Each is a hard link, else a copy, of every file in the carve's folder: cache parts (`outputs/p1c2/cache`), links
    (`outputs/p1c2/links`) and KT (`outputs/p1c3/kt`), and rmatch's and zrc's chains where P1c2 has them (metaqa's and
    webqsp's s1eval carves).
  - Every file's sha256 is checked against its source, and every record must be its source byte for byte.
  - `outputs/wq/overlay.json` lists each folder, its file count and its records' sha256s. Nothing in P1c2's or P1c3's
    roots is written.
- **W10. The gate**, `outputs/wq/gate.json`, is PASS when all of these hold, and FAIL otherwise:
  - W1 to W9's records hold;
  - each of the eleven carves has the folders its source kind has (cache, links and KT for all; chains for webqsp's two
    carves and metaqa's s1eval);
  - webqsp's fit carve has 1,202 questions in its pools, look, cache and KT records, and its KT record's cache parts are
    its cache parts' sha256s;
  - the reproduction check (section 4) is REPRODUCED.

## 4. The reproduction check

Before the gate can pass, W3's and W4's entry points, with zwq's patches, run on webqsp's s1eval carve into
`outputs/wq/check`. They must give step 4d's and P1c2's filed webqsp s1eval arrays, array for array: every key, dtype,
shape and byte. The files' sha256s are not compared, because np.savez_compressed stamps the zip entries with the time.

- For this run only, STEP1_CACHE is step 1's own `outputs/step1/cache`, whose webqsp s1eval records are on the host.
- step 4d checks an s1eval carve against step 4b's filed arrays, and P1c2's pools stage checks it against P1e's
  positions, as when they were filed.
- step 4b's `outputs/step4b/eval/webqsp.npz` and its `.json` are not on the host. They are pushed from the laptop for
  this check (npz sha256 `053cac56c16b6f7509114ebfcde60ab9b128e45e1764accc487ef3096115de1b`).

The record is `outputs/wq/check/check.json`: REPRODUCED or NOT_REPRODUCED. NOT_REPRODUCED stops WQ before any fit, and
the cause is fixed in an amendment here first.

## 5. The base set

- **The arms** are the lean pair, unchanged: zrckp, the MLP (zp1c3p.py: zrc's model, the KT head and the pointwise
  head), and zspk, the GNN (zp1c3.py: zsp's model and the KT head). They keep their code, settings, variants, config,
  hidden size and device (cuda). Only the split, the seed and the roots change.
- **Twelve fits**, seeds 0, 1 and 2 on both splits:
  - `scr-wq-zrckp-s<k>` and `scr-wq-zspk-s<k>` on WQ-musique;
  - `scr-wq-zrckp-s<k>-hp` and `scr-wq-zspk-s<k>-hp` on WQ-hotpotqa.

  The seed-0 fits are WQ's card fits. All twelve run on the host's GPU: one placement for the base set and for every
  WQ screen decided against it.
- **How they train.** `zwq.py train` patches lean_gpu's FIT_CARVE (webqsp's fit carve added) and FITS (the two WQ
  splits added); TRAIN_ORDER is read only at import and is left alone. It patches zp1c3's and zp1c3p's SPLITS, zp1c2's
  roots (CACHE, CH_RM,
  CH_ZRC, LINKS, LOOK), the KT root of zp1c3's two carve classes, and zp1c3's kt_records. It then runs zp1c3's train
  (zspk) or zp1c3p's train (zrckp) unchanged. For zrckp, zp1c3p's fit name is swapped for WQ's for that run, as
  zn2.py does. Every patched name is put back after, also on a failure.
- **The KT root check.** zp1c3.load_kt checks that a KT record was written on the cache root the carve reads.
  - The ten linked KT records name `outputs/p1c2/cache`, so zwq hands load_kt that root for them. load_kt still checks
    their cache parts by sha256: those are the linked files.
  - webqsp's fit record names `outputs/wq/p1c2/cache`, and is checked against it.

  Nothing else in load_kt changes.
- **Gates.** P1c2's gate (PASS) is still checked, by the trainers, unchanged. zwq also requires WQ's gate to be PASS
  before any train or read.
- **Stamps.** The trainers' own stamps keep their committed text: zp1c3's and zp1c3p's say "outputs/p1c2/pools". Their
  KT, rmatch, zrc and link records are read from WQ's roots. zwq adds a `wq` block that names this file, the split, the
  seed, WQ's roots, and the gate's and overlay's sha256s.
- **Reads.** Each fit is read by its trainer's read, under the same patches, on WQ's six s1eval carves. These are the
  files P1c2's fits read, linked.

## 6. The floors

- **The null.** For each arm and split, seeds 1 and 2 are compared with seed 0 by lean_screen's compare: p@swa on the
  six s1eval carves, `outputs/screen/scr-wq-<arm>-s<k>[-hp].json` (k = 1, 2).
- **The rule**, unchanged (docs/SCREENS.md, section 2; screen_recall.floor_of): floor(r) = max(0.0075, 2 x
  sqrt((D1(r)^2 + D2(r)^2) / 2)), with D_s(r) seed s's R@5 minus seed 0's on read r.
- **The records:** `outputs/wq/floors-zrckp.json` and `floors-zspk.json`, each with an `.md`. Each read shows seed 0's
  R@5, D1, D2, the spread and the floor.
- **Their use.** A later WQ screen compares its arm's seed-0 fit with WQ's card fit of the same split and arm, and
  calls each read with these floors (screen_recall's GAIN, WITHIN and LOSS, unchanged). PROMISING needs at least one
  GAIN and no LOSS.

## 7. BASE_READY

`outputs/wq/summary.json` (and `.md`) is BASE_READY when all of these hold:
- the gate is PASS, with the reproduction check REPRODUCED;
- all twelve fits are trained and read, each with its trainer's record, its stamp and WQ's `wq` block;
- the eight null compares exist, each on the six reads and its split's training sets;
- both floors records are OK.

Otherwise it is NOT_READY and names what is missing. No WQ screen runs before BASE_READY.

## 8. Beside (deciding nothing)

Each WQ seed-0 fit is compared with the old protocol's card fit of the matching split and arm, by lean_screen's
compare on the same six s1eval carves:
- zrckp: scr-p1c3p-zrckp (L-musique) and scr-p1c3p-zrckp-hp (L-hotpotqa);
- zspk: scr-p1c3-zspk and scr-p1c3-zspk-hp.

The output is `outputs/wq/beside/scr-wq-<arm>-s0[-hp].json`. It reports what the switch does per read. webqsp is
expected to rise, since it is now trained, and metaqa to fall, since it is now zero-shot. It decides nothing, and no old
verdict changes.

## 9. What comes next (fixed now)

After BASE_READY, each of these gets its own declared file:
1. **The pool-softmax input for the MLP** (the user's 27 October question). Each raw score column of a pool gets one
   parameter-free input: its log-softmax gap below the pool's soft best, at a fixed temperature. The MLP stays MP-free:
   this is a fixed normalisation computed before any row is scored, not a learned exchange between rows. One seed-0 fit
   per WQ split, called under WQ's floors.
2. **WQ on Kaggle's T4**: a KG-style base set under WQ, so that G6r, R2g and pool attention are decided under WQ.
3. **U2s's pool pick, chosen again under WQ**: metaqa leaves the training sets, and webqsp's select carve is the
   selection carve.

If the check is NOT_REPRODUCED or the gate is FAIL, the failing step is reported here and fixed in an amendment before
any fit.

## 10. How it runs

- **Code:** `outputs/mp_unified/zwq.py`, new, with a self-test that passes on the laptop before it is queued. It imports
  the committed modules unchanged. Each command imports only what it needs: the look command imports look_p1c2 first.
- **Commands:**

  | command | step |
  |---|---|
  | `copy-bases` | W1 |
  | `s4d [--check]` | W3, or section 4's step 4d run |
  | `pools [--check]` | W4, or section 4's P1c2 pools run |
  | `check` | section 4's comparison |
  | `look --shard i/3` | W5 |
  | `build --what rmatch\|zrc\|links` | W7 |
  | `kt` | W8 |
  | `overlay` | W9 |
  | `gate` | W10 |
  | `train --split --name --arm --seed` | the base set's fits |
  | `read --name` | their reads |
  | `compare` | lean_screen's compare, unchanged |
  | `floors --arm` | section 6's records |
  | `beside` | section 8's compares |
  | `summary` | section 7's record |

  W0, W2 and W6 are the committed scripts, run as above.
- **The pushes:** zwq.py, this file, and step 4b's webqsp eval file and record (section 4). These are the only inputs
  the host does not already hold.
- **Host items**, `wq-*`, appended to the feeder's items after N2's, so N2's GPU items keep their priority. Sizes are
  P1c2's webqsp items' sizes:
  - W0 and W5 shards: 1.4 CPUs and 8.2 GB each;
  - W2 and W6 parts: 1.5 CPUs and 7.5 GB each;
  - W3, W4, the two check runs and W8: 3 threads, 3 CPUs and 4.5 GB each;
  - rmatch's and zrc's builds: 1 CPU and 8 GB each; links: 1 CPU and 6 GB;
  - copy-bases, overlay, check, gate, floors, beside and summary: 1 CPU and 2 GB or less.

  The trains are sized as N2's L-hotpotqa items: 1.1 CPUs, 7 GB, `--frac 0.27` and GPU share 0.29. Reads take
  1.1 CPUs, 6 GB, `--frac 0.27` and share 0.29. Compares take 1 CPU and 4 GB. Every train waits on the gate.
- **Records.** Small records under `outputs/wq` and the compare files are committed (`git add -f`). Arrays, parts,
  models and the fits' own records are never committed.

## 11. Results

None yet. Filed here after BASE_READY: the gate and the check, the twelve fits' reads, both floors records, and the
beside compares.
