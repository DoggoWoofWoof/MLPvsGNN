# WQG: the GNN seed nulls under WQ on Kaggle's T4 (zg6 and zg6r)

Declared on 30 October 2026, before any WQG fit. The code is `outputs/mp_unified/zwqg.py` (the fits, the take, the
compares, the floors and the verdict) and `outputs/mp_unified/zwqg_session.py` (the Kaggle session and the two stubs).
They import WQK's code (`zwqk.py`, `zwq.py`, `zwqk_session.py`, `zkg_session.py`), G6RW's (`zg6rw.py`,
`zg6rw_session.py`) and U2n's (`zu2n_session.py`) unchanged, and patch them in process. `zg6.py`, `zg6r.py` and the
code under them are imported by the fit and are not edited. No committed file is edited. Nothing here is a result.

## 1. Why this, now

- WQK fitted the MLP arms zrckp and zspk at the null seeds 1 and 2 on both splits and built `outputs/wqk/floors-zrckp`
  and `outputs/wqk/floors-zspk`: per read, max(0.0075, 2 x the root mean square of the two seed-only differences). It
  fitted zg6 once per split (seed 0), and G6RW fitted zg6r once per split (seed 0). Neither GNN has a seed null.
- So every GNN call under WQ so far used zspk's floors as a stand-in: G6 (zg6 against zspk), G6RW (zg6r against zspk and
  against zg6), and R2EG (zg6e against zg6). A GNN's seed spread is not the MLP's. The stand-in was stated each time; WQG
  replaces it with a measured floor.
- Both T4 kernels are free (R2WS and R2EG are filed NOT_ADOPTED; R2's lines close), and a seed null is the same work for
  every later GNN form under WQ: the refits on PKO's pools, and any next form of zg6 or zg6r.
- **Naming.** WQG is the GNN seed nulls: arm A is zg6, arm B is zg6r. What an earlier plan called WQG2 (zg6r at seeds 1
  and 2) is arm B here, not a later file.

## 2. What changes

Nothing in a model. Each fit is the seed-0 fit of its arm and split, trained again with `--seed 1` or `--seed 2`, as
WQK did for zrckp and zspk. The seed is the only thing that changes.

- **Arm A, zg6.** zg6's `train` refuses `--seed` (its `FIXED_TRAIN` holds it: G6 keeps zspk's card, seed 0). WQG calls
  `train` with `--seed <k>` after patching, in process and for that one call, `FIXED_TRAIN` (without `--seed`; every other
  fixed flag stays, a changed hidden size or config still stops) and `FIT["zg6"][split]` (the fit's name, which `train`
  checks against its table). `zg6.py` is unchanged on disk and its sha256 must equal the seed-0 fit's. The self-test shows
  that seeds 0, 1 and 2 give different weights and the same seed the same ones.
- **Arm B, zg6r.** `zg6rw.train` (G6RW's) already appends `--seed <SEED>` from its module constant. WQG calls it unchanged
  with `SEED`, `SPLIT_OF` and `import_g6r` patched around the call, so `--seed <k>` reaches zrm's train and zg6r's net (its
  own offset 3801 on the seed). `zg6rw.read` is called the same way. The record carries G6RW's own `g6rw` block (the KT,
  typed, rmatch, zrc and zlink records, the encoder files' shas, the head) under WQG's names.
- **Reads.** Each fit is read as the seed-0 fit was: p@swa on the six s1eval carves, in the arm's read, on the device it was
  fitted on, with the arm's read cap (zg6 `--rows-cap 65536`, zg6r `--rows-cap 32768`: whole questions per batch, so the
  scores do not change with the cap).
- **Floors.** `outputs/wqg/floors-zg6` and `outputs/wqg/floors-zg6r` are zwq's floor builder (`zwq.floors`), run at WQG's
  names with the arm's seed-0 fit as seed 0.

## 3. The arms and the fits

| arm | seed-0 fit (the base) | WQG's fits | trained by | read cap |
|---|---|---|---|---|
| zg6 (A) | `scr-wqk-zg6-s0` (WQ-musique), `scr-wqk-zg6-s0-hp` (WQ-hotpotqa) | `scr-wqg-zg6-s1`, `-s2`, `-s1-hp`, `-s2-hp` | `zg6.train` | 65,536 |
| zg6r (B) | `scr-g6rw-zg6r`, `scr-g6rw-zg6r-hp` | `scr-wqg-zg6r-s1`, `-s2`, `-s1-hp`, `-s2-hp` | `zg6rw.train` | 32,768 |

- No suffix is the WQ-musique split (webqsp, squad, hotpotqa, 2wiki train; musique held out); `-hp` is WQ-hotpotqa
  (webqsp, squad, musique, 2wiki train; hotpotqa held out). The seeds are WQK's null seeds, 1 and 2.
- Each fit differs from the arm's seed-0 fit of its split in the seed alone. `zwqg.py verdict` checks the card (arm,
  split, train set, basis, variants, config, hidden size, cache root), every code sha256 in the record except the stage
  files' own (zwq, zwqk, zwqg, zg6rw), train.json's fields, the records of the inputs (for zg6: KT, rmatch, zrc, zlink and
  the g6 block; for zg6r: those, the typed records, typed.json, the encoder files and the head) and the trainer's code
  (zg6: zp1c3, zp1c3p, zp1c2, zg6, zg2; zg6r: those and zg6r, zu2n, zg6b2, zg6b, zkg, zu2d).
- Training and reads run under `zwq.on_wq` with WQK's names, roots, mixed KT input and cache root. The wq block carries
  WQK's declaration and the fit's arm and seed.

## 4. Fits and device

- Two kernels, one per arm, built from one commit with the two stubs of section 8. Each kernel has two T4s and four fits
  (two seeds on two splits), and each GPU runs its two fits one after the other, never together:

      GPU 0: s1, then s2-hp          GPU 1: s1-hp, then s2

  The hp fit is the longer, so each GPU holds one short and one long fit and the two queues end together.
- Memory: zg6 peaks at 9.13 GB of a T4's 15 (seed-0 fit, train.json); zg6r at 6.01 GB (train 2.05 / 6.01, read 4.59 /
  6.01 GB). One fit at a time per GPU leaves room. Host RAM about 4.6 to 4.7 GB.
- The six fits of an arm that `verdict` compares (its four and its two seed-0 fits) must have run on a Tesla T4 on KG's
  image tag; a read that ran on another device than its fit is refused (as in WQK).
- No fit is committed. The per-fit folders (outputs/screen/fits/scr-wqg-*), the take's session copies
  (`outputs/wqg/sessions`), the logs and session.json stay local.

## 5. Compares, the null and the floors

- **Compares.** `zwqg.py compare-all --arm <arm>` runs zwq's compare of each WQG fit with the arm's seed-0 fit of its
  split alone, named as the fit (`outputs/screen/scr-wqg-<arm>-s<k>[-hp].json`), p@swa on s1eval, six datasets, stamped
  with WQG's record and the device. The base column of a compare must be the seed-0 fit's own filed read: WQK's
  `scr-wqk-zg6-vs-zspk[-hp]` new column for zg6, G6RW's `scr-g6rw-zg6r-vs-zspk[-hp]` new column for zg6r.
- **Floors.** `zwqg.py floors --arm <arm>`: for each of the twelve reads (two splits x six datasets), D1 and D2 are the
  seed-1 and seed-2 differences in R@5 from the seed-0 fit; the spread is the root mean square of D1 and D2; the floor is
  max(0.0075, 2 x spread). Written to `outputs/wqg/floors-<arm>.json` and `.md`.
- **Rule (per arm).** FLOORS_READY when: WQK's verdict is BASE_READY; the arm's four fits are trained and stamped with
  their seed (screen.json, train.json and the wq block; for zg6r G6RW's block too), read on the six s1eval carves with the
  arm's read cap on one device with the arm's two seed-0 fits, a T4 on KG's image; they differ from the seed-0 fit in the
  seed alone (section 3); every fit and read names a gate that is a filed WQK PASS gate of a WQG session (taken to
  `outputs/wqg/sessions/<gate sha12>`, on the reproduced check and its session's overlay); the four compares are WQG's,
  each against the arm's seed-0 fit alone with that fit's R@5 equal to the seed-0 fit's own filed read; and
  `outputs/wqg/floors-<arm>` is WQG's OK record of those compares' floors (recomputed). Also required: WQK's
  `outputs/wqk/floors-zspk.json` is an OK record (the tables below are set beside it). INCOMPLETE otherwise, naming what
  is missing.
- The verdict is FLOORS_READY or INCOMPLETE and nothing else. It is a null, not a model screen: no fit is adopted or
  rejected by it. Files: `outputs/wqg/null-verdict-<arm>.json` and `.md` (exit 0 on FLOORS_READY, 2 on INCOMPLETE).

## 6. Beside: the re-calls under the real floors (deciding nothing)

With an arm's FLOORS_READY, the verdict also re-calls the filed GNN comparisons that used zspk's stand-in, under the real
floors, and sets each beside the call under the stand-in. These are reported next to the filed verdicts and change nothing.

| pair | arm file | the filed comparison | the pair's floor is built from |
|---|---|---|---|
| zg6e against zg6 | zg6 | R2EG: `scr-r2eg-zg6-vs-zg6[-hp]`, filed NOT_ADOPTED (1 GAIN, 2 LOSS) | zg6's spread |
| zg6r against zspk | zg6r | G6RW's deciding comparison, filed ADOPT_FOR_FULL_RUN (e0e2c51: 5 GAIN, 0 LOSS) | zg6r's and zspk's spreads |
| zg6r against zg6 | zg6r | G6RW's comparison beside, filed in its recall | zg6r's and zg6's spreads |

- **Pair floor.** For a comparison of two arms the floor is max(0.0075, 2 x the root mean square of the two arms'
  spreads) per read; for one arm's form against that arm (zg6e against zg6) it is the arm's own floor. Each arm's spread is
  the root mean square of its two seed-only differences, so a difference between two different arms carries each arm's
  variance once. zspk's spread is WQK's (`outputs/wqk/floors-zspk.json`); zg6's and zg6r's are WQG's. This construction is
  WQG's choice, stated here before any number.
- **Per read** the verdict prints the filed call, the floor under zspk's stand-in and the call under it (recomputed by
  `screen_recall.recall`, which equals the filed call when the filed recall agrees), the real floor and the call under it,
  and marks a changed call. Per pair it prints the rule's reading under each floor (ADOPT_FOR_FULL_RUN when at least one
  GAIN and no LOSS, else NOT_ADOPTED), whether the reading changed, and whether the stand-in calls agree with the form's
  filed recall file.
- A pair is "not available" (with the reason) when the form's compare is not in `outputs/screen` or was made on another
  device, a floors file it needs is missing or not OK, or a compare's column is not the floors' seed-0 R@5. R2EG's compares
  have to be on the laptop for its pair to appear.
- Also printed: the reads where the arm's floor is wider than zspk's (and narrower), so the effect of the stand-in is read
  at a glance.
- **R2EG (NOT_ADOPTED, 1b38abc) and G6RW (ADOPT_FOR_FULL_RUN, e0e2c51) keep their filed verdicts.** The re-calls are
  reported next to them and change nothing, unless a pair's reading flips (zg6e ADOPT_FOR_FULL_RUN, or zg6r NOT_ADOPTED
  against zspk or against zg6). Then the next step is a declared amendment of that form's file, not an edit; the verdict
  names it ("a reading flips: ..., the next step is a declared amendment ... not an edit"). If every reading holds it says
  so ("no reading flips: the filed verdicts stand").

## 7. Outcomes and risks

- **FLOORS_READY** for an arm: the arm is called under `outputs/wqg/floors-<arm>` from here. A later form's declaration
  cites it in place of zspk's stand-in: the refits on PKO's pools, and any next form of zg6 or zg6r under WQ.
- **INCOMPLETE:** what is missing is supplied (a fit, a compare, a session's gate, the floors); nothing else changes.
- **Two fits per seed per split is two floors per read.** A floor from two differences is itself noisy, as WQK's are; the
  0.0075 minimum guards the narrow side.
- **Generator-seed overlap (zg6).** zg6's init seed is seed + 3601 and its generator seed is seed + 3601 + 1, so seed 1's
  generator seed equals seed 2's init seed (the self-test notes it). The streams differ in what they drive; it is not a
  reuse of one stream, but the two seeds are not as independent as 0 and 5 would be. Noted, not changed (no edit to zg6).
- **No `rel_record` for zg6r.** G6RW's diagnostic record of relation coverage is not rebuilt for the WQG fits: a diagnostic,
  not an input to a floor.
- **Cross-arm pair floors are WQG's construction** (section 6); a later file may fix another and say why.
- **Timing is not measured at seeds 1 and 2;** the estimates of section 8 come from the seed-0 fits.

## 8. How it runs

Two kernels from one commit (the repository's main, pushed; WQK's verdict BASE_READY; zg6r also G6RW's ship.json, typed.json
and typed records tracked). On the laptop, two stubs:

    python outputs/mp_unified/zwqg_session.py stub --arm zg6  --out kaggle/wqg_a --gpu0 scr-wqg-zg6-s1,scr-wqg-zg6-s2-hp \
        --gpu1 scr-wqg-zg6-s1-hp,scr-wqg-zg6-s2 --rows-cap 65536
    python outputs/mp_unified/zwqg_session.py stub --arm zg6r --out kaggle/wqg_b --gpu0 scr-wqg-zg6r-s1,scr-wqg-zg6r-s2-hp \
        --gpu1 scr-wqg-zg6r-s1-hp,scr-wqg-zg6r-s2 --rows-cap 32768

Each writes `<out>/kernel/wqg_<arm>_stub.py` (the kernel's code file) and `<out>/code/mpr-<sha12>.bundle` (main's history).
The stub clones the bundle at the commit and runs `zwqg_session.py run --arm <arm> ...`: it checks the names (one arm, WQG's
fits, each once), the rows cap, WQK's verdict and (zg6r) typed.json; finds, checks and links the datasets (zg6: KG's nine and
WQK's groups; zg6r: those and U2n's typed links and G6RW's encoders and typed arrays, exactly G6RW's ship.json); runs
`zwqk.py overlay` and `zwqk.py gate` (a failure of either stops the session before any fit); then runs `zwqg.py fit --name N
--rows-cap C` for each fit, one queue per T4. A fit whose read WQG already stamped is kept; a trained, stamped fit is only
read.

Estimated seconds, per kernel (a GPU's two fits one after the other; setup, verify, link, overlay and gate about 290 s):

- zg6: seed-0 fits took 2,292 s (WQ-musique) and 4,318 s (WQ-hotpotqa) with the read, so each GPU about 6,610 s, with
  setup about 6,900 s (about 1.9 h).
- zg6r: seed-0 fits took 2,908 s and 4,920 s (train plus read, `g6rw_timing.json`), so each GPU about 7,830 s, with setup
  about 8,100 s (2.3 h); 1.2 to 1.35 times zg6's figures puts it at 8,200 to 9,200 s (2.3 to 2.6 h). Both stay inside the
  default `--start-before 8` hours, which only bounds when a fit may start.

On the laptop, after each kernel's output is fetched into a new folder:

    python outputs/mp_unified/zwqg.py take --from DIR              # the fits' records and reads, never their weights
    python outputs/mp_unified/zwqg.py compare-all --arm zg6        # and --arm zg6r
    python outputs/mp_unified/zwqg.py floors --arm zg6             # and --arm zg6r
    python outputs/mp_unified/zwqg.py verdict --arm zg6            # and --arm zg6r

Never committed: the per-fit folders under `outputs/screen/fits`, `outputs/wqg/sessions`, the logs and session.json. Small
records (the compares, the floors, the verdicts and the pair files under `outputs/wqg`) are added with `git add -f`.

## 9. Self-tests

    PYTHONIOENCODING=utf-8 python outputs/mp_unified/zwqg.py --selftest
    PYTHONIOENCODING=utf-8 python outputs/mp_unified/zwqg_session.py --selftest

`zwqg.py`: the names and tables (zwq's names while WQG's checks run, restored after); the seed routes (zg6's patched
`FIXED_TRAIN` and `FIT` reach `train` past its flag check, lift `--seed` alone, restore on a failure; seeds 0, 1 and 2 give
different nets; zg6rw's `train` appends `--seed` from `SEED`, and WQG's patch sets and restores `SEED`, `SPLIT_OF` and
`import_g6r`); the guards (device, read cap per arm, WQK's verdict); the stamps; the kernel's fit order (train then read, a
stamped read kept, a trained fit only read); the take; the compares (kept when stamped from the same fits); the floors
(values checked against the formula); the verdict on a toy of WQK's base set, G6RW's seed-0 fits and a WQG session per arm,
with its INCOMPLETE matrix (every missing or wrong piece named); the beside pairs (floors recomputed by hand: a GAIN that a
real floor turns to WITHIN, a reading that flips and names the amendment, a pair that is not available, a filed recall that
agrees or disagrees); and that nothing is written under the real `outputs/wqg`, `outputs/screen/fits/scr-wqg-*` or
`outputs/screen/scr-wqg-*`. `zwqg_session.py`: for each arm, the run in each layout (the overlay, the gate, the four fits
with the arm's cap, session.json), an overlay or gate that fails stopping the session before any fit, the refusals before
anything runs, the stub's refusals and its two files.

## 10. Results

**WQG: FLOORS_READY for both arms** (`outputs/wqg/null-verdict-zg6.*` and `null-verdict-zg6r.*`; exit 0). Called at 05:02
on 30 October by the amended guard (section 11, 3e84689). The first calls, at 04:57, were INCOMPLETE on `train.json
variants` for all eight fits (section 11); they are the first of two and nothing but the guard changed between them. The
verdict is a null and adopts or rejects no model.

**The sessions.** Both on 2x Tesla T4, one device record (image kaggle-gpu:py3.13.15, torch 2.11.0+cu128, numpy 2.1.3),
every fit exit 0.
- Arm A (zg6, rows cap 65,536): mpr-kg v12, gate a13c1aa6ff6b, session 6,223.2 s (overlay 94.7 s, gate 79.2 s). Fits
  scr-wqg-zg6-s1 2,027.9 s, -s1-hp 3,928.3 s, -s2 2,005.8 s, -s2-hp 3,941.3 s.
- Arm B (zg6r, rows cap 32,768): mpr-kg-g6b2 v10, gate 7121d49c8e4d, session 8,565.4 s (overlay 368.2 s, gate 80.9 s).
  Fits scr-wqg-zg6r-s1 2,786.1 s, -s1-hp 5,157.1 s, -s2 2,559.2 s, -s2-hp 4,656.7 s. Version 9 of mpr-kg-g6b2 had stopped
  within 30 s ("the group u2n is not attached"): the stub did not attach G6RW's dataset, as `kg_push.py g6rw` does. It was
  fixed in the launcher (outside the repo's stage files) and no fit ran in v9.
- Eight compares (`compare-all`, 0 run, 8 filed): `outputs/screen/scr-wqg-<arm>-s<k>[-hp]`.

**The floors.** D1 and D2 are the seed-1 and seed-2 R@5 minus the seed-0 fit's; floor = max(0.0075, 2 x the root mean
square of D1 and D2). Six of twelve floors are above 0.0075 for each arm; four of each arm's are wider than zspk's.

| split | dataset | zg6 seed 0 | D1 | D2 | zg6 floor | zg6r seed 0 | D1 | D2 | zg6r floor | zspk floor (WQK) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WQ-musique | metaqa | 0.4393 | -0.0154 | -0.0428 | 0.0643 | 0.4105 | +0.0139 | -0.0020 | 0.0199 | 0.0439 |
| WQ-musique | squad | 0.9097 | +0.0016 | +0.0027 | 0.0075 | 0.9105 | +0.0003 | +0.0023 | 0.0075 | 0.0075 |
| WQ-musique | musique | 0.5018 | +0.0150 | +0.0093 | 0.0250 | 0.4106 | +0.0502 | +0.0512 | 0.1014 | 0.0545 |
| WQ-musique | hotpotqa | 0.9109 | +0.0015 | +0.0016 | 0.0075 | 0.9134 | -0.0024 | -0.0016 | 0.0075 | 0.0075 |
| WQ-musique | 2wiki | 0.9390 | +0.0012 | +0.0024 | 0.0075 | 0.9369 | +0.0038 | +0.0014 | 0.0075 | 0.0075 |
| WQ-musique | webqsp | 0.6423 | -0.0012 | -0.0093 | 0.0133 | 0.6501 | -0.0074 | -0.0026 | 0.0111 | 0.0174 |
| WQ-hotpotqa | metaqa | 0.4167 | -0.0037 | +0.0025 | 0.0075 | 0.3834 | +0.0521 | +0.0322 | 0.0866 | 0.0308 |
| WQ-hotpotqa | squad | 0.9114 | -0.0004 | +0.0003 | 0.0075 | 0.9111 | +0.0019 | +0.0013 | 0.0075 | 0.0075 |
| WQ-hotpotqa | musique | 0.5961 | -0.0065 | -0.0017 | 0.0095 | 0.5941 | -0.0021 | +0.0037 | 0.0075 | 0.0090 |
| WQ-hotpotqa | hotpotqa | 0.8183 | +0.0078 | +0.0012 | 0.0112 | 0.8529 | -0.0041 | -0.0044 | 0.0085 | 0.0075 |
| WQ-hotpotqa | 2wiki | 0.9414 | -0.0017 | +0.0007 | 0.0075 | 0.9420 | -0.0002 | +0.0012 | 0.0075 | 0.0075 |
| WQ-hotpotqa | webqsp | 0.6421 | -0.0081 | -0.0110 | 0.0193 | 0.6536 | -0.0053 | -0.0135 | 0.0205 | 0.0089 |

What the table shows, read off it and not further:
- **zg6r's spread is large on two zero-shot reads and small on the rest.** WQ-musique musique: seeds 1 and 2 read 0.4608
  and 0.4618 against 0.4106 at seed 0 (floor 0.1014, zspk's 0.0545); WQ-hotpotqa metaqa: 0.4355 and 0.4156 against 0.3834
  (floor 0.0866, zspk's 0.0308). Seed 0 is the lowest of the three seeds on both. zg6's floors on those two reads are
  0.0250 and 0.0075.
- Every in-domain floor of both arms is 0.0075 to 0.0205.

**Beside, deciding nothing: the three re-calls under the real floors** (`outputs/wqg/wqg-*-pair-recall.*`). The floor of a
comparison of two arms with seeds is max(0.0075, 2 x the root mean square of their spreads); zg6e has seed 0 only, so a
comparison with it takes zg6's floor.

| pair | filed call (stand-in floors) | re-call (real floors) | the reads that moved |
|---|---|---|---|
| zg6e against zg6 (R2EG, 1b38abc) | NOT_ADOPTED: 1 GAIN, 2 LOSS | 0 GAIN, 1 LOSS | WQ-musique metaqa LOSS (-0.0443) -> WITHIN (floor 0.0643); WQ-hotpotqa hotpotqa GAIN (+0.0097) -> WITHIN (0.0112). WQ-hotpotqa metaqa stays LOSS (-0.0320, floor 0.0075) |
| zg6r against zspk (G6RW, e0e2c51) | ADOPT_FOR_FULL_RUN: 5 GAIN, 0 LOSS | 5 GAIN, 0 LOSS | WQ-hotpotqa metaqa GAIN (+0.0450) -> WITHIN (floor 0.0650); WQ-musique webqsp WITHIN -> GAIN (+0.0167, floor 0.0146). WQ-musique musique stays WITHIN (-0.0484, floor 0.0814) |
| zg6r against zg6 (G6RW's beside) | 1 GAIN, 2 LOSS | 1 GAIN, 1 LOSS | WQ-hotpotqa metaqa LOSS (-0.0333) -> WITHIN (floor 0.0614). WQ-musique musique stays LOSS (-0.0912, floor 0.0738); WQ-hotpotqa hotpotqa stays GAIN (+0.0346, floor 0.0099) |

No reading flips: R2EG stays NOT_ADOPTED (no GAIN and one LOSS under the real floors) and G6RW stays
ADOPT_FOR_FULL_RUN (no LOSS and five GAINs, with one read in the set swapped for another). Section 7's rule, that a flip
names a declared amendment, is not triggered. The "filed" column in the re-call tables (`outputs/wqg/*-recall.md`) is the
call the compare file itself carries (lean_screen's own, per `screen_recall.py`), not G6RW's or R2EG's adoption call: it
reads GAIN for WQ-musique metaqa of zg6r against zspk (+0.0121), which G6RW's table calls WITHIN against zspk's floor of
0.0439. The adoption calls are the ones in G6RW's and R2EG's own files, and the table above sets them beside the re-calls.

**What this fixes for what comes next.** From here a declaration cites `outputs/wqg/floors-zg6` or `floors-zg6r` in place
of zspk's stand-in. G6RW's weak read (WQ-musique zero-shot musique, 0.4106) was at the bottom of zg6r's spread. The full
run of zg6r and every comparison of zg6r with another arm on a zero-shot read are then called against a floor of 0.0866
to 0.1014 on those two reads, which a single seed cannot resolve: the full run's declaration has to state how many seeds
it fits for zg6r. That is for the declared file; nothing here chooses it.

Files: `outputs/wqg/floors-zg6.*`, `floors-zg6r.*`, `null-verdict-zg6.*`, `null-verdict-zg6r.*`, `wqg-*-pair.*`,
`wqg-*-pair-recall.*`, `outputs/screen/scr-wqg-*`. Not committed: `outputs/wqg/sessions`, `outputs/screen/fits`.

## 11. Amendment A1 (30 October): the seed guard compared a fit's own outputs

**What happened.** Both arms ran, were taken and had their compares and floors built (zg6 at 04:15, zg6r at 04:57). The
first call of `zwqg.py verdict` on each arm, at 04:57, returned INCOMPLETE with four problems per arm, one per fit, each
the same: `not the seed-0 fit <name> with the seed changed: train.json variants`. No other guard named anything.

**Cause.** Section 3 requires a WQG fit to differ from its seed-0 fit in the seed alone, and `SAME_TRAIN` (the fields of
`train.json` the two fits must share) listed `variants`. In a real `train.json`, `variants` is a dict, `{"p": {set, ctx,
blocks, widths, curve, ctx_stats, swa_epochs, seconds}}`, and two of those entries are the fit's own outputs: `curve` (the
loss per epoch, with the step count, which a different shuffle changes) and `seconds`. They differ from one seed to the
next by construction, so the guard refused every seed-k fit. The self-test's fixture wrote `"variants": "p"` (a string), the
shape of `screen.json`'s card and not of `train.json`, so it could not see this. The defect is in the guard, not in a fit:
for all eight fits the guard named no field but `variants`, and a field-by-field diff of two of them (zg6 seed 1 and zg6
seed 2 on the held-out-hotpotqa split) against their seed-0 fits showed only the seed, the weights' sha, the peak
figures, the seconds, the time stamp and the curve (with `variants.p.seconds`) differing.

**Amendment.** `zwqg.py` gains `OWN_OUTPUT = ("curve", "seconds")` and `train_card(t)`, and the guard compares `train_card`
of the two fits instead of `train.json`'s fields as they stand. `train_card` leaves `curve` and `seconds` out of each
variant of `variants` and nothing else; every other field, and every other guard (screen.json's card and code shas, the
`g6` and zg6rw blocks, the records, the trainer code, the seeds, the gate, the device, the read cap), is unchanged.

**What this does not touch.** No fit, take, compare or floor changes: the floors read the compares' R@5 and not
`train.json`, and `floors-zg6` and `floors-zg6r` were built before this amendment and are recomputed by the verdict as
section 5 says. The verdict still has no model number in it (FLOORS_READY or INCOMPLETE). The amendment is committed before
the verdict is called again, and the 04:57 INCOMPLETE calls are the first of two, named in section 10 when it is filled.

**Self-test.** The fixture writes the real shape (a `variants` dict with a curve and seconds that differ per seed, the
seed-0 fit's and each seed-k fit's). Added: a seed-k fit whose curve and seconds are changed again stays FLOORS_READY; a
changed `widths`, `swa_epochs`, `blocks` or `set` is INCOMPLETE and names `train.json variants`. With the old comparison
swapped in, the amended self-test fails with the message the real data gave. `zwqg.py --selftest` and `zwqg_session.py
--selftest` pass.
