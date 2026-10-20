# N2: a fresh seed null on today's lean pair

Declared 20 October 2026, before any number. Code: `outputs/mp_unified/zn2.py`. Not run yet.

Nearly every screen since 15 October is decided against today's lean pair (zrckp and zspk), but its floors still come
from step 1's model of 7 October. N2 measures the seed spread again on the two models the screens are decided against.
It then calls every filed arm on them again with the new floors, beside its filed call, and gates the three refits
queued on adopted arms. Numbers here are development numbers.

## 1. Why

**The rule.** A screen calls a read GAIN or LOSS only past the read's floor: max(0.0075, twice the seed-only spread on
that read) (docs/SCREENS.md, section 2; `outputs/mp_unified/screen_recall.py`). The spread was measured once, on 7
October. Step 1's base arm was trained again with seeds 1 and 2, and each was compared with its seed 0
(`outputs/screen/scr-null-floors.json`). Since 15 October nearly every screen is decided against one of two other
models: zrckp's card fits (`scr-p1c3p-zrckp`, `-hp`) or zspk's (`scr-p1c3-zspk`, `-hp`). Their own seed spread was
never measured.

**The arms.** Sources: `outputs/*/screen-verdict.json`, with each arm's base taken from its filed re-call
(`outputs/screen/scr-<stage>-<arm>-pair-recall.json`). L1 zspks, L3 zrckpo and L3 zspko have their re-calls on the
host only, so their base comes from their compares in the host items.
- 56 arms in 30 verdict files filed since 15 October are decided against zrckp or zspk (P1g-b's pool arms aside): 24
  on zrckp and 32 on zspk.
- 10 are ADOPT_FOR_FULL_RUN and 46 NOT_ADOPTED. The ten: A3v zrckpy; A5c zrckpc and zspkc; A7 zrckps and zspks; G3
  zspr; G4 zspk2 and zspkg; G5 zspx; R1 zrckpr.

**Their filed calls** (the verdicts' calls, 672 reads): 15 GAIN and 26 LOSS (zrckp 5 and 11, zspk 10 and 15).
- All 15 GAINs sit on three reads: L-musique webqsp (5), L-hotpotqa musique (5) and L-hotpotqa webqsp (5).
- The LOSSes: L-musique musique 5, hotpotqa 1, 2wiki 2 and webqsp 12; L-hotpotqa 2wiki 2 and webqsp 4.

**What the 7 October floors did to them.** The 53 filed re-calls on this machine start from their pairs' calls, made
at 0.0075: 48 GAIN and 46 LOSS. The 7 October floors left 15 and 25. They removed 33 GAINs and 21 LOSSes, all on four
reads (`scr-<stage>-<arm>-pair-recall.json`, each row's pair call beside its call):

| read | pairs, GAIN / LOSS | after the 7 Oct floor | 7 Oct floor |
| --- | --- | --- | ---: |
| L-musique musique | 21 / 14 | 0 / 5 | 0.0720 |
| L-hotpotqa webqsp | 14 / 14 | 5 / 4 | 0.0233 |
| L-hotpotqa hotpotqa | 3 / 1 | 0 / 0 | 0.0200 |
| L-musique webqsp | 5 / 12 | 5 / 11 | 0.0102 |

The 7 October floors are raised on 5 of 12 reads: these four and L-hotpotqa metaqa (0.0149). The largest, L-musique
musique, rests on one seed: D1 is +0.0503 and D2 -0.0079.

**Step 1's seed 0 is not today's base.** R@5 on the reads with a raised floor or a GAIN. Step 1's seed 0 (the 7
October null) is shown beside the card fits (`outputs/l1/base-<base>-<split>.json`, each card fit compared with step
1's):

| split | dataset (read) | 7 Oct seed 0 | zrckp card | zspk card | 7 Oct floor |
| --- | --- | ---: | ---: | ---: | ---: |
| L-musique | musique (zero-shot) | 0.2696 | 0.5121 | 0.5012 | 0.0720 |
| L-musique | webqsp (zero-shot) | 0.1779 | 0.3627 | 0.3229 | 0.0102 |
| L-hotpotqa | metaqa (in domain) | 0.6540 | 0.8044 | 0.8040 | 0.0149 |
| L-hotpotqa | musique (in domain) | 0.5550 | 0.5745 | 0.5754 | 0.0075 |
| L-hotpotqa | hotpotqa (zero-shot) | 0.8585 | 0.8513 | 0.8499 | 0.0200 |
| L-hotpotqa | webqsp (zero-shot) | 0.1200 | 0.3475 | 0.2680 | 0.0233 |

The cards nearly double R@5 on musique read zero-shot from L-musique, where the floor is largest. On webqsp, read
zero-shot from both splits, they raise it 1.8 to 2.9 times; two of the three reads that hold every GAIN, and most
LOSSes, are there. A spread measured at R@5 0.27 need not be the same at 0.51, larger or smaller.

**The adopted arms sit close to the floors.** The ten ADOPTs rest on 13 GAIN reads, and 7 of the 10 arms rest on one
GAIN read. 6 of the 13 reads clear their 7 October floor by less than 0.003:

| arm | read | delta R@5 | 7 Oct floor | margin |
| --- | --- | ---: | ---: | ---: |
| G5 zspx | L-hotpotqa webqsp | +0.0235 | 0.0233 | 0.0002 |
| A5c zrckpc | L-hotpotqa webqsp | +0.0239 | 0.0233 | 0.0006 |
| A5c zspkc | L-hotpotqa musique | +0.0083 | 0.0075 | 0.0008 |
| A3v zrckpy | L-musique webqsp | +0.0112 | 0.0102 | 0.0010 |
| G4 zspkg | L-hotpotqa musique | +0.0093 | 0.0075 | 0.0018 |
| G3 zspr | L-hotpotqa musique | +0.0096 | 0.0075 | 0.0021 |

Three refits on P1g-b's pools are queued on these calls: A5cp, A3vp and A7p (docs/A5CP_P1GB_POOLS.md,
docs/A3VP_P1GB_POOLS.md, docs/A7P_P1GB_POOLS.md). So N2 measures the floors again on the models the screens are
decided against, before those refits run.

## 2. The fits

- **Eight fits.** Each card fit is trained again with seeds 1 and 2 on both screen splits: `scr-n2-zrckp-s1`,
  `scr-n2-zrckp-s2`, `scr-n2-zrckp-s1-hp`, `scr-n2-zrckp-s2-hp`, and the same four for zspk. Only the seed changes.
  Everything else is the card fit's: the code (each module sha its record holds), the settings, the training carves,
  today's pools (P1c2's cache), the KT input and the device.
- **The seed reaches the trainer.** zp1c3.py's train passes `--seed` through rmatch's train to lean_screen's and on to
  lean_gpu's fit. That fit seeds torch and numpy and builds the model from the seed. The KT head's first draw follows it
  too (seed + 3301), and the fit's record carries the seed. So zspk's fits run zp1c3.py's train unchanged, with
  `--seed`.
- **zrckp's fits.** zp1c3p.py's train accepts only the card fit's name. `zn2.py train` first checks that the code is
  the card fit's. It then runs zp1c3p.py's train unchanged, with N2's fit name in place of the card's for that run (put
  back after, also when the fit fails), and stamps the fit as N2's (seed, split, card fit). No existing module changes.
- **Read and compare.** Each fit is read by its trainer's read, as its card fit was. It is then compared with its card
  fit alone by lean_screen's compare, p@swa on the six s1eval carves: `outputs/screen/scr-n2-<base>-s<seed>[-hp].json`.
- **What counts as a seed fit.** `zn2.py floors` refuses anything else.
  - The comparison is with the fit's own card fit alone: p@swa on s1eval, on its split's training sets and all six
    datasets.
  - The fit's record is the base arm with seed 1 or 2 on that split, and carries its trainer's stamp (and N2's, for
    zrckp).
  - It has every setting and code sha of the card fit's record.
  - Build records the card's record holds (the KT, rmatch, zrc and link records, the gate's sha) are reported beside,
    deciding nothing.
- No test split is read, webqsp is never trained on, and the encoder does not change.

## 3. The floors and the calls

- **The rule, unchanged** (docs/SCREENS.md, section 2; screen_recall.py): floor(r) = max(0.0075, 2 x sqrt((D1(r)^2 +
  D2(r)^2) / 2)), with D_s(r) seed s's R@5 minus the card fit's on read r. Only the null changes: each base's card fit
  and its seeds 1 and 2, in place of step 1's.
- **The floors**, per base: `outputs/n2/floors-zrckp.json` and `floors-zspk.json` (each with an `.md`). Each read shows
  the card fit's R@5, D1, D2, the spread and the floor. Beside them are the 7 October floor and seed 0, and whether the
  floor went up, down or stayed.
- **The calls.** Each of section 1's 56 arms is called from its filed pair (calls made at 0.0075), by screen_recall's
  pair, unchanged, with its base's fresh floors: `outputs/n2/recall-<stage>-<arm>.json`, `.md`. Each row carries three
  calls: the pair's, the 7 October one (the filed re-call's) and the fresh one.
  - A fresh floor below the 7 October one can give back a call that floor removed.
  - A fresh floor above it can only remove calls.
- **PASS_FRESH**: at least one GAIN and no LOSS under the fresh floors (screen_recall's PROMISING). Otherwise
  **NO_PASS_FRESH**.
- **Before an arm is called**, zn2.py checks:
  - its filed verdict is ADOPT_FOR_FULL_RUN exactly for the ten arms section 1 names;
  - its filed re-call has the verdict's calls and was made from this pair;
  - every pair row's base R@5 is the card fit's (the fresh null's seed 0).
- **The summary**, `outputs/n2/summary.json`, `.md`, holds:
  - the floors then and now;
  - the calls (pairs, 7 October, fresh) per base and read;
  - the ten adopted arms, each PASS_FRESH or NO_PASS_FRESH;
  - the filed NOT_ADOPTED arms that are PASS_FRESH;
  - arms filed on these bases after this declaration (listed, deciding nothing);
  - section 5's routing.

  It is COMPLETE only with all of these: both floors records, each OK; all 56 calls, each the file `recall-all` wrote
  and on those floors; and the reproduction check REPRODUCED. Otherwise it is INCOMPLETE.
- The filed verdicts are never rewritten. N2's calls stand beside them.

## 4. The reproduction check

- `zn2.py repro` makes each of the 56 filed re-calls again from its pair, with the OLD null (step 1's four compares of
  7 October). It goes through the stages' own path: relz's null loader with the base's card comparisons, and
  screen_recall's pair, unchanged. First the filed re-call must be the arm's, on step 1's null, made from this pair.
  Then it must come back exactly:
  - the verdicts and the fits;
  - every row (calls, floors, deltas, D values);
  - the changed calls and the near-floor reads;
  - the null's files.

  It also makes the 7 October floors again from their four compares. They must equal
  `outputs/screen/scr-null-floors.json`.
- **REPRODUCED** needs all three:
  - no arm MISMATCH or FAILED;
  - at least one arm per base REPRODUCED;
  - the 7 October floors reproduced.

  An arm whose files are not on the machine is NOT_CHECKED, which blocks nothing. On the host, where every pair and
  card comparison was made, all should be there (the clean-up never removes a .json file). A check that is not
  REPRODUCED makes N2 INCOMPLETE (section 5).
- **On this machine, before this commit:**
  - the 7 October floors come back exactly from their compares;
  - the 7 arms whose pairs are here reproduce through repro's own path: L1 zrckps, R1 zrckpr and zspkr, A4 zrckpd and
    zspkd, A5 zrckpk and zspkk;
  - all 53 filed re-calls here reproduce from pairs rebuilt out of their rows;
  - G2's base screens (`outputs/g2/base-zspk-*.json`) are on the host only. So for G2, l1's comparisons of the same
    card fit stand in; G2's pair rows carry the zspk card's R@5 on all 12 reads;
  - L1 zspks, L3 zrckpo and L3 zspko are checked on the host only.

## 5. What each outcome leads to (fixed now)

| outcome | what follows |
| --- | --- |
| COMPLETE; every adopted arm of A5c, A3v or A7 PASS_FRESH | that stage's gate (`n2-gate-a5c`, `n2-gate-a3v`, `n2-gate-a7`) exits 0, and its refit (A5cp, A3vp, A7p) runs as its own file declares |
| COMPLETE; an adopted arm of A5c, A3v or A7 NO_PASS_FRESH | the stage's gate exits 1 and its refit's trains do not run. That arm's next form is a second-seed screen of it, in its own declared file; it is not closed. If another adopted arm of the stage passes, that arm's refit can run behind `zn2.py gate --stage <stage> --arm <arm>` (a feeder edit, not N2's) |
| COMPLETE; a G3, G4, G5 or R1 arm NO_PASS_FRESH | nothing is stopped, since these are not gated. It is a caveat on every later stage built on that arm, named in each later file |
| COMPLETE; a filed NOT_ADOPTED arm PASS_FRESH | that arm gets a refit on P1g-b's pools in its own declared file. The summary names it; N2 declares none. Its filed verdict stands |
| INCOMPLETE (a floors record refused, an arm not called, or the reproduction check not REPRODUCED) | every gate exits 2 and nothing is routed. The missing piece is run and the summary made again; the refits are queued again behind their gates |

- **Later screens.** Screens on a zrckp or zspk base are called with N2's floors (`outputs/n2/floors-<base>.json`).
  Screens on a zspx base take zspk's fresh floors, and screens on a zrckpr base take zrckp's, labelled as their
  parent's.
- **The refits' own calls.** A5cp, A3vp and A7p are decided against P1g-b's fits, under the rule their own files
  declare. N2 decides only whether they run.
- **Today's pools.** No model change is adopted on today's pools (the user's rule of 13 October). A PASS_FRESH leads to
  a refit on P1g-b's pools, never to adoption on P1c2's.
- **The gate's exit codes.**
  - 0: the summary is N2's and COMPLETE, and the stage's adopted arms are all PASS_FRESH.
  - 1: the summary is COMPLETE, and one of them is NO_PASS_FRESH.
  - 2: not decidable (no summary, an INCOMPLETE one, one not N2's, or a stage or arm N2 does not gate).

  The feeder takes any non-zero exit as a failed dependency, so the gated trains drop after its one retry.

## 6. How it runs

**Self-test.** `python outputs/mp_unified/zn2.py --selftest` passes on the laptop before this commit (about 17 s). It
checks:
- the declared lists: 56 arms (24 zrckp, 32 zspk), the ten adopted, and the gated and ungated stages;
- the 7 October floors, made again from their compares by screen_recall's floors and by zn2's own arithmetic;
- a toy N2 null. It loads, and 28 wrong inputs are refused:
  - a file missing, repeated or one short;
  - seeds 0 and 3;
  - another arm, split, name or base, and a step-1 or second base;
  - a missing or wrong stamp;
  - a code sha or setting not the card's;
  - another candidate, carve or training set, and five datasets;
  - the two seeds' seed 0 apart;
  - the card's record missing or not seed 0.

  Build records are reported beside;
- a toy over all 56 arms: floors, the reproduction check (56 REPRODUCED), the calls and the summary.
  - PASS_FRESH and NO_PASS_FRESH come out where they were planted.
  - A NOT_ADOPTED arm whose GAIN the 7 October floor removed passes under a lower fresh floor and is named for a
    refit.
  - A late arm is listed; an arm filed before 15 October and P1g-b's pool arm are not.
  - The gates exit 0, 1 and 2 as section 5 says, and INCOMPLETE or foreign summaries give 2.
  - The filed records stay unchanged byte for byte, and no absolute path is written;
- train and compare:
  - the refusals;
  - the fit name swapped in and put back, also on a failure;
  - N2's stamp only after a fit;
  - lean_screen's compare called with exactly its arguments;
- the real filed re-calls (section 4);
- nothing written under `outputs/screen`, `l1`, `p1c2`, `p1c3`, `p1c3p` or the 30 stage folders.

**Host items** (32, `n2-*`; the GPU items first):
- **8 trains**, `n2-train-<base>-s<seed>[-hp]`, with no dependency. Each takes 1.1 CPUs and 4.5 GB (zspk on L-hotpotqa:
  7 GB). The GPU runs at `--frac 0.20` and share 0.22 on L-musique, and at `--frac 0.27` and share 0.29 on L-hotpotqa.
  These are A7's sizes for the same bases on the same pools.
- **8 reads**, `n2-read-...`, each after its train: 1.1 CPUs, 6 GB, `--frac 0.27`, share 0.29.
- **8 compares**, `n2-compare-...`, each after its read: 1 CPU and 4 GB.
- **CPU items**, 1 CPU and 1 GB each (measured peak 0.22 GB with the imports):
  - `n2-recall-floors-zrckp` and `n2-recall-floors-zspk`, each after its base's four compares;
  - `n2-recall-repro`, with no dependency;
  - `n2-recall-all`, after both floors;
  - `n2-recall-summary`, after those four;
  - `n2-gate-a5c`, `n2-gate-a3v` and `n2-gate-a7`, after the summary.
- **Exit codes.**
  - Trains, reads and compares exit 0 when done.
  - floors exits 0, or 2 after writing a REFUSED record.
  - repro, recall-all and summary exit 0 once written; their status is in the record.
  - The gates exit 0, 1 or 2 (section 5).
- **Inputs:** the zrckp card train's 35 (zp1c3p.py's import closure, the two relation-embedding files, retry_cmd.py and
  cuda_alloc.py) plus zn2.py. The floors, repro, recall-all and summary items add the four 7 October null compares and
  their floors file.
- **The refits' gates** (a feeder edit, not part of N2's items): add `n2-gate-a5c` to the dependencies of A5cp's four
  trains, `n2-gate-a3v` to A3vp's two and `n2-gate-a7` to A7p's four. Their reads and later items already follow their
  trains.
- **The pools' files.** The fits read P1c2's cache, chains and links and P1c3's KT columns, as the card fits did. While
  N2's items are queued, the host's clean-up keeps the cache their commands name and the folders named in the source
  of the scripts they run and import (zp1c3p.py, zp1c3.py, zp1c2.py), unless a release record has given those up. If
  any was removed before then, the trains fail and those builds run again first.
- **Disk (estimate):** L1's zrckp fits on these pools take 21.6 MB each: 8.2 MB of weights at train and about 13 MB of
  reads. So about 0.18 GB for the eight fits, and under 1 MB for the compares and N2's records.
- **ETA (estimate):**
  - L1's zrckp fits took 1,635 and 1,659 s on the host beside other work, and P1g-b's zspk fit 1,132 s alone on bigger
    pools. So expect about 20 to 30 minutes a fit and 3 to 7 minutes a read.
  - The GPU work is about 4 to 5 hours one at a time, or about 1.5 to 2 hours three at a time while the GPU is free.
  - The CPU items take minutes, and the gates open minutes after the last compare.

**The commands.** One base, seed and split are shown; the others follow the pattern. On the host the trains and reads
run under `retry_cmd.py --cuda` and `cuda_alloc.py --frac`, as in the items.

    python outputs/mp_unified/zn2.py train --split L-musique --name scr-n2-zrckp-s1 --arm zrckp --seed 1 \
        --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zp1c3.py train --split L-musique --name scr-n2-zspk-s1 --arm zspk --seed 1 \
        --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zp1c3p.py read --name scr-n2-zrckp-s1 --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zp1c3.py read --name scr-n2-zspk-s1 --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zn2.py compare --new outputs/screen/fits/scr-n2-zrckp-s1 \
        --base outputs/screen/fits/scr-p1c3p-zrckp --out outputs/screen/scr-n2-zrckp-s1
    python outputs/mp_unified/zn2.py floors --base zrckp
    python outputs/mp_unified/zn2.py repro
    python outputs/mp_unified/zn2.py recall-all
    python outputs/mp_unified/zn2.py summary
    python outputs/mp_unified/zn2.py gate --stage a5c

## 7. Results

Not run yet.
