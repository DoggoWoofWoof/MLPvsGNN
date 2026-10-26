# GL: the GPU line on U2d's base build, on Modal's L40S, while the host is offline

Declared 27 October 2026. Code: `outputs/mp_unified/zgl.py`. G6's call (section 2) reads fits filed on 22 October;
this file fixes how they are read before the call is made. R2's fits and G6's twin (sections 3 and 4) have not run.

## 1. Why this, now

- **Three stages are stranded.** N2 (the fresh seed null), G6 (the free GNN) and R2 (bridge transforms trained with
  EM) were declared on 20 October to run on the lab host's card. The host went offline that day and has not come back,
  so none of the three has run. The program of 24 October puts them first among the GPU stages, in that order
  (docs/PROGRAM_2026_10_24.md, section 4.3), and says: "Bases and nulls are refit on each new device."
- **U2d already did that, on one device.** U2d (docs/U2D_REFIT_ON_U2E.md) rebuilt today's pools on Modal as its base
  build: P1c2's construction, with P1c3's KT input. On one NVIDIA L40S, from one pinned image, it fitted:
  - zrckp and zspk with seeds 0, 1 and 2 (seeds 1 and 2 are U2d's null, its section 6);
  - zg6, G6's free GNN, with seed 0.

  The base build's KT arrays equal P1c3's on all eleven carves (REPRODUCED, U2d section 11), so the build holds the
  filed state.
- **What that gives.**
  - N2's design, run on the L40S: each base arm trained again with seeds 1 and 2, each compared with its seed-0 fit.
    U2d's floors are filed.
  - G6's deciding pair: zg6 against zspk on the same build, card settings and device.
  - R2's bases: zrckp's and zg6's seed-0 fits.
- **U2d's own file already ties zg6 to zspk's null.** Its section 6, declared on 21 October before U2d's numbers,
  says: "zspk's null calls the GNN arms (zspk, zspkv, zg6). G6's rule also takes zspk's seeds."

So GL decides G6 now, from those fits. R2 and G6's twin run on the same build and device when workspace 3, which holds
the build, renews on 1 November.

## 2. G6, decided on the L40S

**The rule is G6's** (docs/G6_FREE_GNN.md, section 4, fixed on 20 October before any number). So are the next forms
(its section 5). Only the files change:

| G6 as declared (the host's card) | GL (Modal's L40S, U2d's base build) |
| --- | --- |
| zg6's fits `scr-g6-zg6`, `-hp` | `scr-u2d-base-zg6`, `-hp`, filed 22 October |
| zspk's P1c3 card fits `scr-p1c3-zspk`, `-hp` | `scr-u2d-base-zspk`, `-hp` |
| N2's zspk null `scr-n2-zspk-s1`, `-s2`, `-s1-hp`, `-s2-hp` | U2d's zspk null `scr-u2d-null-zspk-s1`, `-s2`, `-s1-hp`, `-s2-hp` |
| G6's compares `scr-g6-zg6`, `-hp` | U2d's compares `scr-u2d-base-zg6-vs-zspk`, `-hp` |
| G6's pair, re-call and verdict | `outputs/screen/scr-gl-g6-zg6-pair`, `scr-gl-g6-zg6-pair-recall`; `outputs/gl/g6-verdict` |

- **ADOPT_FOR_FULL_RUN:** zg6's re-call under U2d's zspk null has no LOSS on any of its twelve reads, and at least one
  GAIN.
- **NOT_ADOPTED:** otherwise. With a GAIN and a LOSS the next form is G6b. With no GAIN it is G6r.
- **INCOMPLETE:** the re-call is missing, is not GL's re-call of zg6 under U2d's zspk null, or does not hold the twelve
  reads.

The next form's own file is written after this call. It runs on workspace 3 from 1 November, beside R2.

**What is the same as G6 declared.**
- The pools and carves: P1c2's pools at each question's frozen size, with P1c3's KT input. U2d's base build reproduces
  them.
- The model: G6's zg6 class, unchanged.
- The card settings: variant p, config `2e-3:1e-4:0.1:8:2`, hidden 128, seed 0, eight epochs, read as p@swa.
- The reads: the six s1eval carves.
- relz's pair and re-call: screen_pair and screen_recall, unchanged.

**What differs.**
- **The device.** Every fit the call reads ran on the L40S from U2d's image (the five fields of U2d section 5). Before
  anything is called, zu2d.py's one-device check runs over the eight fits: zg6's two, zspk's two and the null's four.
- **The read's batch.** U2d read zg6 at the default batch of 262,144 rows, which fits the L40S's memory. G6 set 65,536
  for the host's card. A batch holds whole questions, so the scores do not change (U2d, section 5).
- **No 7 October re-call beside.** G6 re-based it through L1's compares of the host's zspk fits with step 1's. That
  reads fits from another device, which U2d's one-device rule refuses, so it is not computed.
- **The twin comes later.** zg6t has no L40S fit yet; it runs from 1 November (section 4). It decides nothing in G6's
  rule, and its rows are filed beside when it runs.

**Not a blind call.** U2d filed zg6 against zspk on the base build on 22 October, as a reading beside its verdict (U2d
section 11; `outputs/screen/scr-u2d-base-zg6-vs-zspk.md` and `-hp.md`), and U2d's zspk floors with it. So G6's
outcome can be worked out by hand from filed tables, and it was, while this file was written. This file changes nothing
the call reads:
- the rule and the next forms are G6's of 20 October;
- the stand-in null is the one U2d's section 6 named on 21 October;
- the fits and compares are U2d's, as filed.

What the call adds is the record: zg6's twelve reads, called under the floors of the arm it is decided against.

**A limit, G6's, unchanged.** U2d's null measures zspk's seed spread, not zg6's, and a model trained from scratch may
vary more between seeds. That is why an ADOPT leads to a refit and a full run, not to a claim.

**Beside, deciding nothing.**
- zg6 minus zspk on each read: R@5 with its 95% question-bootstrap interval, FC@5 and hit@1 (U2d's compares).
- R@5 per read of rrf, zspk, zg6 and zrckp, all on the L40S. zrckp's seed-0 R@5 comes from U2d's zrckp null compares.
- The MLP share, (zrckp − rrf) / (zg6 − rrf), where zg6's R@5 gain over rrf is above 0.002.
- zspk's R@5 on the host's card beside zspk's on the L40S (L1's compares, `outputs/l1/base-zspk-<split>.json`): the
  same arm on two devices.

G5's zspx and G2's zg2 are not set beside, since their reads are from the host's card.

## 3. R2, on the L40S from 1 November

**The rule is R2's** (docs/R2_BRIDGE_EM.md, section 4). So are its constants, model and next forms (its sections 2
and 5). Only the files change:

| R2 as declared | GL |
| --- | --- |
| zrckpb's fits `scr-r2-zrckpb`, `-hp` | `scr-gl-zrckpb`, `-hp` |
| zg6m's fits `scr-r2-zg6m`, `-hp` | `scr-gl-zg6m`, `-hp` |
| zrckpb's base `scr-p1c3p-zrckp`, `-hp` | `scr-u2d-base-zrckp`, `-hp` |
| zg6m's base `scr-g6-zg6`, `-hp` | `scr-u2d-base-zg6`, `-hp` |
| zrckpb's null: N2's zrckp null | U2d's zrckp null (`scr-u2d-null-zrckp-*`) |
| zg6m's null: N2's zspk null, moved onto zg6 through G6's compares | U2d's zspk null, moved onto zg6 through `scr-u2d-base-zg6-vs-zspk`, `-hp` |
| INCOMPLETE when N2's floors record is missing or not OK | INCOMPLETE when U2d's null files fail zu2d.py's null checks |

- **The move onto zg6** is zu2d.py's re-base, the one U2d declared for its decision G: each read's seed-0 R@5 becomes
  zg6's, through the compare of zg6 with zspk. D1, D2 and the floors stay zspk's. It is the stand-in R2 itself
  declared.
- **The fits.** zr2br.py's two arms, unchanged: R2's head on zrckp and on zg6, with K = 5, r = 32 and LAM = 1, and its
  own generator at seed + 5502. Each trains with R2's EM term on U2d's base build (zu2d.py's roots for the build), with
  U2d's card settings and seed 0. Each runs on Modal's L40S from U2d's image and is stamped with its device. zg6m's
  reads keep R2's `--rows-cap 65536`.
- **The compares.** Each arm against its base's L40S fit alone. Step 1's fit is not set beside, since it is from another
  device. A compare between fits of two devices stops.
- **Beside, deciding nothing.** R2 section 4's beside rows that need no other device:
  - each arm minus its base per read: R@5 with its interval, FC@5 and hit@1;
  - the webqsp and metaqa reads, and musique in-domain (L-hotpotqa);
  - the EM term per epoch.

  The 7 October re-calls are not computed (another device).
- **zrckpb's place.** Whether reading the five bridges' vectors keeps zrckpb inside the MLP track is still the user's
  ruling (R2, section 2). Until then every record calls it "zrckp + R2's bridge head".
- **zg6m** is decided against zg6, whatever G6's verdict (R2, section 3).

## 4. G6's twin, on the L40S from 1 November

zg6t is fitted on U2d's base build as `scr-gl-zg6t` and `-hp`, with the same card settings and device. It is G6's arm,
unchanged: zg6 with no edges reaching its layers.

Beside, deciding nothing:
- the twin against zspk, re-called under U2d's zspk null (G6's beside row);
- zg6 minus zg6t on each read: what the edges add to the same model, with R@5's interval, FC@5 and hit@1.

The twin is also G6b's other half: G6b mixes zg6's score with its twin's. (Corrected 27 October: G6b's twin is its own
no-edge pass, trained together with its edge pass, so G6b does not need these fits; docs/G6B_GATED_BACKOFF.md,
section 1.)

## 5. What waits for the host

- **N2's re-calls of the 56 filed arms**, its reproduction check and its gates. Their pairs and card comparisons are on
  the host. U2d's null is N2's design on the L40S, so the stages GL runs there are called on it. On the host's card,
  N2 stays as declared.
- **A5cp, A3vp, A7p and P5c5**, the refits on P1g-b's pools, which are on the host only. Their ideas reach U2n's
  forms, each as its own declared form (the program of 24 October, section 4.2): A5c's coupled kind offsets, A3v's
  columns, A7's typed columns and P5c's offsets.

## 6. How it runs

**Now, on the laptop.** This takes seconds; it reads filed files only and trains nothing:

    python outputs/mp_unified/zgl.py g6

It checks the eight fits' device, then makes:
- the pair (screen_pair, with GL's loader of U2d's two compares);
- the re-call (screen_recall, with U2d's zspk null loaded by zu2d.py's u2d_null);
- the verdict.

**From 1 November, on workspace 3,** which holds U2d's base build: six fits and their reads (zrckpb, zg6m and zg6t on
both splits). The Modal stages that run them and GL's train, read and compare commands are added in a later commit,
with their self-test, before any of these fits runs. They go into zu2d_modal.py, on U2d's image and L40S, as U2h's
stages were added. Then, on the laptop: the compares, R2's pairs, re-calls and verdict, and the twin's rows.

**Cost and time** come from U2d's measured seconds on the L40S:
- zrckp's fits took 13 to 14 minutes and its reads 5;
- zg6's fits took 22 minutes (L-musique) and 40 minutes (L-hotpotqa), and its reads 3 to 4.

U2d's 24 fits cost about $50 of GPU time, about $2 per fit with its read. So the six cost about $12 of workspace 3's
$30. About 2 hours from the first fit to the last read, two fits at a time.

**Self-test, passed before this commit.** `python outputs/mp_unified/zgl.py --selftest` runs on fake files in a
temporary folder:
- **The pair** is screen_pair's over GL's loader. The loader refuses a compare that is not U2d's, not of zg6 against
  zspk on the base build, of another split, read on other carves or with another candidate, or missing a dataset.
- **The device check** stops when any of the eight fits ran on another device or image.
- **The re-call** is screen_recall's under U2d's zspk null, with no re-base, since the pair's base is the null's
  seed-0 fit. A filed GAIN or LOSS below its read's floor becomes WITHIN.
- **The verdict** follows G6's rule: ADOPT_FOR_FULL_RUN, NOT_ADOPTED with G6b or G6r next, and INCOMPLETE for a
  missing re-call, one that is not GL's or not under U2d's zspk null, or one without the twelve reads. Its beside rows
  are checked.
- Nothing is written under GL's or the screens' real folders.

**Added 27 October, before any of the six fits runs: their code.** `outputs/mp_unified/zgl2.py` trains, reads,
compares and calls R2's two arms and the twin. `zu2d_modal.py` runs the fits through three new stages. On workspace 3:

    modal run outputs/mp_unified/zu2d_modal.py --stage upload
    modal run outputs/mp_unified/zu2d_modal.py --stage gl-probe
    modal run --detach outputs/mp_unified/zu2d_modal.py --stage gl-fits
    modal run outputs/mp_unified/zu2d_modal.py --stage fetch-gl-fits

Then on the laptop:

    python outputs/mp_unified/zgl2.py compare-all
    python outputs/mp_unified/zgl2.py calls
    python outputs/mp_unified/zgl2.py verdict

- **The fits** run as U2h's did, on U2d's base build: one container per fit, which trains and then reads, and stamps
  the fit with its device.
  - A fit's name fixes its arm and split. The seed is 0. U2d's card settings cannot be changed from the command line.
  - R2's arms train inside R2's EM term, through zr2br.py's own wrapper. As in zr2br.py, a fit where no training batch
    carried the term stops without a stamp.
  - The twin trains as zg6.py trains it.
- **The reads' batch** comes from the arm, never from the caller:
  - 65,536 rows for zg6m (R2's setting) and for zg6t (G6's);
  - lean_gpu's default of 262,144 for zrckpb, as in R2's zrckpb reads and U2d's zrckp reads.

  A batch holds whole questions, so the scores do not change.
- **Two fits at a time**, set by the fit runner's container limit. Modal queues the rest in this order: zg6m-hp,
  zg6m, zrckpb-hp, zrckpb, zg6t-hp, zg6t. So R2's four fits come first, the longest first.
- **The run is spawned on the deployed app**, not held by a laptop client: a killed client cancels even a detached
  run.
- **Eight compares.** All the fits in each compare ran on one device.

  | compare | new | base | role |
  | --- | --- | --- | --- |
  | `scr-gl-zrckpb`, `-hp` | zrckpb | `scr-u2d-base-zrckp`, `-hp` | R2: decides zrckpb |
  | `scr-gl-zg6m`, `-hp` | zg6m | `scr-u2d-base-zg6`, `-hp` | R2: decides zg6m |
  | `scr-gl-zg6t`, `-hp` | zg6t | `scr-u2d-base-zspk`, `-hp` | G6's beside row |
  | `scr-gl-zg6-vs-zg6t`, `-hp` | zg6 | zg6t | what the edges add; beside |
- **The calls, for each arm:**
  - `outputs/screen/scr-gl-<arm>-pair`: screen_pair over GL's loader of the arm's two compares;
  - `scr-gl-<arm>-pair-recall`: screen_recall under the arm's U2d null. zg6m's null is moved onto zg6 through
    `scr-u2d-base-zg6-vs-zspk`, `-hp`.

  R2's verdict goes to `outputs/gl/r2-verdict.json` and `.md`. The twin's rows go to `outputs/gl/zg6t-beside.json` and
  `.md`; they decide nothing.
- **Self-test, passed before this commit.** `python outputs/mp_unified/zgl2.py --selftest` runs on fake files in a
  temporary folder:
  - **Train and read refuse before any data is read** on:
    - a name that is not GL's, or another arm or split;
    - a fixed flag: the seed, hidden size, threads, output folder or config;
    - the laptop, another GPU or another image;
    - a read at another rows cap or on another device;
    - a fit stamped with other R2 constants.
  - **What a run uses.** R2's term wraps R2's arms only. Every root is the base build's while a fit runs, and is
    restored after. The read gets the arm's rows cap. The command line reaches train and read.
  - **The loader refuses** a compare that is not GL's; one of another split, base or device; one read on other carves
    or with another candidate; and one missing a dataset.
  - **The re-call** moves zg6m's null onto zg6 and keeps zspk's floors: a GAIN of +0.0101 below a raised floor of
    0.0139 becomes WITHIN.
  - **The verdicts:**
    - ADOPT_FOR_FULL_RUN;
    - NOT_ADOPTED, with R2g or R2e next;
    - INCOMPLETE for a re-call that is not GL's, one without the twelve reads, a missing one, or a missing null file.
  - **compare-all** keeps filed compares and names any it skips.
  - **Nothing is written** under GL's or the screens' real folders.

  A local check of the stages, with no Modal call, also passed. It covers:
  - their fit names and order;
  - their refusal of `--build`, `--datasets`, `--carve` and `--rows-cap`;
  - the fit runner's limit of two containers.

The text encoder stays frozen. webqsp never trains. Test splits are never read. There is no LLM or transformer, and no
new graph.

## 7. Results

### G6, 27 October 2026: NOT_ADOPTED

Run with `python outputs/mp_unified/zgl.py g6` at 5872d26. Records: outputs/screen/scr-gl-g6-zg6-pair.json and .md,
outputs/screen/scr-gl-g6-zg6-pair-recall.json and .md, and outputs/gl/g6-verdict.json and .md. All eight fits ran on
one device: the NVIDIA L40S, on U2d's image (torch 2.8.0+cu128, CUDA 12.8, cuDNN 91002).

zg6 against zspk, re-called under U2d's zspk null, by G6's rule:
- **2 GAIN:** L-musique metaqa (+0.0102) and L-hotpotqa musique (+0.0138).
- **3 LOSS:** L-musique webqsp (-0.0795), L-hotpotqa hotpotqa (-0.0203) and L-hotpotqa webqsp (-0.0584).
- **One call changed** with the re-call: L-hotpotqa 2wiki (+0.0101) became WITHIN, since its floor is 0.0139. No read
  is within rounding of its floor.

**Next form**, as fixed in G6, section 5: **G6b**, a per-question back-off. zg6's score is mixed with its twin's
through a learned gate on the question's edge-family shares, at equal weight at the start. G6b gets its own declared
file. It needs zg6t's fits, which run on workspace 3 from 1 November (section 4). *Corrected 27 October:* it does not.
Its twin is the same network's no-edge pass, trained together with the edge pass, and it runs on KG's T4
(docs/G6B_GATED_BACKOFF.md, section 1).

R@5 per read. Every column is on the L40S except the last. The MLP share is (zrckp - rrf) / (zg6 - rrf):

| read | rrf | zspk | zg6 | zrckp (MLP) | MLP share | floor | call (filed -> re-call) | zspk, host card |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| L-musique metaqa (in-domain) | 0.0047 | 0.8021 | 0.8123 | 0.8008 | 0.9858 | 0.0075 | GAIN | 0.8009 |
| L-musique squad (in-domain) | 0.9054 | 0.9118 | 0.9106 | 0.9121 | 1.2885 | 0.0075 | WITHIN | 0.9114 |
| L-musique musique (zero-shot) | 0.4734 | 0.5143 | 0.5207 | 0.5153 | 0.8858 | 0.0089 | WITHIN | 0.5012 |
| L-musique hotpotqa (in-domain) | 0.6846 | 0.9088 | 0.9118 | 0.9055 | 0.9723 | 0.0075 | WITHIN | 0.9080 |
| L-musique 2wiki (in-domain) | 0.6079 | 0.9346 | 0.9416 | 0.9345 | 0.9787 | 0.0075 | WITHIN | 0.9383 |
| L-musique webqsp (zero-shot) | 0.0535 | 0.3162 | 0.2368 | 0.3561 | 1.6508 | 0.0105 | LOSS | 0.3229 |
| L-hotpotqa metaqa (in-domain) | 0.0047 | 0.8043 | 0.8112 | 0.8033 | 0.9902 | 0.0075 | WITHIN | 0.8040 |
| L-hotpotqa squad (in-domain) | 0.9054 | 0.9117 | 0.9118 | 0.9124 | 1.0937 | 0.0075 | WITHIN | 0.9135 |
| L-hotpotqa musique (in-domain) | 0.4734 | 0.5800 | 0.5937 | 0.5697 | 0.8005 | 0.0107 | GAIN | 0.5754 |
| L-hotpotqa hotpotqa (zero-shot) | 0.6846 | 0.8512 | 0.8309 | 0.8538 | 1.1565 | 0.0145 | LOSS | 0.8499 |
| L-hotpotqa 2wiki (in-domain) | 0.6079 | 0.9290 | 0.9391 | 0.9303 | 0.9734 | 0.0139 | GAIN -> WITHIN | 0.9333 |
| L-hotpotqa webqsp (zero-shot) | 0.0535 | 0.2773 | 0.2189 | 0.3370 | 1.7140 | 0.0405 | LOSS | 0.2680 |

Beside, deciding nothing:
- **Where zg6 wins and loses.** zg6 is above zspk on 7 of its 8 in-domain reads; squad, at -0.0012, is the
  exception. It is below zspk on 3 of its 4 zero-shot reads; musique, at +0.0063, is the exception. So the free GNN
  fits the training graphs better and transfers worse. That is the case G6b's back-off is for.
- **The MLP share.**
  - 0.97 to 0.99 on metaqa, hotpotqa and 2wiki in-domain.
  - 0.80 on L-hotpotqa musique (in-domain) and 0.89 on L-musique musique (zero-shot).
  - Above 1 on squad, both webqsp reads and zero-shot hotpotqa, where zrckp is above zg6.
  - So, on the L40S, musique is where the MLP keeps less than 90% of the free GNN's gain.
- **zspk on two devices.** zspk's L40S fits and its host card fits (the last column) differ by up to 0.0131 (L-musique
  musique, zero-shot: 0.5143 against 0.5012). On webqsp they differ by -0.0067 and +0.0093. This is why GL compares
  fits from one device only.
