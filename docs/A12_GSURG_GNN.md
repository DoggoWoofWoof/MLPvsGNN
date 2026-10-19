# A12: the GNN trained with gradient surgery

Declared 17 October 2026, before any number. The twelfth reopening of docs/VERDICT_AUDIT_2026_10_17.md (section 2,
row 12), under docs/SCREENS.md section 3b. Code: `outputs/mp_unified/za12gs.py` (the arm, the screen and the call).
Filed 19 October: zspkgs NOT_ADOPTED (section 7).

## 1. Why this, now

**Where the idea came from.** Gradient surgery (gsurg, `outputs/mp_unified/lean_screen5.py`, round 5) trains on all
training datasets at once but stops one dataset's update from undoing another's. At each 32-question step, each
dataset's gradient loses its component along any other dataset's gradient in the same step that it conflicts with
(a negative dot product).

**What it did so far.** It reached the MLP's line three times:
- on step 1's model: ADOPT at the re-grade (8 October);
- on zret (zgs, round 13): PROMISING;
- on zrm (zrg, round 21; 9 October): NOT_ADOPTED, with 0 primary GAIN and 0 LOSS. Its non-primary reads gained
  (webqsp zero-shot +0.0273, musique +0.0087).

On zrm the loop found conflicts in 45% to 48% of dataset pairs at every epoch, so the surgery acted throughout.

**Why that closes only its form.** zrg asked whether the surgery helps the MLP. It never reached the GNN, whose step
reads neighbours' scores per link family. The families' mix differs most between datasets (the KBs' typed links
against the passage graphs' hyperlinks and shared entities), so the step may be where datasets' gradients pull
apart most. Under section 3b, the idea stays open until it reaches the GNN.

**The base is zspk,** as the audit row names.

## 2. The arm

**zspkgs** is zspk (zp1c3.ZSPK on P1c3's KT link carve) trained inside lean_screen5's `gsurg_loop`:
- lean_gpu's training loop (`fit_variant`) is swapped for `fit_gsurg` while the fit trains, and restored after;
- `fit_gsurg` is lean_gpu's loop line for line, plus the surgery. Before each dataset's update in a step, it takes the
  other present datasets' gradients on their own questions of the step (in eval mode, so no dropout draw moves), and
  removes from the update its component along each one it conflicts with, in the step's dataset order;
- each epoch's surgery counts (pairs, conflicts, share, per dataset pair) land in train.json's curve.

**Nothing else changes.** zspk's model, carve, card, seed, batches and settings are used unchanged. No parameter, column
or hyperparameter is added. On one training dataset there is no other gradient, and the loop is lean_gpu's bit for
bit.

**Training only.** Reads and serving are zspk's, so any latency figure for zspkgs is zspk's, cold.

**GNN only.** The MLP's line had this loop (zrg on zrm). The MLP's model has changed since (zrckp), so a gain here
brings the loop to zrckp next (section 5).

| arm | the model | the carve | the loop |
| --- | --- | --- | --- |
| zspkgs | zspk, unchanged | zspk's (P1c3's KT link carve) | lean_screen5.fit_gsurg |

## 3. Cost

zrg's fits took 4.9 times zrm's training time (9,929 s against 2,024 s on L-musique), at the same peak GPU memory. Each
step takes every other present dataset's gradient, and a step holds up to four datasets.

zspk's card fits took 2,476 s (L-musique) and 2,559 s (L-hotpotqa), at peaks of 3.7 GB and 5.6 GB on the GPU. So
each zspkgs fit should take about 3.4 to 3.5 hours at zspk's memory.

## 4. The fits and the call

- **Fits:** one card fit on L-musique and one on L-hotpotqa (`scr-a12-zspkgs`, `-hp`), on P1c2's roots with P1c3's
  KT input. Card fit settings: variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128 (zspk's).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against zspk's card fit of the same split (scr-p1c3-zspk, scr-p1c3-zspk-hp), which decides.
  Step 1's fit is reported beside.
- **The re-call's base screens:** L1's, zspk's card fits compared with step 1's of the same split
  (outputs/l1/base-zspk-<split>).
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md). GAIN, LOSS and
  WITHIN are docs/SCREENS.md's, on R@5.

**A fit is A12's only if the loop ran.** `train` stops when the fit's curve lacks the surgery counts in any epoch, and
`read` refuses such a fit (zrg's check).

### The rule, fixed before any number

- **ADOPT_FOR_FULL_RUN:** zspkgs's re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A12's re-call of zspkgs, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- webqsp and metaqa, both splits (webqsp never trains; metaqa trains on both splits);
- musique in domain (L-hotpotqa): Claim 1's read;
- hit@1 and FC@5 per read;
- each fit's surgery counts: the share of conflicting pairs per epoch, and which dataset pairs conflict.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | zspkgs fitted again on P1g-b's pools, then its full run beside zspk's; and A12m, zrckp (the MLP's lean model) trained with gsurg's loop (each its own declared file) |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A12h, the back-off form: half the surgery. Each step's gradient loses half of its component along each conflicting dataset's gradient, so a conflict is damped, not removed (one declared file) |
| NOT_ADOPTED, with no GAIN | A12x, gsurg's loop on zspx (the adopted GNN, three steps, where depth paid) (one declared file) |
| INCOMPLETE | the missing re-call is run; nothing else changes |

No model change is adopted on today's pools (the user's rule of 13 October).

## 6. How it runs

**Self-test, passed before this commit.** `python outputs/mp_unified/za12gs.py --selftest` (17 s) checks:
- **The arm.** zspkgs is zspk's model on P1c3's KT link carve; zspk's, zsp's and gsurg's own arms are unchanged.
- **The swap.** Training swaps in gsurg's loop and zspk's model and carve, and restores all three after.
- **Toy fits** on P1c3's toy KT link carve, as two datasets whose golds lean opposite ways on one rank column (zrg's
  device), so their gradients conflict:
  - gsurg's loop over zspk is zspk's own loop bit for bit until its first conflict, and moves every state after it;
  - a repeat is bit for bit;
  - without the surgery (the other gradients still taken) the loop is zspk's bit for bit;
  - on one training dataset it is zspk's bit for bit, with no pair;
  - the step head (sp_w2) trains.
- **The curve check.** A curve with surgery counts in every epoch passes; lean_gpu's curve, and a missing train.json,
  do not.
- **The refusals.** train refuses zspk's arm, no arm, another split, the wrong fit name, another cache root and the
  CPU. read refuses zspk's fit, an unstamped fit, and a fit trained outside the loop.
- **The rest.** relz's pair and re-call stamped A12's, with relz restored after; the verdict's outcomes and next forms.

**Host items** (`a12-*`), after A11's verdict in the file, each starting when the GPU frees:
- 2 trains and 2 reads, sized from zspk's own items (`--frac 0.32`, share 0.34; memory: trains 3.5 GB, -hp 11 GB;
  reads 5 GB);
- 2 compares, the pair, the re-call and the verdict.

No build: zspkgs reads zspk's carves, already on the host.

## 7. Results

Filed 19 October 2026 from the host's records (`outputs/a12/screen-verdict.json`, written 2026-10-18T10:57:43Z; the
re-calls `outputs/screen/scr-a12-zspkgs-pair-recall.json`, host paths made relative). **zspkgs NOT_ADOPTED:** no GAIN,
and one LOSS.

- **zspkgs: 0 GAIN, 1 LOSS.** webqsp zero-shot falls 0.0137 on L-musique [-0.0220, -0.0057]; musique zero-shot falls
  0.0190 there (filed LOSS, re-called WITHIN against its 0.072 floor); squad in domain falls 0.0034 on L-hotpotqa,
  WITHIN.
- **Beside:** webqsp zero-shot on L-hotpotqa +0.0040; metaqa in domain +0.0016 and -0.0003; musique in domain
  (L-hotpotqa, Claim 1's read) +0.0007; all WITHIN.
- **The surgery acts at every step:** in every epoch of both fits, 0.454 to 0.485 of the dataset pairs conflict (their
  gradients' dot product is negative), and every pair conflicts on close to half of its steps; the four least
  conflicting pairs all involve 2wiki, the most conflicting involve squad or metaqa (table below).

The surgery, acting on about half the dataset pairs at every step, moves no read up past the seed null; webqsp zero-shot
pays for it on L-musique. This closes only this form (gsurg's loop on zspk). The next form, fixed in section 5 and
recorded in the verdict, is **A12x**, gsurg's loop on zspx (the adopted GNN, three steps, where depth paid); one
declared file.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's; a call is shown with its
filed call where the re-call changed it; base R@5 shown as zspk's):

| split | dataset | read | base R@5 | zspkgs: delta [95% CI], call |
| --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8009 | +0.0016 [+0.0006, +0.0026], WITHIN |
| L-musique | squad | in-domain | 0.9114 | +0.0006 [-0.0015, +0.0024], WITHIN |
| L-musique | musique | zero-shot | 0.5012 | -0.0190 [-0.0255, -0.0129], WITHIN (filed LOSS) |
| L-musique | hotpotqa | in-domain | 0.9080 | +0.0010 [-0.0010, +0.0031], WITHIN |
| L-musique | 2wiki | in-domain | 0.9383 | -0.0014 [-0.0028, +0.0000], WITHIN |
| L-musique | webqsp | zero-shot | 0.3229 | -0.0137 [-0.0220, -0.0057], **LOSS** |
| L-hotpotqa | metaqa | in-domain | 0.8040 | -0.0003 [-0.0012, +0.0006], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9135 | -0.0034 [-0.0055, -0.0013], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5754 | +0.0007 [-0.0043, +0.0056], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8499 | +0.0019 [-0.0011, +0.0051], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9333 | +0.0023 [+0.0006, +0.0039], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2680 | +0.0040 [-0.0060, +0.0144], WITHIN |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a12-zspkgs.json` and the others, `-hp` for L-hotpotqa):

| split | dataset | zspkgs FC@5 | zspkgs hit@1 |
| --- | --- | ---: | ---: |
| L-musique | metaqa | +0.0018 | +0.0035 |
| L-musique | squad | +0.0006 | -0.0019 |
| L-musique | musique | -0.0079 | -0.0629 |
| L-musique | hotpotqa | +0.0009 | -0.0007 |
| L-musique | 2wiki | -0.0025 | -0.0069 |
| L-musique | webqsp | -0.0126 | -0.0200 |
| L-hotpotqa | metaqa | -0.0004 | -0.0018 |
| L-hotpotqa | squad | -0.0034 | -0.0009 |
| L-hotpotqa | musique | -0.0017 | -0.0004 |
| L-hotpotqa | hotpotqa | +0.0043 | +0.0108 |
| L-hotpotqa | 2wiki | +0.0070 | -0.0083 |
| L-hotpotqa | webqsp | +0.0013 | -0.0047 |

The surgery per fit (`variants.p.curve[].gsurg` in each fit's train.json; a pair 'a on b' is dataset a's gradient
against b's, summed over the epochs):

| fit | conflicting share per epoch | least conflicting pairs | most conflicting pairs |
| --- | --- | --- | --- |
| L-musique | 0.469, 0.474, 0.460, 0.454, 0.461, 0.468, 0.459, 0.472 | hotpotqa on 2wiki 0.435, 2wiki on hotpotqa 0.439 | hotpotqa on metaqa 0.493, metaqa on squad 0.490 |
| L-hotpotqa | 0.465, 0.480, 0.478, 0.473, 0.462, 0.473, 0.485, 0.465 | 2wiki on musique 0.449, metaqa on 2wiki 0.451 | metaqa on squad 0.498, musique on squad 0.489 |
