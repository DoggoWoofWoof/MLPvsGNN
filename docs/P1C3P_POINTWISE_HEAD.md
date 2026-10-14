# P1c3p: a pointwise KT head for the MLP (zrckp)

Declared 14 October 2026 at 18:53. It comes after P1c3's screen ([P1C3_KT_INPUT.md](P1C3_KT_INPUT.md), section
7, filed d06c384) and before any number of this stage. P1c3's full run ([P1C3_FULL_RUN.md](P1C3_FULL_RUN.md)) is running
and its verdict is not filed. Numbers here are development numbers.

**Why P1c3p.** zrck, P1c3's MLP arm, breaks the MLP's input rule. Its head reads nine inputs per row: the row's eight
KT columns, and the row's score z-scored over its question's rows. That z-score is the row's score minus the mean of
the other rows' scores in the same pool, divided by their spread. It uses no edge and no propagation, but a row's
final score then depends on the other rows' scores.

The rule, docs/SCREENS.md, the twenty-third round (8 October): "A row's inputs are computed before any row is scored,
from the question, the row, the frozen features and the graph's fixed structure, and never from another row's score."

- zrck's z-score is computed after the rows are scored, from the other rows' scores, so it breaks the rule. P1c3's
  declaration said the head reads "its own score's z-score over the question's rows" and called the scoring free of
  message passing. That wording is wrong on this point, and this file corrects it.
- zrc's own inputs are not affected. zret's question-level statistics (each row's first-stage scores set against the
  pool's) come from frozen features, before any row is scored, so they fit the rule.
- zspk, P1c3's GNN arm, keeps the z-scored score. The GNN may read other rows; its propagation does.

## 1. The arm

**zrckp** is zrc (the MLP, zrm.ZRM) plus a pointwise KT head:
- **8 inputs:** the row's eight KT columns, exactly as zrck reads them (`scripts/p1c3_kt.py`, the same files). There is
  no score input.
- **One hidden layer of 16 (GELU), one output added to the score: 161 parameters** (zrck's head has 177).
- The first layer comes from its own generator (seed + 3301, P1c3's) and the last layer is zero. At the start, zrckp's
  forward is zrc's, bit for bit, and the global generator is where zrc leaves it (the self-test).
- Everything else is P1c3's: the carve (zrc's chain carve with the KT input beside), the settings, seeds and batches,
  the card, and P1c2's roots (looks, caches, chains, links, gate).

**Why no score input, and not a score standardised with fixed numbers.** On 14 October the fix was first described
to the user as the score standardised with an average and spread learned in training and frozen when serving, as
BatchNorm keeps them. This file takes the simpler form, for three reasons:
- **It is pointwise by construction.** A row's head output is a function of its own eight columns. Nothing has to be
  learned, frozen or carried between training and serving.
- **A running average and spread would be a buffer.** lean_gpu's SWA averages every state entry except the context
  buffers, so the statistics would be either averaged across epochs or left at their last value. That is a choice
  this stage would have to make and test, and it adds nothing to the question asked here.
- **The head still adds where KIND_T's walks put the row to zrc's score.** zrc's model already reads every other
  input of the row.

What the score input did for zrck, the head can no longer do: weigh the walk positions by how high the model already
ranks the row. Reported beside (section 2): zrckp minus zrck on every read. That measures the price of the fix.

## 2. The screen

- **Fits:** two on the host's card, `scr-p1c3p-zrckp` on L-musique and `scr-p1c3p-zrckp-hp` on L-hotpotqa
  (`zp1c3p.py train`), P1c3's settings and seeds.
- **Reads:** each fit, p@swa, on the six s1eval carves of P1c2's pools (KIND_T's), with P1c3's KT input.
- **Base, which decides:** zrc's card fit of the split on today's pools, `scr-zrct` and `scr-zrct-hp`. These are
  zrck's own bases in P1c3's screen, so the two screens are read on the same footing.
- **Beside, deciding nothing:**
  - zrck's P1c3 fit of the split (`scr-p1c3-zrck`, `scr-p1c3-zrck-hp`): zrckp minus zrck, R@5, FC@5 and hit@1 per
    read, with the 95% question-bootstrap interval of the R@5 difference;
  - P1c2's fit of zrc (`scr-p1c2-zrc`, `-hp`) and step 1's fit of the split.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), stamped as
  P1c3p's. GAIN, LOSS and WITHIN are docs/SCREENS.md's.

## 3. The question

Does the pointwise head keep zrck's screen gains? zrck had 6 GAINs and no LOSS in P1c3's screen (d06c384). zrckp
answers it on the same twelve reads, against the same bases.

## 4. The rule, fixed before any number

- **ADOPT_FOR_FULL_RUN:** zrckp's re-call under the seed null has no LOSS on any of its twelve reads, and at least one
  GAIN. This is P1c3's rule for one arm.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** the re-call is missing, is not P1c3p's re-call of zrckp, or does not hold the twelve reads (the two
  splits on the six datasets).

The verdict is written by `zp1c3p.py verdict` to `outputs/p1c3p/screen-verdict.{json,md}`, with zrckp minus zrck
beside. One input and one rule for all six datasets; nothing is chosen per dataset.

## 5. What follows

- **zrck serves no claim, whatever P1c3's full run says.** It breaks the MLP's input rule. P1c3's full run still files
  its verdict as declared; zspk's part of it stands on its own.
- **ADOPT_FOR_FULL_RUN, and P1c3's full run adopts:** zrckp's full run, in its own declared file: the same six splits
  as P1c3's full run, under grade-nullx. If it is adopted, the lean pair for the claims is (zrckp, zspk).
- **ADOPT_FOR_FULL_RUN, and P1c3's full run does not adopt:** P1c3p stops at its screen. Its record goes to P3's
  ledger beside P1c3's.
- **NOT_ADOPTED:** the lean MLP for the claims stays zrc. The record says, read by read, where the score input
  mattered (zrckp minus zrck), and P3's ledger continues from there.
- **Serving cost.** P4c times zrck's head: nine inputs, with the per-question z-score. zrckp's head reads eight inputs
  and no z-score, so P4c's head time is an upper bound for it. The KT columns and their walks are the same for both.

## 6. How it runs

- **Code:** `outputs/mp_unified/zp1c3p.py` (train, read, compare, pair, recall, verdict). It imports P1c3's
  `zp1c3.py` unchanged: the KT loading and its checks, the carve, and P1c2's roots, gate and card. Its self-test
  passes:
  - the arm registered on P1c3's chain carve, with P1c3's and P1c2's arms unchanged;
  - at the start, zrckp equals zrc bit for bit, in eval and in training, and the global generator is unmoved; 161
    head parameters, the last layer zero;
  - the head by hand. A batch without the KT input equals zrc's. Changing the other rows' KT columns leaves a row's
    head output unchanged;
  - toy fits move the head and repeat bit for bit;
  - train and read refuse another arm, split, fit name, cache root or device, and a fit that is not P1c3p's;
  - relz's pair and re-call against today's card fits, stamped P1c3p's, with relz restored after;
  - the verdict's five cases (adopt, a LOSS, no GAIN, a foreign re-call, fewer reads), a missing re-call, and the
    beside rows.
- **No development run on real data.** P1c2's caches and P1c3's KT files are on the host only.
- **Host items** (`p1c3p-*`):
  - the two fits on the card (`--frac 0.30`, share 0.32, 3.5 GB: zrck's screen fits);
  - the two reads after their fits;
  - then 2 compares, the pair, the re-call and the verdict.
- **Disk:** under 0.1 GB.
- **ETA:** the fits take the card's next free slots, behind P1c3's full run's reads. The zrck fits of P1c3's full
  run took 35 to 54 minutes. Reads, compares and calls take about 10 minutes. The verdict is about 1 hour after
  the fits start.
- The encoder stays frozen. webqsp never trains. Test splits are never read.

## 7. Results

To be filed.
