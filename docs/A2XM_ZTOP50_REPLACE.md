# A2xm: ztop50's replacing form on the lean pair, on both models

Declared 19 October 2026, before any number. The next form A2h's verdict named for its top-50 half (zrckpm and zspkm,
NOT_ADOPTED with no GAIN; docs/A2H_HALVES_APART.md, section 5, and `outputs/a2h/screen-verdict.json`), under
docs/SCREENS.md section 3b and docs/VERDICT_AUDIT_2026_10_17.md. Code: `outputs/mp_unified/za2xm.py`.

## 1. Why this, now

**ztop50** took every pool z-score against the pool's 50 best retrieved rows by rrf, in place of all of them. On step
1's model it gave the largest zero-shot gain of any screen. Its re-call under the seed null
(`outputs/screen/scr-ztop50-pair-recall.json`), R@5 minus step 1's fit of the same split:

| read | delta [95% interval] | floor | call |
| --- | --- | --- | --- |
| L-musique, musique zero-shot (0.2696 to 0.4083) | +0.1387 [+0.1279, +0.1492] | 0.0720 | GAIN |
| L-musique, webqsp zero-shot | +0.0175 [+0.0035, +0.0321] | 0.0102 | GAIN |
| L-hotpotqa, musique | +0.0115 [+0.0056, +0.0178] | 0.0075 | GAIN |
| L-musique, metaqa | −0.0176 [−0.0212, −0.0140] | 0.0075 | LOSS |
| L-hotpotqa, metaqa | −0.0252 [−0.0291, −0.0214] | 0.0149 | LOSS |

The re-call was MIXED and the form was dropped. On rel's inputs (relz, docs/SCREENS.md) the same trade came back:
musique zero-shot +0.1563 (GAIN), metaqa in domain −0.0196 and −0.0242 (LOSS). zret, which takes the z-scores against
all retrieved rows, was adopted instead, and today's lean pair is built on it.

**A2h's top-50 half** put the top-50 z-scores beside zret's, through one weight matrix into l1 (zero at the start). Its
re-calls (`outputs/screen/scr-a2h-zrckpm-pair-recall.json`, `scr-a2h-zspkm-pair-recall.json`), R@5 minus the base arm's
card fit:

| read (floor) | zrckpm (MLP) | zspkm (GNN) |
| --- | --- | --- |
| L-musique, musique zero-shot (0.0720) | +0.0198 [+0.0144, +0.0253]: GAIN filed, WITHIN under the null | +0.0275 [+0.0214, +0.0342]: GAIN filed, WITHIN under the null |
| L-hotpotqa, webqsp zero-shot (0.0233) | −0.0144 [−0.0254, −0.0032]: LOSS filed, WITHIN | −0.0125 [−0.0226, −0.0018]: LOSS filed, WITHIN |
| L-musique, 2wiki (0.0075) | −0.0031 [−0.0048, −0.0015], WITHIN | −0.0083 [−0.0099, −0.0067], LOSS |
| L-musique, metaqa (0.0075) | −0.0053 [−0.0063, −0.0043], WITHIN | −0.0000 [−0.0011, +0.0010], WITHIN |
| L-hotpotqa, metaqa (0.0149) | −0.0039 [−0.0050, −0.0028], WITHIN | −0.0035 [−0.0045, −0.0024], WITHIN |

Both were NOT_ADOPTED with no GAIN, zspkm with one LOSS (L-musique 2wiki). Beside the z-scores, musique zero-shot rose
within its floor and metaqa no longer paid.

**Why replacing now.** Beside the z-scores, an in-domain fit can lean on zret's z-scores and leave the new columns. The
replacing form carried the gain, and it has never run on today's pair. That pair reads musique zero-shot at 0.5121
(zrckp) and 0.5012 (zspk) in L-musique's fit, above rrf's 0.4734 (the base and rrf columns of A2h's re-calls); step
1's model read 0.2696 there. This screen measures whether the cut still adds on top of zret's reference and the lean
pair's chain and KT inputs. If metaqa's loss comes back, the back-off form is fixed for it (section 5).

## 2. The form

zret's reference, a pool's retrieved rows, is cut to its 50 best by rrf. Every within-pool z-score zret takes is taken
against the cut reference instead.

| | zret (today's lean pair) | A2xm |
| --- | --- | --- |
| a pool's reference rows | every retrieved row (rrf > 0) | the 50 best retrieved rows by rrf (rrf > 0, ties by row order) |
| z-scored against them | each block's columns, fixed and SEMB, and rrf's base z-score | the same |
| fewer than two reference rows, or a column constant over them (sd < 1e-6) | the whole pool's z-score | the same |

**Constants.** K = 50 (ztop50's, `lean_screen4.K`). rrf is the rank block's column: the first stage's reciprocal rank
fusion of the dense and SPLADE lists. Ties go by row order (two stable sorts, `lean_screen4.top_ref`). eps is zret's
1e-6. No parameter, column or hyperparameter is added.

**What does not change:**
- the raw columns and the keep flags;
- rmatch's chain term;
- the KT head: zrckp's reads the row's own eight KT columns; zspk's reads them and the score's whole-pool z-score;
- zsp's propagation, which z-scores the score over the whole pool, not against zret's reference.

**How it is built.** Python's method order puts ztop50's forward (`lean_screen4.ZTop`) between the base's heads and
zret's forward. ChainMatch's call to its parent's forward now reaches ZTop. The arm classes have no body.

**Where it changes nothing.** Where no pool has more than 50 retrieved rows, the arm is its base bit for bit. On the
pools ztop50 ran on, the retrieved rows numbered 205 to 379 in the big pools and 50 to 63 in the small ones
(`lean_screen4.py`). How many of P1c2's pools have more than 50 is not counted here; the caches are on the host.

**The MLP rule.** The reference is chosen by rrf alone, a first-stage input column cached with the pool before any row
is scored. No score is read, and the reference is the same whatever the model's weights. What is normalised over a
question's rows is what zret already normalises: input columns (the rank block, SEMB's learned map of query and node,
rrf), now against fewer of the same rows. So zrckpxm keeps the rule: no message passing, no input from another row's
score, nothing normalised over the question's rows from a score. docs/VERDICT_AUDIT_2026_10_17.md, section 2: z-scores
of first-stage retrieval values within the pool are computed before any row is scored, as zret's are. The selftest
asserts it (section 6).

## 3. The arms

| arm | model | carve |
| --- | --- | --- |
| zrckpxm | zrckp's model with ztop50's reference (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, ZTop, ZRet) | zrckp's (`zp1c3.KTChainCarveZRC`) |
| zspkxm | zspk's model with ztop50's reference (MRO: ZSPK, ZProp, KTMix, ZRM, ChainMatch, ZTop, ZRet); zsp's propagation reads the score it changes | zspk's (`zp1c3.KTLinkCarveBase`) |

Each arm has its base's parameters exactly: the same names and shapes, and the same initial values at the same seed.

## 4. The fits and the call

A2h's, unchanged, per arm:
- **Fits:** one card fit on L-musique and one on L-hotpotqa (`scr-a2xm-<arm>`, `-hp`), on P1c2's roots with P1c3's KT
  input, at the card fits' settings (variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split (zrckp: scr-p1c3p-zrckp and -hp; zspk:
  scr-p1c3-zspk and -hp), with step 1's fit beside. The re-call's base screens are L1's.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md).

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A2xm's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names each arm's next form (section 5).

**Reported beside, deciding nothing:**
- musique (the read ztop50 gained on, and Claim 1's read in domain), with both arms side by side;
- metaqa (the read the replacing form lost on step 1's and rel's models) and webqsp;
- hit@1 and FC@5 per read;
- ztop50's and A2h's top-50 deltas, set beside each read when the results are filed.

**Serving cost.** One top-50 reference per question (two stable sorts over the batch's rows) in place of zret's rrf > 0
test; the z-scores are as many as zret's. Flagged, to be timed cold before any latency claim.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools, then its full run |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A2xmg, the learned back-off between the two references: each z-score taken as z_ret + 2σ(⟨w, c⟩ + b) · (z_top − z_ret), w and b zero so the scale starts at 1 (A2xm's form), c the pool's log size and retrieved share (input columns only), on both models |
| NOT_ADOPTED, with no GAIN | A2xmx, ztop50's replacing form on the adopted arms (G5's zspx and R1's zrckpr), on both models; the top-50 idea closes as a model input only if A2xmx finds no GAIN either |
| INCOMPLETE | the missing re-call is run; nothing else changes |

**Why these.**
- **A GAIN with a LOSS** is ztop50's old trade. The scale is per pool, and c says what kind of pool it is: metaqa's big
  KB pools against the passage pools, the cue ztop50's loss pointed at. Each model learns where the cut helps and falls
  back to zret's reference elsewhere. c is computed from the pool's rows and rrf, so the MLP rule holds.
- **No GAIN:** the top-50 reference has then run beside the z-scores (A2, A2h) and in their place (A2xm) on the lean
  pair. The adopted arms are the last models it has not met, so it closes as an input only after them.

No model change is adopted on today's pools (the user's rule of 13 October): an adopt is refitted on P1g-b's pools
before its full run.

## 6. How it runs

**Self-test.** `python outputs/mp_unified/za2xm.py --selftest` passes before this commit (5.1 s). It runs on toy carves
with the KT input, rmatch's chains and zlink's links: 16 pools of 3 to 9 rows, 1 to 8 of them retrieved. It checks:
- **the arms:** the two MROs, with ztop50's forward just above zret's; the arm classes have no body; the base arms,
  zret's and ztop50's arms unchanged; K = 50;
- **the reference:** top_ref against a hand computation at K 50, 3 and 1. On the toy, the top 50 are all the retrieved
  rows. At K 3, a pool with more than 3 retrieved rows keeps its 3 best by rrf, and the others keep all of theirs;
- **the MLP rule,** spied in each arm's forward at K 3:
  - top_ref runs once, on the batch's rrf column itself (no gradient);
  - all three z-scores (the rank block, SEMB, rrf's base term) take that reference;
  - zret's whole retrieved reference is never formed (replaced, not beside);
  - random weights leave the reference unchanged;
  - in zrckpxm, every within-pool normalisation takes an input column and none takes the score;
  - zspkxm's KT head and propagation z-score the score over the whole pool, as zspk's do;
- **each arm against its base,** with every head's last layer set:
  - at K 50: no parameter added, no random number drawn, and bit for bit the base's scores (eval and training) and
    every gradient;
  - at K 3: bit for bit the base model with zret's reference replaced by the top-3 one, scores and every gradient, so
    the reference is all that differs;
  - l1's raw and keep columns are the base's; its z-score columns and the scores move only on the pools with more than
    3 retrieved rows;
- **toy fits** through lean_gpu's loop: at K 50 the base's fit bit for bit; a repeat bit for bit; at K 3 different;
- **the refusals:** train refuses another arm (zrckp, ztop50), another split, another arm's or split's fit name, and
  another cache root or device; read refuses a fit that is not A2xm's, not its arm's or not its split's;
- relz's pair and re-call per arm, stamped A2xm's;
- the verdict's outcomes and their next forms: adopt, back-off, no GAIN, and five kinds of INCOMPLETE (missing,
  another stage's, another arm's, another declared file's, eleven reads).

**Commands** (each on the host through `retry_cmd.py` and `cuda_alloc.py`, as A2h's):

    python outputs/mp_unified/za2xm.py train --split L-musique --name scr-a2xm-zrckpxm --arm zrckpxm --device cuda \
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za2xm.py read --name scr-a2xm-zrckpxm --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za2xm.py compare --new outputs/screen/fits/scr-a2xm-zrckpxm \
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a2xm-zrckpxm
    python outputs/mp_unified/za2xm.py pair --arm zrckpxm \
        --screens outputs/screen/scr-a2xm-zrckpxm.json,outputs/screen/scr-a2xm-zrckpxm-hp.json \
        --out outputs/screen/scr-a2xm-zrckpxm-pair
    python outputs/mp_unified/za2xm.py recall --arm zrckpxm --null N1,N2,N3,N4 \
        --pair outputs/screen/scr-a2xm-zrckpxm-pair.json --out outputs/screen/scr-a2xm-zrckpxm-pair-recall
    python outputs/mp_unified/za2xm.py verdict

N1 to N4 are the four null files (`outputs/screen/scr-null-s1.json`, `-s2`, `-s1-hp`, `-s2-hp`). The same commands run
for L-hotpotqa (`--split L-hotpotqa`, names `-hp`, base scr-p1c3p-zrckp-hp and step 1's L-hotpotqa) and for zspkxm
(base scr-p1c3-zspk and -hp).

**Host items** (17, `a2xm-*`):
- **trains:** 4, with no build to wait on, so they start when the card has room. They run at A2h's fractions
  (L-musique 0.20, L-hotpotqa 0.27, GPU share +0.02) and memory (4.5 GB; 7 for zspkxm's L-hotpotqa fit). The arms read
  their bases' inputs and add nothing, so they need no more than A2h's top-50 half, which added 150 columns;
- **reads:** 4 at 0.27, share 0.29, 6 GB;
- **the call:** 4 compares, 2 pairs, 2 re-calls (after L1's base screens) and the verdict.

**Disk:** about 0.3 GB of fits and reads (A2h's 0.6 GB was four arms).

## 7. Results

Not run yet.
