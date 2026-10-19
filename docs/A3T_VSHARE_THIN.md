# A3t: the share augmentation at vshare's thin range, in both models' training

Declared 18 October 2026, before any number. The next form A3's verdict named for both arms (no GAIN;
docs/A3_VSHARE_LEAN.md, sections 5 and 7), under docs/SCREENS.md section 3b and docs/VERDICT_AUDIT_2026_10_17.md.
Code: `outputs/mp_unified/za3t.py`. Filed 19 October: both arms NOT_ADOPTED (section 7).

## 1. Why this, now

vshare (8 October, `lean_screen7.py`) reweighted a training pool's statistics on step 1's model. Its full run gained
musique zero-shot +0.0891, metaqa zero-shot +0.0180 and squad zero-shot +0.0120, and was not adopted on one in-domain
LOSS (metaqa, −0.0094 against a floor of 0.0075).

**A3** (17 October) moved the share into zret's reference, the reference both of today's models take every z-score
against. A drawn training pool's retrieved rows held a share t in [0.55, 0.95] of the reference's weight. Both arms
were NOT_ADOPTED with no GAIN:
- **zrckpv:** 0 GAIN, 0 LOSS;
- **zspkv:** 0 GAIN, 1 LOSS (webqsp zero-shot −0.0103, against a floor of 0.0102);
- **musique zero-shot rose on both,** within its floor: +0.0095 (MLP) and +0.0186 (GNN).

That closes only A3's form. A3's range kept retrieval the majority of the reference, so the training statistics
stayed near the read's. vshare's other range, t in [0.08, 0.20], makes the reference mostly the pool's other rows,
the side step 1's whole-pool statistics sat on. A3's section 5 fixed that range as the next form for no GAIN, on both
models.

## 2. The form

A3's form with one change, the range:

- **When:** in a training forward only, for each question with at least two retrieved rows and at least one other
  row, with probability 0.5.
- **The draw:** t ~ U[0.08, 0.20].
- **The reference:** the retrieved rows weigh 1 and the other rows weigh r(1 − t) / (t u), so the retrieved rows hold
  a share t of the reference (r retrieved rows, u other rows).
- **What changes:** every z-score zret takes (each block's, SEMB's, and rrf's base term) is taken against this
  reference.
- **Reads** are never reweighted: t = 1, zret's reference.

**What the range means.** At t equal to a pool's own retrieved share r / (r + u), every row weighs 1: the reference is
the whole pool, step 1's. Below that share the other rows weigh more than the whole pool gives them. So on a pool
whose own share is low, the thin range sits near its whole-pool statistics; on a pool where retrieval is most of the
pool, it moves the reference far toward the rows retrieval left out.

**The same draws as A3.** The draws come from A3's generator, the model's own at A3's seed (seed + 5801), made with the
model, so each fit restarts it. A question A3 reweighted, A3t reweights, with its t moved linearly from A3's range to
this one. The two screens differ only in the range. The global stream, dropout's masks included, is the base's.

**Worked example.** A3's six rows: one column, dense_cos, with v1 to v4 retrieved (own share 4/6 = 0.667). At t = 0.14,
b = 4 × 0.86 / (0.14 × 2) = 12.29.

| row | retrieved | dense_cos | z (zret's reference) | z (A3, t = 0.75) | z (A3t, t = 0.14) | z (whole pool) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v1 | yes | 0.81 | 1.11 | 1.17 | 3.14 | 1.27 |
| v2 | yes | 0.78 | 0.83 | 1.00 | 2.91 | 1.10 |
| v3 | yes | 0.62 | −0.65 | 0.08 | 1.68 | 0.23 |
| v4 | yes | 0.55 | −1.29 | −0.32 | 1.14 | −0.15 |
| v5 | no | 0.40 | −2.68 | −1.18 | −0.01 | −0.98 |
| v6 | no | 0.31 | −3.51 | −1.70 | −0.71 | −1.47 |

A3t's reference has mean 0.402 and sd 0.130, near the two rows retrieval left out: the retrieved rows read as far
above them.

**Nothing else changes.** No row is added or removed, the raw inputs and the listwise softmax are the pool's own, and
there is no new parameter. Pools with one retrieved row or no other row (squad's) are never reweighted.

**The MLP rule.** A3t reads no score and normalises nothing over the question's scores. It changes, in training only,
the reference of within-pool statistics of input columns.

## 3. The arms

| arm | model | carve |
| --- | --- | --- |
| zrckpu | zrckp's model with A3TMix just above A3Mix (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, A3TMix, A3Mix, ZRet) | zrckp's (`zp1c3.KTChainCarveZRC`) |
| zspku | zspk's model with A3TMix in the same place (MRO: ZSPK, ZProp, KTMix, ZRM, ChainMatch, A3TMix, A3Mix, ZRet); zsp's propagation reads the score A3t changes | zspk's (`zp1c3.KTLinkCarveBase`) |

A3TMix keeps A3Mix's generator and draws, and replaces only the range in its training forward. A3's arms (zrckpv,
zspkv) are unchanged. No build is needed.

## 4. The fits and the call

A3's, unchanged, per arm:
- **Fits:** one card fit on L-musique and one on L-hotpotqa (`scr-a3t-<arm>`, `-hp`), on P1c2's roots with P1c3's KT
  input, at the card fits' settings (variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split (zrckp: scr-p1c3p-zrckp and -hp; zspk:
  scr-p1c3-zspk and -hp), with step 1's fit beside. The re-call's base screens are L1's.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md).

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A3t's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- metaqa (vshare's one LOSS), musique (vshare's largest gain, and Claim 1's read in domain), squad and webqsp, both
  splits and arms;
- hit@1 and FC@5 per read;
- per fit, the share of training questions reweighted and the mean t (stamped in each fit's screen.json);
- A3's deltas, set beside each read when the results are filed.

**Serving cost: none.** Reads are zret's forward, unchanged.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools, then its full run |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A3ts, the back-off form: A3t with the reference's retrieved share t as an input column (1 at every read), so the model learns what the thinning changes and reads the unweighted reference as a case of its own; on both models |
| NOT_ADOPTED, with no GAIN | A3v, vshare's two-sided rule in zret's reference: a drawn pool whose own retrieved share is under 0.375 takes t in [0.55, 0.95], any other pool t in [0.08, 0.20], on both models. If A3v finds no GAIN either, the share idea goes on as an input before it closes: A3z, each block's and rrf's whole-pool z-score beside zret's, through zero-started columns, on both models |
| INCOMPLETE | the missing re-call is run; nothing else changes |

**Why A3v next on no GAIN.** A3 and A3t each apply one of vshare's ranges to every drawn pool. vshare chose the range
by the pool's own share, so that the share stopped telling the model which kind of graph a pool came from. A3v is that
rule exactly, in the reference today's models read. A training augmentation that finds no GAIN at either range or
under the two-sided rule has not carried the signal; the input form then tests whether the whole-pool reference itself
carries it.

No model change is adopted on today's pools (the user's rule of 13 October): an adopt is refitted on P1g-b's pools
before its full run.

## 6. How it runs

**Self-test.** `python outputs/mp_unified/za3t.py --selftest` passes before this commit (11.7 s). It checks:
- **the arms:** the MROs, with A3TMix just above A3Mix in both models; the base arms and A3's arms unchanged; the
  range is vshare's THIN, read from `lean_screen7`, which this file never runs; A3's own range unchanged;
- **the weights** on A3's hand-made batch of five pools:
  - an applied pool's retrieved rows hold exactly t, now in [0.08, 0.20];
  - the applied pools are A3's on the same draws, and t is A3's moved linearly;
  - the other rows weigh more than under A3;
  - pools not drawn, with one retrieved row, or with no other row keep zret's reference;
  - z-scores against the weights match A3's float64 reference by hand, fallbacks included;
  - t at a pool's own share weighs every row 1, the whole pool's z-scores;
- **each arm** against its base and A3's arm, from one seed:
  - the three states are one, and A3t's generator is A3's;
  - in eval mode bit for bit, and a read draws nothing;
  - in training at P_APPLY 0 bit for bit: scores, every gradient, and the global stream;
  - applied, the scores move on the eligible questions only, equal zret's forward on A3t's reference (zspku through
    its propagation), and differ from A3's arm on the same draws;
- **toy fits** through lean_gpu's loop repeat bit for bit, equal the base arm's fit at P_APPLY 0, reweight exactly the
  questions A3's arm does with a mean t in the thin range, and differ from the base's and A3's fits;
- **the refusals:** train refuses another arm (A3's included), split, fit name, cache root or device; read refuses a fit
  that is not A3t's or not its split's;
- relz's pair and re-call per arm, stamped A3t's, with relz restored after; the verdict's outcomes and next forms.

**Host items** (17, `a3t-*`), after A2h's:
- **trains:** 4, with no build to wait on, at A3's fractions (L-musique 0.20, L-hotpotqa 0.27, GPU share +0.02);
- **reads:** 4 at 0.27, share 0.29;
- **the call:** 4 compares, 2 pairs, 2 re-calls and the verdict;
- **memory:** A3's items (trains 4.5 GB, zspku-hp 7 GB; reads 6 GB; compares 4 GB; the rest 2 GB).

**Disk:** about 0.3 GB of fits and reads.

## 7. Results

Filed 19 October 2026 from the host's records (`outputs/a3t/screen-verdict.json`, written 2026-10-19T04:11:14Z; the
re-calls `outputs/screen/scr-a3t-zrckpu-pair-recall.json` and `-zspku-`, host paths made relative). **Both arms
NOT_ADOPTED:** no GAIN and no LOSS after the seed-null re-call.

- **zrckpu: 0 GAIN, 0 LOSS.** On L-hotpotqa, hotpotqa zero-shot rises +0.0046 [+0.0014, +0.0078] and 2wiki in domain
  falls 0.0052, both WITHIN; webqsp zero-shot falls 0.0142 there [-0.0252, -0.0034], filed LOSS and re-called WITHIN
  against its 0.0233 floor.
- **zspku: 0 GAIN, 0 LOSS.** musique zero-shot rises +0.0083 [+0.0027, +0.0142] on L-musique, filed GAIN and re-called
  WITHIN against its 0.072 floor; webqsp zero-shot rises +0.0088 on L-hotpotqa, WITHIN.
- **Beside:** metaqa in domain (vshare's one LOSS) moves +0.0001 and -0.0004 (MLP), +0.0020 and +0.0002 (GNN); musique
  in domain (L-hotpotqa, Claim 1's read) -0.0024 and -0.0007; squad within 0.002 on all four fits. A3, at t in [0.55,
  0.95], had moved musique zero-shot +0.0095 and +0.0186.
- **The fits:** on both splits about three quarters of the question visits were eligible and half of those were
  reweighted, so 0.37 to 0.38 of all visits were; the mean t is 0.140, the middle of [0.08, 0.20] (table below).

The thin range moves no read past the seed null on either model; A3's range had gained nowhere either. This closes only
this form (A3's draws at t in [0.08, 0.20]). The next form, fixed in section 5 and recorded in the verdict, is **A3v**,
vshare's two-sided rule in zret's reference: a drawn pool whose own retrieved share is under 0.375 takes t in [0.55,
0.95], any other pool t in [0.08, 0.20], on both models, in its own declared file. If A3v finds no GAIN either, the
share idea goes on as an input before it closes: **A3z**, each block's and rrf's whole-pool z-score beside zret's,
through zero-started columns, on both models.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's; a call is shown with its
filed call where the re-call changed it; base R@5 shown as zrckp's / zspk's):

| split | dataset | read | base R@5 | zrckpu: delta [95% CI], call | zspku: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8019 / 0.8009 | +0.0001 [-0.0007, +0.0009], WITHIN | +0.0020 [+0.0012, +0.0030], WITHIN |
| L-musique | squad | in-domain | 0.9116 / 0.9114 | -0.0005 [-0.0024, +0.0014], WITHIN | +0.0018 [-0.0003, +0.0037], WITHIN |
| L-musique | musique | zero-shot | 0.5121 / 0.5012 | +0.0017 [-0.0036, +0.0072], WITHIN | +0.0083 [+0.0027, +0.0142], WITHIN (filed GAIN) |
| L-musique | hotpotqa | in-domain | 0.9065 / 0.9080 | -0.0014 [-0.0034, +0.0007], WITHIN | -0.0017 [-0.0038, +0.0005], WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 / 0.9383 | +0.0026 [+0.0012, +0.0040], WITHIN | -0.0019 [-0.0033, -0.0005], WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 / 0.3229 | -0.0084 [-0.0188, +0.0021], WITHIN | -0.0027 [-0.0127, +0.0077], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8044 / 0.8040 | -0.0004 [-0.0013, +0.0004], WITHIN | +0.0002 [-0.0007, +0.0012], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 / 0.9135 | -0.0003 [-0.0024, +0.0017], WITHIN | -0.0015 [-0.0035, +0.0004], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 / 0.5754 | -0.0024 [-0.0073, +0.0025], WITHIN | -0.0007 [-0.0063, +0.0049], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 / 0.8499 | +0.0046 [+0.0014, +0.0078], WITHIN | +0.0048 [+0.0018, +0.0078], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 / 0.9333 | -0.0052 [-0.0069, -0.0034], WITHIN | +0.0029 [+0.0014, +0.0044], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 / 0.2680 | -0.0142 [-0.0252, -0.0034], WITHIN (filed LOSS) | +0.0088 [-0.0012, +0.0184], WITHIN |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a3t-zrckpu.json` and the others, `-hp` for L-hotpotqa); and zrckpv's R@5 delta (A3, its own
re-call) and zspkv's R@5 delta (A3, its own re-call), on the same read:

| split | dataset | zrckpu FC@5 | zrckpu hit@1 | zspku FC@5 | zspku hit@1 | zrckpv R@5 (A3) | zspkv R@5 (A3) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L-musique | metaqa | -0.0002 | -0.0001 | +0.0029 | +0.0092 | -0.0006 | +0.0007 |
| L-musique | squad | -0.0005 | -0.0024 | +0.0018 | -0.0019 | -0.0008 | -0.0013 |
| L-musique | musique | +0.0145 | +0.0165 | +0.0165 | +0.0352 | +0.0095 | +0.0186 |
| L-musique | hotpotqa | -0.0003 | -0.0117 | -0.0049 | -0.0004 | -0.0007 | -0.0003 |
| L-musique | 2wiki | +0.0078 | -0.0036 | -0.0051 | +0.0004 | +0.0014 | -0.0038 |
| L-musique | webqsp | -0.0153 | +0.0113 | -0.0020 | -0.0033 | +0.0030 | -0.0103 LOSS |
| L-hotpotqa | metaqa | -0.0003 | +0.0013 | +0.0004 | -0.0007 | -0.0009 | -0.0018 |
| L-hotpotqa | squad | -0.0003 | +0.0001 | -0.0015 | -0.0023 | -0.0009 | -0.0014 |
| L-hotpotqa | musique | +0.0000 | -0.0033 | -0.0054 | -0.0116 | +0.0013 | +0.0072 |
| L-hotpotqa | hotpotqa | +0.0092 | -0.0042 | +0.0103 | -0.0012 | -0.0035 | -0.0009 |
| L-hotpotqa | 2wiki | -0.0149 | -0.0049 | +0.0087 | -0.0040 | -0.0000 | -0.0014 |
| L-hotpotqa | webqsp | -0.0120 | +0.0113 | +0.0086 | -0.0053 | -0.0128 | -0.0086 |

The fits (stamped in each fit's screen.json; the two arms drew the same questions and t's, A3's generator):

| fit | question visits (8 epochs) | eligible | reweighted | share reweighted | mean t |
| --- | ---: | ---: | ---: | ---: | ---: |
| L-musique | 189,392 | 142,056 | 71,020 | 0.375 | 0.1402 |
| L-hotpotqa | 178,760 | 131,896 | 65,875 | 0.369 | 0.1403 |
