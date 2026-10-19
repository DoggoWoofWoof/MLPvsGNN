# A3v: vshare's two-sided share rule in zret's reference, in both models' training

Declared 19 October 2026, before any number. The next form A3t's verdict named for both arms (no GAIN;
docs/A3T_VSHARE_THIN.md, section 5; outputs/a3t/screen-verdict.json), under docs/SCREENS.md section 3b and
docs/VERDICT_AUDIT_2026_10_17.md. Code: `outputs/mp_unified/za3v.py`. Filed 19 October: zrckpy ADOPT_FOR_FULL_RUN, zspky
NOT_ADOPTED; next forms: zrckpy refitted on P1g-b's pools, and A3z (section 7).

## 1. Why this, now

**A3** (17 October) and **A3t** (18 October) each gave every drawn training pool one of vshare's two ranges. A drawn
pool's retrieved rows held a share t of zret's reference: t in [0.55, 0.95] under A3, t in [0.08, 0.20] under A3t, on
the same draws. All four arms were NOT_ADOPTED with no GAIN after the seed-null re-call. A3's zspkv kept one LOSS;
A3t's two arms had none (outputs/a3t/screen-verdict.json).

**The reads that moved.** These are A3t's reads whose 95% interval excludes 0 on either arm, with A3's beside on the
same reads. Each is the R@5 delta against the base arm's card fit, from the re-call records
(`outputs/screen/scr-a3t-<arm>-pair-recall.json`, `scr-a3-<arm>-pair-recall.json`):

| read | floor | zrckpu (A3t) | zspku (A3t) | zrckpv (A3) | zspkv (A3) |
| --- | ---: | --- | --- | --- | --- |
| L-musique, metaqa in domain | 0.0075 | +0.0001 [−0.0007, +0.0009] | +0.0020 [+0.0012, +0.0030] | −0.0006 [−0.0015, +0.0002] | +0.0007 [−0.0002, +0.0016] |
| L-musique, musique zero-shot | 0.0720 | +0.0017 [−0.0036, +0.0072] | +0.0083 [+0.0027, +0.0142] | +0.0095 [+0.0047, +0.0141] | +0.0186 [+0.0131, +0.0243] |
| L-musique, 2wiki in domain | 0.0075 | +0.0026 [+0.0012, +0.0040] | −0.0019 [−0.0033, −0.0005] | +0.0014 [+0.0000, +0.0028] | −0.0038 [−0.0052, −0.0024] |
| L-hotpotqa, hotpotqa zero-shot | 0.0200 | +0.0046 [+0.0014, +0.0078] | +0.0048 [+0.0018, +0.0078] | −0.0035 [−0.0065, −0.0005] | −0.0009 [−0.0039, +0.0020] |
| L-hotpotqa, 2wiki in domain | 0.0075 | −0.0052 [−0.0069, −0.0034] | +0.0029 [+0.0014, +0.0044] | −0.0000 [−0.0015, +0.0016] | −0.0014 [−0.0031, +0.0003] |
| L-hotpotqa, webqsp zero-shot | 0.0233 | −0.0142 [−0.0252, −0.0034] | +0.0088 [−0.0012, +0.0184] | −0.0128 [−0.0229, −0.0026] | −0.0086 [−0.0191, +0.0015] |

What the records show:
- **No read passed its floor upward.** The re-call moved filed calls to WITHIN: musique zero-shot's GAIN on zspku
  (and on both of A3's arms), and L-hotpotqa's webqsp zero-shot LOSS on zrckpu (and on zrckpv).
- **The rich range moved musique zero-shot most:** +0.0095 and +0.0186 under A3, against +0.0017 and +0.0083 under A3t.
  musique's pools are big: 2,092 rows a question on average on s1eval (docs/P1C2_POOLS_KIND_T.md).
- **The thin range moved hotpotqa zero-shot up on both models:** +0.0046 and +0.0048, both intervals above 0. The rich
  range gave −0.0035 and −0.0009 there. hotpotqa's pools are small: 94 rows a question.
- **A3's one LOSS moved back:** L-musique's webqsp zero-shot on the GNN was −0.0103 [−0.0198, −0.0012] under A3 (floor
  0.0102) and −0.0027 [−0.0127, +0.0077] under A3t.
- 2wiki in domain moved in opposite directions on the two models, under both ranges, inside its floor.

**vshare's rule.** vshare (8 October, `lean_screen7.py`) never gave every pool one range. It chose the range by the
pool's own share: [0.55, 0.95] when the pool's ranked share is under 0.375, [0.08, 0.20] otherwise. Its measured
shares were 0.08 to 0.20 on the big pools and 0.56 to 0.69 on the small passage pools (docs/SCREENS.md, vshare's row).
So each kind of pool was seen in training at the other kind's share, and a ranked share no longer named the kind of
graph. A3v is that rule in the reference today's models read. It asks whether one fit keeps both moves: the rich range
on the big pools and the thin range on the small ones. A3t's section 5 fixed it as the next form for no GAIN, on both
models.

## 2. The form

A3's form, with vshare's rule for the range:

- **When:** in a training forward only, for each question with at least two retrieved rows (rrf > 0) and at least one
  other row, with probability 0.5. These are A3's draws.
- **The side:** the pool's own retrieved share s = r / (r + u), for r retrieved and u other rows, in float64.
  - s under 0.375: the rich side;
  - s at 0.375 or above: the thin side (a share of exactly 0.375 is thin).
- **The draw:** with v the question's uniform from A3's draws, t = 0.55 + 0.40 v on the rich side and
  t = 0.08 + 0.12 v on the thin side. So t ~ U[0.55, 0.95] or t ~ U[0.08, 0.20].
- **The reference:** the retrieved rows weigh 1 and the other rows weigh r(1 − t) / (t u), so the retrieved rows hold
  a share t of the reference.
- **What changes:** every z-score zret takes (each block's, SEMB's, and rrf's base term) is taken against this
  reference.
- **Reads** are never reweighted: t = 1, zret's reference.

**The constants.** Each is read from the code that set it, and the self-test asserts each:

| constant | value | from |
| --- | --- | --- |
| P_APPLY | 0.5 | vshare's `P_APPLY` (`lean_screen7.VShareCarve`); A3's and A3t's |
| RICH | [0.55, 0.95] | vshare's `RICH`; A3's range |
| THIN | [0.08, 0.20] | vshare's `THIN`; A3t's range |
| MID | 0.375 | vshare's `MID` |
| MIN_R | 2 retrieved rows | zret's floor; A3's |
| other rows | at least 1 | A3's |
| seed | the model's seed + 5801 | A3's generator, so A3's draws |

**How it differs from vshare.** vshare reweighted step 1's model, drew from its own seed (20261008), and took any pool
with a ranked and an unranked row. A3v keeps A3's eligibility, draws and reference. Only the choice of range is
vshare's.

**The same draws as A3 and A3t.** The draws come from A3's generator, the model's own at A3's seed, made with the
model, so each fit restarts it. A question A3 and A3t reweighted, A3v reweights. On a pool under 0.375 its weights and t
are A3's, bit for bit; on any other pool they are A3t's. So A3v differs from A3 only on the thin side's pools, and from
A3t only on the rich side's. The global stream, dropout's masks included, is the base's.

**Worked example.** A3's six rows, one column (dense_cos), at the draw v = 0.5, which gave A3's t = 0.75 and A3t's
t = 0.14.
- **v1 to v4 retrieved** (own share 4/6 = 0.667): the thin side. t = 0.14, the other rows weigh 12.29, and the
  z-scores are A3t's column in docs/A3T_VSHARE_THIN.md (mean 0.402, sd 0.130).
- **Only v1 and v2 retrieved** (own share 2/6 = 0.333): the rich side. t = 0.75, and the other rows weigh
  2 × 0.25 / (0.75 × 4) = 0.167, A3's weights:

| row | retrieved | dense_cos | z (zret's reference) | z (A3v = A3, t = 0.75) | z (A3t, t = 0.14) | z (whole pool) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v1 | yes | 0.81 | 1.00 | 0.63 | 1.84 | 1.27 |
| v2 | yes | 0.78 | −1.00 | 0.43 | 1.66 | 1.10 |
| v3 | no | 0.62 | −11.67 | −0.61 | 0.65 | 0.23 |
| v4 | no | 0.55 | −16.33 | −1.06 | 0.22 | −0.15 |
| v5 | no | 0.40 | −26.33 | −2.04 | −0.72 | −0.98 |
| v6 | no | 0.31 | −32.33 | −2.62 | −1.29 | −1.47 |

zret's reference here is two rows (mean 0.795, sd 0.015), so every other row reads as far below. A3v's reference
(mean 0.714, sd 0.154) lies between it and the whole pool (mean 0.578, sd 0.183). A3t's range would move this pool's
reference past the whole pool (mean 0.516), toward the rows retrieval left out. A3v leaves that to pools where
retrieval is most of the pool.

**Nothing else changes.** No row is added or removed, the raw inputs and the listwise softmax are the pool's own, and
there is no new parameter. Pools with one retrieved row or no other row (squad's) are never reweighted.

**The MLP rule.** A3v reads no score and normalises nothing over the question's scores. The side comes from the count
of the pool's retrieved rows, the flag zret's reference already uses. It changes, in training only, the reference of
within-pool statistics of input columns.

**On record per fit.** Today's pools are P1c2's, not the step 1 pools vshare measured. Each fit's screen.json stamps
the training questions eligible and reweighted on each side of 0.375 and each side's mean t, so the split each fit met
is on record. A fit whose reweighted pools all fell on one side would repeat A3's or A3t's fit (the self-test shows
both limits).

## 3. The arms

| arm | model | carve |
| --- | --- | --- |
| zrckpy | zrckp's model with A3VMix just above A3Mix (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, A3VMix, A3Mix, ZRet) | zrckp's (`zp1c3.KTChainCarveZRC`) |
| zspky | zspk's model with A3VMix in the same place (MRO: ZSPK, ZProp, KTMix, ZRM, ChainMatch, A3VMix, A3Mix, ZRet); zsp's propagation reads the score A3v changes | zspk's (`zp1c3.KTLinkCarveBase`) |

A3VMix keeps A3Mix's generator and draws, and replaces only the reference in its training forward. `two_sided_ref`
builds A3's weights (`za3vshare.share_ref` at RICH) and A3t's (at THIN) from the same uniforms, and keeps for each pool
the one its side names. A3's arms (zrckpv, zspkv) and A3t's (zrckpu, zspku) are unchanged. No build is needed.

## 4. The fits and the call

A3t's, unchanged, per arm:
- **Fits:** one card fit on L-musique and one on L-hotpotqa (`scr-a3v-<arm>`, `-hp`), on P1c2's roots with P1c3's KT
  input, at the card fits' settings (variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split (zrckp: scr-p1c3p-zrckp and -hp; zspk:
  scr-p1c3-zspk and -hp), with step 1's fit beside. The re-call's base screens are L1's.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md).

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A3v's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- metaqa (vshare's one LOSS), musique (vshare's largest gain, and Claim 1's read in domain), squad and webqsp, both
  splits and arms;
- hit@1 and FC@5 per read;
- per fit, the training questions eligible and reweighted on each side of 0.375, and each side's mean t (stamped in
  each fit's screen.json);
- A3's and A3t's deltas, set beside each read when the results are filed.

**Serving cost: none.** Reads are zret's forward, unchanged.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools, then its full run, its own declared file |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A3vs, the back-off form: A3v with the reference's retrieved share t as an input column (1 at every read), so the model learns what the two-sided reweighting changes and reads the unweighted reference as a case of its own; on both models, its own declared file |
| NOT_ADOPTED, with no GAIN | the share idea goes on as an input before it closes. A3z: each block's and rrf's whole-pool z-score beside zret's, zero-started columns, on both models; its own declared file |
| INCOMPLETE | the missing re-call is run; nothing else changes |

**Why A3z next on no GAIN.** A3, A3t and A3v put the share into zret's reference at each of vshare's ranges and under
its two-sided rule. If none of them finds a GAIN, the training augmentation has not carried the signal. The idea then
reaches both models as an input before it closes (SCREENS.md section 3b): the whole-pool z-scores, step 1's statistics,
beside zret's, through zero-started columns, so each model starts as its base and learns how much to read them. They
are computed from the rows' input columns, not from a score, so the MLP rule allows them. The share idea closes only
after A3z.

No model change is adopted on today's pools (the user's rule of 13 October): an adopt is refitted on P1g-b's pools
before its full run.

## 6. How it runs

**Self-test.** `python outputs/mp_unified/za3v.py --selftest` passes before this commit (6.0 s). It checks:
- **the arms:** the MROs, with A3VMix just above A3Mix in both models; the base arms, A3's arms and A3t's arms
  unchanged; the two ranges, the split at 0.375 and the probability are vshare's, read from `lean_screen7`, which this
  file never runs; A3's range is the rich side and A3t's the thin side; the A3z text is A3t's own, word for word;
- **the weights** on a hand-made batch of eight pools (A3's five and three more, one at a share of exactly 0.375):
  - the applied pools are A3's and A3t's on the same draws;
  - a pool under 0.375 takes A3's weights and t, any other A3t's, bit for bit; 0.375 itself is thin;
  - an applied pool's retrieved rows hold exactly t; the other rows weigh under 1 on the rich side and over 1 on the
    thin side;
  - pools not drawn, with one retrieved row, or with no other row keep zret's reference;
  - z-scores against the weights match A3's float64 reference by hand, fallbacks included;
  - a split of 1 gives A3's weights everywhere, a split of 0 A3t's;
- **each arm** against its base, A3's arm and A3t's arm, from one seed, on a toy carve with eligible pools on both
  sides:
  - the four states are one, and A3v's generator is A3's and A3t's;
  - in eval mode bit for bit, and a read draws nothing;
  - in training at P_APPLY 0 bit for bit: scores, every gradient, and the global stream;
  - applied, the global stream is the base's; the scores move on the eligible questions only and equal zret's forward
    on A3v's reference (zspky through its propagation); they are A3's arm's on the pools under 0.375 and A3t's arm's
    on the others, bit for bit, and differ from each on the other side;
- **toy fits** through lean_gpu's loop:
  - they repeat bit for bit;
  - they reweight exactly the questions A3's arm does, on both sides, each side's mean t in its range;
  - they equal the base arm's fit at P_APPLY 0, A3's arm's fit at a split of 1, and A3t's at a split of 0;
  - they differ from all three at 0.375;
- **the refusals:** train refuses another arm (A3's and A3t's included), split, fit name, cache root or device; read
  refuses a fit that is not A3v's or not its split's;
- relz's pair and re-call per arm, stamped A3v's, with relz restored after; the verdict's outcomes and next forms.

**The commands**, for zrckpy's L-musique fit. zspky's are the same against scr-p1c3-zspk. The `-hp` fits take
`--split L-hotpotqa` and step 1's L-hotpotqa fit beside.

```
python outputs/mp_unified/za3v.py train --split L-musique --name scr-a3v-zrckpy --arm zrckpy --device cuda --host --cache-root outputs/p1c2/cache
python outputs/mp_unified/za3v.py read --name scr-a3v-zrckpy --device cuda --host --cache-root outputs/p1c2/cache
python outputs/mp_unified/za3v.py compare --new outputs/screen/fits/scr-a3v-zrckpy --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a3v-zrckpy
python outputs/mp_unified/za3v.py pair --arm zrckpy --screens outputs/screen/scr-a3v-zrckpy.json,outputs/screen/scr-a3v-zrckpy-hp.json --out outputs/screen/scr-a3v-zrckpy-pair
python outputs/mp_unified/za3v.py recall --arm zrckpy --null outputs/screen/scr-null-s1.json,outputs/screen/scr-null-s2.json,outputs/screen/scr-null-s1-hp.json,outputs/screen/scr-null-s2-hp.json --pair outputs/screen/scr-a3v-zrckpy-pair.json --out outputs/screen/scr-a3v-zrckpy-pair-recall
python outputs/mp_unified/za3v.py verdict
```

On the host, trains and reads run under `retry_cmd.py --cuda` and `cuda_alloc.py`, as A3t's do.

**Host items** (17, `a3v-*`):
- **trains:** 4, with no build to wait on, at A3's fractions (L-musique 0.20, L-hotpotqa 0.27, GPU share +0.02);
- **reads:** 4 at 0.27, share 0.29;
- **the call:** 4 compares, 2 pairs, 2 re-calls (after L1's base screens) and the verdict;
- **memory:** A3t's items (trains 4.5 GB, zspky-hp 7 GB; reads 6 GB; compares 4 GB; the rest 2 GB).

**Disk:** about 0.3 GB of fits and reads.

## 7. Results

Filed 19 October 2026 from the host's records (`outputs/a3v/screen-verdict.json`, written 2026-10-19T13:44:28Z; the
re-calls `outputs/screen/scr-a3v-zrckpy-pair-recall.json` and `-zspky-`, host paths made relative). **zrckpy
ADOPT_FOR_FULL_RUN, zspky NOT_ADOPTED:** the MLP gains webqsp zero-shot on L-musique with no LOSS; the GNN gains nothing
past the seed null.

- **zrckpy: 1 GAIN, 0 LOSS.** webqsp zero-shot on L-musique rises +0.0112 [+0.0008, +0.0211], just past its 0.0102
  floor. musique zero-shot on L-musique falls 0.0111 [-0.0167, -0.0054], WITHIN (filed LOSS) against its 0.072 floor,
  while its FC@5 moves +0.0054. hotpotqa zero-shot on L-hotpotqa rises +0.0038 [+0.0007, +0.0069], within its 0.02
  floor.
- **zspky: 0 GAIN, 0 LOSS.** musique zero-shot on L-musique rises +0.0243 [+0.0183, +0.0306], WITHIN (filed GAIN), with
  FC@5 +0.0310 and hit@1 +0.0240. webqsp zero-shot on L-hotpotqa rises +0.0087 [-0.0006, +0.0177], within its 0.0233
  floor; 2wiki in domain on L-musique moves -0.0041 [-0.0055, -0.0026].
- **Beside:** metaqa, vshare's one LOSS, stays within its floor on every read: +0.0003 and -0.0019 on the MLP, +0.0003
  and +0.0003 on the GNN. On the GNN, musique zero-shot rises further than under A3's rich range (+0.0186) or A3t's thin
  one (+0.0083); on the MLP it falls, where A3 (+0.0095) and A3t (+0.0017) had it rising. The MLP's GAIN read, webqsp
  zero-shot on L-musique, moved +0.0030 under A3 and -0.0084 under A3t. Claim 1's read, musique in domain on L-hotpotqa,
  moves +0.0004 (MLP) and -0.0029 (GNN).
- **The fits:** A3v reweights from A3's draws, as A3t did, and its counts are A3t's: 71,020 and 65,875 question visits
  (0.375 and 0.369 of them). Of those, 22,776 and 22,711 sit in pools whose retrieved share is under 0.375 and take A3's
  rich range (mean t 0.749 and 0.751); the others take A3t's thin one (mean t 0.140 and 0.140). Table below.

zrckpy adopts the form for its full run only. Its next form, fixed in section 5 and recorded in the verdict, is the arm
fitted again on P1g-b's pools, then its full run, each in its own declared file. zspky's next form is A3z: the share
idea as an input before it closes, each block's and rrf's whole-pool z-score beside zret's, through zero-started
columns, on both models, in its own declared file. The share idea closes on the GNN only after A3z.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's; a call is shown with its
filed call where the re-call changed it; base R@5 shown as zrckp's / zspk's):

| split | dataset | read | base R@5 | zrckpy: delta [95% CI], call | zspky: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8019 / 0.8009 | +0.0003 [-0.0006, +0.0011], WITHIN | +0.0003 [-0.0006, +0.0012], WITHIN |
| L-musique | squad | in-domain | 0.9116 / 0.9114 | +0.0015 [-0.0004, +0.0034], WITHIN | +0.0008 [-0.0012, +0.0026], WITHIN |
| L-musique | musique | zero-shot | 0.5121 / 0.5012 | -0.0111 [-0.0167, -0.0054], WITHIN (filed LOSS) | +0.0243 [+0.0183, +0.0306], WITHIN (filed GAIN) |
| L-musique | hotpotqa | in-domain | 0.9065 / 0.9080 | -0.0022 [-0.0043, -0.0002], WITHIN | -0.0034 [-0.0058, -0.0010], WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 / 0.9383 | +0.0014 [-0.0001, +0.0029], WITHIN | -0.0041 [-0.0055, -0.0026], WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 / 0.3229 | +0.0112 [+0.0008, +0.0211], **GAIN** | -0.0014 [-0.0100, +0.0072], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8044 / 0.8040 | -0.0019 [-0.0028, -0.0011], WITHIN | +0.0003 [-0.0005, +0.0013], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 / 0.9135 | +0.0003 [-0.0016, +0.0022], WITHIN | -0.0004 [-0.0024, +0.0015], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 / 0.5754 | +0.0004 [-0.0050, +0.0058], WITHIN | -0.0029 [-0.0081, +0.0027], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 / 0.8499 | +0.0038 [+0.0007, +0.0069], WITHIN | +0.0003 [-0.0024, +0.0030], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 / 0.9333 | -0.0013 [-0.0028, +0.0003], WITHIN | +0.0024 [+0.0009, +0.0039], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 / 0.2680 | +0.0036 [-0.0067, +0.0139], WITHIN | +0.0087 [-0.0006, +0.0177], WITHIN |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a3v-zrckpy.json` and the others, `-hp` for L-hotpotqa); and A3's (zrckpv, zspkv) and A3t's (zrckpu,
zspku) R@5 deltas, their own re-calls, on the same read:

| split | dataset | zrckpy FC@5 | zrckpy hit@1 | zspky FC@5 | zspky hit@1 | zrckpv R@5 (A3) | zspkv R@5 (A3) | zrckpu R@5 (A3t) | zspku R@5 (A3t) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L-musique | metaqa | +0.0020 | -0.0016 | +0.0004 | +0.0004 | -0.0006 | +0.0007 | +0.0001 | +0.0020 |
| L-musique | squad | +0.0015 | -0.0008 | +0.0008 | +0.0013 | -0.0008 | -0.0013 | -0.0005 | +0.0018 |
| L-musique | musique | +0.0054 | -0.0004 | +0.0310 | +0.0240 | +0.0095 | +0.0186 | +0.0017 | +0.0083 |
| L-musique | hotpotqa | -0.0028 | -0.0132 | -0.0073 | -0.0097 | -0.0007 | -0.0003 | -0.0014 | -0.0017 |
| L-musique | 2wiki | +0.0045 | -0.0026 | -0.0114 | -0.0025 | +0.0014 | -0.0038 | +0.0026 | -0.0019 |
| L-musique | webqsp | +0.0013 | +0.0160 | -0.0013 | +0.0000 | +0.0030 | -0.0103 LOSS | -0.0084 | -0.0027 |
| L-hotpotqa | metaqa | -0.0019 | -0.0047 | +0.0006 | -0.0006 | -0.0009 | -0.0018 | -0.0004 | +0.0002 |
| L-hotpotqa | squad | +0.0003 | +0.0004 | -0.0004 | -0.0029 | -0.0009 | -0.0014 | -0.0003 | -0.0015 |
| L-hotpotqa | musique | -0.0025 | -0.0079 | -0.0054 | -0.0116 | +0.0013 | +0.0072 | -0.0024 | -0.0007 |
| L-hotpotqa | hotpotqa | +0.0078 | -0.0117 | +0.0019 | +0.0035 | -0.0035 | -0.0009 | +0.0046 | +0.0048 |
| L-hotpotqa | 2wiki | -0.0018 | -0.0035 | +0.0090 | -0.0041 | -0.0000 | -0.0014 | -0.0052 | +0.0029 |
| L-hotpotqa | webqsp | +0.0027 | +0.0086 | +0.0086 | -0.0007 | -0.0128 | -0.0086 | -0.0142 | +0.0088 |

The fits (stamped in each fit's screen.json; the two arms drew the same questions and t's, A3's generator; a pool is
under 0.375 when its own retrieved share r / (r + u) is):

| fit | question visits (8 epochs) | eligible: under 0.375 / other | reweighted: rich / thin | share reweighted | mean t: rich / thin |
| --- | ---: | ---: | ---: | ---: | ---: |
| L-musique | 189,392 | 45,632 / 96,424 | 22,776 / 48,244 | 0.375 | 0.7491 / 0.1404 |
| L-hotpotqa | 178,760 | 45,632 / 86,264 | 22,711 / 43,164 | 0.369 | 0.7512 / 0.1402 |
