# A5: zck's kind head on both models, its gradient kept off the shared weights

Declared 17 October 2026, before any number. The fifth reopening of docs/VERDICT_AUDIT_2026_10_17.md (section 2,
row 5), under docs/SCREENS.md section 3b. Code: `outputs/mp_unified/za5kind.py`. Filed 18 October: both arms NOT_ADOPTED (section 7).

## 1. Why this, now

zck (round twenty-eight, 9 October) put zkind's head on zrc's entries. The head is a learned offset on zret's output
layer and on rrf's base weight, acting only on a batch from a graph with typed relations (a batch with chain entries:
metaqa and webqsp). Its screen, against zrc's card fits, was MIXED under the seed null:

| read | L-musique's fit | L-hotpotqa's fit |
| --- | --- | --- |
| webqsp (zero-shot) | **+0.0160 GAIN** [+0.0042, +0.0270] | **+0.0402 GAIN** [+0.0257, +0.0557] |
| 2wiki | −0.0041 WITHIN | **−0.0082 LOSS** [−0.0103, −0.0061] |
| every other read | WITHIN (−0.0032 to +0.0039) | WITHIN |

- **The LOSS did not come from the head.** 2wiki has no chain entries, so the offsets never act on its batches. The
  shift came through the weights both kinds of graph share in joint training (SCREENS.md, round 28's result).
- **What round 28 asked for:** "a narrower screen that keeps the webqsp gain without the 2wiki loss would be declared
  here". None was declared. It never ran on zret or the KT input, and never reached the GNN. Under section 3b the idea
  is still open.

## 2. The form

On a typed batch the score is the model's score plus the head:

    score = s + kd_base · z(rrf) + kd_w · h + kd_b

- **s** is the arm's base model's score (zrckp's, or zspk's after its propagation).
- **h** is zret's last hidden units, detached; **z(rrf)** is zret's own base term (rrf's z-score against the retrieved
  rows). A5Stash, just above zret, keeps both from zret's own forward; zret's score is unchanged.
- **kd_w, kd_b, kd_base** have zck's offsets' shapes (128 + 1 + 1 = 130 parameters per model) and start at 0. They draw
  nothing, so from the same seed each arm's trunk is its base's.
- **Any other batch's score is s.**

Training minimises, on a typed batch,

    L(s) + L(s.detach() + head)

with L lean_gpu's listwise loss; on any other batch, L(s).
- The first term is the base model's loss on the base model's score, so **every shared weight gets the base model's
  gradient bit for bit** (zret, the KT head, rmatch's chain term, zsp's propagation). lean_gpu's loop clips nothing and
  Adam steps each parameter on its own, so the trunk's fit is the base's fit.
- The second term trains the head alone.
- So **every read of a graph without typed relations is the base model's** (up to the card's training noise between
  two fits), and the typed graphs' reads move only by the head. Section 1's LOSS has no path left.
- The logged training loss is the sum of both terms: twice the base's at the start, where the head is 0.

**Where the head goes: the top of each model.** On zspk it acts after the propagation, as a back-off on the final
score. Placed below the propagation, its offsets would spread to neighbours through the score zsp propagates.

**The MLP rule.** The head reads zret's hidden units and an input column; no score, nothing normalised over the
question's scores. The graph's kind is the graph's own property (whether its batches carry chain entries).

**What trains it.** webqsp never trains, so the head is fitted on metaqa's batches in both splits (L-musique and
L-hotpotqa) and read on metaqa in domain and on webqsp zero-shot.

## 3. The arms

| arm | model | carve |
| --- | --- | --- |
| zrckpk | zrckp's model with the kind head (MRO: A5Top, ZRCKP, KTMixP, ZRM, ChainMatch, A5Stash, ZRet) | zrckp's (`zp1c3.KTChainCarveZRC`) |
| zspkk | zspk's model with the kind head (MRO: A5Top, ZSPK, ZProp, KTMix, ZRM, ChainMatch, A5Stash, ZRet) | zspk's (`zp1c3.KTLinkCarveBase`) |

## 4. The fits and the call

- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-a5-<arm>`, `-hp`), on P1c2's roots with
  P1c3's KT input. These use the card fits' settings: variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128.
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split, which decides:
  - zrckp: scr-p1c3p-zrckp and -hp;
  - zspk: scr-p1c3-zspk and -hp.

  Step 1's fit is reported beside. The re-call's base screens are L1's.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), per arm. GAIN,
  LOSS and WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A5's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- metaqa and webqsp, both splits and arms (the head's graphs);
- 2wiki, both splits (zck's LOSS);
- the largest |ΔR@5| on the four graphs without typed relations: 0 by construction, up to the card's training noise;
- musique in domain (L-hotpotqa), both arms: Claim 1's read;
- hit@1 and FC@5 per read;
- per fit, the number of typed and untyped training forwards.

**Serving cost:** one 128-long product and two adds per row, on typed graphs only.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools, then its full run |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A5w, the narrow head: the kind's own level and base weight only (kd_b and kd_base, two parameters per model), the shared weights kept off as here |
| NOT_ADOPTED, with no GAIN | A5c, zck's coupled form on the lean pair: the offsets inside the output layer and base weight, the shared weights trained jointly as zck was, with 2wiki in domain beside as its known risk |
| INCOMPLETE | the missing re-call is run; nothing else changes |

- **If both arms pass**, the kind head joins both lean models.
- **If only one passes**, its full run is declared. The next file decides whether F1's rule needs the other model to
  carry the head too.
- **What does not happen.** No model change is adopted on today's pools (the user's rule of 13 October).

## 6. How it runs

**Self-test.** `python outputs/mp_unified/za5kind.py --selftest` passes before this commit (3.9 s). It checks:
- the arms' MROs, with A5Top at the top and A5Stash just above zret; the base arms unchanged; the head's parameter
  names are zck's;
- each arm against its base:
  - from the same seed, the base's state plus three zero parameters, with no draw;
  - at zero, the base bit for bit, typed or untyped, eval or train;
  - set, a typed batch is the base plus the head computed by hand from zret's own h and z(rrf); an untyped batch
    (zkind's `Untyped` carve) is the base bit for bit;
  - under A5's loss, every shared weight's gradient is the base's bit for bit on a typed and an untyped batch, and only
    the typed batch gives the head a gradient. Without A5's loss the shared weights' gradients differ (the check is
    live);
- toy fits through lean_gpu's loop under A5's loss: every shared weight of every epoch's state and of the SWA state is
  the base fit's bit for bit, the head moves, and a repeat is bit-identical;
- the refusals; relz's pair and re-call per arm, stamped A5's; the verdict's outcomes and next forms.

**Host items** (17, `a5-*`) go just after A4's verdict. They are sized as A3's and A4's are, from the measured peaks
of the base arms (the head adds 130 parameters):
- L-musique trains: `--frac 0.20`, GPU share 0.22;
- -hp trains and the reads: `--frac 0.27`, GPU share 0.29;
- memory: trains 4.5 GB (zspkk-hp 7 GB), reads 6 GB, compares 4 GB, pairs, re-calls and the verdict 2 GB.

## 7. Results

Filed 18 October 2026 from the host's records (`outputs/a5/screen-verdict.json`, written 2026-10-17T18:38:49Z; the
re-calls `outputs/screen/scr-a5-zrckpk-pair-recall.json` and `-zspkk-`, host paths made relative). **Both arms
NOT_ADOPTED, with no GAIN and no LOSS.**

- **On the four passage graphs nothing moves** (+0.0000 on every read, both arms), as the form requires: the kind
  head reads only typed relations and its gradient is kept off the shared weights.
- **On the KBs:** metaqa in domain falls 0.0003 to 0.0019; webqsp zero-shot rises on L-hotpotqa (+0.0176 MLP,
  +0.0136 GNN), filed GAIN and re-called WITHIN its 0.023 floor, and moves -0.0005 and +0.0066 on L-musique.

So a kind head that cannot change the shared weights reaches the KBs but does not clear the seed null. This closes
only this form. The next form, fixed in section 5 for no GAIN: **A5c**, zck's coupled form on the lean pair (the
offsets inside the output layer and base weight, trained with the shared weights), 2wiki in domain beside as its
known risk, in its own declared file.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's; base R@5 shown as zrckp's /
zspk's):

| split | dataset | read | base R@5 | zrckpk: delta [95% CI], call | zspkk: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8019 / 0.8009 | -0.0019 [-0.0025, -0.0014], WITHIN | -0.0008 [-0.0012, -0.0004], WITHIN |
| L-musique | squad | in-domain | 0.9116 / 0.9114 | +0.0000 [+0.0000, +0.0000], WITHIN | +0.0000 [+0.0000, +0.0000], WITHIN |
| L-musique | musique | zero-shot | 0.5121 / 0.5012 | +0.0000 [+0.0000, +0.0000], WITHIN | +0.0000 [+0.0000, +0.0000], WITHIN |
| L-musique | hotpotqa | in-domain | 0.9065 / 0.9080 | +0.0000 [+0.0000, +0.0000], WITHIN | +0.0000 [+0.0000, +0.0000], WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 / 0.9383 | +0.0000 [+0.0000, +0.0000], WITHIN | +0.0000 [+0.0000, +0.0000], WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 / 0.3229 | -0.0005 [-0.0055, +0.0042], WITHIN | +0.0066 [+0.0024, +0.0112], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8044 / 0.8040 | -0.0005 [-0.0009, -0.0002], WITHIN | -0.0003 [-0.0006, +0.0000], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 / 0.9135 | +0.0000 [+0.0000, +0.0000], WITHIN | +0.0000 [+0.0000, +0.0000], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 / 0.5754 | +0.0000 [+0.0000, +0.0000], WITHIN | +0.0000 [+0.0000, +0.0000], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 / 0.8499 | +0.0000 [+0.0000, +0.0000], WITHIN | +0.0000 [+0.0000, +0.0000], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 / 0.9333 | +0.0000 [+0.0000, +0.0000], WITHIN | +0.0000 [+0.0000, +0.0000], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 / 0.2680 | +0.0176 [+0.0099, +0.0252], WITHIN (filed GAIN) | +0.0136 [+0.0089, +0.0191], WITHIN (filed GAIN) |
