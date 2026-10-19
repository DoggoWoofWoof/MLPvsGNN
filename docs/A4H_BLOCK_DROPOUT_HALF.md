# A4h: block dropout at half the rate, in both models' training

Declared 18 October 2026, before any number. The back-off form A4's verdict named for zspkd
(docs/A4_BLOCK_DROPOUT_LEAN.md, sections 5 and 7), under docs/SCREENS.md section 3b and
docs/VERDICT_AUDIT_2026_10_17.md. Code: `outputs/mp_unified/za4h.py`.
Filed 19 October: both arms NOT_ADOPTED (section 7).

## 1. Why this, now

A4 put bdrop20's block dropout (each (question, block) kept with probability 0.8 in training) at the top of both lean
models. Filed 18 October, both arms NOT_ADOPTED:

| arm | GAIN | LOSS |
| --- | --- | --- |
| zrckpd (MLP) | none | musique zero-shot −0.1195 (L-musique) |
| zspkd (GNN) | webqsp zero-shot +0.0134 (L-musique) | musique zero-shot −0.1394 (L-musique) |

The GNN arm had a GAIN and a LOSS, so A4's section 5 fixed the next form: the same dropout at half the rate, on both
models. A4's other next form, A4f (the family form, docs/A4F_FAMILY_DROPOUT.md), is declared beside this file.

## 2. The form

A4's form, one number changed:
- In a training forward, each (question, block) of the model's blocks is kept with probability **0.9** (A4: 0.8).
- The draws come from the model's own generator, with bdrop20's and A4's seed (seed + 101).
- The incoming keep flag is multiplied by the draw, so an absent block stays absent.
- A read keeps every block.

Where it acts is A4's:

| layer | reads the keep flags? | under A4h |
| --- | --- | --- |
| zret's blocks (the eight fixed blocks and SEMB) | each block enters as [raw m, z m, m] | a dropped block's columns and flag are 0 |
| rmatch's chain term (`ChainMatch`) | gated by SEMB's flag | drops with SEMB |
| the KT head (KTMix, KTMixP) | no | unchanged |
| zsp's propagation (ZProp, the GNN only) | no | unchanged; it reads the score A4h changes |

No new parameter, no score read; the global random stream, dropout's masks included, is the base's.

**The MLP rule.** A4h changes only which inputs a training forward sees. It reads no score and normalises nothing over
the question's scores.

## 3. The arms

| arm | model | carve |
| --- | --- | --- |
| zrckph | zrckp's model with A4HTop at the top (MRO: A4HTop, ZRCKP, KTMixP, ZRM, ChainMatch, ZRet) | zrckp's (`zp1c3.KTChainCarveZRC`) |
| zspkh | zspk's model with A4HTop at the top (MRO: A4HTop, ZSPK, ZProp, KTMix, ZRM, ChainMatch, ZRet) | zspk's (`zp1c3.KTLinkCarveBase`) |

## 4. The fits and the call

A4's, unchanged:
- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-a4h-<arm>`, `-hp`), on P1c2's roots with
  P1c3's KT input, at the card fits' settings (variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split (zrckp: scr-p1c3p-zrckp and -hp; zspk:
  scr-p1c3-zspk and -hp), step 1's fit beside. The re-call's base screens are L1's.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), per arm.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A4h's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:** musique on both splits (A4's loss zero-shot, Claim 1's read in domain); webqsp
zero-shot (A4's GNN gain); metaqa; hit@1 and FC@5 per read; per fit, the share of cells dropped. When the results are
filed, A4's deltas (A4's section 7) are set beside each read, so the rate's effect reads directly.

**Serving cost: none.** Reads keep every block.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools, then its full run |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A4hs, the back-off over blocks: dropout at 0.1 on the eight fixed blocks only, SEMB (and with it rmatch's chain term) always kept, on both models |
| NOT_ADOPTED, with no GAIN | the rate closes; the idea goes on in A4f (declared beside), on both models |
| INCOMPLETE | the missing re-call is run; nothing else changes |

No model change is adopted on today's pools (the user's rule of 13 October): an adopt is refitted on P1g-b's pools
before its full run.

## 6. How it runs

**Self-test.** `python outputs/mp_unified/za4h.py --selftest` passes before this commit (9.9 s). It is A4's, at the new
rate:
- the arms' MROs with A4HTop at the top; the base arms unchanged and A4's arms not imported; the rate 0.1 beside
  bdrop20's 0.2; bdrop20's seed;
- a draw of 40,000 cells keeps 0.9 within 0.01;
- each arm against its base: in eval mode bit for bit with no draw; in training at a rate of 0 bit for bit (scores,
  every gradient, the global stream); with a fixed mask, the base model's forward on keep times the mask, bit for
  bit, the scores moving only on the masked questions; a question whose SEMB is dropped loses the chain term;
- toy fits through lean_gpu's loop repeat bit for bit, equal the base arm's at a rate of 0, and differ with A4h;
- the refusals; relz's pair and re-call per arm, stamped A4h's; the verdict's outcomes and next forms.

**Host items** (17, `a4h-*`) go after the queued A13 verdict and P5c, sized by A4's measured peaks (A4h adds no
parameter and no memory): trains and reads at A4's `--frac` and GPU share (L-musique trains 0.20/0.22, the rest
0.27/0.29); memory from A4's measured peaks, item by item.

## 7. Results

Filed 19 October 2026 from the host's records (`outputs/a4h/screen-verdict.json`, written 2026-10-18T13:24:30Z; the
re-calls `outputs/screen/scr-a4h-zrckph-pair-recall.json` and `-zspkh-`, host paths made relative). **Both arms
NOT_ADOPTED:** the MLP's arm has no GAIN and no LOSS; the GNN's has one GAIN and one LOSS.

- **zrckph: 0 GAIN, 0 LOSS.** musique zero-shot falls 0.0530 on L-musique [-0.0601, -0.0454]: filed LOSS, re-called
  WITHIN against that read's 0.072 floor. At A4's rate (0.2) it had fallen 0.1195. webqsp zero-shot rises +0.0127 on
  L-hotpotqa (filed GAIN, re-called WITHIN), and metaqa in domain falls 0.0036 there, WITHIN.
- **zspkh: 1 GAIN, 1 LOSS.** webqsp zero-shot rises +0.0248 [+0.0131, +0.0354] on L-hotpotqa; musique zero-shot falls
  0.1143 on L-musique [-0.1243, -0.1040]. At A4's rate the two reads were +0.0192 and -0.1394.
- **Half the rate more than halves the MLP's musique loss, not the GNN's:** zrckph falls 0.0530 where zrckpd fell
  0.1195, while zspkh falls 0.1143 where zspkd fell 0.1394. The GNN's webqsp gain grows at the lower rate on L-hotpotqa
  (+0.0248 against +0.0192) but is gone on L-musique (-0.0030 against zspkd's +0.0134, a GAIN there).
- **Beside:** musique in domain (L-hotpotqa, Claim 1's read) -0.0064 (MLP) and -0.0017 (GNN), both WITHIN; metaqa in
  domain within 0.004 on all four fits.
- **The fits:** 0.100 of the eligible cells dropped on both splits, and no question lost every block (table below).

This closes only this form (each fixed block dropped at 0.1, at the top of the forward). The next forms, fixed in
section 5 and recorded in the verdict:
- **zrckph (no GAIN):** the rate closes on the MLP; the idea goes on in A4f, declared beside it (filed with it today:
  NOT_ADOPTED, docs/A4F_FAMILY_DROPOUT.md, section 7).
- **zspkh (a GAIN and a LOSS): A4hs**, the back-off over blocks: block dropout at 0.1 on the eight fixed blocks only,
  with SEMB, and with it rmatch's chain term, always kept, on both models, in its own declared file.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's; a call is shown with its
filed call where the re-call changed it; base R@5 shown as zrckp's / zspk's):

| split | dataset | read | base R@5 | zrckph: delta [95% CI], call | zspkh: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8019 / 0.8009 | +0.0006 [-0.0003, +0.0015], WITHIN | +0.0023 [+0.0014, +0.0034], WITHIN |
| L-musique | squad | in-domain | 0.9116 / 0.9114 | -0.0012 [-0.0029, +0.0005], WITHIN | -0.0009 [-0.0029, +0.0011], WITHIN |
| L-musique | musique | zero-shot | 0.5121 / 0.5012 | -0.0530 [-0.0601, -0.0454], WITHIN (filed LOSS) | -0.1143 [-0.1243, -0.1040], **LOSS** |
| L-musique | hotpotqa | in-domain | 0.9065 / 0.9080 | -0.0016 [-0.0038, +0.0007], WITHIN | -0.0020 [-0.0041, +0.0001], WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 / 0.9383 | +0.0004 [-0.0012, +0.0020], WITHIN | -0.0019 [-0.0033, -0.0004], WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 / 0.3229 | -0.0004 [-0.0108, +0.0096], WITHIN | -0.0030 [-0.0119, +0.0056], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8044 / 0.8040 | -0.0036 [-0.0045, -0.0027], WITHIN | -0.0017 [-0.0027, -0.0008], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 / 0.9135 | -0.0012 [-0.0031, +0.0008], WITHIN | -0.0016 [-0.0035, +0.0003], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 / 0.5754 | -0.0064 [-0.0116, -0.0013], WITHIN | -0.0017 [-0.0071, +0.0036], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 / 0.8499 | +0.0048 [+0.0017, +0.0079], WITHIN | +0.0037 [+0.0007, +0.0067], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 / 0.9333 | -0.0034 [-0.0050, -0.0018], WITHIN | +0.0022 [+0.0008, +0.0036], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 / 0.2680 | +0.0127 [+0.0021, +0.0238], WITHIN (filed GAIN) | +0.0248 [+0.0131, +0.0354], **GAIN** |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a4h-zrckph.json` and the others, `-hp` for L-hotpotqa); and zrckpd's R@5 delta (A4, its own
re-call) and zspkd's R@5 delta (A4, its own re-call), on the same read:

| split | dataset | zrckph FC@5 | zrckph hit@1 | zspkh FC@5 | zspkh hit@1 | zrckpd R@5 (A4) | zspkd R@5 (A4) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L-musique | metaqa | +0.0002 | +0.0027 | +0.0025 | +0.0139 | -0.0026 | +0.0006 |
| L-musique | squad | -0.0012 | -0.0067 | -0.0009 | -0.0044 | -0.0013 | -0.0011 |
| L-musique | musique | -0.0285 | -0.1332 | -0.0687 | -0.2693 | -0.1195 LOSS | -0.1394 LOSS |
| L-musique | hotpotqa | -0.0015 | -0.0035 | -0.0045 | +0.0011 | -0.0050 | -0.0047 |
| L-musique | 2wiki | +0.0001 | -0.0023 | -0.0052 | -0.0015 | -0.0033 | -0.0032 |
| L-musique | webqsp | -0.0047 | -0.0160 | -0.0047 | -0.0093 | +0.0019 | +0.0134 GAIN |
| L-hotpotqa | metaqa | -0.0044 | -0.0128 | -0.0026 | -0.0066 | -0.0024 | -0.0018 |
| L-hotpotqa | squad | -0.0012 | -0.0040 | -0.0016 | -0.0048 | -0.0001 | -0.0043 |
| L-hotpotqa | musique | -0.0074 | -0.0083 | -0.0091 | -0.0037 | -0.0053 | -0.0032 |
| L-hotpotqa | hotpotqa | +0.0097 | -0.0061 | +0.0080 | -0.0117 | +0.0016 | +0.0009 |
| L-hotpotqa | 2wiki | -0.0112 | -0.0062 | +0.0056 | +0.0002 | -0.0010 | -0.0043 |
| L-hotpotqa | webqsp | +0.0067 | +0.0047 | +0.0173 | +0.0140 | -0.0030 | +0.0192 |

The fits (stamped in each fit's screen.json; both arms drew the same cells):

| fit | cells | dropped | share dropped | questions with every cell dropped |
| --- | ---: | ---: | ---: | ---: |
| L-musique | 1,704,528 | 170,502 | 0.1000 | 0 |
| L-hotpotqa | 1,608,840 | 160,886 | 0.1000 | 0 |
