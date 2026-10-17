# A4h: block dropout at half the rate, in both models' training

Declared 18 October 2026, before any number. The back-off form A4's verdict named for zspkd
(docs/A4_BLOCK_DROPOUT_LEAN.md, sections 5 and 7), under docs/SCREENS.md section 3b and
docs/VERDICT_AUDIT_2026_10_17.md. Code: `outputs/mp_unified/za4h.py`.

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
