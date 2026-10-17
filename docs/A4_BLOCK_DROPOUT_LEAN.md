# A4: block dropout in both models' training

Declared 17 October 2026, before any number. The fourth reopening of docs/VERDICT_AUDIT_2026_10_17.md (section 2,
row 4), under docs/SCREENS.md section 3b. Code: `outputs/mp_unified/za4bdrop.py`. Not run yet.

## 1. Why this, now

bdrop20 (7 October, `lean_screen.py`) kept each (question, block) with probability 0.8 in training. Its screen on step
1's model (L-musique, against step 1's fit):

| metaqa | squad | musique (zero-shot) | hotpotqa | 2wiki | webqsp (zero-shot) |
| --- | --- | --- | --- | --- | --- |
| −0.0047 WITHIN | −0.0007 WITHIN | **+0.1344 GAIN** [+0.124, +0.145] | −0.0048 WITHIN | −0.0050 WITHIN | −0.0037 WITHIN |

- **Its full run was not adopted (15:45)** on its first LOSS: webqsp zero-shot −0.0081 in L-metaqa's fit.
  - That LOSS was called at the floor of the time, 0.0075. The seed null filed since puts webqsp zero-shot's floors
    at 0.0102 to 0.0295.
  - Three of its fits were cancelled, so the run never had a complete grade to re-grade.
- **padbd20** (pad with it) was PROMISING (musique zero-shot +0.1095) and got no full run.

It never ran on zret or on anything later, and it never reached the GNN. Under section 3b the idea is still open.

## 2. The form

bdrop20's mask is unchanged:
- In a training forward, each (question, block) of the model's blocks is kept with probability 0.8.
- The draws come from the model's own generator, with bdrop20's seed (seed + 101).
- A read keeps every block.

What is new is where the mask goes: at the top of each of today's models, so that every layer reading the keep flags
sees the same dropped blocks.

| layer | reads the keep flags? | under A4 |
| --- | --- | --- |
| zret's blocks (the eight fixed blocks and SEMB) | each block enters as [raw m, z m, m] | a dropped block's columns and flag are 0, lean_mlp's missing-block input, as in bdrop20 |
| rmatch's chain term (`ChainMatch`) | gated by SEMB's flag | drops with SEMB |
| the KT head (KTMix, KTMixP) | no | unchanged |
| zsp's propagation (ZProp, the GNN only) | no | unchanged; it reads the score A4 changes |

- **The flags.** A4 multiplies the incoming flag by the draw, so a block that is absent stays absent. In training
  every incoming flag is 1, so this is bdrop20's mask exactly.
- **Nothing else changes.** There is no new parameter and no score read. The global random stream, dropout's masks
  included, is the base's. The self-test checks all of this.

**Why the top and not just above zret.** Placed just above zret's forward, the mask would drop SEMB's columns while
rmatch's chain term, which is gated by SEMB's flag, kept reading SEMB as present. The model would then learn from an
input state that never happens at a read.

**The MLP rule.** A4 changes only which inputs a training forward sees. It reads no score and normalises nothing over
the question's scores.

## 3. The arms

| arm | model | carve |
| --- | --- | --- |
| zrckpd | zrckp's model with A4Top at the top (MRO: A4Top, ZRCKP, KTMixP, ZRM, ChainMatch, ZRet) | zrckp's (`zp1c3.KTChainCarveZRC`) |
| zspkd | zspk's model with A4Top at the top (MRO: A4Top, ZSPK, ZProp, KTMix, ZRM, ChainMatch, ZRet) | zspk's (`zp1c3.KTLinkCarveBase`) |

## 4. The fits and the call

- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-a4-<arm>`, `-hp`), on P1c2's roots with
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
- **INCOMPLETE:** its re-call is missing, is not A4's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- musique, both splits and arms: zero-shot (L-musique) was bdrop20's gain, and in domain (L-hotpotqa) is Claim 1's
  read;
- webqsp zero-shot: bdrop20's LOSS;
- metaqa;
- hit@1 and FC@5 per read;
- per fit, the share of (question, block) cells dropped and the number of questions with every block dropped.

**Serving cost: none.** Reads keep every block, so both models serve exactly as their bases do.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools, then its full run |
| NOT_ADOPTED, with at least one GAIN and a LOSS | the back-off form, A4h: block dropout at half the rate, 0.1 per question and block, on both models |
| NOT_ADOPTED, with no GAIN | A4f, the family form: the inputs above zret are dropped as blocks too. The KT columns and rmatch's chain term each become one more (question, block) cell at 0.2, on both models. The propagation's edge families are A9's |
| INCOMPLETE | the missing re-call is run; nothing else changes |

- **If both arms pass**, block dropout becomes the lean pair's training, in both models.
- **If only one passes**, its full run is declared. The next file decides whether F1's rule needs the other model to
  train the same way.
- **What does not happen.** No model change is adopted on today's pools (the user's rule of 13 October).

## 6. How it runs

**Self-test.** `python outputs/mp_unified/za4bdrop.py --selftest` passes before this commit (4.0 s). It checks:
- the arms' MROs, with A4Top at the top; the base arms unchanged; bdrop20's rate and its seed (a BDrop built with
  seed 7 has mask seed 108);
- a draw of 40,000 cells keeps 0.8 within 0.01;
- each arm against its base:
  - in eval mode bit for bit, with no draw;
  - in training at a rate of 0 bit for bit: scores, every gradient, and the global stream;
  - with a fixed mask, the arm equals the base model's forward on keep times the mask, bit for bit. The scores move
    only on the masked questions;
  - a question whose SEMB is dropped loses rmatch's chain term;
- toy fits through lean_gpu's loop repeat bit for bit, equal the base arm's fit at a rate of 0, and differ with A4;
- the refusals; relz's pair and re-call per arm, stamped A4's; the verdict's outcomes and next forms.

**Host items** (17, `a4-*`) go just after A3's verdict. They are sized as A3's are, from the measured peaks of the
base arms, since A4 adds no parameter:
- L-musique trains: `--frac 0.20`, GPU share 0.22;
- -hp trains and the reads: `--frac 0.27`, GPU share 0.29;
- memory: trains 4.5 GB (zspkd-hp 7 GB), reads 6 GB, compares 4 GB, pairs, re-calls and the verdict 2 GB.
