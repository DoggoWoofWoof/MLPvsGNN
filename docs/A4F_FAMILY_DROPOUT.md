# A4f: block dropout in its family form, in both models' training

Declared 18 October 2026, before any number. The next form A4's verdict named for zrckpd
(docs/A4_BLOCK_DROPOUT_LEAN.md, sections 5 and 7), under docs/SCREENS.md section 3b and
docs/VERDICT_AUDIT_2026_10_17.md. Code: `outputs/mp_unified/za4f.py`.
Filed 19 October: both arms NOT_ADOPTED (section 7).

## 1. Why this, now

A4 dropped each (question, block) of zret's blocks with probability 0.2 in training and left the two inputs above zret
always on: the KT head's term (P1c3's eight KIND_T columns) and rmatch's chain term. Both arms were NOT_ADOPTED; the
MLP arm had no GAIN (musique zero-shot −0.1195), and the GNN arm lost musique zero-shot too (−0.1394).

bdrop20, the same mask on step 1's model, which had neither input, *gained* musique zero-shot +0.1344. One reading of
the sign flip: with the KT term and the chain term never dropped, a model trained under A4 leans harder on them, and
they transfer worse to musique. The family form tests that reading directly, by dropping them as blocks too.
A4's section 5 fixed it as zrckpd's next form (no GAIN), on both models.

## 2. The form

In a training forward, each question draws len(blocks) + 2 cells from the model's own generator (bdrop20's and A4's
seed, seed + 101), each kept with probability **0.8**:

| cell | when it is dropped |
| --- | --- |
| the first len(blocks) | multiply the keep flags: A4 exactly (a dropped block's columns and flag are 0) |
| the KT cell | the KT head's term is 0 on the question's rows |
| the chain cell | rmatch's chain term is 0 on the question's rows |

- **The chain term keeps SEMB's gate.** At a read it needs SEMB, so it is off when either its own cell or SEMB's is.
- **No rescaling.** A kept term is exactly the read's term; bdrop20's blocks are not rescaled either.
- **Reads keep everything.** Both models serve exactly as their bases do.
- **Nothing else changes.** No new parameter, no score read; the global random stream, dropout's masks included, is
  the base's.

The masks act inside the models' own forwards: A4FTop wraps the KT head (`kt_head`) and rmatch's chain features
(`chain_feats`) and multiplies them by the question's cell on each row. In the GNN both terms are added before zsp's
propagation reads the score, so a dropped term is dropped for the step too. The propagation's edge families are A9's
(docs/A9_EDGE_FAMILY_DROPOUT.md), not this file's.

**The MLP rule.** A4f changes only which inputs a training forward sees. It reads no score and normalises nothing
over the question's scores.

## 3. The arms

| arm | model | carve |
| --- | --- | --- |
| zrckpf | zrckp's model with A4FTop at the top (MRO: A4FTop, ZRCKP, KTMixP, ZRM, ChainMatch, ZRet) | zrckp's (`zp1c3.KTChainCarveZRC`) |
| zspkf | zspk's model with A4FTop at the top (MRO: A4FTop, ZSPK, ZProp, KTMix, ZRM, ChainMatch, ZRet) | zspk's (`zp1c3.KTLinkCarveBase`) |

## 4. The fits and the call

A4's, unchanged:
- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-a4f-<arm>`, `-hp`), on P1c2's roots with
  P1c3's KT input, at the card fits' settings (variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split (zrckp: scr-p1c3p-zrckp and -hp; zspk:
  scr-p1c3-zspk and -hp), step 1's fit beside. The re-call's base screens are L1's.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), per arm.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A4f's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:** musique on both splits (A4's loss zero-shot, Claim 1's read in domain); webqsp
zero-shot (A4's GNN gain); metaqa; hit@1 and FC@5 per read; per fit, the share of cells dropped and how often each
term was dropped. When the results are filed, A4's deltas (A4's section 7) are set beside each read: if the reading in
section 1 is right, musique zero-shot recovers relative to A4.

**Serving cost: none.** Reads keep everything.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools, then its full run |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A4fh, the family form at half the rate (0.1 per question on each cell), on both models |
| NOT_ADOPTED, with no GAIN | A4fx, the family form on the adopted arms (G5's zspx and R1's zrckpr, R1's columns one more cell) |
| INCOMPLETE | the missing re-call is run; nothing else changes |

No model change is adopted on today's pools (the user's rule of 13 October): an adopt is refitted on P1g-b's pools
before its full run.

## 6. How it runs

**Self-test.** `python outputs/mp_unified/za4f.py --selftest` passes before this commit (8.8 s). On toy carves that
carry the KT input and rmatch's chains, it checks:
- the arms' MROs with A4FTop at the top; the base arms unchanged and A4's arms not imported; A4's rate and seed;
- a draw of 80,000 cells keeps 0.8 within 0.01;
- each arm against its base (the KT head's last layer and rmatch's gate set nonzero, so both terms move scores):
  - in eval mode bit for bit, with no draw, and the row masks cleared;
  - in training at a rate of 0 bit for bit: scores, every gradient, and the global stream;
  - with fixed cells (a block, SEMB, the KT term, the chain term and pairs of them off on different questions), each
    question's rows equal the base model's forward on keep times its block cells, with the KT head's last layer at
    zero where its KT cell is off and rmatch's gate at zero where its chain cell is off, bit for bit; the scores move
    only on those questions, each term's own cell moves its question, and the counts are the cells drawn;
  - a question whose SEMB is dropped loses the chain term;
- toy fits through lean_gpu's loop repeat bit for bit, equal the base arm's at a rate of 0, and differ with A4f;
- the refusals; relz's pair and re-call per arm, stamped A4f's; the verdict's outcomes and next forms.

**Host items** (17, `a4f-*`) go right after A4h's, sized by A4's measured peaks (A4f adds no parameter and no memory):
trains and reads at A4's `--frac` and GPU share (L-musique trains 0.20/0.22, the rest 0.27/0.29); memory from A4's
measured peaks, item by item.

## 7. Results

Filed 19 October 2026 from the host's records (`outputs/a4f/screen-verdict.json`, written 2026-10-18T14:11:19Z; the
re-calls `outputs/screen/scr-a4f-zrckpf-pair-recall.json` and `-zspkf-`, host paths made relative). **Both arms
NOT_ADOPTED:** no GAIN, and two LOSSes on each.

- **zrckpf: 0 GAIN, 2 LOSS.** musique zero-shot falls 0.1147 on L-musique [-0.1246, -0.1048], and webqsp zero-shot falls
  0.0236 on L-hotpotqa [-0.0358, -0.0115].
- **zspkf: 0 GAIN, 2 LOSS.** musique zero-shot falls 0.2264 on L-musique [-0.2372, -0.2158], and webqsp zero-shot 0.0125
  there [-0.0243, -0.0010]; on L-hotpotqa webqsp zero-shot falls 0.0128 (filed LOSS, re-called WITHIN against its 0.0233
  floor).
- **Section 1's reading does not hold.** Dropping the KT term and rmatch's chain term as two more cells was to let
  musique zero-shot recover against A4. It did not: the MLP falls about as far as at A4 (-0.1147 against -0.1195), and
  the GNN falls further (-0.2264 against -0.1394).
- **Beside:** musique in domain (L-hotpotqa, Claim 1's read) -0.0035 (MLP) and +0.0021 (GNN), WITHIN; metaqa in domain
  -0.0057 on the MLP's L-hotpotqa fit, WITHIN.
- **The fits:** 0.200 of the cells dropped on both splits; the KT term and the chain term each dropped on 0.20 of the
  question visits, as declared; no question lost every cell (table below).

This closes only this form (the family form at 0.2 on the lean pair). The next form, fixed in section 5 for no GAIN and
recorded in the verdict, is **A4fx**: the family form on the adopted arms (G5's zspx and R1's zrckpr, R1's columns one
more cell), in its own declared file.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's; a call is shown with its
filed call where the re-call changed it; base R@5 shown as zrckp's / zspk's):

| split | dataset | read | base R@5 | zrckpf: delta [95% CI], call | zspkf: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8019 / 0.8009 | -0.0038 [-0.0052, -0.0025], WITHIN | -0.0039 [-0.0055, -0.0025], WITHIN |
| L-musique | squad | in-domain | 0.9116 / 0.9114 | -0.0008 [-0.0027, +0.0010], WITHIN | -0.0009 [-0.0032, +0.0013], WITHIN |
| L-musique | musique | zero-shot | 0.5121 / 0.5012 | -0.1147 [-0.1246, -0.1048], **LOSS** | -0.2264 [-0.2372, -0.2158], **LOSS** |
| L-musique | hotpotqa | in-domain | 0.9065 / 0.9080 | -0.0022 [-0.0046, +0.0003], WITHIN | -0.0023 [-0.0046, +0.0000], WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 / 0.9383 | -0.0024 [-0.0041, -0.0007], WITHIN | -0.0054 [-0.0070, -0.0039], WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 / 0.3229 | +0.0026 [-0.0093, +0.0145], WITHIN | -0.0125 [-0.0243, -0.0010], **LOSS** |
| L-hotpotqa | metaqa | in-domain | 0.8044 / 0.8040 | -0.0057 [-0.0070, -0.0043], WITHIN | -0.0017 [-0.0029, -0.0003], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 / 0.9135 | +0.0003 [-0.0018, +0.0024], WITHIN | -0.0022 [-0.0043, +0.0000], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 / 0.5754 | -0.0035 [-0.0089, +0.0018], WITHIN | +0.0021 [-0.0034, +0.0076], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 / 0.8499 | -0.0024 [-0.0055, +0.0007], WITHIN | +0.0039 [+0.0007, +0.0074], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 / 0.9333 | -0.0051 [-0.0069, -0.0034], WITHIN | +0.0030 [+0.0013, +0.0047], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 / 0.2680 | -0.0236 [-0.0358, -0.0115], **LOSS** | -0.0128 [-0.0248, -0.0010], WITHIN (filed LOSS) |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a4f-zrckpf.json` and the others, `-hp` for L-hotpotqa); and zrckpd's R@5 delta (A4, its own
re-call) and zspkd's R@5 delta (A4, its own re-call), on the same read:

| split | dataset | zrckpf FC@5 | zrckpf hit@1 | zspkf FC@5 | zspkf hit@1 | zrckpd R@5 (A4) | zspkd R@5 (A4) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L-musique | metaqa | -0.0048 | -0.0157 | -0.0059 | -0.0130 | -0.0026 | +0.0006 |
| L-musique | squad | -0.0008 | -0.0051 | -0.0009 | -0.0019 | -0.0013 | -0.0011 |
| L-musique | musique | -0.0691 | -0.2631 | -0.1171 | -0.3695 | -0.1195 LOSS | -0.1394 LOSS |
| L-musique | hotpotqa | -0.0008 | -0.0134 | -0.0043 | -0.0073 | -0.0050 | -0.0047 |
| L-musique | 2wiki | -0.0060 | -0.0112 | -0.0154 | -0.0035 | -0.0033 | -0.0032 |
| L-musique | webqsp | -0.0053 | -0.0073 | -0.0120 | -0.0286 | +0.0019 | +0.0134 GAIN |
| L-hotpotqa | metaqa | -0.0074 | -0.0244 | -0.0034 | -0.0072 | -0.0024 | -0.0018 |
| L-hotpotqa | squad | +0.0003 | +0.0006 | -0.0022 | -0.0048 | -0.0001 | -0.0043 |
| L-hotpotqa | musique | -0.0021 | -0.0054 | -0.0008 | -0.0058 | -0.0053 | -0.0032 |
| L-hotpotqa | hotpotqa | -0.0016 | +0.0068 | +0.0107 | -0.0112 | +0.0016 | +0.0009 |
| L-hotpotqa | 2wiki | -0.0122 | -0.0099 | +0.0077 | -0.0034 | -0.0010 | -0.0043 |
| L-hotpotqa | webqsp | -0.0253 | -0.0146 | -0.0113 | -0.0166 | -0.0030 | +0.0192 |

The fits (stamped in each fit's screen.json; both arms drew the same cells; the KT and chain terms' shares are of the
question visits):

| fit | cells | dropped | share dropped | KT term dropped | chain term dropped | questions with every cell dropped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L-musique | 2,083,312 | 416,916 | 0.2001 | 37,833 (0.200) | 38,176 (0.202) | 0 |
| L-hotpotqa | 1,966,360 | 393,318 | 0.2000 | 35,706 (0.200) | 35,998 (0.201) | 0 |
