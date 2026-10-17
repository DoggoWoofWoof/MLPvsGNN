# A4f: block dropout in its family form, in both models' training

Declared 18 October 2026, before any number. The next form A4's verdict named for zrckpd
(docs/A4_BLOCK_DROPOUT_LEAN.md, sections 5 and 7), under docs/SCREENS.md section 3b and
docs/VERDICT_AUDIT_2026_10_17.md. Code: `outputs/mp_unified/za4f.py`.

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
