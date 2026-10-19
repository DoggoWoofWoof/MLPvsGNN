# A4hs: block dropout on the eight fixed blocks only, SEMB always kept, in both models' training

Declared 19 October 2026, before any number. The back-off form A4h's verdict named for zspkh
(docs/A4H_BLOCK_DROPOUT_HALF.md, section 5), under docs/SCREENS.md section 3b and docs/VERDICT_AUDIT_2026_10_17.md.
Code: `outputs/mp_unified/za4hs.py`. Filed 19 October: both arms NOT_ADOPTED with no GAIN (zspkhs with one LOSS, webqsp
zero-shot on L-musique); the next form is A4he (section 7).

## 1. Why this, now

A4h put A4's block dropout at half the rate at the top of both lean models: in training, each (question, block) of the
model's nine blocks was kept with probability 0.9. A dropped SEMB took rmatch's chain term with it, because the term is
gated by SEMB's flag.

Its verdict, from the host's records (`outputs/a4h/screen-verdict.json`, written 2026-10-18T13:24:30Z; the re-calls
`outputs/screen/scr-a4h-zrckph-pair-recall.json` and `-zspkh-`): **both arms NOT_ADOPTED.** The reads that crossed a
floor, filed or re-called:

| arm | read | delta R@5 [95% CI] | floor | filed | re-call |
| --- | --- | --- | --- | --- | --- |
| zspkh (GNN) | L-hotpotqa, webqsp zero-shot | +0.0248 [+0.0131, +0.0354] | 0.0233 | GAIN | GAIN |
| zspkh (GNN) | L-musique, musique zero-shot | −0.1143 [−0.1243, −0.1040] | 0.0720 | LOSS | LOSS |
| zrckph (MLP) | L-hotpotqa, webqsp zero-shot | +0.0127 [+0.0021, +0.0238] | 0.0233 | GAIN | WITHIN |
| zrckph (MLP) | L-musique, musique zero-shot | −0.0530 [−0.0601, −0.0454] | 0.0720 | LOSS | WITHIN |

Every other read of both arms was within its floor, filed and re-called. The GNN arm had a GAIN and a LOSS, so A4h's
section 5 fixed this form, on both models.

The three block-dropout forms so far, on zspk (A4's section 7 and `outputs/screen/scr-a4-zspkd-pair-recall.json`; A4f's
verdict `outputs/a4f/screen-verdict.json` and `outputs/screen/scr-a4f-zspkf-pair-recall.json`):

| form (zspk's arm) | what a training forward drops | musique zero-shot (L-musique) | webqsp zero-shot (L-musique) | webqsp zero-shot (L-hotpotqa) |
| --- | --- | --- | --- | --- |
| A4 (zspkd) | 0.2 per (question, block), nine blocks | −0.1394 [−0.1503, −0.1282], LOSS | +0.0134 [+0.0032, +0.0237], GAIN | +0.0192 [+0.0085, +0.0297], WITHIN (filed GAIN) |
| A4h (zspkh) | 0.1 per (question, block), nine blocks | −0.1143 [−0.1243, −0.1040], LOSS | −0.0030 [−0.0119, +0.0056], WITHIN | +0.0248 [+0.0131, +0.0354], GAIN |
| A4f (zspkf) | 0.2 per cell: nine blocks, the KT term, the chain term | −0.2264 [−0.2372, −0.2158], LOSS | −0.0125 [−0.0243, −0.0010], LOSS | −0.0128 [−0.0248, −0.0010], WITHIN (filed LOSS) |

One reading: musique zero-shot falls more the more a training forward drops. And webqsp zero-shot, a graph whose rows
carry the chain term, gained only while the chain term dropped no more often than SEMB (A4, A4h). It lost when the
chain term and the KT term had cells of their own too (A4f). A4hs drops less again: never SEMB, so never the chain
term. If the loss comes with
SEMB's drop and the gain with the fixed blocks', A4hs keeps the gain without the loss. Section 5 says what each other
outcome leads to.

## 2. The form

A4h's form, with SEMB taken out of the drop:
- **The blocks.** Variant p's nine blocks, in the model's order (lean_mlp8's PICK, lean_gpu's `SETS["pick"]`): rank,
  dense_cos, topo_STRUCT, depth_STRUCT, **SEMB**, SEED, WALK, DISTS, WALKF. SEMB is the fifth (index 4). The eight
  fixed blocks are the other eight: rank, dense_cos, topo_STRUCT, depth_STRUCT, SEED, WALK, DISTS, WALKF (lean_gpu's
  `SETS["pns"]`).
- **The rate.** In a training forward, each (question, fixed block) is dropped with probability **0.1**, so kept with
  0.9 (A4h's rate).
- **The draw is A4h's.** The model's own generator, with bdrop20's, A4's and A4h's seed (seed + 101), draws one uniform
  per (question, block) of the model's blocks. A cell is kept when its uniform is at least 0.1. So a fixed block's
  cells are A4h's, cell for cell.
- **SEMB is never dropped.** SEMB's cell is then set to kept. lean_mlp's drop mode kept rank the same way
  (`NEVER = ("rank",)`). rmatch's chain term, gated by SEMB's flag, therefore always stays.
- **The flags.** Each flag is the incoming flag times its cell, so an absent block stays absent, SEMB included.
- **Reads** keep every block.
- **The fit's blocks.** lean_gpu leaves out of a fit any block that is constant on every training row (`dead_blocks`),
  as for the base. The arm finds SEMB by name, as rmatch does (`blocks.index("SEMB")`, rmatch's `j_semb`). The fixed
  blocks are whichever others the fit has. A model without SEMB is refused (rmatch's own check).

The constants (`za4hs.py`): `P_DROP = 0.1`, `SEED_OFF = 101`, `KEPT = "SEMB"`, `FIXED` = variant p's blocks without
SEMB.

Where it acts:

| layer | reads the keep flags? | under A4hs |
| --- | --- | --- |
| zret's eight fixed blocks | each block enters as [raw m, z m, m] | a dropped block's columns and flag are 0 |
| zret's SEMB block | the same | never dropped |
| zret's base term (rrf's z-score) | no; it reads the rank block's raw columns | unchanged, also when rank drops |
| rmatch's chain term (`ChainMatch`) | gated by SEMB's flag | never dropped (A4h dropped it with SEMB) |
| the KT head (KTMix, KTMixP) | no | unchanged |
| zsp's propagation (ZProp, the GNN only) | no | unchanged; it reads the score A4hs changes |

No new parameter, no score read; the global random stream, dropout's masks included, is the base's.

**The MLP rule.** A4hs changes only which inputs a training forward sees. It reads no score and normalises nothing over
the question's scores.

## 3. The arms

| arm | model | carve |
| --- | --- | --- |
| zrckphs | zrckp's model with A4HSTop at the top (MRO: A4HSTop, ZRCKP, KTMixP, ZRM, ChainMatch, ZRet) | zrckp's (`zp1c3.KTChainCarveZRC`) |
| zspkhs | zspk's model with A4HSTop at the top (MRO: A4HSTop, ZSPK, ZProp, KTMix, ZRM, ChainMatch, ZRet) | zspk's (`zp1c3.KTLinkCarveBase`) |

## 4. The fits and the call

A4h's, unchanged:
- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-a4hs-<arm>`, `-hp`), on P1c2's roots with
  P1c3's KT input, at the card fits' settings (variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split (zrckp: scr-p1c3p-zrckp and -hp; zspk:
  scr-p1c3-zspk and -hp), step 1's fit beside. The re-call's base screens are L1's.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), per arm.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A4hs's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:** musique on both splits (A4h's loss zero-shot, Claim 1's read in domain); webqsp
zero-shot (A4h's GNN gain); metaqa (the one training graph whose rows carry the chain term); hit@1 and FC@5 per read;
per fit, the share of fixed-block cells dropped, and how many of SEMB's draws fell under the cut (where A4h dropped SEMB
and the chain term, and A4hs keeps them). When the results are filed, A4h's deltas (section 1) are set beside each
read, so the effect of keeping SEMB reads directly.

**Serving cost: none.** Reads keep every block.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools, then its full run, its own declared file |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A4hg, the back-off over blocks one step further: dropout at 0.1 on the six graph and seed blocks only (topo_STRUCT, depth_STRUCT, SEED, WALK, DISTS, WALKF); the question-match blocks rank, dense_cos and SEMB, and with SEMB rmatch's chain term, always kept; on both models, its own declared file |
| NOT_ADOPTED, with no GAIN | A4he, the complement: dropout at 0.1 on SEMB alone (and with it rmatch's chain term), the eight fixed blocks always kept, on both models, its own declared file. Reason: A4h's GAIN may have come with SEMB's drop |
| INCOMPLETE | the missing re-call is run; nothing else changes |

No model change is adopted on today's pools (the user's rule of 13 October): an adopt is refitted on P1g-b's pools
before its full run.

## 6. How it runs

**Commands** (on the host each train and read goes through `outputs/host_ops/retry_cmd.py` and
`outputs/host_ops/cuda_alloc.py`, as the items show):

```
python outputs/mp_unified/za4hs.py train --split L-musique --name scr-a4hs-zrckphs --arm zrckphs --device cuda --host --cache-root outputs/p1c2/cache
python outputs/mp_unified/za4hs.py train --split L-hotpotqa --name scr-a4hs-zrckphs-hp --arm zrckphs --device cuda --host --cache-root outputs/p1c2/cache
python outputs/mp_unified/za4hs.py train --split L-musique --name scr-a4hs-zspkhs --arm zspkhs --device cuda --host --cache-root outputs/p1c2/cache
python outputs/mp_unified/za4hs.py train --split L-hotpotqa --name scr-a4hs-zspkhs-hp --arm zspkhs --device cuda --host --cache-root outputs/p1c2/cache
python outputs/mp_unified/za4hs.py read --name scr-a4hs-zrckphs --device cuda --host --cache-root outputs/p1c2/cache
    (and scr-a4hs-zrckphs-hp, scr-a4hs-zspkhs, scr-a4hs-zspkhs-hp)
python outputs/mp_unified/za4hs.py compare --new outputs/screen/fits/scr-a4hs-zrckphs --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a4hs-zrckphs
python outputs/mp_unified/za4hs.py compare --new outputs/screen/fits/scr-a4hs-zrckphs-hp --base outputs/screen/fits/scr-p1c3p-zrckp-hp,outputs/step1/fits/L-hotpotqa --out outputs/screen/scr-a4hs-zrckphs-hp
python outputs/mp_unified/za4hs.py compare --new outputs/screen/fits/scr-a4hs-zspkhs --base outputs/screen/fits/scr-p1c3-zspk,outputs/step1/fits/L-musique --out outputs/screen/scr-a4hs-zspkhs
python outputs/mp_unified/za4hs.py compare --new outputs/screen/fits/scr-a4hs-zspkhs-hp --base outputs/screen/fits/scr-p1c3-zspk-hp,outputs/step1/fits/L-hotpotqa --out outputs/screen/scr-a4hs-zspkhs-hp
python outputs/mp_unified/za4hs.py pair --arm zrckphs --screens outputs/screen/scr-a4hs-zrckphs.json,outputs/screen/scr-a4hs-zrckphs-hp.json --out outputs/screen/scr-a4hs-zrckphs-pair
python outputs/mp_unified/za4hs.py recall --arm zrckphs --null outputs/screen/scr-null-s1.json,outputs/screen/scr-null-s2.json,outputs/screen/scr-null-s1-hp.json,outputs/screen/scr-null-s2-hp.json --pair outputs/screen/scr-a4hs-zrckphs-pair.json --out outputs/screen/scr-a4hs-zrckphs-pair-recall
    (pair and recall the same for zspkhs)
python outputs/mp_unified/za4hs.py verdict
```

**Self-test.** `python outputs/mp_unified/za4hs.py --selftest` passes before this commit (4.8 s). It is A4h's, with
SEMB kept:
- the arms' MROs with A4HSTop at the top; the base arms unchanged, and A4h's and A4's arms not imported; the rate 0.1
  beside bdrop20's 0.2; bdrop20's seed; the eight fixed blocks are variant p's nine without SEMB (lean_gpu's pns), and
  SEMB is the fifth; a model without SEMB is refused;
- on variant p's nine blocks, 40,000 questions' cells: SEMB's flag never drops (an absent SEMB stays absent, and at a
  rate of 1 every fixed cell drops while SEMB stays); the fixed blocks' cells are A4h's, cell for cell, each kept 0.9
  within 0.01; the counts are the cells drawn;
- each arm against its base, on toy carves with rmatch's chains (rmatch's gate set nonzero):
  - in eval mode bit for bit, with no draw;
  - in training at a rate of 0 bit for bit: scores, every gradient, and the global stream;
  - with fixed cells, the base model's forward on keep times A4hs's cells, bit for bit; the scores move only on the
    questions where a fixed block drops;
  - a question whose SEMB draw falls under the cut keeps SEMB and the chain term: its scores do not move, A4h's cells
    would move them, and zeroing rmatch's gate moves them; a question whose SEMB is absent stays without the chain term;
- toy fits through lean_gpu's loop repeat bit for bit, equal the base arm's at a rate of 0, and differ with A4hs; the
  last fit's counts are its own;
- the refusals (A4h's arm name and an A4h fit record among them); relz's pair and re-call per arm, stamped A4hs's; the
  verdict's outcomes and next forms (a re-call stamped A4h's, or declared in A4h's file, or with eleven reads, is
  INCOMPLETE).

**Host items** (17, `a4hs-*`), sized as A4h's (A4hs adds no parameter and no memory): trains and reads at A4h's
`--frac` and GPU share (L-musique trains 0.20/0.22, the rest 0.27/0.29); memory as A4h's, item by item. Trains wait for
nothing; each re-call waits for L1's base screens, as A4h's did.

## 7. Results

Filed 19 October 2026 from the host's records (`outputs/a4hs/screen-verdict.json`, written 2026-10-19T14:37:40Z; the
re-calls `outputs/screen/scr-a4hs-zrckphs-pair-recall.json` and `-zspkhs-`, host paths made relative). **Both arms
NOT_ADOPTED, with no GAIN:** with SEMB always kept, block dropout on the eight fixed blocks gains nothing on either
model that the seed null cannot explain, and the GNN loses webqsp zero-shot on L-musique.

- **zrckphs: 0 GAIN, 0 LOSS.** Every read stays within its floor. The largest move is hotpotqa zero-shot on L-hotpotqa,
  -0.0117 [-0.0149, -0.0084], WITHIN (filed LOSS) against its 0.02 floor, with FC@5 -0.0227. musique zero-shot on
  L-musique moves +0.0049 [+0.0004, +0.0097].
- **zspkhs: 0 GAIN, 1 LOSS.** webqsp zero-shot on L-musique falls 0.0109 [-0.0216, -0.0005], past its 0.0102 floor.
  musique zero-shot on L-musique rises +0.0253 [+0.0190, +0.0316], WITHIN (filed GAIN) against its 0.072 floor. webqsp
  zero-shot on L-hotpotqa moves +0.0092 [-0.0001, +0.0179], within its 0.0233 floor.
- **Beside: A4h's gain and its loss both went with SEMB's drop.** A4h dropped SEMB along with the fixed blocks. Its GNN
  arm lost musique zero-shot on L-musique (-0.1143) and gained webqsp zero-shot on L-hotpotqa (+0.0248). With SEMB
  always kept, the loss is gone (+0.0253 now), and the gain shrinks to +0.0092, within its floor. The MLP moves the same
  way within its floors: A4h's -0.0530 and +0.0127 become +0.0049 and +0.0021.

Neither arm is adopted. This closes only this form, block dropout on the eight fixed blocks with SEMB kept
(docs/SCREENS.md section 3b). The next form, fixed in section 5 and recorded in the verdict, is A4he, the complement:
dropout at 0.1 on SEMB alone, and with it rmatch's chain term, the eight fixed blocks always kept, on both models, in
its own declared file. One reading of the two screens is that SEMB's drop carried both of A4h's effects; A4he tests that
drop alone.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's; a call is shown with its
filed call where the re-call changed it; base R@5 shown as zrckp's / zspk's):

| split | dataset | read | base R@5 | zrckphs: delta [95% CI], call | zspkhs: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8019 / 0.8009 | -0.0010 [-0.0018, -0.0001], WITHIN | +0.0016 [+0.0006, +0.0026], WITHIN |
| L-musique | squad | in-domain | 0.9116 / 0.9114 | +0.0005 [-0.0014, +0.0024], WITHIN | +0.0005 [-0.0014, +0.0024], WITHIN |
| L-musique | musique | zero-shot | 0.5121 / 0.5012 | +0.0049 [+0.0004, +0.0097], WITHIN | +0.0253 [+0.0190, +0.0316], WITHIN (filed GAIN) |
| L-musique | hotpotqa | in-domain | 0.9065 / 0.9080 | -0.0007 [-0.0030, +0.0016], WITHIN | -0.0043 [-0.0065, -0.0020], WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 / 0.9383 | +0.0015 [+0.0001, +0.0031], WITHIN | -0.0026 [-0.0041, -0.0012], WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 / 0.3229 | -0.0016 [-0.0103, +0.0066], WITHIN | -0.0109 [-0.0216, -0.0005], **LOSS** |
| L-hotpotqa | metaqa | in-domain | 0.8044 / 0.8040 | -0.0010 [-0.0019, -0.0001], WITHIN | -0.0013 [-0.0021, -0.0005], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 / 0.9135 | -0.0003 [-0.0021, +0.0017], WITHIN | -0.0006 [-0.0025, +0.0013], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 / 0.5754 | -0.0008 [-0.0064, +0.0049], WITHIN | +0.0006 [-0.0047, +0.0056], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 / 0.8499 | -0.0117 [-0.0149, -0.0084], WITHIN (filed LOSS) | -0.0064 [-0.0095, -0.0034], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 / 0.9333 | -0.0016 [-0.0032, +0.0001], WITHIN | +0.0036 [+0.0019, +0.0052], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 / 0.2680 | +0.0021 [-0.0078, +0.0123], WITHIN | +0.0092 [-0.0001, +0.0179], WITHIN |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a4hs-zrckphs.json` and the others, `-hp` for L-hotpotqa); and A4h's R@5 deltas with SEMB in the
drop (zrckph and zspkh, their own re-calls), on the same read:

| split | dataset | zrckphs FC@5 | zrckphs hit@1 | zspkhs FC@5 | zspkhs hit@1 | zrckph R@5 (A4h) | zspkh R@5 (A4h) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L-musique | metaqa | -0.0004 | -0.0015 | +0.0022 | +0.0083 | +0.0006 | +0.0023 |
| L-musique | squad | +0.0005 | -0.0040 | +0.0005 | -0.0008 | -0.0012 | -0.0009 |
| L-musique | musique | +0.0074 | -0.0041 | +0.0199 | +0.0132 | -0.0530 | -0.1143 LOSS |
| L-musique | hotpotqa | -0.0011 | -0.0024 | -0.0090 | +0.0014 | -0.0016 | -0.0020 |
| L-musique | 2wiki | +0.0038 | +0.0011 | -0.0056 | +0.0011 | +0.0004 | -0.0019 |
| L-musique | webqsp | -0.0080 | +0.0000 | -0.0100 | -0.0086 | -0.0004 | -0.0030 |
| L-hotpotqa | metaqa | -0.0012 | -0.0037 | -0.0014 | -0.0038 | -0.0036 | -0.0017 |
| L-hotpotqa | squad | -0.0003 | -0.0026 | -0.0006 | -0.0015 | -0.0012 | -0.0016 |
| L-hotpotqa | musique | +0.0074 | -0.0128 | -0.0037 | -0.0058 | -0.0064 | -0.0017 |
| L-hotpotqa | hotpotqa | -0.0227 | +0.0078 | -0.0113 | -0.0057 | +0.0048 | +0.0037 |
| L-hotpotqa | 2wiki | -0.0054 | +0.0006 | +0.0114 | -0.0007 | -0.0034 | +0.0022 |
| L-hotpotqa | webqsp | -0.0013 | +0.0007 | +0.0060 | -0.0007 | +0.0127 | +0.0248 GAIN |
