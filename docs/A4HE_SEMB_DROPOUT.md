# A4he: dropout on SEMB alone, the eight fixed blocks always kept, in both models' training

Declared 19 October 2026, before any number. The next form A4hs's verdict named (docs/A4HS_FIXED_BLOCK_DROPOUT.md,
sections 5 and 7: both arms NOT_ADOPTED with no GAIN), under docs/SCREENS.md section 3b and
docs/VERDICT_AUDIT_2026_10_17.md. Code: `outputs/mp_unified/za4he.py`.

## 1. Why this, now

A4h put block dropout at 0.1 at the top of both lean models: in training, each (question, block) of the model's nine
blocks was kept with probability 0.9. A dropped SEMB took rmatch's chain term with it, because the term is gated by
SEMB's flag. A4hs drew the same cells and never dropped SEMB, so only the eight fixed blocks dropped.

A4hs's verdict, from the host's records (`outputs/a4hs/screen-verdict.json`, written 2026-10-19T14:37:40Z; the re-calls
`outputs/screen/scr-a4hs-zrckphs-pair-recall.json` and `-zspkhs-`): **both arms NOT_ADOPTED, with no GAIN.** Set
beside A4h's re-calls (`outputs/screen/scr-a4h-zrckph-pair-recall.json` and `-zspkh-`), on the two reads where A4h moved
(delta R@5 against the same base, zrckp's / zspk's card fits):

| re-call | what a training forward drops | musique zero-shot, L-musique (floor 0.0720) | webqsp zero-shot, L-hotpotqa (floor 0.0233) |
| --- | --- | --- | --- |
| zrckph (A4h, MLP) | nine blocks; the chain term with SEMB | −0.0530 [−0.0601, −0.0454], WITHIN (filed LOSS) | +0.0127 [+0.0021, +0.0238], WITHIN (filed GAIN) |
| zspkh (A4h, GNN) | nine blocks; the chain term with SEMB | −0.1143 [−0.1243, −0.1040], **LOSS** | +0.0248 [+0.0131, +0.0354], **GAIN** |
| zrckphs (A4hs, MLP) | the eight fixed blocks | +0.0049 [+0.0004, +0.0097], WITHIN | +0.0021 [−0.0078, +0.0123], WITHIN |
| zspkhs (A4hs, GNN) | the eight fixed blocks | +0.0253 [+0.0190, +0.0316], WITHIN (filed GAIN) | +0.0092 [−0.0001, +0.0179], WITHIN |

The bases on these reads: musique zero-shot 0.5121 / 0.5012, webqsp zero-shot 0.3475 / 0.2680. A4hs's one LOSS was
zspkhs's webqsp zero-shot on L-musique, −0.0109 [−0.0216, −0.0005] against its 0.0102 floor (A4h's zspkh: −0.0030,
WITHIN).

With SEMB kept, both of A4h's effects are gone, on both models. So the cells A4hs kept are the ones that carried them:
SEMB's drops, and with them the chain term's. A4he drops exactly those cells and nothing else. It shares A4h's draw, so
the SEMB cells it drops are the ones A4h dropped and A4hs set to kept. If SEMB's drop carries both effects, A4he shows
the GNN's GAIN and its LOSS together, and section 5 then splits the drop in two (SEMB's columns, the chain term). If the
gain needed the fixed blocks' drop beside SEMB's, A4he finds no GAIN, and A4h's whole form goes to the adopted arms
(section 5).

The lean pair (zrckp, zspk) stays the base, as for A4h and A4hs, so the three forms read against the same fits.

## 2. The form

A4h's form, with the eight fixed blocks taken out of the drop:
- **The blocks.** Variant p's nine blocks, in the model's order (lean_mlp8's PICK, lean_gpu's `SETS["pick"]`): rank,
  dense_cos, topo_STRUCT, depth_STRUCT, **SEMB**, SEED, WALK, DISTS, WALKF. SEMB is the fifth (index 4). The eight
  fixed blocks are the other eight (lean_gpu's `SETS["pns"]`).
- **The rate.** In a training forward, each question's SEMB is dropped with probability **0.1**, so kept with 0.9
  (A4h's rate).
- **The draw is A4h's.** The model's own generator, with bdrop20's, A4's, A4h's and A4hs's seed (seed + 101), draws one
  uniform per (question, block) of the model's blocks. A cell is kept when its uniform is at least 0.1. So SEMB's cells
  are A4h's, cell for cell.
- **The fixed blocks are never dropped.** Their cells are then set to kept (A4hs set SEMB's the same way).
- **The chain term goes with SEMB.** rmatch's chain term is gated by SEMB's flag, so a dropped SEMB takes it too, as
  in A4h.
- **The flags.** Each flag is the incoming flag times its cell, so an absent block stays absent.
- **Reads** keep every block.
- **The fit's blocks.** lean_gpu leaves out of a fit any block that is constant on every training row (`dead_blocks`),
  as for the base. The arm finds SEMB by name, as rmatch does (`blocks.index("SEMB")`, rmatch's `j_semb`). The fixed
  blocks are whichever others the fit has. A model without SEMB is refused (rmatch's own check).

The constants (`za4he.py`): `P_DROP = 0.1`, `SEED_OFF = 101`, `DROPPED = "SEMB"`, `FIXED` = variant p's blocks
without SEMB.

Where it acts:

| layer | reads the keep flags? | under A4he |
| --- | --- | --- |
| zret's eight fixed blocks | each block enters as [raw m, z m, m] | never dropped |
| zret's SEMB block | the same | dropped at 0.1 per question; a dropped block's columns and flag are 0 |
| zret's base term (rrf's z-score) | no; it reads the rank block's raw columns | unchanged |
| rmatch's chain term (`ChainMatch`) | gated by SEMB's flag | dropped with SEMB (as in A4h) |
| the KT head (KTMix, KTMixP) | no | unchanged |
| zsp's propagation (ZProp, the GNN only) | no | unchanged; it reads the score A4he changes |

No new parameter, no score read; the global random stream, dropout's masks included, is the base's.

**The MLP rule.** A4he changes only which inputs a training forward sees. It reads no score and normalises nothing over
the question's scores.

## 3. The arms

| arm | model | carve |
| --- | --- | --- |
| zrckphe | zrckp's model with A4HETop at the top (MRO: A4HETop, ZRCKP, KTMixP, ZRM, ChainMatch, ZRet) | zrckp's (`zp1c3.KTChainCarveZRC`) |
| zspkhe | zspk's model with A4HETop at the top (MRO: A4HETop, ZSPK, ZProp, KTMix, ZRM, ChainMatch, ZRet) | zspk's (`zp1c3.KTLinkCarveBase`) |

## 4. The fits and the call

A4hs's, unchanged:
- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-a4he-<arm>`, `-hp`), on P1c2's roots with
  P1c3's KT input, at the card fits' settings (variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split (zrckp: scr-p1c3p-zrckp and -hp; zspk:
  scr-p1c3-zspk and -hp), step 1's fit beside. The re-call's base screens are L1's.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), per arm.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A4he's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:** musique on both splits (A4h's loss zero-shot, Claim 1's read in domain); webqsp
zero-shot (A4h's GNN gain); metaqa (the one training graph whose rows carry the chain term); hit@1 and FC@5 per read;
per fit, the share of SEMB cells dropped, and how many of the fixed blocks' draws fell under the cut (where A4h dropped
them and A4he keeps them). When the results are filed, A4h's and A4hs's deltas are set beside each read, so the two
halves of A4h's drop read side by side.

**Serving cost: none.** Reads keep every block.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools, then its full run, its own declared file |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A4hc, SEMB's drop split in two on both models: SEMB's columns dropped at 0.1 with rmatch's chain term kept, and the chain term dropped at 0.1 with SEMB's columns kept, the eight fixed blocks always kept, its own declared file |
| NOT_ADOPTED, with no GAIN | A4hx, A4h's whole form (0.1 per question over all nine blocks, the chain term going with SEMB) on the adopted arms, R1's zrckpr and G5's zspx, each against its own base, its own declared file. Block dropout at 0.1 closes as a training form only if A4hx finds no GAIN either |
| INCOMPLETE | the missing re-call is run; nothing else changes |

No model change is adopted on today's pools (the user's rule of 13 October): an adopt is refitted on P1g-b's pools
before its full run.

## 6. How it runs

**Commands** (on the host each train and read goes through `outputs/host_ops/retry_cmd.py` and
`outputs/host_ops/cuda_alloc.py`, as the items show):

```
python outputs/mp_unified/za4he.py train --split L-musique --name scr-a4he-zrckphe --arm zrckphe --device cuda --host --cache-root outputs/p1c2/cache
python outputs/mp_unified/za4he.py train --split L-hotpotqa --name scr-a4he-zrckphe-hp --arm zrckphe --device cuda --host --cache-root outputs/p1c2/cache
python outputs/mp_unified/za4he.py train --split L-musique --name scr-a4he-zspkhe --arm zspkhe --device cuda --host --cache-root outputs/p1c2/cache
python outputs/mp_unified/za4he.py train --split L-hotpotqa --name scr-a4he-zspkhe-hp --arm zspkhe --device cuda --host --cache-root outputs/p1c2/cache
python outputs/mp_unified/za4he.py read --name scr-a4he-zrckphe --device cuda --host --cache-root outputs/p1c2/cache
    (and scr-a4he-zrckphe-hp, scr-a4he-zspkhe, scr-a4he-zspkhe-hp)
python outputs/mp_unified/za4he.py compare --new outputs/screen/fits/scr-a4he-zrckphe --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a4he-zrckphe
python outputs/mp_unified/za4he.py compare --new outputs/screen/fits/scr-a4he-zrckphe-hp --base outputs/screen/fits/scr-p1c3p-zrckp-hp,outputs/step1/fits/L-hotpotqa --out outputs/screen/scr-a4he-zrckphe-hp
python outputs/mp_unified/za4he.py compare --new outputs/screen/fits/scr-a4he-zspkhe --base outputs/screen/fits/scr-p1c3-zspk,outputs/step1/fits/L-musique --out outputs/screen/scr-a4he-zspkhe
python outputs/mp_unified/za4he.py compare --new outputs/screen/fits/scr-a4he-zspkhe-hp --base outputs/screen/fits/scr-p1c3-zspk-hp,outputs/step1/fits/L-hotpotqa --out outputs/screen/scr-a4he-zspkhe-hp
python outputs/mp_unified/za4he.py pair --arm zrckphe --screens outputs/screen/scr-a4he-zrckphe.json,outputs/screen/scr-a4he-zrckphe-hp.json --out outputs/screen/scr-a4he-zrckphe-pair
python outputs/mp_unified/za4he.py recall --arm zrckphe --null outputs/screen/scr-null-s1.json,outputs/screen/scr-null-s2.json,outputs/screen/scr-null-s1-hp.json,outputs/screen/scr-null-s2-hp.json --pair outputs/screen/scr-a4he-zrckphe-pair.json --out outputs/screen/scr-a4he-zrckphe-pair-recall
    (pair and recall the same for zspkhe)
python outputs/mp_unified/za4he.py verdict
```

**Self-test.** `python outputs/mp_unified/za4he.py --selftest` passes before this commit (10.1 s). It is A4hs's, with
the cut moved from the fixed blocks to SEMB:
- the arms' MROs with A4HETop at the top; the base arms unchanged, and A4hs's, A4h's and A4's arms not imported; the
  rate 0.1 beside bdrop20's 0.2; bdrop20's seed; the eight fixed blocks are variant p's nine without SEMB (lean_gpu's
  pns), and SEMB is the fifth; a model without SEMB is refused;
- on variant p's nine blocks, 40,000 questions' cells: no fixed block's flag drops (an absent block stays absent, and
  at a rate of 1 every SEMB cell drops while the fixed blocks stay); SEMB's cells are A4h's, cell for cell, kept 0.9
  within 0.01; the counts are the cells drawn;
- each arm against its base, on toy carves with rmatch's chains (rmatch's gate set nonzero):
  - in eval mode bit for bit, with no draw;
  - in training at a rate of 0 bit for bit: scores, every gradient, and the global stream;
  - with fixed cells, the base model's forward on keep times A4he's cells, bit for bit; the scores move only on the
    questions whose SEMB drops;
  - a question whose rank draw falls under the cut keeps rank: its scores do not move, and A4h's cells would move them;
    a dropped SEMB takes the chain term with it (zeroing rmatch's gate moves nothing there, and moves a question that
    keeps SEMB); a question whose SEMB is absent stays without the chain term;
- toy fits through lean_gpu's loop repeat bit for bit, equal the base arm's at a rate of 0, and differ with A4he; the
  last fit's counts are its own;
- the refusals (A4hs's and A4h's arm names and an A4hs fit record among them); relz's pair and re-call per arm, stamped
  A4he's; the verdict's outcomes and next forms (a re-call stamped A4hs's, or declared in A4hs's file, or with eleven
  reads, is INCOMPLETE).

**Host items** (17, `a4he-*`), sized as A4hs's (A4he adds no parameter and no memory): trains and reads at A4hs's
`--frac` and GPU share (L-musique trains 0.20/0.22, the rest 0.27/0.29); memory as A4hs's, item by item. Trains wait
for nothing; each re-call waits for L1's base screens, as A4hs's did. They go after A2xmx's items. Disk: about 0.3 GB
(four fits and their reads).

## 7. Results

Filed 27 October 2026 from the host's records (`outputs/a4he/screen-verdict.json`, written 2026-10-20T18:50:14Z; the
re-calls `outputs/screen/scr-a4he-zrckphe-pair-recall.json` and `scr-a4he-zspkhe-pair-recall.json`, host paths made
relative). **zrckphe ADOPT_FOR_FULL_RUN, zspkhe NOT_ADOPTED:** with SEMB alone dropped at 0.1 per question in training
(the chain term with it, the eight fixed blocks always kept), each model gains one webqsp zero-shot read. Both fall on
musique zero-shot on L-musique: past its floor on the GNN, and on the MLP by 0.0014 less than its floor.

- **zrckphe: 1 GAIN, 0 LOSS.** The GAIN is webqsp zero-shot on L-musique: +0.0181 [+0.0077, +0.0293] against its 0.0102
  floor, with FC@5 +0.0080. musique zero-shot on L-musique falls -0.0706 [-0.0784, -0.0626], a LOSS by its pair's
  interval that is WITHIN under its 0.0720 floor (FC@5 -0.0401, hit@1 -0.1324); the adopt stands on that floor. Five
  more intervals exclude 0, all inside their floors: on L-musique, hotpotqa -0.0036 [-0.0058, -0.0012] and 2wiki -0.0072
  [-0.0088, -0.0054]; on L-hotpotqa, metaqa -0.0013 [-0.0020, -0.0005], hotpotqa zero-shot +0.0058 [+0.0029, +0.0087]
  and 2wiki -0.0024 [-0.0039, -0.0007].
- **zspkhe: 1 GAIN, 1 LOSS.** The GAIN is webqsp zero-shot on L-hotpotqa: +0.0312 [+0.0211, +0.0415] against its 0.0233
  floor, with FC@5 +0.0233. The LOSS is musique zero-shot on L-musique: -0.1036 [-0.1133, -0.0937] against its 0.0720
  floor, with FC@5 -0.0633 and hit@1 -0.2362. hotpotqa zero-shot on L-hotpotqa rises +0.0099 [+0.0068, +0.0130], a GAIN
  by its pair's interval that is WITHIN under its 0.0200 floor. Four more intervals exclude 0, all inside their floors:
  on L-musique, metaqa +0.0013 [+0.0004, +0.0022], squad -0.0021 [-0.0040, -0.0003] and 2wiki -0.0016 [-0.0030,
  -0.0002]; on L-hotpotqa, 2wiki -0.0025 [-0.0042, -0.0010].
- **Beside: SEMB carries musique zero-shot on L-musique.** With the eight fixed blocks dropped and SEMB kept (A4hs),
  that read moved up on both models (zrckphs +0.0049, zspkhs +0.0253); with SEMB alone dropped it falls on both
  (-0.0706, -0.1036). A4hs's one call past a floor, zspkhs's LOSS on webqsp zero-shot on L-musique (-0.0109), is gone
  here.

The verdict is per arm. **zrckphe** is adopted for a full run; its next form, fixed in section 5 and recorded in the
verdict, is the arm fitted again on P1g-b's pools, then its full run, its own declared file. P1g-b's pools have since
been left out of the adopted models (P1g-c, docs/P1GC_ADOPTED_ON_P1GB.md section 8: the full run takes P1c2's pools), so
the pools of that refit are its own file's to fix. **zspkhe** is not adopted; this closes only its form, SEMB's drop at
0.1 on the GNN (docs/SCREENS.md section 3b). Its next form, for a GAIN with a LOSS, is A4hc, SEMB's drop split in two on
both models: SEMB's columns dropped at 0.1 with rmatch's chain term kept, and the chain term dropped at 0.1 with SEMB's
columns kept, the eight fixed blocks always kept, its own declared file.

The floors are the seed null of 7 October (docs/SCREENS.md section 2: `outputs/screen/scr-null-s1.json`,
`scr-null-s2.json` and their `-hp` files), not N2's fresh floors (docs/N2_FRESH_SEED_NULL.md).

Records, filed with this section: the verdict `outputs/a4he/screen-verdict.json` and `.md`; for each arm (`zrckphe`,
`zspkhe`), the compares `outputs/screen/scr-a4he-<arm>.json` and `-hp.json`, the pair `-pair.json` and its re-call
`-pair-recall.json`, each with its `.md`. The host paths were made relative in each re-call's two `rel` entries and
nowhere else, so the `recall_sha256` each verdict holds is the host file's, from before that change. The re-calls' nulls
also read the base screens `outputs/l1/base-zrckp-L-musique.json` and `outputs/l1/base-zrckp-L-hotpotqa.json` and
`outputs/l1/base-zspk-L-musique.json` and `outputs/l1/base-zspk-L-hotpotqa.json`, already in the repository.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's, the base's card fit of the
split; a call is shown with its filed call where the re-call changed it):

| split | dataset | read | zrckp R@5 | zrckphe: delta [95% CI], call | zspk R@5 | zspkhe: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8019 | -0.0002 [-0.0012, +0.0007], WITHIN | 0.8009 | +0.0013 [+0.0004, +0.0022], WITHIN |
| L-musique | squad | in-domain | 0.9116 | -0.0008 [-0.0025, +0.0010], WITHIN | 0.9114 | -0.0021 [-0.0040, -0.0003], WITHIN |
| L-musique | musique | zero-shot | 0.5121 | -0.0706 [-0.0784, -0.0626], WITHIN (filed LOSS) | 0.5012 | -0.1036 [-0.1133, -0.0937], **LOSS** |
| L-musique | hotpotqa | in-domain | 0.9065 | -0.0036 [-0.0058, -0.0012], WITHIN | 0.9080 | -0.0008 [-0.0029, +0.0012], WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 | -0.0072 [-0.0088, -0.0054], WITHIN | 0.9383 | -0.0016 [-0.0030, -0.0002], WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 | +0.0181 [+0.0077, +0.0293], **GAIN** | 0.3229 | -0.0039 [-0.0130, +0.0051], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8044 | -0.0013 [-0.0020, -0.0005], WITHIN | 0.8040 | -0.0001 [-0.0010, +0.0008], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 | -0.0005 [-0.0024, +0.0014], WITHIN | 0.9135 | -0.0003 [-0.0023, +0.0014], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 | +0.0016 [-0.0032, +0.0062], WITHIN | 0.5754 | +0.0006 [-0.0044, +0.0059], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 | +0.0058 [+0.0029, +0.0087], WITHIN | 0.8499 | +0.0099 [+0.0068, +0.0130], WITHIN (filed GAIN) |
| L-hotpotqa | 2wiki | in-domain | 0.9350 | -0.0024 [-0.0039, -0.0007], WITHIN | 0.9333 | -0.0025 [-0.0042, -0.0010], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 | -0.0010 [-0.0114, +0.0102], WITHIN | 0.2680 | +0.0312 [+0.0211, +0.0415], **GAIN** |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a4he-<arm>.json` and `-hp.json`); and A4hs's R@5 deltas with the eight fixed blocks dropped instead
(zrckphs against zrckp, zspkhs against zspk; their own re-calls, `outputs/screen/scr-a4hs-<arm>-pair-recall.json`), on
the same read (a call is shown where it is not WITHIN):

| split | dataset | zrckphe FC@5 | zrckphe hit@1 | zspkhe FC@5 | zspkhe hit@1 | zrckphs (A4hs) R@5 | zspkhs (A4hs) R@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L-musique | metaqa | -0.0011 | +0.0026 | +0.0012 | +0.0004 | -0.0010 | +0.0016 |
| L-musique | squad | -0.0008 | -0.0019 | -0.0021 | -0.0021 | +0.0005 | +0.0005 |
| L-musique | musique | -0.0401 | -0.1324 | -0.0633 | -0.2362 | +0.0049 | +0.0253 |
| L-musique | hotpotqa | -0.0035 | -0.0054 | -0.0046 | +0.0008 | -0.0007 | -0.0043 |
| L-musique | 2wiki | -0.0187 | -0.0001 | -0.0021 | +0.0013 | +0.0015 | -0.0026 |
| L-musique | webqsp | +0.0080 | +0.0047 | -0.0047 | -0.0100 | -0.0016 | -0.0109 LOSS |
| L-hotpotqa | metaqa | -0.0021 | -0.0029 | +0.0004 | -0.0018 | -0.0010 | -0.0013 |
| L-hotpotqa | squad | -0.0005 | +0.0000 | -0.0003 | -0.0003 | -0.0003 | -0.0006 |
| L-hotpotqa | musique | +0.0046 | +0.0037 | -0.0037 | -0.0099 | -0.0008 | +0.0006 |
| L-hotpotqa | hotpotqa | +0.0130 | -0.0112 | +0.0192 | -0.0026 | -0.0117 | -0.0064 |
| L-hotpotqa | 2wiki | -0.0045 | -0.0046 | -0.0062 | -0.0029 | -0.0016 | +0.0036 |
| L-hotpotqa | webqsp | -0.0060 | -0.0020 | +0.0233 | +0.0060 | +0.0021 | +0.0092 |
