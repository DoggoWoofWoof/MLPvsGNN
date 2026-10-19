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

Not run yet.
