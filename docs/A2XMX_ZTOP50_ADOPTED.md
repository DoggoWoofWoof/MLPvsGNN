# A2xmx: ztop50's replacing form on the adopted arms, on both models

Declared 19 October 2026, before any number. The next form A2xm's verdict named (zrckpxm and zspkxm, both NOT_ADOPTED
with no GAIN; docs/A2XM_ZTOP50_REPLACE.md, sections 5 and 7, and `outputs/a2xm/screen-verdict.json`), under
docs/SCREENS.md section 3b and docs/VERDICT_AUDIT_2026_10_17.md. Code: `outputs/mp_unified/za2xmx.py`.

## 1. Why this, now

**A2xm** took every z-score zret takes against the pool's 50 best retrieved rows by rrf, in place of all of them, on
the lean pair (zrckp and zspk). Its re-calls (`outputs/screen/scr-a2xm-zrckpxm-pair-recall.json` and `-zspkxm-`), R@5
minus the base arm's card fit:

| read (floor) | zrckpxm (MLP) | zspkxm (GNN) |
| --- | --- | --- |
| L-musique, musique zero-shot (0.0720) | +0.0113 [+0.0053, +0.0173]: GAIN filed, WITHIN under the null | +0.0096 [+0.0028, +0.0164]: GAIN filed, WITHIN under the null |
| L-hotpotqa, webqsp zero-shot (0.0233) | −0.0351 [−0.0494, −0.0208], LOSS | −0.0141 [−0.0261, −0.0029]: LOSS filed, WITHIN under the null |
| L-hotpotqa, metaqa (0.0149) | −0.0012 [−0.0021, −0.0002], WITHIN | −0.0047 [−0.0059, −0.0036], WITHIN |

Both were NOT_ADOPTED with no GAIN. The verdict closed only the replacing form on zrckp and zspk.

**The rise shrinks as the model gets better.** On musique zero-shot (L-musique), the top-50 reference moved:
- step 1's model +0.1387, from 0.2696 (ztop50, `outputs/screen/scr-ztop50-pair-recall.json`), a GAIN;
- the lean pair +0.0198 (MLP) and +0.0275 (GNN), beside zret's z-scores (A2h's top-50 half, `scr-a2h-zrckpm-` and
  `scr-a2h-zspkm-pair-recall.json`), both WITHIN;
- the lean pair +0.0113 and +0.0096 in their place (A2xm), both WITHIN.

**The adopted arms change what the z-scores feed.** R1's zrckpr (the MLP) and G5's zspx (the GNN) are the models P1g-c
carries to P1g-b's pools. Their own re-calls against the lean pair (`outputs/screen/scr-r1-zrckpr-pair-recall.json`,
`scr-g5-zspx-pair-recall.json`), on the reads above:

| read (floor) | zrckpr − zrckp (R1) | zspx − zspk (G5) |
| --- | --- | --- |
| L-musique, musique zero-shot (0.0720) | +0.0076 [+0.0028, +0.0125]: GAIN filed, WITHIN under the null | +0.0061 [+0.0006, +0.0119], WITHIN |
| L-musique, webqsp zero-shot (0.0102) | +0.0135 [+0.0032, +0.0239], GAIN | −0.0025 [−0.0112, +0.0066], WITHIN |
| L-hotpotqa, webqsp zero-shot (0.0233) | −0.0064 [−0.0156, +0.0028], WITHIN | +0.0235 [+0.0133, +0.0327], GAIN |
| L-hotpotqa, musique in domain (0.0075) | +0.0034 [−0.0015, +0.0086], WITHIN | +0.0147 [+0.0095, +0.0198], GAIN |

zspx gains on webqsp zero-shot on L-hotpotqa, the read where the replacing form cost both lean models most. A2xm cannot
say whether the cut still adds, or still costs, on top of R1's head and zspx's three increments. The adopted arms are
the last models the top-50 reference has not met. A2xm's section 5 fixed it: the idea closes as a model input only if
A2xmx finds no GAIN either.

## 2. The form

A2xm's, unchanged: zret's reference, a pool's retrieved rows, is cut to its 50 best by rrf. Every within-pool z-score
zret takes is taken against the cut reference instead.

| | zret (the adopted arms today) | A2xmx |
| --- | --- | --- |
| a pool's reference rows | every retrieved row (rrf > 0) | the 50 best retrieved rows by rrf (rrf > 0, ties by row order) |
| z-scored against them | each block's columns, fixed and SEMB, and rrf's base z-score | the same |
| fewer than two reference rows, or a column constant over them (sd < 1e-6) | the whole pool's z-score | the same |

**Constants.** K = 50 (ztop50's, `lean_screen4.K`). rrf is the rank block's column: the first stage's reciprocal rank
fusion of the dense and SPLADE lists. Ties go by row order (two stable sorts, `lean_screen4.top_ref`). eps is zret's
1e-6. No parameter, column or hyperparameter is added.

**What does not change:**
- the raw columns and the keep flags;
- rmatch's chain term;
- the KT head: zrckpr's reads the row's own eight KT columns; zspx's reads them and the score's whole-pool z-score;
- R1's head on zrckpr, over the row's own nine columns (R1's att, pmax and smax per link family, from raw cosines,
  built before any row is scored; docs/R1_RESIDUAL_INPUT.md);
- zspx's three increments (its step with the gradient kept, G3's residual message, G4's second step;
  docs/G5_COMBINED_GNN.md). Each z-scores the score over the whole pool, not against zret's reference.

**How it is built.** Python's method order puts ztop50's forward (`lean_screen4.ZTop`) between the adopted arm's heads
and zret's forward. ChainMatch's call to its parent's forward now reaches ZTop. The arm classes have no body, and ZTop is
in neither base's method order.

**Where it changes nothing.** Where no pool has more than 50 retrieved rows, the arm is its base bit for bit. A2xm's
fits differ from their bases', so some of P1c2's pools have more than 50. How many is not counted here; the caches are
on the host.

**The MLP rule.** As in A2xm (its section 2). The reference is chosen by rrf alone, a first-stage input column cached
with the pool before any row is scored. No score is read, and the reference is the same whatever the model's weights.
What is normalised over a question's rows is what zret already normalises: input columns (the rank block, SEMB's
learned map of query and node, rrf), now against fewer of the same rows. R1's head reads only the row's own columns. So
zrckprxm keeps the rule: no message passing, no input from another row's score, nothing normalised over the question's
rows from a score. The selftest asserts it (section 6).

## 3. The arms

| arm | model | carve |
| --- | --- | --- |
| zrckprxm | zrckpr's model with ztop50's reference (MRO: ZRCKPR, ZRCKP, KTMixP, ZRM, ChainMatch, ZTop, ZRet) | zrckpr's (`zr1res.R1KTChainCarveZRC`) |
| zspxxm | zspx's model with ztop50's reference (MRO: ZSPX, ZSPR, ZSPK, ZProp, KTMix, ZRM, ChainMatch, ZTop, ZRet); zspx's three increments read the score it changes | zspx's (`zg3res.ResKTLinkCarveBase`) |

Each arm has its base's parameters exactly: the same names and shapes, and the same initial values at the same seed.

## 4. The fits and the call

A4fx's, on the same adopted arms:
- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-a2xmx-<arm>`, `-hp`), on P1c2's roots with
  P1c3's KT input and the base arm's own columns (R1's for zrckpr, G3's c and s for zspx), at the card fits' settings
  (variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split (zrckpr: scr-r1-zrckpr and -hp; zspx:
  scr-g5-zspx and -hp), step 1's fit beside.
- **Base screens for the re-call:** each base fit against step 1's. For zrckpr they are P1g-c's
  (`outputs/p1gc/base-pools-zrckpr-<split>.json`); for zspx, A9's (`outputs/a9/base-zspx-<split>.json`).
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), per arm.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A2xmx's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names each arm's next form (section 5).

**Reported beside, deciding nothing:**
- musique (Claim 1's read in domain, and the zero-shot read ztop50 gained on), with both arms side by side;
- metaqa (the read the replacing form lost on step 1's and rel's models) and webqsp (where A2xm's MLP lost);
- hit@1 and FC@5 per read;
- A2xm's deltas on the lean pair, on the same reads, in the verdict (from A2xm's two re-calls); ztop50's and A2h's top-50
  deltas, set beside each read when the results are filed.

**Serving cost.** One top-50 reference per question (two stable sorts over the batch's rows) in place of zret's rrf > 0
test; the z-scores are as many as zret's. Flagged, to be timed cold before any latency claim.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools, then its full run, its own declared file |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A2xmg, the learned back-off between the two references on the adopted arms: each z-score taken as z_ret + 2σ(⟨w, c⟩ + b) · (z_top − z_ret), w and b zero so the scale starts at 1 (A2xmx's form), c the pool's log size and retrieved share (input columns only), on both models |
| NOT_ADOPTED with no GAIN, the other arm with a GAIN | A2xmg on this model as well: the other arm's GAIN keeps the top-50 idea open on both models |
| NOT_ADOPTED with no GAIN, the other arm INCOMPLETE | decided with the other arm's re-call |
| NOT_ADOPTED with no GAIN on both arms | none for the top-50 reference as a model input, which closes: it has met both models beside zret's z-scores (A2, A2h) and in their place, on the lean pair (A2xm) and on the adopted arms (A2xmx), without a GAIN; ztop50's gain on step 1's model stays in the gap ledger, its shrinking trend beside |
| INCOMPLETE | the missing re-call is run; nothing else changes |

**Why these.**
- **A GAIN with a LOSS** is ztop50's old trade. The scale is per pool, and c says what kind of pool it is: metaqa's big
  KB pools against the passage pools, the cue ztop50's loss pointed at. Each model learns where the cut helps and falls
  back to zret's reference elsewhere. c is computed from the pool's rows and rrf, so the MLP rule holds.
- **One arm's GAIN** keeps the idea open on both models, so the arm without a GAIN goes to the back-off as well
  (docs/SCREENS.md section 3b: an idea with signal reaches both models before it closes).
- **No GAIN on either arm:** the top-50 reference has then met both models beside the z-scores, through a weight that
  starts at zero (A2h), and in their place, on the lean pair (A2xm) and on the adopted arms (A2xmx). It closes as a
  model input. ztop50's gain on step 1's model stays in the gap ledger.

No model change is adopted on today's pools (the user's rule of 13 October): an adopt is refitted on P1g-b's pools
before its full run.

## 6. How it runs

**Self-test.** `python outputs/mp_unified/za2xmx.py --selftest` passes before this commit (about 5 s). It runs on A2xm's
toy carves with the KT input, rmatch's chains, zlink's links, G3's c and s and R1's columns: 16 pools of 3 to 9 rows,
1 to 8 of them retrieved. It checks:
- **the arms:** the two MROs, with ztop50's forward just above zret's and in neither base's; the arm classes have no
  body; the base arms, A2xm's, zret's and ztop50's arms unchanged; K = 50; the zspx base screens are A9's;
- **the reference:** top_ref against a hand computation at K 50, 3 and 1. On the toy, the top 50 are all the retrieved
  rows. At K 3, a pool with more than 3 retrieved rows keeps its 3 best by rrf, and the others keep all of theirs;
- **the MLP rule,** spied in each arm's forward at K 3:
  - top_ref runs once, on the batch's rrf column itself (no gradient);
  - all three z-scores (the rank block, SEMB, rrf's base term) take that reference;
  - zret's whole retrieved reference is never formed (replaced, not beside);
  - random weights leave the reference unchanged;
  - in zrckprxm, nothing else is normalised over a pool: every within-pool normalisation takes an input column, none
    the score, and R1's head is the row's own;
  - in zspxxm, the KT head and the three increments z-score the score over the whole pool, as zspx's do: exactly four,
    each checked against the chain-matched score, the KT-mixed score, the score after the step and after G3's residual,
    and the arm's score is rebuilt from them bit for bit;
- **each arm against its base,** with every head's last layer set:
  - at K 50: no parameter added, no random number drawn, and bit for bit the base's scores (eval and training) and
    every gradient;
  - at K 3: bit for bit the base model with zret's reference replaced by the top-3 one, scores and every gradient, so
    the reference is all that differs;
  - l1's raw and keep columns are the base's; its z-score columns and the scores move only on the pools with more than
    3 retrieved rows;
- **toy fits** through lean_gpu's loop: at K 50 the base's fit bit for bit; a repeat bit for bit; at K 3 different;
- **the refusals:** train refuses another arm (zrckpr, zrckpxm, ztop50), another split, another arm's or split's fit
  name, and another cache root or device; read refuses a fit that is not A2xmx's, not its arm's or not its split's;
- relz's pair and re-call per arm against its base arm's card fits, stamped A2xmx's, with relz restored after;
- the verdict's outcomes and their next forms: an adopt, a GAIN with a LOSS, no GAIN beside the other arm's GAIN, no
  GAIN on both arms, no GAIN beside an incomplete arm, and five kinds of INCOMPLETE (missing, another stage's, another
  arm's, another declared file's, eleven reads); A2xm's re-calls beside, and "not found" when they are missing or
  another arm's.

**Commands** (each on the host through `retry_cmd.py` and `cuda_alloc.py`, as A4fx's):

    python outputs/mp_unified/za2xmx.py train --split L-musique --name scr-a2xmx-zrckprxm --arm zrckprxm --device cuda \
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za2xmx.py read --name scr-a2xmx-zrckprxm --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za2xmx.py compare --new outputs/screen/fits/scr-a2xmx-zrckprxm \
        --base outputs/screen/fits/scr-r1-zrckpr,outputs/step1/fits/L-musique --out outputs/screen/scr-a2xmx-zrckprxm
    python outputs/mp_unified/za2xmx.py pair --arm zrckprxm \
        --screens outputs/screen/scr-a2xmx-zrckprxm.json,outputs/screen/scr-a2xmx-zrckprxm-hp.json \
        --out outputs/screen/scr-a2xmx-zrckprxm-pair
    python outputs/mp_unified/za2xmx.py recall --arm zrckprxm --null N1,N2,N3,N4 \
        --pair outputs/screen/scr-a2xmx-zrckprxm-pair.json --out outputs/screen/scr-a2xmx-zrckprxm-pair-recall
    python outputs/mp_unified/za2xmx.py verdict

N1 to N4 are the four null files (`outputs/screen/scr-null-s1.json`, `-s2`, `-s1-hp`, `-s2-hp`). The same commands run
for L-hotpotqa (`--split L-hotpotqa`, names `-hp`, base scr-r1-zrckpr-hp and step 1's L-hotpotqa) and for zspxxm
(base scr-g5-zspx and -hp).

**Host items** (17, `a2xmx-*`), A4fx's 17 renamed, at A4fx's sizes:
- zrckprxm at R1's sizes: `--frac 0.30`, GPU share 0.32, 4.5 GB for the trains and 6 GB for the reads.
- zspxxm at A9's sizes for zspx: `--frac 0.40`, GPU share 0.42, 6 GB, and 15 GB for the L-hotpotqa train.
- A2xmx adds no parameter and no memory: the top-50 reference replaces zret's mask over the batch's rows.
- zrckprxm's trains and reads wait on the n19-r1c-* rebuilds of R1's columns, zspxxm's on the n19-res-* rebuilds of G3's
  c and s, by A4fx's lists. Both sets are the shared rebuilds of 19 October, all done; A2xmx adds no rebuild item.
- zrckprxm's re-call waits on P1g-c's base screens (p1gc-base-pools-zrckpr-L-musique and -L-hotpotqa); zspxxm's waits
  on A9's (a9-base-zspx and -hp).
- Compares take 1 CPU and 4 GB; pairs, re-calls and the verdict take 1 CPU and 2 GB. The verdict also reads A2xm's two
  re-calls.
- The items go behind P5c5's in the feeder's file, so P1g-c's redo chain and P5c5 keep the host's memory first.

**Disk:** about 0.3 GB of fits and reads, as A2xm's.

The encoder stays frozen. webqsp never trains. Test splits are never read. No LLM or transformer, and no new graph.

## 7. Results

Not run yet.
