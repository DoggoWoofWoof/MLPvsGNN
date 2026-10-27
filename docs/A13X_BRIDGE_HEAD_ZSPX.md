# A13x: zbr's bridge head on the adopted GNN

Declared 19 October 2026, before any number. The next form A13's verdict named (zspkb NOT_ADOPTED with no GAIN;
outputs/a13/screen-verdict.json, fixed in docs/A13_BRIDGE_COLUMNS_GNN.md section 5). Under docs/SCREENS.md section 3b.
Code: `outputs/mp_unified/za13x.py` (the arm, the screen and the call). Filed 27 October:
NOT_ADOPTED with no GAIN and no LOSS; the next form is A13xs (section 5).

## 1. Why this, now

**What A13 found.** A13 put zbr's bridge head on zspk, before its one propagation step (zspkb). After the seed-null
re-call it had 0 GAIN and 0 LOSS (outputs/screen/scr-a13-zspkb-pair-recall.json). Two reads moved. Both were filed
GAINs, and both fell inside the null's floor:

| split | read | zspk R@5 | zspkb R@5 | delta [95% CI] | floor | filed → re-call |
| --- | --- | ---: | ---: | --- | ---: | --- |
| L-musique | musique (zero-shot) | 0.5012 | 0.5150 | +0.0138 [+0.0084, +0.0191] | 0.0720 | GAIN → WITHIN |
| L-hotpotqa | hotpotqa (zero-shot) | 0.8499 | 0.8579 | +0.0080 [+0.0050, +0.0110] | 0.0200 | GAIN → WITHIN |

The other ten reads were WITHIN as filed. The largest drop was 2wiki in domain on L-hotpotqa, -0.0043 [-0.0058,
-0.0028] (floor 0.0075). webqsp zero-shot fell 0.0066 and 0.0056, inside floors of 0.0102 and 0.0233.

So the head lifted the two zero-shot multi-hop reads, but not past the null.

**Why that does not close the idea.**
- zspk's step reads the score without the gradient. zspkb's head shifted a row's score, and the step carried the shift
  to the row's neighbours. But the head never learned from what the step did with it.
- zspx, the adopted GNN (G5, ADOPT_FOR_FULL_RUN), keeps the gradient through its first step (G4's zspkg). So here the
  head learns through the step. Two more steps follow, G3's residual message and G4's second step, and both read the
  shifted score.
- Depth paid where the bridge matters. zspx's two GAINs against zspk were both on L-hotpotqa: musique in domain,
  +0.0147 [+0.0095, +0.0198], and webqsp zero-shot, +0.0235 [+0.0133, +0.0327] (docs/G5_COMBINED_GNN.md, section 5).
- Under section 3b, an idea stays open until it reaches both models. zbr's columns reached the MLP on zrc (round 29,
  NO_GAIN) and the GNN on zspk (A13). They have not reached the adopted GNN.

**The base is zspx,** decided against zspx's card fits (scr-g5-zspx, scr-g5-zspx-hp), as in A9 and A10.

## 2. The form

**zspxb** is zspx (zg5mix.ZSPX on G3's carve) whose score adds zbr's head over zbr's 13 inputs, before zspx's first
step reads the score (A13's placement, and R1's):

    s0  = zrm's score + the KT head                        (zspx's, unchanged)
    s0' = s0 + head(bridge_inputs(rrf's order, the pool's links, dense_cos))
    s1  = s0' + h_sp(prop_inputs_grad(s0'))                zspx's first step, gradient kept
    s2  = s1 + h_rs(res_inputs(s1, c, s))                  G3's residual message, no gradient
    out = s2 + h_p2(prop_inputs(s2))                       G4's second step, no gradient

**The inputs** are `zbr.bridge_inputs`, unchanged. Per row:
- log1p of its links from RRF's top 1, top 2 and top 5, per link family (9);
- log1p of its two-step paths from RRF's top 5, over any family (1);
- the bridge flag: linked to RRF's top 2 and not one of them (1);
- the bridge rank: 1 / (1 + the row's rank by dense_cos among its question's flagged rows), 0 when not flagged (1);
- the source rank: 1 / (1 + the best RRF rank among its neighbours), 0 when it has none (1).

They are fixed before any row is scored: RRF's order (the batch's base_z), zlink's pool edges, and the row's cached
dense_cos column. The head reads no score.

**The head** is A13's (`za13br.BrMix`), unchanged: zbr's Linear(13, 32), GELU, Linear(32, 1), 481 parameters. The first
layer is drawn from its own generator (seed + 2901, zbr's draw); the last starts at zero. So at the start zspxb is zspx
bit for bit, and no draw of zspx's moves.

**The class** is `ZSPXB(zg5mix.ZSPX, za13br.BrMix)`. Its class order is ZSPX, ZSPR, ZSPK, ZProp, BrMix, KTMix, ZRM.
zspx's forward asks for the score that comes before ZProp's step. In this order that is BrMix's: the KT score plus the
bridge head. P1g-c's zspxr places R1's head the same way.

**What learns through what.**
- The head's shift enters the first step's inputs with the gradient kept. So the head learns from its own term and
  from how the step passes the shift to the row's neighbours.
- G3's residual message and G4's second step read the shifted score without the gradient, as in zspx.

**The carve** is zspx's (G3's: P1c3's KT link carve with c and s beside), with each row's dense_cos beside its links,
by A13's `cos_carve`, unchanged. Every block, row, edge, KT column, residual and gold is unchanged. It subclasses zspx's
carve, so P1c2's roots (patched on zlink's and rmatch's carves) reach it. No build: dense_cos is a column of P1c2's
cache, which zspx's carve already loads.

| constant | value |
| --- | --- |
| inputs | 13 (zbr.N_IN) |
| hidden | 32 (zbr.HID) |
| the head's generator | seed + 2901 (zbr.SEED_OFF); zspx's heads use 2301 (step), 3301 (KT), 3401 (residual), 4401 (second step) |
| parameters added | 481 |
| everything else | zspx's: card settings, seed, batches, KT input, residuals, loss and loop |

**GNN only.** zbr's MLP form closed on zrc (round 29), before the MLP's model changed. A gain here brings the head to
the MLP next (section 5). zbr's inputs are legal there: none reads a score.

## 3. The arms

| arm | the model | the carve | added |
| --- | --- | --- | --- |
| zspxb | zspx + zbr's head before its first step | zspx's (G3's) + dense_cos beside the links | 481 parameters |

**Cost.** zspx's own fit times are not filed in this repository. G3's and G4's fits took 47 to 54 minutes each on the
card, and their reads about 5 minutes (docs/G5_COMBINED_GNN.md). On the MLP, zbr's head made zrc's card fits 1.28 and
1.41 times longer (2,777 s and 2,984 s against 2,166 s and 2,114 s), at the same peak GPU memory. Part of that was
loading zlink's links, which zspx already pays (docs/A13_BRIDGE_COLUMNS_GNN.md, section 3). So zspxb should cost a
little more than zspx, at zspx's memory.

Serving: per question, two sorts of its pool and three passes over its edges, all from fixed inputs. Any latency figure
for zspxb is cold, timed in its own declared stage.

## 4. The fits and the call

- **Fits:** one card fit on L-musique and one on L-hotpotqa (`scr-a13x-zspxb`, `-hp`), on P1c2's roots with P1c3's KT
  input and G3's residuals. Card fit settings: variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128 (zspx's).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against zspx's card fit of the same split (scr-g5-zspx, scr-g5-zspx-hp), which decides.
  Step 1's fit is reported beside.
- **The re-call's base screens:** A9's, zspx's card fits compared with step 1's of the same split
  (outputs/a9/base-zspx-<split>.json).
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md). GAIN, LOSS and
  WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

- **ADOPT_FOR_FULL_RUN:** zspxb's re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A13x's re-call of zspxb, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- webqsp and metaqa, both splits (webqsp never trains; metaqa trains on both splits);
- musique in domain (L-hotpotqa): Claim 1's read, and one of zspx's two GAINs;
- the two reads where A13's filed GAINs were: musique zero-shot (L-musique) and hotpotqa zero-shot (L-hotpotqa);
- squad in domain, the one dataset where G1a's rule was BELOW;
- hit@1 and FC@5 per read.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | zspxb fitted again on P1g-b's pools, then its full run beside zspx's; and A13xm, zbr's head on zrckpr (R1's MLP, ADOPT_FOR_FULL_RUN at its screen), since zbr's MLP form closed on zrc before the MLP's model changed (each its own declared file) |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A13xb, the back-off form: the bridge head bounded (0.2 tanh(h / 0.2), the MP-unified track's bound) on zspx, so the steps cannot lean on it where the rule fails (squad, comparison questions) (one declared file) |
| NOT_ADOPTED, with no GAIN | A13xs, the columns inside the step: zbr's 13 inputs read by zspx's first step head beside zprop's 10 (its first layer widened, the new weights zero at the start, so it starts as zspx). The step then weighs a row's neighbours by the row's bridge quantities, instead of adding a shift. It is P1g-c's zspxs form, for zbr's columns (one declared file) |
| INCOMPLETE | the missing re-call is run; nothing else changes |

Why A13xs and not a close on no GAIN: so far zbr's columns reached the GNN only as a shift added to the score. A step
that reads them is a form not yet tried, and it can back off per row.

No model change is adopted on today's pools (the user's rule of 13 October).

## 6. How it runs

**Self-test, passed before this commit.** `python outputs/mp_unified/za13x.py --selftest` (5 s) checks, on a toy carve
with G3's residuals, P1c3's KT input and a dense_cos column:
- **The arm.** zspxb is zspx plus A13's BrMix (zbr's head: its shape, sizes and seed offset, apart from zspx's four
  heads') on zspx's carve with dense_cos beside the links; class order ZSPX, ZSPR, ZSPK, ZProp, BrMix, KTMix, ZRM.
  zspx's, zspkb's, zspk's and zbr's own arms are unchanged. P1c2's roots reach A13x's carve; its KT and residual roots
  are zspx's. A9's base screens are the re-call's.
- **The carve:** its batch is zspx's (rows, links, KT columns, residuals, base_z and golds), plus each row's cached
  dense_cos column.
- **The inputs** are zbr's by hand (Python sorts and loops over each question's edge list).
- **The start.** From one seed, zspxb is zspx's state plus zbr's head as zbr and zspkb draw it, the last layer zero;
  the global generator ends where zspx leaves it. Its forward is zspx's bit for bit, in eval and in training with the
  same dropout draws, with zspx's three step heads and its KT head moved.
- **The placement.** With the head moved, zspxb is zspx's chain over (zspx's pre-step score + zbr's head): the first
  step, the residual message and the second step all read the shifted score, and zspxb minus zspx is more than the
  head's own term. Without links it is zspx's scorer; links without dense_cos are refused.
- **The gradient.** The head's gradient is the explicit chain's, and it differs from the gradient under a first step
  that reads the score detached (zspkb's case): the head learns through the step.
- **Toy fits** through lean_gpu's loop: a repeat is bit for bit; the head trains; through the first step the shared
  weights match zspx's own fit (the head adds 0 there), and every state after it differs; without links the fit is
  zspx's bit for bit on every shared key, the head untouched.
- **The refusals.** train refuses zspx's, zspkb's and zbr's arms, another split, the wrong fit name, another cache root
  and the CPU. read refuses zspx's fit, zspkb's fit, an unstamped fit and a zspxb fit stamped A13's.
- **The rest.** relz's pair and re-call stamped A13x's, with relz restored after; the verdict's outcomes and next
  forms, and INCOMPLETE for a missing re-call, another stage's, or fewer than twelve reads.

**Host items** (`a13x-*`, 9 in all), each starting when the GPU frees:
- 2 trains and 2 reads, sized as A9's from zspx's runs: `--frac 0.40` (share 0.42); memory: trains 6 GB (-hp 15 GB),
  reads 6 GB;
- the trains wait on the rebuild of G3's residual arrays for their split's fit carves (`n19-res-*`, by A9's lists),
  the reads on the six s1eval carves';
- 2 compares, the pair, the re-call (after A9's base compares, `a9-base-zspx` and `-hp`) and the verdict.

No build: zspxb reads zspx's carves and computes its inputs in the batch.

The commands. The -hp fit is the same with `--split L-hotpotqa` and the `-hp` names. On the host each GPU command runs
under `retry_cmd.py --tries 3 --wait 90 --cuda` and `cuda_alloc.py --frac 0.40 --gc 0.9`. N1 to N4 are the four null
files: outputs/screen/scr-null-s1.json, scr-null-s2.json, scr-null-s1-hp.json and scr-null-s2-hp.json.

    python outputs/mp_unified/za13x.py train --split L-musique --name scr-a13x-zspxb --arm zspxb --device cuda --host \
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za13x.py read --name scr-a13x-zspxb --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za13x.py compare --new outputs/screen/fits/scr-a13x-zspxb \
        --base outputs/screen/fits/scr-g5-zspx,outputs/step1/fits/L-musique --out outputs/screen/scr-a13x-zspxb
    python outputs/mp_unified/za13x.py pair \
        --screens outputs/screen/scr-a13x-zspxb.json,outputs/screen/scr-a13x-zspxb-hp.json \
        --out outputs/screen/scr-a13x-zspxb-pair
    python outputs/mp_unified/za13x.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-a13x-zspxb-pair.json \
        --out outputs/screen/scr-a13x-zspxb-pair-recall
    python outputs/mp_unified/za13x.py verdict

The encoder stays frozen. webqsp never trains. Test splits are never read. No LLM or transformer, and no new graph.

## 7. Results

Filed 27 October 2026 from the host's records (`outputs/a13x/screen-verdict.json`, written 2026-10-20T09:34:29Z; the
re-call `outputs/screen/scr-a13x-zspxb-pair-recall.json`, host paths made relative). **NOT_ADOPTED, with no GAIN and no
LOSS:** with zbr's bridge head before zspx's first step, no read moves past its floor, and the two filed LOSSes, musique
zero-shot on L-musique and webqsp zero-shot on L-hotpotqa, are inside their floors under the re-call.

- **zspxb: 0 GAIN, 0 LOSS.** Two filed LOSSes are WITHIN under the re-call: musique zero-shot on L-musique, -0.0119
  [-0.0176, -0.0063] against its 0.0720 floor (FC@5 -0.0153), and webqsp zero-shot on L-hotpotqa, -0.0133 [-0.0216,
  -0.0055] against its 0.0233 floor (FC@5 -0.0080). Three more intervals exclude 0, all on L-hotpotqa and inside their
  floors: squad +0.0023 [+0.0006, +0.0040], musique in domain -0.0065 [-0.0111, -0.0017] and 2wiki -0.0015 [-0.0030,
  -0.0000] (its upper end below 0 before rounding: the record holds -0.0).
- **Beside: zbr's head moves musique zero-shot on L-musique the other way on zspx.** On zspk (A13's zspkb) it rose
  +0.0138; on zspx it falls -0.0119. Every other read of A13's arm stayed within 0.0080 of its base.

Not adopted. This closes only this form: zbr's bridge head added before zspx's first step (docs/SCREENS.md section 3b).
The next form, fixed in section 5 and recorded in the verdict, is A13xs, the columns inside the step: zbr's 13 inputs
read by zspx's first step head beside zprop's 10 (its first layer widened, the new weights zero at the start, so it
starts as zspx). The step then weighs a row's neighbours by the row's bridge quantities, instead of adding a shift. It
is P1g-c's zspxs form, for zbr's columns (one declared file).

The floors are the seed null of 7 October (docs/SCREENS.md section 2: `outputs/screen/scr-null-s1.json`,
`scr-null-s2.json` and their `-hp` files), not N2's fresh floors (docs/N2_FRESH_SEED_NULL.md).

Records, filed with this section: the verdict `outputs/a13x/screen-verdict.json` and `.md`; for the arm `zspxb`, the
compares `outputs/screen/scr-a13x-zspxb.json` and `-hp.json`, the pair `-pair.json` and its re-call `-pair-recall.json`,
each with its `.md`. The host paths were made relative in each re-call's two `rel` entries and nowhere else, so the
`recall_sha256` each verdict holds is the host file's, from before that change. The re-calls' nulls also read the base
screens `outputs/a9/base-zspx-L-musique.json` and `outputs/a9/base-zspx-L-hotpotqa.json`, fetched from the host with
this filing (A9's own filing had left them there).

The twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's, zspx's card fit of the split; a
call is shown with its filed call where the re-call changed it):

| split | dataset | read | zspx R@5 | zspxb: delta [95% CI], call |
| --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8030 | +0.0002 [-0.0005, +0.0011], WITHIN |
| L-musique | squad | in-domain | 0.9092 | +0.0004 [-0.0015, +0.0023], WITHIN |
| L-musique | musique | zero-shot | 0.5073 | -0.0119 [-0.0176, -0.0063], WITHIN (filed LOSS) |
| L-musique | hotpotqa | in-domain | 0.9074 | +0.0014 [-0.0008, +0.0034], WITHIN |
| L-musique | 2wiki | in-domain | 0.9351 | -0.0011 [-0.0025, +0.0003], WITHIN |
| L-musique | webqsp | zero-shot | 0.3204 | +0.0065 [-0.0003, +0.0136], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8039 | -0.0003 [-0.0011, +0.0004], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9129 | +0.0023 [+0.0006, +0.0040], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5901 | -0.0065 [-0.0111, -0.0017], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8547 | +0.0013 [-0.0013, +0.0040], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9312 | -0.0015 [-0.0030, -0.0000], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2914 | -0.0133 [-0.0216, -0.0055], WITHIN (filed LOSS) |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a13x-zspxb.json` and `-hp.json`); and A13's R@5 deltas with the head on zspk (zspkb, its own
re-call `outputs/screen/scr-a13-zspkb-pair-recall.json`, against zspk), on the same read:

| split | dataset | zspxb FC@5 | zspxb hit@1 | zspkb (A13) R@5 |
| --- | --- | ---: | ---: | ---: |
| L-musique | metaqa | +0.0005 | -0.0003 | +0.0003 |
| L-musique | squad | +0.0004 | +0.0001 | -0.0004 |
| L-musique | musique | -0.0153 | +0.0025 | +0.0138 |
| L-musique | hotpotqa | +0.0035 | -0.0068 | -0.0001 |
| L-musique | 2wiki | -0.0040 | +0.0038 | -0.0011 |
| L-musique | webqsp | +0.0067 | -0.0040 | -0.0066 |
| L-hotpotqa | metaqa | -0.0010 | -0.0041 | +0.0013 |
| L-hotpotqa | squad | +0.0023 | +0.0004 | -0.0005 |
| L-hotpotqa | musique | -0.0128 | -0.0041 | -0.0018 |
| L-hotpotqa | hotpotqa | +0.0011 | +0.0080 | +0.0080 |
| L-hotpotqa | 2wiki | -0.0078 | -0.0060 | -0.0043 |
| L-hotpotqa | webqsp | -0.0080 | +0.0060 | -0.0056 |
