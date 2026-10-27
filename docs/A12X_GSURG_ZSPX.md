# A12x: the adopted GNN trained with gradient surgery

Declared 19 October 2026, before any number. The next form A12's verdict named (zspkgs NOT_ADOPTED with no GAIN;
outputs/a12/screen-verdict.json, fixed in docs/A12_GSURG_GNN.md section 5), under docs/SCREENS.md section 3b and the
twelfth reopening of docs/VERDICT_AUDIT_2026_10_17.md (section 2, row 12). Code: `outputs/mp_unified/za12x.py` (the
arm, the screen and the call). Filed 27 October: NOT_ADOPTED with no GAIN and no LOSS; the next form is A12m
(section 5).

## 1. Why this, now

**A12's verdict.** zspkgs, zspk trained with gsurg's loop, was NOT_ADOPTED: 0 GAIN and 1 LOSS among its twelve reads
(outputs/a12/screen-verdict.json, written 2026-10-18T10:57:43Z). The reads that moved, from its re-call
(outputs/screen/scr-a12-zspkgs-pair-recall.json; delta = zspkgs's R@5 minus zspk's):

| split | dataset | read | zspk R@5 | delta [95% CI] | floor | call (filed -> re-call) |
| --- | --- | --- | ---: | --- | ---: | --- |
| L-musique | musique | zero-shot | 0.5012 | -0.0190 [-0.0255, -0.0129] | 0.0720 | LOSS -> WITHIN |
| L-musique | webqsp | zero-shot | 0.3229 | -0.0137 [-0.0220, -0.0057] | 0.0102 | **LOSS** |
| L-musique | metaqa | in-domain | 0.8009 | +0.0016 [+0.0006, +0.0026] | 0.0075 | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9135 | -0.0034 [-0.0055, -0.0013] | 0.0075 | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9333 | +0.0023 [+0.0006, +0.0039] | 0.0075 | WITHIN |

The other seven reads' intervals hold zero.

- On L-musique the surgery cost both zero-shot reads. musique fell 0.0190, inside its wide floor (0.0720). webqsp fell
  0.0137, past its floor (0.0102).
- On L-hotpotqa no read moved past its floor. Its zero-shot reads rose a little, their intervals holding zero:
  hotpotqa +0.0019 [-0.0011, +0.0051], webqsp +0.0040 [-0.0060, +0.0144].
- So on zspk the loop added no gain and cost one zero-shot read. That closes only the zspk form (section 3b).

**Why zspx.** A12's section 5 named this form for no GAIN: "A12x, gsurg's loop on zspx (the adopted GNN, three steps,
where depth paid)".
- zspx is the GNN that passed its screen: G5's ADOPT_FOR_FULL_RUN, 2 GAIN and 0 LOSS against zspk
  (outputs/g5/screen-verdict.json, written 2026-10-17T05:27:11Z). Its GAINs: L-hotpotqa musique in domain
  +0.0147 [+0.0095, +0.0198], and L-hotpotqa webqsp zero-shot +0.0235 [+0.0133, +0.0327].
- zspx has three steps over the score where zspk has one. Depth paid there: G4's two arms (zspk's step end to end,
  and a second step) each gained on L-hotpotqa's musique read, and zspx's three changes together gained more on it
  (+0.0147) than any one alone (+0.0093 to +0.0115; docs/G5_COMBINED_GNN.md).
- A12 took the loop to the GNN because the step is where the datasets' gradients may pull apart most: the step reads
  the link families, and their mix differs most between datasets. zspx has three such steps.
- On zspx, the two reads zspkgs lost stand at 0.5073 (L-musique musique, +0.0061 [+0.0006, +0.0119] over zspk) and
  0.3204 (L-musique webqsp, -0.0025 [-0.0112, +0.0066]), both WITHIN (outputs/screen/scr-g5-zspx-pair-recall.json).
  They are reported beside (section 4).

## 2. The form

**zspxgs** is zspx (`zg5mix.ZSPX` on G3's residual KT link carve) trained inside lean_screen5's `gsurg_loop`, exactly
as A12 trained zspk:
- lean_gpu's training loop (`fit_variant`) is swapped for `fit_gsurg` while the fit trains, and restored after;
- `fit_gsurg` is lean_gpu's loop line for line, plus the surgery;
- each epoch's surgery counts (pairs, conflicts, share, per dataset pair) land in train.json's curve.

**The surgery** (`lean_screen5.project`, unchanged). At each 32-question step, each training dataset present in the
step updates in turn:
1. its loss on its questions of the step is taken in training mode, and its gradient g_i is one flat vector over
   every parameter;
2. each other present dataset's gradient g_j is taken on that dataset's own questions of the step, the same way but
   in eval mode, so no dropout draw moves;
3. for each g_j, in the order of the step's datasets, when g_i . g_j < 0 (dot products in float64), g_i loses its
   component along g_j: g_i <- g_i - (g_i . g_j / |g_j|^2) g_j, on g_i as projected so far;
4. the optimizer steps with the projected g_i.

A g_j that is not finite is left out and counted.

**Where the loop sits: outside zspx's three steps.**

    s0 = zrc's scorer + the KT head          zspk's score before its step
    s1 = s0 + h_sp(prop_inputs_grad(s0))     step 1, its inputs with the gradient kept (zspkg's)
    s2 = s1 + h_rs(res_inputs(s1, c, s))     step 2, G3's residual message, inputs without gradient
    s3 = s2 + h_p2(prop_inputs(s2))          step 3, G4's second step, inputs without gradient
    loss = listwise(s3)                      one loss per dataset per step, on zspx's score
    g = the loss's gradient over (scorer, KT head, h_sp, h_rs, h_p2) as one vector: the surgery acts here

- Each dataset's forward runs zspx whole, and its loss is taken on s3, the score zspx reads and serves.
- The surgery acts on one vector: the gradient of every parameter at once. Each other dataset's vector is taken the
  same way, on s3.
- No step has a surgery of its own. No part of the model is left out of the dot products.

**Why this placement: it is A12's, applied to zspx.** A12's doc fixes it:
- "`fit_gsurg` is lean_gpu's loop line for line, plus the surgery." The loop sees the model's parameters as one list
  and the loss on the model's score. It knows nothing of steps. On zspk that put the scorer, the KT head and the
  step's head in one vector, with the loss on zspk's final score. The same swap on zspx puts every parameter in one
  vector, with the loss on s3.
- "No parameter, column or hyperparameter is added." A surgery per step would have to choose which parameters make
  each step's block, and where each step's loss is taken (s1, s2 or s3). Those are new choices, so a different form.
- "Training only." Reads and serving are zspx's, so any latency figure for zspxgs is zspx's, cold.
- A12 went to the GNN because "the step may be where datasets' gradients pull apart most". With the whole gradient in
  every dot product, each of zspx's three step heads has its part in every dataset's vector, so a conflict in any step
  counts. zspx's first step keeps its gradient, so the scorer's part also carries what flows back through it.

**Nothing else changes.** zspx's model, carve, card, seed, batches and settings are used unchanged:
- card fit settings: variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128 (zspx's, which are zspk's);
- the heads as filed: the KT head (177 parameters, seed + 3301) and the three step heads (385 parameters each; seeds
  + 2301, + 3401 and + 4401), each with its last layer at zero at the start;
- the carve: G3's `ResKTLinkCarveBase`, P1c3's KT link carve with G3's per-row cosine c and per-edge cosine s beside,
  on P1c2's roots.

On one training dataset there is no other gradient, and the loop is lean_gpu's bit for bit.

**GNN first.** The MLP's line had this loop (step 1's model, zgs on zret, zrg on zrm), but never the MLP's current
model, zrckp. Section 5 brings the loop there on an adopt and on no GAIN.

## 3. The arms

One arm.

| arm | the model | the carve | the loop | decided against |
| --- | --- | --- | --- | --- |
| zspxgs | zspx, unchanged | zspx's (G3's residual KT link carve) | lean_screen5.fit_gsurg, on the whole gradient, outside the three steps | zspx's card fits (scr-g5-zspx, scr-g5-zspx-hp) |

zspx's, zspk's and gsurg's own arms are unchanged.

## 4. The fits and the call

- **Fits:** one card fit on L-musique and one on L-hotpotqa (`scr-a12x-zspxgs`, `-hp`), on P1c2's roots, P1c3's KT
  input and G3's residual builds.
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against zspx's card fit of the same split (scr-g5-zspx, scr-g5-zspx-hp), which decides.
  Step 1's fit is reported beside.
- **The re-call's base screens:** A9's, zspx's card fits compared with step 1's of the same split
  (outputs/a9/base-zspx-<split>.json).
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md). GAIN, LOSS and
  WITHIN are docs/SCREENS.md's, on R@5.

**A fit is A12x's only if the loop ran.** `train` stops when the fit's curve lacks the surgery counts in any epoch, and
`read` refuses such a fit (zrg's check, as in A12). read also refuses zspx's fit, A12's zspkgs fit and a fit without
A12x's stamp.

**The residual arrays.** zspx's card fits read G3's first build of the residual arrays. The host's clean-up removed
those arrays and kept their records. The n19-res-* items rebuild them with G3's own build commands, from P1c2's looks
and links (ba4a7c0). The rebuilt arrays' sha256s are compared with the saved records before any screen on them is
filed (ba4a7c0). Each fit stamps the records it read (residual, KT, chain and link records).

### The rule, fixed before any number

- **ADOPT_FOR_FULL_RUN:** zspxgs's re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A12x's re-call of zspxgs, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- webqsp and metaqa, both splits (webqsp never trains; metaqa trains on both splits);
- musique in domain (L-hotpotqa): Claim 1's read, and zspx's GAIN;
- L-musique's two zero-shot reads (musique and webqsp), where zspkgs lost on zspk, beside A12's deltas there;
- hit@1 and FC@5 per read;
- each fit's surgery counts: the share of conflicting pairs per epoch, and which dataset pairs conflict.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | zspxgs fitted again on P1g-b's pools, then its full run beside zspx's (its own declared file); and A12m, zrckp (the MLP's lean model) trained with gsurg's loop (its own declared file), so the loop reaches both of today's models |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A12xh, the back-off form on zspx: half the surgery. Each step's gradient loses half of its component along each conflicting dataset's gradient, so a conflict is damped, not removed (one declared file) |
| NOT_ADOPTED, with no GAIN | A12m, gsurg's loop on zrckp, the MLP's lean model (one declared file, whose own section 5 names what follows it) |
| INCOMPLETE | the missing re-call is run; nothing else changes |

**Why A12m on no GAIN, and not a closure.** A12 sent the loop to zrckp only after a GNN gain. With no GAIN on both GNN
forms (zspk and zspx), the loop's signal sits only on the MLP's line: step 1's model (ADOPT at the re-grade,
8 October) and zrg on zrm (webqsp zero-shot +0.0273 and musique +0.0087 beside, 9 October). The MLP's model has
changed since, and zrckp has never had the loop. Under SCREENS.md section 3b an idea closes only after it has reached
both of today's models, so the MLP goes next.

No model change is adopted on today's pools (the user's rule of 13 October).

## 6. How it runs

**Self-test, passed before this commit.** `python outputs/mp_unified/za12x.py --selftest` (7.4 s) checks:
- **The arm.** zspxgs is zspx's model on G3's residual KT link carve; zspx's, zspk's and gsurg's own arms are
  unchanged.
- **The swap.** Training swaps in gsurg's loop and zspx's model and carve, and restores all three after.
- **Toy fits** on G3's toy residual KT link carve (links, KT columns, c and s), as two datasets whose golds lean
  opposite ways on one rank column (zrg's device), so their gradients conflict:
  - the first conflict comes in epoch 0, and gsurg's loop moves every state from there on; a repeat is bit for bit;
  - without the surgery (the other gradients still taken) the loop is zspx's own loop bit for bit;
  - on one training dataset it is zspx's bit for bit, with no pair;
  - the KT head and all three step heads (sp, rs, p2) train.
- **The placement.** All 24 surgeries of a toy fit act on one vector of all 172,425 parameters of the toy model. The
  first one's two vectors equal zspx's whole gradients computed by hand, bit for bit: the loss on s3, the dataset's
  own in training mode and the other's in eval mode. The scorer, the KT head, sp, rs and p2 each have a non-zero part
  in both.
- **The curve check.** A curve with surgery counts in every epoch passes; lean_gpu's curve, and a missing train.json,
  do not.
- **The refusals.** train refuses zspx's arm, A12's arm (zspkgs), no arm, another split, the wrong fit name, another
  cache root (P1g-b's) and the CPU. read refuses zspx's fit, A12's fit, an unstamped fit and a fit trained outside the
  loop.
- **The rest.** relz's pair and re-call stamped A12x's, with relz restored after; the verdict's outcomes and next
  forms; INCOMPLETE for a missing re-call and for A12's own.

**Host items** (`a12x-*`, 9 in all), each starting when the GPU frees:
- 2 trains and 2 reads, sized as A9's from zspx's runs: `--frac 0.40` (share 0.42); memory: trains 6 GB
  (-hp 15 GB), reads 6 GB;
- the trains wait on the rebuild of G3's residual arrays for their split's fit carves (`n19-res-*`, by A9's lists:
  11 parts for L-musique, 16 for L-hotpotqa); the reads wait on their train and on the six s1eval carves' 22 parts;
- 2 compares, the pair, the re-call (after A9's base compares, `a9-base-zspx` and `-hp`) and the verdict.

No build of A12x's own: zspxgs reads zspx's carves, and n19-res-* bring back the residual arrays.

The commands. The -hp fit is the same with `--split L-hotpotqa` and the `-hp` names. On the host each GPU command runs
under `retry_cmd.py --tries 3 --wait 90 --cuda` and `cuda_alloc.py --frac 0.40 --gc 0.9`. N1 to N4 are the four null
files: outputs/screen/scr-null-s1.json, scr-null-s2.json, scr-null-s1-hp.json and scr-null-s2-hp.json.

    python outputs/mp_unified/za12x.py train --split L-musique --name scr-a12x-zspxgs --arm zspxgs --device cuda \
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za12x.py read --name scr-a12x-zspxgs --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za12x.py compare --new outputs/screen/fits/scr-a12x-zspxgs \
        --base outputs/screen/fits/scr-g5-zspx,outputs/step1/fits/L-musique --out outputs/screen/scr-a12x-zspxgs
    python outputs/mp_unified/za12x.py pair \
        --screens outputs/screen/scr-a12x-zspxgs.json,outputs/screen/scr-a12x-zspxgs-hp.json \
        --out outputs/screen/scr-a12x-zspxgs-pair
    python outputs/mp_unified/za12x.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-a12x-zspxgs-pair.json \
        --out outputs/screen/scr-a12x-zspxgs-pair-recall
    python outputs/mp_unified/za12x.py verdict

**Cost.** zrg's fits took 4.9 times zrm's training time at the same peak GPU memory (A12's doc): each step takes every
other present dataset's gradient. zspx's own fit times are not in the laptop's records. G3's and G4's fits, each one of
zspx's changes on zspk, took 47 to 54 minutes on the card (G5's doc). So each zspxgs fit should take about 3.8 to 4.4
hours, more if zspx's own fits ran longer. The two fits can run at once (share 0.42 each). Reads take about 5 minutes
each.
- `fit_gsurg` holds every present dataset's batch of a step at once, where lean_gpu's loop builds them one at a time.
  A12's fits ran with this loop at zspk's own sizes, so zspx's own sizes are kept here.
- **Disk:** about 0.05 GB.

The encoder stays frozen. webqsp never trains. Test splits are never read. No LLM or transformer, and no new graph.

## 7. Results

Filed 27 October 2026 from the host's records (`outputs/a12x/screen-verdict.json`, written 2026-10-20T10:18:22Z; the
re-call `outputs/screen/scr-a12x-zspxgs-pair-recall.json`, host paths made relative). **NOT_ADOPTED, with no GAIN and no
LOSS:** zspx trained with gsurg's loop moves no read past its floor, and the two filed LOSSes, musique zero-shot on
L-musique and webqsp zero-shot on L-hotpotqa, are inside their floors under the re-call.

- **zspxgs: 0 GAIN, 0 LOSS.** Two filed LOSSes are WITHIN under the re-call: musique zero-shot on L-musique, -0.0133
  [-0.0196, -0.0074] against its 0.0720 floor (FC@5 -0.0116), and webqsp zero-shot on L-hotpotqa, -0.0094 [-0.0174,
  -0.0012] against its 0.0233 floor (FC@5 -0.0073). Five more intervals exclude 0, all inside their floors: on
  L-musique, metaqa +0.0012 [+0.0003, +0.0022] and hotpotqa +0.0027 [+0.0005, +0.0050]; on L-hotpotqa, metaqa -0.0014
  [-0.0023, -0.0006], hotpotqa zero-shot -0.0033 [-0.0061, -0.0004] and 2wiki +0.0051 [+0.0037, +0.0066].
- **Beside: A12's LOSS is gone on zspx.** On zspk (A12's zspkgs) the loop lost webqsp zero-shot on L-musique (-0.0137);
  on zspx that read moves -0.0004. musique zero-shot on L-musique falls on both (-0.0190 on zspk, -0.0133 on zspx),
  inside its floor each time.

Not adopted. This closes only this form: gsurg's loop on the adopted GNN (docs/SCREENS.md section 3b). The next form,
fixed in section 5 and recorded in the verdict, is A12m, gsurg's loop on zrckp, the MLP's lean model (one declared file,
whose own section 5 names what follows it).

The floors are the seed null of 7 October (docs/SCREENS.md section 2: `outputs/screen/scr-null-s1.json`,
`scr-null-s2.json` and their `-hp` files), not N2's fresh floors (docs/N2_FRESH_SEED_NULL.md).

Records, filed with this section: the verdict `outputs/a12x/screen-verdict.json` and `.md`; for the arm `zspxgs`, the
compares `outputs/screen/scr-a12x-zspxgs.json` and `-hp.json`, the pair `-pair.json` and its re-call
`-pair-recall.json`, each with its `.md`. The host paths were made relative in each re-call's two `rel` entries and
nowhere else, so the `recall_sha256` each verdict holds is the host file's, from before that change. The re-calls' nulls
also read the base screens `outputs/a9/base-zspx-L-musique.json` and `outputs/a9/base-zspx-L-hotpotqa.json`, fetched
from the host with this filing (A9's own filing had left them there).

The twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's, zspx's card fit of the split; a
call is shown with its filed call where the re-call changed it):

| split | dataset | read | zspx R@5 | zspxgs: delta [95% CI], call |
| --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8030 | +0.0012 [+0.0003, +0.0022], WITHIN |
| L-musique | squad | in-domain | 0.9092 | -0.0003 [-0.0023, +0.0019], WITHIN |
| L-musique | musique | zero-shot | 0.5073 | -0.0133 [-0.0196, -0.0074], WITHIN (filed LOSS) |
| L-musique | hotpotqa | in-domain | 0.9074 | +0.0027 [+0.0005, +0.0050], WITHIN |
| L-musique | 2wiki | in-domain | 0.9351 | -0.0010 [-0.0025, +0.0005], WITHIN |
| L-musique | webqsp | zero-shot | 0.3204 | -0.0004 [-0.0099, +0.0084], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8039 | -0.0014 [-0.0023, -0.0006], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9129 | -0.0001 [-0.0019, +0.0019], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5901 | -0.0050 [-0.0102, +0.0001], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8547 | -0.0033 [-0.0061, -0.0004], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9312 | +0.0051 [+0.0037, +0.0066], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2914 | -0.0094 [-0.0174, -0.0012], WITHIN (filed LOSS) |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a12x-zspxgs.json` and `-hp.json`); and A12's R@5 deltas with the loop on zspk (zspkgs, its own
re-call `outputs/screen/scr-a12-zspkgs-pair-recall.json`, against zspk), on the same read (a call is shown where it is
not WITHIN):

| split | dataset | zspxgs FC@5 | zspxgs hit@1 | zspkgs (A12) R@5 |
| --- | --- | ---: | ---: | ---: |
| L-musique | metaqa | +0.0015 | +0.0011 | +0.0016 |
| L-musique | squad | -0.0003 | -0.0023 | +0.0006 |
| L-musique | musique | -0.0116 | -0.0095 | -0.0190 |
| L-musique | hotpotqa | +0.0039 | +0.0007 | +0.0010 |
| L-musique | 2wiki | -0.0014 | +0.0017 | -0.0014 |
| L-musique | webqsp | +0.0040 | -0.0160 | -0.0137 LOSS |
| L-hotpotqa | metaqa | -0.0015 | -0.0085 | -0.0003 |
| L-hotpotqa | squad | -0.0001 | -0.0005 | -0.0034 |
| L-hotpotqa | musique | -0.0095 | -0.0066 | +0.0007 |
| L-hotpotqa | hotpotqa | -0.0069 | +0.0072 | +0.0019 |
| L-hotpotqa | 2wiki | +0.0117 | -0.0010 | +0.0023 |
| L-hotpotqa | webqsp | -0.0073 | +0.0027 | +0.0040 |
