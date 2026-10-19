# A8e: round 11's depth prior, inside both models

Declared 19 October 2026, before any number. The next form A8's verdict named for both arms (docs/A8_QDEPTH_INPUT.md,
section 5), under docs/SCREENS.md section 3b and docs/VERDICT_AUDIT_2026_10_17.md (section 2, row 8). Code:
`outputs/mp_unified/za8e.py`. Filed 20 October: both arms NOT_ADOPTED with no GAIN and no LOSS (zspkqe's two filed
GAINs, musique and webqsp zero-shot, fall within their seed-null floors); the next form is A8ex (section 5).

## 1. Why this, now

**Round 11** (8 October, docs/SCREENS.md) learned a prior over five depth classes for each question and added it to
every row's score, inside step 1's MLP only. Its pair was MIXED: webqsp zero-shot LOSS (−0.0204) in L-musique's fit,
hotpotqa zero-shot −0.0125 in L-hotpotqa's (a LOSS, WITHIN under the null). Its one GAIN, musique zero-shot, was not
the prior's own: eleven of the seventeen arms screened against that fit moved the read by +0.08 to +0.24.

**A8** (filed 18 October, `outputs/a8/screen-verdict.md`) gave the same prior to both lean models as a fixed input: fitted
on its own depth-only objective, cross-fitted, six columns through a pointwise head. Both arms were NOT_ADOPTED with no
GAIN and no LOSS: all 24 reads WITHIN under the seed null. Five reads have a 95% interval that excludes 0, all below
their floors:

| arm | read | delta R@5 [95% CI] | floor |
| --- | --- | --- | --- |
| zrckpq (MLP) | L-musique hotpotqa | −0.0026 [−0.0049, −0.0003] | 0.0075 |
| zrckpq (MLP) | L-musique 2wiki | −0.0062 [−0.0078, −0.0046] | 0.0075 |
| zspkq (GNN) | L-musique 2wiki | −0.0070 [−0.0084, −0.0056] | 0.0075 |
| zspkq (GNN) | L-hotpotqa hotpotqa (zero-shot) | −0.0057 [−0.0086, −0.0027] | 0.0200 |
| zspkq (GNN) | L-hotpotqa 2wiki | +0.0051 [+0.0035, +0.0067] | 0.0075 |

The oracle's reads did not move. metaqa ran from −0.0006 to +0.0009 on both arms. webqsp's largest read was zspkq's
L-hotpotqa +0.0101 [−0.0012, +0.0216] (floor 0.0233). musique in domain was −0.0007 (zrckpq) and +0.0011 (zspkq).
Sources: `outputs/screen/scr-a8-zrckpq-pair-recall.json` (sha256 283dbf23…), `scr-a8-zspkq-pair-recall.json`
(238a93d6…) and `outputs/a8/screen-verdict.json` (f76fb4e6…).

With no GAIN on either arm, A8's section 5 fixed this file's form: round 11's own form on both lean models.

**What changes against A8.**
- **The prior learns with the model.** Each model's listwise loss trains U, W and b with the rest of its weights. A8
  fitted the prior alone, then froze it.
- **It adds to the score directly.** Each row gets l_q at its own class. A8's head read a log-weight normalised over
  the five classes, beside the one-hot.
- **No cross-fitting.** One prior per fit, from the fit's own training questions, like every other weight.

**What changes against round 11.** The models: zrckp and zspk on P1c2's pools with P1c3's KT input, where round 11
used step 1's MLP. In the GNN, zsp's propagation reads the score with the prior in it.

## 2. The form

Round 11's form, unchanged (`outputs/mp_unified/qdepth.py`):

    s(v) = s_base(v) + l_q[depth(v)],    l_q = q U + P_q W + b    (five values per question)

- **q** is the question's embedding: the P1c2 cache's `q_emb`, SEMB's input.
- **P_q** is its pool's profile (qdepth.profile): the share of its rows in each depth class, and log(1 + the pool's
  size). Six values.
- **depth(v)** is qdepth.depth_of's class of the row, from WALK's raw columns:
  - 0, a seed (column 15). A seed that is also a first hop is a seed.
  - 1 to 3, the first structural hop from a seed (columns 12 to 14). The nearest wins when a row carries two.
  - 4, unreached.
- **U** is 1,536 × 5, **W** 6 × 5, **b** 5: 7,715 parameters, all zero at the start. They draw nothing from any
  generator, so every other weight and the global random stream are the base's.
- **Training:** the model's listwise loss (lean_gpu's), with its own Adam, learning rate, weight decay and SWA (the card
  config 2e-3:1e-4:0.1:8:2). Only the differences between a question's five values move its ranking: the loss is a
  softmax over the question's rows.
- **A dropped block:** a question whose WALK or SEMB block is dropped takes no prior (round 11's rule). Fits and reads
  keep every block, so this never acts here.
- **The code** is qdepth.QDepth.depth_logits itself. The self-test checks the term against QDepth's, bit for bit.

**No columns, no builder.** A8e reads none of A8's columns. q and WALK are blocks of P1c2's cache that both base arms
already read under variant p, so nothing is built and nothing needs rebuilding.

**The MLP rules hold.**
- The prior reads the question's embedding and its pool's class counts. The classes come from the walks, fixed before
  any row is scored.
- It reads no label and no row's score, and passes nothing over graph neighbours. P_q's shares are over the
  question's rows, but of a feature (the depth class), not of a score.
- A row's term depends only on its question's five values and its own class.

## 3. The arms

| arm | model | where the prior enters | carve |
| --- | --- | --- | --- |
| zrckpqe | zrckp's model plus the prior (MRO: ZRCKPQE, ZRCKP, KTMixP, ZRM, ...) | added to the final score, after the pointwise KT head | zrckp's (`zp1c3.KTChainCarveZRC`), unchanged |
| zspkqe | zspk's model with the prior in its score (MRO: ZSPKQE, ZProp, QDMix, KTMix, ZRM, ...) | added to zrm's score and the KT head, before zsp's propagation reads it; the propagation's term adds on top | zspk's (`zp1c3.KTLinkCarveBase`), unchanged |

Each arm starts as its base, bit for bit.

## 4. The fits and the call

A8's, unchanged:
- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-a8e-<arm>`, `-hp`), on P1c2's roots with
  P1c3's KT input. Card fit settings: variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128.
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
- **INCOMPLETE:** its re-call is missing, is not A8e's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- webqsp and metaqa, both splits and arms (the oracle's reads);
- musique in domain (L-hotpotqa), both arms: Claim 1's read;
- hit@1 and FC@5 per read;
- per fit, the prior's norms |U|, |W| and |b| in each saved state (stamped by `train` as `prior_norms`).

When the results are filed, A8's deltas are set beside each read.

**Serving cost** (reported later, not part of this call): per question, one 1,536 × 5 product and the pool's class
counts; per row, one add. The classes come from WALK, which both models already read. The timed form is declared with
the full run if one follows.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools, then its full run, its own declared file |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A8ek, the back-off form: the prior split by the graph's kind (one U, W and b for the KB graphs, metaqa and webqsp, and one for the passage graphs, each at zero), so one kind's depth weights cannot move the other kind's scores; on both models |
| NOT_ADOPTED, with no GAIN | A8ex, the same prior on the adopted arms (G5's zspx, added before its propagation reads the score; R1's zrckpr, added to its final score), on both models; the depth prior closes as a model term only if A8ex finds no GAIN either |
| INCOMPLETE | the missing re-call is run; nothing else changes |

- **If both arms pass,** the prior joins both lean models after the refit.
- **If only one passes,** its refit and full run are declared; the next file decides whether F1's rule needs the other
  model to carry the prior too.
- **What does not happen.** No model change is adopted on today's pools (the user's rule of 13 October): an adopt is
  refitted on P1g-b's pools before its full run.

## 6. How it runs

**The commands** (zspkqe's are the same with its name and zspk's fits, `scr-p1c3-zspk` and `-hp`):

    python outputs/mp_unified/za8e.py train --split L-musique --name scr-a8e-zrckpqe --arm zrckpqe --device cuda \
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za8e.py train --split L-hotpotqa --name scr-a8e-zrckpqe-hp --arm zrckpqe --device cuda \
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za8e.py read --name scr-a8e-zrckpqe --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za8e.py compare --new outputs/screen/fits/scr-a8e-zrckpqe \
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a8e-zrckpqe
    python outputs/mp_unified/za8e.py pair --arm zrckpqe \
        --screens outputs/screen/scr-a8e-zrckpqe.json,outputs/screen/scr-a8e-zrckpqe-hp.json \
        --out outputs/screen/scr-a8e-zrckpqe-pair
    python outputs/mp_unified/za8e.py recall --arm zrckpqe \
        --null outputs/screen/scr-null-s1.json,outputs/screen/scr-null-s2.json,outputs/screen/scr-null-s1-hp.json,outputs/screen/scr-null-s2-hp.json \
        --pair outputs/screen/scr-a8e-zrckpqe-pair.json --out outputs/screen/scr-a8e-zrckpqe-pair-recall
    python outputs/mp_unified/za8e.py verdict

On the host the trains and reads run under `retry_cmd.py --cuda` and `cuda_alloc.py`, as A8's. `train` refuses another
arm, split, fit name or cache root, and a device other than the card; `read` refuses a fit that is not A8e's fit of its
split.

**Self-test.** `python outputs/mp_unified/za8e.py --selftest` passes before this commit (4.3 s). It runs on rmatch's
toy look, with zrm's toy carve given a WALK block in lean_mlp's layout (all five classes, a seed that is also a first
hop, rows with two first hops) and P1c3's toy KT input. It checks:
- **the arms:** the MROs; the base arms unchanged and A8's arms not imported; qdepth's constants; 7,715 added
  parameters; the toy walks' classes are the drawn ones;
- **the start:** from the same seed each arm is its base's state plus U, W and b at zero, the global random stream
  where the base leaves it. Each arm is its base bit for bit:
  - its scores, in eval and in training (the same dropout draws);
  - every gradient of a first listwise step but the prior's. The prior's are finite and nonzero, and b's sum to 0 over
    the classes;
- **the term:** with the KT head (and zsp's output layer) set alike, still the base. With U, W and b set:
  - the term equals a float64 hand computation from the raw walks and embeddings;
  - zrckpqe's score is the base's plus the term, bit for bit;
  - zspkqe's propagation reads the score with the term added, bit for bit, and moves;
- **never a row's score:** the prior from WALK and the question's embedding alone is the same bit for bit, and it stays
  the same when every other weight moves every score. A question whose WALK or SEMB is dropped takes none; the others
  keep theirs;
- **round 11's term:** it is what qdepth.QDepth adds to lean_gpu's model, bit for bit;
- **toy fits** through lean_gpu's loop repeat bit for bit and move U, W and b. With the prior frozen at zero, each
  arm's fit is its base's fit bit for bit on every shared key;
- the refusals; relz's pair and re-call per arm, stamped A8e's; the verdict's outcomes and next forms.

**Host items** (17, `a8e-*`): A8's 17 screen items under the new names.
- **No column dependency.** A8e reads no column, so no builder runs and nothing is rebuilt. Trains have no dependency;
  each read waits for its own fit; the re-calls wait for L1's base screens, as A8's.
- **Sizes: A8's.** The prior adds 7,715 parameters and one 1,536 × 5 product per batch.
  - L-musique trains `--frac 0.20` (share 0.22); -hp trains and the reads `--frac 0.27` (0.29).
  - Memory: trains 4.5 GB (zspkqe-hp 7 GB), reads 6 GB, compares 4 GB, the pairs, re-calls and verdict 2 GB.

## 7. Results

Filed 20 October 2026 from the host's records (`outputs/a8e/screen-verdict.json`, written 2026-10-19T18:43:35Z; the
re-calls `outputs/screen/scr-a8e-zrckpqe-pair-recall.json` and `-zspkqe-`, host paths made relative). **Both arms
NOT_ADOPTED, with no GAIN:** learned with each model and added to every row's score, round 11's depth prior moves no
read past the seed null on either model; the GNN's two interval gains, musique and webqsp zero-shot, sit within their
floors.

- **zrckpqe: 0 GAIN, 0 LOSS.** Every read stays within its floor. Three intervals exclude 0, all falls in domain and all
  below their floors: hotpotqa on L-musique -0.0068 [-0.0091, -0.0045] (floor 0.0075), and 2wiki on L-musique -0.0058
  [-0.0075, -0.0041] and on L-hotpotqa -0.0040 [-0.0055, -0.0023]. musique zero-shot on L-musique moves +0.0031
  [-0.0020, +0.0083]; webqsp zero-shot moves +0.0024 and +0.0030.
- **zspkqe: 0 GAIN, 0 LOSS.** Two reads were GAINs by their pairs' intervals and are WITHIN under the seed null: musique
  zero-shot on L-musique, +0.0084 [+0.0023, +0.0147] against its 0.072 floor, and webqsp zero-shot on L-hotpotqa,
  +0.0133 [+0.0030, +0.0232] against 0.0233, with FC@5 +0.0120. hotpotqa zero-shot on L-hotpotqa rises +0.0053 [+0.0023,
  +0.0083] (floor 0.02). The largest fall is musique in domain on L-hotpotqa, -0.0072 [-0.0127, -0.0014], just inside
  its 0.0075 floor.
- **Beside: the GNN's zero-shot moves grew from A8's.** With the prior as a fixed input (A8), zspkq moved webqsp
  zero-shot on L-hotpotqa +0.0101 and musique zero-shot on L-musique +0.0048; learned with the model, the moves are
  +0.0133 and +0.0084. hotpotqa zero-shot on L-hotpotqa turns from -0.0057 to +0.0053. The MLP's 2wiki fall on L-musique
  stays (-0.0062 at A8, -0.0058 now).

Neither arm is adopted. This closes only this form: round 11's depth prior learned inside zrckp and zspk and added to
each row's score (docs/SCREENS.md section 3b). The next form, fixed in section 5 and recorded in the verdict, is A8ex:
the same prior on the adopted arms, G5's zspx (added before its propagation reads the score) and R1's zrckpr (added to
its final score), on both models, in its own declared file. The depth prior closes as a model term only if A8ex finds no
GAIN either.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's; a call is shown with its
filed call where the re-call changed it; base R@5 shown as zrckp's / zspk's):

| split | dataset | read | base R@5 | zrckpqe: delta [95% CI], call | zspkqe: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8019 / 0.8009 | -0.0000 [-0.0011, +0.0010], WITHIN | +0.0008 [-0.0002, +0.0018], WITHIN |
| L-musique | squad | in-domain | 0.9116 / 0.9114 | -0.0008 [-0.0026, +0.0011], WITHIN | +0.0003 [-0.0017, +0.0023], WITHIN |
| L-musique | musique | zero-shot | 0.5121 / 0.5012 | +0.0031 [-0.0020, +0.0083], WITHIN | +0.0084 [+0.0023, +0.0147], WITHIN (filed GAIN) |
| L-musique | hotpotqa | in-domain | 0.9065 / 0.9080 | -0.0068 [-0.0091, -0.0045], WITHIN | -0.0010 [-0.0031, +0.0011], WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 / 0.9383 | -0.0058 [-0.0075, -0.0041], WITHIN | -0.0042 [-0.0057, -0.0027], WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 / 0.3229 | +0.0024 [-0.0087, +0.0135], WITHIN | -0.0007 [-0.0100, +0.0080], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8044 / 0.8040 | +0.0002 [-0.0007, +0.0011], WITHIN | -0.0008 [-0.0018, +0.0002], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 / 0.9135 | -0.0005 [-0.0022, +0.0014], WITHIN | -0.0024 [-0.0045, -0.0004], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 / 0.5754 | +0.0002 [-0.0046, +0.0049], WITHIN | -0.0072 [-0.0127, -0.0014], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 / 0.8499 | +0.0003 [-0.0026, +0.0033], WITHIN | +0.0053 [+0.0023, +0.0083], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 / 0.9333 | -0.0040 [-0.0055, -0.0023], WITHIN | +0.0037 [+0.0021, +0.0053], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 / 0.2680 | +0.0030 [-0.0080, +0.0136], WITHIN | +0.0133 [+0.0030, +0.0232], WITHIN (filed GAIN) |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a8e-zrckpqe.json` and the others, `-hp` for L-hotpotqa); and A8's R@5 deltas with the prior as a
fixed input (zrckpq and zspkq, their own re-calls), on the same read:

| split | dataset | zrckpqe FC@5 | zrckpqe hit@1 | zspkqe FC@5 | zspkqe hit@1 | zrckpq R@5 (A8) | zspkq R@5 (A8) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L-musique | metaqa | +0.0002 | +0.0014 | +0.0009 | +0.0062 | -0.0003 | -0.0006 |
| L-musique | squad | -0.0008 | -0.0014 | +0.0003 | +0.0019 | -0.0017 | +0.0009 |
| L-musique | musique | +0.0037 | +0.0112 | +0.0108 | +0.0095 | -0.0052 | +0.0048 |
| L-musique | hotpotqa | -0.0117 | +0.0046 | -0.0035 | -0.0020 | -0.0026 | +0.0018 |
| L-musique | 2wiki | -0.0147 | -0.0049 | -0.0092 | -0.0118 | -0.0062 | -0.0070 |
| L-musique | webqsp | -0.0040 | -0.0027 | -0.0027 | +0.0000 | -0.0066 | -0.0022 |
| L-hotpotqa | metaqa | +0.0004 | +0.0035 | -0.0011 | -0.0076 | +0.0001 | +0.0009 |
| L-hotpotqa | squad | -0.0005 | +0.0017 | -0.0024 | +0.0012 | -0.0003 | -0.0008 |
| L-hotpotqa | musique | +0.0021 | -0.0091 | -0.0149 | -0.0165 | -0.0007 | +0.0011 |
| L-hotpotqa | hotpotqa | +0.0005 | -0.0066 | +0.0105 | -0.0107 | -0.0002 | -0.0057 |
| L-hotpotqa | 2wiki | -0.0104 | -0.0078 | +0.0098 | -0.0041 | +0.0006 | +0.0051 |
| L-hotpotqa | webqsp | +0.0020 | -0.0047 | +0.0120 | +0.0033 | -0.0012 | +0.0101 |
