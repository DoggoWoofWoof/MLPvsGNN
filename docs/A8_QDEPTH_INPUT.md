# A8: round 11's depth prior, as a fixed input to both models

Declared 17 October 2026, before any number. The eighth reopening of docs/VERDICT_AUDIT_2026_10_17.md (section 2,
row 8), under docs/SCREENS.md section 3b. Code: `scripts/a8_qdepth_cols.py` (the prior's fits and its columns) and
`outputs/mp_unified/za8qd.py` (the arms, the screen and the call). Filed 19 October: both arms NOT_ADOPTED (section 7).

## 1. Why this, now

Round 11 (8 October, docs/SCREENS.md) learned, for each question, a prior over five depth classes of its pool's rows
(0 a seed, 1 to 3 the first structural hop from a seed, 4 unreached) and added it to every row's score.
- **The headroom is real.** A perfect question-to-depth attention lifted the MLP's metaqa 3-hop hit@1 by 0.074 to
  0.083 and doubled its webqsp zero-shot hit@1 (SCREENS.md, 'Hop depth of the top-1 errors').
- **Round 11 did not find it.** On step 1's MLP the pair was MIXED: webqsp zero-shot LOSS (−0.0204) in L-musique's
  fit, hotpotqa zero-shot −0.0125 (LOSS, WITHIN under the null) in L-hotpotqa's, metaqa in domain −0.0014 and
  −0.0034. Its one GAIN, musique zero-shot 0.270 to 0.380, is not the prior's own effect: eleven of the seventeen arms
  screened against that fit moved the same read by +0.08 to +0.24.
- **It never reached zrc or the GNN.** The prior ran once, inside step 1's MLP, trained only through the whole
  model's listwise loss. Under section 3b the idea stays open until it reaches both models.

**What changes in this form.**
- **The prior is fitted on its own.** Round 11 left it to the residual of the whole model's gradient, from zero, with
  the model's own learning rate. Here it is fitted on its own objective (section 2) to convergence, with an L2
  strength chosen on held-out questions.
- **It is cross-fitted.** Each question's columns come from a prior that never saw that question.
- **Each model decides how much to read it.** The head's last layer starts at zero, so a model that gains nothing
  from the prior can leave it at zero. That is the back-off round 11's additive form lacked on webqsp.

## 2. The form

**The prior** is round 11's (outputs/mp_unified/qdepth.py): five logits per question,

    l_q = x_q Theta + b,    x_q = (q, P_q), standardised

- q is the question's embedding: the P1c2 cache's `q_emb`, SEMB's input.
- P_q is its pool's profile: the share of its rows in each depth class, and log(1 + the pool's size)
  (qdepth.profile).
- depth(v) is qdepth.depth_of's class of the row, from WALK's raw columns.
- Theta is 1,542 × 5 and b is 5: the same 7,715 parameters as round 11's U, W and b.

**Its objective** is the listwise loss of the depth-only score l_q[depth(v)] over the question's rows: lean_gpu's
listwise, the mean over the question's golds of −log softmax over its rows. With n_q,d the question's rows and g_q,d
its golds in class d (G_q golds in all), it needs only those counts:

    loss_q = logsumexp_d (l_q,d + log n_q,d) − (1 / G_q) Σ_d g_q,d l_q,d

**The fit.** One prior per screen split and fold, on the split's four training datasets' fit carves of P1c2's cache.
The held-out dataset never trains it (L-musique's prior never sees musique; L-hotpotqa's never sees hotpotqa), and
neither does webqsp.
- **Questions:** the fold's questions, parity of sha256('dataset:question id'). A fifth of them are held out, by P5b2's
  split_of with seed 0. Of the rest, those with a gold in the pool train.
- **Objective:** each dataset's mean loss weighted alike, plus lam ‖Theta‖² (b is not penalised).
- **Inputs:** standardised by the training questions under the same weights.
- **Solver:** L-BFGS (strong Wolfe) in float64 from zero, round 11's start; at most 2,000 iterations, history 20.
- **lam:** a grid from 1e-1 to 1e-5 in half decades, strongest first, each run starting from the previous solution.
  The kept lam is the one with the lowest dataset-balanced held-out loss; ties go to the stronger.

**The columns,** six per cache row, float16, each question's from the prior of the fold that never saw it:

| column | what it is |
| --- | --- |
| lw | log_softmax(l_q)[depth(v)]: the prior's log-weight of the row's class, normalised over the five classes |
| d0..d4 | the row's depth class, one-hot |

**The MLP rules hold.**
- No column reads a label or any row's score.
- None passes over graph neighbours at read time. P_q counts the pool's rows by class from the walks, which are
  fixed before any row is scored; lw is normalised over the five classes, not over the question's rows.
- WALK already holds the class flags. The one-hot gives the head the class beside lw, so it can weigh the prior
  differently per class.

## 3. The arms

Both models read the columns through the same pointwise head: Linear(6, 16), GELU, Linear(16, 1). It has 129
parameters, the last layer zero at the start and the first drawn from its own generator (seed + 8801).

| arm | the model | where the head enters | the carve |
| --- | --- | --- | --- |
| zrckpq | zrckp's (zp1c3p.py) | added to the final score | zrckp's, with the columns beside |
| zspkq | zspk's (zp1c3.py) | added with the KT head, before zsp's propagation reads the score | zspk's, with the columns beside |

Each arm starts as its base, bit for bit. Each split reads only its own priors' columns.

## 4. The fits and the call

- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-a8-<arm>`, `-hp`), on P1c2's roots with
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
- **INCOMPLETE:** its re-call is missing, is not A8's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- webqsp and metaqa, both splits and arms (the oracle's reads);
- musique in domain (L-hotpotqa), both arms: Claim 1's read;
- hit@1 and FC@5 per read;
- per fit: the kept lam; the held-out loss per dataset, against the uniform prior's and the class bias alone's (does
  the question add anything over a per-class bias?); the iterations and final gradient per lam;
- per column part: the quantiles of lw, the class shares, the float16 error.

**Serving cost** (reported later, not part of this call): per question, one 1,542 × 5 product and a softmax over five
values; the classes come from WALK, which both models already compute. The timed form is declared with the full run
if one follows.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools (the prior fitted and its columns built again on P1g-b's cache), then its full run |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A8k, the back-off form: the six columns split by the graph's kind (the KB graphs' and the passage graphs'; twelve columns, the other kind's six at 0), so the KB graphs' depth weights cannot move the passage graphs' head weights; both models |
| NOT_ADOPTED, with no GAIN | A8e, round 11's own form on both lean models: the prior l_q[depth(v)] added to each row's score (qdepth.QDepth's U, W and b at zero), trained by each model's listwise loss; on zrckp, and on zspk before zsp's propagation reads the score |
| INCOMPLETE | the missing re-call is run; nothing else changes |

- **If both arms pass,** the columns join both lean models.
- **If only one passes,** its full run is declared; the next file decides whether F1's rule needs the other model to
  read the columns too.
- **What does not happen.** No model change is adopted on today's pools (the user's rule of 13 October).

## 6. How it runs

**Self-tests, both passed before this commit.**

`python scripts/a8_qdepth_cols.py --selftest` (45 s) checks, on a toy cache in lean_cache's part layout (metaqa,
squad and musique fit carves; four s1eval carves; two parts each), whose questions put their golds at a hidden depth
their embeddings encode:
- the depth classes are qdepth.depth_of's bit for bit, the nearest first hop winning; the profile is qdepth.profile's;
- the count form of the loss equals lean_gpu's listwise loss of the depth-only score over the rows;
- each fit's training and held-out questions are the fold's less its held-out fifth, with a gold in the pool (counts
  checked); the kept lam has the lowest held-out loss; every lam's solution is stationary (largest gradient entry
  below 1e-5); a repeat is bit for bit; the prior beats the uniform one on held-out questions;
- each row's columns are its question's prior of the other fold, against a float64 brute force from the raw walks, and
  differ from the own fold's;
- a part-wise build equals a one-part build of the same carve, and the build reads no label (the one-part copy has no
  gold array);
- the refusals: a split whose held-out dataset would train the prior, a fold out of range, a carve no split reads, a
  fit under other settings, another stage or another state, a walk that is not its record's.

`python outputs/mp_unified/za8qd.py --selftest` (8 s) checks:
- the arms' MROs; the base arms unchanged; 129 head parameters; the builder's constants and settings are the model
  module's;
- a carve reads only under `on_split`, only its split's columns, only columns built on its own step-1 parts under the
  builder's settings;
- a batch's columns are its rows' own;
- each arm starts as its base bit for bit; with the head set, zrckpq adds it and zspkq's propagation reads the score
  with it added; without the columns each is its base;
- toy fits through lean_gpu's loop repeat bit for bit and move the head; without the columns each fit is its base's;
- the refusals; relz's pair and re-call per arm, stamped A8's; the verdict's outcomes and next forms.

**Sizes.** The fits read about 6,000 questions per fit carve; the columns cover P1c2's eleven carves the two splits read
(about 97 million rows over both splits, 1.2 GB as float16).

**Host items** (`a8-*`), after A7's verdict:
- 4 prior fits (two splits, two folds), 1 CPU and 3 GB each, no dependency;
- 11 column items, 1 CPU and 3 GB each, after the four fits;
- the 17 screen items, the trains gated on their split's fit-carve columns and the reads on the six s1eval carves'.
  GPU sizing as A3's to A7's, from the base arms' measured peaks: L-musique trains `--frac 0.20` (share 0.22), -hp
  trains and the reads `--frac 0.27` (0.29); memory: trains 4.5 GB (zspkq-hp 7 GB), reads 6 GB.

## 7. Results

Filed 19 October 2026 from the host's records (`outputs/a8/screen-verdict.json`, written 2026-10-17T22:15:09Z; the
re-calls `outputs/screen/scr-a8-zrckpq-pair-recall.json` and `-zspkq-`, host paths made relative). **Both arms
NOT_ADOPTED:** no GAIN and no LOSS after the seed-null re-call.

- **zrckpq: 0 GAIN, 0 LOSS.** 2wiki in domain falls 0.0062 on L-musique [-0.0078, -0.0046] and webqsp zero-shot 0.0066
  there, both WITHIN; every other read is within 0.006.
- **zspkq: 0 GAIN, 0 LOSS.** 2wiki in domain falls 0.0070 on L-musique and hotpotqa zero-shot 0.0057 on L-hotpotqa, both
  WITHIN; webqsp zero-shot rises +0.0101 on L-hotpotqa, WITHIN.
- **Beside:** webqsp zero-shot -0.0066 and -0.0012 (MLP), -0.0022 and +0.0101 (GNN); metaqa in domain within 0.001 on
  all four fits; musique in domain (L-hotpotqa, Claim 1's read) -0.0007 (MLP) and +0.0011 (GNN).
- **The prior is informative on its own:** on every fit its held-out loss is below the class bias alone's, on every
  dataset (tables below), so the question adds to the depth classes' base rates. The solver: L-musique fold 0: 125 to
  2000 iterations (the 2,000 cap at lam 1e-05), final gradient at most 1.4e-06; L-musique fold 1: 152 to 2000 iterations
  (the 2,000 cap at lam 3e-05, 1e-05), final gradient at most 6.8e-07; L-hotpotqa fold 0: 147 to 1871 iterations, final
  gradient at most 1.4e-06; L-hotpotqa fold 1: 145 to 2000 iterations (the 2,000 cap at lam 1e-05), final gradient at
  most 9.1e-07.
- **The columns:** float16 error at most 0.004; the depth shares follow the graphs (on metaqa and musique most rows are
  at depth 2 or 3, or unreached; on squad most are unreached).

So the prior carries information about where the gold sits, but as six fixed columns through a pointwise head it moves
no read past the seed null. This closes only this form. The next form, fixed in section 5 and recorded in the verdict,
is **A8e**, round 11's own form on both lean models: the prior l_q[depth(v)] added to each row's score (qdepth.QDepth's
U, W and b at zero), trained by each model's listwise loss, on zrckp, and on zspk before its propagation reads the
score; one declared file.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's; a call is shown with its
filed call where the re-call changed it; base R@5 shown as zrckp's / zspk's):

| split | dataset | read | base R@5 | zrckpq: delta [95% CI], call | zspkq: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8019 / 0.8009 | -0.0003 [-0.0012, +0.0007], WITHIN | -0.0006 [-0.0016, +0.0003], WITHIN |
| L-musique | squad | in-domain | 0.9116 / 0.9114 | -0.0017 [-0.0037, +0.0003], WITHIN | +0.0009 [-0.0011, +0.0029], WITHIN |
| L-musique | musique | zero-shot | 0.5121 / 0.5012 | -0.0052 [-0.0106, +0.0003], WITHIN | +0.0048 [-0.0014, +0.0107], WITHIN |
| L-musique | hotpotqa | in-domain | 0.9065 / 0.9080 | -0.0026 [-0.0049, -0.0003], WITHIN | +0.0018 [-0.0003, +0.0039], WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 / 0.9383 | -0.0062 [-0.0078, -0.0046], WITHIN | -0.0070 [-0.0084, -0.0056], WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 / 0.3229 | -0.0066 [-0.0160, +0.0028], WITHIN | -0.0022 [-0.0128, +0.0074], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8044 / 0.8040 | +0.0001 [-0.0008, +0.0010], WITHIN | +0.0009 [-0.0000, +0.0019], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 / 0.9135 | -0.0003 [-0.0019, +0.0015], WITHIN | -0.0008 [-0.0028, +0.0011], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 / 0.5754 | -0.0007 [-0.0057, +0.0044], WITHIN | +0.0011 [-0.0041, +0.0065], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 / 0.8499 | -0.0002 [-0.0031, +0.0028], WITHIN | -0.0057 [-0.0086, -0.0027], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 / 0.9333 | +0.0006 [-0.0009, +0.0022], WITHIN | +0.0051 [+0.0035, +0.0067], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 / 0.2680 | -0.0012 [-0.0119, +0.0098], WITHIN | +0.0101 [-0.0012, +0.0216], WITHIN |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a8-zrckpq.json` and the others, `-hp` for L-hotpotqa):

| split | dataset | zrckpq FC@5 | zrckpq hit@1 | zspkq FC@5 | zspkq hit@1 |
| --- | --- | ---: | ---: | ---: | ---: |
| L-musique | metaqa | +0.0010 | -0.0010 | -0.0001 | +0.0005 |
| L-musique | squad | -0.0017 | -0.0011 | +0.0009 | +0.0016 |
| L-musique | musique | +0.0012 | +0.0050 | +0.0091 | -0.0050 |
| L-musique | hotpotqa | -0.0030 | +0.0069 | +0.0026 | +0.0078 |
| L-musique | 2wiki | -0.0142 | -0.0001 | -0.0203 | +0.0011 |
| L-musique | webqsp | -0.0113 | +0.0086 | -0.0033 | -0.0033 |
| L-hotpotqa | metaqa | -0.0007 | +0.0031 | +0.0014 | +0.0040 |
| L-hotpotqa | squad | -0.0003 | -0.0004 | -0.0008 | -0.0027 |
| L-hotpotqa | musique | +0.0017 | -0.0066 | -0.0017 | -0.0132 |
| L-hotpotqa | hotpotqa | +0.0007 | -0.0088 | -0.0095 | -0.0123 |
| L-hotpotqa | 2wiki | +0.0017 | -0.0057 | +0.0165 | +0.0009 |
| L-hotpotqa | webqsp | -0.0020 | -0.0033 | +0.0080 | +0.0027 |

The prior's fits (`outputs/a8/fits/<split>-f<fold>.json`): every fit kept lam 0.03; mean listwise loss on each dataset's
held-out fifth, at the kept lam, against the class bias alone and the uniform prior:

| split | fold | dataset | held-out loss at the kept lam | class bias alone | uniform prior |
| --- | ---: | --- | ---: | ---: | ---: |
| L-musique | 0 | metaqa | 5.880 | 6.922 | 7.607 |
| L-musique | 0 | squad | 2.296 | 2.455 | 3.913 |
| L-musique | 0 | hotpotqa | 2.897 | 3.028 | 4.508 |
| L-musique | 0 | 2wiki | 3.263 | 3.382 | 4.637 |
| L-musique | 0 | **mean** | **3.584** | **3.947** | **5.166** |
| L-musique | 1 | metaqa | 5.866 | 6.892 | 7.608 |
| L-musique | 1 | squad | 2.234 | 2.359 | 3.912 |
| L-musique | 1 | hotpotqa | 2.955 | 3.102 | 4.529 |
| L-musique | 1 | 2wiki | 3.291 | 3.422 | 4.643 |
| L-musique | 1 | **mean** | **3.587** | **3.944** | **5.173** |
| L-hotpotqa | 0 | metaqa | 5.922 | 7.093 | 7.607 |
| L-hotpotqa | 0 | squad | 2.280 | 2.415 | 3.913 |
| L-hotpotqa | 0 | musique | 4.921 | 5.291 | 7.630 |
| L-hotpotqa | 0 | 2wiki | 3.294 | 3.389 | 4.637 |
| L-hotpotqa | 0 | **mean** | **4.104** | **4.547** | **5.947** |
| L-hotpotqa | 1 | metaqa | 5.897 | 7.060 | 7.608 |
| L-hotpotqa | 1 | squad | 2.231 | 2.308 | 3.912 |
| L-hotpotqa | 1 | musique | 4.982 | 5.373 | 7.647 |
| L-hotpotqa | 1 | 2wiki | 3.307 | 3.425 | 4.643 |
| L-hotpotqa | 1 | **mean** | **4.104** | **4.541** | **5.952** |

The columns (`outputs/a8/cols/...`; shares weighted by rows):

| split | dataset | carve | rows | lw 5% / 50% / 95% (first part) | shares of depth 0 / 1 / 2 / 3 / unreached | float16 error |
| --- | --- | --- | ---: | --- | --- | ---: |
| L-musique | metaqa | fit | 12,021,002 | -8.59 / -6.49 / -2.98 | 0.004 / 0.010 / 0.187 / 0.428 / 0.371 | 0.0039 |
| L-musique | metaqa | s1eval | 19,742,463 | -8.99 / -6.83 / -3.56 | 0.004 / 0.010 / 0.183 / 0.425 / 0.377 | 0.0039 |
| L-musique | squad | fit | 292,994 | -6.91 / -4.89 / -0.04 | 0.151 / 0.203 / 0.044 / 0.007 / 0.595 | 0.0039 |
| L-musique | squad | s1eval | 594,063 | -6.89 / -4.92 / -0.04 | 0.151 / 0.190 / 0.061 / 0.010 / 0.587 | 0.0039 |
| L-musique | musique | s1eval | 5,057,436 | -7.23 / -5.35 / -1.42 | 0.004 / 0.087 / 0.357 / 0.358 / 0.195 | 0.0037 |
| L-musique | hotpotqa | fit | 557,195 | -7.57 / -5.00 / -0.12 | 0.084 / 0.335 / 0.270 / 0.030 / 0.281 | 0.0039 |
| L-musique | hotpotqa | s1eval | 693,828 | -7.43 / -5.00 / -0.12 | 0.085 / 0.330 / 0.270 / 0.030 / 0.284 | 0.0039 |
| L-musique | 2wiki | fit | 628,354 | -6.97 / -4.85 / -0.24 | 0.078 / 0.272 / 0.287 / 0.037 / 0.325 | 0.0038 |
| L-musique | 2wiki | s1eval | 1,328,715 | -6.91 / -4.85 / -0.21 | 0.077 / 0.284 / 0.271 / 0.040 / 0.327 | 0.0039 |
| L-musique | webqsp | s1eval | 3,186,165 | -7.36 / -5.35 / -2.10 | 0.004 / 0.089 / 0.251 / 0.256 / 0.399 | 0.0039 |
| L-hotpotqa | metaqa | fit | 12,021,002 | -8.67 / -6.43 / -2.93 | 0.004 / 0.010 / 0.187 / 0.428 / 0.371 | 0.0039 |
| L-hotpotqa | metaqa | s1eval | 19,742,463 | -9.01 / -6.83 / -3.56 | 0.004 / 0.010 / 0.183 / 0.425 / 0.377 | 0.0039 |
| L-hotpotqa | squad | fit | 292,994 | -6.94 / -5.00 / -0.02 | 0.151 / 0.203 / 0.044 / 0.007 / 0.595 | 0.0039 |
| L-hotpotqa | squad | s1eval | 594,063 | -6.89 / -4.98 / -0.02 | 0.151 / 0.190 / 0.061 / 0.010 / 0.587 | 0.0039 |
| L-hotpotqa | musique | s1eval | 5,057,436 | -8.33 / -6.38 / -3.97 | 0.004 / 0.087 / 0.357 / 0.358 / 0.195 | 0.0039 |
| L-hotpotqa | musique | s1fit | 9,623,206 | -8.32 / -6.47 / -4.38 | 0.004 / 0.092 / 0.356 / 0.362 / 0.186 | 0.0039 |
| L-hotpotqa | hotpotqa | s1eval | 693,828 | -7.54 / -5.30 / -0.04 | 0.085 / 0.330 / 0.270 / 0.030 / 0.284 | 0.0039 |
| L-hotpotqa | 2wiki | fit | 628,354 | -6.87 / -4.86 / -0.22 | 0.078 / 0.272 / 0.287 / 0.037 / 0.325 | 0.0039 |
| L-hotpotqa | 2wiki | s1eval | 1,328,715 | -6.84 / -4.86 / -0.18 | 0.077 / 0.284 / 0.271 / 0.040 / 0.327 | 0.0038 |
| L-hotpotqa | webqsp | s1eval | 3,186,165 | -8.33 / -6.11 / -3.37 | 0.004 / 0.089 / 0.251 / 0.256 / 0.399 | 0.0039 |
