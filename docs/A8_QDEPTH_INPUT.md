# A8: round 11's depth prior, as a fixed input to both models

Declared 17 October 2026, before any number. The eighth reopening of docs/VERDICT_AUDIT_2026_10_17.md (section 2,
row 8), under docs/SCREENS.md section 3b. Code: `scripts/a8_qdepth_cols.py` (the prior's fits and its columns) and
`outputs/mp_unified/za8qd.py` (the arms, the screen and the call). Not run yet.

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
