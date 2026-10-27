# P5c5: the learned offsets at five folds, and the GNN's log p

Declared 19 October 2026, before any number. Code: `outputs/mp_unified/zp5c5.py` builds the columns again at five
folds with P5c's builder's functions (`scripts/p5c_offset_cols.py`, imported unchanged), then trains and reads the
arms. Two arms, one per model. Both read P5c's four columns, now from g's cross-fitted over five folds. The GNN's arm
also reads P5b's log p as a fifth column. This is the next form that P5c's section 5 fixed for an outcome with no GAIN
([P5C_OFFSET_INPUT.md](P5C_OFFSET_INPUT.md)), still under reopening #1 of
[VERDICT_AUDIT_2026_10_17.md](VERDICT_AUDIT_2026_10_17.md).

## 1. Why this, now

- **P5c gained nothing past the seed null.** Its verdict (`outputs/p5c/screen-verdict.json`, written
  2026-10-18T12:34:32Z) has both arms NOT_ADOPTED, with no GAIN on any of the 24 reads. These reads moved, from the
  re-calls (`outputs/screen/scr-p5c-zrckpo-pair-recall.json` and `-zspko-`; R@5 of the arm minus its base's):

  | split | dataset | read | arm | delta R@5 [95% CI] | floor | call |
  | --- | --- | --- | --- | --- | ---: | --- |
  | L-hotpotqa | musique | in-domain | zrckpo | +0.0062 [+0.0016, +0.0112] | 0.0075 | WITHIN |
  | L-hotpotqa | hotpotqa | zero-shot | zrckpo | +0.0027 [+0.0001, +0.0056] | 0.0200 | WITHIN |
  | L-hotpotqa | metaqa | in-domain | zrckpo | -0.0009 [-0.0017, -0.0002] | 0.0149 | WITHIN |
  | L-hotpotqa | 2wiki | in-domain | zrckpo | -0.0023 [-0.0039, -0.0007] | 0.0075 | WITHIN |
  | L-hotpotqa | webqsp | zero-shot | zrckpo | -0.0121 [-0.0219, -0.0016] | 0.0233 | WITHIN (filed LOSS) |
  | L-musique | metaqa | in-domain | zspko | +0.0014 [+0.0006, +0.0023] | 0.0075 | WITHIN |
  | L-musique | 2wiki | in-domain | zspko | -0.0037 [-0.0050, -0.0022] | 0.0075 | WITHIN |
  | L-hotpotqa | 2wiki | in-domain | zspko | -0.0111 [-0.0126, -0.0094] | 0.0075 | LOSS |

  Every other read's interval holds 0. The MLP's musique in-domain read, Claim 1's, rose by +0.0062, under its 0.0075
  floor.
- **Each of P5c's g's fitted on half the data.** P5c cross-fitted g over two folds, so each g fitted on one fold: half
  of its split's fit questions with a counted gold. P5c's section 5 named that halving as the first suspect. Five
  folds give each g four fifths of them, 1.6 times as many. A question still takes its columns only from a g that never
  fitted on it.
- **No model read the value P5b's reads ranked by.** P5b ordered each question's candidates by g's log p: per (seed,
  head), g's softmax over the candidates, mixed by π. P5c's pt is the same mixture without the softmax's normaliser.
  That normaliser is a sum over the question's rows. So the MLP may not read log p, and P5c computed it only as a
  check. The GNN may read a value normalised over the question's rows, since its propagation already mixes rows'
  scores. P5c's section 5 named log p as the GNN arm's fifth column.
- **Both models still get the five-fold columns** (F1, docs/F1_ONE_INPUT_SET.md). Each arm is decided against its own
  model's card fit, as in P5c. No GNN number chooses whether the MLP keeps them.

## 2. The form

### The five folds

- **fold(q)** is sha256("dataset:question id") read as an integer, mod 5. Every question of every carve has one.
- **Per split, five fits, g_0 to g_4.** g_k fits on the split's four training datasets' fit questions that have a
  counted gold (a gold among the candidates) and whose fold is not k. That is four fifths of them.
  - L-musique: metaqa, squad, hotpotqa and 2wiki;
  - L-hotpotqa: metaqa, squad, musique and 2wiki.

  webqsp never fits. A split's held-out dataset never fits that split's g's.
- **Each fit is P5c's, unchanged.** P5b's g (four heads, 200,465 weights) on P5a's V. P5b2's held-out stop inside the
  fit questions: 20% held out, split seed 0, the epoch with the lowest loss on held-out new golds kept, at most 30
  (epoch 0 is g's start). Adam 1e-3, batches of 32, float32 with TF32 off, EM over the latent (seed, head).
- **Each question q takes its columns from g_fold(q).** That is the one g of its split that never fitted on q. A
  training question's other four g's fitted on it. The s1eval questions, the held-out dataset's and webqsp's were
  fitted by none, and follow the same rule.
- **At two folds this is P5c's cross-fit.** g_k then fits on the other fold, which P5c called g_(1-k). A question of
  fold f takes g_f, which is P5c's g_(1-f).
- **The guard.** Each g's record lists every question it fitted on, its training and its held-out questions. The
  column build stops if a question would take its columns from a g whose record lists it.

### The columns

- **pt, mx, pt0 and mx0 are P5c's, unchanged** (P5c, section 2). pt0 and mx0 do not depend on g, so they are P5c's
  values. pt and mx now come from the five-fold g's.
- **lp, the GNN's fifth column, is P5b's log p of the row.** With ℓ and π as in P5c, and C the question's candidates
  (its pool minus its seeds):

      lp(v) = log Σ_{s,h} π_{s,h} exp( ℓ_{s,h}(v) − log Σ_{c in C} exp ℓ_{s,h}(c) )

  - For a candidate, lp is the log of P5b's p(v | q). Over the candidates, exp lp sums to 1.
  - A seed row gets the same formula, against the candidates' normaliser.
  - A question whose pool holds only seeds gets lp = pt.
- **lp is normalised over the question's rows, so only the GNN reads it.** The MLP's carve never opens its file.
- **Storage, as P5c's.** Each column is clipped to [-40, 40], divided by 10 and stored as float16. Per P1c2 cache part
  and split there is one `cols.npy` (rows × 4) and one `lp.npy` (rows × 1). A question with no seed gets -4 in every
  column, lp too.
- **Every question is built**, on P1c2's KIND_T pools, as in P5c. The units are P5c's stage, run again into this
  stage's folder. On each training dataset's fit carve, every question in P5b's filed base must come out as P5b's: its
  seeds, ranks, degrees, candidates and golds exactly, x and m_s within 1e-5.
- **Its own folders.** Every build is under `outputs/p5c5`: `units`, `g` and `cols/<split>/<dataset>/<carve>/<part>`.
  Nothing here reads or writes P5c's builds (`outputs/p5c`).

**A worked example, with made-up numbers.** One question, two seeds s₁ and s₂, one head each, π = 0.5 for both, and two
candidates. s₁'s target sits near v₁: ℓ(v₁) = -0.2, ℓ(v₂) = -3.0. s₂'s target sits far from both, nearer v₂:
ℓ(v₁) = -9.0, ℓ(v₂) = -6.0.

| row | pt | lp | stored (pt, lp) |
| --- | ---: | ---: | --- |
| v₁ | -0.893 | -0.703 | -0.0893, -0.0703 |
| v₂ | -3.645 | -0.683 | -0.3645, -0.0684 |

pt follows the nearest target, so s₂ adds almost nothing and v₁ leads by far. In lp each (seed, head) casts one vote
over the candidates: s₂ gives v₂ 0.95 of its vote, and v₂ edges ahead. The GNN's head sees both, before any score
exists.

## 3. The arms

| arm | model | carve | its base |
| --- | --- | --- | --- |
| zrckpo5 | zrckpo's model unchanged: zrckp plus P5c's head over the four columns (`ZRCKPO5`) | zrckp's (`KTChainCarveZRC`) with the four five-fold columns beside (`P5c5KTChainCarveZRC`) | `scr-p1c3p-zrckp`, `-hp` |
| zspko5 | zspk plus a head over five columns, added with the KT head before zsp's propagation reads the score (`ZSPKO5`) | zspk's (`KTLinkCarveBase`) with the four five-fold columns and lp beside (`P5c5KTLinkCarveBase`) | `scr-p1c3-zspk`, `-hp` |

- **zrckpo5's head** is P5c's: Linear(4, 16), GELU, Linear(16, 1), 97 parameters, over the row's own four columns,
  added to the score. Its model is P5c's zrckpo bit for bit. Only its columns differ.
- **zspko5's head** is the same over five columns: Linear(5, 16), GELU, Linear(16, 1), 113 parameters. It is added
  with the KT head, before zsp's propagation reads the score. So lp reaches other rows only through the GNN's own
  propagation.
- Neither head has a score input. Its last layer starts at zero, so each arm starts as its base bit for bit. Its first
  layer is drawn from its own generator (seed + 5701).
- **Each split's fits read that split's columns.** A fit of L-musique never sees a g that trained on musique, and a
  fit of L-hotpotqa never sees one that trained on hotpotqa.
- Everything else is the base's, unchanged: the carve, KT input, settings, seeds, batches, loss and loop. Neither arm
  adds a graph, an encoder, text or a message. The MLP's carve gets the four columns and no edge, and never opens lp.

## 4. The fits and the call

- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa: `scr-p5c5-zrckpo5`, `scr-p5c5-zrckpo5-hp`,
  `scr-p5c5-zspko5` and `scr-p5c5-zspko5-hp`. They run on P1c2's roots, P1c3's KT input and this stage's columns of
  that split.
- **Reads:** each fit on the six s1eval carves at p@swa. No test split is read.
- **Compares:** each fit against its base arm's card fit of the same split (`scr-p1c3p-zrckp` and `scr-p1c3-zspk`,
  with `-hp`), which decides. Step 1's fit is reported beside. The re-call's base screens are L1's
  (`outputs/l1/base-<base arm>-<split>.json`).
- **Call:** relz's pair and its re-call under the seed null, per arm, with the four null files of docs/SCREENS.md
  (`outputs/screen/scr-null-s1.json`, `-s2`, `-s1-hp`, `-s2-hp`). GAIN, LOSS and WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not P5c5's re-call of that arm, or does not hold the twelve reads.

The verdict (`outputs/p5c5/screen-verdict.json` and `.md`) closes only this form (SCREENS.md section 3b), and it names
the next form (section 5).

**Reported beside, deciding nothing:**
- webqsp (zero-shot in both splits) and metaqa (in-domain), per split and arm, with their calls and differences;
- musique's in-domain read (L-hotpotqa) for both arms side by side. Claim 1's musique share depends on it;
- hit@1 and FC@5 per read, as the compares give them;
- each read's delta beside P5c's delta of the same read and model (P5c, section 7). For the MLP that is the move from
  two folds to five alone;
- each g fit's kept epoch, beta and number of fit questions; each part's column quantiles and clipped shares, lp's
  included, and its rows by fold.

**Flagged.**
- **zspko5 changes two things at once:** the folds and lp. Its outcome alone does not say which part moved it.
  zrckpo5 reads the five-fold columns alone, so the MLP's arm isolates the folds.
- **lp is normalised over the question's rows.** The GNN may read it, and it reaches other rows only through zsp's
  propagation. The MLP may not, and its carve never opens the file: the self-test loads the MLP's carve with lp's file
  removed.
- **Graph aggregates, π and serving cost** are as P5c's (P5c, section 4). lp adds one log-sum-exp over the
  candidates per (seed, head), 40 per question. That is small next to the 40 distances per row.

## 5. What each outcome leads to (fixed now)

Per arm, the next form, each in its own file declared before its numbers. The verdict record writes it for each arm.

| this arm's re-call | the other arm's | the next form |
| --- | --- | --- |
| ADOPT_FOR_FULL_RUN | any | the arm fitted again on P1g-b's pools, then its full run, its own declared file |
| NOT_ADOPTED, with a GAIN | any | the back-off form, its own declared file: P5b3's gate, refit per split and cross-fitted over the same five folds, as one more column. For zrckpo5 it is a fifth column from the six gate inputs the MLP may read: log n, ring1, ring2, the log of the number of seeds, pi_max and delta_rms. For zspko5 it is a sixth column from all ten |
| NOT_ADOPTED, no GAIN | ADOPT_FOR_FULL_RUN | zrckpo5: none on the MLP. It read the four columns at two folds and at five with no GAIN, and may not read lp. zspko5's full-run file decides under F1 whether the GNN keeps an input the MLP does not read. zspko5: zrckpo5's input on the GNN, its own declared file: zspk with the four five-fold columns and no lp, one input set for both models (F1) |
| NOT_ADOPTED, no GAIN | NOT_ADOPTED, with a GAIN | the other arm's back-off form reaches this model too (SCREENS.md, 3b) |
| NOT_ADOPTED, no GAIN | NOT_ADOPTED, no GAIN | none: the learned offsets close as a fixed input to the lean pair, with this file named as the close (VERDICT_AUDIT_2026_10_17.md, section 6). They reached both models at two folds and at five, the GNN also with P5b's normalised log p, with no GAIN on any read |
| NOT_ADOPTED, no GAIN | INCOMPLETE | decided with the other arm's verdict, once its missing re-call is run |
| INCOMPLETE | any | the missing re-call is run; nothing else changes |

- **If both arms pass**, the input is the lean pair's, read by both.
- **If only one passes**, its full run is declared. That file decides whether F1's one-input-set rule keeps the input.
- **What does not happen.** No model change is adopted on today's pools (the user's rule of 13 October).

## 6. How it runs

- **Self-test.** `python outputs/mp_unified/zp5c5.py --selftest` passes before this commit, in 32.6 s on the laptop.
  - **The builds** (28.1 s), on P5c's toy units and look, with P5c's builder's functions:
    - every build's root is under `outputs/p5c5`, never under P5c's `outputs/p5c`; P5c's splits, carves, columns and
      transform are the ones used;
    - the folds are deterministic and balanced (each within 0.03 of a fifth over 4,000 ids). At two folds they are
      P5c's parity, and g_k's questions are P5c's other fold;
    - **at two folds this builder is P5c's.** On the same toy, P5c's builder and this one give the same g's (this
      g_k is P5c's g_(1-k), and the two folds' g's differ) and the same columns, with equal sha256;
    - **at five folds, g_k fits on exactly the counted questions outside fold k.** So each counted question sits in
      four of the five fits;
    - every row's four columns and lp are its own fold's g's, bit for bit. They match a float64 reference built seed
      by seed, head by head and row by row within 1e-4. lp is P5b's log p within 2e-3, and lp is not pt;
    - **no g scores a question it fitted on**, while the other four g's fitted on it. Another fold's g gives other
      columns;
    - the guard: a g record that lists a question of its own fold stops the columns;
    - refusals: two-fold fits in a five-fold build, a sixth fold, carves no split reads, and P5c's units read as
      this stage's.
  - **The arms:**
    - the arms are zrckpo's and zspk's models plus the head; the base arms and P5c's arms are unchanged;
    - a carve reads only under its split, and only this stage's five-fold build of its parts. P5c's build and the
      two-fold settings are refused. The MLP's carve loads with lp's file removed; the GNN's stops;
    - a batch's columns are its rows' own, in the cache's row order;
    - zrckpo5 is P5c's zrckpo bit for bit, and each arm starts as its base bit for bit. With the head moved,
      zrckpo5's score is its base's plus the head, exactly. zspko5's propagation reads its base score plus the head
      over all five columns, and lp moves it;
    - toy fits repeat bit for bit and the head learns. Without the columns, each arm's fit equals its base's on every
      shared key;
    - train refuses another arm, split, fit name or cache root. read refuses a fit of another split, or one that is
      not P5c5's;
    - relz's pair and re-call per arm, stamped P5c5's;
    - the verdict's outcomes, each with section 5's next form;
    - P5c's folder and this stage's are untouched.
- **No development run.** The caches exist only on the host. The units stage is P5c's code, called unchanged, and
  P5c's development run stands (P5c, section 6).
- **The builds never skip.** Each build stage writes its files again when its record exists, as P5c's do.
- **Host items** (`p5c5-*`, 49 in all, 32 of them builds named `p5c5-b-*`):
  - on the CPU: 11 `p5c5-b-units-<dataset>-<carve>`, one per carve the splits read, 2 CPUs each. Memory is P5c's
    plus 0.5 GB, for the model stack this script imports;
  - on the card at share 0.12 (`cuda_alloc --frac 0.1`, P5c's), 2 CPUs and 10 GB: ten `p5c5-b-fitg-k<k>` and
    `p5c5-b-fitg-hp-k<k>`, one per split and fold. Each runs after its split's four fit-carve units. P5c's fits had
    8 GB; each g now holds four fifths of the fit questions, not half;
  - on the CPU: 11 `p5c5-b-cols-<dataset>-<carve>`, each after its carve's units and the five fits of every split
    that reads it. Memory is P5c's plus 0.5 GB;
  - on the card: four trains, then four reads, at `--frac 0.30` and share 0.32 (P5c's). Each train runs after its
    carves' columns;
  - on the CPU: four compares, two pairs, two re-calls (after L1's base screens) and the verdict.
  - Each build item names P5c's inputs, and also P5a's node tables of its datasets
    (`outputs/p5a/space/u_<dataset>.npy` and `u0_<dataset>.npy`), which P5c's items did not name. The feeder holds an
    item while an input is missing. So a table that the host's clean-up removed holds the chain instead of failing it.
- **Inputs the host must still hold.** The builds read what P5c's read: P5a's frame and node tables, P5b's filed base,
  P1c2's pools, cache and look, and the link stores. This stage does not rebuild them. P5a's space is a trained fit,
  and P5b's check needs the filed one.
  - **P5a's node tables were gone.** The host's clean-up removed them at 00:08 on 19 October, as builds; that is fixed
    in 82ebdb4. `scripts/p5a_restore.py` (45cbcbe, items `p5ar-*`) makes them again with P5a's own code, unchanged. A
    table goes back into `outputs/p5a/space` only if its sha256 is the one P5a's records name, and never over a file.
    So the units read P5a's tables bit for bit, or they stay held. squad's space is fit first, and the other five
    only if it matches.
  - **If a table does not come back bit for bit,** these builds stay held. This stage is then declared again, in its
    own file, on a refit of P5a's space with P5b's g fitted again on it. P5b's filed base was fit on the old space.
- **Disk.**
  - units: about 0.5 GB, as P5c's;
  - columns: P5c's rows, 97,270,441 row-splits in all (P5c, section 7), × 4 columns × 2 bytes, about 0.78 GB. lp adds
    about 0.19 GB;
  - the ten g fits: about 10 MB, with their fit-question lists.

  About 1.5 GB in all.
- **ETA** (estimates):
  - units: about 2 to 15 minutes each;
  - fitg: P5c's section 6 put a fit on about 5,000 questions at 45 to 60 minutes. A P5c5 fit holds 1.6 times as
    many, so about 75 to 100 minutes each, all ten at once if the card has room;
  - cols: about 5 to 20 minutes each;
  - trains and reads: about 2.5 to 3 hours, as P5c's.

  The verdict is due about 4 to 6 hours after the first units start, if the card has room.
- **Fixed rules.** The encoder stays frozen. webqsp never trains. Test splits are never read. No LLM or transformer.

## 7. Results

Filed 27 October 2026 from the host's records (`outputs/p5c5/screen-verdict.json`, written 2026-10-20T17:12:11Z; the
re-calls `outputs/screen/scr-p5c5-zrckpo5-pair-recall.json` and `scr-p5c5-zspko5-pair-recall.json`, host paths made
relative). **Both arms NOT_ADOPTED, with no GAIN; the learned offsets close as a fixed input to the lean pair.** At five
folds, the MLP's arm moves no read past its floor, and the GNN's arm, with P5b's log p as a fifth column, loses webqsp
zero-shot on L-musique.

- **zrckpo5: 0 GAIN, 0 LOSS.** webqsp zero-shot on L-hotpotqa falls -0.0120 [-0.0223, -0.0014], a LOSS by its pair's
  interval that is WITHIN under its 0.0233 floor (FC@5 -0.0086). Three more intervals exclude 0, all inside their
  floors: on L-musique, 2wiki +0.0018 [+0.0004, +0.0032]; on L-hotpotqa, metaqa -0.0008 [-0.0015, -0.0002] and hotpotqa
  zero-shot +0.0031 [+0.0005, +0.0058].
- **zspko5: 0 GAIN, 1 LOSS.** The LOSS is webqsp zero-shot on L-musique: -0.0142 [-0.0250, -0.0034] against its 0.0102
  floor, with FC@5 -0.0126. musique zero-shot on L-musique rises +0.0146 [+0.0087, +0.0208], a GAIN by its pair's
  interval that is WITHIN under its 0.0720 floor (FC@5 +0.0116). Three more intervals exclude 0, all inside their
  floors: metaqa in domain, +0.0043 [+0.0033, +0.0054] on L-musique and +0.0020 [+0.0009, +0.0031] on L-hotpotqa; and
  2wiki in domain on L-musique, -0.0014 [-0.0027, -0.0000] (its upper end below 0 before rounding: the record holds
  -0.0).
- **Beside: five folds move P5c's one LOSS and add another.** At two folds (P5c), zspko lost 2wiki in domain on
  L-hotpotqa (-0.0111); at five, with lp, zspko5 moves that read +0.0003 and loses webqsp zero-shot on L-musique, which
  zspko moved -0.0066. zrckpo's twelve reads at two folds were all WITHIN, as zrckpo5's are at five.

Not adopted. This closes only this form: P5c's four offset columns cross-fitted over five folds, on both models, with
P5b's log p as a fifth column on the GNN (docs/SCREENS.md section 3b). With no GAIN on either arm, the next form fixed
in section 5 and recorded in the verdict for both arms is none: the learned offsets close as a fixed input to the lean
pair, this file named as the close (docs/VERDICT_AUDIT_2026_10_17.md, section 6): they reached both models at two folds
and at five, the GNN also with P5b's normalised log p, with no GAIN on any read.

The floors are the seed null of 7 October (docs/SCREENS.md section 2: `outputs/screen/scr-null-s1.json`,
`scr-null-s2.json` and their `-hp` files), not N2's fresh floors (docs/N2_FRESH_SEED_NULL.md).

Records, filed with this section: the verdict `outputs/p5c5/screen-verdict.json` and `.md`; for each arm (`zrckpo5`,
`zspko5`), the compares `outputs/screen/scr-p5c5-<arm>.json` and `-hp.json`, the pair `-pair.json` and its re-call
`-pair-recall.json`, each with its `.md`. The host paths were made relative in each re-call's two `rel` entries and
nowhere else, so the `recall_sha256` each verdict holds is the host file's, from before that change. The re-calls' nulls
also read the base screens `outputs/l1/base-zrckp-L-musique.json` and `outputs/l1/base-zrckp-L-hotpotqa.json` and
`outputs/l1/base-zspk-L-musique.json` and `outputs/l1/base-zspk-L-hotpotqa.json`, already in the repository.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's, the base's card fit of the
split; a call is shown with its filed call where the re-call changed it):

| split | dataset | read | zrckp R@5 | zrckpo5: delta [95% CI], call | zspk R@5 | zspko5: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8019 | -0.0006 [-0.0016, +0.0003], WITHIN | 0.8009 | +0.0043 [+0.0033, +0.0054], WITHIN |
| L-musique | squad | in-domain | 0.9116 | -0.0003 [-0.0019, +0.0012], WITHIN | 0.9114 | -0.0008 [-0.0029, +0.0012], WITHIN |
| L-musique | musique | zero-shot | 0.5121 | +0.0011 [-0.0032, +0.0056], WITHIN | 0.5012 | +0.0146 [+0.0087, +0.0208], WITHIN (filed GAIN) |
| L-musique | hotpotqa | in-domain | 0.9065 | +0.0003 [-0.0016, +0.0022], WITHIN | 0.9080 | +0.0007 [-0.0014, +0.0028], WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 | +0.0018 [+0.0004, +0.0032], WITHIN | 0.9383 | -0.0014 [-0.0027, -0.0000], WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 | -0.0076 [-0.0172, +0.0016], WITHIN | 0.3229 | -0.0142 [-0.0250, -0.0034], **LOSS** |
| L-hotpotqa | metaqa | in-domain | 0.8044 | -0.0008 [-0.0015, -0.0002], WITHIN | 0.8040 | +0.0020 [+0.0009, +0.0031], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 | -0.0007 [-0.0024, +0.0013], WITHIN | 0.9135 | +0.0008 [-0.0012, +0.0025], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 | +0.0030 [-0.0019, +0.0080], WITHIN | 0.5754 | +0.0053 [-0.0004, +0.0109], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 | +0.0031 [+0.0005, +0.0058], WITHIN | 0.8499 | +0.0009 [-0.0021, +0.0038], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 | -0.0004 [-0.0018, +0.0010], WITHIN | 0.9333 | +0.0003 [-0.0012, +0.0019], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 | -0.0120 [-0.0223, -0.0014], WITHIN (filed LOSS) | 0.2680 | +0.0046 [-0.0064, +0.0161], WITHIN |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-p5c5-<arm>.json` and `-hp.json`); and P5c's R@5 deltas with the offsets at two folds (zrckpo
against zrckp, zspko against zspk; their own re-calls, `outputs/screen/scr-p5c-<arm>-pair-recall.json`), on the same
read (a call is shown where it is not WITHIN):

| split | dataset | zrckpo5 FC@5 | zrckpo5 hit@1 | zspko5 FC@5 | zspko5 hit@1 | zrckpo (P5c) R@5 | zspko (P5c) R@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L-musique | metaqa | -0.0001 | -0.0035 | +0.0039 | +0.0196 | +0.0007 | +0.0014 |
| L-musique | squad | -0.0003 | -0.0026 | -0.0008 | -0.0012 | +0.0003 | +0.0015 |
| L-musique | musique | +0.0012 | +0.0137 | +0.0116 | +0.0037 | +0.0017 | -0.0013 |
| L-musique | hotpotqa | +0.0018 | -0.0074 | -0.0003 | +0.0012 | -0.0001 | -0.0011 |
| L-musique | 2wiki | +0.0045 | -0.0038 | -0.0024 | -0.0053 | +0.0006 | -0.0037 |
| L-musique | webqsp | -0.0133 | +0.0033 | -0.0126 | +0.0040 | -0.0084 | -0.0066 |
| L-hotpotqa | metaqa | -0.0009 | -0.0020 | +0.0029 | +0.0049 | -0.0009 | +0.0004 |
| L-hotpotqa | squad | -0.0007 | +0.0009 | +0.0008 | +0.0024 | -0.0005 | +0.0001 |
| L-hotpotqa | musique | +0.0054 | -0.0108 | +0.0025 | -0.0037 | +0.0062 | +0.0008 |
| L-hotpotqa | hotpotqa | +0.0089 | +0.0011 | +0.0039 | -0.0039 | +0.0027 | +0.0020 |
| L-hotpotqa | 2wiki | -0.0002 | -0.0002 | +0.0044 | -0.0041 | -0.0023 | -0.0111 LOSS |
| L-hotpotqa | webqsp | -0.0086 | -0.0013 | +0.0013 | -0.0013 | -0.0121 | +0.0040 |
