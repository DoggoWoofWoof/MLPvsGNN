# One-fit screens

Declared 7 October 2026 at about 14:45, before any screen number was read. Code:

- `outputs/mp_unified/lean_screen.py`: the arms, train, read, compare, ablate and the smoke.
- `outputs/mp_unified/screen_null.py`: the null check.

The queue is the `scr-*` items of `outputs/host_ops/hostfeed_items.txt`.

## 1. Why

The user's rule (7 October, 13:50): one training run per idea, on one split. No extra seeds, splits, variants or select
carves. An idea gets a full run (six fits and a declared grade, in its own file) only after its screen shows a gain.

Steps 1 to 4 spent six fits, four variants and select-carve picks on every idea. That is about two hours of card per
fit. Step 1's verdict took a day and came back NOT_ADOPTED. A screen costs one fit (about 25 minutes on a quarter of
the card) and six reads.

## 2. Protocol

- **Split:** step 1's L-musique (docs/STEP1_MATCHED_SELECTION.md).
  - Trained on the fit carves of metaqa, squad, hotpotqa and 2wiki; basis 2wiki.
  - Those four are read in-domain. musique (a passage graph) and webqsp (a KB) are read zero-shot.
  - Its baseline already exists.
- **One fit:** variant p, seed 0, the SWA state p@swa.
  - Variant p is the pick set (rank, dense_cos, topo_STRUCT, depth_STRUCT, SEMB, SEED, WALK, DISTS, WALKF), with no
    context.
  - Config 2e-3:1e-4:0.1:8:2 (lr, wd, dropout, epochs, SWA from epoch 2); hidden 128.
  - On the card, under lean_gpu.py's flags: deterministic algorithms, TF32 off, 2 torch threads.
  - lean_gpu.py's loop, batches and train/read commands are called unchanged; an arm swaps in its own model class or
    carve class. Every arm changes every training dataset alike.
- **Reads:** the six s1eval carves.
- **Baseline:** step 1's L-musique p@swa, read on the same carves (`outputs/step1/fits/L-musique`, models sha256
  `493723a1…`).
- **Call per read, on R@5:**
  - Δ is new minus base, paired by question.
  - **GAIN:** Δ ≥ 0.0075 and the 95% interval of a 2,000-resample question bootstrap lies above 0.
  - **LOSS:** the mirror case.
  - **WITHIN:** everything else.
  - FC@5 and hit@1 differences are reported, not called.
- **Verdict:**
  - **PROMISING:** at least one GAIN and no LOSS. The idea gets its full run, declared in its own file.
  - **MIXED:** GAIN and LOSS both. No full run as it stands; a narrower screen may follow, declared here.
  - **NO_GAIN:** no GAIN. The idea is dropped.
  - **Amended 7 October, about 15:55, after scr-padbd20's numbers:** a PROMISING screen whose arm contains an arm
    that was NOT_ADOPTED in its own full run gets a full run only if it also GAINs against that arm's screen. Without
    such a gain, the full run would repeat the failed arm's loss with nothing added.
- **Amended 7 October, about 20:35, at the user's request ("yeah do them"), before any number of the fourth round:
  two fits per screen.** From the fourth round on, a screen trains its arm on two splits:
  - **L-musique**, as before;
  - **L-hotpotqa**, which reads hotpotqa zero-shot and musique in-domain.

  Each fit is compared with step 1's fit of the same split (p@swa), read by read, under the call above. The verdict
  above is taken over all twelve reads: PROMISING needs at least one GAIN and no LOSS among the twelve.
  `outputs/mp_unified/screen_pair.py` files the verdict over both fits, and the full-run gate reads it.

  Why: both PROMISING screens so far failed their full runs on reads the L-musique fit cannot make. bdrop20 lost on
  musique in-domain (J5); zret lost on hotpotqa and 2wiki read zero-shot. Under this rule, zret's two fits (its screen
  and its full run's L-hotpotqa fit) are MIXED: the L-hotpotqa fit LOSEs on hotpotqa zero-shot (−0.014) and on metaqa
  in-domain (−0.012). The two fits run side by side on the card, so a verdict comes about one fit later than before.
- **Amended 7 October, about 23:45, before any of its numbers and before vshare's L-hotpotqa numbers: the seed null.**
  Step 1's base arm, unchanged, is trained with seeds 1 and 2 on both screen splits. Each fit is compared with step
  1's seed-0 p@swa of its split by the compare above, which gives two seed-only differences on each of the twelve
  reads.
  - **Why:**
    - On L-hotpotqa, hotpotqa read zero-shot has LOST under four different arms: zret (−0.014), ztop50 (−0.008),
      gsurg (−0.015) and hubwalk (−0.016). hubwalk does not touch the pool-share confound the others target.
    - If the arms were neutral on that read, four losses in four would come about one time in sixteen. A seed-0
      baseline that is a high draw on that read would explain them.
    - The same can turn a low draw into an apparent GAIN, by regression to the mean: step 1's L-musique fit reads
      musique zero-shot at 0.270, below rrf's 0.473.
    - Models that agree in-domain can differ widely out of distribution (underspecification, D'Amour et al. 2020),
      and the floor 0.0075 was measured in-domain.
  - **Fits:** `scr-null-s1` and `scr-null-s2` (L-musique), and `scr-null-s1-hp` and `scr-null-s2-hp` (L-hotpotqa).
    - Each is the base arm through lean_screen2.py's train with `--seed 1` or `--seed 2`.
    - Everything else is step 1's: variant p, config 2e-3:1e-4:0.1:8:2, hidden 128, p@swa, the six s1eval reads.
    - Seed 0 is step 1's fit itself.
  - **Floors:** for each read r (split × dataset), let Δ_s(r) be seed s's R@5 minus seed 0's. Then
    floor(r) = max(0.0075, 2·√((Δ_1(r)² + Δ_2(r)²) / 2)), twice the seed-only spread.
    - A read is GAIN when Δ ≥ floor(r) and its 95% interval lies above 0, LOSS in the mirror case, WITHIN otherwise.
    - The verdict rule over the twelve reads is unchanged.
    - The form is fixed here and is not tuned.
  - **Use:** `outputs/mp_unified/screen_recall.py` re-calls a pair under the floors and files
    `scr-<arm>-pair-recall.{md,json}` beside its verdict. It is committed before the null's numbers exist.
    - Every pair filed so far (ztop50, gsurg, prank, hubwalk) is re-called when the null lands. Later pairs (vshare,
      pret, and every round after) are called both ways. **The re-call decides.**
    - A pair that is PROMISING under its re-call gets its full run as its FULL file declares it. The run is
      re-queued under new item names, behind a gate on the re-call.
    - vshare's and pret's full runs now wait on their re-calls in place of their pair verdicts (docs/FULL_VSHARE.md
      and docs/FULL_PRET.md, amended at the same time).
    - A full-run grade's twelve reused screen reads take the floors. Its other 24 reads keep 0.0075, since no null
      covers their splits, and the grade says so.
  - **How screen_recall.py calls a read** (committed about 23:55, before any of the null's numbers):
    - A floor of at least 0.0075 can only turn a GAIN or a LOSS into WITHIN, never the reverse. So a read whose floor
      is 0.0075 keeps its filed call, and a read with a larger floor keeps a filed GAIN (LOSS) only when its filed
      difference is at least its floor (at most minus its floor).
    - The interval half of the rule is the filed call's, computed before rounding. The floor half uses the filed
      difference, rounded to 4 places like the null's. A read within 0.00015 of its floor is flagged; its call stands.
    - The null itself is refused unless it is the declared one: the base arm, seeds 1 and 2 on each of L-musique and
      L-hotpotqa (read from each fit's screen.json), compared with step 1's fits on p@swa and s1eval. A pair is refused
      when its seed-0 R@5 on a read is not the null's.
    - `screen_recall.py floors` files the twelve floors as `outputs/screen/scr-null-floors.{md,json}`, and
      `screen_recall.py grade` re-calls a full run's grade into its `grade-recall.{md,json}`.
  - **Order:** the four null fits queue behind round eight's screen and ahead of every full run. They take about 25 to
    35 minutes each; the null lands about 01:00 to 01:30.
- **Amended 8 October, about 02:45, before any of its numbers: the seed null over every full-run split.** Step 1's
  base arm, unchanged, is trained with seeds 1 and 2 on the four splits only full runs train (L-2wiki, L-squad, J5,
  L-metaqa), as the first null was on L-musique and L-hotpotqa. A full run's 36 reads then all take a floor from the
  null, and none keeps 0.0075 for want of one.
  - **Why:** every full run graded so far was NOT_ADOPTED on a LOSS of 0.008 to 0.014 on a read no null covers.
    - bdrop20: L-metaqa, webqsp read zero-shot −0.0081; J5, musique in-domain −0.0082.
    - zret: L-squad, musique in-domain −0.0093; L-2wiki, 2wiki read zero-shot −0.0081.
    - vshare: L-squad, metaqa in-domain −0.0094 and musique in-domain −0.0127; J5, musique in-domain −0.0108.
    - rel: L-2wiki, 2wiki read zero-shot −0.0136 (02:26, the comparison that prompted this amendment), beside metaqa
      in-domain +0.0795 and webqsp read zero-shot +0.1156.
    - On the two splits it covers, the null raised 5 of the 12 floors above 0.0075 (to 0.0102 to 0.0720), every
      zero-shot read but one among them. With 24 reads at 0.0075, a run of an arm with no effect at all would likely
      draw a LOSS somewhere, so the gate could not adopt anything.
  - **It works both ways.** A raised floor turns a GAIN within it into WITHIN as surely as a LOSS. rel's GAINs on the
    four new splits face the same floors as its LOSS.
  - **Fits:** `scr-nullx-<split>-s1` and `-s2` for L-2wiki, L-squad, J5 and L-metaqa: the base arm through
    lean_screen2.py's train with `--seed 1` or `--seed 2`, everything else step 1's, each compared with step 1's
    seed-0 p@swa of its split on the six s1eval carves. Seed 0 is step 1's fit itself.
  - **Floors:** the first null's form, unchanged and not tuned, on all 36 reads. `outputs/mp_unified/nullx.py`
    (committed before the null's numbers; its selftest passes) runs screen_recall.py's floors and grade unchanged,
    with the null's splits widened to all six. It files `outputs/screen/scr-nullx-floors.{md,json}`, and each full
    run's `grade-nullx.{md,json}`. **A full run's re-grade under the null over every split decides.**
  - **Which runs it re-grades:**
    - rel (docs/FULL_ROUND9.md): its remaining fits and reads keep running, since under this null they can change its
      verdict.
    - zret (docs/FULL_ZRET.md): its grade is complete.
    - vshare (docs/FULL_VSHARE.md): its L-metaqa fit ended before its items were taken out, so its read, comparison
      and grade are re-queued (`-x` names) for the re-grade.
    - gsurg's `-rc` run and relz's run (docs/FULL_ROUND10.md) when they are graded.
    - bdrop20's run stopped with three of its fits never run, so it has no complete grade to re-grade.
  - **If more than one arm is ADOPTED,** each re-grade is filed, and the next base is declared in a later round,
    before its numbers. Arms that change the same thing (zret's and ztop50's z-scores) do not stack by default.
  - **Order:** the eight fits queue behind round ten's screen and ahead of gsurg's `-rc` fits not yet started. They
    take about 20 to 35 minutes each alone. The floors land about 04:30 to 05:30.
    - Moved at about 03:15, before any of the null's or relz's numbers: the null's fits go ahead of round ten's
      screen fits. relz's full run needs rel's re-grade under this null either way, so the screen gains nothing by
      going first, and every re-grade lands about 40 minutes sooner. No rule changes.
- **rrf** (plain retrieval, no learned scorer) is reported beside every read as the zero-shot floor.
- **The floor 0.0075** is the lean track's measured one-seed training noise: across the lean_mlp to lean_mlp8 fits,
  one-seed differences under about 0.75 R@5 points are noise. The bootstrap covers question sampling and the floor
  covers training noise. One seed cannot separate an effect smaller than the floor from noise, and a screen does not
  try.
- **The null check** (`scr-null`): the base arm trained for one epoch on the same carves must give step 1's p@ep0 state
  bit for bit (`screen_null.py` writes `outputs/screen/null.json`).
  - IDENTICAL: a screen's difference is its arm's alone.
  - DIFFERENT: the harness adds noise of its own, the floor carries it, and this file says so.
- **The smoke** (`scr-smoke`, 14:24, passed): every arm trains one epoch on 2wiki select twice, then reads, compares
  and runs the CPU ablate. Each arm's repeat was IDENTICAL.

## 3. What a screen is not

A screen is not a result and not a grade. It gives development numbers on one split and one seed, and it never adopts
anything. An adopted change's numbers come from its full run. The paper's numbers come from one declared confirmation
run.

## 4. The first screens

Each arm is one idea, aimed at all six datasets at once. The zero-shot gaps of step 1 (docs/STEP1_MATCHED_SELECTION.md)
are training's, not selection's; these arms test what in training makes them.

| screen | arm | idea |
| --- | --- | --- |
| scr-zonly | zonly | Each block enters as its pool z-score and keep flag only; the raw values are dropped. The raw columns carry each dataset's own scales (cosines, ranks, walk masses, distances); the z-scores do not. |
| scr-dnorm | dnorm | The raw values stay, each column standardised by its own carve's mean and sd. Label-free; at read time the read carve's own statistics are used. The z-scores and rrf's base are unchanged. |
| scr-bound1 | bound1 | The correction is bounded: s = base_w·z(rrf) + tanh(out), at most one pool sd of rrf's z-score either way. The unbounded correction can overturn rrf's order on an unseen graph. In the QD track, a tanh bound kept zero-shot graphs within noise. |
| scr-bdrop20 | bdrop20 | Block dropout in training: each (question, block) is kept with probability 0.8, from its own generator (seed + 101). The scorer learns not to lean on any one block; in the lean track, one block (SEED) carried squad's loss. Every block is present at read time. |
| scr-pools | base, step 4c's cache | Step 4c's P_F pools: one fit instead of 4c's six. |

The scr-pools screen is compared with two baselines:

- step 1's L-musique, on the frozen pools (this one decides);
- step 4c's early read of the unchanged L-musique fit on P_F (reported only).

**The diagnosis** (`scr-ablate-L-musique`, CPU, no training) reads step 1's L-musique p@swa with each block's
within-pool information removed: the block's rows are set to their pool mean, so its z-score is 0 and its level stays.
It is run four ways:

- one block at a time (−B);
- every block but one (+B);
- every block;
- none.

It shows which blocks carry the in-domain gain and which carry the zero-shot loss. The next screens come from it.

**Step 1's other variants, read as screens (no training; declared 14:55).** Step 1's L-musique fit trained four
variants on the screen split: p, pf (p with FiLM context), n (p without SEMB) and nf (n with FiLM context). Every
state was read on the six s1eval carves. pf@swa, n@swa and nf@swa are each compared with p@swa by this rule
(`outputs/mp_unified/screen_variants.py`, `outputs/screen/variants-L-musique.md`). n tests whether the model does
better without SEMB, the block that carries metaqa.

**Step 2, read as a screen.** Step 2's eval-mix weighted fits of J5, L-metaqa and L-squad trained before the rule. Its
other three fits (L-musique, L-hotpotqa, L-2wiki) were cancelled at about 14:00, 18 to 40 minutes into training, so
step 2's declared grade does not run. Each of the three is compared with step 1's fit of the same split by this rule (`outputs/screen/s2-*.md`).

**Held at about 14:40:**

- Every queued seed-1 and seed-2 run of parts 9 to 15, and the grades that need them (112 items).
- Step 4c's six fits; scr-pools replaces them.
- The seed-0 GNN runs of parts 10 and 11 (29 items), until the screens' reads are sent.

cs26-g-k6w-s0 was cancelled at 14:30 so that the screens get the card.

### Second round (declared about 15:20, before any of its numbers)

Code: `outputs/mp_unified/lean_screen2.py`. It runs lean_screen's arms, commands and rule unchanged and adds two arms.
Both are scored by section 2's rule against step 1's L-musique p@swa.

| screen | arm | idea |
| --- | --- | --- |
| scr-pad | pad | Pool padding in training. With probability 0.5, a training question's pool gets copies of its own unranked non-gold rows (rrf 0) until retrieval's ranked rows are a share t of it. t is uniform in [0.08, 0.20], the measured ranked shares of the big pools (metaqa 0.10, webqsp 0.15, musique 0.18); a pool holds at most 4,096 rows. A pool with no unranked non-gold row (squad's) or no ranked row is never padded. Each training carve draws from its own generator, reset at the start of each fit. Reads are never padded. |
| scr-padbd20 | pad and bdrop20 | Both together: do they add? Compared with step 1's L-musique (decides) and with scr-bdrop20 (reported only). |

**Why padding.** The pool composition (Results) shows that in this split's training set, the only pools in which
retrieval ranks few rows are metaqa's, and retrieval misses most of metaqa's golds. musique's pools have as few
ranked rows, but retrieval ranks almost all of musique's golds. Padding gives the passage datasets such pools with
ranked golds, so a low ranked share no longer marks a KB. The copies enter the pool z-scores, rrf's base z-score and
the listwise softmax as rows of their own.

**The smoke** (`scr2-smoke`) trains both arms for one epoch on 2wiki select, twice, and reads them. A repeat must be
IDENTICAL and padding must pad some questions. It also checks every split's fit carves and basis, as lean_screen's
train builds them, against step 1's train.json. A failure drops the second round and the full run below.

**bdrop20's full run** (PROMISING at 14:57) is declared in `docs/FULL_BDROP20.md` and runs beside these screens.

**scr-pools was MIXED (15:12),** so step 4c's six fits do not run (docs/STEP4C_WALK_POOLS_RETRAINED.md).

### Third round (declared about 15:55, before any of its numbers)

Code: `outputs/mp_unified/lean_screen3.py` (its selftest passes). It runs lean_screen2's commands and lean_screen's
rule unchanged and adds one arm, scored by section 2's rule against step 1's L-musique p@swa.

| screen | arm | idea |
| --- | --- | --- |
| scr-zret | zret | Every pool z-score is taken against the pool's retrieved rows (rrf > 0) instead of all of its rows: each block's z-scores in the model, and rrf's base z-score. A pool with fewer than two retrieved rows, or a column constant over them, keeps its whole-pool z-score there. Label-free, the same in training and at read, no new parameters. On squad's pools, where every row is retrieved, it is the base model bit for bit. |

**Why zret.** scr-pad showed that the share of a pool's rows that retrieval ranked is part of what musique's zero-shot
read misuses: padding the passage pools down to the big pools' share lifted musique by 0.083. Padding changes the
training data, and that cost hotpotqa 0.009 in-domain. zret takes the share out of the inputs instead. A retrieved
row's z-scores no longer depend on how many unretrieved rows share its pool. Nor does rrf's base z-score: against the
whole pool, a top row's z-score grows roughly as one over the square root of the retrieved share, so it is two to
three times larger in a pool with a share of 0.1 than in one with 0.7.

**The smoke** (`scr3-smoke`) trains zret one epoch on 2wiki select, twice, and reads it. The repeat must be
IDENTICAL. A stronger block dropout (0.4) was drafted beside zret and dropped before any run, because block dropout
was NOT_ADOPTED in its full run.

**If scr-zret is PROMISING, its full run starts by itself** (docs/FULL_ZRET.md, declared about 16:15, before zret's
numbers): a gate on the host reads the screen's verdict, and on any other verdict the full run's items are
dropped unrun.

### Fourth round (declared about 19:55, before any of its numbers)

Code: `outputs/mp_unified/lean_screen4.py` (its selftest passes). It runs lean_screen2's commands and lean_screen's
rule unchanged and adds one arm, scored by section 2's rule against step 1's L-musique p@swa. zret's screen fit
(scr-zret) is a second base, reported only.

| screen | arm | idea |
| --- | --- | --- |
| scr-ztop50 | ztop50 | zret with a reference of fixed size. Every pool z-score, rrf's base z-score included, is taken against the pool's top 50 retrieved rows by rrf (rrf > 0; ties by row order) instead of all of its retrieved rows. A pool with fewer than two reference rows, or a column constant over them, keeps its whole-pool z-score there, as in zret. Label-free, the same in training and at read, no new parameters. Where no pool has more than 50 retrieved rows it is zret bit for bit; on squad's pools (50 rows, all retrieved) it is the base model bit for bit. |

**Why ztop50.** zret's full run (docs/FULL_ZRET.md) gained on the big pools read zero-shot (metaqa +0.064, musique
+0.120, webqsp +0.025) and lost on the small ones (hotpotqa −0.014, 2wiki −0.008). zret took the retrieved share
out of the z-scores, but not the number of retrieved rows they are taken over. That number is 205 to 379 in the big
pools and 57 to 63 in hotpotqa's and 2wiki's (pool composition, Results). A top row's z-score depends on it: the best
of 300 rows sits further out in the tail than the best of 60. So the same z-score still means different things in
the two kinds of pool, and a fit learns that from its training datasets. With the top 50 as every pool's reference,
a z-score is measured against the same number of rows, drawn from the head of retrieval's list, in every pool. 50
is the most that squad's pools hold, and close to the small pools' medians (55 and 60).

**What the screen can show.** On L-musique it reads musique zero-shot, where zret gained 0.120. It cannot show the
small-pool losses: hotpotqa and 2wiki are in this split's training set, and only L-hotpotqa and L-2wiki read them
zero-shot. Both earlier full runs failed on reads their screens did not make. Those reads come only from the full run.

**The smoke** (`scr4-smoke`) trains ztop50 one epoch on 2wiki select, twice, and reads it. The repeat must be
IDENTICAL.

**If scr-ztop50 is PROMISING, its full run starts by itself** (docs/FULL_ZTOP50.md, declared with this round, before
its numbers). On any other verdict the full run's items are dropped unrun.

**Amended about 20:35, before any of this round's numbers (section 2: two fits per screen).** A second fit,
**scr-ztop50-hp**, trains ztop50 on L-hotpotqa: the same config, seed and caps as zret's full-run fit of that split
(cap 0.26, share 0.28). It is read on the six s1eval carves and compared with step 1's L-hotpotqa p@swa (decides) and
with zret's full-run L-hotpotqa fit (reported only). It reads hotpotqa zero-shot, where zret lost 0.014, and musique
in-domain. The screen's verdict is taken over both fits' twelve reads (`scr-pair-ztop50`, through screen_pair.py), and
the full run's gate reads that verdict instead of scr-ztop50's alone. 2wiki read zero-shot still comes only from the
full run.

### Fifth round (declared about 21:15, before any of its numbers)

Code: `outputs/mp_unified/lean_screen5.py` (its selftest passes). It runs lean_screen2's commands and lean_screen's
rule unchanged and adds two arms. Each screen trains two fits under section 2's amended rule: L-musique and
L-hotpotqa, each compared with step 1's p@swa of its split, the verdict taken over both fits' twelve reads
(`screen_pair.py`).

| screen | arm | idea |
| --- | --- | --- |
| scr-prank, scr-prank-hp | prank | **Rank inputs (a preprocessing step).** Each fixed block's within-pool z-score is replaced by two rank columns: the column's competition rank in the pool from the top and from the bottom (ties share the best position), each as 60 / (60 + min(rank, 50)). That is reciprocal rank fusion's form, cut at ztop50's reference size. The raw values and the presence flags stay. SEMB (learned) keeps its z-score. rrf's base z-score is step 1's, unchanged. Label-free, the same in training and at read time; no new hyperparameter. |
| scr-gsurg, scr-gsurg-hp | gsurg | **Gradient surgery across the training datasets (an objective; PCGrad, Yu et al. 2020; for domain generalisation, Mansilla et al. 2021).** lean_gpu's loop step for step: one step per training dataset per chunk of 32 questions, the same batches. Before each step, its gradient loses its component along any other training dataset's gradient in the same chunk that it conflicts with (a negative dot product). The other datasets are taken in sorted order, on the gradient as projected so far. Their gradients are taken at the step's parameters with dropout off, so the fit's own dropout draws are step 1's. Where no pair conflicts, a step is step 1's exactly. Reads are the base model's. No hyperparameter. |

**Why these two.** Every arm so far changed how a pool's z-scores are taken (zonly, dnorm, zret, ztop50) or what the
fit sees (padding, block dropout). Each traded metaqa in-domain for musique zero-shot or the reverse. The reason is
the same in each case. In L-musique's training set only metaqa has big pools, so a fit learns "big pool, so trust rrf
less" from one dataset, and every change to the pool statistics moves that one dataset against the others. ztop50's
musique gain also came mostly from a stronger rrf base z-score: musique read zero-shot is still below rrf alone
(0.408 against 0.473).

- **prank** keeps step 1's base, so a gain here comes from the correction's inputs alone. A z-score moves with the
  pool's size and tail: a retrieved node in a 2,000-row pool sits far out in its column's tail, and in a 60-row pool
  it does not. Raw scales also differ between graphs. A node's rank among the pool's top 50 does neither. This is the
  preprocessing that rank fusion and learning-to-rank's query-level normalisation rest on (Cormack et al. 2009; the
  LETOR benchmarks).
- **gsurg** changes no input and adds no parameter. It acts on the conflict itself. Where metaqa's step says "trust
  rrf less" and a passage dataset's says "trust it more", neither step may undo the other along the shared direction.
  The selftest's toy datasets showed that early in a fit, rrf's weight (base_w) carries most of a step's gradient.
- Both change every training dataset alike, and prank every read alike. No new graph, encoder, text or model is used.

**What the screens can show.** L-musique reads musique and webqsp zero-shot. L-hotpotqa reads hotpotqa and webqsp
zero-shot and musique in-domain. 2wiki read zero-shot comes only from a full run.

**The smoke** (`scr5-smoke`) trains each arm for one epoch twice and reads it. prank trains on 2wiki select; gsurg on
2wiki and hotpotqa select, so that its fit meets other datasets' gradients. Each repeat must be IDENTICAL, and
gsurg's fit must meet at least one conflicting pair. The four fits wait on it.

**If a pair verdict is PROMISING, that arm's full run starts by itself** (docs/FULL_ROUND5.md, declared with this
round, before its numbers). On any other verdict the arm's full-run items are dropped unrun.

### Sixth round (declared about 21:25, before any of its numbers)

Code: `outputs/mp_unified/hubwalk.py` (its selftest passes). It runs lean_screen2's commands and lean_screen's rule
unchanged, and adds one arm and the build of its arrays. As in the fifth round, the screen trains two fits, L-musique
and L-hotpotqa. Each is compared with step 1's p@swa of its split, and the verdict is taken over both fits' twelve
reads.

| screen | arm | idea |
| --- | --- | --- |
| scr-hubwalk, scr-hubwalk-hp | hubwalk | **Hub-discounted walks (a preprocessing step).** Every walk column the model reads becomes the walk's probability mass instead of its path count. A walk starts uniform on its seeds; at each hop, each node's mass splits evenly over its out-edges in that walk's edge set. A column is log1p(n × mass): the mass's lift over the pool's uniform distribution, for a pool of n nodes. This covers WALK's eleven walk columns and WALKF's thirteen. The degree column, the first-hop flags and the seed flag stay step 1's bit for bit, and so does every other block (topo_STRUCT's two-hop count is a column of the look's and stays). Label-free, the same in training and at read time, on step 1's graphs and seeds; no new hyperparameter. |

**Why hubwalk.** Step 1's walk features are path counts (log1p). A path count grows with the pool's density. Through
a KB hub (a genre, a country), a two-hop count runs into the thousands; in a passage pool of about 100 nodes it stays
in single digits. So the same column means different things on the KB graphs and on the passage graphs, and a fit
learns the difference from the datasets it trains on: the big-pool confound again, in the walk features.

- **Mass splits at every hub**, so a path through a node with many edges counts for less. That is resource
  allocation (Zhou, Lü and Zhang 2009), Adamic and Adar's weighting (2003), and personalised PageRank's random-walk
  mass, read hop by hop.
- **The lift n × mass stays put when a pool's size and degrees grow together.** A pool made of two copies of a pool
  gives the same columns, and the selftest checks this.

**The build** (`hubwalk.py build`, CPU, one item per carve) covers:
- the five fit carves of the two splits' training sets: metaqa, squad, hotpotqa and 2wiki fit, and musique s1fit;
- the six s1eval carves;
- the smoke's 2wiki select carve.

Every query is also computed as path counts. Those must equal step 1's cached walk and walkf bit for bit, so the rows
line up. The mass columns must reach exactly the nodes the counts reach. Any difference refuses the carve.

**The smoke** (`scr6-smoke`) runs on 2wiki select. It reads the carve through the arm and through step 1's carve: the
mass columns must have moved, and the reached nodes and every other column must not. Then it trains one epoch twice
(the repeat must be IDENTICAL) and reads.

**If the pair verdict is PROMISING, its full run starts by itself** (docs/FULL_HUBWALK.md, declared with this round,
before its numbers). On any other verdict its items are dropped unrun.

**Amended about 21:27, before any of this round's numbers.** Two of the thirteen builds (metaqa fit, webqsp s1eval)
failed on the host. Moving a finished file into place hit a file another process held for a moment (WinError 32 at
os.replace). The other eleven passed, their path counts equal to step 1's bit for bit. hubwalk.py now retries the move
for up to two minutes; nothing it computes changed. The two builds, and every item the feeder dropped behind them, run
again under the same commands with `-b` added to their item names (`fhw-gate-b` for the gate).

**Builds and smoke, 21:37.** All thirteen carves are built: 40 parts, 53,885,351 rows, and every part's path counts
equal step 1's bit for bit (`path_counts_against_step1: IDENTICAL` in each part's record.json). The smoke passed
(`outputs/screen/smoke6/smoke.json`):
- On WALK and WALKF, the kept columns equal step 1's, the reached nodes are the same, and the mass moved; every other
  column is unchanged.
- The one-epoch repeat is IDENTICAL, and the read runs.

The feeder runs GPU items in file order, and the smoke (0.12 of the card) was queued behind scr-train-prank-hp, which
waits for 0.28. About 21:34 it was moved ahead of that item, into the 0.145 of the card left idle; prank-hp still fits
when the first running fit ends. This changes when the smoke ran, not anything it computes.

### Seventh round (declared about 22:35, before any of its numbers)

Code: `outputs/mp_unified/lean_screen7.py` (its selftest passes). It runs lean_screen2's commands and lean_screen's
rule unchanged and adds one arm. As in rounds five and six, the screen trains two fits, L-musique and L-hotpotqa. Each
is compared with step 1's p@swa of its split, and the verdict is taken over both fits' twelve reads.

| screen | arm | idea |
| --- | --- | --- |
| scr-vshare, scr-vshare-hp | vshare | **Share augmentation of the pool statistics (a preprocessing step, in training only; feature-statistics augmentation for domain generalisation, as in MixStyle, Zhou et al. 2021, and DSU, Li et al. 2022).** With probability 0.5, a training question's pool z-scores (every fixed block's, SEMB's and rrf's base z-score) are taken with row weights that make retrieval's ranked rows a share t of the pool's weight. t is uniform in [0.08, 0.20] (the big pools' measured shares) when the pool's own share is at least 0.375, and in [0.55, 0.95] (the small passage pools' 0.56 to 0.69, and above) when it is below. Ranked rows weigh 1; unranked rows weigh r(1 − t) / (t u), for r ranked and u unranked rows. No row is added or removed: the listwise softmax, the raw values and the golds are the pool's own. A pool with no ranked or no unranked row (squad's) is never reweighted. Each training carve draws from its own generator, reset at the start of each fit. Label-free; reads unchanged; no new parameter. |

**Why vshare.** The diagnosis located the channel (pool composition, 14:57; scr-pad, 15:46; scr-zret, 16:46). The
share of a pool's rows that retrieval ranked sets every pool z-score, and in L-musique's training set only metaqa's
pools are thinly ranked.
- **pad** changed the share in one direction only, by copying rows. musique read zero-shot gained 0.083, but hotpotqa
  in-domain lost 0.0086: half of hotpotqa's training pools were full of copies the softmax had to rank.
- **zret and ztop50** took the share out of the z-scores at read time too, and each moved metaqa and musique in
  opposite directions.
- **vshare** keeps the reads and the softmax as they are. It changes only the statistics a training pool is
  normalised by, in both directions: small passage pools are seen as thinly ranked and the big pools as richly ranked.
  A ranked share then no longer names the kind of graph in training, and the fit must tell metaqa's unranked golds
  from musique's unranked noise by the rows themselves.
- **A weight w on a row is that row copied w times** in the statistics (the selftest checks it on integer weights).
  The thinning half is pad's statistics without pad's copies in the softmax.
- It changes every training dataset alike. No new graph, encoder, text or model is used.

**What the screens can show.** As in round five: L-musique reads musique and webqsp zero-shot. L-hotpotqa reads
hotpotqa and webqsp zero-shot, and musique in-domain, whose big pools are enriched there beside metaqa's.

**The smoke** (`scr7-smoke`) trains the arm for one epoch twice on 2wiki's and metaqa's select carves, so that it both
thins (2wiki) and enriches (metaqa), then reads both. The repeat must be IDENTICAL, and the fit must have thinned and
enriched at least one pool each. The two fits wait on it.

**If the pair verdict is PROMISING, its full run starts by itself** (docs/FULL_VSHARE.md, declared with this round,
before its numbers). On any other verdict its items are dropped unrun.

### Eighth round (declared about 23:20, before any of its numbers)

Code: `outputs/mp_unified/lean_screen8.py` (its selftest passes). It runs lean_screen2's commands and lean_screen's
rule unchanged and adds one arm: prank narrowed. Section 2 allows this after a MIXED screen ("a narrower screen may
follow, declared here"). As in rounds five to seven, the screen trains two fits, L-musique and L-hotpotqa. Each is
compared with step 1's p@swa of its split, and the verdict is taken over both fits' twelve reads.

| screen | arm | idea |
| --- | --- | --- |
| scr-pret, scr-pret-hp | pret | **Rank inputs for the retrieval blocks only (a preprocessing step; prank, narrowed).** The two retrieval blocks take prank's form: the raw values, each column's competition rank in the pool from the top and from the bottom as 60 / (60 + min(rank, 50)), and the presence flag (`lean_screen5.rank_inputs`, unchanged). They are rank (dense_rr, splade_rr, rrf, agreement, is_seed) and dense_cos: 6 of the 86 fixed columns, where prank ranked all 86. Every other block keeps step 1's form [raw, z, flag]: the structure blocks (topo_STRUCT, depth_STRUCT, WALK, WALKF, SEED, DISTS) and SEMB keep their within-pool z-scores. rrf's base z-score is step 1's. Label-free, the same in training and at read time; no new hyperparameter. With no block ranked the model is step 1's, and with every fixed block ranked it is prank: the selftest checks both, weight for weight. |

**Why pret.** prank's L-musique fit (`outputs/screen/scr-prank.md`) is MIXED. It has the largest zero-shot gain any
screen has had, and it loses in-domain:
- **Gains:** musique read zero-shot +0.2016 (0.270 to 0.471; rrf 0.473), and webqsp zero-shot +0.0269.
- **Losses:** metaqa in-domain −0.0310 and hotpotqa in-domain −0.0080. 2wiki −0.0068 is WITHIN.

A rank cut at 50 keeps a row's place among a column's top 50 and erases everything past it.
- **Retrieval columns:** a ranked row's z-score moves with the share of the pool that retrieval ranked. In a thinly
  ranked pool the ranked rows sit far out in the column's tail; its rank among the top 50 does not move (the
  selftest checks this on one column). That share is the confound the diagnosis found: in L-musique's training set
  only metaqa's pools are thinly ranked.
- **Structure columns:** in metaqa's pools of about 2,000 rows, retrieval misses 57% of the golds. The walks, seed
  similarity and distances find them, mostly past the top 50 of those columns, where a cut at 50 flattens them and a
  z-score does not.

pret ranks the retrieval columns and keeps the structure z-scores. It asks one question: does the zero-shot gain come
through the retrieval columns?
- **If it does,** pret keeps musique's gain without metaqa's loss.
- **If it does not,** pret is NO_GAIN on musique. The gain then needs the structure ranks too: the structure z-scores
  carry the share as well (as vshare's declaration says, the share sets every pool z-score).

Timing and the amendment:
- prank's L-hotpotqa fit was still training when this was declared; pret does not wait on its numbers.
- prank has no full run (its pair cannot be PROMISING), so section 2's amendment of about 15:55 does not apply.

**What the screens can show.** As in round five: L-musique reads musique and webqsp zero-shot. L-hotpotqa reads
hotpotqa and webqsp zero-shot, and musique in-domain.

**The smoke** (`scr8-smoke`) trains the arm for one epoch twice on 2wiki's select carve, then reads it. The repeat must
be IDENTICAL. The two fits wait on it.

**If the pair verdict is PROMISING, its full run starts by itself** (docs/FULL_PRET.md, declared with this round,
before its numbers). On any other verdict its items are dropped unrun.

### Ninth round (declared 8 October at about 01:00, before any of its numbers)

Code: `outputs/mp_unified/relcols.py` (its selftest passes). It runs lean_screen2's commands and lean_screen's rule
unchanged and adds two arms. Each adds a block set that the look already compiles and step 1's pick does not read. As
in rounds five to eight, each screen trains two fits, L-musique and L-hotpotqa. Each fit is compared with step 1's
p@swa of its split, and the verdict is taken over both fits' twelve reads. Both pairs are re-called under the seed
null, and the re-call decides (section 2).

| screen | arm | idea |
| --- | --- | --- |
| scr-rel, scr-rel-hp | rel | **Question-relation inputs (the KB systems' matching, as a preprocessing step).** Three blocks of the look's compiled columns, 26 in all, enter after step 1's nine as [raw, z, flag]. typed_rel (10): the cosine of the question with each relation's text over the typed edges to a seed and over those incident to the node (max, max, mean); relation IEF; relation diversity; direction; edges to and from the seeds; the best two-step relation chain from a seed. typed_v2 (6): retrieval mass moved 2 and 3 steps over typed in-edges weighted exp(cos(q, e_r) / 0.1); typed walk counts; the best, mean and weakest-step question-relation match along 3-step typed walks from a seed. ordered (10): the best 3-step walk's question-relation match, direction and consecutive-relation similarity, step by step. On a graph without typed relations (the four passage graphs) every one is 0. |
| scr-gcs, scr-gcs-hp | gcs | **Parameter-free propagation of retrieval (GraphER's GCS: a two-step personalised PageRank from retrieval's scores).** Two columns: max(p_2, s) with p_{t+1} = 0.5 s + 0.5 W p_t, s = rrf / max rrf in the pool, W the row-normalised pool graph over every edge family (gcs_full) or over the structural edges (gcs_struct). Live on every graph. |

Both arms are label-free and the same in training and at read time, and they change every dataset alike. No new graph,
text, encoder or model is used. The values are the look's float16 columns row for row, the inputs the twin and the GNN
read, copied as step 1's 35 cached columns are. The build checks that the same chunks give step 1's cached columns bit
for bit.

**Why these two (8 October; the user asked which features or methods we are missing against the published systems).**
- **The gap on the KB datasets is relational.** NuTrea and ReaRev (MetaQA, WebQSP) score a node by matching the
  question against the relations on the paths to it. Step 1's MLP reads no relation at all: its pick was chosen on
  2wiki, an untyped graph.
- **The twin reads these 26 columns** (twin0: the six-dataset MLP on the look's 129 columns) and is ahead of J5's MLP
  on metaqa at every hop, both in-domain. s1eval R@5 / hit@1 (`outputs/step1/grade.json`), J5 against the twin:
  - 1-hop: .9215 / .7950 against .9523 / .8883.
  - 2-hop: .7513 / .7265 against .8190 / .8835.
  - 3-hop: .3387 / .4253 against .4576 / .5707.

  The twin also reads 66 other columns step 1 does not, and it trains on webqsp too. So the screen tests whether the
  relation columns carry that gap; the twin's lead does not show it.
- **It is not the topic-entity oracle.** The KB papers start from the gold topic entity. Our seeds (dense top-5 and
  SPLADE top-5) hold metaqa's topic entity for 98.1% of questions (by hop .992, .993, .959).
- **gcs** is GraphER's parameter-free arm, within 0.2 to 1.9 PR@10 of its GAT (its Table 5). It is also HippoRAG's
  mechanism in short form: retrieval's own scores spread over the graph. Step 1's walks spread a seed indicator, not
  retrieval's scores.
- **Out of reach of a preprocessing step:** MuSiQue R@5 against HippoRAG 2 (74.7). Its encoder alone (NV-Embed-v2)
  gives 69.7, on a corpus ten times smaller; our structure lifts rrf by 8.8 points, theirs lifts its retriever by 5.0.
  The encoder stays frozen.

**What the screens can show.**
- Both fits train metaqa, the only typed KB in training, and read webqsp zero-shot. metaqa in-domain is where rel
  should gain. webqsp tests whether a channel learned on metaqa's 9 relations carries to webqsp's 7,058.
- The passage reads see the rel blocks at 0, so only a change in how training weighs the other blocks can move them.
  Their calls test that.
- gcs is live on all six reads.

**A split with no KB in training (L-metaqa) has its rel blocks at 0 on every training row.** lean_gpu's dead-block
rule drops them, so that fit is step 1's model. rel can change only the splits that train metaqa.

**The smoke** (`scr9-smoke`) first loads metaqa select, webqsp s1eval and 2wiki select through each arm's carve. The
base matrix, golds and rows must equal lean_gpu's carve, and the arm's columns its build's. It then trains each arm for
one epoch twice on metaqa's select carve: the repeat must be IDENTICAL and the arm's blocks live. Then it reads that
carve. The fits wait on it.

**If a pair's re-call is PROMISING, its full run starts by itself** (docs/FULL_ROUND9.md, declared with this round,
before its numbers). On any other re-call verdict its items are dropped unrun.

### Tenth round (declared 8 October at about 02:35, before any of its numbers)

Code: `outputs/mp_unified/relz.py` (its selftest passes). One arm, relz: ztop50's model (lean_screen4.py) on rel's carve
and blocks (relcols.py), both used unchanged. As in rounds five to nine, the screen trains two fits, L-musique and
L-hotpotqa, with rel's screen fits' config, seed and caps.

| screen | arm | idea |
| --- | --- | --- |
| scr-relz, scr-relz-hp | relz | **Pool normalisation on top of the relation inputs.** rel's blocks (step 1's nine, then typed_rel, typed_v2 and ordered) under ztop50's forward: every block's within-pool z-score, rel's included, and rrf's base z-score are taken against the pool's top 50 retrieved rows by rrf, not the whole pool. A pool with fewer than two such rows, or a column with no spread over them, keeps the whole-pool z-score. No new column, block or hyperparameter. |

**The base is rel, not step 1.**
- Each fit is compared with rel's screen fit of its split (`scr-rel`, `scr-rel-hp`), which decides, and with step 1's,
  reported.
- relz's model has rel's parameters and widths: ztop50 changes only the forward. With the same seed, its initial
  weights and batch order are rel's, so a difference comes from ztop50's forward and what training does with it.
- The pair's verdict is screen_pair.py's rule over the twelve reads. Its re-call is screen_recall.py's under the seed
  null's floors, with each read's base R@5 rel's screen fit's (`relz.py pair` and `relz.py recall`: both files are
  used unchanged, only their loaders are swapped). rel's screens were compared with the null's seed 0, and the
  re-call checks that. The re-call decides.

**Why (8 October; the gap to the published systems).**
- **ztop50 gave the largest zero-shot gain of any screen:** musique read zero-shot +0.1387 in the L-musique fit
  (0.270 to 0.408), and musique in-domain +0.0115 in the L-hotpotqa fit.
  - Under the seed null its only LOSSes left are metaqa in-domain, in both fits: −0.0176, and −0.0252 against a floor
    of 0.0149.
  - Its hotpotqa and webqsp zero-shot LOSSes in the L-hotpotqa fit are within their floors.
- **The reading filed then:** a reference of fixed size takes the pool's size out of the z-scores. metaqa, the only
  training dataset with big pools, lost what its fit had learned from them.
- **rel gives metaqa's rows a signal of their own,** the question-relation match (hit@1 0.637 to 0.781). If metaqa's
  loss under ztop50 came from losing the one cue that set its rows apart, rel's columns now carry another, and the
  two may add.
- **They may also overlap.** rel's musique zero-shot gain (+0.0868) and ztop50's (+0.1387) may be one effect: metaqa's
  big pools no longer pulling the weights the passage graphs share. Then relz adds little on musique over rel, and
  the screen shows that.
- **This departs from the note filed at 21:22 on 7 October** (no further screen of how a pool's z-scores are taken).
  That trade, metaqa against musique, was measured on a model with no KB signal. One pair tests whether rel removes
  it.

**What the screen can show.**
- **metaqa in-domain:** whether rel's columns hold metaqa when ztop50 takes the pool's size away. Against rel, no
  LOSS is the test.
- **musique,** read zero-shot (L-musique) and in-domain (L-hotpotqa): whether ztop50's gain survives on rel's base.
- **webqsp read zero-shot:** rel's relation channel under the new normalisation.
- **The passage graphs in-domain:** ztop50 left them WITHIN on step 1's base.

**The smoke** (`scr10-smoke`, `relz.py smoke`) trains rel and relz for one epoch on metaqa's select carve, relz twice
(the repeat must be IDENTICAL), and reads that carve. relz's fit must hold rel's blocks live, with the same blocks as
rel's fit, the same questions, and scores that differ from rel's. The fits wait on it.

**If the re-call is PROMISING and rel's full run is ADOPTED under its re-grade, relz's full run starts by itself**
(docs/FULL_ROUND10.md, declared with this round, before its numbers). Otherwise its items are dropped unrun. The pair
and its re-call are filed either way.

**Order and ETAs.**
- The smoke and the two fits go ahead of gsurg's `-rc` fits not yet started, so the screen's verdict comes before
  that full run's.
- rel's L-squad and J5 fits end about 02:45 to 03:00, and its L-metaqa fit runs next.
- The smoke takes about 5 minutes. Each fit takes about 45 to 60 minutes beside the others (ztop50's L-musique fit
  took 43 alone), then a read of about 10.
- The pair and its re-call land about 04:15 to 05:00.

### Eleventh round (declared 8 October at about 03:50, before any of its numbers)

Code: `outputs/mp_unified/qdepth.py` (its selftest passes). One idea, a learned depth prior for each question, screened
on one base: rel's if rel's full run is ADOPTED under the seed null over every split, step 1's if it is NOT_ADOPTED. As
in rounds five to ten, the screen trains two fits, L-musique and L-hotpotqa.

| screen | arm | idea |
| --- | --- | --- |
| scr-qdepth, scr-qdepth-hp (step 1's base); scr-relqd, scr-relqd-hp (rel's base) | qdepth; relqd | **A question-to-depth prior.** Every row's score gains l_q[depth(v)]. depth(v) is the row's class from WALK's raw columns: 0 a seed, 1 to 3 the first structural hop from a seed (the nearest, should a row carry two), 4 unreached. l_q = q U + P_q W + b, five values per question, from its embedding q (SEMB's input) and its pool's profile P_q: the share of its rows in each depth class, and log(1 + the pool's size). U (1536 × 5), W (6 × 5) and b (5) start at zero, so each fit starts as its base model, bit for bit, from the same initialisation and batches; the listwise loss alone trains them, with the model's own Adam, learning rate and weight decay. 7,715 parameters. No new column, block or hyperparameter, and nothing reads a label at read time. A question whose WALK or SEMB block is dropped takes no prior. |

**Why (section 'Hop depth of the top-1 errors' in the results, filed at about 03:06).**
- **A perfect question-to-depth attention lifts the MLPs' metaqa 3-hop hit@1 by 0.074 and 0.083,** about ten standard
  errors, and the GNN's top-1 sits off-depth about half as often as the MLPs'. The KB systems (NuTrea, ReaRev,
  TransferNet) weigh a node by the hop count the question asks for. Step 1's MLP reads each row's depth but not the
  question's hop count, so it can only prefer one depth for every question alike.
- **On webqsp, read zero-shot, the MLPs rank a seed first on 0.60 to 0.95 of the questions,** where 0.07 of the golds
  are seeds, and the oracle doubles their hit@1. On metaqa the same fits rank a seed first on 0.02 to 0.03. The
  question alone cannot say which graph it is asked on; the pool's profile is in the prior for that.
- **What it cannot do:** most of rel's 3-hop misses sit at a depth that holds a gold (0.263 of the questions, the
  GNN's 0.108). A depth prior closes about a third of rel's 3-hop gap to the GNN, at most.

**The base is named by rel's re-grade, not by this screen.**
- `qdepth.py base --rel-grade outputs/full_rel/grade-nullx.json --want rel` exits 0 only on ADOPT, and `--want step1`
  only on NOT_ADOPTED. A missing or INCOMPLETE re-grade runs neither.
- **On rel's base (relqd):** rel's carve and blocks (relcols.py). Each fit is compared with rel's screen fit of its
  split, which decides, and with step 1's, reported. The pair, its re-call and the grade are relz.py's (rel's R@5 as
  each read's base), run under relqd's name: `qdepth.py pair-rel`, `recall-rel` and `grade-rel`. This is
  docs/FULL_ROUND9.md's rule: after ADOPT, rel is the base of every later screen.
- **On step 1's base (qdepth):** each fit is compared with step 1's fit of its split. The pair and its re-call are
  screen_pair.py's and screen_recall.py's, unchanged.
- The fits wait for the re-grade. That costs no time: the null's fits and round ten's hold the card until about then.
- Caps: qdepth step 1's screen caps, 0.26 (share 0.28) on both fits; relqd rel's, 0.26 (0.28) on L-musique and 0.30
  (0.32) on L-hotpotqa. Reads 0.28 (0.30).

**What the screen can show.**
- **metaqa in-domain** (both fits train on metaqa): the 3-hop questions are the test. The oracle is a ceiling, not a
  target.
- **webqsp, read zero-shot:** whether the prior, from the question and its pool's profile, stops ranking seeds first
  on a graph no fit trains on.
- **musique and hotpotqa** (each read zero-shot by one fit and in-domain by the other): whether a depth prior helps
  or harms the passage graphs, where depth comes from the same WALK columns.
- **The passage graphs in-domain:** the prior starts at zero, and the loss decides its size. No LOSS is the test.

**The smoke** (`scr11-smoke`, `qdepth.py smoke`) trains step 1's base arm once, qdepth twice (the repeat must be
IDENTICAL) and relqd once, one epoch each on metaqa's select carve, and reads that carve.
- qdepth's and relqd's priors must have moved from zero and be finite, and the base arm's fit must hold none.
- relqd must hold rel's blocks live, and qdepth's blocks must be the base arm's.
- All three must read the same questions, and qdepth's scores must differ from the base arm's.
- It runs at once, in the card's free share (0.14). The fits wait on it.

**If the re-call is PROMISING, the full run starts by itself** on the same base (docs/FULL_ROUND11.md, declared with
this round, before its numbers). Otherwise its items are dropped unrun. The pair and its re-call are filed either way.

**Order and ETAs.**
- The smoke takes about 5 minutes and starts now.
- rel's re-grade lands about 06:15 to 06:30, when the null's fits end. The two gate items run at once after it.
- The screen's items queue after round ten's. Each fit takes about 45 to 60 minutes beside the others, then a read of
  about 10. The pair and its re-call land about 07:15 to 07:45.

### Twelfth round (declared 8 October at about 05:30, before any of its numbers)

Code: `outputs/mp_unified/rmatch.py` (its selftest passes). One idea, a learned match between the question and the
relations along the graph's own typed chains, screened on one base, as in round eleven: rel's if rel's full run is
ADOPTED under the seed null over every split, step 1's if it is NOT_ADOPTED. The screen trains two fits, L-musique and
L-hotpotqa.

| screen | arm | idea |
| --- | --- | --- |
| scr-rmatch, scr-rmatch-hp (step 1's base); scr-relrm, scr-relrm-hp (rel's base) | rmatch; relrm | **A learned question-relation match along typed chains.** Every row's score gains g_m log(1 + S_m(v)/0.01) + g_r log(1 + S_r(v)). S_m(v) sums w_q(c) m_c(v) over the typed relation chains c that reach row v from the question's seeds, and S_r(v) sums w_q(c). A chain is up to three typed steps z = 2r + d (relation r, read forward or back). m_c(v) is the mass chainpop17's exact walk sends to v along c: 1/\|S\| on each seed of one bucket, split equally over each node's out-edges of type z in the pool, renormalised over the chain's non-seed rows. Per row and bucket the 64 heaviest entries are kept. log w_q(c) = log π_q(L) + β_b + Σ_k log σ(h_qk(z_k)), with h_qk(2r + d) = κ_k zcos_q(r) + (A_k q)·(B ẽ_r + D_d) + dir_kd + bias_k. q is the question's unit embedding (SEMB's input) and e_r the name embedding of relation r from the same frozen encoder (`outputs/m3b/relations`). zcos_q(r) is their cosine, z-scored over the relations on the question's own pool; ẽ_r is e_r's unit vector less the graph's mean. π_q is a softmax over the chain length from q. A (3 × 8 × 1536) starts random, κ at 1, and B (8 × 1536), D, dir, bias, π's weights, β and the gates at zero. So each fit starts as its base model, bit for bit, from the same initialisation and batches; the listwise loss alone trains the match, with the model's own Adam, learning rate and weight decay. 53,795 parameters. No new column, block or hyperparameter of the fit, and nothing reads a label at read time. A graph without typed relations (squad, musique, hotpotqa, 2wiki) has no chains, and there the arm is its base model. A question whose SEMB block is dropped takes no match. |

**Why (section 'Within the depth that holds a gold' in the results, filed at about 04:07).**
- **The gap is in the match, not the inputs.** On rel's metaqa 3-hop misses at a depth that holds a gold, the gold
  and the top-1 never share their relation columns (same_rel 0.000). Yet no fixed question-relation match column
  favours the gold: the best does on 0.533. The GNN prefers the gold on 0.743 of them, the twin on 0.338.
- **The KB systems learn which relation each hop of the question asks for.** TransferNet, ReaRev and NuTrea score
  each hop's relations against the question. Step 1's and rel's MLPs read a fixed cosine between the question and
  each relation name, summed over paths. This arm learns that match, per hop
  position, from the same frozen embeddings, along the chains the existing graph already holds.
- **Zero-shot on webqsp:** B, D and κ are learned on metaqa's nine relations. On webqsp's 7,058 the match leans on the
  per-pool z-scored cosine and on B's map of the name embeddings. Whether that transfers is part of the test.

**The chains are built once per carve** (`rmatch.py build`, on the host's CPU), from the look's chunks, beside step
1's cache: metaqa fit, select and s1eval, and webqsp s1eval. Each part is tied to its step-1 part's record, and its
pool sizes are checked against step 1's. No other carve has typed relations.

**The base is named by rel's re-grade, as in round eleven.**
- `rmatch.py base --rel-grade outputs/full_rel/grade-nullx.json --want rel` exits 0 only on ADOPT, and `--want
  step1` only on NOT_ADOPTED. A missing or INCOMPLETE re-grade runs neither.
- **On rel's base (relrm):** rel's carve and blocks. Each fit is compared with rel's screen fit of its split, which
  decides, and with step 1's, reported. The pair, its re-call and the grade are relz.py's under relrm's name
  (`rmatch.py pair-rel`, `recall-rel`, `grade-rel`).
- **On step 1's base (rmatch):** each fit is compared with step 1's fit of its split. The pair and its re-call are
  screen_pair.py's and screen_recall.py's, unchanged.
- Caps: rmatch 0.26 (share 0.28) on both fits; relrm 0.26 (0.28) on L-musique and 0.30 (0.32) on L-hotpotqa. Reads
  0.30 (0.32). The chain entries stay in host memory, and each batch's go to the card a chunk at a time.

**What the screen can show.**
- **metaqa in-domain** (both fits train on metaqa): the 3-hop questions are the test, where the GNN leads most.
- **webqsp, read zero-shot:** whether a match learned on nine relations transfers to 7,058.
- **The passage graphs:** they have no typed chains, so there the arm differs from its base only through training on
  metaqa's batches beside theirs. No LOSS is the test.

**Speed is timed cold (the user's rule, 8 October).** The chains are a query-local compile: a new question's chains
must be walked from its own pool when it arrives. The screen builds them ahead of time only to save the card's time,
and no speed figure may rest on that.
- Any latency figure for this arm times each question from scratch, with no warm-up pass and nothing kept from an
  earlier question: the typed walk from the pool's arrays (the typed graph, the walk, the seed drop and the row cap),
  the move of its entries to the device, and the forward. It goes in the cost table's split: graph index,
  query-local compile, forward.
- The relation-name tables are built once per graph. They are index cost, on their own row.
- **Measured already** (the laptop, unpinned, one process, on the local look chunks): the walk takes about 0.02
  seconds a question on metaqa and 0.3 on webqsp, cold. On webqsp that is about twice the GNN's single-query forward
  p50 there (about 0.155 seconds, docs/UNIVERSAL_GNN_SIX.md; another thread setting). On a graph with webqsp's
  relation fan-out the arm may cost the MLP its speed lead. An adoption is reported with that cost, and its latency
  stage is declared in its own file.

**The smoke** (`scr12-smoke`, `rmatch.py smoke`).
- It first checks the carves on metaqa select, webqsp s1eval and 2wiki select. The base's matrix, golds and rows must
  be unchanged, and typed carves must hold entries. On their first questions the arm at its start must score as its
  base model, bit for bit, with finite chain features.
- It then trains step 1's base arm once, rmatch twice (the repeat must be IDENTICAL) and relrm once, one epoch each
  on metaqa's select carve, and reads that carve.
- rmatch's and relrm's gates and B must have moved from zero and be finite, and the base arm's fit must hold none.
- relrm must hold rel's blocks live, and rmatch's blocks must be the base arm's.
- All three must read the same questions, and rmatch's scores must differ from the base arm's.
- It runs in the card's free share (0.16) once the chains of metaqa select and webqsp s1eval are built. The fits wait
  on it.

**If the re-call is PROMISING, the full run starts by itself** on the same base (docs/FULL_ROUND12.md, declared with
this round, before its numbers). Otherwise its items are dropped unrun. The pair and its re-call are filed either way.

**Order and ETAs.**
- The four chain builds start now on the host's idle CPU: a few minutes for each of metaqa's carves, and about 10
  to 15 for webqsp s1eval (measured on the laptop: 0.02 seconds a question on metaqa, 0.3 on webqsp). The smoke
  follows, about 10 minutes.
- rel's re-grade lands about 06:15 to 06:30. The two gate items run at once after it.
- The screen's items queue after round eleven's. Each fit takes about 45 to 60 minutes beside the others, then a read
  of about 10 to 15. The pair and its re-call land about 08:00 to 09:00.

### Thirteenth round (declared 8 October at about 06:45, before any of its numbers)

Code: `outputs/mp_unified/zgs.py` (its selftest passes). One arm, the two adopted arms together, decided against
zret's fits. The screen trains two fits, L-musique and L-hotpotqa.

**The base.** Both re-grades under the null over every split are ADOPT: zret's (05:14) and gsurg's (06:21, filed in the
results below). They change different things, zret the inputs' normalisation and gsurg the objective. So by section 2
and docs/FULL_ROUND5.md they combine, and a screen of the two together decides the next base.
- **zret is the base this screen is decided against, and the base of every later screen and run unless zgs is
  ADOPTED.** Of the two, zret has two primary GAINs: musique read zero-shot in L-musique's fit (+0.1202) and metaqa
  read zero-shot in L-metaqa's (+0.0644). gsurg has one: musique read zero-shot (+0.0842). Eleven of the seventeen
  arms screened against step 1's L-musique fit before qdepth moved that read by +0.08 to +0.24, gsurg among them.
- **zret's fits:** L-musique is its screen fit (`outputs/screen/fits/scr-zret`). Every other split, L-hotpotqa
  included, is its full run's (`outputs/full_zret/fits/<split>`), since zret's screen came before the two-fit screens.
- Section 2's amendment of 15:55 does not apply: neither arm is NOT_ADOPTED.

| screen | arm | idea |
| --- | --- | --- |
| scr-zgs, scr-zgs-hp | zgs | **zret's model trained by gsurg's loop.** zret's forward (every block's within-pool z-score, and rrf's base z-score, taken against the pool's retrieved rows), trained with gsurg's gradient surgery across the training datasets (before each step, its gradient loses its component along any other dataset's gradient in the same chunk that it conflicts with). Both parts are used unchanged: no new column, block or hyperparameter. Reads are zret's model, so its serving cost is zret's. |

**How it is decided.**
- Each fit is compared with zret's fit of its split, which decides. gsurg's fit and step 1's are reported beside it.
  The reads are the six s1eval carves at p@swa.
- The pair and its re-call under the seed null are relz.py's, run under zgs's name, with zret's R@5 as each read's
  base and the null's floors, as relz was decided against rel's screen fits.
- **If the re-call is PROMISING, the full run starts by itself** (docs/FULL_ROUND13.md, declared with this round). Its
  four fits are graded against zret's fits and re-graded under the null over every split. ADOPT makes zgs the base of
  every later screen and run. On any other result zret stays the base, and gsurg's objective is not carried.

**What the screen can show.** gsurg's one primary GAIN is on the read most arms move, and zret moves it further. A GAIN against zret's fits with no LOSS means the surgery adds something on zret's inputs. NO_GAIN means the two
do the same work there, and zret alone stays.

**The smoke** (`scr13-smoke`, `zgs.py smoke`) trains zret once, gsurg once and zgs twice (the repeat must be
IDENTICAL), one epoch each on 2wiki's and hotpotqa's select carves. Each is read on 2wiki's.
- zgs must hold zret's model, and its loop must meet conflicting pairs.
- Its scores must differ from zret's (the same initialisation and batches, so the difference is the surgery) and from
  gsurg's (the same loop, so the difference is zret's forward).
- It runs as soon as 0.16 of the card is free (cap 0.14). The fits wait on it.

**Caps and order.**
- gsurg's caps: 0.26 (share 0.28) for both fits, reads 0.28 (0.30). The items go at the end of the feeder's list.
- rmatch's two screen reads hold 0.64 of the card until about 06:45.
- If rmatch's re-call is PROMISING, its full run's fits come first in the list (GPU items start in the list's order),
  and zgs's fits start as those free the card, about 07:30. Otherwise zgs's fits start after the smoke, about 07:00.
- gsurg's loop takes an extra forward and backward for each other dataset in a chunk. gsurg's screen fits took about
  1.5 hours each, and its full run's up to about 3.5 hours beside three to five other items on the card. So each zgs
  fit takes about 1.5 to 2 hours, then a read of about 10 minutes. The pair and its re-call land about 09:00 to
  10:00.
- **Moved at about 06:47, before any of its numbers.** The smoke passed at 06:40, and the two fits started at 06:41,
  a minute before rmatch's gate passed. rmatch's full run, first in the list, then waited behind them. The two fits
  were re-queued under `-b` names behind rmatch's fits, with every item after them (the same commands, outputs and
  gates), and the two running fits were cancelled. A fit writes nothing until it ends. No rule changes.
- **Moved again at about 07:05, before any of its numbers.** Round fourteen's smoke and screen go ahead of zgs's two
  fits in the list, behind rmatch's full run (fourteenth round, 'Caps and order'). zgs's fits had not started. No rule
  changes.
- **Moved a third time at about 07:40, before any of its numbers.** Round fifteen's smoke and screen go ahead of
  zgs's L-hotpotqa fit (`scr-train-zgs-hp-b`) in the list (fifteenth round, 'Caps and order'). That fit had not
  started. zgs's L-musique fit is running and keeps its place. No rule changes.

### Fourteenth round (declared 8 October at about 07:05, before any of its numbers)

Code: `outputs/mp_unified/zrm.py` (its selftest passes). One arm, rmatch's match on zret's base, decided against zret's
fits. The screen trains two fits, L-musique and L-hotpotqa.

**Why this arm.** rmatch's screen and its re-call are PROMISING against step 1's fits (06:40), with the largest gains on
the typed graphs of any arm so far, and its full run is on the card. zret is ADOPTED, and it is the base of every later
screen and run unless zgs is ADOPTED (thirteenth round). The two change different parts of the score: zret how each
block is normalised, rmatch a learned match along the graph's typed relation chains. So the question is whether the
match adds to zret's base as it adds to step 1's.
- **zret's fits** are the thirteenth round's: its screen fit on L-musique, its full run's fits on every other split.
- **rmatch's fits** are reported beside them: its screen fits (`scr-rmatch`, `scr-rmatch-hp`) and its full run's
  (`outputs/full_rmatch/fits/<split>`).
- **rmatch's own grade does not gate this round.** zrm's full run faces the same test against zret's fits, over every
  split, that rmatch's faces against step 1's.

| screen | arm | idea |
| --- | --- | --- |
| scr-zrm, scr-zrm-hp | zrm | **rmatch's match added to zret's model.** zret's forward (every block's within-pool z-score, and rrf's base z-score, taken against the pool's retrieved rows), plus rmatch's learned match of the question to each hop of the typed relation chains that reach a row from the question's seeds. The match's gates start at zero, so the arm starts as zret's model bit for bit. Both parts are used unchanged, on rmatch's chains as built for the twelfth round: no new column, block or hyperparameter. On the untyped graphs (squad, musique, hotpotqa, 2wiki) it is zret's model. |

**How it is decided.**
- Each fit is compared with zret's fit of its split, which decides. rmatch's fit and step 1's are reported beside it.
  The reads are the six s1eval carves at p@swa.
- The pair and its re-call under the seed null are relz.py's, run under zrm's name, with zret's R@5 as each read's
  base and the null's floors, as in the thirteenth round.
- **If the re-call is PROMISING, the full run starts by itself** (docs/FULL_ROUND14.md, declared with this round). Its
  four fits are graded against zret's fits and re-graded under the null over every split.
  - ADOPT makes zrm the base of every later screen and run.
  - If zgs is ADOPTED too, each re-grade is filed and the next base is declared in a later round, before its numbers.
  - On any other result zret stays the base (or zgs, if it is ADOPTED), and the match is not carried.

**What the screen can show.** rmatch's GAINs were on the typed graphs (metaqa in-domain, webqsp read zero-shot) and on
musique read zero-shot in L-musique's fit. zret's model has no relation match, so a GAIN on the typed graphs with no
LOSS means the match adds on zret's base. musique read zero-shot is the read zret already moves (+0.1202 against step
1's fit). If zrm moves it little against zret's fit, rmatch's move there was work zret already does.

**The smoke** (`scr14-smoke`, `zrm.py smoke`):
- The carve check on metaqa select, webqsp s1eval and 2wiki select: on the first questions, zrm at its start scores as
  zret's model bit for bit, with finite chain features on the typed graphs.
- Then zret once, rmatch once and zrm twice (the repeat must be IDENTICAL), one epoch each on metaqa's and 2wiki's
  select carves. Each is read on both.
- zrm's match must move from zero and stay finite, and zret's fit must hold none.
- zrm's scores must differ from zret's on metaqa (the same initialisation and batches, so the difference is the match)
  and from rmatch's on 2wiki (the same match, so the difference is zret's forward).
- It runs as soon as 0.16 of the card is free (cap 0.14). The fits wait on it.

**Speed.** The chains are a query-local compile, as for rmatch (8216ffe; twelfth round, 'Speed is timed cold'). Any
latency figure for zrm is cold.

**Caps and order.**
- rmatch's caps: 0.26 (share 0.28) for both fits, reads 0.30 (0.32).
- **The items go ahead of round thirteen's two fits in the feeder's list, behind rmatch's full run.** rmatch's screen
  fits took about 15 minutes each, and zgs's take about 1.5 to 2 hours. Going first, the short screen returns its
  verdict about two hours sooner, and it holds zgs's fits back by at most its own length. No rule changes.
- **ETAs:** rmatch's full run holds the card until about 07:30 to 08:00, and its reads come first. Then the smoke, and
  the two fits and their reads take about 30 to 45 minutes. The pair and its re-call land about 08:15 to 09:00.

### Fifteenth round (declared 8 October at about 07:40, before any of its numbers)

Code: `outputs/mp_unified/zrs.py` (its selftest passes). One arm, rmatch's match trained on zret's finished fit,
decided against zret's fits. The screen trains two fits, L-musique and L-hotpotqa.

**Why this arm.** rmatch's full run is NOT_ADOPTED under the null over every split (07:32; 'rmatch's full run' in
the results below). In every fit that trains metaqa it GAINs on the two typed graphs: metaqa in-domain (+0.125 to
+0.135) and webqsp read zero-shot (+0.138 to +0.163). Its one LOSS is 2wiki read zero-shot in L-2wiki's fit (−0.0136,
floor 0.0112).
- 2wiki has no typed relations, so the match adds nothing to its scores. The change there comes from training the
  match and the model together: the model's own weights move while the match learns. The same joint training moved
  two other untyped reads up (squad read zero-shot in L-squad's fit, +0.0096; musique read zero-shot in L-musique's,
  +0.2324).
- zrm (fourteenth round) trains the two together too, on zret's base.
- zrs trains the match alone, on top of zret's finished fit. The question is whether the match keeps its typed gains
  when the model does not move. zrs gives up every move of the untyped reads, the gains with the LOSS.

| screen | arm | idea |
| --- | --- | --- |
| scr-zrs, scr-zrs-hp | zrs | **rmatch's match trained on zret's frozen fit.** zret's fit of the split (its p@swa state) is loaded and frozen, and scores as it reads (no dropout). Only the match's parameters train: on the fit's typed training carves (metaqa's; webqsp never trains), against the frozen model's scores, with the fit's own config, seed, loss and batches of 32 questions. The model is zrm's (rmatch's match over zret's forward), on rmatch's chains as built for the twelfth round: no new column, block or hyperparameter. |

**What this arm can and cannot change.**
- Every state of a zrs fit holds zret's p@swa state bit for bit. Training checks this, and refuses a base trained as
  another arm or with other carves, basis, blocks, config, seed or hidden size.
- So on the four passage graphs (squad, musique, hotpotqa, 2wiki) its scores are zret's fit's bit for bit. Its reads
  there are WITHIN by construction, not by evidence, and the results will say so.
- **This round tests the match on the typed graphs only:** metaqa in-domain and webqsp read zero-shot.
- L-metaqa's fit trains no typed graph (metaqa is held out and webqsp never trains). There the match takes no step and
  its gates stay at zero, so that fit is zret's bit for bit, metaqa's held-out read included.

**How it is decided.**
- Each fit is compared with zret's fit of its split, which decides. rmatch's fit and step 1's are reported beside it.
  The reads are the six s1eval carves at p@swa.
- zrm's screen is compared with the same zret fits on the same questions, so the two arms' differences from zret can
  be set side by side as they are. That is reported, not decided.
- The pair and its re-call under the seed null are relz.py's, run under zrs's name, with zret's R@5 as each read's
  base and the null's floors, as in the thirteenth and fourteenth rounds.
- **If the re-call is PROMISING, the full run starts by itself** (docs/FULL_ROUND15.md, declared with this round). Its
  four fits are graded against zret's fits and re-graded under the null over every split.
  - ADOPT makes zrs (zret's fits with the match trained on them) the base of every later screen and run.
  - If zgs or zrm is ADOPTED too, each re-grade is filed and the next base is declared in a later round, before its
    numbers.
  - On any other result zret stays the base (or the arm adopted in round thirteen or fourteen), and the match is not
    carried.

**What the screen can show.** If zrs keeps most of rmatch's typed gains, they do not need the model to move, and the
2wiki LOSS was the price of training the two together. If it keeps little, the match's gains came with the model's
own change, and so did the LOSS.

**The smoke** (`scr15-smoke`, `zrs.py smoke`):
- The carve check on metaqa select, webqsp s1eval and 2wiki select, under train's and read's flags: zret's model
  repeats bit for bit, and zrs at its start (zero gates) scores as zret's model bit for bit, with finite chain
  features on the typed graphs.
- Then zret once, and zrs twice on that zret fit (the repeat must be IDENTICAL), one epoch each on metaqa's and
  2wiki's select carves. Each is read on both.
- Every state of zrs's fit must hold that zret fit's p@swa state bit for bit. zrs's match must move from zero and stay
  finite, and zret's fit must hold none.
- zrs's scores must equal zret's on 2wiki (no typed relations) bit for bit, and differ on metaqa.
- It runs as soon as 0.16 of the card is free (cap 0.14). The fits wait on it.

**Speed.** zrs serves as zrm does: zret's forward plus the match on the chains, which are a query-local compile
(8216ffe; twelfth round, 'Speed is timed cold'). Any latency figure for zrs is cold.

**Caps and order.**
- rmatch's caps: 0.26 (share 0.28) for both fits, reads 0.30 (0.32). A fit loads its training carves as zret's did,
  though only metaqa's questions train.
- **The items go after round fourteen's and ahead of zgs's L-hotpotqa fit** (`scr-train-zgs-hp-b`) in the feeder's
  list. A zrs fit makes one pass over metaqa's fit questions an epoch, with no backward pass through the model, so it
  takes minutes; zgs's fit takes about 1.5 to 2 hours. The short screen holds zgs's fit back by at most its own
  length. zgs's L-musique fit is running and keeps its place. No rule changes.
- **ETAs:** zrm's two fits hold the card until about 07:45 to 08:00, and their reads come first. Then the smoke, and
  the two fits and their reads take about 30 to 45 minutes. The pair and its re-call land about 08:30 to 09:30.

### Sixteenth round (declared 8 October at about 08:00, before any of its numbers)

Code: `outputs/mp_unified/zsep.py` (its selftest passes). One arm, an objective on zret's base, decided against zret's
fits. The screen trains two fits, L-musique and L-hotpotqa.

**Why this arm.** Every arm so far changed the inputs, the model or the training loop. None changed what a question's
loss asks of its golds. A multi-hop question (musique, hotpotqa and 2wiki ask for two to four supporting passages) and
a KB question with several answers has more than one gold in its pool.
- lean_gpu's loss (listwiseD, the loss of every fit since step 1) scores a question as the mean over its golds of
  minus the log of each gold's softmax share over every row of the pool. The question's other golds sit in each gold's
  denominator, so the golds compete with each other.
- Once one gold leads, its own term pushes it down: its gradient is its share minus one over the number of golds.
- The non-gold rows between the leader and a trailing gold get almost no push, since their share is small next to the
  leader's. That is the question R@5 and FC@5 count against on the multi-hop datasets: its first gold is found, and
  its second is not.
- zsep sets each gold against the question's non-gold rows only. No gold is pushed down. Each gold's term pushes the
  non-gold rows down, the highest first, and hardest for the golds that trail them.

| screen | arm | idea |
| --- | --- | --- |
| scr-zsep, scr-zsep-hp | zsep | **Each gold against the non-gold rows only (an objective).** zret's model, carves, batches, config and seed, trained with a question's loss as the mean over its golds of −log(e^{s_g} / (e^{s_g} + Σ e^{s_n})), the sum over the pool's non-gold rows, in place of the mean over its golds of −log softmax over every row. A question with one gold trains as before (the same loss and gradient, up to rounding), and a pool whose rows are all gold adds nothing. The objective acts in training only, so reads and serving are zret's model. No new column, block or hyperparameter. |

**What this arm can and cannot change.**
- Only the terms of questions with two or more golds in the pool change. Each fit records, per training carve, how
  many questions have two or more (screen.json, `zsep`, `census`). Where a training dataset has none, its own terms
  are zret's.
- The model's weights are shared, so the changed terms move every read, single-gold and zero-shot reads included. All
  36 reads are decided as usual.
- The objective changes training only. zsep's reads and its serving cost are zret's model.

**How it is decided.**
- Each fit is compared with zret's fit of its split, which decides. Step 1's fit is reported beside it. The reads are
  the six s1eval carves at p@swa. FC@5's difference (every gold in the top 5) is in each comparison's table, and does
  not decide.
- The pair and its re-call under the seed null are relz.py's, run under zsep's name, with zret's R@5 as each read's
  base and the null's floors, as in rounds thirteen to fifteen.
- **If the re-call is PROMISING, the full run starts by itself** (docs/FULL_ROUND16.md, declared with this round). Its
  four fits are graded against zret's fits and re-graded under the null over every split.
  - ADOPT makes zsep the base of every later screen and run.
  - If an arm of rounds thirteen to fifteen is ADOPTED too, each re-grade is filed and the next base is declared in a
    later round, before its numbers.
  - On any other result zret stays the base (or the arm adopted in rounds thirteen to fifteen), and the objective is
    not carried.

**What the screen can show.** A GAIN on the multi-hop reads (musique, hotpotqa, 2wiki) with no LOSS means the golds'
competition held zret back there. If FC@5 moves with it, the gain is in questions whose last gold was missing.
NO_GAIN means the competition costs nothing at R@5.

**The smoke** (`scr16-smoke`, `zsep.py smoke`):
- The loss on real carves, 2wiki's and hotpotqa's select. Each carve's census is recorded. On its first 64 questions,
  under zret's model at its start, the loss and its gradient must match a float64 reference. Where a question has one
  gold, they must equal listwiseD's, up to rounding.
- Then zret once and zsep twice (the repeat must be IDENTICAL), one epoch each on both carves. Each is read on 2wiki's.
- zsep must hold zret's model (its state's keys). Its census must be the check's, with questions of two or more golds.
- Its scores must differ from zret's: the same initialisation and batches, so the difference is the objective.
- It runs as soon as 0.16 of the card is free (cap 0.14). The fits wait on it.

**Speed.** The objective changes training only. zsep reads and serves as zret's model does, so any latency figure for
it is zret's, and cold (8 October): each question from scratch, with no warm-up pass and nothing kept from an earlier
question.

**Caps and order.**
- zret's caps: 0.26 (share 0.28) for both fits, reads 0.28 (0.30).
- **The items go at the end of the feeder's list, after zgs's L-hotpotqa fit** (`scr-train-zgs-hp-b`). GPU items
  start in the list's order, so zsep's smoke and fits start once that fit has started, as the card has room beside
  it. zgs's fits keep their places. No rule changes.
- **ETAs:** zrm's and zrs's screens hold the card until about 08:30 to 09:00, then zgs's L-hotpotqa fit starts. The
  smoke takes about 10 minutes. The two fits take about 15 to 25 minutes each, and their reads about 10. The pair and
  its re-call land about 09:30 to 10:30.

### Seventeenth round (declared 8 October at about 09:10, before any of its numbers)

Code: `outputs/mp_unified/zrc.py` (its selftest passes). One arm, a preprocessing change to rmatch's chain entries,
decided against zrm's fits. Nothing is trained: zrm's two screen fits, and its four full-run fits if the screen passes,
are read again with the new entries.

**Why this arm.**
- rmatch's build keeps, per row and seed bucket, the 64 chain entries with the most walk mass. Walk mass is set by the
  graph's degrees, not by the question.
- The diagnosis filed below ('what rmatch's chain caps cost a chain match'): on webqsp's s1eval carve those entries
  keep 0.132 of the starting match's sum on gold rows. Ranking rows by the match at its start alone, R@5 is 0.2109
  with the build's entries, 0.2138 with every entry, and 0.2493 with the 64 entries per row and bucket that add most
  to the match.
- On metaqa's s1eval carve no cap binds, and the three are equal (0.2938).
- zrm's largest gain is on webqsp read zero-shot (R@5 0.177 to 0.313 in L-musique's fit). Better entries may lift it
  further, with no training.

| screen | arm | idea |
| --- | --- | --- |
| scr-zrc, scr-zrc-hp | zrc | **Chain entries kept by what they add to the match (preprocessing).** rmatch's walk, chain cap (20,000 chains), renormalised mass and arrays, with one change: per row and seed bucket, the 64 entries kept are those with the largest w0(c)·m_c(v) in place of the largest m_c(v). w0 is the match at its start, (1/3)∏σ(zcos_q(r_k)) over the chain's relations (rmatch.ChainMatch before training), from the frozen encoder's question and relation embeddings. Ties go to the earlier chain. zrm's model and weights, unchanged. No new column, block or hyperparameter. |

**Why nothing is trained.**
- webqsp never trains. metaqa is the only typed graph a fit trains on (every split but L-metaqa), on its fit carve.
- If that carve never fills a row's 64 entries, zrc's build of it is rmatch's array for array. Then zrc's training is
  zrm's bit for bit (the same arrays, initialisation, batches and seed), and zrm's fits are zrc's.
- **The identity gate** (`zrc-identity`: `zrc.py identity --carves metaqa=fit`) checks every part of that carve, array
  for array. Only if it is IDENTICAL are zrm's fits copied (`zrc.py fork`: models and training records unchanged,
  under arm zrc, in new folders) and read on the six s1eval carves with zrc's entries. zrm's folders are not written.
- If the gate is not IDENTICAL, nothing is forked or read, and the round stops. A round that trains would be declared
  in its own file.

**What this arm can and cannot change.**
- Only the reads of the typed graphs can change: webqsp's (read zero-shot in every split), and metaqa's if its s1eval
  carve fills a row's 64 entries (the diagnosis says it does not).
- The reads of squad, musique, hotpotqa and 2wiki must be zrm's bit for bit. Each comparison files that check beside
  it (`<out>-same.md`: every array of every read against zrm's). A difference there is a bug, and the round stops.
- L-metaqa's fit trains no typed graph, so its match's gates stay at zero, and its reads must be zrm's too.

**How it is decided.**
- Each read is compared with zrm's fit of its split, which decides. zret's fit and step 1's are reported beside it.
- The pair and its re-call under the seed null are relz.py's, run under zrc's name. Each read's base R@5 is zrm's
  (zrm's fits compared with step 1's, `outputs/zrc/base-zrm-<split>.json`), with the null's floors.
- **If the re-call is PROMISING, the full run starts by itself** (docs/FULL_ROUND17.md, declared with this round).
  zrm's four full-run fits are forked and read, graded against zrm's fits, and re-graded under the null over every
  split.
- **Adoption is relative to zrm.**
  - If zrm's own re-grade (docs/FULL_ROUND14.md) is ADOPT, zrc ADOPT makes zrc the base of every later screen and run.
  - If zrm is NOT_ADOPTED, zrc's grade decides nothing about the base. A grade of zrc against zret's fits would be
    declared in a later round, before its numbers.
  - If another arm of rounds thirteen to sixteen is ADOPTED too, each re-grade is filed and the next base is declared
    in a later round, before its numbers.

**What the screen can show.**
- A GAIN on webqsp read zero-shot means the build's mass rule held zrm's match back on a graph it never trained on.
  The diagnosis puts that at about 0.035 to 0.04 of R@5 for the match at its start, before its trained weights.
- NO_GAIN means the trained match already makes up for it.
- A LOSS on webqsp means the trained match leans on the entries kept by mass.

**Speed.** The rule changes which entries a row keeps, at build time. It takes one cosine per pool relation and a
product per chain, inside the walk the build already makes. Any latency figure for this arm is cold (8216ffe): each
question from scratch, the walk, the selection and the forward included, with no warm-up pass and nothing kept from
an earlier question.

**Caps and order.**
- The builds, the identity gate and the comparisons are CPU only: 1 CPU and 2 GB per build (the diagnosis's walk
  peaked under 0.5 GB). The forks copy two files (1 CPU, 1 GB).
- Reads: 0.30 of the card (share 0.32) and 6 GB, as zrm's reads (measured peak 3.6 GB).
- **The items go after round fifteen's (zrs's full run), ahead of zgs's re-queued screen** (`scr-train-zgs-b`'s
  block): reads of minutes ahead of a fit of about 2 hours. zrm's full run and zrs's keep their places. No rule
  changes.
- **ETAs:** the builds and the gate take about 5 minutes. The screen's two reads wait for zrm's full-run reads and
  zrs's fits on the card, then take about 3 minutes each. The pair and its re-call land about 09:45 to 10:30. If it
  passes, each full-run fork waits for its zrm fit and its comparison for zrm's read. The grade lands about 10:30 to
  11:30.

## 5. ETAs (7 October)

- **The four arm fits** started 14:25 to 14:27. An epoch takes 154 to 168 s, so each fit finishes about 14:50 to 14:55.
- **The null check and the pools fit** start as the card frees, about 14:52. The null takes about 5 minutes; pools
  about 25.
- **Reads** take about 8 minutes each.
- **Compares:** about 15:05 to 15:15 for the four arms, about 15:30 for pools.
- **The ablate** (CPU): about 14:50 to 15:00.
- **Second round** (queued about 15:25): the smoke takes about 10 minutes. The two fits take about 25 to 35 minutes
  each (padded pools add rows to hotpotqa's and 2wiki's batches), then reads and compares. Verdicts about 16:15 to
  16:30. (They landed at 15:46 and 15:48.)
- **Third round** (queued about 16:00): the smoke takes about 5 minutes, the fit about 25 (8 epochs at about 3
  minutes), the read about 2. Verdict about 16:35 to 16:45. (It landed at 16:46: the fit took 43 minutes.)
- **Fourth round** (queued about 20:00, beside the seed-0 GNN chain, which holds half the card until about 22:30):
  the smoke takes about 10 minutes, the fit about 45 to 55 (zret's took 43 minutes; ztop50 adds a sort per batch),
  then the read and comparison take about 10. Verdict about 21:00 to 21:15.
  - Amended about 20:35: the L-musique fit was at epoch 4 of 8 at 20:26 (about 2 minutes an epoch), so its read and
    comparison land about 20:50. scr-ztop50-hp starts when that fit frees the card (it needs 0.28; 0.24 was free at
    20:26), about 20:45 to 21:00. It takes about 50 to 60 minutes, then about 15 more for its read and comparison.
    The verdict over both fits comes about 22:00 to 22:15.
- **Fifth round** (queued about 21:15, beside the seed-0 GNN chain until about 22:30):
  - The smoke takes about 10 minutes; it fits beside the chain and scr-ztop50-hp.
  - The four fits cap 0.26 of the card each (share 0.28). Until about 22:30, one runs beside the chain; after that,
    three at a time. gsurg's fits start first: they take longest. Each step adds a forward and backward over each
    other training dataset's batch of the same chunk (three more on these splits), while the batches are built once,
    so about 50 to 80 minutes each. prank's take about 25 to 35.
  - Verdicts: prank about 23:15 to 23:30, gsurg about 23:40 to 00:15.
- **Sixth round** (queued about 21:30):
  - The builds take a few minutes each on the host's idle CPUs (2wiki select, 1,496 questions, took about 1 second on
    the laptop).
  - The smoke takes about 10 minutes, once the card has room.
  - hubwalk's two fits queue behind round five's (cap 0.26, share 0.28). They start as those finish, about 22:50 to
    23:10, and take about 25 to 35 minutes each: the arm adds no work to a step.
  - Verdict about 23:45 to 00:30.
- **Amended about 21:40,** from the epochs logged so far:
  - The seed-0 GNN chain had freed the card by 21:22, so three of round five's fits started together at 21:22.
  - With three fits sharing the card, prank's epochs take about 387 s and gsurg's about 490 to 500 s. Over 8 epochs,
    scr-prank ends about 22:14 and both gsurg fits about 22:27 to 22:30.
  - gsurg's verdict comes about 22:40 to 22:50 and prank's about 23:15 to 23:25, since scr-prank-hp starts only when
    scr-prank frees the card.
  - hubwalk's fits start about 22:35 to 22:45, after the round-five reads get the card. Its verdict comes about 23:50
    to 00:10.
- **Seventh round** (queued about 22:40):
  - The smoke takes about 10 minutes once the card has room.
  - The two fits (cap 0.26, share 0.28) queue behind round six's. The arm adds a weighted sum per z-score and nothing
    else to a step, so each takes about as long as step 1's fit: 25 to 35 minutes alone, about 50 with three fits on
    the card. They start as prank's and hubwalk's fits free the card, about 23:10 to 23:45.
  - Verdict about 00:30 to 01:15.
- **Eighth round** (queued about 23:30):
  - The smoke takes about 5 to 10 minutes on 0.12 of the card.
  - The two fits (cap 0.26, share 0.28) queue behind round seven's fits and prank-hp's read. pret ranks 6
    columns where prank ranked 86, so a step costs about step 1's: 25 to 35 minutes alone, about 50 with three
    fits on the card. They start as prank-hp and hubwalk free the card, about 23:45 to 00:30.
  - Verdict about 01:00 to 01:45.
  - **Amended about 23:40:** the smoke passed at 23:23 (repeat IDENTICAL, 12 s) and scr-train-pret started at 23:35,
    so the verdict comes nearer 00:20 to 00:45. vshare's L-musique fit is PROMISING, so if its pair is too, its four
    full-run fits would be ready before pret's reads, and the feeder runs GPU items in file order. Round eight's
    screen items were therefore moved ahead of vshare's full run in the feeder's file. This changes when its reads
    (about 3 minutes each) run, not anything they compute.

- **Ninth round** (queued about 01:05 on 8 October):
  - The thirteen CPU builds read the look chunks once each, a few minutes per carve on the host's idle cores.
  - The smoke takes about 5 to 10 minutes once its three carves are built.
  - The four fits queue ahead of every gated full run, behind the seed null's reads. rel caps 0.30 of the card (share
    0.32): its 26 columns add about 0.7 GB on L-musique and 1.2 GB on L-hotpotqa to the base fit's measured 5.2 GB.
    gcs keeps the earlier cap, 0.26 (share 0.28). Each takes about 25 to 35 minutes alone, about 50 with three on the
    card. They start as the null's reads free the card, about 01:20 to 01:40.
  - Verdicts and re-calls about 02:30 to 03:15.

## Results

Development numbers; the paper's numbers come from one declared confirmation run.

### Step 2, eval-mix weights: NO_GAIN on all three splits. Dropped.

Read at 14:24 (`outputs/screen/s2-J5.md`, `s2-L-metaqa.md`, `s2-L-squad.md`). Each cell is R@5 of step 2's p@swa minus
step 1's p@swa on the same split, with the call. rrf is in brackets.

| dataset | J5 | L-metaqa | L-squad |
| --- | --- | --- | --- |
| metaqa | +0.0019 WITHIN | **−0.0086 LOSS** (zero-shot; rrf 0.005) | −0.0025 WITHIN |
| squad | −0.0016 WITHIN | +0.0014 WITHIN | +0.0059 WITHIN (zero-shot; rrf 0.905) |
| musique | −0.0050 WITHIN | −0.0065 WITHIN | **−0.0166 LOSS** |
| hotpotqa | −0.0014 WITHIN | −0.0071 WITHIN | **−0.0086 LOSS** |
| 2wiki | +0.0001 WITHIN | +0.0047 WITHIN | −0.0021 WITHIN |
| webqsp (zero-shot; rrf 0.054) | **−0.0099 LOSS** | **−0.0092 LOSS** | −0.0078 WITHIN |

- No read gains. Five of the eighteen lose: webqsp twice, zero-shot metaqa once, and musique and hotpotqa in-domain
  once each.
- Squad read zero-shot (L-squad): rrf alone (0.905) is above both fits (step 1's 0.880, step 2's 0.886). The learned
  scorer costs squad, a control whose graph exposes no extra gold, about 0.02 to 0.03 R@5 when it has not trained on
  it. That is the zero-shot harm these screens target.

### The null check: IDENTICAL (14:52)

`outputs/screen/null.json`: the base arm trained one epoch through lean_screen.py gives step 1's L-musique p@ep0 bit
for bit, all nine tensors. A screen's difference from step 1's fit is its arm's alone.

### scr-zonly: NO_GAIN. Dropped.

`outputs/screen/scr-zonly.md`. R@5 of zonly minus step 1's L-musique p@swa:

| metaqa | squad | musique (zero-shot) | hotpotqa | 2wiki | webqsp (zero-shot) |
| --- | --- | --- | --- | --- | --- |
| −0.1072 LOSS | −0.0035 WITHIN | −0.1297 LOSS | −0.0115 LOSS | −0.0374 LOSS | −0.0014 WITHIN |

The raw values carry real information in-domain, and dropping them does not help musique either (0.140 against
0.270 and rrf's 0.473). The z-scores alone carry the big-pool shift too: in a big pool, the few retrieved nodes sit
far out in each rank column's tail.

### scr-dnorm: NO_GAIN. Dropped. scr-bound1: MIXED. Step 1's variants pf, n, nf: NO_GAIN (14:56)

`outputs/screen/scr-dnorm.md`, `scr-bound1.md`, `variants-L-musique.md`. R@5 minus step 1's L-musique p@swa:

| screen | metaqa | squad | musique (zero-shot) | hotpotqa | 2wiki | webqsp (zero-shot) | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dnorm | +0.0004 | +0.0000 | **−0.1330** | +0.0025 | +0.0005 | −0.0002 | NO_GAIN |
| bound1 | **−0.6226** | −0.0006 | **+0.1420** | **−0.0190** | **−0.0420** | **−0.1000** | MIXED |
| pf (FiLM context) | **−0.0080** | +0.0028 | **−0.0293** | +0.0034 | −0.0040 | **−0.0217** | NO_GAIN |
| n (no SEMB) | **−0.1199** | −0.0012 | **−0.0229** | **−0.0076** | **−0.0462** | +0.0031 | NO_GAIN |
| nf (no SEMB, FiLM) | **−0.1108** | +0.0023 | **−0.0992** | −0.0011 | **−0.0430** | **−0.0535** | NO_GAIN |

Bold: a GAIN or LOSS call.

- **bound1 shows the trade.** Kept within one pool sd of rrf, the model recovers most of musique (0.412 against
  0.270; rrf 0.473), and metaqa falls to rrf's level (0.031). metaqa's golds are the nodes rrf ranks lowest, so a
  model that must stay near rrf cannot find them. One model has to stay near rrf on musique and move far from it on
  metaqa, and in this training set nothing but metaqa has big pools.
- **Per-dataset standardisation (dnorm) changes nothing in-domain and makes musique worse**, as zonly did.
- p stays the best of step 1's four variants on this split.

### The diagnosis: block ablation of step 1's L-musique p@swa (14:50)

`outputs/screen/ablate-L-musique/ablate.md`. Pool sizes per question: musique 2,092 rows and webqsp 2,120 (both read
zero-shot), metaqa 2,017, 2wiki 106, hotpotqa 94, squad 50. Every big-pool dataset in training is a KB.

| read | model | rrf | only rank kept (+rank) | only dense_cos kept | only SEED kept |
| --- | --- | --- | --- | --- | --- |
| musique (zero-shot) | 0.270 | 0.473 | 0.154 | 0.479 | 0.023 |
| webqsp (zero-shot) | 0.178 | 0.054 | 0.034 | 0.091 | 0.000 |
| metaqa (in-domain) | 0.654 | 0.005 | 0.004 | 0.008 | 0.005 |
| hotpotqa (in-domain) | 0.902 | 0.685 | 0.701 | 0.681 | 0.011 |
| 2wiki (in-domain) | 0.873 | 0.608 | 0.631 | 0.569 | 0.018 |
| squad (in-domain) | 0.910 | 0.905 | 0.912 | 0.892 | 0.019 |

- **No single block makes musique's zero-shot loss.** Removing any one block's order (−B) leaves musique no better
  than the full model (rank −0.086, dense_cos −0.059, WALK −0.042, WALKF −0.033, DISTS −0.018; the other four within
  ±0.006). The loss is in how the blocks are combined.
- **The rank block reads differently on big pools.** With every other block flattened, the model's use of the
  retrieval ranks lifts rrf on the three small-pool passage datasets (+0.006 to +0.023), but on musique it costs
  0.32 (0.154 against rrf's 0.473). On metaqa it is flat. musique's big pools make the model act as it does on
  metaqa, whose golds rrf does not find (rrf R@5 0.005).
- **What follows** (corrected about 15:20). An earlier version of this bullet said the raw values carry the pool's
  size. They do not. The rank block's raw values are reciprocal ranks in each retriever's top-1000 list over the
  whole corpus, the same scale in every pool. What differs with pool size is how many of a pool's rows retrieval
  ranked at all: 10 to 18% in the big pools against 56 to 100% in the small ones (the composition below). That share
  sets every pool z-score, and in training the only pools with a low share are a KB's. zonly and dnorm changed the
  raw values and found nothing there. On webqsp, the KB read zero-shot, the structure blocks carry the model's lead
  over rrf (only DISTS kept: +0.099; only WALK kept: +0.070).

### scr-bdrop20: PROMISING (14:57). Its full run: docs/FULL_BDROP20.md

`outputs/screen/scr-bdrop20.md`. R@5 of bdrop20 minus step 1's L-musique p@swa, with the 95% interval:

| metaqa | squad | musique (zero-shot) | hotpotqa | 2wiki | webqsp (zero-shot) |
| --- | --- | --- | --- | --- | --- |
| −0.0047 WITHIN | −0.0007 WITHIN | **+0.1344 GAIN** [+0.124, +0.145] | −0.0048 WITHIN | −0.0050 WITHIN | −0.0037 WITHIN |

- musique, read zero-shot, rises from 0.270 to 0.404 (hit@1 +0.171, FC@5 +0.067). rrf alone is 0.473, so the model
  is still below plain retrieval there.
- **Its full run is NOT_ADOPTED (15:45).** On L-metaqa, webqsp read zero-shot loses 0.0081 and metaqa read zero-shot
  does not move (docs/FULL_BDROP20.md).
- metaqa, hotpotqa and 2wiki each fall about 0.005, with intervals below 0. That is under the 0.0075 floor (one-seed
  training noise), so the calls are WITHIN. The full run shows whether it repeats.
- Unlike bound1, nothing collapses: metaqa keeps 0.649.

### scr-pools: MIXED (15:12). Step 4c's six fits do not run

`outputs/screen/scr-pools.md`. One fit of step 1's L-musique on step 4c's P_F pools, read on P_F:

| metaqa | squad | musique (zero-shot) | hotpotqa | 2wiki | webqsp (zero-shot) |
| --- | --- | --- | --- | --- | --- |
| +0.0027 WITHIN | +0.0007 WITHIN | **+0.0203 GAIN** | +0.0020 WITHIN | −0.0005 WITHIN | **−0.0200 LOSS** |

Against step 4c's early read (step 1's unchanged fit read on P_F; reported only), retraining on P_F gains on metaqa
(+0.0108) and is within on the rest. Read on P_F pools, step 1's unchanged fit already lost 0.012 on webqsp.

### Pool composition (a diagnosis, 14:57)

`outputs/mp_unified/pool_composition.py`, `outputs/screen/pool_composition.md`. Per question, averaged:

| carve | rows (median) | ranked rows (median) | ranked share | golds in pool | golds ranked |
| --- | ---: | ---: | ---: | ---: | ---: |
| metaqa fit | 2,017 (2,037) | 205 (206) | 0.102 | 6.36 | 0.427 |
| squad fit | 50 (50) | 50 (50) | 1.000 | 0.98 | 1.000 |
| hotpotqa fit | 94 (90) | 63 (60) | 0.693 | 1.96 | 0.946 |
| 2wiki fit | 106 (104) | 57 (55) | 0.557 | 2.29 | 0.802 |
| musique s1fit | 2,092 (2,134) | 369 (357) | 0.178 | 2.13 | 0.959 |
| musique s1eval | 2,092 (2,132) | 379 (365) | 0.182 | 2.40 | 0.931 |
| webqsp s1eval | 2,120 (2,124) | 322 (308) | 0.152 | 3.32 | 0.581 |

The s1eval carves of metaqa, squad, hotpotqa and 2wiki match their fit carves to the second decimal. A ranked row
is one with rrf > 0: a node in the dense or the splade top-1000 list.

- In L-musique's training set, the only pools in which retrieval ranks few rows are metaqa's, and retrieval misses
  57% of metaqa's golds.
- musique's pools are as big and as thinly ranked, but retrieval ranks 93 to 96% of musique's golds.
- A model that learns "few ranked rows, so look past the ranked ones" from metaqa applies it to musique. That is
  the zero-shot loss the ablation located in the rank block's use on big pools. scr-pad tests it.

### scr-pad: MIXED (15:46). scr-padbd20: PROMISING, but no full run (15:48)

`outputs/screen/scr-pad.md`, `scr-padbd20.md`. R@5 minus step 1's L-musique p@swa (the last row against scr-bdrop20,
reported only):

| screen | metaqa | squad | musique (zero-shot) | hotpotqa | 2wiki | webqsp (zero-shot) | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pad | −0.0036 | −0.0005 | **+0.0831** | **−0.0086** | −0.0058 | +0.0001 | MIXED |
| padbd20 | −0.0053 | +0.0010 | **+0.1095** | −0.0055 | −0.0069 | −0.0111 | PROMISING |
| padbd20 against scr-bdrop20 | −0.0007 | +0.0017 | **−0.0248** | −0.0007 | −0.0018 | −0.0074 | NO_GAIN |

Bold: a GAIN or LOSS call.

- **Padding alone lifts musique from 0.270 to 0.353** (hit@1 +0.104). The share of ranked rows is part of what the
  zero-shot read misuses, as the pool composition suggested.
- **It costs hotpotqa 0.0086 in-domain.** Half of hotpotqa's training pools were padded, and the reads are not.
- **Padding adds nothing to block dropout.** Together they are 0.025 below block dropout alone on musique.
- **padbd20 gets no full run.** It is PROMISING by the rule, but its block dropout failed its own full run, and it gains
  nothing against bdrop20's screen. That is section 2's amendment of about 15:55, made after these numbers.
- **pad is not followed with a narrower padding screen.** zret (the third round) tests the same cause without changing
  the training data.

### scr-zret: PROMISING (16:46). Its full run is NOT_ADOPTED (docs/FULL_ZRET.md)

`outputs/screen/scr-zret.md`. R@5 of zret minus step 1's L-musique p@swa, with the 95% interval:

| metaqa | squad | musique (zero-shot) | hotpotqa | 2wiki | webqsp (zero-shot) |
| --- | --- | --- | --- | --- | --- |
| −0.0041 WITHIN | +0.0007 WITHIN | **+0.1202 GAIN** [+0.110, +0.130] | +0.0046 WITHIN | −0.0018 WITHIN | −0.0012 WITHIN |

- **musique, read zero-shot, rises from 0.270 to 0.390** (hit@1 +0.154, FC@5 +0.068). rrf alone is 0.473, so the
  model is still below plain retrieval there.
- **The share of retrieved rows was the channel.** Taking it out of the z-scores recovers most of what padding and
  block dropout recovered, with no change to the training data and no new parameters.
- **Elsewhere it is cleaner than block dropout's screen.** hotpotqa +0.0046 (bdrop20 −0.0048), 2wiki −0.0018
  (−0.0050), webqsp −0.0012 (−0.0037). metaqa −0.0041 (−0.0047) has an interval below 0, but under the 0.0075
  floor, so the call is WITHIN.
- **It trains about twice as slowly:** epochs of 308 to 341 s against about 160 s, from the second set of per-block
  z-scores. The fit took 43 minutes, not the 25 expected.
- **The gate passed at 16:47.** J5, L-metaqa and L-squad started at once; L-hotpotqa and L-2wiki start as the card
  frees.
- **Its full run is NOT_ADOPTED (17:57).** Three of the eleven primary reads GAIN, all read zero-shot: metaqa +0.064,
  musique +0.120 (this fit) and webqsp +0.025. Four of the thirty-six reads LOSE: hotpotqa and 2wiki read zero-shot
  (−0.014, −0.008), metaqa in L-hotpotqa's fit (−0.012) and musique in L-squad's fit (−0.009). zret gains on the
  big, thinly retrieved pools and loses on the small ones (docs/FULL_ZRET.md).

### scr-ztop50 (L-musique): MIXED (20:39). The pair cannot be PROMISING, so ztop50's full run does not start

`outputs/screen/scr-ztop50.md`. R@5 of ztop50 minus step 1's L-musique p@swa, with the 95% interval:

| metaqa | squad | musique (zero-shot) | hotpotqa | 2wiki | webqsp (zero-shot) |
| --- | --- | --- | --- | --- | --- |
| **−0.0176 LOSS** [−0.021, −0.014] | +0.0013 WITHIN | **+0.1387 GAIN** [+0.128, +0.149] | +0.0004 WITHIN | −0.0041 WITHIN | **+0.0175 GAIN** [+0.004, +0.032] |

- **musique, read zero-shot, rises from 0.270 to 0.408** (hit@1 +0.176, FC@5 +0.074), 0.019 above zret's screen fit
  (reported base, a GAIN). rrf alone is 0.473, so the model is still below plain retrieval there. webqsp read
  zero-shot gains too (+0.0175; zret's screen: −0.0012).
- **metaqa in-domain LOSEs (−0.0176; against zret's screen −0.0135).** metaqa is the only dataset with big pools in
  this split's training set. A reference of fixed size takes the pool's size out of its z-scores, and with it what
  the fit had learned from metaqa's pools alone.
- One LOSS among the twelve reads already rules out PROMISING. scr-ztop50-hp still runs to its end and is reported
  here; fzt-gate fails on the pair verdict and the feeder drops the full run's items.

### scr-ztop50-hp (L-hotpotqa): MIXED (21:22). The pair is MIXED, and ztop50's full run was dropped (21:25)

`outputs/screen/scr-ztop50-hp.md`, `outputs/screen/scr-ztop50-pair.md`. R@5 of ztop50 minus step 1's L-hotpotqa
p@swa:

| metaqa | squad | musique | hotpotqa (zero-shot) | 2wiki | webqsp (zero-shot) |
| --- | --- | --- | --- | --- | --- |
| **−0.0252 LOSS** | −0.0003 WITHIN | **+0.0115 GAIN** | **−0.0082 LOSS** | −0.0025 WITHIN | **−0.0172 LOSS** |

- **Over both fits: MIXED**, with GAINs on 3 of the 12 reads and LOSSes on 4. `fzt-gate` failed at 21:25 and the
  feeder dropped the full run's items.
- **Read zero-shot, hotpotqa still loses** (−0.008). zret's full-run L-hotpotqa fit, reported here as a second base,
  lost 0.014. ztop50 recovers 0.006 of that (WITHIN against zret's fit) but still loses against step 1. webqsp, read
  zero-shot, loses 0.017 here, where it gained 0.0175 in the L-musique fit.
- **metaqa in-domain loses in both fits** (−0.018 and −0.025); **musique gains in both roles** (+0.139 zero-shot,
  +0.012 in-domain).
- Every change to how a pool's z-scores are taken (zonly, dnorm, zret, ztop50) moves metaqa and musique in opposite
  directions. No further screen of that kind is planned. Rounds five and six change the inputs and the objective
  instead.

### scr-gsurg (L-musique): PROMISING (23:04). scr-gsurg-hp (L-hotpotqa): MIXED (23:07). The pair is MIXED, so gsurg's full run does not start

`outputs/screen/scr-gsurg.md`, `outputs/screen/scr-gsurg-hp.md`, `outputs/screen/scr-gsurg-pair.md`. R@5 of gsurg minus
step 1's p@swa of the same split, with the 95% interval where a call is made:

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | +0.0012 WITHIN | −0.0011 WITHIN | **+0.0842 GAIN** (zero-shot) [+0.076, +0.092] | +0.0003 WITHIN | +0.0014 WITHIN | −0.0007 WITHIN (zero-shot) |
| L-hotpotqa | +0.0016 WITHIN | −0.0019 WITHIN | +0.0049 WITHIN | **−0.0147 LOSS** (zero-shot) [−0.019, −0.011] | **+0.0091 GAIN** [+0.007, +0.011] | **+0.0124 GAIN** (zero-shot) [+0.002, +0.023] |

- **gsurg is the first arm that lifts musique, read zero-shot, without costing metaqa in-domain.** In the L-musique
  fit, musique rises from 0.270 to 0.354 (hit@1 +0.067, FC@5 +0.038) while metaqa stays at 0.655 (+0.0012). zret,
  ztop50 and pad each traded metaqa for musique. rrf alone is still higher on musique (0.473).
- **hotpotqa, read zero-shot, loses again** (−0.0147; FC@5 −0.028, hit@1 −0.007): mostly the second gold. It was
  also the LOSS read of zret's full run (−0.014) and of ztop50's pair (−0.008). In the same fit, 2wiki gains
  in-domain (+0.0091) and webqsp gains zero-shot (+0.0124).
- **Over both fits: MIXED**, with GAINs on 3 of the 12 reads and a LOSS on 1. `fgs-gate` fails on the pair verdict
  and the feeder drops gsurg's full run. prank's pair (round five's other arm) is still to come.

### scr-hubwalk (L-musique): NO_GAIN (23:10). scr-hubwalk-hp (L-hotpotqa): NO_GAIN (23:31). The pair is NO_GAIN, so hubwalk is dropped

`outputs/screen/scr-hubwalk.md`, `outputs/screen/scr-hubwalk-hp.md`, `outputs/screen/scr-hubwalk-pair.md`. R@5 of
hubwalk minus step 1's p@swa of the same split, with the 95% interval where a call is made:

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | +0.0041 WITHIN | −0.0002 WITHIN | **−0.0288 LOSS** (zero-shot) [−0.037, −0.021] | +0.0009 WITHIN | −0.0009 WITHIN | **−0.0131 LOSS** (zero-shot) [−0.024, −0.003] |
| L-hotpotqa | −0.0073 WITHIN | +0.0003 WITHIN | +0.0035 WITHIN | **−0.0155 LOSS** (zero-shot) [−0.019, −0.012] | −0.0026 WITHIN | −0.0053 WITHIN (zero-shot) |

- **Walk mass in place of path counts gains nothing.** Every read that moves past the floor is a zero-shot LOSS:
  musique and webqsp in the L-musique fit, and hotpotqa in the L-hotpotqa fit (FC@5 −0.031). Step 1's path counts
  transfer better on all three.
- **In-domain reads stay within the floor.** metaqa in the L-hotpotqa fit (−0.0073) has its interval below 0 but sits
  just under the floor.
- **hotpotqa, read zero-shot, is the LOSS read for the fourth time** (zret, ztop50, gsurg, hubwalk).
- **Over both fits: NO_GAIN,** with no GAIN among the 12 reads and a LOSS on 3. `fhw-gate-b` fails, and the feeder drops
  hubwalk's full run.

### scr-vshare (L-musique): PROMISING (23:35). scr-vshare-hp (L-hotpotqa): NO_GAIN (00:00). The pair is MIXED; the seed null's re-call decides

`outputs/screen/scr-vshare.md`, `outputs/screen/scr-vshare-hp.md`, `outputs/screen/scr-vshare-pair.md`. R@5 of
vshare minus step 1's p@swa of the same split, with the 95% interval where a call is made:

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | −0.0030 WITHIN | −0.0004 WITHIN | **+0.0891 GAIN** (zero-shot) [+0.079, +0.099] | −0.0017 WITHIN | −0.0011 WITHIN | +0.0045 WITHIN (zero-shot) |
| L-hotpotqa | −0.0055 WITHIN | +0.0003 WITHIN | +0.0030 WITHIN | **−0.0140 LOSS** (zero-shot) [−0.018, −0.010] | −0.0013 WITHIN | −0.0057 WITHIN (zero-shot) |

- **musique, read zero-shot, gains the most of any arm that keeps metaqa:** 0.270 → 0.359, with hit@1 +0.149 and
  FC@5 +0.043. metaqa in-domain stays within the floor (−0.0030). For comparison, prank lifted musique to 0.471 but lost
  0.031 on metaqa, and gsurg lifted it by 0.084 with metaqa at +0.0012.
- **hotpotqa, read zero-shot, LOSES for the fifth arm in five** (zret −0.014, ztop50 −0.008, gsurg −0.015, hubwalk
  −0.016, vshare −0.014), with FC@5 −0.029. This is the read the seed null was declared for (section 2), before these
  numbers.
- **Over both fits: MIXED,** with 1 GAIN and 1 LOSS among the 12 reads. `fvs-gate` fails, but nothing waits on it any
  more. vshare's full run waits on `fvs-gate-r`, the re-call under the null's floors, about 01:20.
- **What the re-call needs (arithmetic on the declared rule, written before the null's numbers):** the LOSS turns
  WITHIN when the null's spread on L-hotpotqa's hotpotqa read is above 0.0070, a floor above 0.0140. The GAIN holds
  while the spread on L-musique's musique read is at most 0.0445, a floor of at most 0.0891. Every other read is WITHIN
  under any floor. So the re-call is PROMISING exactly when both hold.

### scr-prank (L-musique): MIXED (23:08). scr-prank-hp (L-hotpotqa): NO_GAIN (00:07). The pair is MIXED, so prank's full run does not start

`outputs/screen/scr-prank.md`, `outputs/screen/scr-prank-hp.md`, `outputs/screen/scr-prank-pair.md`. R@5 of prank
minus step 1's p@swa of the same split, with the 95% interval where a call is made:

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | **−0.0310 LOSS** [−0.035, −0.027] | +0.0011 WITHIN | **+0.2016 GAIN** (zero-shot) [+0.190, +0.213] | **−0.0080 LOSS** [−0.011, −0.005] | −0.0068 WITHIN | **+0.0269 GAIN** (zero-shot) [+0.013, +0.041] |
| L-hotpotqa | **−0.0320 LOSS** [−0.036, −0.028] | −0.0035 WITHIN | −0.0067 WITHIN | **−0.0255 LOSS** (zero-shot) [−0.030, −0.021] | −0.0004 WITHIN | +0.0054 WITHIN (zero-shot) |

- **Ranks in place of z-scores close musique's zero-shot gap to plain retrieval:** 0.270 → 0.471 against rrf's
  0.473, with hit@1 +0.252 and FC@5 +0.124. webqsp, read zero-shot, gains 0.027.
- **metaqa in-domain loses in both fits** (−0.031 and −0.032). Ranks cut at 50 drop the structure magnitudes that
  metaqa's unranked golds need. This is the reason for pret (eighth round), which ranks the retrieval blocks only.
  hotpotqa in-domain also loses in the L-musique fit (−0.008).
- **hotpotqa, read zero-shot, LOSES for the sixth arm in six,** and by the most (−0.0255, FC@5 −0.048).
- **Over both fits: MIXED,** with 2 GAINs and 4 LOSSes among the 12 reads. `fpr-gate` fails and the feeder drops the
  full run. Under the re-call (about 01:20), each metaqa in-domain LOSS would need a floor above 0.031, an in-domain
  seed spread above 0.016, before prank could turn PROMISING (written before the null's numbers).

### scr-pret (L-musique): PROMISING (00:19). scr-pret-hp (L-hotpotqa): NO_GAIN (00:45). The pair is MIXED; the seed null's re-call decides

`outputs/screen/scr-pret.md`, `outputs/screen/scr-pret-hp.md`, `outputs/screen/scr-pret-pair.md`. R@5 of pret minus
step 1's p@swa of the same split, with the 95% interval where a call is made:

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | +0.0006 WITHIN | +0.0003 WITHIN | **+0.0219 GAIN** (zero-shot) [+0.010, +0.035] | −0.0028 WITHIN | +0.0035 WITHIN | −0.0090 WITHIN (zero-shot) |
| L-hotpotqa | −0.0064 WITHIN | −0.0013 WITHIN | −0.0038 WITHIN | **−0.0253 LOSS** (zero-shot) [−0.030, −0.021] | +0.0037 WITHIN | −0.0042 WITHIN (zero-shot) |

- **Ranking only the retrieval blocks keeps metaqa** (+0.0006 and −0.0064, both within the floor), where prank lost
  0.031 and 0.032. But it keeps only a ninth of prank's musique gain: 0.270 → 0.292, against prank's 0.471, with
  hit@1 −0.060. So most of prank's zero-shot gain needs the structure ranks too: close to the declaration's second
  case, with a small part through the retrieval columns.
- **hotpotqa, read zero-shot, LOSES for the seventh arm in seven** (−0.0253, FC@5 −0.049): zret, ztop50, gsurg,
  hubwalk, vshare, prank and pret.
- **Over both fits: MIXED,** with 1 GAIN and 1 LOSS among the 12 reads. `fpt-gate` fails; nothing waits on it.
  pret's full run waits on `fpt-gate-r`.
- **What the re-call needs (arithmetic on the declared rule, written before the null's numbers):** the LOSS turns
  WITHIN when the null's spread on L-hotpotqa's hotpotqa read is above 0.01265, a floor above 0.0253. The GAIN holds
  while the spread on L-musique's musique read is at most 0.01095, a floor of at most 0.0219. Every other read is WITHIN
  under any floor. So the re-call is PROMISING exactly when both hold.

### The seed null (00:54): five of the twelve floors are above 0.0075. The re-calls make gsurg and vshare PROMISING, and their full runs start

`outputs/screen/scr-null-floors.md`; the four fits are in `outputs/screen/scr-null-s1.md`, `scr-null-s2.md`,
`scr-null-s1-hp.md` and `scr-null-s2-hp.md`. Each fit is step 1's base arm with seed 1 or 2, compared with step 1's
seed-0 p@swa of its split (R@5). The floors above 0.0075:

| split | read | seed-0 R@5 | seed 1 | seed 2 | floor |
| --- | --- | --- | --- | --- | --- |
| L-musique | musique, zero-shot | 0.2696 | +0.0503 | −0.0079 | **0.0720** |
| L-musique | webqsp, zero-shot | 0.1779 | −0.0070 | −0.0018 | **0.0102** |
| L-hotpotqa | metaqa, in-domain | 0.6540 | −0.0092 | −0.0051 | **0.0149** |
| L-hotpotqa | hotpotqa, zero-shot | 0.8585 | −0.0055 | −0.0130 | **0.0200** |
| L-hotpotqa | webqsp, zero-shot | 0.1200 | +0.0102 | +0.0129 | **0.0233** |

The other seven reads are all in-domain. Each moves by at most 0.0023 between seeds and keeps the floor 0.0075.

- **Both suspicions in the declaration hold (section 2).**
  - Seeds 1 and 2 both read hotpotqa zero-shot below seed 0 (−0.0055, −0.0130). So seed 0 is a high draw on the
    read that had lost under seven arms in seven.
  - On musique read zero-shot, seed 1 alone gains 0.0503 with training unchanged. The read where every musique gain
    was seen swings by five points with the seed.
  - Zero-shot reads move by up to 0.0503. That is over twenty times the most any in-domain read moves (0.0023), except
    L-hotpotqa's metaqa (−0.0092, −0.0051).
- **The re-calls** (`outputs/screen/scr-<arm>-pair-recall.md`):

| arm | filed | re-called | calls the floors change, and what is left | full run |
| --- | --- | --- | --- | --- |
| gsurg | MIXED | **PROMISING** | L-hotpotqa: hotpotqa zero-shot LOSS → WITHIN (−0.0147), webqsp zero-shot GAIN → WITHIN (+0.0124). GAINs left: musique zero-shot in L-musique (+0.0842, floor 0.0720) and 2wiki in-domain in L-hotpotqa (+0.0091) | starts (`fgs-*-rc`, docs/FULL_ROUND5.md) |
| vshare | MIXED | **PROMISING** | L-hotpotqa: hotpotqa zero-shot LOSS → WITHIN (−0.0140). GAIN left: musique zero-shot in L-musique (+0.0891) | started 00:55 (`fvs-*`, docs/FULL_VSHARE.md) |
| ztop50 | MIXED | MIXED | L-hotpotqa: hotpotqa and webqsp zero-shot LOSS → WITHIN. metaqa in-domain still LOSES in both fits (−0.0176; −0.0252, floor 0.0149) | dropped |
| prank | MIXED | MIXED | none. metaqa in-domain (−0.0310, −0.0320), hotpotqa in-domain in L-musique (−0.0080) and hotpotqa zero-shot in L-hotpotqa (−0.0255) still LOSE | dropped |
| pret | MIXED | NO_GAIN | L-musique: musique zero-shot GAIN → WITHIN (+0.0219). hotpotqa zero-shot in L-hotpotqa still LOSES (−0.0253) | dropped |
| hubwalk | NO_GAIN | NO_GAIN | musique and hotpotqa zero-shot LOSS → WITHIN. webqsp zero-shot in L-musique still LOSES (−0.0131, floor 0.0102) | none |

- **The arithmetic written before the null's numbers called vshare and pret correctly.**
  - vshare: its hotpotqa spread, 0.0100, is above 0.0070, and its musique spread, 0.0360, is at most 0.0445.
  - pret: its hotpotqa spread is not above 0.01265, and its musique spread is above 0.01095.
- **What is left of the hotpotqa zero-shot pattern.** Six arms are re-called. Four of them now read hotpotqa
  zero-shot WITHIN seed noise: ztop50, gsurg, hubwalk and vshare. Two still LOSE, by 0.025 each: prank and pret, the
  two arms that rank the retrieval columns within the pool.
- **The musique zero-shot gains.** gsurg's +0.0842 clears the floor 0.0720 by 0.012, and vshare's +0.0891 by 0.017.
  ztop50 (+0.1387) and prank (+0.2016) clear it by far, but both lose metaqa in-domain.
- **What runs now (01:10).**
  - vshare's L-2wiki, L-squad and J5 fits have been on the card since 00:55. They end about 01:45.
  - **Round nine's thirteen builds all passed on the host (00:58 to 01:00).** Each is IDENTICAL to step 1's cached
    columns, with no non-finite value. rel is live on 99.4% of metaqa's rows and 85% of webqsp's, and 0 on the
    passage graphs. gcs is live on all six.
  - **Round nine's smoke passed (01:00, 43 s; `outputs/screen/smoke9/smoke.json`).**
    - On metaqa select, webqsp s1eval and 2wiki select, each arm's carve keeps lean_gpu's base matrix, golds and rows,
      and the arm's columns equal its build's.
    - Each arm's one-epoch repeat is IDENTICAL. Its blocks follow step 1's nine in the fit and are live, and its read
      runs.
  - Round nine's four fits come first in file order and take the card as vshare's fits leave it, about 01:45. Their
    pair verdicts and re-calls land about 03:00 to 03:30. That is later than section 5 gave, because vshare's run took
    the card first.
  - vshare's reads and its L-metaqa fit follow. Its grade and re-grade land about 03:30 to 04:15.
  - gsurg's four fits come after those. Its grade lands about 04:30 to 05:30.

### scr-rel (L-musique): PROMISING. scr-rel-hp (L-hotpotqa): MIXED. The pair is MIXED; its re-call is PROMISING (01:46), and rel's full run starts

`outputs/screen/scr-rel.md`, `outputs/screen/scr-rel-hp.md`, `outputs/screen/scr-rel-pair.md`,
`outputs/screen/scr-rel-pair-recall.md`. R@5 of rel minus step 1's p@swa of the same split, with the 95% interval where
a call is made, and the call under the seed null's floors:

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | **+0.0716 GAIN** [+0.067, +0.076] | −0.0008 WITHIN | **+0.0868 GAIN** (zero-shot; floor 0.0720) [+0.077, +0.097] | +0.0005 WITHIN | −0.0012 WITHIN | **+0.0635 GAIN** (zero-shot) [+0.048, +0.079] |
| L-hotpotqa | **+0.0677 GAIN** [+0.063, +0.072] | −0.0012 WITHIN | +0.0053 WITHIN | −0.0132 LOSS → WITHIN (zero-shot; floor 0.0200) | +0.0015 WITHIN | **+0.0702 GAIN** (zero-shot) [+0.051, +0.089] |

- **metaqa in-domain.** The relation columns pass the twin and close more than half of the MLP's gap to the GNN.
  - hit@1 0.637 → 0.781 (L-musique) and 0.638 → 0.777 (L-hotpotqa). R@5 +0.072 and +0.068, FC@5 +0.061 and +0.062.
  - On the same carve, step 1's grade has the twin at 0.721 / 0.596 / 0.771 and the six-dataset GNN at 0.784 /
    0.667 / 0.889 (R@5 / FC@5 / hit@1). The L-musique fit now reads 0.725 / 0.600 / 0.781: above the twin on all
    three (by 0.004, 0.004 and 0.010), and 57% of step 1's hit@1 gap to the GNN closed.
  - Added to step 1's model, the 26 columns are worth more than the twin's whole lead over it on metaqa (hit@1
    +0.144 against +0.134), though the twin reads 68 more of the look's columns and trains on webqsp.
- **webqsp read zero-shot.** A channel learned on metaqa's 9 relations carries to webqsp's 7,058.
  - R@5 0.178 → 0.241 and 0.120 → 0.190. hit@1 0.069 → 0.124 and 0.059 → 0.099.
  - Absolute webqsp zero-shot is still low. The twin and the GNN train on webqsp (R@5 0.60 and 0.63 there), and no
    fit here does.
- **musique read zero-shot gains (+0.0868, 0.270 → 0.356, above its floor of 0.0720), though rel is 0 on every
  musique row.**
  - A likely reading: the relation columns tell the fit which rows are metaqa's, so metaqa's big pools no longer pull
    the weights the passage graphs share. That is the big-pool confound every pool-normalisation arm traded on
    (ztop50, zret, prank).
  - The gain is only 0.015 above its floor, so the full run's other splits are what can confirm it.
- **The passage graphs in-domain** (squad, hotpotqa, 2wiki, and musique in the L-hotpotqa fit) are all WITHIN
  (|Δ| ≤ 0.0053).
  - Calls are on R@5 (section 2). Two secondary metrics on 2wiki in-domain have intervals below 0: FC@5 in the
    L-musique fit (−0.0047 [−0.0085, −0.0009]) and hit@1 in the L-hotpotqa fit (−0.0085 [−0.0121, −0.0051]). The
    full run reads 2wiki in-domain in three more fits.
- **hotpotqa read zero-shot** in the L-hotpotqa fit (−0.0132) LOSES on the filed rule and is WITHIN under its floor of
  0.0200, like the other arms (the seed null).
- **The full run** (docs/FULL_ROUND9.md) starts on `frl-gate-rc`.
  - At 01:50 its items were moved ahead of gsurg's `-rc` run in the feeder. rel's re-call is the stronger lead, and it
    answers the SOTA gap the round was declared for. vshare's run, already on the card, finishes first.
  - ETAs: its four fits about 02:00 to 03:00, the grade and re-grade about 03:15 to 03:45.

### scr-gcs (L-musique): MIXED. scr-gcs-hp (L-hotpotqa): NO_GAIN. The pair is MIXED and its re-call NO_GAIN (02:03); gcs is not run

`outputs/screen/scr-gcs.md`, `outputs/screen/scr-gcs-hp.md`, `outputs/screen/scr-gcs-pair.md`,
`outputs/screen/scr-gcs-pair-recall.md`. R@5 of gcs minus step 1's p@swa of the same split, with the call under the
seed null's floors:

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | −0.0060 WITHIN | −0.0018 WITHIN | +0.0436 GAIN → WITHIN (zero-shot; floor 0.0720) | −0.0022 WITHIN | −0.0018 WITHIN | **−0.0220 LOSS** (zero-shot; floor 0.0102) [−0.033, −0.011] |
| L-hotpotqa | −0.0084 LOSS → WITHIN (floor 0.0149) | −0.0009 WITHIN | +0.0032 WITHIN | −0.0132 LOSS → WITHIN (zero-shot; floor 0.0200) | +0.0050 WITHIN | −0.0005 WITHIN (zero-shot) |

- **GraphER's propagation of retrieval's scores adds nothing step 1's model lacks.**
  - No in-domain read gains. The largest, 2wiki in the L-hotpotqa fit, is +0.0050 [+0.0030, +0.0072], below 0.0075
    (its FC@5 +0.0160).
  - musique read zero-shot (+0.0436) is within its floor, and webqsp read zero-shot in the L-musique fit LOSES beyond
    its floor.
  - A likely reading: rank and dense_cos already carry retrieval's scores, and step 1's walks (WALK, WALKF, DISTS)
    already spread the seeds over the pool graph. The two columns are a two-step mix of what the model already reads.
- **Against the published systems.** GCS is the parameter-free part of GraphER's method that step 1's model did not
  have. Our MLP's FC@5 was already above GraphER's on hotpotqa and 2wiki (the deck: 84.1 and 70.8 against 78.0 to
  78.9 and 42.5 to 44.1), so this round's open gap was the KB rows, which rel answers.
- **hotpotqa read zero-shot in the L-hotpotqa fit** is −0.0132 in both of round nine's arms (FC@5 −0.0259 and
  −0.0258), though rel's columns are 0 on every hotpotqa row and gcs's are not.
  - A likely reading: an added live block widens the first layer, which changes the initial weights and every later
    torch draw (the batch order is numpy's and stays the same). Each arm is then another seed on that read.
  - Its floor (0.0200) covers both.
- **The full run** (`fgc-*`, docs/FULL_ROUND9.md) does not run: `fgc-gate-rc` exits 1. With gcs out, round nine's
  combined screen (rel with gcs) is not needed.

### Hop depth of the top-1 errors (a diagnosis, declared 8 October about 03:10, before its numbers)

`outputs/mp_unified/hopdiag.py` (its selftest passes), on the host's CPU: no training, and it decides nothing. It
asks whether a question-to-depth input, the KB systems' hop attention (NuTrea, ReaRev, TransferNet weigh a node by how
many hops the question asks for), is worth a screen.
- **Why:** metaqa's 3-hop questions hold most of the gap left after rel (hit@1 0.595 against the GNN's 0.802 in the
  screens). Step 1's MLP reads each node's depth from the seeds (WALK's first-hop and is-seed columns), but nothing
  tells it the question's hop count, so it can only prefer a depth for every question alike.
- **What it reads:** step 1's J5 fit and rel's J5 fit (p@swa), with rrf and the look's twin0 and gnn0 scores, on
  metaqa and webqsp s1eval. A node's depth: 0 is-seed, 1 to 3 the first structural hop from a seed, 4 unreached.
  - hit@1 and R@5, as the reads compute them;
  - **off-depth:** the top-1 is not gold and sits at a depth where no in-pool gold is;
  - **the oracle:** hit@1 and R@5 with every row at such a depth removed. That is a perfect question-to-depth attention
    over this classing, an upper bound and not a model.
  - Each is given over all questions and by metaqa's hop label (from the question ids).
- **How it is used:** if the oracle lifts the MLP's 3-hop hit@1 well past its misses' noise, and the GNN's top-1s sit
  off-depth far less often than the MLP's, a question-to-depth arm is declared as round eleven before its numbers.
  Otherwise the gap is within a depth (which relation chain), and round eleven goes there.
- `outputs/diag/hopdiag-J5.{md,json}`, about 03:30.

**Filed at about 03:06** (`outputs/diag/hopdiag-J5.{md,json}`; 191 s on the host's CPU). The record's host placement
is redacted for the public repository, with the original's sha256.

**metaqa, the 3-hop questions** (3,569; a gold in the pool 0.972; the golds by depth 0 to 4: 0.003, 0.075, 0.144,
0.778, 0.001):

| scorer | hit@1 | oracle hit@1 | R@5 | oracle R@5 | off-depth | off-depth share of misses |
| --- | --- | --- | --- | --- | --- | --- |
| step 1's J5 | 0.425 | 0.499 | 0.339 | 0.403 | 0.206 | 0.358 |
| rel's J5 | 0.576 | 0.659 | 0.451 | 0.495 | 0.161 | 0.380 |
| twin0 | 0.571 | 0.676 | 0.458 | 0.507 | 0.200 | 0.467 |
| gnn0 | 0.802 | 0.839 | 0.586 | 0.599 | 0.090 | 0.453 |

- **Both halves of the rule hold, so round eleven is a question-to-depth arm** (section 4, eleventh round).
  - The oracle lifts the MLPs' 3-hop hit@1 by 0.074 (step 1) and 0.083 (rel). One standard error of a hit@1 over
    3,569 questions is about 0.008, so each lift is about ten of them.
  - The GNN's top-1 sits off-depth on 0.090 of the 3-hop questions, the MLPs' on 0.161 and 0.206: about half as often.
- **A depth prior can close about a third of rel's 3-hop gap to the GNN, at most.** rel's hit@1 is 0.226 below the
  GNN's, and its oracle is still 0.180 below the GNN's oracle. Most of rel's misses sit at a depth that holds a gold
  (0.263 of the questions, the GNN's 0.108). Which chain of relations, within a depth, is the larger part of the gap.
- On the 1-hop and 2-hop questions the oracle adds 0.034 and 0.022 to rel's hit@1 (0.858 to 0.892, 0.897 to 0.919).
  Over all metaqa questions: rel 0.770 (oracle 0.817), the GNN 0.889 (0.913), step 1 0.634 (0.677).

**webqsp, read zero-shot** (1,503 questions; a gold in the pool 0.921; the golds by depth 0 to 4: 0.066, 0.505, 0.310,
0.109, 0.010):

| scorer | hit@1 | oracle hit@1 | R@5 | oracle R@5 | off-depth | top-1 a seed |
| --- | --- | --- | --- | --- | --- | --- |
| step 1's J5 | 0.055 | 0.121 | 0.117 | 0.246 | 0.802 | 0.945 |
| rel's J5 | 0.116 | 0.239 | 0.218 | 0.371 | 0.667 | 0.595 |
| twin0 | 0.563 | 0.642 | 0.602 | 0.647 | 0.242 | 0.150 |
| gnn0 | 0.637 | 0.695 | 0.629 | 0.671 | 0.214 | 0.170 |

- **The MLPs rank a seed first on most webqsp questions** (0.945 and 0.595 of them), where 0.066 of the golds are
  seeds. The oracle doubles their hit@1.
- **This is a failure to transfer, not a KB failure.** On metaqa, a KB graph J5 trains on, the same fits rank a seed
  first on 0.026 and 0.020 of the questions. The question alone cannot say which graph it is asked on, so round
  eleven's prior also reads the pool's profile.

### Within the depth that holds a gold: what tells the gold from the top-1 (a diagnosis, declared 8 October about 04:05, before its numbers)

`outputs/mp_unified/chaindiag.py` (its selftest passes), on the host's CPU. There is no training, and it decides
nothing about adoption. It names the direction of round twelve.
- **Why:** most of rel's metaqa 3-hop misses sit at a depth that holds a gold: 0.263 of the questions, against the
  GNN's 0.108. Round eleven's depth prior cannot reach these. So the question is whether the MLP's inputs can tell the
  gold from its wrong top-1 at that depth, or whether they can and the fit does not.
- **What it reads:** step 1's J5 fit and rel's J5 fit (p@swa), with the look's twin0 and gnn0 scores, on metaqa and
  webqsp s1eval.
  - An **at-depth miss**: the top-1 is not gold, and it sits at a depth that holds a gold.
  - g is the gold at that depth that the fit scores highest.
- **What it measures**, over all questions and by metaqa's hop label:
  - **same_rel:** the share of misses where g and the top-1 are equal on rel's 26 relation columns. same_own and
    same_all are the same over the fit's own columns except SEMB, and over all 112 (step 1's eight blocks and rel's
    three).
  - **favours_gold:** per column, the share of misses where g's value is above the top-1's (ties count half).
  - **qrel_max:** the largest favours_gold over the twelve question-relation match columns.
  - **gnn0 and twin0 prefer the gold:** the share of misses where the look's GNN (or twin) scores g above the top-1.
- **A caution on favours_gold:** it is read on the fit's own errors, so selection bends it. A column the fit scores
  higher tends to sit higher on the top-1 the fit chose. A better match should score higher, so for the match columns
  the bias works against the gold. That makes qrel_max >= 0.65 a conservative sign.
- **How it is read** (rel's J5 fit, metaqa s1eval, the 3-hop questions; the script applies the rule):
  - **INPUT_GAP** if same_rel >= 0.5. On at least half the misses the relation columns are identical. Round twelve
    adds columns from the existing graph that tell relation chains apart.
  - **TRAINING_GAP** otherwise, if qrel_max >= 0.65. A match column favours the gold against the selection. Round
    twelve is an objective: a contrast between each gold and the rows at its own depth, on every dataset.
  - **MATCH_GAP** otherwise. The relation columns differ, but no match favours the gold. Round twelve learns the
    question-relation match from the frozen embeddings, in place of the fixed cosine.
  - Whichever it is gets its own declared round before its numbers, as the same change on all six datasets.
- `outputs/diag/chaindiag-J5.{md,json}`, about 04:20 to 04:30.

**Filed at about 04:07** (`outputs/diag/chaindiag-J5.{md,json}`; 155 s on the host's CPU). The record's host placement
is redacted for the public repository, with the original's sha256.

**The reading is MATCH_GAP.** The metaqa 3-hop questions (3,569), on each fit's at-depth misses:

| fit | at-depth misses (share) | same_rel | same_all | columns differing (mean) | qrel_max (column) | gnn0 prefers gold | twin0 prefers gold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| step 1's J5 | 1,316 (0.369) | 0.000 | 0.000 | 37.8 | 0.697 (relpath_max_h3) | 0.866 | 0.673 |
| rel's J5 | 938 (0.263) | 0.000 | 0.000 | 37.7 | 0.533 (opath_h3_q3) | 0.743 | 0.338 |

- **It is not an input gap.** The gold and the top-1 never share their relation columns (same_rel 0.000). They differ
  on about 38 of the 112 columns.
- **On rel's misses no match column favours the gold.** The best is 0.533. The fixed cosine between the question and
  the relation names does not tell the right chain from the wrong one there.
- **The GNN does tell them apart:** it prefers the gold on 0.743 of rel's misses. The twin reads the same 129 columns
  without message passing and prefers the gold on 0.338. So what separates them is how the GNN uses the relations,
  not a column the MLP lacks.
- **On step 1's misses the match columns do favour the gold** (step 1 does not read them): 0.697 on the 3-hop
  questions, and relchain2_max 0.907 on the 2-hop ones. That is the part rel already recovered: rel's at-depth misses
  are 0.263 of the 3-hop questions against step 1's 0.369, and 0.071 of the 2-hop ones against 0.235.
- **webqsp, read zero-shot:** rel's at-depth misses are 0.217 of the questions, step 1's 0.142. Step 1 mostly ranks a
  seed first (off-depth), and rel moves its top-1 to the right depth but to the wrong node there. The GNN prefers the
  gold on 0.865 of rel's misses.

So round twelve learns the question-relation match from the frozen embeddings, declared next before its numbers.

### The seed null over every split (05:14): 17 of the 36 floors are above 0.0075. zret is ADOPTED; rel and vshare are not

`outputs/screen/scr-nullx-floors.md`. The eight new fits are `outputs/screen/scr-nullx-<split>-s1.md` and `-s2.md` for
L-2wiki, L-squad, J5 and L-metaqa. With the first null's four fits, every split now has two seed-only differences on
each of its six reads.

- **Twelve of the 24 new reads take a floor above 0.0075:**
  - J5: metaqa 0.0154, musique 0.0158, webqsp read zero-shot 0.0295.
  - L-metaqa: metaqa read zero-shot 0.0168, 2wiki 0.0175.
  - L-squad: musique 0.0239, 2wiki 0.0117, webqsp read zero-shot 0.0260.
  - L-2wiki: squad 0.0099, hotpotqa 0.0089, 2wiki read zero-shot 0.0112, webqsp read zero-shot 0.0473.
- **In-domain reads move with the seed too.** On the first null's two splits, in-domain reads stayed within 0.0023 of
  seed 0, except L-hotpotqa's metaqa. Over all six splits, 8 of the 30 in-domain reads take a raised floor. musique
  moves by up to 0.0144 (L-squad, seed 2), and 2wiki by up to 0.0110 (L-metaqa, seed 1).
- **The LOSSes that stopped the full runs were this size.** They were 0.008 to 0.014 (section 2). With training
  unchanged, the seed alone moves 13 of the 36 reads by more than 0.008.

**The re-grades** (`outputs/full_<run>/grade-nullx.md`):

| run | filed | under the null | calls the floors change | what decides |
| --- | --- | --- | --- | --- |
| zret (docs/FULL_ZRET.md) | NOT_ADOPTED | **ADOPT** | All four LOSSes become WITHIN: musique in L-squad (−0.0093, floor 0.0239), metaqa in L-hotpotqa (−0.0121, 0.0149), hotpotqa read zero-shot (−0.0140, 0.0200), 2wiki read zero-shot (−0.0081, 0.0112). Three webqsp zero-shot GAINs become WITHIN | Two primary reads GAIN: metaqa read zero-shot in L-metaqa (+0.0644, floor 0.0168) and musique read zero-shot in L-musique (+0.1202, 0.0720). No read LOSES |
| rel (docs/FULL_ROUND9.md) | NOT_ADOPTED | NOT_ADOPTED | musique in L-squad (−0.0093, 0.0239) and hotpotqa read zero-shot (−0.0132, 0.0200) become WITHIN | Three primary GAINs stand, and one LOSS is left: 2wiki read zero-shot in L-2wiki, −0.0136 against 0.0112 |
| vshare (docs/FULL_VSHARE.md) | NOT_ADOPTED | NOT_ADOPTED | musique in J5 (−0.0108, 0.0158) and in L-squad (−0.0127, 0.0239), and hotpotqa read zero-shot (−0.0140, 0.0200), become WITHIN | Three primary GAINs, and one LOSS is left: metaqa in L-squad, −0.0094 against 0.0075 |

- **zret is the first screen arm adopted.** It takes each pool z-score against the pool's retrieved rows instead of all
  of its rows. Read zero-shot, musique's R@5 rises from 0.270 to 0.390 and metaqa's from 0.077 to 0.142. J5, trained on
  all five datasets, is within the floor on every one of them.
- **rel misses by one read,** 0.0024 beyond its floor. Its gains stand under the null: metaqa in-domain +0.068 to
  +0.080 in every fit that trains on a KB, and webqsp read zero-shot +0.064 to +0.116. rel's columns are 0 on every
  passage graph, so its difference on 2wiki can only come through the weights it trains.
- **vshare misses by one read,** metaqa in-domain in L-squad's fit, which seeds 1 and 2 moved by at most 0.0037.

**What follows, as declared:**
- **Rounds eleven and twelve run on step 1's base.** Their gates read rel's re-grade and dropped the rel branches with
  everything behind them (`scr11-base-rel` at 05:15 and `scr12-base-rel` at 05:32, each exiting 1 with "NOT_ADOPTED;
  the rel base does not run"). qdepth and rmatch go on.
- **relz's full run is dropped at its gate,** which needs rel's ADOPT (`relz.py gate`). Its screen pair and re-call are
  still filed, as a report, about 06:00 to 06:30.
- **gsurg's `-rc` run is the last re-grade** (`fgs-grade-nullx`). Its L-metaqa fit ends about 06:15, and the re-grade
  lands about 06:35 to 06:50.
- **zret's adoption does not change rounds already declared.** Section 2 says that when more than one arm is ADOPTED,
  the next base is declared in a later round, before its numbers. So that round is declared once gsurg's re-grade is
  in. zret's z-scores and gsurg's gradient surgery do not change the same thing.

**Round eleven's smoke passed** (03:45, 37 s; `outputs/screen/smoke11/smoke.json`).
- qdepth's repeat is IDENTICAL.
- Both arms' depth priors moved off zero, and the base arm has none.
- relqd's relation blocks are live, and every arm read the same questions.

**Round twelve's four chain builds passed** (05:32 to 05:35, on the host's CPU). Each part's pools are IDENTICAL to step
1's cache, and every record names rmatch.py's committed sha256.

| carve | questions | entries a question (mean; max) | walks stopped by the chain cap (bucket 0; 1) | walk mass the row cap keeps (bucket 0; 1) | took |
| --- | --- | --- | --- | --- | --- |
| metaqa select | 1,497 | 7,722; 18,309 | 0; 0 | 1.000; 1.000 | 13 s |
| metaqa fit | 5,960 | 7,699; 18,610 | 0; 0 | 1.000; 1.000 | 42 s |
| metaqa s1eval | 9,785 | 7,730; 21,807 | 0; 0 | 1.000; 1.000 | 67 s |
| webqsp s1eval | 1,503 | 89,001; 193,417 | 593; 1,078 | 0.383; 0.297 | 150 s |

- **On metaqa the arm sees every chain.** No walk reaches the 20,000-chain cap. The 64-entry row cap drops 4 entries in
  all, out of the fit carve's 46 million.
- **On webqsp it sees a cut-down walk.** The chain cap stops 593 of the 1,503 bucket-0 walks and 1,078 of the bucket-1
  walks, each after the level that crossed it. The row cap then keeps 0.383 and 0.297 of the walk's mass. Both caps were
  declared before these numbers. On webqsp, read zero-shot, a WITHIN or LOSS can come from the caps, not only from the
  match.
- The webqsp build took 2.5 minutes, not the 10 to 15 the declaration gave from the laptop's 0.3 seconds a question.

### scr-relz (L-musique): MIXED. scr-relz-hp (L-hotpotqa): MIXED. The pair and its re-call are MIXED (05:52), and its full run was dropped at its gate

`outputs/screen/scr-relz.md`, `scr-relz-hp.md`, `scr-relz-pair.md` and `scr-relz-pair-recall.md`. R@5 of relz minus rel's
screen fit of the same split (relz's base), called with the null's floors. Unmarked reads are WITHIN. zs: read
zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | −0.0196 LOSS | +0.0051 | +0.1563 GAIN zs | +0.0018 | +0.0005 | +0.0127 zs |
| L-hotpotqa | −0.0242 LOSS | +0.0030 | +0.0051 | −0.0117 zs (LOSS → WITHIN under the null) | −0.0018 | +0.0601 GAIN zs |

- **ztop50's trade comes back on rel's inputs.**
  - musique read zero-shot rises from rel's 0.356 to 0.513, above rrf's 0.473.
  - webqsp read zero-shot in L-hotpotqa's fit rises from 0.190 to 0.250.
  - metaqa in-domain loses in both fits (−0.0196, −0.0242), as ztop50's did on step 1's inputs.
- **Every arm that ranks or clips a pool's statistics (ztop50, prank, relz) lifts musique read zero-shot far past its
  floor and costs metaqa in-domain.** zret, now adopted, takes a smaller musique gain (0.270 to 0.390) and keeps metaqa
  within its floors.
- **relz's full run needed rel's ADOPT as well** (`relz.py gate`), so it was dropped either way when its re-call landed.
- The re-call record names rel's two screen comparisons by absolute host paths. The committed copy has them relative
  to the workspace, with the host original's sha256 in its `redacted` field.

### Round twelve's first smoke failed on the card in 4 s (05:54), before any training. Fixed and re-queued (06:05)

- **What failed.** The smoke's first step, the carve check, runs a forward of the base model and of the arm. It ran
  before `lean_mlp`'s z-score had been bound to the device, which train and read do first, so on the card the
  z-score's index and its values sat on different devices. The CPU selftest cannot see this.
- **The fix** (`rmatch.py`, a bug fix only): the carve check binds the device ops first, as train, read, chaindiag and
  hopdiag do. Nothing else changes, and the selftest passes. No chain, fit or read existed yet, so nothing is redone.
- **The feeder dropped every item behind the smoke** (05:55): rmatch's two screen fits, their reads and comparisons,
  the pair, its re-call and the full run. They are re-queued under -b names with the same commands, outputs and
  gates. The smoke goes ahead of gsurg's reads again, in the card's free share (0.16).
- **relrm stays dropped**, as declared: rel's re-grade is NOT_ADOPTED, so its base gate (`scr12-base-rel`) exited 1 at
  05:33, and `scr12-base-step1` exited 0.
- **ETAs:** the smoke takes about 10 minutes once it is sent. rmatch's two fits follow round eleven's, and the pair and
  its re-call land about 08:00 to 09:00.

**The re-queued smoke passed** (06:05, 57 s; `outputs/screen/smoke12/smoke.json`).
- The carve checks hold on metaqa select, webqsp s1eval and 2wiki select. The base's matrix, golds and rows are
  unchanged. At its start the arm scores as its base model, bit for bit, and its chain features are finite. On the first
  questions the chains reach 0.988 of metaqa's rows and 0.771 of webqsp's. 2wiki has no typed chains.
- rmatch's repeat is IDENTICAL. Both arms' gates and relation maps moved from zero and are finite, and the base arm's
  fit holds none. relrm's relation blocks are live, and rmatch's blocks are the base's.
- All three arms read the same questions, and rmatch's scores differ from the base's. After one epoch, metaqa select's
  hit@1 is 0.445 for the base, 0.464 for rmatch and 0.558 for relrm. That is a check of the mechanics, not a result.

### scr-qdepth (L-musique): MIXED. scr-qdepth-hp (L-hotpotqa): NO_GAIN. The pair and its re-call are MIXED (06:16), so round eleven's full run was dropped at its gate

`outputs/screen/scr-qdepth.md`, `scr-qdepth-hp.md`, `scr-qdepth-pair.md` and `scr-qdepth-pair-recall.md`. R@5 of
qdepth minus step 1's fit of the same split (rel's re-grade is NOT_ADOPTED, so the prior ran on step 1's base and relqd
never ran), called with the null's floors. Unmarked reads are WITHIN. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | −0.0014 | +0.0001 | +0.1106 GAIN zs | −0.0038 | −0.0054 | −0.0204 LOSS zs |
| L-hotpotqa | −0.0034 | +0.0013 | +0.0035 | −0.0125 zs (LOSS → WITHIN under the null) | +0.0035 | −0.0035 zs |

- **The learned prior does not find the depth the oracle found.** metaqa in-domain moves hit@1 by +0.0065 and +0.0022
  over all its questions, and R@5 by −0.0014 and −0.0034. A perfect question-to-depth attention lifted the 3-hop
  questions' hit@1 alone by 0.074 to 0.083 (section 'Hop depth of the top-1 errors').
- **webqsp, read zero-shot, loses in L-musique's fit:** R@5 0.178 to 0.158 (−0.0204, floor 0.0102), hit@1 −0.0086;
  in L-hotpotqa's fit hit@1 −0.0100. A prior learned on the training graphs' pools does not stop the seeds ranking
  first on webqsp.
- **musique read zero-shot rises from 0.270 to 0.380** (+0.1106, floor 0.0720). Eleven of the seventeen arms screened
  against step 1's L-musique fit moved this read by +0.08 to +0.24, so it is not the prior's own effect.
- `fqd-gate` exited 1 at 06:17, and the feeder dropped the four fits with their reads, comparisons and grades.

### gsurg's full run, re-graded under the null over every split: ADOPT (06:21). Both zret and gsurg are now adopted

`outputs/full_gsurg/grade-nullx.md` (filed NOT_ADOPTED in `grade.md` and `grade-recall.md`). R@5 of gsurg minus step
1's fit of the same split. Only the reads that are not WITHIN under the null are shown; P marks a primary read.

| split | dataset | read | step 1 | gsurg | delta | floor | filed → re-call |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| L-musique | musique | zero-shot (P) | 0.2696 | 0.3537 | +0.0842 | 0.0720 | GAIN → GAIN |
| L-squad | webqsp | zero-shot | 0.1302 | 0.1576 | +0.0274 | 0.0260 | GAIN → GAIN |
| L-hotpotqa | 2wiki | in-domain | 0.8672 | 0.8763 | +0.0091 | 0.0075 | GAIN → GAIN |
| L-2wiki | metaqa | in-domain | 0.6468 | 0.6571 | +0.0103 | 0.0075 | GAIN → GAIN |

- **One primary GAIN and no LOSS, so gsurg is ADOPTED.** Its filed LOSSes fall within their floors: hotpotqa read
  zero-shot in L-hotpotqa's fit (−0.0147, floor 0.0200) and musique in-domain in L-squad's (−0.0094, floor 0.0239).
  Four filed GAINs fall within theirs as well, J5's metaqa among them (+0.0121, floor 0.0154).
- **Its one primary GAIN is the read most arms move.** zret moves the same read further (0.270 to 0.390, against
  gsurg's 0.354) and also GAINs on metaqa read zero-shot in L-metaqa's fit (+0.0644), where gsurg is WITHIN
  (−0.0025).
- **What follows (section 2):** zret and gsurg change different things, so the next base comes from a screen of the
  two together, declared as the thirteenth round before its numbers.

### Round thirteen's smoke passed (06:40; `outputs/screen/smoke13/smoke.json`)

- zgs's repeat is IDENTICAL, and it holds zret's model.
- Its loop met conflicting pairs, 70 of 188 (gsurg's own loop met 60 of 188 on step 1's inputs). zret's loop has no
  surgery.
- All three read the same questions. zgs's scores differ from zret's (the surgery) and from gsurg's (zret's forward).
- One epoch's 2wiki select hit@1 (zret 0.898, gsurg 0.904, zgs 0.904) checks the mechanics only.
- The two fits were re-queued behind rmatch's full run at about 06:47 (thirteenth round, 'Caps and order').

### scr-rmatch (L-musique): PROMISING. scr-rmatch-hp (L-hotpotqa): PROMISING. The pair and its re-call are PROMISING (06:40), and its full run has started

`outputs/screen/scr-rmatch.md`, `scr-rmatch-hp.md`, `scr-rmatch-pair.md` and `scr-rmatch-pair-recall.md`. R@5 of rmatch
minus step 1's fit of the same split, called with the null's floors. Step 1's fits are rmatch's base, since rel's
re-grade is NOT_ADOPTED. Unmarked reads are WITHIN. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | +0.1268 GAIN | +0.0007 | +0.2324 GAIN zs | +0.0023 | +0.0003 | +0.1376 GAIN zs |
| L-hotpotqa | +0.1250 GAIN | −0.0015 | +0.0038 | −0.0072 zs | +0.0051 | +0.1446 GAIN zs |

- **The largest gains on both typed graphs of any arm screened so far.**
  - **metaqa in-domain:** R@5 rises from 0.654 to 0.781 in L-musique's fit and to 0.779 in L-hotpotqa's. hit@1 rises
    from 0.637 to 0.892 and from 0.638 to 0.891. rel's relation columns moved the same R@5 by +0.068 to +0.080 in its
    full run.
  - **webqsp read zero-shot** (webqsp never trains): R@5 rises from 0.178 to 0.316 and from 0.120 to 0.265, and hit@1
    from 0.069 to 0.170 and from 0.059 to 0.130. rel moved this R@5 by +0.064 to +0.116. The match is learned on
    metaqa's relations alone and carries to webqsp's unseen ones through their frozen text embeddings.
- **musique read zero-shot in L-musique's fit rises from 0.270 to 0.502,** above plain retrieval's 0.473; hit@1 rises
  from 0.283 to 0.641.
  - musique's graph has no typed relations, so the match adds nothing to its scores. The change is in the base
    model's weights, trained beside the match.
  - Eleven of the seventeen earlier arms moved this read by +0.08 to +0.24.
- **The untyped reads hold.** squad, hotpotqa and 2wiki in-domain stay within their floors, and so does hotpotqa read
  zero-shot in L-hotpotqa's fit (−0.0072, floor 0.0200).
- **What follows.** `frm-gate-b` exited 0 at 06:41. The full run (docs/FULL_ROUND12.md) trains L-2wiki, L-squad, J5 and
  L-metaqa, decided against step 1's fits. If it is ADOPTED, three arms are adopted (zret, gsurg, rmatch), and the
  next base is declared in a later round, before its numbers. Any speed figure for this arm is cold (8216ffe).

### Round fourteen's first smoke failed (07:06) on its identity check alone. Re-queued under `-b` names (07:13)

`outputs/screen/smoke14/smoke.json`.
- **Every other check passed.** zrm's repeat is IDENTICAL. rmatch's and zrm's matches moved and are finite, and zret's
  fit holds none. All three fits have the same blocks and read the same questions. zrm's scores differ from zret's on
  metaqa and from rmatch's on 2wiki. The chain features are finite on both typed carves.
- **The identity at the start failed on all three carves, 2wiki's untyped one included.** There zrm's forward is
  zret's line for line, so two forwards of the same model differed, not the models.
- **Cause.** The carve check ran its forwards before train's and read's flags (`lean_gpu.set_flags`: deterministic
  algorithms, TF32 off). zret's z-scores against the retrieved rows sum with `index_add_`, which on the card is not
  deterministic without them. rmatch's carve check passed the same test because lean_gpu's own z-score does not sum
  that way. Every fit and read runs under the flags, which is why zrm's training repeat is IDENTICAL.
- **The fix** is in the smoke's check only (`zrm.py`, documented there). The carve check sets the flags first. It also
  records and requires zret's own repeat: two forwards of zret's model must be equal. The model, training and reading
  code are unchanged.
- **Re-queued** under `-b` names with the same commands and outputs, at the same place in the list. The smoke writes
  `outputs/screen/smoke14b`, and the full run's gate is `fzm-gate-b`. No rule changes.
- **Order.** zgs's L-musique fit (`scr-train-zgs-b`) was sent at 07:05 while the smoke ran (only ready items lead the
  card), and it keeps its place. zgs's L-hotpotqa fit waits behind round fourteen's items, as declared.
- **ETAs:** the smoke takes about 2 minutes once 0.16 of the card is free. The pair and its re-call land about 08:30 to
  09:15.

### Round fourteen's re-queued smoke passed (07:16; `outputs/screen/smoke14b/smoke.json`)

- **The identity holds on all three carves** (metaqa select, webqsp s1eval, 2wiki select): two forwards of zret's
  model are equal, and zrm at its start scores as zret's model bit for bit. The chain features are finite on both
  typed carves.
- The rest is as in the first smoke. zrm's repeat is IDENTICAL. rmatch's and zrm's matches moved and are finite, and
  zret's fit holds none. All three fits have the same blocks and read the same questions. zrm's scores differ from
  zret's on metaqa and from rmatch's on 2wiki.
- It ran `zrm.py` as committed in 1e54e4b.
- One epoch's metaqa select hit@1 (zret 0.487, rmatch 0.528, zrm 0.518) checks the mechanics only.
- **What follows:** the two screen fits start as the card frees, ahead of zgs's L-hotpotqa fit. The pair and its
  re-call land about 08:30 to 09:15.

### rmatch's full run, re-graded under the null over every split: NOT_ADOPTED (07:32). Four primary GAINs and one LOSS, on 2wiki read zero-shot

`outputs/full_rmatch/grade-nullx.md` (filed NOT_ADOPTED in `grade.md`). R@5 of rmatch minus step 1's fit of the same
split. Only the reads that are not WITHIN under the null, or were not WITHIN as filed, are shown; P marks a primary
read.

| split | dataset | read | step 1 | rmatch | delta | floor | filed → re-call |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| J5 | metaqa | in-domain (P) | 0.6443 | 0.7788 | +0.1345 | 0.0154 | GAIN → GAIN |
| J5 | musique | in-domain (P) | 0.5609 | 0.5685 | +0.0077 | 0.0158 | GAIN → WITHIN |
| J5 | webqsp | zero-shot (P) | 0.1167 | 0.2795 | +0.1628 | 0.0295 | GAIN → GAIN |
| L-squad | metaqa | in-domain | 0.6535 | 0.7785 | +0.1251 | 0.0075 | GAIN → GAIN |
| L-squad | squad | zero-shot (P) | 0.8797 | 0.8893 | +0.0096 | 0.0075 | GAIN → GAIN |
| L-squad | webqsp | zero-shot | 0.1302 | 0.2711 | +0.1409 | 0.0260 | GAIN → GAIN |
| L-musique | metaqa | in-domain | 0.6537 | 0.7805 | +0.1268 | 0.0075 | GAIN → GAIN |
| L-musique | musique | zero-shot (P) | 0.2696 | 0.5020 | +0.2324 | 0.0720 | GAIN → GAIN |
| L-musique | webqsp | zero-shot | 0.1779 | 0.3155 | +0.1376 | 0.0102 | GAIN → GAIN |
| L-hotpotqa | metaqa | in-domain | 0.6540 | 0.7790 | +0.1250 | 0.0149 | GAIN → GAIN |
| L-hotpotqa | webqsp | zero-shot | 0.1200 | 0.2646 | +0.1446 | 0.0233 | GAIN → GAIN |
| L-2wiki | metaqa | in-domain | 0.6468 | 0.7803 | +0.1334 | 0.0075 | GAIN → GAIN |
| L-2wiki | webqsp | zero-shot | 0.1075 | 0.2607 | +0.1532 | 0.0473 | GAIN → GAIN |
| L-2wiki | 2wiki | zero-shot (P) | 0.8074 | 0.7937 | −0.0136 | 0.0112 | LOSS → LOSS |

- **Four primary GAINs and one LOSS, so rmatch is NOT_ADOPTED.** The LOSS is 2wiki read zero-shot in L-2wiki's fit,
  beyond its floor (−0.0136 against 0.0112). Its hit@1 holds (+0.0006): the loss is in ranks 2 to 5.
- **The typed gains hold in every fit that trains metaqa.**
  - metaqa in-domain: +0.125 to +0.135 R@5, and +0.25 to +0.27 hit@1. In J5, R@5 rises from 0.644 to 0.779.
  - webqsp read zero-shot (webqsp never trains): +0.138 to +0.163 R@5. In J5, R@5 rises from 0.117 to 0.280 and hit@1
    by +0.089.
- **L-metaqa's fit is step 1's bit for bit:** all six reads +0.0000. It trains no typed graph (metaqa is held out and
  webqsp never trains), so the match took no step, and the model trained as step 1's did, from the same
  initialisation and batches. metaqa read zero-shot gets nothing from the match: no fit without metaqa learns one.
- **The untyped reads move only through the model's own weights.** The match adds nothing to a graph without typed
  relations. Yet training it beside the model moved three untyped reads: squad read zero-shot in L-squad's fit up
  (+0.0096), musique read zero-shot in L-musique's up (+0.2324), and 2wiki read zero-shot in L-2wiki's down (−0.0136).
- **What follows.** rmatch is not adopted, and zret stays the base. Round fourteen (zrm: rmatch's match and zret's
  model trained together) is screening. Round fifteen (zrs: the match trained alone on zret's frozen fits, so the
  untyped reads cannot move) is declared next, before its numbers.

### Round fifteen's smoke passed (07:57; `outputs/screen/smoke15/smoke.json`)

- **The identity holds on all three carves** (metaqa select, webqsp s1eval, 2wiki select), under train's and read's
  flags. Two forwards of zret's model are equal, and zrs at its start scores as zret's model bit for bit. The chain
  features are finite on both typed carves.
- **zrs's repeat is IDENTICAL, and every state of its fit holds zret's p@swa state bit for bit.** Its match moved and
  is finite, and only metaqa's select carve trained it (47 steps). zret's fit holds no match. Both fits have the same
  blocks and read the same questions.
- **On 2wiki, which has no typed relations, zrs's scores equal zret's bit for bit.** On metaqa they differ.
- It ran `zrs.py` as committed in ae5976c, in 65 seconds.
- One epoch's metaqa select hit@1 (zret 0.487, zrs 0.499) checks the mechanics only.
- **What follows:** the two screen fits start as the card frees. A zrs fit trains the match alone, so it takes minutes.
  The pair and its re-call land about 08:30 to 09:30.

### A diagnosis: where zret's lost R@5 sits (08:05; `outputs/diag/goldsplit-zret.md`, decides nothing)

`outputs/mp_unified/goldsplit.py` (e2098d2) reads zret's six fits' s1eval reads at p@swa and counts, per dataset, the
questions with two or more golds, FC@5 among them, and how the lost R@5 (one minus each question's R@5) splits between
questions with some of their golds in the top 5 but not all (partly found) and questions with none in it.
- **Read in-domain, the multi-hop datasets lose most of their R@5 in partly-found questions.** Every musique, hotpotqa
  and 2wiki question has two or more golds. In J5, partly-found questions hold 0.770 of musique's lost R@5, 0.620 of
  hotpotqa's and 0.884 of 2wiki's. FC@5 among them is 0.257, 0.839 and 0.701. metaqa (0.605 of its questions have two
  or more golds; the 90th percentile is 20) holds 0.637. This is the question zsep (round sixteen) aims at.
- **Read zero-shot, the KB datasets lose nearly all of it in questions with no gold in the top 5.** webqsp: 0.89 to
  0.94 in every fit. metaqa read zero-shot in L-metaqa's fit: 0.916 (R@5 0.142). An objective that separates a
  question's golds cannot reach these questions.
- **musique read zero-shot in L-musique's fit is between:** R@5 0.390 (0.564 in J5), with 0.460 of its loss in
  none-found questions (0.230 in J5). Its R@5 is below plain retrieval's (rrf 0.473).
- squad has one gold per question, so all of its loss is none-found, and zsep trains its questions as zret did.
- The same count on zrm's two screen fits is queued after their reads (`diag-goldsplit-zrm`).

### scr-zrm (L-musique): PROMISING. scr-zrm-hp (L-hotpotqa): PROMISING. The pair and its re-call are PROMISING (08:09), and zrm's full run has started

`outputs/screen/scr-zrm.md`, `scr-zrm-hp.md`, `scr-zrm-pair.md` and `scr-zrm-pair-recall.md` (the `-b` items of round
fourteen). R@5 of zrm minus zret's fit of the same split, called with the null's floors. Unmarked reads are WITHIN.
zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | +0.1293 GAIN | +0.0004 | +0.1334 GAIN zs | −0.0021 | −0.0005 | +0.1361 GAIN zs |
| L-hotpotqa | +0.1381 GAIN | −0.0012 | −0.0023 | +0.0012 zs | −0.0047 | +0.1221 GAIN zs |

- **rmatch's gains carry onto zret's model.**
  - metaqa in-domain: R@5 rises from 0.650 to 0.779 in L-musique's fit and from 0.642 to 0.780 in L-hotpotqa's.
    hit@1 rises by +0.253 and +0.265.
  - webqsp read zero-shot (webqsp never trains): R@5 rises from 0.177 to 0.313 and from 0.120 to 0.242.
- **musique read zero-shot in L-musique's fit rises from 0.390 to 0.523** (floor 0.0720), above plain retrieval's
  0.473. Beside rmatch's fit (reported only) it is +0.0211 GAIN. musique's graph has no typed relations, so the change
  is in the base model's weights, trained beside the match.
- **The gains are in questions that had no gold in the top 5** (the diagnosis on zrm's fits,
  `outputs/diag/goldsplit-zrm.md`, beside zret's):
  - musique read zero-shot: the none-found share of the lost R@5 falls from 0.460 to 0.263.
  - webqsp read zero-shot: from 0.889 to 0.782 in L-musique's fit, and from 0.923 to 0.828 in L-hotpotqa's.
  - metaqa in-domain: FC@5 among questions with two or more golds rises from 0.284 to 0.466.
- **The untyped reads hold.** The largest move is 2wiki in-domain in L-hotpotqa's fit, −0.0047 (CI −0.0068 to
  −0.0025, floor 0.0075; FC@5 −0.0130). rmatch's full run lost on 2wiki read zero-shot in L-2wiki's fit, a read the
  full run now trains.
- In the filed re-call record, the host's absolute paths of zret's two comparisons are cut to relative ones. Nothing
  else in it changed.
- **What follows.** `fzm-gate-b` exited 0 at 08:09, and `fzm-train-L-2wiki-b` started at 08:10. The full run
  (docs/FULL_ROUND14.md) trains L-2wiki, L-squad, J5 and L-metaqa, decided against zret's fits and re-graded under the
  null over every split. Any speed figure for this arm is cold (8216ffe). ETA of the grade: about 10:00 to 10:45.

### scr-zrs (L-musique): PROMISING. scr-zrs-hp (L-hotpotqa): PROMISING. The pair and its re-call are PROMISING (08:15), so zrs's full run starts

`outputs/screen/scr-zrs.md`, `scr-zrs-hp.md`, `scr-zrs-pair.md` and `scr-zrs-pair-recall.md`. R@5 of zrs minus zret's
fit of the same split, called with the null's floors. Unmarked reads are WITHIN. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | +0.1140 GAIN | +0.0000 | +0.0000 zs | +0.0000 | +0.0000 | +0.0857 GAIN zs |
| L-hotpotqa | +0.1223 GAIN | +0.0000 | +0.0000 | +0.0000 zs | +0.0000 | +0.0905 GAIN zs |

- **The untyped reads are zret's bit for bit, as declared.** A zrs fit trains the match alone on zret's frozen fit,
  and the match adds nothing to a graph without typed relations.
- **On the typed graphs, the match trained alone gains, but less than trained jointly.** Beside zrm's fits of the same
  splits (round fourteen):

  | read | zret | zrs (match alone) | zrm (match and model together) |
  | --- | ---: | ---: | ---: |
  | L-musique, metaqa in-domain | 0.650 | 0.764 | 0.779 |
  | L-musique, webqsp zs | 0.177 | 0.262 | 0.313 |
  | L-musique, musique zs | 0.390 | 0.390 | 0.523 |
  | L-hotpotqa, metaqa in-domain | 0.642 | 0.764 | 0.780 |
  | L-hotpotqa, webqsp zs | 0.120 | 0.210 | 0.242 |

  Beside rmatch's L-musique fit (reported only), zrs LOSES on metaqa (−0.0168), musique read zero-shot (−0.1123) and
  webqsp read zero-shot (−0.0531). Training the match beside the model adds to the match's own gain on the typed
  graphs, and it is the only way the match moves an untyped read.
- **What follows.** `fzs-gate` runs on this re-call. The full run (docs/FULL_ROUND15.md) trains L-2wiki, L-squad, J5
  and L-metaqa; each fit trains the match alone, so it takes minutes. Its items follow zrm's full run on the card. If
  both zrm and zrs are ADOPTED, each re-grade is filed and the next base is declared in a later round, before its
  numbers. Any speed figure for this arm is cold (8216ffe).

### A diagnosis: what rmatch's chain caps cost a chain match (08:47; `outputs/diag/chaincov-webqsp-s1eval.md`, `chaincov-metaqa-s1eval.md`; decides nothing)

`outputs/mp_unified/chaincov.py` (af8b615) walks rmatch's chains on the typed graphs' s1eval carves, as rmatch's build
does, and scores each row by the match at its start alone: S0(v), the sum over a row's entries of w0(c)·m_c(v), with
w0(c) = (1/3)∏σ(zcos_q(r_k)) (rmatch.ChainMatch before training). Nothing is trained or read. Three entry sets: every
entry the walk makes ('all'), the build's 64 per row and seed bucket with the most mass ('mass'), and the 64 with the
largest w0(c)·m_c(v) ('contrib').

webqsp (1,503 questions, each with a gold; the chain cap stopped bucket 0's walk on 593 and some bucket's on 1,259):

| entry set | R@5 by S0 | R@5 by S0, questions the chain cap stopped | S0 kept on gold rows | on other rows |
| --- | ---: | ---: | ---: | ---: |
| all | 0.2138 | 0.2051 | 1.000 | 1.000 |
| mass (the build's) | 0.2109 | 0.2051 | 0.132 | 0.305 |
| contrib | 0.2493 | 0.2466 | 0.174 | 0.358 |

- **The chain cap costs little.** Some chain reaches 0.993 of the golds in the pools of questions it stopped (4,164 of
  4,195), and 0.995 elsewhere (787 of 791).
- **The row cap costs little R@5 as built** (0.2109 against 0.2138 with every entry), though it keeps only 0.132 of
  the starting match's sum on gold rows.
- **Keeping entries by what they add to the match lifts R@5 by 0.0384** over the build's entries, and by 0.0355 over
  every entry.
- **metaqa** (9,785 questions): no cap binds, and all three sets give 0.2938. Some chain reaches 62,693 of its 62,706
  golds in the pools.
- This scores the match at its start, without zrm's trained weights. It is why round seventeen (zrc) re-reads zrm's
  fits with the 'contrib' entries.

### Round seventeen stopped at its identity gate (09:13; `outputs/zrc/identity.md`). Nothing was forked or read

- **The gate filed DIFFERENT.** zrc's build of metaqa's fit carve is rmatch's, array for array, on six of its eight
  parts. On part_2of8 and part_3of8 (748 questions each), ent_row, ent_z and ent_m differ; q_ent, ent_b, q_nrel and
  q_rel are IDENTICAL.
- **Why.** The round assumed that no row cap binds on the fit carve, as on metaqa's s1eval carve (the diagnosis of
  08:47). The build records say it binds in two parts, in bucket 1. Each build drops 2 entries there: of 4,669,299 in
  part 2 and of 4,942,862 in part 3 (zrc's dropped walk mass is 0.001 of 114,057 and 0.002 of 119,321). The two builds
  do not drop the same entries. Nothing else differs among the carve's 45,883,297 entries.
- **As declared, nothing was forked, read or graded.** The feeder dropped the forks and everything after them: the
  screen's reads, comparisons, pair and re-call, and the full run behind `fzc-gate`. zrm's base comparisons
  (`zrc-base-S`: zrm's fits against step 1's) still run. Later rounds on zrm's fits use them.
- **A fit trained on zrc's entries would be a round of its own,** declared in its own file. The diagnosis below
  decides whether one is worth two fits on the card.

### A diagnosis: zrm's screen fits read with zrc's entries (zrcd; declared 8 October about 09:30, before its numbers; decides nothing about any arm)

Code: `outputs/mp_unified/zrcd.py` (its selftest passes). Nothing trains.

- **What it reads.** zrm's two screen fits, `scr-zrm` and `scr-zrm-hp`, copied unchanged into `outputs/diag/fits/zrcd`
  and `zrcd-hp` under the arm zrcd. There is no gate: each fork's record names the gate's verdict and says the fit
  trained on rmatch's entries. Each fork is read on the six s1eval carves with zrc's entries (`outputs/zrc/cache`).
  zrm's folders are only read.
- **What it compares.**
  - Each read with zrm's fit of its split, with zret's and step 1's beside (`outputs/diag/zrcd-<split>.md`).
  - Every read's arrays against zrm's (`-same.md`).
  - The pair and its re-call under the seed null are relz.py's, decided against zrm's screen fits, under zrcd's name
    (`outputs/diag/zrcd-pair`, `zrcd-pair-recall`). Each read's base R@5 is zrm's
    (`outputs/zrc/base-zrm-<split>.json`).
- **What it counts.** `zrcd.py count` on metaqa's fit carve: the questions, rows and entries whose kept set differs
  between the two builds, and their walk mass (`outputs/diag/zrcd-count-metaqa-fit.md`).
- **What can move.**
  - The reads of squad, musique, hotpotqa and 2wiki must be zrm's bit for bit. A difference is a bug, and then no number
    here is filed.
  - metaqa's reads should be zrm's too, since no row cap binds on its s1eval carve.
  - webqsp's reads (zero-shot in both fits) can move.
- **Why it is not zrc.** The fits trained on rmatch's entries. The two builds of the training carve differ in at most
  four entries of 45.9 million. Even so, a fit trained on zrc's entries would not be zrm's bit for bit, and only such a
  fit can be graded as zrc.
- **The rule, declared before its numbers.**
  - Round eighteen is declared in its own file, before any of its numbers, if both of these hold:
    - the re-call is PROMISING (a GAIN and no LOSS among its twelve reads, each called with the seed null's floor);
    - every read of squad, musique, hotpotqa and 2wiki is IDENTICAL to zrm's.

    Round eighteen trains zrc and reads it with its own entries. Its screen trains two fits (L-musique, L-hotpotqa) and
    is decided against zrm's.
  - Otherwise the contribution rule is dropped. Nothing else follows from this diagnosis.
- **Caps and order.**
  - CPU only, 1 CPU each: the count (2 GB), the forks, the comparisons, the pair and the re-call (1 to 4 GB).
  - The reads take 0.30 of the card (share 0.32) and 6 GB, as zrm's.
  - They go right after round seventeen's items, ahead of zgs's re-queued screen and zsep's fits. No rule changes.
- **ETAs.** The forks and the count take minutes. Each read waits for a free share (zrs's full-run reads hold the card
  now), then takes about 3 minutes. The re-call lands about 10:00 to 10:30.
- **Speed.** This times nothing. Any latency figure for zrc's entries is cold (8216ffe).

### zrs's full run, re-graded under the null over every split: ADOPT (09:38). Two primary GAINs and no LOSS

`outputs/full_zrs/grade-nullx.md` (filed ADOPT in `grade.md`). R@5 of zrs minus zret's fit of the same split. Only
the reads that are not WITHIN are shown; P marks a primary read. The other 26 reads are zret's bit for bit (+0.0000).

| split | dataset | read | zret | zrs | delta | floor | filed → re-call |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| J5 | metaqa | in-domain (P) | 0.6431 | 0.7636 | +0.1205 | 0.0154 | GAIN → GAIN |
| J5 | webqsp | zero-shot (P) | 0.1417 | 0.2329 | +0.0912 | 0.0295 | GAIN → GAIN |
| L-squad | metaqa | in-domain | 0.6472 | 0.7631 | +0.1159 | 0.0075 | GAIN → GAIN |
| L-squad | webqsp | zero-shot | 0.1469 | 0.2264 | +0.0795 | 0.0260 | GAIN → GAIN |
| L-musique | metaqa | in-domain | 0.6496 | 0.7637 | +0.1140 | 0.0075 | GAIN → GAIN |
| L-musique | webqsp | zero-shot | 0.1766 | 0.2623 | +0.0857 | 0.0102 | GAIN → GAIN |
| L-hotpotqa | metaqa | in-domain | 0.6419 | 0.7642 | +0.1223 | 0.0149 | GAIN → GAIN |
| L-hotpotqa | webqsp | zero-shot | 0.1195 | 0.2100 | +0.0905 | 0.0233 | GAIN → GAIN |
| L-2wiki | metaqa | in-domain | 0.6464 | 0.7642 | +0.1178 | 0.0075 | GAIN → GAIN |
| L-2wiki | webqsp | zero-shot | 0.1338 | 0.2108 | +0.0770 | 0.0473 | GAIN → GAIN |

- **Two primary GAINs and no LOSS among the 36 reads, so zrs is ADOPTED.** They are J5's metaqa in-domain (+0.1205
  R@5, +0.2052 hit@1) and J5's webqsp read zero-shot (+0.0912 R@5, +0.0532 hit@1). No call changed under the null.
- **The match lifts both typed graphs in every fit that trains metaqa.**
  - metaqa in-domain: +0.114 to +0.122 R@5, and +0.19 to +0.23 hit@1.
  - webqsp read zero-shot (webqsp never trains): +0.077 to +0.091 R@5.
- **Every untyped read is zret's bit for bit, as declared.** Each fit trains the match alone on zret's frozen fit.
- **L-metaqa's fit is zret's on all six reads.** It trains no typed graph, so the match's gates stay at zero. metaqa
  read zero-shot stays at 0.1417 and webqsp at 0.0997: no fit without metaqa learns a match.
- **Beside rmatch's J5 fit (reported only), zrs is lower on both typed reads:** metaqa −0.0152 and webqsp read
  zero-shot −0.0466. The screen's fits showed the same pattern: the match gains more when it trains beside the model.
- **What follows (docs/FULL_ROUND15.md, section 4).** zrs is the base of every later screen and run, unless zrm's
  re-grade is ADOPT too (due about 10:15 to 10:45). In that case the next base is declared in a later round, before its
  numbers. That round is declared next, before zrm's grade lands.

### The next base if zrm is ADOPTED too (declared 8 October about 09:50, before zrm's re-grade; docs/BASE_ZRM_ZRS.md)

- **zrs is the incumbent.** It was ADOPTED first, and it changes less: zret's fits are frozen and the match is added on
  top.
- **zrm's six fits are compared with zrs's, read by read** (36 reads, under the seed null over every split). zrm
  becomes the base if at least one primary read GAINs and none of the 36 LOSEs. Otherwise zrs stays the base.
- **It runs only if zrm's re-grade is ADOPT.** Otherwise zrs is the base.
- Both arms' reads against zret's fits are already known on five of six splits, so this is not blind. The rule is the
  one every round uses, unchanged.

### zrcd: PROMISING, and every read but webqsp's is zrm's bit for bit (09:45; decides nothing about any arm). Round eighteen is declared next

Records: `outputs/diag/zrcd-L-musique.md` and `zrcd-L-hotpotqa.md` with their `-same.md`, `zrcd-pair.md`,
`zrcd-pair-recall.md` and `zrcd-count-metaqa-fit.md`. The table gives R@5 of zrm's screen fits read with zrc's
entries, against the same fits read with rmatch's.

| split | read | zrm | zrcd | delta R@5 [95% CI] | floor | call | delta hit@1 |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: |
| L-musique | webqsp zs | 0.3128 | 0.3342 | +0.0215 [+0.0074, +0.0358] | 0.0102 | GAIN | −0.0459 |
| L-hotpotqa | webqsp zs | 0.2416 | 0.2860 | +0.0444 [+0.0313, +0.0577] | 0.0233 | GAIN | −0.0053 |

- **The re-call is PROMISING.** webqsp read zero-shot GAINs in both fits. The other ten reads are +0.0000.
- **Every read of squad, musique, hotpotqa, 2wiki and metaqa is zrm's bit for bit** (the `-same` files: IDENTICAL).
  Only webqsp's reads differ: the number of golds in the top five changes for 312 and 263 of its 1,503
  questions.
- **On webqsp, R@5 rises but hit@1 falls in both fits,** by 0.046 in L-musique's fit and 0.005 in L-hotpotqa's. The
  screens decide on R@5. The hit@1 drop is reported beside it, and round eighteen reports it too.
- **The count.** On metaqa's fit carve, the two builds keep a different set in 2 of 5,960 questions, in one row each.
  Each swaps one entry, so 2 of 45,883,297 entries differ, and their walk mass rounds to 0.0000.
- **The declared rule holds on both counts:** the re-call is PROMISING, and every untyped read is IDENTICAL. So round
  eighteen is declared next, in its own file, before its numbers. It trains zrc on its own entries, with two screen
  fits decided against zrm's.

### zrm's full run, re-graded under the null over every split: ADOPT (09:51). Three primary GAINs and no LOSS

`outputs/full_zrm/grade-nullx.md` (filed ADOPT in `grade.md`). R@5 of zrm minus zret's fit of the same split. Only
the reads that are not WITHIN are shown; P marks a primary read.

| split | dataset | read | zret | zrm | delta | floor | filed → re-call |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| J5 | metaqa | in-domain (P) | 0.6431 | 0.7788 | +0.1357 | 0.0154 | GAIN → GAIN |
| J5 | webqsp | zero-shot (P) | 0.1417 | 0.2558 | +0.1140 | 0.0295 | GAIN → GAIN |
| L-squad | metaqa | in-domain | 0.6472 | 0.7792 | +0.1321 | 0.0075 | GAIN → GAIN |
| L-squad | webqsp | zero-shot | 0.1469 | 0.2505 | +0.1036 | 0.0260 | GAIN → GAIN |
| L-musique | metaqa | in-domain | 0.6496 | 0.7789 | +0.1293 | 0.0075 | GAIN → GAIN |
| L-musique | musique | zero-shot (P) | 0.3897 | 0.5231 | +0.1334 | 0.0720 | GAIN → GAIN |
| L-musique | webqsp | zero-shot | 0.1766 | 0.3128 | +0.1361 | 0.0102 | GAIN → GAIN |
| L-hotpotqa | metaqa | in-domain | 0.6419 | 0.7801 | +0.1381 | 0.0149 | GAIN → GAIN |
| L-hotpotqa | webqsp | zero-shot | 0.1195 | 0.2416 | +0.1221 | 0.0233 | GAIN → GAIN |
| L-2wiki | metaqa | in-domain | 0.6464 | 0.7794 | +0.1330 | 0.0075 | GAIN → GAIN |
| L-2wiki | webqsp | zero-shot | 0.1338 | 0.2644 | +0.1306 | 0.0473 | GAIN → GAIN |

- **Three primary GAINs and no LOSS among the 36 reads, so zrm is ADOPTED:**
  - J5's metaqa in-domain: +0.1357 R@5, +0.2694 hit@1.
  - J5's webqsp read zero-shot: +0.1140 R@5, +0.0699 hit@1.
  - musique read zero-shot in L-musique's fit: +0.1334 R@5, +0.2305 hit@1.

  No call changed under the null.
- **zrm gains more than zrs on the typed graphs, in every fit that trains metaqa.**

  | read | zrm | zrs |
  | --- | --- | --- |
  | metaqa in-domain | +0.129 to +0.138 | +0.114 to +0.122 |
  | webqsp read zero-shot | +0.104 to +0.136 | +0.077 to +0.091 |
- **The untyped reads move, but only within their floors.** The model's own weights train beside the match. The
  largest moves, all within 0.0075:
  - squad read zero-shot in L-squad's fit, +0.0051;
  - 2wiki in L-hotpotqa's fit, −0.0047;
  - 2wiki in J5's, +0.0046.

  2wiki read zero-shot in L-2wiki's fit was rmatch's LOSS on step 1's base. Here it is −0.0020 (floor 0.0112).
- **L-metaqa's fit is zret's bit for bit** on all six reads, since it trains no typed graph.
- **What follows.** Both zrm and zrs are ADOPTED, so docs/BASE_ZRM_ZRS.md runs: zrm's fits against zrs's, read by read.
  It was declared at 09:50, before this grade. Its gate passes on these two re-grades, and its verdict lands about
  10:15 to 10:30.

### Eighteenth round: zrc trained on its own entries (declared 8 October about 10:00, before any of its numbers; full run docs/FULL_ROUND18.md)

Code: `outputs/mp_unified/zrct.py` (its selftest passes).

- **Why: zrcd's declared rule holds.** zrcd's re-call is PROMISING, and every read but webqsp's is zrm's bit for bit.
  Round seventeen could not fork zrm's fits as zrc's, since the two training carves differ in two entries. So zrc trains.
- **What trains.** The arm zrc, (zrm.ZRM, zrc.ChainCarveZRC): zrm's model, settings and training (rmatch.py's train),
  over zrc's builds (`outputs/zrc/cache`). Seed 0. Only the entries kept on the typed graphs change.
- **The screen's two fits:** `scr-zrct` (L-musique) and `scr-zrct-hp` (L-hotpotqa). Each is read on the six s1eval
  carves with zrc's entries.
- **Decided against zrm's screen fits** (`scr-zrm` and `scr-zrm-hp`), with zret's and step 1's beside. The pieces are
  zrc.py's: the comparison (with its `-same` file), the pair, and the re-call under the seed null. Each read's base R@5
  comes from `outputs/zrc/base-zrm-<split>.json`.
- **What can move.**
  - webqsp's reads, through the entries.
  - Every other read, through the weights alone: two entries of metaqa's training carve differ, so the weights may
    differ slightly.
  - hit@1 is reported beside R@5. On webqsp, zrcd's R@5 rose while its hit@1 fell.
- **What follows.**
  - The full run (docs/FULL_ROUND18.md) starts only if both hold: the re-call is PROMISING (a GAIN and no LOSS among
    its twelve reads), and zrm is the base (docs/BASE_ZRM_ZRS.md re-graded ADOPT).
  - If zrs stays the base, nothing follows here. A grade of zrc against zrs's fits would be declared in a later round.
- **Caps and order.**
  - Each fit takes 0.26 of the card (share 0.28) and 9 GB. Each read takes 0.30 (share 0.32) and 6 GB, as
    zrcd's did (zrm's J5 read peaked at 3.9 GB).
  - The two fits go after zgs's and zsep's screen reads and ahead of the queued full runs' fits in the feeder's list.
    zsep's reads were moved up from the end of the list, so its verdict does not wait behind these fits. The full run
    goes at the end. Nothing is preempted.
- **ETAs.** Each fit takes about 35 minutes once the card has room. zgs's and zsep's fits hold the card until about
  10:10 to 10:25. The re-call lands about 11:00 to 11:30.
- **Speed.** Any latency figure is cold (8216ffe), timing the walk, w0, the selection and the forward per question.

### The next base: zrm (10:05). zrm's fits against zrs's, re-graded under the null over every split: ADOPT

`outputs/zbase/grade-nullx.md` (filed ADOPT in `grade.md`; declared in docs/BASE_ZRM_ZRS.md). The table gives R@5 of
zrm minus zrs's fit of the same split. Only the reads that are not WITHIN are shown; P marks a primary read.

| split | dataset | read | zrs | zrm | delta | floor | filed → re-call | delta hit@1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| J5 | metaqa | in-domain (P) | 0.7636 | 0.7788 | +0.0152 | 0.0154 | GAIN → WITHIN | +0.0642 |
| J5 | webqsp | zero-shot (P) | 0.2329 | 0.2558 | +0.0229 | 0.0295 | GAIN → WITHIN | +0.0166 |
| L-squad | metaqa | in-domain | 0.7631 | 0.7792 | +0.0162 | 0.0075 | GAIN → GAIN | +0.0445 |
| L-squad | webqsp | zero-shot | 0.2264 | 0.2505 | +0.0242 | 0.0260 | GAIN → WITHIN | +0.0140 |
| L-musique | metaqa | in-domain | 0.7637 | 0.7789 | +0.0153 | 0.0075 | GAIN → GAIN | +0.0621 |
| L-musique | musique | zero-shot (P) | 0.3897 | 0.5231 | +0.1334 | 0.0720 | GAIN → GAIN | +0.2305 |
| L-musique | webqsp | zero-shot | 0.2623 | 0.3128 | +0.0504 | 0.0102 | GAIN → GAIN | +0.0466 |
| L-hotpotqa | metaqa | in-domain | 0.7642 | 0.7801 | +0.0159 | 0.0149 | GAIN → GAIN | +0.0490 |
| L-hotpotqa | webqsp | zero-shot | 0.2100 | 0.2416 | +0.0316 | 0.0233 | GAIN → GAIN | +0.0160 |
| L-2wiki | metaqa | in-domain | 0.7642 | 0.7794 | +0.0152 | 0.0075 | GAIN → GAIN | +0.0441 |
| L-2wiki | webqsp | zero-shot | 0.2108 | 0.2644 | +0.0536 | 0.0473 | GAIN → GAIN | +0.0319 |

- **One primary GAIN and no LOSS among the 36 reads, so zrm is the base of every later round.** The GAIN is musique
  read zero-shot in L-musique's fit: +0.1334 R@5 (floor 0.0720) and +0.2305 hit@1. zrs's untyped reads are zret's, so
  this is what training the match jointly with the model's own weights adds there.
- **The typed reads gain too.**
  - metaqa in-domain: +0.015 to +0.016 in all five fits that train it. Four are GAINs; J5's +0.0152 is WITHIN its floor
    of 0.0154.
  - webqsp read zero-shot: +0.023 to +0.054. Three are GAINs; J5's and L-squad's turn WITHIN under the null.
- **The untyped reads stay within their floors.** The largest moves are squad read zero-shot in L-squad's fit
  (+0.0051), 2wiki in L-hotpotqa's (−0.0047) and 2wiki in J5's (+0.0046). L-metaqa's fit is zret's under both arms, so
  its six reads are +0.0000.
- **What follows.**
  - zrm is the base. Round eighteen's full run (docs/FULL_ROUND18.md) now waits only on its screen's re-call.
  - zgs (round thirteen) and zsep (round sixteen) are still decided against zret's fits, as declared. If either is
    adopted, it is combined with zrm in a later round, declared before its numbers.
  - Any latency figure for zrm is cold (8216ffe).

### Amendment: with zrm the base, a PROMISING re-call of zgs or zsep leads to a combination with zrm, not a full run against zret (declared 8 October about 10:15, before either re-call)

- **Why.**
  - zrm became the base at 10:05 (docs/BASE_ZRM_ZRS.md). zgs (round thirteen) and zsep (round sixteen) were declared
    against zret's fits, and an arm adopted against zret is only combined with the base in a later round
    (docs/BASE_ZRM_ZRS.md, section 4). So their full runs against zret can no longer change the base.
  - Those runs would still hold the card ahead of round eighteen's full run, which can. zgs's four fits take about 2 to
    3.5 hours beside others (gsurg's loop).
- **What changes.**
  - The re-calls of zgs and zsep are filed as declared, and still decide whether each idea goes on.
  - A PROMISING re-call no longer starts a full run against zret's fits. Instead the idea is combined with zrm in a
    later round, declared before its numbers. That round has two screen fits decided against zrm's screen fits, then a
    full run against zrm's fits, re-graded under the null over every split, as every round.
  - The bar for adoption is unchanged: a full run against the base.
- **In the feeder,** the full-run items of rounds thirteen and sixteen are commented out unrun:
  - `fzg-train-*-b` to `fzg-grade-nullx-b`;
  - `fzp-train-*` to `fzp-grade-nullx`.

  Their gates (`fzg-gate-b`, `fzp-gate`) stay. They record each re-call's verdict and start nothing.
- **Round eighteen's full run (`fzk-*`) is unaffected.** It still starts only on its screen's PROMISING re-call.
- **What exists at this point.**
  - zgs: its L-musique fit's comparison was computed on the host at 09:45. Its L-hotpotqa fit is still training,
    and its pair and re-call do not exist.
  - zsep: its two fits are still training, so none of its numbers exist.
  - Round eighteen: no numbers exist.

  The amendment applies to both arms whatever their re-calls say.

### scr-zsep (L-musique): NO_GAIN. scr-zsep-hp (L-hotpotqa): MIXED. The pair is MIXED and its re-call NO_GAIN (10:28); zsep is dropped

`outputs/screen/scr-zsep.md`, `outputs/screen/scr-zsep-hp.md`, `outputs/screen/scr-zsep-pair.md`,
`outputs/screen/scr-zsep-pair-recall.md`. R@5 of zsep minus zret's fit of the same split, with the call under the seed
null's floors:

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | +0.0043 WITHIN | −0.0012 WITHIN | −0.0687 LOSS → WITHIN (zero-shot; floor 0.0720) | −0.0030 WITHIN | −0.0007 WITHIN | +0.0003 WITHIN (zero-shot) |
| L-hotpotqa | +0.0104 GAIN → WITHIN (floor 0.0149) | −0.0015 WITHIN | **−0.0101 LOSS** (floor 0.0075) [−0.0163, −0.0043] | −0.0026 WITHIN (zero-shot) | −0.0003 WITHIN | −0.0006 WITHIN (zero-shot) |

- **No GAIN among the twelve reads, and one LOSS:** musique read in-domain in L-hotpotqa's fit. So the re-call is
  NO_GAIN.
- **hit@1 rises, R@5 does not.**
  - hit@1 rises on metaqa, hotpotqa and 2wiki read in-domain: metaqa +0.0169 and +0.0216, hotpotqa +0.0205, 2wiki
    +0.0151 and +0.0120.
  - On hotpotqa and 2wiki in-domain, R@5 and FC@5 fall slightly (R@5 −0.0003 to −0.0030, FC@5 −0.0035 to −0.0068).
    metaqa's R@5 rises within its floors (+0.0043 and +0.0104).
  - The objective was meant to help the golds that trail a leading gold, which R@5 counts. On hotpotqa and 2wiki it
    moved the top one instead.
- **musique falls both ways.**
  - Read zero-shot in L-musique's fit: −0.0687 R@5, −0.0401 FC@5 and −0.0530 hit@1. That is within its floor of
    0.0720, but the largest move in the pair.
  - Read in-domain in L-hotpotqa's fit: the LOSS.
- **The loss changed most training questions.** Questions with two or more golds in the pool, per training carve:
  - metaqa: 3,488 of 5,960;
  - hotpotqa: 5,741 of 5,930;
  - 2wiki: 5,621 of 5,928;
  - musique: 4,224 of 4,601;
  - squad: none (one gold each).
- **What follows.**
  - By the amendment (10:15), only a PROMISING re-call led to a combination with zrm. None is declared.
  - `fzp-gate` exits 1. Its full-run items were already commented out.
  - zgs's re-call (round thirteen) is next, about 11:50 to 12:00. Round eighteen's re-call lands about 11:05 to 11:30.

### Nineteenth round: a head for graphs with typed relations, on zrm (declared 8 October about 11:00, before any of its numbers; full run docs/FULL_ROUND19.md)

Code: `outputs/mp_unified/zkind.py` (its selftest passes).

- **Why: the typed graphs need their own read.**
  - On metaqa and webqsp a seed is the question's topic entity. It holds only 0.066 of webqsp's in-pool golds and
    0.005 of metaqa's. On the passage graphs a seed is a retrieved passage, and often gold.
  - What the MLPs learn of metaqa does not carry to webqsp. step 1's and rel's J5 fits rank a seed first for 0.9454
    and 0.5948 of webqsp's questions, against 0.0262 and 0.0198 of metaqa's (outputs/diag/hopdiag-J5). hopdiag reads
    only step 1's and rel's fits, so zrm's rate is not measured.
  - zrm reads metaqa in-domain at 0.78 R@5, and webqsp zero-shot at 0.24 to 0.31 in the same five fits
    (outputs/zbase/grade-nullx.json).
  - One output layer serves both kinds of graph.
- **What trains.** The arm zkind, (zkind.ZKind, rmatch.ChainCarveBase): zrm's model, settings and training
  (rmatch.py's train), with learned offsets on its output layer's weights and bias and on rrf's base weight. Seed 0.
  - The offsets apply only to a batch that carries chain entries: a graph with typed relations (metaqa, webqsp). They
    start at zero: 130 new weights.
  - The graph's kind is read from the batch, never from a dataset's name. So the head applies to any typed graph,
    seen in training or not.
  - An offset takes a gradient only from a typed graph's batch. With no typed graph in training (L-metaqa's fit) the
    offsets stay at zero, and the fit is zrm's bit for bit. The full run's L-metaqa comparison checks this (`-same`).
- **The screen's two fits:** `scr-zkind` (L-musique) and `scr-zkind-hp` (L-hotpotqa). Both train metaqa, so both fit
  the offsets. Each is read on the six s1eval carves, as zrm's fits were.
- **Decided against zrm's screen fits** (`scr-zrm` and `scr-zrm-hp`), with zret's and step 1's beside. The pieces are
  zkind.py's: the comparison, the pair, and the re-call under the seed null (relz.py's, with zrc.py's mapping to zrm's
  fits). Each read's base R@5 comes from `outputs/zrc/base-zrm-<split>.json`.
- **What can move.**
  - metaqa's reads (in-domain in both fits) and webqsp's (zero-shot in both), through the offsets.
  - The passage datasets' reads, through the shared weights alone.
  - hit@1 is reported beside R@5.
- **What follows.** The full run (docs/FULL_ROUND19.md) starts only if both hold: the re-call is PROMISING (a GAIN and
  no LOSS among its twelve reads), and zrm is the base (docs/BASE_ZRM_ZRS.md re-graded ADOPT, as it is since 10:05).
- **Caps and order.**
  - Each fit takes 0.26 of the card (share 0.28) and 9 GB. Each read takes 0.30 (share 0.32) and 6 GB, as round
    eighteen's.
  - The two fits go after round eighteen's screen items, ahead of the queued full runs' fits. The full run goes at
    the end of the list. Nothing is preempted.
- **ETAs.** The fits start when round eighteen's reads leave room on the card, about 11:10, and take about 40 minutes
  each. The re-call lands about 11:55 to 12:15.
- **Numbering.** If zgs's re-call (round thirteen, about 11:50) is PROMISING, its combination with zrm is round
  twenty, by the amendment (10:15).
- **Speed.** On a typed graph the offsets add one sum per weight, once per batch. Any latency figure is cold
  (8216ffe): each question timed from scratch, with the walk, the move of its entries to the device and the forward.

### scr-zrct (L-musique): PROMISING. scr-zrct-hp (L-hotpotqa): PROMISING. The pair and its re-call are PROMISING (11:08), so round eighteen's full run starts

`outputs/screen/scr-zrct.md` and `scr-zrct-hp.md` with their `-same.md`, `scr-zrct-pair.md` and
`scr-zrct-pair-recall.md`. R@5 of zrc (trained on its own entries) minus zrm's fit of the same split, called with the
null's floors. Unmarked reads are WITHIN. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | −0.0007 | −0.0006 | +0.0004 zs | +0.0001 | +0.0024 | +0.0257 GAIN zs (floor 0.0102) |
| L-hotpotqa | −0.0008 | −0.0003 | +0.0045 | −0.0006 zs | +0.0051 | +0.0450 GAIN zs (floor 0.0233) |

- **webqsp read zero-shot GAINs in both fits,** as zrcd's diagnosis found. R@5 rises from 0.313 to 0.338 in
  L-musique's fit and from 0.242 to 0.287 in L-hotpotqa's. Trained on its own entries, zrc keeps what zrcd measured
  with zrm's weights (+0.0215 and +0.0444).
- **hit@1 on webqsp falls in both fits,** by 0.0532 and 0.0160 (zrcd: 0.046 and 0.005). The screens decide on R@5; the
  full run reports hit@1 beside it.
- **The other ten reads move by 0.0051 or less.**
  - 2wiki in-domain rises +0.0024 and +0.0051, with FC@5 +0.0054 and +0.0146.
  - metaqa in-domain moves −0.0007 and −0.0008.
- **Every read differs from zrm's** (the `-same` files: DIFFERENT on all six datasets in both fits). That is expected:
  two entries of metaqa's training carve differ, so the weights differ. The golds in the top five change for 124 to
  906 questions a read.
- **No call changed under the null.** In the filed re-call record, the host's absolute paths of zrm's two comparisons
  are cut to relative ones. Nothing else in it changed.
- **What follows.**
  - `fzk-gate` passes: the re-call is PROMISING, and zrm is the base (ADOPT, 10:05).
  - The full run (docs/FULL_ROUND18.md) trains L-2wiki, L-squad, J5 and L-metaqa when the card has room, decided
    against zrm's fits and re-graded under the null over every split.
  - L-metaqa's fit trains no typed graph, so its six reads must be zrm's bit for bit (`compare-L-metaqa-same.md`).
  - Any speed figure for this arm is cold (8216ffe).
- **ETAs.** The card holds zgs's L-hotpotqa fit until about 11:40 and round nineteen's two screen fits until about
  11:45 to 11:50. The four fits take about 40 minutes each, two or three at a time. The grade lands about 13:30 to
  14:30.

### Twentieth round: an objective for recall at five, on zrm (declared 8 October about 11:33, before any of its numbers; full run docs/FULL_ROUND20.md)

Code: `outputs/mp_unified/zrk.py` (its selftest passes).

- **Why: the screens decide on R@5, and zrm loses most of its R@5 on questions it finds only in part.**
  - Where zrm's screen fits lose R@5, some of the question's golds are found and some are not. That is 0.737 and 0.765
    of musique's lost R@5 in L-musique's and L-hotpotqa's fits, 0.887 and 0.875 of 2wiki's, 0.605 and 0.725 of
    hotpotqa's, and 0.825 and 0.818 of metaqa's (outputs/diag/goldsplit-zrm). FC@5 of musique's questions is 0.215
    and 0.252.
  - lean_gpu's loss (listwiseD) trains for R@5 only through a softmax over the whole pool. Each gold's term pushes on
    every row by its share, so most of the push goes to the rows already on top.
  - zsep (round sixteen) took each question's other golds out of each gold's softmax. It was NO_GAIN, with a LOSS on
    musique read in-domain in L-hotpotqa's fit.
- **What trains.** The arm zrk: zrm's model, carve, settings and training (rmatch.py's train), seed 0, with a smooth
  recall at five added to listwiseD:
  - loss = listwiseD + 1 x (1 - the question's smooth recall at five), averaged over the questions with a gold, as
    listwiseD is;
  - a gold's rank in its pool is made smooth by a sigmoid, temperature 0.1 score units, over the pool's other rows
    (ApproxNDCG's form, Qin, Liu and Li 2010);
  - "rank at most five" is made smooth by a sigmoid, temperature 1 rank (the Recall@k surrogate of Patel, Tolias and
    Matas, CVPR 2022);
  - another gold above a gold counts toward its rank but carries no gradient, since two golds that swap never change
    R@5. So no gold is pushed down by another, and no row but a gold is pushed up;
  - the surrogate's gradient sits on the golds just below the fifth row and on the rows just above them.
- **The same loss on every training question, on every dataset.** The settings are fixed before any number: K = 5 (the
  metric's cutoff), temperatures 0.1 and 1, weight 1. No new column, block, carve, graph or model. Reads and serving
  are zrm's.
- **A check before training.** Each fit checks the loss and its gradient on the first eight questions of every
  training carve, at the fit's start, against a float64 reference. A mismatch stops the fit (exit 1). Each fit also
  records its training carves' census: questions with two or more golds, the largest pool, and the largest golds-by-rows
  count of one question.
- **The screen's two fits:** `scr-zrk` (L-musique) and `scr-zrk-hp` (L-hotpotqa). Each is read on the six s1eval carves,
  as zrm's fits were.
- **Decided against zrm's screen fits** (`scr-zrm` and `scr-zrm-hp`), with zret's and step 1's beside. The pieces are
  zrk.py's: the comparison, the pair, and the re-call under the seed null (relz.py's, with zrc.py's mapping to zrm's
  fits). Each read's base R@5 comes from `outputs/zrc/base-zrm-<split>.json`.
- **What can move.** Every read, since the loss changes training on every dataset. hit@1 is reported beside R@5. It may
  fall, since the surrogate does not care about the order inside the top five.
- **What follows.** The full run (docs/FULL_ROUND20.md) starts only if both hold: the re-call is PROMISING (a GAIN and
  no LOSS among its twelve reads), and zrm is the base (docs/BASE_ZRM_ZRS.md re-graded ADOPT, as it is since 10:05).
- **Caps and order.**
  - Each fit takes 0.26 of the card (share 0.28) and 9 GB, as zrm's screen fits did. Their torch peaks were 3.5 and
    5.2 GB under a 6.2 GB cap. On metaqa's training carve the surrogate's largest batch holds about 0.5 million gold-row
    pairs, about 2 MB a tensor.
  - Each read takes 0.30 (share 0.32) and 6 GB.
  - The fits go after round nineteen's screen items, ahead of the queued full runs' fits. The full run goes at the end
    of the list. Nothing is preempted.
- **ETAs.** The fits start when the card has room, about 11:55 to 12:15, and take about 35 minutes each (zrm's screen
  fits took 34). The re-call lands about 12:45 to 13:15.
- **Numbering.** Rounds are numbered in the order they are declared. zgs's re-call (round thirteen) is due about 11:50.
  If it is PROMISING, its combination with zrm (zrg, by the amendment of 10:15) is declared next, as round twenty-one.
- **Speed.** The objective changes training only. zrk reads and serves as zrm does, so any latency figure for it is
  zrm's, and cold (8216ffe): each question timed from scratch, with the walk, the move of its entries to the device and
  the forward.

### scr-zgs (L-musique): MIXED. scr-zgs-hp (L-hotpotqa): PROMISING. The pair is MIXED and its re-call PROMISING (11:47), so gsurg's loop is combined with zrm (round twenty-one)

`outputs/screen/scr-zgs.md` and `scr-zgs-hp.md`, `scr-zgs-pair.md` and `scr-zgs-pair-recall.md`. R@5 of zgs (gsurg's
loop on zret's model, the thirteenth round) minus zret's fit of the same split, called with the null's floors. Unmarked
reads are WITHIN. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | +0.0030 | +0.0009 | −0.0100 zs (LOSS as filed; floor 0.0720) | −0.0014 | +0.0005 | +0.0109 GAIN zs (floor 0.0102) |
| L-hotpotqa | +0.0078 (GAIN as filed; floor 0.0149) | −0.0010 | −0.0065 | −0.0053 zs | +0.0009 | +0.0032 zs |

- **One GAIN, just above its floor:** webqsp read zero-shot in L-musique's fit, +0.0109 R@5 (0.177 to 0.188) against
  a floor of 0.0102. hit@1 there barely moves (+0.0006).
- **No LOSS under the null.** musique read zero-shot in L-musique's fit falls −0.0100 R@5 (hit@1 −0.0165), a LOSS at
  the floor 0.0075 and WITHIN its floor of 0.0720. metaqa in L-hotpotqa's fit (+0.0078) turns from GAIN to WITHIN.
- **musique read in-domain in L-hotpotqa's fit falls −0.0065** (hit@1 −0.0182), under its floor of 0.0075.
- **The loop is slow.** zgs's L-hotpotqa fit took 2.1 hours (8 epochs of about 15.5 minutes), about 3.5 times
  lean_gpu's loop.
- **What follows.** By the amendment of 10:15, the full run against zret's fits does not run (`fzg-gate-b` starts
  nothing). gsurg's loop is combined with zrm, the base, in round twenty-one, declared next before its numbers.

### Twenty-first round: gsurg's loop on zrm (declared 8 October about 11:55, before any of its numbers; full run docs/FULL_ROUND21.md)

Code: `outputs/mp_unified/zrg.py` (its selftest passes).

- **Why.** zgs's screen (round thirteen: gsurg's loop on zret's model) re-called PROMISING at 11:47 (the section
  above). By the amendment of 10:15 ('Amendment: with zrm the base'), a PROMISING re-call of zgs leads to a combination
  with zrm, the base, declared before its numbers. This is that round.
- **What trains.** The arm zrg: zrm's model, carve, settings and training (rmatch.py's train), run inside lean_screen5's
  gsurg loop, seed 0.
  - At each step, the gradient on each training dataset's questions loses its component along the gradient of every
    other dataset in the step that it conflicts with (lean_screen5.fit_gsurg, as gsurg and zgs).
  - Each epoch's surgery counts land in train.json's curve. A fit whose curve lacks them trained outside the loop and
    stops (exit 1).
- **The same loop on every training question, on every dataset.** No new column, block, carve, graph, model or
  hyperparameter. Reads and serving are zrm's. On L-metaqa's fit (no typed graph in training) zrm is zret's model bit
  for bit, so there zrg is zgs's model and loop.
- **The screen's two fits:** `scr-zrg` (L-musique) and `scr-zrg-hp` (L-hotpotqa). Each is read on the six s1eval carves,
  as zrm's fits were.
- **Decided against zrm's screen fits** (`scr-zrm` and `scr-zrm-hp`), with zgs's screen fits and step 1's beside. The
  pieces are zrg.py's: the comparison, the pair, and the re-call under the seed null (relz.py's, with zrc.py's mapping
  to zrm's fits). Each read's base R@5 comes from `outputs/zrc/base-zrm-<split>.json`.
- **What follows.** The full run (docs/FULL_ROUND21.md) starts only if both hold: the re-call is PROMISING (a GAIN and
  no LOSS among
  its twelve reads), and zrm is the base (docs/BASE_ZRM_ZRS.md re-graded ADOPT, as it is since 10:05).
- **Caps and order.**
  - Each fit takes 0.30 of the card (share 0.32) and 9 GB. zgs's L-hotpotqa fit in the loop reserved 5.5 GB of its
    6.2 GB cap (0.26), and zrm's chain entries add to the loop's batches. The full run's L-metaqa fit takes 0.18
    (share 0.20), as gsurg's did.
  - Each read takes 0.30 (share 0.32) and 6 GB.
  - The screen's fits go after round eighteen's full-run items, which are under way, and ahead of the later full runs'
    fits (rounds nineteen and twenty). They take about two hours each, so ahead of round eighteen's remaining fits they
    would delay its grade by about an hour. The full run goes at the end of the list. Nothing is preempted.
- **ETAs.** The fits start once round eighteen's last fits have started, about 12:30 to 13:15, and take about 2 to 2.5
  hours each (zgs's L-hotpotqa fit took 2.1 hours: 8 epochs of about 15.5 minutes). The re-call lands about 15:00 to
  16:00.
- **Speed.** The loop changes training only (about 3.5 times lean_gpu's time per epoch, since each step takes every
  other present dataset's gradient). zrg reads and serves as zrm does, so any latency figure for it is zrm's, and cold
  (8216ffe): each question timed from scratch, with the walk, the move of its entries to the device and the forward.

### scr-zkind (L-musique): PROMISING. scr-zkind-hp (L-hotpotqa): MIXED. The pair is MIXED and its re-call PROMISING (11:51), so round nineteen's full run starts

`outputs/screen/scr-zkind.md` and `scr-zkind-hp.md`, `scr-zkind-pair.md` and `scr-zkind-pair-recall.md`. R@5 of zkind
(zrm with a head for graphs with typed relations) minus zrm's fit of the same split, called with the null's floors.
Unmarked reads are WITHIN. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | −0.0003 | −0.0001 | +0.0100 zs (GAIN as filed; floor 0.0720) | −0.0012 | −0.0022 | +0.0168 GAIN zs (floor 0.0102) |
| L-hotpotqa | −0.0022 | +0.0024 | +0.0078 GAIN (floor 0.0075) | −0.0113 zs (LOSS as filed; floor 0.0200) | +0.0007 | +0.0591 GAIN zs (floor 0.0233) |

- **webqsp read zero-shot GAINs in both fits, and its hit@1 rises with it.** R@5 rises from 0.313 to 0.330 in
  L-musique's fit and from 0.242 to 0.301 in L-hotpotqa's; hit@1 rises +0.021 and +0.049 (zrc's fell, −0.053 and
  −0.016).
- **musique read in-domain in L-hotpotqa's fit GAINs:** +0.0078 R@5 against a floor of 0.0075, FC@5 +0.0132, hit@1
  −0.0108.
- **hotpotqa read zero-shot in L-hotpotqa's fit falls −0.0113 R@5** (FC@5 −0.0245): a LOSS at the floor 0.0075 and
  WITHIN its floor of 0.0200. The full run reuses this fit, and its re-grade under the null decides.
- **musique read zero-shot in L-musique's fit rises +0.0100 R@5, but its hit@1 falls −0.0381.**
- **The untyped reads move although the head acts on typed batches only.** The offsets train on metaqa's batches, so
  the shared weights take a different path. Elsewhere every read moves by 0.0024 or less.
- **What follows.**
  - `fkd-gate` passes: the re-call is PROMISING, and zrm is the base (ADOPT, 10:05).
  - The full run (docs/FULL_ROUND19.md) trains L-2wiki, L-squad, J5 and L-metaqa, decided against zrm's fits and
    re-graded under the null over every split.
  - L-metaqa's fit trains no typed graph, so its six reads must be zrm's bit for bit, or the round stops.
  - Its items sit behind round twenty-one's two screen fits (zrg, about two hours each), as declared at 11:55.
  - Any speed figure for this arm is cold (8216ffe).
- **ETAs.** The four fits start as the card frees, about 13:30 to 15:00, and take about 35 to 60 minutes each. The grade
  lands about 16:00 to 17:30.

### Twenty-second round: a row's link to the pool's leading rows, on zrm (declared 8 October about 18:15, before any of its numbers; full run docs/FULL_ROUND22.md)

Code: `outputs/mp_unified/zlink.py` (its selftest passes).

- **Why.** docs/DIAG_BRIDGE.md, read together:
  - **D1, REACHABLE_NOT_RANKED.** In partly-found questions, the golds zrm misses sit next to a found gold in the
    question's own pool graph: 2wiki 0.77, hotpotqa 0.92 to 0.93, musique 0.38 (0.77 within two hops). That is 6 to
    14 times the rate of non-gold rows. No fixed re-rank uses it.
  - **D2, FEATURE_LIMIT.** Boosted trees on zrm's per-row inputs plus zrm's own score add at most +0.006 R@5. The gap
    is in the inputs, not the scorer.
  - **What the inputs lack.** Every per-row input is computed before any row is scored. The graph inputs (WALK, WALKF,
    SEED, DISTS) tie a row to the retrieval's fixed seeds. None ties it to the rows the scorer itself ranks highly.
- **What trains.** The arm zlk: zrm's model, settings and training (rmatch.py's train), seed 0, plus a small head added
  to zrm's score.
  - zrm scores the pool. Its top five rows and its top row, ranked in lean_gpu.top_hit's order under no_grad, are the
    question's leaders.
  - Each row gets 9 inputs:
    - its edges from the top five, per edge family (3);
    - its edges from the top row, per family (3);
    - its two-step paths from the top five over any family (1);
    - its top-five flag (1);
    - the z-score of zrm's score in its pool (1, detached).
    The counts enter as log1p.
  - The head is Linear(9, 32), GELU, Linear(32, 1), 353 weights. Its last layer starts at zero and its first is drawn
    from its own generator, so at the start zlk's forward is zrm's bit for bit and no draw of zrm's moves (selftest;
    `zlink.py check` on real carves).
- **The edges.** Each question's pool graph as the look's chunks already hold it (structural, NER and kNN on the
  passage graphs; the KB's relation edges on metaqa and webqsp). They are read undirected: both directions, once per
  family, no self-loops. `zlink.py build` caches them per step-1 part, tied to its record (`outputs/zlink/cache`).
- **The same rule on all six datasets.** No new graph, column, text, encoder or model. The leaders come from zrm's own
  scores, so the inputs are query-local and label-free, the same in training and at read.
- **The screen's two fits:** `scr-zlk` (L-musique) and `scr-zlk-hp` (L-hotpotqa), each read on the six s1eval carves.
- **Decided against zrm's screen fits** (`scr-zrm` and `scr-zrm-hp`), with zret's screen fits and step 1's beside.
  zlink.py's comparison, pair and re-call under the seed null (relz.py's, with zrc.py's mapping to zrm's fits) decide.
  Each read's base R@5 comes from `outputs/zrc/base-zrm-<split>.json`.
- **What follows.** The full run (docs/FULL_ROUND22.md) starts only if both hold:
  - the re-call is PROMISING (a GAIN and no LOSS among its twelve reads);
  - zrm is the base (docs/BASE_ZRM_ZRS.md re-graded ADOPT, as it is since 10:05).
- **Checks before the fits.** The eleven edge builds (the five training datasets' fit carves and the six s1eval
  carves) are host CPU items. `zlk-check-hotpotqa` and `zlk-check-webqsp` load a passage and a KB carve with their
  edges, and require zlk's forward at the start to equal zrm's, in eval and in training with the same dropout draws.
  The fits wait on both.
- **Caps and order.**
  - Each fit takes 0.30 of the card (share 0.32) and 11 GB; each read 0.30 (share 0.32) and 8 GB. That is zrm's, plus
    the edges in host memory (about 4 to 10 per row).
  - The screen's GPU items go ahead of every waiting GPU item but round twenty's two reads (the user's priority), and
    behind the fits already running. The full run goes at the end of the list. Nothing is preempted.
- **ETAs.** The builds and checks take about 15 minutes. The fits start when the card has room, about 18:45 to 19:30,
  and take about 45 to 60 minutes each. The re-call lands about 20:15 to 21:00.
- **Speed.** Per question, one sort of its pool's scores and one pass over its pool's edges, after zrm's forward. Any
  latency figure for zlk is cold (8216ffe): each question timed from scratch, with the walk, the move of its entries
  and edges to the device and the forward, no warm-up pass and nothing kept from an earlier question.

### scr-zrk (L-musique): NO_GAIN. scr-zrk-hp (L-hotpotqa): NO_GAIN. The pair and its re-call are NO_GAIN (19:39), so zrk is dropped

`outputs/screen/scr-zrk.md` and `scr-zrk-hp.md`, `scr-zrk-pair.md` and `scr-zrk-pair-recall.md`. R@5 of zrk (zrm with a
smooth recall at five added to listwiseD, the twentieth round) minus zrm's screen fit of the same split, called with
the null's floors. Every read is WITHIN. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | −0.0003 | +0.0002 | +0.0007 zs | +0.0005 | +0.0015 | +0.0069 zs (floor 0.0102) |
| L-hotpotqa | +0.0006 | +0.0014 | +0.0003 | −0.0041 zs (floor 0.0200) | +0.0004 | +0.0016 zs |

- **The objective moves nothing.** Eleven of the twelve reads move by 0.0041 R@5 or less. The largest move is webqsp
  read zero-shot in L-musique's fit, +0.0069 (0.313 to 0.320), under its floor of 0.0102; its hit@1 falls −0.0153.
- **FC@5 does not move either**, though the loss was aimed at the questions found only in part: 2wiki +0.0042 and
  +0.0024, musique −0.0033 zs and −0.0054. hotpotqa read zero-shot in L-hotpotqa's fit falls −0.0108 FC@5
  (−0.0041 R@5).
- **What this says.** Re-weighting the gradient toward the golds outside the top five changes no ranking that zrm's
  listwise loss had not already set. With D2 (FEATURE_LIMIT), the missing R@5 is not in how zrm is trained on its
  inputs but in the inputs.
- **What follows.** `frk-gate` exits 1, so the feeder drops round twenty's full run (docs/FULL_ROUND20.md): its four
  fits, reads, comparisons and grades never run.

### scr-zlk (L-musique): NO_GAIN. scr-zlk-hp (L-hotpotqa): PROMISING. The pair and its re-call are PROMISING (21:49); flk-gate decides the full run

`outputs/screen/scr-zlk-pair-recall.md`. **Message passing (the GNN track):** zrm plus a head over each row's links to
zrm's own leading rows (one propagation step). R@5 of zlk minus zrm's screen fit of the same split, called with the
null's floors. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | −0.0044 | −0.0007 | −0.0027 zs | +0.0001 | −0.0041 | +0.0083 zs (floor 0.0102) |
| L-hotpotqa | −0.0014 | +0.0019 | **+0.0084 GAIN** | +0.0011 zs | +0.0047 | +0.0055 zs |

- **One GAIN, no LOSS.** musique in L-hotpotqa's fit, 0.560 to 0.568 [+0.003, +0.014], the 2-hop bridge D1 pointed
  at (missed golds linked to found ones).
- **L-musique's fit leans down in-domain:** metaqa −0.0044 and 2wiki −0.0041, both inside the floor of 0.0075.
  Both webqsp reads rise (+0.0083, +0.0055 zs), under their floors.
- **What follows.** flk-gate waits for zb-grade-nullx as declared, then starts docs/FULL_ROUND22.md's full run. Its
  numbers are the GNN track's, never the MLP's.

### Twenty-third round: each row's neighbours' scores, on zrm, on the host's CPU (declared 8 October about 20:05, before any of its numbers; full run docs/FULL_ROUND23.md)

Code: `outputs/mp_unified/zprop.py` (its selftest passes).

- **Why.** The same reading of docs/DIAG_BRIDGE.md as round twenty-two, from the other side:
  - **D1.** The golds zrm misses in partly-found questions sit next to rows it ranks highly (2wiki 0.77, hotpotqa 0.92
    to 0.93, musique 0.38 and 0.77 within two hops), but zrm's score does not single them out among those rows'
    neighbours (musique median rank 5 to 9 of about 45).
  - **D2.** A stronger scorer on the per-row inputs gains nothing (FEATURE_LIMIT); what they lack is relational.
  - **Round twenty-two** gives each row its edges to zrm's top five and top row: a cut at five. When the found gold is
    ranked sixth or tenth, its neighbours get nothing from it. This round gives each row every neighbour's score, and
    the head learns how much a neighbour's rank counts.
- **What trains.** The arm zsp: zrm's model, settings and training (rmatch.py's train), seed 0, plus a small head added
  to zrm's score.
  - zrm scores the pool. Its scores are z-scored within each pool under no_grad.
  - Each row gets 10 inputs:
    - per edge family, the mean of its neighbours' z-scores (3);
    - per family, their soft maximum, the log of the sum of their exponentials, each z-score clipped to [-8, 8] (3);
    - per family, log1p of its degree (3);
    - its own z-score (1).
    A row with no neighbour in a family reads 0 there.
  - The head is Linear(10, 32), GELU, Linear(32, 1), 385 weights. Its last layer starts at zero and its first is drawn
    from its own generator, so at the start zsp's forward is zrm's bit for bit and no draw of zrm's moves (selftest;
    `zprop.py check` on real carves).
- **The edges are round twenty-two's** (`outputs/zlink/cache`, built and checked at 18:15): each question's own pool
  graph, undirected, once per family, no self-loops. Nothing new is built.
- **The same rule on all six datasets.** No new graph, column, text, encoder or model. The inputs come from zrm's own
  scores, so they are query-local and label-free, the same in training and at read.
- **On the host's CPU, against zrm on the CPU.** The card holds about half of what it did, so its queue runs one fit at
  a time while the CPU idles. This round runs on the CPU beside it.
  - Four fits, on 6 threads each: zsp's `scr-zsp` (L-musique) and `scr-zsp-hp` (L-hotpotqa), and zrm's refits on the
    same splits, `scr-zrm-cpu` and `scr-zrm-cpu-hp` (`zprop.py base`: zrm.py's train with `--device cpu`). Each fit is
    read on the six s1eval carves on the CPU.
  - **Each read is decided against zrm's CPU fit of its split**, so both sides ran on the same device. zrm's card fits
    (`scr-zrm`, `scr-zrm-hp`) and step 1's are reported beside.
  - Each read's base R@5 for the re-call comes from `outputs/zprop/base-zrm-cpu-<split>.json`: zrm's CPU fit compared
    with step 1's (the null's seed 0). The same comparison reports zrm's CPU fit against its card fit: the device's
    own spread. That is reported only and decides nothing.
  - zprop.py refuses to train or read zsp, or train zrm's base, on any device but the CPU, and its pair and grade
    refuse a comparison whose new fit or deciding base is not a CPU fit (their screen.json stamps).
- **The rule is the screens' rule.** zprop.py's comparison, pair and re-call under the seed null (relz.py's, mapped to
  zrm's CPU fits) decide. The full run (docs/FULL_ROUND23.md) starts only if both hold:
  - the re-call is PROMISING (a GAIN and no LOSS among its twelve reads);
  - zrm is the base (docs/BASE_ZRM_ZRS.md re-graded ADOPT, as it is since 10:05).
- **If rounds twenty-two and twenty-three both pass.** Each full run decides its own arm against zrm. Which becomes the
  base, or whether the two combine, is decided in a file declared before either grade, as docs/BASE_ZRM_ZRS.md was.
- **Checks before the fits.** `zsp-check-hotpotqa` and `zsp-check-webqsp` load a passage and a KB carve with their
  edges, and require zsp's forward at the start to equal zrm's, in eval and in training with the same dropout draws, and
  every input to be finite. The fits wait on both.
- **Caps and order.** Each fit and read takes 6 CPUs and 16 GB of host memory; no card. The four fits start together,
  beside the card's queue, and take nothing from it.
- **ETAs.** The checks take about 5 minutes. A CPU fit's pace is not yet measured: the first epoch's time gives it, and
  the ETAs are restated then. At 3 to 6 times the card's 6.5 minutes per epoch, a fit takes about 2.5 to 5 hours, so
  the re-call lands about 23:30 to 03:00.
- **Speed.** Per question, one pass over its pool's edges after zrm's forward. Any latency figure for zsp is cold
  (8216ffe): each question timed from scratch, with the walk, the move of its entries and edges to the device and the
  forward, no warm-up pass and nothing kept from an earlier question.

### scr-zsp (L-musique): NO_GAIN. scr-zsp-hp (L-hotpotqa): PROMISING. The pair and its re-call are PROMISING (21:43); the full run started

`outputs/screen/scr-zsp-pair-recall.md`. **Message passing (the GNN track):** zrm plus one propagation step over the
pool graph. R@5 of zsp minus zrm refit on the host's CPU (scr-zrm-cpu, scr-zrm-cpu-hp), called with the null's floors.
zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | +0.0006 | −0.0018 | +0.0004 zs | +0.0013 | +0.0052 | +0.0035 zs |
| L-hotpotqa | +0.0013 | +0.0035 | +0.0034 | −0.0074 zs (floor 0.0200) | **+0.0100 GAIN** | −0.0030 zs |

- **One GAIN, no LOSS.** 2wiki in L-hotpotqa's fit, 0.866 to 0.876 [+0.008, +0.012]. 2wiki also rises in L-musique's
  fit, +0.0052 [+0.003, +0.007], under its floor of 0.0075. These are the passage graphs where D1 found most missed
  golds linked to found ones (2wiki 0.77).
- **Zero-shot is flat.** hotpotqa read zero-shot falls −0.0074, within its floor of 0.0200. webqsp moves by 0.0035 or
  less.
- **What follows.** `fsp-gate` exits 0, so the full run of docs/FULL_ROUND23.md started at 21:45, on the host's CPU.
  Its numbers are the GNN track's, never the MLP's.

### Label: rounds twenty-two and twenty-three are message passing (8 October about 20:20)

- zlk (round twenty-two) and zsp (round twenty-three) add one propagation step after zrm's forward. Each row
  aggregates from its neighbours in the question's own pool graph: zlk counts its edges from zrm's top rows, and zsp
  pools its neighbours' scores. The step has no learned weights and passes no gradient, but it runs over the edges at
  read time and uses other rows' scores. That is message passing (an MLP plus one propagation step, as Correct &
  Smooth is), not the MLP.
- They keep running as declared, and their rules and verdicts are unchanged. Every record, table and claim names them
  **"zrm + one propagation step (message passing)"**. Neither is ever called the MLP or non-MP, and neither is set
  against the GNN as the MLP.
- zrm stays the MLP's base. An ADOPT of zlk or zsp makes it the base of a hybrid track, not of the MLP's screens.
- The MLP's screens stay message-passing-free from here on. A row's inputs are computed before any row is scored,
  from the question, the row, the frozen features and the graph's fixed structure, and never from another row's
  score.

## Twenty-fourth round: feature selection for the MLP over all six datasets (zfs; declared 8 October about 20:45)

Declared before any of its numbers. `outputs/mp_unified/zfeat.py`; full run docs/FULL_ROUND24.md. **No message
passing:** every candidate is a column of the look, computed in preprocessing from the question, the row, the frozen
embeddings and the graph's fixed structure around the fixed seeds, before any row is scored. None reads another row's
score, and nothing passes over neighbours at read time.

**Why.** zrm's inputs are step 1's pick, and the pick came from one greedy on 2wiki's select carve (l3-2w, 2 October).
That greedy dropped the NER, kNN and all-family views of the structure, the all-family depth rings and the seed
similarities "at no select cost" on 2wiki alone. A later greedy with the edge-label block (l4-2w) took the NER and kNN
topologies, the all-family depth and DISTF back. D1 finds the golds zrm misses linked to the ones it finds, on musique
mostly over NER edges, and D2 finds that zrm's own inputs hold nothing more. No selection of zrm's inputs has looked at
all six datasets or at a dataset read zero-shot. The user (8 October): select "the best possible combinations of the
features that help both across all the datasets and also help in generalization".

**The candidates** (76 look columns in seven blocks the pick leaves out):

| block | columns | what |
| --- | --- | --- |
| topo_NER, topo_KNN, topo_FULL | 11 each | distances to the seeds, seeds within one and two hops, pool and global degree, walks and the seeds' component, in the NER, kNN and all-family views (topo_STRUCT, in the pick, is the structural view) |
| depth_FULL | 17 | depth_STRUCT's rings and seed counts over every edge family |
| seedcond | 6 | the row's cosine with the seeds' prototype, its nearest seed and the reached seeds |
| typed_rel, ordered | 10 each | the question's cosine with the relation text on the row's typed edges and along the best typed walk from a seed (0 on graphs without types) |

Left out as message passing, for the GNN track: nbr_agg (neighbours' frozen features pooled per family), gcs and
typed_v2 (retrieval scores moved over the pool graph).

**The screen, per split (L-musique and L-hotpotqa, two fits as every round):**

1. **Build.** The candidates' columns, part by part as step 1's cache holds each carve (fit, select and s1eval carves of
   the six datasets, 16 carves), from the same look chunks. Step 1's cached columns and pool sizes from the same
   chunks must equal its cache bit for bit.
2. **Train: zfs.** zrm's model, settings and training with the seven blocks after the pick's. Each enters as every block
   does: raw, within-pool z and keep flag. In training each (question, candidate block) is kept with probability 0.5,
   drawn from the model's own generator (bdrop20's form, on the candidates only). A masked block then reads as the
   model learned to read its absence. The pick's blocks are never dropped. Seed 0, p@swa, on the card.
3. **Select.** Both fits' p@swa are read on the five training datasets' select carves: four in-domain, the held-out one
   zero-shot. Every one of the 128 subsets of the candidates is kept in turn, the rest masked. Each subset's R@5 minus
   R@5 with every candidate masked gives ten reads.
   - The chosen subset has the largest mean of the ten with no read below −0.002. Ties go to fewer blocks.
   - Below a mean of +0.001 the choice is empty: NO_SELECTION, and the round stops.
   - The selection reads no evaluation carve. webqsp has no other carve, so it is graded but not selected on.
   - One subset serves both fits.
4. **Read.** Each fit on the six s1eval carves with the chosen subset kept and the other candidates masked.
5. **Compare** against zrm's screen fit of the split (scr-zrm, scr-zrm-hp; same card, same training), with zret's and
   step 1's beside. lean_screen's comparison, then the pair and the re-call under the seed null, as every round
   (relz.py under zfs's name, zrc.py's mapping).

**Verdict.** The re-call's, as every round: PROMISING is at least one GAIN and no LOSS among the twelve reads.
PROMISING opens the full run (docs/FULL_ROUND24.md, `ffs-gate`).

**What it changes and what it does not.**
- Nothing new is computed: the columns are the look's, already compiled for the twin and the GNN. No graph, text,
  encoder or embedding changes.
- The 0.5 block dropout on the candidates is the one change to training. zrm's settings are otherwise unchanged.
- No label, gold flag or dataset name enters any input. webqsp never trains. Test splits are never read.

**The GNN side.** The same seven blocks plus the three left out above are the GNN track's candidates, selected the same
way on the GNN in a round of its own.

*Amended before any of this round's numbers (8 October about 21:00):* the selection in step 3 is joint with the GNN's.
One subset serves zfs and zgn: the largest smaller of the two models' mean gains, admissible for both (twenty-fifth
round). The GNN's candidates are these seven, so that the two models read the same features. nbr_agg, gcs and typed_v2
stay out of both.

**Order and ETAs (8 October, about 20:45).**
- The 16 builds are CPU items (1 CPU, about 5 minutes each); they start now, beside round twenty-three's CPU fits.
- The two fits are card items placed straight after round twenty-two's screen, ahead of the full runs (rounds
  eighteen, nineteen and twenty-one). Round twenty-two's L-hotpotqa fit starts about 21:00. The zfs fits start
  when the card has room for them: from about 21:00 if they fit beside it, about 21:55 if not. A zrm fit on the card
  takes about 55 minutes.
- The selection takes about 15 minutes, the reads about 10, then compares, pair and re-call.
- The selection lands about 22:15 to 23:00, the re-call about 22:45 to 23:30.

### Tracks (8 October about 20:45)

The user: rounds twenty-two (zlk) and twenty-three (zsp) count as **GNN improvements**. They are the GNN track's,
which aims at the state of the art. The MLP's screens aim to come as close to the GNN as possible without message
passing; round twenty-four is the first. Each track's features are selected over all six datasets, including reads
zero-shot.

## Twenty-fifth round: feature selection for the GNN, with the MLP's features (zgn; declared 8 October about 21:00)

Declared before any of its numbers, and before any number of round twenty-four. `outputs/mp_unified/zgnn.py`. **The
GNN track. Message passing:** zgn passes each row's hidden state over its question's pool graph. It is never "the MLP"
and never "non-MP".

**Why.** The user (8 October about 20:55): "also declare the gnn feature selection round, because want both models to
have the same features". The MLP and the GNN should read one feature set, chosen on all six datasets and on zero-shot
reads, so that whatever separates them is the message passing and not their inputs.

**The model.** zgn is zfs's model (round twenty-four: zrm over the pick and the seven candidates, the candidates dropped
at 0.5 per question and block in training and masked at read outside the chosen subset) with two residual
message-passing layers between its hidden layers and its output:
- Each row takes the mean hidden state of its neighbours, separately per edge family (0 for a family with none).
- Then h ← h + GELU(W_t [a_0, a_1, a_2] + b_t), for t = 1, 2. This is GraphSAGE's mean form.
- The graph is round twenty-two's pool graph (zlink.py): the look's pool edges, undirected, once per family, with no
  self-loops. On the passage graphs the families are structural, NER and kNN; on metaqa and webqsp they are the KB's
  relation edges.
- W_t and b_t start at zero and draw nothing, and the messages have no dropout. At the start zgn's forward is zfs's,
  and every random draw of zfs's (initial weights, dropout, candidate masks) is also zgn's.
- 98,560 extra weights at H = 128.
- Settings and training are zrm's, on the card, seed 0, p@swa.

**The same features.** The candidates are round twenty-four's seven blocks. nbr_agg, gcs and typed_v2 stay out of both
models: they move frozen features or retrieval scores over the pool graph, and zgn moves its own hidden state, so it
needs none of them. The two models' inputs are therefore the same columns.

**Round twenty-four amended before any of its numbers.** The selection is joint: one subset for both models.
- Both models' p@swa are read on the same ten select reads: two fits, the five training datasets, in-domain on four and
  zero-shot on the held-out one. Every one of the 128 subsets is read, with the rest masked.
- The chosen subset has the largest smaller of the two models' mean gains, and is admissible for both: no read of
  either model below −0.002.
- Ties go to fewer blocks. Below +0.001 the choice is empty for both, NO_SELECTION, and both rounds stop.
- Each model's own best subset is reported beside, and is not used.
- zfs's screen, re-call and full run are otherwise as declared, read with the common subset.
- The amendment's code is in zfeat.py (`joint`, `select --gnn-fits`). The selftest passes.

**The screen, per split (L-musique and L-hotpotqa):**
1. **Build** the select carves' pool edges (zlink.py build; the fit and s1eval carves' edges exist from round
   twenty-two).
2. **Train** zgn: scr-zgn and scr-zgn-hp.
3. **Select** jointly with zfs (above).
4. **Read** each zgn fit on the six s1eval carves twice:
   - with the common subset kept;
   - with every candidate masked, into a work copy of the same fit (scr-zgn-none, scr-zgn-none-hp), so that the
     fit's own folder is never written by the second read.
5. **Compare** the subset read with the same fit's every-candidate-masked read; this decides. zfs's, zrm's and step
   1's beside. zgn against zfs, with the same features, shows what the message passing adds, and is reported only.
6. **Pair and re-call** under the seed null, as every round: relz.py's machinery, each read's base R@5 the masked
   read's (zgnn.py on_none; base compares of scr-zgn-none against step 1's fits, as zrc's base-zrm files).

**Verdict.** The re-call's: PROMISING is at least one GAIN and no LOSS among the twelve reads. A full run, if any, is
declared in its own file before any of its numbers.

**What it changes and what it does not.**
- No new graph, column, text, encoder or embedding. The edges are round twenty-two's, from the look.
- No label, gold flag or dataset name enters any input. webqsp never trains. Test splits are never read.
- Its numbers are the GNN track's. They are never cited as the MLP's.

**Order and ETAs (8 October, about 21:00).**
- The five edge builds are CPU items, about 5 minutes each; they start when the CPU cap frees, about 21:00 to 21:15.
- The two zgn fits are card items after zfs's fits, about an hour each. They start when the card has room, about 21:30
  to 22:30.
- The joint selection runs after all four fits (about 25 minutes), then the reads, about 23:30 to midnight.
- The re-calls of rounds twenty-four and twenty-five land about midnight to 00:45.

### Bases after round twenty-three's full run (9 October 00:52)

- **docs/FULL_ROUND23.md is ADOPT, and ADOPT under the seed null over all six splits.** Its two primary GAINs are both
  zero-shot: metaqa in L-metaqa's fit, +0.0303; 2wiki in L-2wiki's, +0.0141. No read of 36 LOSEs.
- **zsp (zrm + one propagation step, message passing) is the base of the GNN track's later screens and runs,** its
  fits the host CPU's, as FULL_ROUND23.md section 4 declares. Rounds twenty-two and twenty-five keep the bases they were
  declared on; a combination with zsp is a round of its own.
- **zrm stays the MLP's base.** zsp's numbers are never cited as the MLP's.

### Round twenty-five's fits: a bug fix (9 October about 01:00, before any of its numbers)

- **The failure.** Both zgn fits stopped at their first step. zgn's forward asked for chains on every carve, but the
  graphs without typed relations (squad, musique, hotpotqa, 2wiki) hold none.
- **The fix.** zgnn.py skips the chain match on a carve without chains, as rmatch.ChainMatch's forward does. The
  selftest now checks that a carve without chains runs and is zfs at the start, and that a carve without pool edges is
  refused. Nothing else changes.
- **The requeue.** The two fits and every item after them run under new names:
  - round twenty-four's select, reads, compares, pair, re-call and gate (`scr24b-*`, `ffs-gate-b`);
  - round twenty-five's (`scr25b-*`).

## Twenty-sixth round: a graph-shaped space, read without the graph, on zrm (zgf; declared 9 October about 02:05)

Declared before any of its numbers. Code: `outputs/mp_unified/zgf.py` (its selftest passes). **The MLP track:** at read
a row's score uses only the question, the row and their frozen vectors. No edge, neighbour, walk or other row's score
enters it. The graph is used only in training, as a loss.

- **Why.** The user (9 October): can the semantic space itself do the retrieval, through a new space built from the
  graph (offsets as in the parallelogram rule, transformations, relation attention), so that the graph is never needed;
  and can passages and KB share one representation, so that what is learned on one carries to the other?
  docs/DIAG_GAPS.md (first run, a1e441d) and docs/DIAG_BRIDGE.md give the target:
  - the MLP loses to the GNN track on partly-found and multi-gold questions (2wiki, hotpotqa) and on unseen graphs
    (metaqa one-hop golds +0.140 zero-shot);
  - no per-row input carries a row's place beside the rows it is linked to (D2, FEATURE_LIMIT).

  This round asks whether that place can be written into the vectors once, in training, so that at read a row's vector
  alone says where it sits.
- **What trains.** The arm zgf: zrm's model, settings and training (rmatch.py's train), seed 0, plus a space head added
  to zrm's score.
  - **One space for every graph.** The question's frozen 1536-wide embedding and the row's frozen 257-wide store vector
    (SEMB's two sides) are mapped by two learned matrices to a 64-wide space and set to unit length. The same maps serve
    all six graphs, passages and KB alike: one representation for both.
  - **Offsets (the parallelogram rule).** One learned offset per edge family (3). From the question, ten targets: the
    question itself, one step (each family's offset) and two steps (each pair's sum), each set to unit length.
  - **Relation attention.** The question's softmax over the ten targets.
  - **Three inputs a row**, z-scored in its question's pool: the attention-weighted cosine to the targets, the best
    cosine, and the plain cosine to the question.
  - **The head** is Linear(3, 16), GELU, Linear(16, 1), added to zrm's score; a question whose SEMB is masked reads zrm's
    score. In all, 115,675 weights, most of them the two maps.
  - **Start.** The last layer starts at zero and every other weight is drawn from the head's own generator (seed +
    2601), so at the start zgf's forward is zrm's bit for bit and no draw of zrm's moves (selftest; `zgf.py check` on
    real carves).
- **The graph, in training only.** Each batch's pool edges are round twenty-two's (`outputs/zlink/cache`): each question's
  own pool graph, undirected, once per family. Nothing new is built. For at most 20,000 edges a batch, drawn from the
  model's own generator, the row moved by its family's offset should sit nearer its neighbour than a row drawn from the
  same pool: a softplus margin on the cosines, temperature 0.1, weight 0.1 beside zrm's listwise loss. The loss reported
  per epoch stays zrm's. At read no edge is loaded into the score: the selftest requires the same scores with the edges
  removed.
- **The same rule on all six datasets.** No new graph, column, text, encoder or model; the encoder stays frozen.
- **On the card, against zrm's card fits.** Two fits: `scr-zgf` (L-musique) and `scr-zgf-hp` (L-hotpotqa), each read on
  the six s1eval carves, with deterministic algorithms as every card fit. Each read is decided against zrm's card fit of
  its split (`scr-zrm`, `scr-zrm-hp`; zlink.py's mapping), with zret's and step 1's beside.
- **The rule is the screens' rule.** zgf.py's comparison, pair and re-call under the seed null (relz.py's, mapped to
  zrm's fits) decide. A PROMISING re-call (a GAIN and no LOSS among its twelve reads) earns a full run, declared in its
  own file before its numbers.
- **What this round does not test.** The KB's relations get no offset of their own here: the cached edges carry their
  family, not their relation. Offsets built from the relation's text (so a relation never seen still gets one) need the
  relation of each edge, and are a later round of their own. Nor does it read the space alone as a retriever (whether it
  reaches golds without zrm); that read is reported with the later round.
- **Checks before the fits.** `zgf-check-hotpotqa` and `zgf-check-webqsp` load a passage and a KB carve with their
  edges, and require zgf's forward at the start to equal zrm's, in eval and in training with the same dropout draws, and
  every input and the edge loss to be finite. They report beside, without deciding, the untrained space's z-scores for
  gold rows against the rest. The fits wait on both.
- **Caps and order.** The checks take 2 CPUs and 8 to 10 GB each and start at once. Each fit and read takes 1.1 CPUs,
  11 GB and 0.32 of the card. They queue after round twenty-five's card items.
- **ETAs.** The checks take about 5 minutes. zgn's fits hold the card until about 02:25 and 02:45; round twenty-five's
  reads follow. A zrm-sized card fit takes about an hour, so zgf's fits run about 02:45 to 04:00, the reads to about
  04:30, and the re-call lands about 04:30 to 05:00.
- **Speed.** Per question, two small matrix products (its pool's rows and its question to 64 wide) and ten cosines a
  row. No edge is read. Any latency figure for zgf is cold (8216ffe).

## Twenty-seventh round: the KB's relations as the space's helper (zgr; declared 9 October about 02:50)

Declared before any of its numbers, and before any number of round twenty-six. Code: `outputs/mp_unified/zgr.py` (its
selftest passes). **The MLP track:** at read a row's score uses the question, the row, their frozen vectors and the
names of the relations its question's pool holds (the graph's fixed structure, as zrm's own chain match reads them). No
edge, neighbour, walk mass or other row's score enters it.

- **Why.** Round twenty-six gives each edge family one offset, so the KB's relations share one. The user (9 October):
  "fix this too we should use that rel what we have as a helper". The KB graphs store each triple's relation, and each
  relation's name has a vector by the same frozen encoder (`outputs/m3b/relations`, the tables rmatch reads). zrm's
  chain carve already hands every KB batch its pools' relations and the typed chains from the question's seeds.
  Nothing new is built.
- **What trains.** The arm zgr: round twenty-six's zgf (zrm plus the shared 64-wide space and its ten family targets),
  plus relation offsets.
  - **An offset per relation, from its name.** A learned map of the relation's frozen name vector into the space. Every
    relation gets one through the same map, so a relation no fit saw (webqsp's 7,058, all read zero-shot) still gets
    one.
  - **The question moved along a relation**, either way: question plus or minus the offset, set to unit length.
  - **Relation attention.** Over the relations on the question's pool: a fixed text match (the question's cosine to
    the relation's name, its weight learned from 1) plus a learned term (starting at zero). The 16 highest are kept and
    softmaxed.
  - **Two more inputs a row:** the attention-weighted best cosine to the moved question, and the best over the kept
    relations. They join zgf's three, z-scored in the pool, into Linear(5, 16), GELU, Linear(16, 1). On a graph without
    typed relations (squad, musique, hotpotqa, 2wiki), or a pool with none, the two read 0.
  - **Start.** The last layer starts at zero and every new weight is drawn from zgr's own generator (seed + 2701), so at
    the start zgr's forward is zrm's bit for bit and no draw of zrm's moves.
- **The graph, in training only.**
  - zgf's family-edge loss stays.
  - The **chains** add one more: for at most 20,000 of the batch's typed chain entries (a row reached from the
    question's seeds by up to three typed steps), the question moved by the chain's relation offsets, forward plus and
    back minus, should sit nearer that row than a row drawn from the same pool. Same margin, temperature and weight as
    zgf's.
  - The chains are the KB's walks from the seeds, gold or not; no label enters.
  - At read no edge or chain entry is used: the selftest requires the same inputs with both removed.
- **The same rule on all six datasets.** No new graph, column, text, encoder or model; the encoder stays frozen. The
  relation vectors are the existing ones.
- **On the card, against zrm's card fits,** as round twenty-six:
  - two fits, `scr-zgr` (L-musique) and `scr-zgr-hp` (L-hotpotqa), each read on the six s1eval carves;
  - each read is decided against zrm's card fit of its split, with zret's and step 1's beside.
  - zgr against zgf (what the relations add) is reported from the two rounds' records and decides nothing.
- **The rule is the screens' rule.** zgr.py's comparison, pair and re-call under the seed null decide. A PROMISING
  re-call earns a full run declared in its own file.
- **Checks before the fits.** `zgr-check-metaqa` and `zgr-check-webqsp` (s1eval) require zgr at the start to equal zrm
  in eval and training, and every input and both losses to be finite. They report beside, without deciding, the
  untrained relation inputs for gold rows against the rest.
- **Caps and order.** The checks take 2 CPUs and 10 GB each. Each fit and read takes 1.1 CPUs, 11 GB and 0.32 of the
  card, queued after round twenty-six's.
- **ETAs.** The checks take about 5 minutes. zgf's fits hold the card from about 02:45 to 04:00; zgr's run about 04:00
  to 05:15, the reads to about 05:45, and the re-call lands about 06:00.
- **Speed.** Per question, its pool's relations mapped into the space, a sort of their logits, and 32 cosines a row. No
  edge is read. Any latency figure for zgr is cold (8216ffe).

## Rounds twenty-four and twenty-five: result (filed 9 October about 05:45)

**NO_SELECTION: both rounds stop, as declared** (`outputs/zfeat/select.json`, `select.md`; the placement block is
redacted). The joint selection over the 128 subsets of the seven candidate blocks found no subset whose smaller mean
gain (zfs's or zgn's, ten select reads each) reached +0.001.

- The best admissible subset (topo_KNN, depth_FULL, seedcond, typed_rel, ordered) gains +0.0007 for zfs and +0.0008
  for zgn. The best of all, without seedcond, gains +0.0008 but has a zgn read at -0.0020, so it is not admissible.
- On its own, zfs would choose depth_FULL, seedcond, typed_rel and ordered (+0.0012). zgn alone would choose nothing.
  Both are reported beside and are not used.
- Each block alone moves either model's mean by at most 0.0013 either way. The three topology blocks are at or below
  zero for both.
- What this says: the seven dropped look blocks hold nothing either model reads beyond the pick's blocks, on any of the
  five training datasets, in-domain or zero-shot. With the same inputs, zgn's message passing does not make a block
  useful that zfs cannot use. The MLP-to-GNN gap is not in these columns.
- No s1eval carve was read. The reads, comparisons, pair and re-call of both rounds were dropped with the selection, so
  zgn has no screen verdict. The select-carve R@5 in the record are selection numbers, not results.

## Round twenty-one: result (zrg; filed 9 October about 08:05)

**PROMISING under the seed null** (`outputs/screen/scr-zrg-pair-recall.md`; filed MIXED). The full run
(docs/FULL_ROUND21.md) is gated on it: `fzrg-gate` passed at 07:50 and its four fits are queued behind rounds
twenty-six and twenty-seven's screens.

- One GAIN of 12: musique, in-domain, in L-hotpotqa's fit, +0.0087 R@5 [+0.0032, +0.0142] against zrm's card fit.
- The filed LOSS, hotpotqa read zero-shot in L-hotpotqa's fit (-0.0094), is WITHIN under its null floor.
- L-musique's fit is NO_GAIN: every read is within 0.005 of zrm's, and musique read zero-shot is -0.0040.
- Against zgs's fits (reported only), L-musique's fit reads PROMISING. gsurg's loop on zrm keeps zrm's level and adds
  little on top of it at screen size.

## Twenty-eighth round: zrc's entries under zkind's head (zck; declared 9 October about 08:10)

Declared before any of its numbers. Code: `outputs/mp_unified/zck.py` (its selftest passes). **The MLP track:** zrc's and
zkind's inputs, no edge or neighbour read.

- **Why.** zrc (round eighteen) and zkind (round nineteen) were both ADOPTED against zrm, and neither beats the other
  (docs/BASE_ZRC_ZKIND.md, NOT_ADOPTED 08:01: zrc is the base). Both declarations name their combination as a round of
  its own. zrc changes which entries a typed row keeps (the 64 w0 ranks highest), and zkind the head that reads them (an
  offset on the output layer and rrf's base weight for a batch from a graph with typed relations). Both gain on webqsp
  read zero-shot. Whether the gains add is the question.
- **What trains.** The arm zck: zkind's model (`zkind.ZKind`) over zrc's builds (`zrc.ChainCarveZRC`, the entries
  `outputs/zrc/cache`). The offsets add no draw, so from the same seed zck is zrc's model and state plus three zero
  offsets, and its forward is zrc's bit for bit until an offset moves (the selftest requires it). No new column, block,
  carve, graph or hyperparameter; settings and training are zrm's (rmatch.py's train).
- **The same rule on all six datasets.** The offsets act only on a batch with chain entries (metaqa and webqsp); every
  other batch is zrc's. The encoder stays frozen. webqsp never trains.
- **On the card, against zrc's card fits** (the base): two fits, `scr-zck` (L-musique) and `scr-zck-hp` (L-hotpotqa), each
  read on the six s1eval carves. Each read is decided against zrc's card fit of its split (`scr-zrct`, `scr-zrct-hp`),
  with zrm's and step 1's beside. The re-call's base R@5 is zrc's against step 1's (`outputs/zbase2/base-zrc-S`, from
  docs/BASE_ZRC_ZKIND.md).
- **The rule is the screens' rule.** zck.py's comparison, pair and re-call under the seed null decide. PROMISING (at
  least one GAIN, no LOSS among the twelve reads) earns a full run declared in its own file.
- **Caps and order.** As round nineteen's: each fit 1.1 CPUs, 9 GB and 0.28 of the card; each read 0.32. Queued after
  round twenty-seven's screens, ahead of the full runs fzrg and flk.
- **ETAs.** zkind's fits took about 35 minutes each. zck's fits start when round twenty-six's or twenty-seven's free the
  card, about 09:00; the re-call lands about 10:30.
- **Speed.** zkind's one product per row on a typed graph, over zrc's entries. Any latency figure for zck is cold
  (8216ffe), timed in its own declared stage.

## Round twenty-six: result (zgf; filed 9 October about 09:05)

**NO_GAIN under the seed null** (`outputs/screen/scr-zgf-pair-recall.md`; filed NO_GAIN). No GAIN in 12 reads, and three
LOSSes, all in L-hotpotqa's fit. No full run.

- L-musique's fit (`scr-zgf`) is level with zrm's card fit: every read WITHIN, from -0.0035 (squad) to +0.0060 (musique
  read zero-shot) and +0.0043 (2wiki).
- L-hotpotqa's fit (`scr-zgf-hp`) loses on three passage graphs: musique in-domain -0.0599 [-0.0683, -0.0511], hotpotqa
  read zero-shot -0.0437, squad -0.0143. 2wiki is -0.0055, metaqa and webqsp level.
- Its training ran cleanly (loss falling each epoch to 1.286, no retry). Why the space head harms this fit and not
  L-musique's is not diagnosed here. zrm's card fit's training log is no longer on the host, so the two training losses
  cannot be compared.
- What this says for question three (a graph-free space shaped by the graph's edges): at screen size, a 64-wide space
  shared by passages and KB, read through a zero-started head, adds nothing over zrm's inputs, and on one fit it hurts.
- Round twenty-seven (zgr) is zgf plus relation offsets, so its reads are read beside these.

## Round twenty-eight: result (zck; filed 9 October about 10:15)

**MIXED under the seed null** (`outputs/screen/scr-zck-pair-recall.md`; filed MIXED, no call changed). Two GAINs and one
LOSS in 12 reads, against zrc's card fits. No full run (section 2: a MIXED screen does not earn one).

- **The gains add on webqsp read zero-shot.** L-musique's fit +0.0160 [+0.0042, +0.0270] (floor 0.0102), L-hotpotqa's
  +0.0402 [+0.0257, +0.0557] (floor 0.0233). zkind against zrc read -0.0141 to +0.0142 there (docs/BASE_ZRC_ZKIND.md),
  so zrc's entries under zkind's head gain where neither alone gains over the other.
- **The LOSS:** L-hotpotqa's fit on 2wiki in-domain, -0.0082 [-0.0103, -0.0061] against the floor 0.0075. L-musique's
  fit reads 2wiki -0.0041, WITHIN. 2wiki has no chain entries, so the offsets never act on its batches; the shift comes
  through the shared weights in joint training.
- Every other read is WITHIN: metaqa -0.0001 to +0.0000, squad +0.0011/+0.0015, musique +0.0039 (zero-shot) and +0.0030,
  hotpotqa +0.0000 and -0.0032 (zero-shot).
- zrc stays the base. A narrower screen that keeps the webqsp gain without the 2wiki loss would be declared here, before
  its numbers; none is declared now.

## Round twenty-seven: result (zgr; filed 9 October about 10:25)

**NO_GAIN under the seed null** (`outputs/screen/scr-zgr-pair-recall.md`; filed NO_GAIN, no call changed). No GAIN in 12
reads against zrm's card fits, and four LOSSes. No full run.

- **The KB relation offsets from relation-name vectors do not move the KB graphs.** metaqa reads +0.0034 and +0.0026,
  webqsp read zero-shot -0.0004 and -0.0065, all WITHIN. Beside round twenty-six's zgf (the same space with no
  relation offsets) they add at most +0.0011 on metaqa and nothing on webqsp.
- **L-musique's fit** loses on squad (-0.0118 [-0.0149, -0.0086]); hotpotqa -0.0063 and musique read zero-shot -0.0057
  are WITHIN. zgf's L-musique fit read squad -0.0035, so here the offsets cost passage graphs a little.
- **L-hotpotqa's fit** repeats zgf's harm on the passage graphs, smaller: musique in-domain -0.0474 (zgf -0.0599), hotpotqa
  read zero-shot -0.0263 (zgf -0.0437), squad -0.0141 (zgf -0.0143). Two rounds now show it, so the shared space read
  through its zero-started head harms L-hotpotqa's fit on passage graphs; it is not a single fit's accident. Its cause is
  still not diagnosed.
- What this says for question three: at screen size, neither the shared space nor relation offsets built from the KB's
  relation names add to zrm's inputs on any of the six datasets.

## Twenty-ninth round: the first stage's bridge rows as per-row inputs, on zrc (zbr; declared 9 October about 16:45)

Declared before any of its numbers. Code: `outputs/mp_unified/zbr.py` (its selftest passes). This is G1b of
docs/G1A_UNIVERSAL_BRIDGE_SIX.md. **The MLP track:** every input is fixed before any row is scored; no score of any row
enters another's (not message passing in section "rounds twenty-two and twenty-three are message passing").

- **Why.** G1a (65e66e1) applied B1d's training-free bridge rule, untuned, to all six datasets:
  - it gains over RRF on metaqa, hotpotqa, 2wiki and webqsp (+6.9 to +11.6);
  - it is level on musique;
  - it loses on squad (-2.2).

  A fixed rule gives up slots 4-5 on single-hop and comparison questions as well as on bridge questions. Its quantities
  are known before scoring, so a model can learn per question when a bridge row beats a first-stage row.
  docs/DIAG_BRIDGE.md's D1 found the missed golds next to found ones; D2 found the present inputs at their limit.
- **What trains.** The arm zbr: zrc's model (`zrm.ZRM` over `zrc.ChainCarveZRC`) plus a head over 13 per-row inputs
  added to its score. The inputs are computed from rrf's order in the pool (the batch's base_z) and zlink's pool edges
  (outputs/zlink/cache; undirected, once per family, no self-loops):
  - edges from rrf's top 1, top 2 and top 5, per family (9);
  - two-step paths from the top 5 (1);
  - the bridge flag: linked to rrf's top 2 and not one of them (1);
  - the bridge rank: 1/(1+rank by dense_cos among the flagged rows), which is G1a's rule's order over every family (1);
  - the source rank: 1/(1+the best rrf rank among the row's neighbours) (1).

  Head: Linear(13, 32), GELU, Linear(32, 1). The last layer starts at zero and the first comes from its own generator
  (seed + 2901). From the same seed zbr is zrc's state plus the head, and its forward is zrc's bit for bit (the
  selftest requires it). Settings and training are zrm's (rmatch.py's train).
- **The same rule on all six datasets.** zlink's edges exist for every carve; on the KB graphs they are the relation
  edges (families 0 and 2). No new graph, encoder, text or hyperparameter. webqsp never trains.
- **First, a carve check** (`zbr.py check`, hotpotqa and webqsp s1eval): identity against zrc at the start. Reported,
  not decided: the gold share of rows outside rrf's top five, split by bridge rank and unflagged. A check that is not
  IDENTICAL stops the round.
- **On the card, against zrc's card fits** (the base): two fits, `scr-zbr` (L-musique) and `scr-zbr-hp` (L-hotpotqa).
  Each is read on the six s1eval carves and decided against zrc's card fit of its split (`scr-zrct`, `scr-zrct-hp`),
  with zrm's and step 1's beside. The re-call's base R@5 is `outputs/zbase2/base-zrc-<split>`.
- **The rule is the screens' rule.** zbr.py's comparison, pair and re-call under the seed null decide. PROMISING (at
  least one GAIN, no LOSS among the twelve reads) earns a full run declared in its own file. The full run is the one that
  B1b would re-read under its own declared re-run.
- **Caps and order.** As round twenty-two's:
  - fits 1.1 CPUs, 11 GB and 0.32 of the card;
  - reads 1.1 CPUs, 8 GB and 0.32;
  - checks 2 CPUs and 8 GB.

  Queued after B1b's items, ahead of step 4e's block.
- **ETAs.** zlk's fits took about 40 minutes each, so the re-call lands about 2.5 hours after the card frees from B1b's
  reads.
- **Speed.** Per question, one sort of rrf, one sort of the flagged rows' dense_cos and three passes over the pool's
  edges, all before the forward. Any latency figure for zbr is cold (8216ffe), timed in its own declared stage.

## Thirtieth round: zsp's propagation repeated, on zsp (zdp; declared 9 October about 16:50)

Declared before any of its numbers. Code: `outputs/mp_unified/zdeep.py` (its selftest passes). **The GNN track:**
message passing over the pool's own graph, more steps.

- **Why.** zsp, the GNN track's base, propagates the neighbours' scores once. Its gains have come on 2-hop questions;
  musique's 3- and 4-hop slices gain nothing. A chain's third or fourth passage sits two or three edges from the first
  stage's rows. HippoRAG 2 (PPR) and GFM-RAG (NBFNet-style) propagate over many steps. Depth is the first lever that
  docs/PROGRAM_2026_10_09.md's stage G1 names, ahead of relation-conditioned messages, and Claim 2's musique gap is
  where it should show.
- **What trains.** The arm zdp (`zdeep.ZDeep`) runs zsp's forward (`zprop.ZProp`) and then two more steps. Each step adds
  a new head over `zprop.prop_inputs` of the score so far: per family, the neighbours' mean and soft-maximum z-score
  and the degree, plus the row's own z-score.
  - Each head is Linear(10, 32), GELU, Linear(32, 1). The last layer starts at zero; the first is drawn from its own
    generator (seed + 3001, + 3002).
  - From the same seed zdp is zsp's state plus 770 parameters, and its forward is zsp's bit for bit (the selftest
    requires it).
  - Settings and training are zrm's (rmatch.py's train).
- **The same rule on all six datasets,** on zlink's edges. No new graph, column, encoder or text. webqsp never trains.
- **On the host's CPU, against zsp's CPU screen fits** (`scr-zsp`, `scr-zsp-hp`): two fits, `scr-zdp` (L-musique) and
  `scr-zdp-hp` (L-hotpotqa), each read on the six s1eval carves. zrm's CPU fits and step 1's are beside. The re-call's
  base R@5 is zsp's screen fits against step 1's (`outputs/zdeep/base-zsp-<split>`, made first).
- **The rule is the screens' rule.** zdeep.py's comparison, pair and re-call under the seed null decide. PROMISING earns
  a full run declared in its own file. Reported beside, not decided: the musique hop slices.
- **Caps and order.** As round twenty-three's: 6 CPUs and 16 GB per fit and read. Queued after round twenty-nine's
  items, ahead of step 4e's block.
- **ETAs.** zsp's CPU fits took a few hours each. Three passes over the edges instead of one add perhaps 30%. The
  re-call lands this evening or overnight.
- **Speed.** Three passes over each question's pool edges after zrm's forward, instead of zsp's one. Any latency figure
  for zdp is cold (8216ffe), timed in its own declared stage.

## Round twenty-nine: result (zbr; filed 9 October about 19:15)

**NO_GAIN** (`outputs/screen/scr-zbr-pair-recall.md`; the seed-null re-call changed no call). All 12 reads are WITHIN,
against zrc's card fits. No full run.

- **musique**, the setting B1b names as the gap, moves least where it matters:
  - L-musique's fit, musique read zero-shot: +0.0053 [+0.0003, +0.0099];
  - L-hotpotqa's fit, musique in-domain: +0.0015 [-0.0042, +0.0071].
- **Every other read** lies between -0.0056 (webqsp zero-shot, L-musique) and +0.0042 (webqsp zero-shot, L-hotpotqa).
- **Reading.** The first stage's bridge rows, given to zrc as per-row inputs, add nothing the walks do not already
  carry. This agrees with D2 (FEATURE_LIMIT): the missed bridges are reachable but not separable from the inputs zrc
  has. The musique gap needs new information in the graph (the universal links, docs/U1A_*), not a re-description of
  the same lists.

## Round thirty: result (zdp; filed 10 October about 01:25)

**NO_GAIN** (`outputs/screen/scr-zdp-pair-recall.md`): the pair was filed MIXED. Under the seed null, L-hotpotqa's one
GAIN (webqsp zero-shot, +0.0173) falls WITHIN its floor (0.0233). L-musique keeps a LOSS on webqsp zero-shot (-0.0133
against a floor of 0.0102). No full run.

- **musique**, the setting depth was for, does not move: -0.0029 zero-shot (L-musique), -0.0071 in-domain
  (L-hotpotqa).
- **Reading.** Two more propagation steps over the same pool edges reach nothing new. The bridges musique misses are
  reachable, but the pool's edges do not single them out (D1, D2, round 29). The next lever is the graph itself: U1d's
  precise links (docs/U1D_PRECISE_LINKS.md), then step 4h's pools, then one retrain on both (the user, 10 October).

## Thirty-first round: a query-conditioned deeper GNN on zsp (zg1; declared 10 October about 01:55)

Declared before any of its numbers. Code: `outputs/mp_unified/zg1.py` (its selftest passes). **The GNN track, stage G1
of docs/PROGRAM_2026_10_09.md:** learned message passing over the pool's own graph, several layers.

- **Why.** Round thirty (zdp) repeated zsp's fixed summary of the neighbours' scores twice more and changed nothing, on
  musique included. A fixed summary of scores cannot carry a path; a learned state can. The golds zsp misses sit next
  to found rows (D1), and on musique's 3- and 4-hop questions two or three edges from them. HippoRAG 2 (PPR) and GFM-RAG
  (NBFNet-style) propagate learned or weighted signals over many steps; musique is Claim 2's gap (66.3 against 74.7).
- **What trains.** The arm zg1 (`zg1.ZG1`) runs zsp's forward (`zprop.ZProp`), then a 4-layer stack with 32-wide row
  states, NBFNet-style:
  - the boundary: each row starts from `zprop.prop_inputs` of zsp's score (10 per row, under no_grad as zsp's). The
    question enters through zsp's query-conditioned score;
  - each layer: m = W_self h + the sum over families of W_f times the neighbours' mean state + w_z z + b, with z the
    row's own z-score fed again at every layer; then h = h + GELU(layer_norm(m));
  - the output Linear(32, 1) starts at zero and is added to zsp's score.
  - All draws come from the module's own generator (seed + 3101). From the same seed zg1 is zsp's state plus 17,025
    parameters, and its forward is zsp's bit for bit (the selftest requires it). Settings and training are zrm's
    (rmatch.py's train).
- **The same rule on all six datasets,** on zlink's edges. No new graph, column, encoder or text. webqsp never trains.
- **On the host's card, against zsp on the card.** zsp's fits so far were trained on the CPU. So this round first
  refits zsp on the card on both splits (`scr-zspg`, `scr-zspg-hp`), then trains `scr-zg1` (L-musique) and
  `scr-zg1-hp` (L-hotpotqa). Each is read on the six s1eval carves and decided against zsp's card fit of its split.
  zsp's CPU screen fits and step 1's are beside. The re-call's base R@5 is zsp's card fits against step 1's
  (`outputs/zg1/base-zspg-<split>`).
- **The rule is the screens' rule.** zg1.py's comparison, pair and re-call under the seed null decide. PROMISING (at
  least one GAIN and no LOSS among the twelve reads) earns a full run, declared in its own file. Reported beside, not
  decided: the musique hop slices.
- **Caps and order.** GPU items under cuda_alloc (frac 0.45, so two share the card), 2 CPUs and 16 GB each.
- **ETAs.** zrm-sized fits take minutes on the card; four fits and four reads should land within about two hours.
- **Speed.** Four passes over each question's pool edges with 32-wide states after zsp's forward. Any latency figure for
  zg1 is cold (8216ffe), timed in its own declared stage.

## Round thirty-one: result (zg1; filed 10 October about 03:35)

**NO_GAIN** (`outputs/screen/scr-zg1-pair-recall.md`): the pair was filed MIXED. Under the seed null, L-musique's one
GAIN (musique zero-shot, +0.0077 [+0.0017, +0.0136]) falls WITHIN its floor (0.0720). L-musique keeps a LOSS on webqsp
zero-shot (-0.0156 against a floor of 0.0102). No full run.

- **2wiki in-domain moves on both fits**, just under its floor: +0.0062 [+0.0043, +0.0082] (L-musique) and +0.0072
  [+0.0048, +0.0094] (L-hotpotqa), against 0.0075. Every other read lies between -0.0024 and +0.0050.
- **musique** in-domain (L-hotpotqa): +0.0030 [-0.0027, +0.0090].
- **Reading.** A learned four-layer state over the pool's own edges finds a little more on 2wiki, whose pools carry
  dense links, and nothing on musique, whose bridges the pool's edges do not single out (D1, D2, rounds 29 and 30).
  Three rounds now agree: on these graphs, the GNN track gains only with new information in the graph. U1d's links and
  step 4h's pools come next, then U1c's retrain of zrc and zsp on both.

## Thirty-second round: a query-conditioned offset from the leading rows, on zrc (zof; declared 10 October about 03:35)

Declared before any of its numbers. Code: `outputs/mp_unified/zof.py` (its selftest passes, including a repeat fit
identical bit for bit). **The MLP track, stage S1 of docs/PROGRAM_2026_10_09.md:** an offset space learned end to end.

- **Why.** D1 found the golds zrc misses linked to rows it found, and D2 found zrc's per-row inputs at their limit.
  Rounds 29 to 31 counted the leading rows' edges (zbr) or propagated scores over them (zdp, zg1). None of them asks
  what the next hop's text should look like, given the question and a row already found. An offset space does: a
  relation, set by the question, that maps a found row onto the row it leads to (S1). The old offset screen
  (docs/OFFSET_SCREEN_RESULTS.md) found such translations sharpen the top ranks; here the offset is an input to zrc, not
  a replacement for it.
- **What trains.** The arm zof (`zof.ZOff`) is zrc's model (zrm.ZRM over zrc.ChainCarveZRC) plus a head added to its
  score:
  - **anchors:** each question's top 5 rows by rrf (the batch's base_z, ranked by zlink.pool_rank). No row's score
    enters and no edge is read, so the head is not message passing; it reads fixed rows, as SEED and DISTS do;
  - **offset scores:** 4 channels, each 16 wide, in DistMult's form: 16 * sum(norm(P_anchor Wa) * norm(q Wq) *
    norm(P_row Wt)). P is the row's decoded store vector (as SEMB reads it), q the question's embedding. The encoder's
    vectors are read, never changed;
  - **the head's inputs (25 per row):** the score against each anchor in rrf's order (0 for the row itself and a
    missing anchor), each channel's maximum, and whether the row is an anchor;
  - **head:** Linear(25, 32), GELU, Linear(32, 1). The last layer starts at zero, and every other draw comes from the
    module's own generator (seed + 3201). From the same seed zof's forward is zrc's bit for bit (the selftest requires
    it). Settings and training are zrm's (rmatch.py's train).
- **The same rule on all six datasets.** No new graph, column, encoder, text or edge. webqsp never trains.
- **On the host's card, against zrc's fits** (scr-zrct, scr-zrct-hp; the re-call's base is outputs/zbase2), as round 29
  was decided. zof trains `scr-zof` (L-musique) and `scr-zof-hp` (L-hotpotqa), each read on the six s1eval carves.
- **The rule is the screens' rule.** zof.py's comparison, pair and re-call under the seed null decide. PROMISING (at
  least one GAIN and no LOSS among the twelve reads) earns a full run, declared in its own file.
- **Caps and order.** GPU items under cuda_alloc (frac 0.3, so both fits share the card), queued ahead of U1d's block
  so the link shards' memory hold does not keep the card idle.
- **ETAs.** Round 29's fits of the same size took under an hour each on the card; the round should land within about
  two hours.
- **Speed.** Per row, 5 x 4 x 16 products against fixed rows, query-local. Any latency figure for zof is cold
  (8216ffe), timed in its own declared stage.

## Round thirty-two: result (zof; filed 10 October about 04:20)

**NO_GAIN, and it harms.** Both fits are NO_GAIN, filed and under the seed null. No read is a GAIN and 7 of the 12 are
a LOSS (outputs/screen/scr-zof-pair{,-recall}.{md,json}).

| fit | read | delta R@5 [95% CI] | re-call |
| --- | --- | --- | --- |
| L-hotpotqa | musique in-domain | −0.087 [−0.096, −0.078] | LOSS |
| L-hotpotqa | hotpotqa zero-shot | −0.049 [−0.054, −0.044] | LOSS |
| both | squad in-domain | −0.017 | LOSS |
| L-hotpotqa | webqsp zero-shot | −0.025 | LOSS |
| L-hotpotqa | 2wiki in-domain | −0.012 | LOSS |
| L-musique | hotpotqa in-domain | −0.008 | LOSS |
| both | metaqa in-domain | +0.004 to +0.005 | under its floor |
| L-musique | musique zero-shot | −0.038 | under its 0.072 floor |

What this says:
- zof starts as zrc bit for bit, but it trains jointly, so the base moves along with the head (as rmatch's did).
- A query-conditioned offset against rrf's top five rows does not add what zrc misses. It costs most where the
  anchors are least often gold (musique, webqsp).
- The two-hop "translate from the leading rows" signal that the offset spaces sharpened in docs/OFFSET_SCREEN_RESULTS.md
  does not carry over as an input on these graphs.

No full run follows. A GNN-track counterpart of zof is not queued.
