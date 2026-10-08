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
