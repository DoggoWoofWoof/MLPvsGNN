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
  - **Order:** the four null fits queue behind round eight's screen and ahead of every full run. They take about 25 to
    35 minutes each; the null lands about 01:00 to 01:30.
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
