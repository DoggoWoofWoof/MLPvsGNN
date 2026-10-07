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

**Step 2, read as a screen.** Step 2's eval-mix weighted fits of J5, L-metaqa and L-squad trained before the rule. Its
other three fits (L-musique, L-hotpotqa, L-2wiki) were cancelled at about 14:00, so step 2's declared grade does not
run. Each of the three is compared with step 1's fit of the same split by this rule (`outputs/screen/s2-*.md`).

**Held at about 14:40:**

- Every queued seed-1 and seed-2 run of parts 9 to 15, and the grades that need them (112 items).
- Step 4c's six fits; scr-pools replaces them.
- The seed-0 GNN runs of parts 10 and 11 (29 items), until the screens' reads are sent.

cs26-g-k6w-s0 was cancelled at 14:30 so that the screens get the card.

## 5. ETAs (7 October)

- **The four arm fits** started 14:25 to 14:27. An epoch takes 154 to 168 s, so each fit finishes about 14:50 to 14:55.
- **The null check and the pools fit** start as the card frees, about 14:52. The null takes about 5 minutes; pools
  about 25.
- **Reads** take about 8 minutes each.
- **Compares:** about 15:05 to 15:15 for the four arms, about 15:30 for pools.
- **The ablate** (CPU): about 14:50 to 15:00.

## Results

Development numbers; the paper's numbers come from one declared confirmation run.

None yet.
