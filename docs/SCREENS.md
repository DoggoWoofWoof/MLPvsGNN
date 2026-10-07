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

## 5. ETAs (7 October)

- **The four arm fits** started 14:25 to 14:27. An epoch takes 154 to 168 s, so each fit finishes about 14:50 to 14:55.
- **The null check and the pools fit** start as the card frees, about 14:52. The null takes about 5 minutes; pools
  about 25.
- **Reads** take about 8 minutes each.
- **Compares:** about 15:05 to 15:15 for the four arms, about 15:30 for pools.
- **The ablate** (CPU): about 14:50 to 15:00.

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
- **What follows.** The raw values carry the pool's size (ranks reach 2,000 in big pools and about 100 in small
  ones), and in training the only big pools are a KB's. zonly drops the raw values and dnorm standardises them per
  dataset; their screens test this directly. On webqsp, the KB read zero-shot, the structure blocks carry the
  model's lead over rrf (only DISTS kept: +0.099; only WALK kept: +0.070).
