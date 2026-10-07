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

### scr-zret: PROMISING (16:46). Its full run started by itself (docs/FULL_ZRET.md)

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
