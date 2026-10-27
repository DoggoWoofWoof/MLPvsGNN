# PSM: a pool-softmax input for the MLP (zrckpg)

Declared 27 October 2026 at 23:10, under WQ ([WQ_WEBQSP_TRAINS.md](WQ_WEBQSP_TRAINS.md), section 9, step 1). WQ's base
set has no number yet and no PSM fit has run. A PSM verdict is a WQ verdict only.

## 1. Why

- **The user's question**, 27 October: "also can we normalize the features across the pool would that help, like
  doing softmax like how transformers work etc?" Before it: "we are just normalizing across the pool for a better
  representation of scores, if we are already giving raw scores doesnt the model know score of feature x of this node
  is more than the other node?"
- **What zrckp reads now.** For each block it reads three things per row:
  - the row's raw columns;
  - their z-scores over the question's pool, taken against the pool's retrieved rows (zret);
  - the block's keep flag.

  A z-score says how far a row sits from the pool's middle, in units of the pool's spread. It does not say how far
  the row sits below the pool's top, or how many rows crowd that top. A pool with one clear leader and a pool with
  twenty near-ties look alike in z.
- **What a softmax adds.** For a column with z-scores z over the question's rows:

  g_i = log softmax(z)_i = z_i - log sum_j exp(z_j)

  This is the row's gap below the pool's soft best. When one row stands out, its gap is near 0 and every other row's
  is far below. When k rows tie at the top, each gets about -log k. The denominator is the normalisation a
  transformer's attention uses, here with no learned query or key.
- **The MLP's rule** (docs/SCREENS.md, the twenty-third round, 8 October, and its 14 October reading) holds. In one
  line: **this input aggregates over the question's rows, but only over the cache's fixed columns, as zret's z-scores
  do. No learned value and no score of any row enters it, and it is computed before any row is scored.** It is a fixed
  normalisation, not a learned exchange between rows.
- **What stays out.**
  - **SEMB.** Its columns are a learned product of the question's and the node's maps, so a softmax over them would
    pass learned values between rows.
  - **The eight KT columns.** They are already positions within the pool's orders, and the KT head stays pointwise.
  - **The chain match's features.** They are learned.
- **One open question for the user.** This file does not decide it, and PSM leaves it as it is. zret, like every lean
  MLP since lean_mlp, z-scores SEMB's learned product over the pool. P1c3p's file called zret's statistics
  frozen-feature statistics. For SEMB they are statistics of a learned product, so under the strict reading a row's
  input depends on the other rows' learned values through their mean and spread. It needs the user's ruling before
  Claim 1 is worded.

## 2. The input

The input covers every fixed block of zrckp's pick set, which is every block but SEMB: rank 5, dense_cos 1,
topo_STRUCT 11, depth_STRUCT 18, SEED 3, WALK 16, DISTS 14 and WALKF 18, so **86 columns**. For each block:
1. **z** is zret's z-score of the block's raw columns over the question's pool, against its retrieved rows
   (lean_screen3.seg_zscore_ref). It is the same z that zret already feeds the model.
2. **g** is the log-softmax of z over the question's rows, column by column, at temperature 1. The unit is z, so each
   column's own spread over the pool sets its scale, and one fixed temperature means the same on every column.
3. **g is floored at -10**, so it stays finite and bounded. For scale: -log N is -3.9 for squad's 50-row pools and
   -7.7 for webqsp's 2,120-row pools.
4. **g is multiplied by the block's keep flag**, as the block's other inputs are.

g is computed without gradients, in float32. Every column is treated the same whatever its direction (a distance's
best is its smallest): the transform is monotone in the row's own value, and the MLP learns the sign. No column is
picked by hand.

**The runtime guard.** A block's gap is taken only when the block's raw columns are the batch's own tensor; otherwise
the forward stops. lean_screen.raw_of returns the batch's tensor for every block but SEMB.

## 3. The arm

**zrckpg** is zrckp (zp1c3p.py, imported unchanged) with the 86 gaps added to its first layer:

h1 = GELU(l1([raw m, z m, m] over the blocks) + W_g [g m] over the fixed blocks)

- **W_g** is 128 x 86, so **11,008 parameters**. It starts at zero, has no bias and is drawn from no generator. These
  are the only new parameters; the input itself has none.
- **Everything else is zrckp's:**
  - zrc's model (zrm.ZRM: zret plus the gated chain match) and the pointwise KT head (161 parameters);
  - P1c3's chain carve with the KT input;
  - the card: lr 0.002, wd 1e-4, dropout 0.1, 8 epochs, SWA from epoch 2, Adam, hidden 128;
  - the seed, the batches, the device (cuda) and WQ's roots.
- **At the start, zrckpg's forward is zrckp's bit for bit,** in eval and with the same dropout draws in training. The
  global generator is where zrckp leaves it (the self-test).
- **The class:** `ZRCKPG(ZRCKP, GapRet)`, where GapRet is zret with the gaps. The method order puts GapRet right after
  the chain match, so the KT head and the chain match wrap it unchanged.

## 4. The screen

- **Two fits**, seed 0, on WQ's card and roots (zwq.on_wq), on the host's GPU (WQ's placement):
  - `scr-psm-zrckpg` on WQ-musique;
  - `scr-psm-zrckpg-hp` on WQ-hotpotqa.
- **Reads:** each fit is read on WQ's six s1eval carves at p@swa.
- **Compares:** each fit is compared, by lean_screen's compare, with WQ's card fit of its split alone
  (`scr-wq-zrckp-s0`, `scr-wq-zrckp-s0-hp`). The output is `outputs/screen/scr-psm-zrckpg[-hp].json`.
- **When it runs:** only once WQ's summary is BASE_READY (WQ, section 7). The train checks it and stops otherwise.

## 5. The rule, fixed before any number

- **The re-call.** Each of the 12 reads (2 splits x 6 datasets) is called again under WQ's floors on zrckp
  (`outputs/wq/floors-zrckp.json`, status OK), by screen_recall.recall on the filed call and the R@5 difference. A
  read's base R@5 must equal the floors record's seed-0 R@5 for that read (the same card fit). Otherwise the verdict
  is INCOMPLETE.
- **The verdict:**
  - **ADOPT_FOR_FULL_RUN** when no read is LOSS and at least one is GAIN;
  - **NOT_ADOPTED** otherwise;
  - **INCOMPLETE** when a comparison or the floors are missing, are not WQ's or PSM's, or do not hold 12 reads.
- **Beside, deciding nothing:** the FC@5 and hit@1 differences per read.
- **The record:** `outputs/psm/screen-verdict.json`, with an `.md`.

## 6. What follows (fixed now)

Each next form gets its own declared file before its fit.
- **ADOPT_FOR_FULL_RUN → zspkg:** the same input on the GNN (zspk), so that F1's controlled pair keeps one input set.
  Then the pair's full run under WQ (seeds 0 to 2).
- **NOT_ADOPTED with a GAIN and a LOSS → zrckpu:** the gap without the pool-size term, log(N_q p_i) floored at 0, which
  is the row's lift above a uniform pool.
- **NOT_ADOPTED with no GAIN → zrckph:** the hard gap, z_i minus the pool's largest z, floored at -10. This is the
  softmax's limit as the temperature goes to 0.
- **INCOMPLETE:** the missing piece reruns. Nothing is decided.

## 7. How it runs

- **Code:** `outputs/mp_unified/zpsm.py`, new. It holds the arm, the train, read, compare and verdict commands, and a
  self-test that passes on the laptop before anything is queued. It imports zwq.py and, through zwq.on_wq, zp1c3p.py
  unchanged.
- **Commands:**

  | command | step |
  |---|---|
  | `train --split WQ-musique --name scr-psm-zrckpg --device cuda --host --cache-root outputs/wq/p1c2/cache` | a fit |
  | `read --name scr-psm-zrckpg --device cuda --host --cache-root outputs/wq/p1c2/cache` | its read |
  | `compare --split WQ-musique` | lean_screen's compare against `scr-wq-zrckp-s0` |
  | `verdict` | section 5's record |
  | `--selftest` | the laptop check |

- **Host items**, `psm-*`, are appended after WQ's summary item. Sizes are WQ's base-set sizes:
  - **psm-train-zrckpg[-hp]:** after wq-summary; 1.1 CPUs, 7 GB, `--frac 0.27`, GPU share 0.29.
  - **psm-read-zrckpg[-hp]:** after its train; 1.1 CPUs, 6 GB, the same GPU settings.
  - **psm-compare-zrckpg[-hp]:** after its read; 1 CPU, 4 GB.
  - **psm-verdict:** after both compares; 1 CPU, 2 GB.
- **Records.** The verdict and the two compare files are committed (`git add -f`). The fits' own records, arrays and
  models are never committed.

## 8. Results

None yet.
