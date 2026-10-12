# P1c: the models' pools from P1b2's pick at today's sizes, one screen per family

Declared 13 October 2026, about 05:09, after P1b2's verdict (docs/P1B2_RANK_GATE.md, section 7) and before any
number of this stage. P1's fourth stage in [PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md), as section 5 of
[P1B2_RANK_GATE.md](P1B2_RANK_GATE.md) names it. Numbers here are development numbers.

## 1. The question

P1b2's verdict picked **MX**, P1b's mixture, for the models' pools. Not all six datasets were AT_OR_ABOVE, so section 5
of P1b2 applied: the models read the higher of RC and MX in mean ALL at |I_q| over the five training datasets' s1sel
carves. MX led, 0.936 against RC's 0.926 (P1b2, section 7).

At each question's own pool size |I_q|, MX's order holds a question's golds whole more often than today's pools on all
six datasets (P1b2's filed matched-size table, s1eval):

| dataset | rows (mean) | today's pools | MX's top \|I_q\| |
| --- | ---: | ---: | ---: |
| metaqa | 2,018 | 0.895 | 0.923 |
| squad | 50 | 0.980 | 0.981 |
| musique | 2,092 | 0.803 | 0.892 |
| hotpotqa | 94 | 0.968 | 0.969 |
| 2wiki | 106 | 0.906 | 0.914 |
| webqsp | 2,120 | 0.688 | 0.806 |

These columns are the order's own top |I_q|. P1c's pool keeps today's base and seeds and fills the rest from the order
(section 2), so its own coverage is measured by the pools stage and reported beside.

Today's lean bases were trained and read on today's pools:
- **the MLP base, zrc** (zrm's model over zrc's chain entries; no message passing);
- **the GNN base, zsp** (zrc's inputs plus one propagation head over each row's neighbours' scores, on zlink's pool
  edges).

A pool that holds more golds raises what a ranker can reach. It also changes the negatives the ranker trains on. P1c
asks whether the two bases rank better when trained and read on MX's pools at the same sizes, in-domain and
zero-shot. Neither model, nor its settings, seeds or batches, changes. The graph, the encoder and the carves are
today's.

## 2. The pools

- **Per question:** base ∪ seeds, then the first B_q nodes of MX's order outside them, with
  B_q = |I_q| − |base ∪ seeds|. That is `m3b_compile.build_pool(base, seeds, expansion)` with this expansion in place
  of the frozen one, step 4c's construction (scripts/step4c_walk_pools.py `replace_pools`, unchanged). Every pool
  keeps its frozen size and is sorted by node id. A question whose order offers fewer than B_q nodes keeps them all
  and is counted as short.
- **MX's order:** P1b's mixture ([P1B_RESTART_MIXTURE.md](P1B_RESTART_MIXTURE.md), section 2), which is P1b2's class
  MX.
  - Six sources per question: the seeds, our linked nodes and the RRF lists, each unwalked and walked on the
    dataset's regime union graph with step 4e's kernel.
  - Their weights are w(q) = softmax(W x(q)) over P1b's fifteen label-free features. W is P1b's filed gate: six
    sources by sixteen weights, fit by EM to the golds' mass on the five training datasets' fit carves.
  - The order is the mixture's mass, ties by node position. It is cut at 7,000, as P1b2 cut it.
- **squad:** |I_q| = |base ∪ seeds| = 50 on every question, so its pools are today's, bit for bit. That is the stage's
  identity check (section 3).
- **The carves:** the screen's eleven. The fit carve of each training dataset (fit on metaqa, squad, hotpotqa and
  2wiki; s1fit on musique), and s1eval on all six. The ids are step 1's (`outputs/step1/carves.json`).
- **Checks, run by `scripts/p1c_pools.py pools`; any failure stops the stage:**
  - the construction is the one P1b2's verdict picked, and the one this file declares;
  - on a fit carve, the fifteen features equal P1b2's gather question by question, and the chosen class's area
    equals the gather's;
  - on s1eval, the order's ALL and recall at |I_q| equal P1b2's filed record for MX;
  - step 1's host cache rows are the frozen pool sizes built here (step 4c's check).
- **Written to** `outputs/p1c/pools/<D>/<D>__<carve>.npz` in step 4c's format, with the chosen class per question,
  and a manifest `pools.json` holding each file's sha256 and counts.

P1b's gate was fit on these fit carves' golds (its EM reads their mass). The pools of a training question therefore see
its golds through 96 weights over label-free features. They never see them any other way. No s1eval question was ever
fit.

## 3. Everything computed from the pools

The pipeline is U1c's (docs/U1C_RETRAIN_ON_U.md), under P1c's roots. Today's files are only read.

- **Looks:** `outputs/mp_unified/look_p1c.py`, look_step4c's binding with P1c's pools file. After
  `m3b_compile.prepare` builds a carve's frozen pools, each question's pool is replaced by its P1c pool. The look stops
  on a missing question, base and seed set or frozen size that is not the filed one, or an expansion that is not the
  filed one's length or touches base and seeds. Everything else is look_x_six's scoring pass, with `--host --full`.
  Output goes to `outputs/p1c/look/<D>/<carve>`, with each shard's pools record. Step 4c's shard counts apply.
- **Caches:** `lean_cache.py` unchanged, `--look-root outputs/p1c/look --out-root outputs/p1c/cache`, with step 1's
  part counts. Then step 1's bases are copied into P1c's cache root (`p1c_pools.py copy-bases`).
- **Chain entries and pool edges:** `outputs/mp_unified/zp1c.py build`, U1c's builds with P1c's roots: rmatch's and
  zrc's chains on the typed datasets' carves (metaqa fit and s1eval, webqsp s1eval) into `outputs/p1c/chains`, and
  zlink's pool edges, today's graph inside each P1c pool, on all eleven carves into `outputs/p1c/links`. The relation
  tables are today's.
- **The gate** (`p1c_pools.py gate`), before any fit:
  - squad's two carves: every cache array equals step 1's, sha256 by sha256. One array, score2 (the six pair's stored
    scores, which no lean model reads), must be identical or within 1e-4, and its largest difference is filed.
  - every carve: each question's rows equal its filed pool size, and step 1's rows equal its |I_q|;
  - every look shard filed a pools record naming the manifest's file and counts.

  Training needs its PASS.

## 4. The screen

- **Fits:** four on the host's card: zrc and zsp, each on L-musique and L-hotpotqa (`scr-p1c-zrc`, `scr-p1c-zrc-hp`,
  `scr-p1c-zsp`, `scr-p1c-zsp-hp`). Each is `zp1c.py train`, which is rmatch's train with P1c's roots and cache.
- **Reads:** each fit, p@swa, on the six s1eval carves of P1c's pools.
- **Bases:** each arm's own card fit of the split on today's pools, trained the same way on the same device:
  - zrc: `scr-zrct` and `scr-zrct-hp`;
  - zsp: `scr-zspg` and `scr-zspg-hp`.

  The same questions and gold totals. R@5 is over the question's golds, so a gold the pools lose counts as a miss,
  and a gold they add counts as a hit only when it is ranked in the top five.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), each arm decided
  against its own base. GAIN, LOSS and WITHIN are docs/SCREENS.md's.

### The rule, fixed before any number

- **ADOPT_FOR_FULL_RUN:** neither arm's re-call has a LOSS on any of its twelve reads, and the two arms together have
  at least one GAIN.
- **NOT_ADOPTED:** otherwise.

Both families read the same pools, so a pool that costs either base is not adopted. One construction and one rule for
all six datasets; nothing is chosen per dataset.

Reported beside, deciding nothing:
- FC@5 and hit@1 per read, as the compares give them;
- each carve's ALL and gold recall at |I_q| under today's pools and P1c's, the short and changed pools, and the share
  of questions each class takes (from `pools.json`).

## 5. What follows

- **ADOPT_FOR_FULL_RUN:** the full run, six splits per arm under grade-nullx, in its own file. After it, P1c's
  construction is the pools of P3's gap ledger and of P5.
- **NOT_ADOPTED:** today's pools stay. The record says, read by read, whether the pools' extra golds reached the top
  five, which is where P3's ledger starts.
- **Latency.** The pools keep today's sizes, so the looks, the compiled columns and the models' per-question cost are
  unchanged. Building a pool is not: P1b2's seven walks per question replace today's breadth-first expansion. That
  cost is shared by both families of a pair, and it is timed cold in its own stage (P4) before any latency claim.

Flagged, as every pool so far: the walks are propagation over the graph. The pool is built before either model runs,
and both models read the same pool, so the MLP's own scoring stays free of message passing.

## 6. How it runs

- **Code:**
  - `scripts/p1c_pools.py`: stages `pools`, `copy-bases` and `gate`. Its self-test passes: the screen's eleven carves;
    the order outside base and seeds keeps its order; RC takes the hard read and MX takes class MX, and nothing else
    runs; step 4c's file and replacement keep each frozen size and count short pools; coverage; the gate's row,
    look-record and pick checks.
  - `outputs/mp_unified/look_p1c.py`.
  - `outputs/mp_unified/zp1c.py`: builds, train, read, compare, pair and recall. Its self-test passes: arms zrc and
    zsp as registered; P1c's chain and link roots in place and restored, with today's relation tables untouched; train
    takes no other arm, split, cache root or device and needs P1c's gate PASS; read refuses a fit that is not P1c's;
    relz's pair and re-call run against each arm's card fits.
- **Development run, no number:** on the laptop, `pools musique --limit 64` and `pools metaqa --limit 64` ran every
  step on both carves of each and wrote nothing. On the first 64 questions of each fit carve, the features and MX's
  areas equal P1b2's gather.
- **Host items** (131, `p1c-*`):
  - the four fits and reads on the card (`--frac 0.47`, share 0.49, two at a time), placed above the CPU chain;
  - `p1c-pools-<D>` (light lane, five threads), `p1c-copy-bases`;
  - 51 look shards and 39 cache parts (heavy lane, step 4c's measured sizes);
  - 17 builds and the gate;
  - then 4 compares, 2 pairs and 2 re-calls.
- **Disk:** step 1's caches for these eleven carves take 13.8 GB on the host. P1c's caches are the same size. With
  looks, chains and links, the stage writes about 45 GB. It runs once the host has room above the feeder's
  heavy-writer line (clean-up 15).
- **ETA:**
  - the pools: about 45 minutes, from P1b2's gather and apply times (2wiki and hotpotqa about 20 minutes each);
  - looks and caches: about 2 hours on the CPU share;
  - the four fits and reads: about 3 hours on the card, two at a time (U1c's fits took 47 minutes to 1 hour 20
    minutes each, its reads 10 to 15 minutes).
- The encoder stays frozen. webqsp never trains. Test splits are never read.
