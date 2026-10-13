# P1c2: the models' pools from the final pool, KIND_T, at today's sizes, one screen per family

Declared 13 October 2026, about 15:10. It comes after P1e's verdict ([P1E_ANCHOR_KIND.md](P1E_ANCHOR_KIND.md),
section 8) and before any number of this stage. Numbers here are development numbers.

**Why P1c2 replaces P1c.** P1c ([P1C_POOLS_RETRAINED.md](P1C_POOLS_RETRAINED.md)) was declared on P1b2's pick, MX,
whose gate is fit to golds. It was held at 10:45, before any fit, under the user's rule:
- nothing is learned until the pools are final;
- the links and the pool are parameter-free and the same for all six datasets;
- the pool holds the most golds in the fewest nodes.

P1e's KIND_T is that pool. P1c's MX pools ran on the host (`outputs/p1c/pools`) and are never read. P1c2 is P1c's
stage, unchanged, on KIND_T's pools, under its own roots (`outputs/p1c2/`).

## 1. The question

P1e chose **KIND_T** as the final pool. Its call is NOT_ALL_SIX: below P1a's bar by at most 0.005 on metaqa and 0.007
on hotpotqa.

Nothing in KIND_T is fit, and it reads no gold. Its candidates were shaped on the first 600 questions of each fit
carve, and it was chosen on s1sel (P1e, section 2).
- **An entity question** starts from nodes that are all name-only: its corroborated links, or its seeds when it has
  none. It takes ROUTEst. On the route (every link corroborated), that is the walk from the links on the structural
  family, backfilled by A1. Otherwise it is A1.
- **A passage question** takes RRF5T: RRF (c = 60) over A1 and the two walks from the links, each on the regime's
  structural union and on the union of all families, plus the first stage's RRF lists (R0). The walks from the links
  are steered by the first stage's lists.
- **Every order is cut at 7,000.**

At each question's own pool size |I_q|, KIND_T's order holds a question's golds whole more often than today's pools on
four datasets. On squad and hotpotqa it is within 0.004 below. These are P1e's filed records, s1eval:

| dataset | rows (mean) | ALL, today's pools | ALL, KIND_T's top \|I_q\| | recall, today's pools | recall, KIND_T's top \|I_q\| |
| --- | ---: | ---: | ---: | ---: | ---: |
| metaqa | 2,018 | 0.895 | **0.962** | 0.957 | **0.989** |
| squad | 50 | 0.980 | 0.979 | 0.980 | 0.979 |
| musique | 2,092 | 0.803 | **0.900** | 0.921 | **0.963** |
| hotpotqa | 94 | 0.968 | 0.965 | 0.980 | 0.979 |
| 2wiki | 106 | 0.906 | **0.960** | 0.964 | **0.984** |
| webqsp | 2,120 | 0.688 | **0.832** | 0.811 | **0.909** |

These columns are the order's own top |I_q|. P1c2's pool keeps today's base and seeds and fills the rest from the
order (section 2), so its own coverage is measured by the pools stage and reported beside.
- **squad:** its pools are today's, bit for bit (section 2).
- **hotpotqa:** a pool that keeps base and seeds may land on either side of today's.

**On these six datasets the kind acts almost as a KB switch** (s1eval shares):
- entity questions: every metaqa question and 0.967 of webqsp's;
- passage questions: every question of the four passage corpora except one of hotpotqa's 7,405.

The two bases are trained and read on today's pools:
- **the MLP base, zrc**: zrm's model over zrc's chain entries, with no message passing;
- **the GNN base, zsp**: zrc's inputs plus one propagation head over each row's neighbours' scores, on zlink's pool
  edges.

A pool that holds more golds raises what a ranker can reach. It also changes the negatives the ranker trains on. P1c2
asks whether the two bases rank better when trained and read on KIND_T's pools at the same sizes, in-domain and
zero-shot. Neither model changes, nor do their settings, seeds or batches. The graph, the encoder and the carves are
today's.

## 2. The pools

- **Per question:** base ∪ seeds, then the first B_q nodes of KIND_T's order outside them, with
  B_q = |I_q| − |base ∪ seeds|.
  - This is `m3b_compile.build_pool(base, seeds, expansion)` with this expansion in place of the frozen one. It is
    step 4c's construction (scripts/step4c_walk_pools.py `replace_pools`, unchanged).
  - Every pool keeps its frozen size and is sorted by node id.
  - A question whose order offers fewer than B_q nodes keeps them all and is counted as short.
- **KIND_T's order:** P1e's sources, kind and rule, imported unchanged (`scripts/p1e_anchor_kind.py`: the sources,
  the steered walks, ROUTE's condition, the kind, `p1e_orders`).
- **squad:** |I_q| = |base ∪ seeds| = 50 on every question, so its pools are today's, bit for bit. That is the stage's
  identity check (section 3).
- **The carves:** the screen's eleven. That is the fit carve of each training dataset (fit on metaqa, squad, hotpotqa
  and 2wiki; s1fit on musique), and s1eval on all six. The ids are step 1's (`outputs/step1/carves.json`).
- **Checks, run by `scripts/p1c2_pools.py pools`; any failure stops the stage:**
  - P1e's verdict chose KIND_T, the construction this file declares;
  - P1e's part for the dataset was written by P1e's filed script, on this freeze (the pinned RECORD_SHA256), with no
    limit;
  - the part's name-only node count is the one found here;
  - on s1eval, KIND_T puts every gold at the position P1e filed, question for question, and its ALL and recall at
    |I_q| equal P1e's filed record;
  - step 1's host cache rows are the frozen pool sizes built here (step 4c's check).
- **Written to** `outputs/p1c2/pools/<D>/<D>__<carve>.npz` in step 4c's format, with two flags per question: whether
  it is an entity question, and whether it takes the route. A manifest `pools.json` holds each file's sha256 and
  counts:
  - the changed and short pools;
  - B_q;
  - ALL and recall at |I_q| for today's pools and P1c2's;
  - the kind and route shares.

KIND_T has no weight, so no question's pool sees its golds in any way. P1c's MX pools saw the fit carves' golds through
96 weights; P1c2's do not.

## 3. Everything computed from the pools

The pipeline is P1c's (and U1c's, docs/U1C_RETRAIN_ON_U.md), under P1c2's roots. Today's files are only read.

- **Looks:** `outputs/mp_unified/look_p1c2.py`, P1c's binding (`look_p1c.py`, imported unchanged) with P1c2's pools
  file.
  - After `m3b_compile.prepare` builds a carve's frozen pools, each question's pool is replaced by its P1c2 pool.
  - The look stops on a missing question, on a base and seed set or frozen size that is not the filed one, or on an
    expansion that is not the filed one's length or touches base and seeds. It also refuses a pools file that is not
    KIND_T's.
  - Everything else is look_x_six's scoring pass, with `--host --full`.
  - Output goes to `outputs/p1c2/look/<D>/<carve>`, with each shard's pools record. Step 4c's shard counts apply.
- **Caches:** `lean_cache.py`, unchanged, with `--look-root outputs/p1c2/look --out-root outputs/p1c2/cache` and step
  1's part counts. Then step 1's bases are copied into P1c2's cache root (`p1c2_pools.py copy-bases`).
- **Chain entries and pool edges:** `outputs/mp_unified/zp1c2.py build`. These are U1c's builds with P1c2's roots:
  - rmatch's and zrc's chains on the typed datasets' carves (metaqa fit and s1eval, webqsp s1eval), into
    `outputs/p1c2/chains`;
  - zlink's pool edges (today's graph inside each P1c2 pool) on all eleven carves, into `outputs/p1c2/links`.

  The relation tables are today's.
- **The gate** (`p1c2_pools.py gate`) runs before any fit. It is P1c's gate on P1c2's roots:
  - squad's two carves: every cache array equals step 1's, sha256 by sha256. One array, score2, must be identical or
    within 1e-4, and its largest difference is filed. score2 holds the six pair's stored scores, which no lean model
    reads.
  - every carve: each question's rows equal its filed pool size, and step 1's rows equal its |I_q|;
  - every look shard filed a pools record naming the manifest's file and counts, under KIND_T.

  Training needs its PASS.

## 4. The screen

- **Fits:** four on the host's card: zrc and zsp, each on L-musique and L-hotpotqa. They are named `scr-p1c2-zrc`,
  `scr-p1c2-zrc-hp`, `scr-p1c2-zsp` and `scr-p1c2-zsp-hp`. Each is `zp1c2.py train`, which is rmatch's train with
  P1c2's roots and cache.
- **Reads:** each fit, p@swa, on the six s1eval carves of P1c2's pools.
- **Bases:** each arm's own card fit of the split on today's pools, trained the same way on the same device:
  - zrc: `scr-zrct` and `scr-zrct-hp`;
  - zsp: `scr-zspg` and `scr-zspg-hp`.

  The reads use the same questions and gold totals. R@5 is over the question's golds, so a gold the pools lose counts
  as a miss. A gold they add counts as a hit only when it is ranked in the top five.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md). Each arm is
  decided against its own base. GAIN, LOSS and WITHIN are docs/SCREENS.md's.

### The rule, fixed before any number (P1c's, unchanged)

- **ADOPT_FOR_FULL_RUN:** neither arm's re-call has a LOSS on any of its twelve reads, and the two arms together have
  at least one GAIN.
- **NOT_ADOPTED:** otherwise.

Both families read the same pools, so a pool that costs either base is not adopted. There is one construction and
one rule for all six datasets, and nothing is chosen per dataset.

Reported beside, deciding nothing:
- FC@5 and hit@1 per read, as the compares give them;
- each carve's ALL and gold recall at |I_q|, under today's pools and P1c2's;
- the short and changed pools, and the share of entity and route questions (from `pools.json`).

## 5. What follows

- **ADOPT_FOR_FULL_RUN:** the full run, six splits per arm under grade-nullx, in its own file. After it, P1c2's pools
  are the pools of P3's gap ledger and of P5. P5 is the node-space EM and the learned offsets g(question, seed, graph
  features), as rerankers on the final pool.
- **NOT_ADOPTED:** no full run.
  - The record says, read by read, whether the pools' extra golds reached the top five. That is where P3's ledger
    starts.
  - P5 still builds on the final pool.
  - Today's models stay the claims' models until a model on the final pool beats them.
- **Latency.** The pools keep today's sizes, so the looks, the compiled columns and the models' per-question cost do
  not change. Building a pool does change: KIND_T's walks per question replace today's breadth-first expansion. That
  cost is shared by both families of a pair. It is timed cold in its own stage (P4) before any latency claim.

Flagged, as for every pool so far: the walks are propagation over the graph. The pool is built before either model
runs, and both models read the same pool, so the MLP's own scoring stays free of message passing.

## 6. How it runs

- **Code:**
  - `scripts/p1c2_pools.py`: stages `pools`, `copy-bases` and `gate`. Its self-test passes:
    - P1c's self-test;
    - KIND_T on toy sources: the route, off the route, and a passage question;
    - step 4c's file and replacement keep each frozen size and count short pools;
    - the positions check;
    - the refusals of a position file that is not P1e's and of a verdict that did not choose KIND_T.
  - `outputs/mp_unified/look_p1c2.py`.
  - `outputs/mp_unified/zp1c2.py`: builds, train, read, compare, pair and recall. Its self-test passes:
    - arms zrc and zsp as registered;
    - P1c2's chain and link roots in place and restored, with today's relation tables untouched;
    - train takes no other arm, split, cache root (P1c's and U1c's refused) or device, and needs P1c2's gate PASS;
    - read refuses a fit that is not P1c2's;
    - relz's pair and re-call run against each arm's card fits and are stamped P1c2's.
- **Development runs, no number:** on the laptop, `pools musique --limit 64` and `pools metaqa --limit 64` ran every
  step on both carves of each and wrote nothing. On s1eval, the 64 questions' golds sit at the positions P1e filed,
  and the at-|I_q| check ran.
- **Host items** (131, `p1c2-*`):
  - the four fits and reads on the card (`--frac 0.47`, share 0.49, two at a time), placed above the CPU chain;
  - `p1c2-pools-<D>` and `p1c2-copy-bases`, in the light lane. They are sized from P1e's measured runs: 2wiki 8 CPUs
    and 7 GB, hotpotqa 6 and 5.5, metaqa 3 and 4, webqsp 3 and 3.5, musique 2 and 3, squad 2 and 2;
  - 51 look shards and 39 cache parts, in the heavy lane, at step 4c's measured sizes;
  - 17 builds and the gate;
  - then 4 compares, 2 pairs and 2 re-calls.
- **Disk:** step 1's caches for these eleven carves take 13.8 GB on the host, and P1c2's caches are the same size.
  With looks, chains and links, the stage writes about 45 GB. The looks, caches and builds run once the host has room
  above the feeder's heavy-writer line (clean-up 15). The pools write well under 1 GB and run now.
- **ETA:**
  - the pools: about 45 minutes. This is P1e's measured runs scaled to the fit carves; 2wiki is the longest, at about
    42 minutes.
  - looks and caches: about 2 hours on the CPU share, after clean-up 15;
  - the four fits and reads: about 3 hours on the card, two at a time.
- The encoder stays frozen. webqsp never trains. Test splits are never read.
