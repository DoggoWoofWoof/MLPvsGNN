# P1c3: KIND_T's walk positions as an input to both models, one screen per family

Declared 14 October 2026, about 15:55. It comes after P1c2's verdict ([P1C2_POOLS_KIND_T.md](P1C2_POOLS_KIND_T.md),
section 7, filed 9c8ca88) and before any number of this stage. Numbers here are development numbers.

**Why P1c3.** The user, 14 October: "if there are good gains then we should try improving no". P1c2 retrained both
models on KIND_T's pools, the final pool. It was NOT_ADOPTED on one LOSS, but the pools gave real gains:
- metaqa +0.020 to +0.021 and 2wiki +0.039 to +0.045, on both models and both fits;
- on musique and webqsp the pools added golds (+0.042 and +0.108 of gold recall), and the models did not rank them
  into the top five. These are RANKED gaps: the gold is in the pool and ranked out.

P1c3 keeps P1c2's pools, roots, models, settings and rule. It adds one input, the same for both models: where KIND_T's
construction put each pool row.

## 1. The question

The models read each row's features from the look, including two walk columns (WALK and WALKF, the look's walk from
the seeds). They do not read where KIND_T put the row:
- **a passage question's** order is RRF5T, the RRF of A1 and the two steered walks from the links, each on the
  structural union and on all families, with the first stage's lists (R0);
- **an entity question's** order is ROUTEst on the route (the walk from the links, backfilled by A1), and A1 off it.

KIND_T's order is itself a ranker, built from no gold. Its R@5 on s1eval (P1e's position files) beside the models'
reads on KIND_T's pools (P1c2's filed reads):

| dataset | KIND_T's order | zrc on P1c2's pools (L-musique / L-hotpotqa) | zsp on P1c2's pools (L-musique / L-hotpotqa) |
| --- | ---: | --- | --- |
| metaqa | 0.2359 | 0.7994 / 0.8002 | 0.8008 / 0.8008 |
| squad | 0.9003 | 0.9123 / 0.9134 | 0.9119 / 0.9111 |
| musique | 0.5025 | zero-shot 0.4789 / 0.5651 | zero-shot 0.4792 / 0.5707 |
| hotpotqa | 0.7948 | 0.9036 / zero-shot 0.8363 | 0.9062 / zero-shot 0.8475 |
| 2wiki | 0.7887 | 0.9128 / 0.9125 | 0.9174 / 0.9171 |
| webqsp | 0.1030 | zero-shot 0.3466 / 0.3142 | zero-shot 0.3078 / 0.2603 |

webqsp is zero-shot on every fit (it never trains).

- **On musique zero-shot, both models rank KIND_T's pool below KIND_T's own order** (0.4789 and 0.4792 against 0.5025).
- Everywhere else the models are above the order, so the order alone is not the answer. The question is whether its
  positions, as one more input, lift the RANKED gaps without a LOSS anywhere, in-domain and zero-shot.

## 2. The KT input

Eight columns per pool row, computed when the pool is built, from no gold (`scripts/p1c3_kt.py feats`):

| column | what it is |
| --- | --- |
| KIND_T | the row's place in KIND_T's order |
| A1st | its place in A1's walk (HippoRAG 2's restart) on the structural union |
| A1all | the same on the union of all families |
| Kst | its place in the walk from the corroborated links on the structural union |
| KTst | its place in P1e's steered walk from the links on the structural union |
| KTall | the same on the union of all families |
| fill | 1.0 when the row came from KIND_T's fill, not from base or seeds |
| entity | 1.0 on every row of an entity question (P1e's kind) |

- **A place p is scaled to log1p(p) / log1p(7,000).** A row not in the order gets 1.0. Every order is cut at 7,000, so a
  row in it scores below 1.
- **Not a column:** R0, the first stage's lists. The models already read the lists' ranks and scores.
- **The sources, kind and order are P1e's,** imported unchanged (`scripts/p1e_anchor_kind.py`), and recomputed on the
  same questions as P1c2's pools (`scripts/p1c2_pools.py`, unchanged).
- **The entity column is almost a dataset switch on these six.** It is 1.0 on every metaqa question and on 0.967 of
  webqsp's, and 0 on the passage corpora except one hotpotqa question. It is read from the node texts (a name-only
  start), not from the dataset's name. The pools already treat the two kinds differently.

**Checks, run by `feats`; any failure stops the stage:**
- P1e's verdict chose KIND_T. P1e's part for the dataset was written by P1e's filed script, on the pinned freeze, with
  no limit, and its name-only count is the one found here. P1d's cut is 7,000.
- The pools file is P1c2's KIND_T file and holds the carve's questions in order.
- The cache's parts are P1c2's: each array's sha256 is its record's, each question appears once, and every question is
  the carve's.
- On every question:
  - |I_q|, base and seeds are the filed ones;
  - KIND_T's fill, recomputed here, is P1c2's filed fill, node for node;
  - the cache's rows are base, seeds and the fill, each node once.
- An order that holds a node twice, or more than 7,000 nodes, stops the stage.

**Written to** `outputs/p1c3/kt/<D>/<carve>/`:
- `kt.npy`: rows × 8, float16, in the cache's row order;
- `n.npy`: the rows per question;
- `ids.json`;
- `record.json`: the sha256 of every input (cache parts, pools file and manifest, the four scripts, P1e's verdict, the
  freeze).

The arrays (about 0.86 GB over the eleven carves) are never committed. The records are fetched and committed with the
filing.

## 3. The two arms

- **zrck** is zrc (the MLP) plus the KT head. **zspk** is zsp (the GNN) plus the KT head, added before zsp's
  propagation reads the score.
- **The head:**
  - 9 inputs: the eight columns, and the row's score z-scored over its question's rows and clipped at ±8, as zprop's
    input is;
  - one hidden layer of 16 (GELU), and one output added to the score: 177 parameters;
  - the first layer from its own generator (seed + 3301), the last layer zero.

  At the start, each arm's forward is its P1c2 arm's, bit for bit, and the global generator is where that arm leaves
  it (the self-test).
- **Everything else is P1c2's:** the settings, seeds and batches, the card, and P1c2's roots (looks, caches, chains,
  links, gate).
- **Loading** (`zp1c3.load_kt`): the KT file must be P1c3's, on the same cache parts and root, with the same questions
  in the same order and the same rows per question.

Flagged: the KT columns come from walks over the graph, done when the pool is built and before any row is scored. Both
models read the same columns, as they read WALK and WALKF today. The MLP's head reads only its own row: its columns and
its own score's z-score over the question's rows, with no edge. So its scoring stays free of message passing.

## 4. The screen

- **Fits:** four on the host's card: `scr-p1c3-zrck`, `scr-p1c3-zrck-hp`, `scr-p1c3-zspk` and `scr-p1c3-zspk-hp`,
  each arm on L-musique and L-hotpotqa (`zp1c3.py train`).
- **Reads:** each fit, p@swa, on the six s1eval carves of P1c2's pools.
- **Bases, which decide:** each family's card fit of the split on today's pools, as in P1c2:
  - zrck: `scr-zrct` and `scr-zrct-hp`;
  - zspk: `scr-zspg` and `scr-zspg-hp`.
- **Beside, deciding nothing:** P1c2's own fit of the same arm and split (`scr-p1c2-zrc`, `scr-p1c2-zsp` and their -hp
  fits), which reads the KT input's own step on KIND_T's pools, and step 1's fit of the split.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md). GAIN, LOSS and
  WITHIN are docs/SCREENS.md's.

### The rule, fixed before any number (P1c2's, unchanged)

- **ADOPT_FOR_FULL_RUN:** neither arm's re-call has a LOSS on any of its twelve reads, and the two arms together have
  at least one GAIN.
- **NOT_ADOPTED:** otherwise.

There is one input and one rule for all six datasets, and nothing is chosen per dataset.

Reported beside, deciding nothing:
- FC@5 and hit@1 per read;
- each read against P1c2's fit of the same arm and split;
- KIND_T's order's own R@5 (section 1).

## 5. What follows

- **ADOPT_FOR_FULL_RUN:** the full run, six splits per arm under grade-nullx, in its own file. Then the KT models are
  the claims' models once the serving cost below is timed.
- **NOT_ADOPTED:** today's models stay the claims' models. The record says, read by read, where the KT step moved R@5
  against P1c2's fits, and P3's ledger continues from there.
- **Serving cost, declared now.** The input reads six orders per question. KIND_T's build (P4a's timed form) runs only
  the walks a question's kind needs:
  - a passage question: A1st and A1all, and KTst and KTall when it has links;
  - an entity question on the route: Kst, then A1st only when Kst offers fewer than B_q nodes outside base and seeds;
  - an entity question off the route: A1st.

  Serving the KT input adds every walk its build skips: Kst on passage questions, and A1all, KTst and KTall (and Kst
  or A1st where the build skips them) on entity questions.
  - Both models read the same input, so the pair stays fair.
  - The end-to-end time of both rises. P4a found the walks are 88 to 97% of the build, and on hotpotqa and 2wiki the
    build is 99% of either model's end to end.
  - That cost is timed cold, in P4's table, before any claim uses a KT model. P4b's faster exact walks apply to these
    walks too.

## 6. How it runs

- **Code:**
  - `scripts/p1c3_kt.py`: stage `feats`, one dataset, and with `--carve`, one carve. Its self-test passes:
    - each place equals a brute force on 300 cases, and an order holding a node twice is refused;
    - the columns by hand, and an order longer than the cut is refused;
    - P1c2's fill, the three checks and six refusals, and a short question;
    - the cache's parts, and a changed array refused.
  - `outputs/mp_unified/zp1c3.py`: train, read, compare, pair and recall. Its self-test passes:
    - the two arms registered, with zspk's head before zsp's propagation;
    - the KT batch equals the KT rows of the batch's questions;
    - eight refusals of a KT file that is not this stage's or not the carve's;
    - at the start, each arm equals its P1c2 arm bit for bit, in eval and in training, and the global generator is
      unmoved;
    - the head and zspk's propagation by hand; a batch without the input equals the base; the clip;
    - toy fits move the head and repeat bit for bit;
    - train and read refuse another arm, split or fit;
    - relz's pair and re-call against today's fits, stamped P1c3's.
- **No development run on real data.** P1c2's caches are on the host only. The checks run on every question, and a
  carve writes nothing unless they all hold.
- **Host items** (`p1c3-*`):
  - eleven `p1c3-kt-<D>-<carve>` items on the CPU. They are sized from P1c2's pools items plus the cache's rows and the
    columns: 2wiki 8 CPUs and 8 GB, hotpotqa 6 and 6.5, metaqa 3 and 5.5, webqsp 3 and 4.5, musique 2 and 4, squad 2
    and 3. The fit carves come first.
  - the four fits on the card (`--frac 0.30`, share 0.32, three at a time; the measured peaks of P1c2's fits are 3.5
    and 5.2 GB, and the input adds at most 0.4 GB). Each fit waits for its training carves' items. zspk-hp, the
    longest, goes first.
  - the four reads, after all six s1eval items;
  - then 4 compares, 2 pairs and 2 re-calls.
- **Disk:** under 1 GB.
- **ETA:**
  - the fit carves' inputs: about 12 minutes (hotpotqa's fit is the longest, from P1c2's pools at 6 threads);
  - the s1eval inputs: about 20 minutes, beside;
  - the fits: about an hour, three at a time (P1c2's zsp-hp fit took 57 minutes);
  - reads, compares and calls: about 10 minutes.

  The verdict is about 1 hour 30 minutes after the items start.
- The encoder stays frozen. webqsp never trains. Test splits are never read.

## 7. Results

To be filed.
