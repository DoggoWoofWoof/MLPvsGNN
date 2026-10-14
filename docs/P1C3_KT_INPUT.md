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

Filed 14 October 2026, about 17:30. **ADOPT_FOR_FULL_RUN.**
- **zrck:** 6 GAINs and no LOSS in its 12 reads after the re-call (PROMISING).
- **zspk:** 6 GAINs and no LOSS (PROMISING).

Neither arm has a LOSS, and together they have twelve GAINs, so section 4's rule gives ADOPT_FOR_FULL_RUN. The full run
is [P1C3_FULL_RUN.md](P1C3_FULL_RUN.md).

**The gate** (`zp1c3f.py gate`) applied the rule to the two re-calls at 17:13:59 and wrote
`outputs/p1c3/screen-verdict.{json,md}`. Its record names zp1c3f.py by its sha256 with line ends normalized,
1e3e45f5…, which is 64d640f's file.

**How the items ran.** The first eleven KT items stopped at the package import, before any question was read. de806f3
made `--host` load the frozen package from the mirror, as P1c2's pools stage does, and the items ran again as
`p1c3-2-*`. Nothing else changed.

**Files:**
- `outputs/screen/scr-p1c3-{zrck,zspk}{,-hp}.{md,json}`;
- `outputs/screen/scr-p1c3-{zrck,zspk}-pair{,-recall}.{md,json}`. Host paths were cut from the two re-call records, so
  their bytes differ from the sha256 the gate recorded. The host's copies are unchanged.
- `outputs/p1c3/screen-verdict.{json,md}`;
- the eleven `outputs/p1c3/kt/<D>/<carve>/record.json`.

**delta R@5 against each arm's base on today's pools, with the re-call** (W = WITHIN; zs = zero-shot):

| arm | fit | metaqa | squad | musique | hotpotqa | 2wiki | webqsp (zs) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zrck | L-musique | +0.0225 GAIN | +0.0013 W | −0.0176 W (zs) | +0.0022 W | +0.0632 GAIN | +0.0372 GAIN |
| zrck | L-hotpotqa | +0.0259 GAIN | +0.0009 W | +0.0050 W | +0.0113 W (zs) | +0.0397 GAIN | +0.0820 GAIN |
| zspk | L-musique | +0.0215 GAIN | −0.0011 W | −0.0100 W (zs) | +0.0015 W | +0.0659 GAIN | −0.0017 W |
| zspk | L-hotpotqa | +0.0233 GAIN | +0.0026 W | +0.0107 GAIN | +0.0064 W (zs) | +0.0589 GAIN | +0.0418 GAIN |

The re-call changed three calls:
- L-musique's musique zero-shot read, on both arms: LOSS to WITHIN (floor 0.072);
- zrck's hotpotqa zero-shot read on L-hotpotqa's fit: GAIN to WITHIN (+0.0113, floor 0.0200).

**Against P1c2's fit of the same arm and split**, which is the KT input's own step on KIND_T's pools. These are called
at the floor 0.0075, with no re-call, and decide nothing:

| arm | fit | metaqa | squad | musique | hotpotqa | 2wiki | webqsp (zs) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zrck | L-musique | +0.0014 W | −0.0006 W | +0.0271 GAIN (zs) | +0.0026 W | +0.0238 GAIN | +0.0290 GAIN |
| zrck | L-hotpotqa | +0.0050 W | −0.0013 W | +0.0044 W | +0.0201 GAIN (zs) | +0.0003 W | +0.0544 GAIN |
| zspk | L-musique | +0.0001 W | −0.0005 W | +0.0220 GAIN (zs) | +0.0018 W | +0.0209 GAIN | +0.0150 GAIN |
| zspk | L-hotpotqa | +0.0032 W | +0.0024 W | +0.0047 W | +0.0024 W (zs) | +0.0162 GAIN | +0.0077 W |

**Did the pools' extra golds reach the top five?** This is P1c2's table ([P1C2_POOLS_KIND_T.md](P1C2_POOLS_KIND_T.md),
section 7) with P1c3 beside it. Each cell is delta R@5 against today's base, P1c2's arm → P1c3's arm. In brackets:
P1c3's share of the pools' change in gold recall.

| dataset | pools' recall change | zrc → zrck, L-musique | zrc → zrck, L-hotpotqa | zsp → zspk, L-musique | zsp → zspk, L-hotpotqa |
| --- | ---: | ---: | ---: | ---: | ---: |
| metaqa | +0.0324 | +0.0211 → +0.0225 (0.69) | +0.0209 → +0.0259 (0.80) | +0.0214 → +0.0215 (0.66) | +0.0201 → +0.0233 (0.72) |
| musique | +0.0421 | −0.0447 → −0.0176 (zs) | +0.0006 → +0.0050 (0.12) | −0.0320 → −0.0100 (zs) | +0.0060 → +0.0107 (0.25) |
| 2wiki | +0.0202 | +0.0394 → +0.0632 (3.13) | +0.0394 → +0.0397 (1.97) | +0.0450 → +0.0659 (3.26) | +0.0426 → +0.0589 (2.92) |
| webqsp | +0.1080 | +0.0082 → +0.0372 (0.34) | +0.0276 → +0.0820 (0.76) | −0.0167 → −0.0017 | +0.0341 → +0.0418 (0.39) |

squad's pools are today's, and hotpotqa's pools lose 0.004 of gold recall. Their rows are in the first table.

What this says:
- **metaqa and 2wiki gain on both arms and both fits, as with P1c2.**
  - metaqa's gain is the pools': the KT input adds +0.0001 to +0.0050 to P1c2's fits.
  - On 2wiki the KT input adds +0.0003 to +0.0238 to the pools' +0.039 to +0.045.
- **musique zero-shot: the KT input wins back about 0.6 to 0.7 of what KIND_T's pools cost.**
  - P1c2's fits lost 0.045 (zrc) and 0.032 (zsp) there. P1c3's lose 0.018 and 0.010, both WITHIN under the seed null.
  - KIND_T's own order reads 0.5025 there (section 1). zrck is now above it (0.5059) and zspk just below it (0.5012).
    With P1c2's pools both were below it (0.4789 and 0.4792).
  - In-domain (L-hotpotqa's fit), musique moves +0.0050 (W, zrck) and +0.0107 (GAIN, zspk).
- **webqsp, never trained: more of the pools' golds reach the top five.**
  - zrck keeps 0.34 and 0.76 of the pools' +0.108 (P1c2's zrc: 0.08 and 0.26).
  - zspk keeps 0.39 on L-hotpotqa's fit (P1c2's zsp: 0.32). On L-musique's fit, P1c2's LOSS (−0.0167) is now
    −0.0017, WITHIN.
- **hotpotqa's R@5 moves +0.0015 to +0.0113 (WITHIN on all four), but its hit@1 falls on all four fits**, each
  interval below 0:
  - zrck: −0.0186 [−0.0253, −0.0120] in-domain, −0.0424 [−0.0502, −0.0346] zero-shot;
  - zspk: −0.0081 [−0.0144, −0.0019] in-domain, −0.0433 [−0.0513, −0.0355] zero-shot.

  So on hotpotqa a gold is ranked first less often. hit@1 decides nothing here, and the full run reports it on all six
  splits.
- **musique zero-shot's FC@5 falls**: zrck −0.0331 [−0.0451, −0.0207] and zspk −0.0207 [−0.0314, −0.0103]. Its hit@1
  rises (+0.0534 and +0.0314). Fewer questions have all their golds in the top five, and more have a gold first.
- **2wiki's FC@5 rises the most**, by +0.059 to +0.133, and its hit@1 by +0.06 to +0.08.
- **squad** is the identity again: its pools are today's, and R@5 moves by at most 0.0026.

What follows, by the rule (section 5):
- **The full run goes ahead** ([P1C3_FULL_RUN.md](P1C3_FULL_RUN.md)).
  - It was declared at 17:09 (64d640f), before this verdict, and its eight fits started at 17:10 on the card's free
    slots.
  - Their reads waited on this gate and now follow each fit.
  - Its verdict, one for the pair over 36 reads per arm, is due about 19:40.
- **The KT models are not the claims' models yet.** That needs the full run's ADOPT, and then the KT input's serving
  cost timed cold (section 5) on P4b's faster exact walks. Until then, today's models on today's pools stay the
  claims' models.
