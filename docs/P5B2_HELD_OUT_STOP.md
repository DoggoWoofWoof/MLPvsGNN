# P5b2: P5b's learned offset, its fit stopped on held-out new golds (a diagnostic)

Declared 13 October 2026, about 16:45, before any number of the stage. It is the next form that P5b's filed result
names ([P5B_LEARNED_OFFSETS.md](P5B_LEARNED_OFFSETS.md), section 8). It builds on P5b's filed files (a13b484).

Nothing here is adopted, and no ranking model trains or reads.

## 1. The question

P5b's g, fused with H2, found more of the golds zsp leaves low on metaqa and 2wiki, and fewer on musique. Its fit's
own loss on musique's fit questions fell to 0.88 nats per gold, while on s1eval it was near chance there.

**The question:** stopped at the epoch where it best predicts golds that no training question had, does g find more of
the low golds on some dataset and fewer on none?

**Beside it:** are g's gains on golds that the fit questions had (seen golds) or on new ones?

- The user's rule (13 October, about 10:40) stands. The pools are final and parameter-free, and g is a reranker on
  them: "the offset etc was for reranking".
- **Only the fit's stop changes.** Everything else is P5b's: g, its inputs, the two spaces, the orders, the reads, the
  grid and the call.

## 2. What changes: the stop

- **Data:** P5b's five base files as filed, on the host (`outputs/p5b/base/fit_<dataset>.npz`). Each is checked by
  sha256 against its filed record, and each record against the sha256 filed at a13b484. No base is rebuilt.
- **The split.** Per training dataset, the questions are permuted with seed 0 (`numpy.random.default_rng(0)`). The
  first round(0.2 n) are held out, and the rest train g.
- **New golds:** a held-out question's counted golds that no train question of the same dataset has as a counted gold.
- **The fit is P5b's, unchanged, on the train questions only:**
  - Adam, learning rate 1e-3;
  - batches of 32, drawn in a seeded order (seed 0);
  - float32 with TF32 off;
  - every dataset weighs the same.

  The held-out questions give no gradient.
- **The stop.** After every epoch, and at g's start (epoch 0), the held-out loss is computed:
  - for each held-out question with a new gold, −log p averaged over its new golds;
  - the mean over those questions, per dataset;
  - the mean over the datasets that have any.

  Each fit keeps the epoch with the lowest held-out loss, ties to the earlier epoch, out of at most 30 epochs (P5b's
  count). Epoch 0 is g's start, which orders the candidates as NEAR does.
- **Two fits, as in P5b:** G on V (the call) and G0 on V0 (beside), each stopped by its own held-out loss.
- **kappa** is P5b's, per space and dataset, checked against P5b's train record.
- **Why new golds.** On golds that the train questions share, the loss can fall by memorizing those nodes. A gold that
  no train question had can only be reached by what g learned about the question, the seed and the graph. That is
  what the s1eval read needs.
- **What it costs.** g trains on 80% of P5b's questions, so it may be weaker for that alone. P5b's own G is read
  beside it on the same questions (GB, section 3).

## 3. The read (s1eval)

**As P5b's section 4:**
- the same questions, pools, seeds and counted golds (low and pool);
- the same orders: F, H2, G, G0, F0, NEAR, FN, COS;
- the same N and found(N).

The s1eval sample is read a second time. The stop is fixed here and chosen on fit questions alone, so s1eval chooses
nothing.

**Two orders are added beside:**
- **GB:** P5b's filed G, from P5b's g file on the host (sha256 5c850a0b..., checked against P5b's train record);
- **FB:** GB fused with H2.

GB and FB should reproduce P5b's G and F. The number of questions where they differ is recorded, and 0 is expected.

**Seen and new golds.**
- A counted gold is **seen** when it is a counted gold of some question in P5b's fit sample of the same dataset.
  Otherwise it is **new**.
- That sample covers P5b2's train and held-out questions alike: P5b's fit touched both by gradient, and P5b2's by
  gradient or by the stop.
- webqsp never fits, so all its golds are new.

Every found(N) is also counted on seen and on new golds apart.

## 4. The call, fixed before any number

The checks come first. Any failure stops the stage:
- P5b's script, its five base records and its train record are the filed ones, by sha256. Each base file is its
  record's, and the node tables are the ones the base ran on.
- kappa equals P5b's.
- Some held-out question has a new gold.
- On the read, all of P5b's section 5 checks:
  - the sample is P5a's read1 sample;
  - today's pool and seeds equal read1's;
  - the final pools are P1c2's KIND_T files.
- Every fit's loss, on the train questions and held out, is finite.
- Every read ran on one script, one g and one GB, whose sha256s are recorded.

**The call is P5b's, unchanged.** For each dataset (all six, webqsp zero-shot), at N = 5, 10, 25, 50 and 100:
- Delta(N) = found_F(N) − found_H2(N) on the low read;
- its 95% interval comes from a question bootstrap (2,000 resamples, seed 0);
- a point counts when |Delta| > 0.002 and its interval is clear of 0.

The labels are P5a's: MORE, FEWER, MIXED and TIE. F **FINDS_MORE** when it is MORE on at least one dataset and FEWER
on none. The stage **GOES_FURTHER** if F FINDS_MORE, and otherwise **STOPS**.

Reported beside, deciding nothing:
- each of P5b's beside orders, and GB and FB, against H2 on the low read, labelled the same way;
- seen and new golds apart: found(N) for every order, and F, G, GB and FB against H2, labelled the same way;
- the fits: the held-out loss by epoch and by dataset (on new golds, and on all golds), the train loss, the kept
  epoch and β;
- as in P5b: the pool read, found(N) by ring, the golds F and G find that H2 does not, the mean |Δ| and the largest
  weight π.

## 5. What follows

- **GOES_FURTHER:** P5c, as P5b's section 6 says, on P5b2's g. Named now: g's fit questions are among the models'
  fit questions. So the models' fit rows need g's scores out of fold (from a g fit without those questions). P5c's own
  file declares how.
- **STOPS:** the learned offsets close in this form. The record says where they fell short:
  - F0 or FN MORE where F is not: V, or the fit;
  - the kept epoch of G is 0: "g at its start: no epoch predicted new golds better";
  - otherwise: "what g reaches in this space".

  What comes next is decided after the result, in its own file.
- The encoder stays frozen, nothing is finer than a node, and test splits are never read.

## 6. How it runs

- **Code:** `scripts/p5b2_stop.py`. It imports P5b's script unchanged, and through it P5a's and step 4c's.
- **The self-test passes:**
  - the split is seeded, disjoint and covering, with round(0.2 n) held out;
  - a subset of a base file equals its questions' rows, with fresh pointers;
  - new golds are the held-out golds that no train question has;
  - the held-out loss equals a question-by-question sum over seeds and heads, over all golds and over new golds;
  - the stop's means are as declared, with datasets that have no new gold left out;
  - on P5b's toy (golds at a question-chosen offset from a seed), the kept epoch has the lowest held-out loss, below
    half of g's start, and g comes back at that epoch;
  - on made-up reads, the verdict's labels, call and diagnosis come out as declared, and GB and FB are checked
    against made-up P5b reads.
- **Development runs, no number.** `--limit Q` runs a stage on the first Q sampled questions and writes nothing. It logs
  only that every step ran, the checks, and that the fit's losses are finite. Where P5b's files are not on disk, it
  works in memory:
  - it builds this dataset's base (P5b's development path);
  - P5b2's g is a one-epoch stopped fit on that base;
  - g at its start stands in for P5b's g.

  Ran before this file was committed, 40 questions each: train on musique; reads on musique, metaqa and webqsp.
  Every step ran and every check held, on P1c2's own files for musique and webqsp (stand-in pools on metaqa).
- **Host items** (light lane; P5b's measured sizes):

  | item | what | where |
  | --- | --- | --- |
  | `p5b2-train` | G and G0, each stopped | the card (0.1); 2 CPUs; 6 GB; reads P5b's bases on the host |
  | `p5b2-read-<dataset>` (six) | every order, both counts, seen and new golds | the card (0.1); 2 CPUs; 4 to 10 GB; after `p5b2-train` |
  | `verdict` | run locally, on the fetched records, beside P5b's fetched reads | |

  - Records go to `outputs/p5b2/`. Only the records and the verdict are tracked. g stays on the host; the read arrays
    are fetched for the verdict and never tracked.
  - They write under 0.2 GB in all.
- **ETA:**
  - the two fits take about 3 minutes on the card (P5b's took 2, and the held-out check adds a little each epoch);
  - the reads take 35 seconds to 5 minutes each, side by side.

  The call should land about 15 minutes after the items start, if the card is free.
