# P5b: learned offsets, read as a reranker on the final pools (a diagnostic)

Declared 13 October 2026, about 16:00, before any number of the stage. It is P5's second stage
([PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md), "Offsets: learned functions, not fixed vectors"). It comes after:
- P5a stopped ([P5A_NODE_SPACE.md](P5A_NODE_SPACE.md), section 8);
- the pools became final: P1e's KIND_T ([P1E_ANCHOR_KIND.md](P1E_ANCHOR_KIND.md)), written per question by P1c2
  ([P1C2_POOLS_KIND_T.md](P1C2_POOLS_KIND_T.md)).

Nothing here is adopted, and no ranking model trains or reads.

## 1. The question

On each s1eval question's final pool, does a learned offset g(question, seed, graph features) put golds near the top
that the graph's hops and the encoder's cosine leave low?

- **The user's direction** (13 October, about 08:50): "the offset is learned offset or transformations it can be a
  combination of complex functions with input being both query seed and graph features etc". The user's rule (about
  10:40) makes it a reranker, and only after the pools are final: "the offset etc was for reranking".
- **What changes from P5a.** P5a's map picked the offsets per corpus from the question alone, and it collapsed: one
  chain took nearly every question. Its oracle chain (one chain per question, picked with the golds) was MORE than H2
  on the low read on five of six datasets. The space held more than the map read. g sets each offset from three things:
  - the question;
  - the seed;
  - the seed's place in the graph.

  The offset is a continuous vector, with four heads per seed.
- **Why the call fuses.** The models already read the hops and the cosine, which are H2's two parts. What g can add is
  the golds that H2 ranks low. So the call order is H2 fused with g's order by reciprocal rank (constant 60). It beats
  H2 only where g's top places hold golds that H2 puts lower. g alone against H2 is reported beside.
- **One read: the low read.** P5a's file said its next offset stage would be read on the same two reads. The miss
  read counted golds outside the pool. The pools are now final and parameter-free, and g is a reranker that never
  feeds them (the user's rule). So only the low read applies, and its pool is the final pool.

## 2. g

- **The space.** P5a's filed node tables, unchanged:
  - V (`u_<dataset>`, each node's learned vector) is the call's space;
  - V0 (`u0_<dataset>`, the frozen start) is beside.

  The frame (mu, P) is P5a's. A question's input is x(q) = [P (q − mu), 1].
- **Inputs, per question q and seed s (262 wide):**
  - x(q) (65);
  - u_s (64), the seed's vector in the space;
  - x(q)[:64] − u_s (64);
  - m_s (64): the mean of u_v − u_s over the seed's neighbours. The neighbours are the union over the three link
    families (M3B's stores), with the seed itself excluded. m_s is zero for a seed with no neighbour;
  - log(1 + the seed's degree) in each family (3);
  - 1/(60 + rank) of the seed in the question's dense list, and again in its SPLADE list (2). Ranks start at 1. A
    list that does not hold the seed gives its depth + 1 (1,001), as in P1e's steer.
- **What is computed when.**
  - m_s and the degrees depend only on the node and the graph. They form an index-time table, like the node vectors,
    and are flagged as index-time graph aggregations.
  - The ranks come from the question's first stage.
  - Nothing comes from the question's scored rows, so a model that reads g stays free of message passing.
- **The network:** LayerNorm(262) → Linear(262, 256) → GELU → Linear(256, 256) → GELU → Linear(256, 4 × 65).
  - For each of 4 heads h, the output is an offset Δ_h (64) and a weight logit a_h (1).
  - The last layer starts at zero, so every Δ starts at 0 and the weights start equal.
  - With β (below), g has 200,465 weights: one parameter set for all six corpora.
- **The score.**
  - The target: t_{s,h} = u_s + Δ_h(q, s).
  - ℓ_{s,h}(v) = −κ · e^β · |t_{s,h} − u_v|².
    - κ = 1 / (the median of |u_a − u_b|² over 100,000 random pairs of the corpus's nodes, seed 0), per corpus and
      space. It is a label-free scale.
    - β is learned and starts at 0.
  - p(v | q) = Σ over (s, h) of π_{s,h} · softmax over the candidates of ℓ_{s,h}(v).
    - π is the softmax of a over the question's (seed, head) pairs.
    - The candidates C_q are the final pool minus the seeds.

## 3. The fit

- **Data:** P5a's sample of each training dataset's fit carve, up to 2,000 questions drawn with seed 0 (metaqa,
  squad, hotpotqa and 2wiki on fit; musique on s1fit).
  - The pools are those carves' final pools, from P1c2's files.
  - A question's counted golds are its golds in C_q. A question with none is skipped and counted.
  - webqsp never fits.
- **The objective:** −log p(g | q).
  - It is averaged over the question's counted golds, then over the questions.
  - Every dataset weighs the same.
- **EM.** Which seed and head explain a gold is latent. The gradient of the mixture's log-likelihood is each
  (seed, head) pair's own gradient, weighted by its posterior responsibility. So each step is an E-step and a gradient
  M-step in one (online generalized EM).
- **Settings, fixed here and selected on nothing:**
  - Adam, learning rate 1e-3;
  - 30 epochs of 32 questions per batch, drawn in a seeded order (seed 0);
  - float32, with TF32 off;
  - the last epoch's g is read.
- **Two fits:** G on V (the call) and G0 on V0 (beside). Both use the same settings and the same order.

## 4. The read (s1eval)

- **Questions:** P5a's s1eval sample, up to 2,000 per dataset (all 1,503 of webqsp's). It is the sample that P5a's
  read1 read zsp on.
- **The pool:** the final pool P_q. It holds base and seeds, and KIND_T's first B_q nodes outside them (P1c2's s1eval
  file).
  - **Seeds:** today's: the dense top 5, then the SPLADE top 5.
  - **Candidates:** P_q minus the seeds.
- **Counted golds:**
  - **low (the call):** the golds in P_q that are neither seeds nor in zsp's top 5 on today's pool.
    - zsp's top 5 on the final pool needs P1c2's zsp fit, which waits for clean-up 15. Today's top 5 (P5a's read1)
      stands in for it.
    - A gold that KIND_T newly brings into the pool counts, since today's zsp never saw it.
  - **pool (beside):** every gold in P_q that is not a seed.
- **The orders.** Each order covers all candidates, ties by node id.
  - **F (the call):** the reciprocal-rank fusion of G's order and H2's order: 1/(60 + rank in G) + 1/(60 + rank in
    H2).
  - **H2 (the comparator):** P5a's low order. Ring 1 from the seeds first, then ring 2, then the rest, over the three
    families' full adjacency; within each, by the frozen cosine to the question.
  - **Beside:**
    - G alone;
    - G0 alone;
    - F0: the fusion of G0 and H2;
    - NEAR: g before any fit, on V (every Δ zero, equal weights, β = 0). It orders the candidates by nearness to the
      seeds;
    - FN: the fusion of NEAR and H2;
    - COS: the candidates by frozen cosine alone.
- **N:** 5, 10, 25, 50 and 100 (P5a's low grid). On hotpotqa and 2wiki the pools hold about 100 nodes, so N = 100
  reaches nearly all of them.
- **found(N)** is P5a's: the sum over questions of |the order's top N ∩ T(q)|, divided by the sum of |T(q)|. It is
  counted on the questions whose T(q) is not empty.

## 5. The call, fixed before any number

The checks come first. Any failure stops the stage:
- The sample is P5a's read1 sample: the sha256 of its ids, and every position.
- Today's pool and seeds equal read1's, question by question, as sets.
- The final pools file is P1c2's KIND_T file for the carve:
  - its sha256 equals the manifest's;
  - the manifest names docs/P1C2_POOLS_KIND_T.md and KIND_T.
- Each final pool is the file's:
  - base and seeds equal the filed ones, and today's pool's sha256 equals the filed one;
  - the expansion repeats no node and holds none of base and seeds;
  - the pool has its expected size, and it holds every seed.
- Every fit's loss is finite.
- Every read ran on one script and one g, whose sha256s are recorded.
- The node tables are P5a's filed ones, checked by sha256 against its space records.

For each dataset (all six, webqsp zero-shot), at every N:
- Delta(N) = found_F(N) − found_H2(N) on the low read. Its 95% interval comes from a question bootstrap (2,000
  resamples, seed 0).
- A point counts when |Delta| > 0.002 and its interval is clear of 0.
- The labels are P5a's:
  - **MORE:** some point counts above, none below;
  - **FEWER:** the reverse;
  - **MIXED:** both;
  - **TIE:** neither.

F **FINDS_MORE** when it is MORE on at least one dataset and FEWER on none. The stage **GOES_FURTHER** if F FINDS_MORE,
and otherwise **STOPS**.

Reported beside, deciding nothing:
- G, G0, F0, NEAR, FN and COS against H2 on the low read, labelled the same way;
- every order on the pool read;
- found(N) by the gold's ring from the seeds (1, 2, or beyond 2);
- at each N, the golds F finds that H2 does not, and the golds G finds that H2 does not;
- the fits:
  - the loss by epoch and by dataset;
  - β;
  - the mean |Δ| and the largest weight π, by dataset.

## 6. What follows

- **GOES_FURTHER:** P5c, in its own file. The models read g's outputs per candidate as inputs: log p(v | q), and the
  least scaled distance to a target.
  - One screen fit per family (zrc and zsp), on P1c2's final pools, against P1c2's fits, by P1c's rule.
  - g is fixed before any row is scored, and its inputs are index-time tables and the question's first stage. So the
    MLP stays free of message passing.
  - P5c reads P1c2's caches, so it comes after P1c2's results.
- **STOPS:** the learned offsets close in this form. The record says where they fell short:
  - F0 or FN MORE where F is not points at V or at the fit;
  - nothing MORE points at what g can reach in this space.
- The encoder stays frozen, nothing is finer than a node, and test splits are never read.

## 7. How it runs

- **Code:** `scripts/p5b_offsets.py`.
  - It imports two scripts unchanged:
    - P5a's: the sample, the frame, the rings, the cosines, H2's low order, and the bootstrap and labels. Through it,
      step 4e's carve runner;
    - step 4c's: the pools file's loader and its checks.
  - The self-test passes:
    - the batched score equals a question-by-question sum over seeds and heads, with padding;
    - at the start, g orders the candidates as the seeds' nearness alone;
    - on a toy problem whose golds sit at a question-chosen offset from a seed, the fit's loss falls and g finds the
      golds better than at its start;
    - the low order equals P5a's h2_orders, the fusion equals a brute-force sum of reciprocal ranks, and ties go by
      node id;
    - first-stage ranks are 1-based, with depth + 1 for an absent node; m_s and the degrees equal a brute-force
      neighbour mean;
    - the final pool is the union of base and seeds and the expansion. It refuses another base, another pool's
      sha256, an expansion that touches base and seeds, or a wrong size;
    - the verdict's labels and call come out as declared on made-up reads, in a temporary folder.
- **Development runs, no number.** `--limit Q` runs a stage on the first Q sampled questions and writes nothing. It
  logs only that every step ran, the checks, and that the fit's loss is finite. Where a stage's inputs are not on
  disk, it builds them in memory:
  - P5a's start from this corpus stands in for both spaces;
  - today's pools stand in for the final ones, where P1c2's file is not on disk;
  - for train and read, it builds this dataset's base and a one-epoch g.

  Ran before this file was committed, 40 questions each: base and train on musique; reads on musique, metaqa and
  webqsp. Every step ran and every check held, on P1c2's own files for musique and webqsp (stand-in pools on metaqa).
- **Host items** (light lane; each card share is the measured use plus room):

  | item | what | where |
  | --- | --- | --- |
  | `p5b-base-<dataset>` (five) | the fit sample's questions, seeds, ranks, graph features and candidates | 2 CPUs; 2 to 7 GB; after P1c2's pools for the dataset |
  | `p5b-train` | G and G0 | the card (0.1); 2 CPUs; 6 GB; after the five bases |
  | `p5b-read-<dataset>` (six) | every order and both counts | the card (0.1); 2 CPUs; 4 to 10 GB; after `p5b-train` and P1c2's pools |
  | `verdict` | run locally, on the fetched records | |

  - P5a's reads held every question's rings at once (34.7 GB on musique). These reads build each question's rings
    when they order it and drop them, and they read the cosines of the candidates alone.
  - Records go to `outputs/p5b/`. Only the records and the verdict are tracked. The base files and g stay on the
    host; the read arrays are fetched for the verdict and never tracked.
  - They write under 0.2 GB in all.
- **ETA:**
  - P1c2's six pools jobs ended by 15:48, every check passing;
  - the bases take about 10 minutes, side by side, and the two fits about 10;
  - the reads take 5 to 15 minutes each, side by side.

  The call should land about 45 minutes after the items start, if the card is free.

## 8. Results (filed 13 October 2026, about 16:45): STOPS

`python scripts/p5b_offsets.py verdict` on the fetched records (`outputs/p5b/verdict.json` and `verdict.md`). Every
check held:
- on all 2,000 sampled questions (all 1,503 on webqsp), today's pool and seeds equal read1's, and each final pool is
  P1c2's KIND_T file's;
- the node tables are P5a's filed ones, and kappa is the fit's;
- every read ran on one script and one g;
- every fit's loss is finite.

The host ran the stage in about 10 minutes: the bases in 10 seconds to 2 minutes, the two fits in 2 minutes on the
card, the reads in 35 seconds to 4.5 minutes.

**The call: STOPS.** F does not find more on every dataset: it is MORE on metaqa and 2wiki, FEWER on musique and MIXED
on webqsp.

| low read, against H2 | metaqa | squad | musique | hotpotqa | 2wiki | webqsp (zero-shot) |
| --- | --- | --- | --- | --- | --- | --- |
| **F (the call)** | **MORE** | TIE | **FEWER** | TIE | **MORE** | **MIXED** |
| G (beside) | MORE | FEWER | FEWER | FEWER | MORE | FEWER |
| F0 (beside) | MORE | TIE | FEWER | TIE | MORE | FEWER |
| FN (beside) | FEWER | TIE | FEWER | FEWER | FEWER | FEWER |
| where it fell short (the record) | - | what g reaches | what g reaches | what g reaches | - | what g reaches |
| low golds (questions) | 9,267 (681) | 79 (79) | 1,959 (1,318) | 238 (225) | 509 (438) | 6,436 (1,054) |

F against H2, Delta at N = 5, 10, 25, 50 and 100:
- metaqa: +0.026, +0.041, +0.079, +0.137, +0.210;
- 2wiki: +0.118, +0.187, +0.163, +0.081, +0.024;
- musique: -0.038, -0.066, -0.070, -0.097, -0.099;
- webqsp: +0.006, +0.008, -0.005 (not counted), -0.037, -0.080.

**What the reads say.**
- **Where g works: 2wiki and metaqa.** At N = 10, G alone finds 0.322 of 2wiki's low golds against H2's 0.136, and
  0.076 of metaqa's against 0.009. It reaches past the hops. At N = 25:
  - 2wiki: 0.458 of the ring-2 golds (H2 0.034) and 0.394 of those beyond ring 2 (H2 none);
  - metaqa: 0.196 of ring 2 (H2 0.000) and 0.137 beyond (H2 none).

  On 2wiki these are the golds one link from a found gold, which is what message passing buys there (P3b's BRIDGE).
- **Where it fails: musique, and webqsp zero-shot.**
  - musique: G's top 10 holds 0.007 of the low golds, against 0.159 for H2, 0.077 for g before any fit (NEAR) and
    0.140 for the cosine. Its pools hold about 2,083 candidates, so chance at N = 10 is about 0.005.
  - webqsp, which g never saw: G finds 0.012 against H2's 0.026.
  - hotpotqa and squad: G is below H2 alone; fused, it ties.
- **The fit memorized its questions (read beside, after the call).**
  - The fit's own loss on musique's fit questions fell to 0.88 nats per gold: the golds took about 0.42 of g's
    probability, as a geometric mean. On s1eval, g is near chance there. On squad's 104 fit questions the loss fell to
    0.004.
  - g has 200,465 weights and fit 5,783 questions for 30 epochs, with no held-out check.
  - Its offsets are 5 to 10 times the median distance between two nodes (root mean square |Delta| 5.5 to 9.6, against
    0.9 to 1.0). So its targets sit far outside the cloud of nodes, and the score acts as a direction in the space, not
    a nearby point.

  The record's diagnosis reads "what g reaches in this space", because neither F0 nor FN is MORE where F is not. The
  gap between the fit's own loss and s1eval says the fit is the first thing to fix. Whether 2wiki's and metaqa's gains
  are golds that fit questions also had is not in these records: the reads count golds per question, not which.
- **V against V0.** G and G0 move together: F0 is MORE where F is, and FEWER on webqsp. The space does not decide it.

**What follows (section 6, STOPS).** The learned offsets close in this form, and P5c does not start from it. The next
form is P5b2:
- the same g, with its fit stopped at the epoch that best predicts, on held-out fit questions, the golds no training
  question had;
- every number split into golds that fit questions had and golds they did not.

It is declared in its own file before any number. Nothing here is adopted, and nothing in P1-P3's files changes.
