# P5a: the node space read as a retriever on its own (a diagnostic)

Declared 13 October 2026, before any number of the stage. P5's first stage in
[PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md) ("P5: a fourth offset attempt, and what changes"). Nothing here is
adopted, and no ranking model trains or reads.

## 1. The question

From each s1eval question's seeds, at an equal candidate count, does a learned node space find more of the golds than
the graph's own hops from the same seeds? Two kinds of gold are counted:
- **the golds today's pools miss** (not in I_q): what a pool could take from the space;
- **the golds zsp ranks below 5** (in I_q, not a seed, not in zsp's top 5): what a model could read from the space's
  distances.

The program fixes the rule: a space that finds fewer goes no further. The three filed offset attempts (zgf, zgr and
zof) were read only through a model's head, and lost there (DIAG_ZGF_HARM: HEAD_AT_READ). This stage reads the space
alone first.

What changes from those attempts:
- every node has its own learned vector, fit to its corpus's graph (the earlier attempts mapped the frozen vectors
  through one shared matrix);
- the targets are composed offsets, seed + r1 and seed + r1 + r2, looked up by distance, with no walk at read time.

## 2. The space (label-free, per corpus, at index time)

- **The frame.** One 64-wide frame for all six corpora: the mean and the top 64 principal directions of 50,000 frozen
  node vectors drawn from each corpus (all of squad's 20,233 and metaqa's 43,234), seed 0.
  - The frozen vectors are the substrate's dense node vectors (1,536 wide, unit length). The encoder stays frozen.
  - It reads no label.
- **The start.** Each node's start is its frozen vector in the frame: u0 = P (x − mu), kept at 16 bits for all six
  corpora. The fit starts from the kept start. A question's input is its own frozen vector in the frame, with a
  trailing 1: x(q) = [P (q − mu), 1].
- **The edges.** All three link families on every corpus, so squad has a space too (its pools use no family, but its
  families exist). Self-loops are dropped.

  | family | edges | offsets |
  | --- | --- | --- |
  | structural, metaqa and webqsp | directed KB triples | one per named relation (9 and 7,058), each starting at its edges' mean displacement at the start |
  | structural, the four passage corpora | directed links, one relation written down (the link's direction fixes the sign) | 8 latent offsets; the E-step picks each edge's offset |
  | ner and knn, all six | one per unordered pair | 8 latent offsets each; the E-step picks each pair's offset and sign |

  A latent family's offsets start from a k-means (20 steps) over up to 200,000 sampled start displacements,
  sign-folded for ner and knn.
- **The fit:** a translation objective, as TransE fits a knowledge graph.
  - d(h, r, t) = |u_h + r − u_t|.
  - The loss per edge is softplus(d − gamma), plus the mean of softplus(gamma − d) over 16 tails drawn uniformly from
    the corpus.
  - gamma is the 5th percentile of the start's distance between 100,000 random node pairs: a label-free scale.
  - Each node's vector is u = u0 + delta. A prior of weight 1 pulls each step's touched nodes back to their start
    (their mean squared delta). The frozen vector is the start and the prior, and it is never changed.
  - The fit runs in float32 on the card, with TF32 off. The space is kept at 16 bits.
- **EM, online and hard.** At every step, each sampled edge of a latent family takes the offset (and, for ner and
  knn, the sign) that fits it best. Then one gradient step moves the node vectors and the offsets together.
- **The settings, fixed here and selected on nothing:**
  - 3,000 steps of 65,536 edges, shared equally by the three families;
  - SparseAdam for the node vectors and Adam for the offsets, both at a learning rate of 1e-3;
  - seed 0.
- **V0, the start's space (beside):** the same fit with every node held at its start, so only the offsets fit. It
  shows what the node fit adds over the frozen encoder.

## 3. The chain map (EM on the training datasets' fit carves)

- **Chains.** A one-hop chain is one signed offset of the corpus (each offset forward and backward). A two-hop chain
  is an ordered pair of them. Its target from seed s is u_s + r_c1 (+ r_c2).
- **The map,** one parameter set for all six corpora. The offsets are each corpus's own; the map reads them through the
  shared frame.
  - The hop gate: h(q) = sigmoid(g · x(q)).
  - One hop: pi(c | q) = h(q) · softmax_c(<W1 x(q), r_c>).
  - Two hops: pi(c1, c2 | q) = (1 − h(q)) · softmax_c1(<W1 x(q), r_c1>) · softmax_c2(<W2 x(q), r_c2>).
  - W1 and W2 are 64 by 65, and g is 65 wide: 8,385 weights.
- **The evidence** of chain c for question q is its golds' rank around its targets, the read's own measure (P1b2's
  lesson: fit to rank, not mass). It is read on the kept 16-bit tables.
  - For each gold g that is not a seed: the best over the seeds of 1/rank. rank is g's position among all nodes
    (seeds excluded) by distance to u_s + r_c.
  - A node counts as nearer only when its squared distance is below the gold's by more than
    1e-5 (|t|² + |u_g|²). That margin is float32 rounding, so a rounding tie does not cost the gold a place.
  - e(q, c) is the mean over the golds.
  - It is counted for the 32 chains whose targets come nearest to a gold. The search covers every one-hop chain, and
    the two-hop chains whose first hop is among the 64 one-hop chains nearest a gold (all of them on a corpus with 64
    or fewer). Every other chain gets 1/n (n the corpus's node count).
  - A question whose golds are all seeds carries no evidence. It is skipped and counted.
- **EM.**
  - E-step: each question's responsibility over its chains, pi(c | q) e(q, c) normalized.
  - M-step: the map's responsibility-weighted log-likelihood with an L2 penalty of 1e-3 (L-BFGS, strong Wolfe, up to
    100 inner steps). A two-hop chain's log pi is the gate's term plus each hop's log softmax, so the weighted sum
    needs only each hop's marginal responsibility. The M-step uses that exact form.
  - The map starts at zero. The loop stops when the penalized log-likelihood gains less than 1e-9 relatively, or after
    200 iterations.
- **Data:** up to 2,000 questions sampled from each training dataset's fit carve (metaqa, squad, hotpotqa and 2wiki
  fit; musique s1fit), seed 0. Every dataset weighs the same, and every question within it. webqsp never fits.
- V0 gets its own map, fit the same way on its own evidence.
- The map fits every two-hop chain, so it refuses a corpus with more than 128 one-hop chains. The five training
  corpora have 48 (the passage corpora) and 50 (metaqa). webqsp's 14,148 are read with the map, never fit.

## 4. The reads (s1eval)

- **Questions:** up to 2,000 sampled from each dataset's s1eval carve, seed 0 (all of webqsp's 1,503). The same
  sample in every item.
- **Seeds:** today's seeds (the dense top 5, then the SPLADE top 5, first occurrence kept). **Pools:** today's frozen
  pools I_q.
- **The orders**, each from the same seeds:
  - **V, the space:**
    - the map's 8 most likely one-hop chains, and its 32 most likely two-hop chains among the 8 by 8 best first and
      second hops;
    - each seed and chain gives a target, and the target lists the read's candidates by distance (ties by node id);
    - the lists are fused by the largest pi(c | q) / position, with ties broken by distance, then node id;
    - each list is read only as deep as can still reach the fused top 1,000 (exact).
  - **V0:** the same, on the start's space with its own map (beside).
  - **VO, the oracle chain (beside):** per question, on V's space, the one chain whose targets come nearest to a
    counted gold, every seed weighted the same. Its two-hop search takes the 64 nearest first hops.
    - It reads the golds, so it shows what a better map could reach over this space.
    - It is not a bound: V fuses 40 chains, and VO reads one.
  - **H2, the graph's hops (the comparator):**
    - rings 1 and 2 from the seeds over all three families' full adjacency (M3B's stores, which the pools read);
    - each ring by the frozen cosine to the question, ties by node id. The cosine is computed in float32 on the card,
      with TF32 off, in one pass over the corpus's frozen vectors;
    - in the miss read, then the dense first-stage list's remaining nodes.

    It is the strongest simple order of the hops: the graph's own reach, ranked by the encoder.
  - **H1, step 4's own hop expansion (beside):** expand_hops at h3_c25_v2000 over all three families, in its own
    order. That is 3 hops, at most 25 per seed and per frontier, and at most 2,000 visited.
- **Two reads,** each order cut to the read's candidates:

  | read | candidates | N | counted golds T(q) |
  | --- | --- | --- | --- |
  | miss | nodes outside I_q | 10, 25, 50, 100, 200, 500, 1,000 | the golds not in I_q |
  | low | I_q's nodes, seeds excluded | 5, 10, 25, 50, 100 | the golds in I_q that are not seeds and not in zsp's top 5 |

  - In the low read, V scores every candidate against its targets. H2 orders the candidates by ring (1, 2, then the
    rest), each by cosine.
  - zsp is today's GNN base, the filed J5 fit. Its top 5 is re-read on the sample through P2a's reader, unmasked: the
    same compile and read as P2a part 2, whose unmasked rows equal the filed J5 reads.
    - It runs in P2a's own parts, one thread each, as P2a part 2 ran.
    - Each part's pool must equal step 1's, question by question, and its R@5 is checked against P2a's filed read
      (section 5).
- **found(N)** = sum over questions of |the order's top N ∩ T(q)|, over the sum of |T(q)|, on the questions whose T(q)
  is not empty.

## 5. The call, fixed before any number

The checks come first. Any failure stops the stage:
- the read's pools and seeds equal P2a's reader's (zsp's), question by question, as sets;
- zsp's R@5 on the sample equals P2a's filed read on all but at most 1% of each dataset's questions;
- the sample is the same in every item (the sha256 of its ids);
- every fit's loss is finite;
- every read ran on one script, one frame and one map (their sha256s are recorded).

For each dataset (all six, webqsp zero-shot) and each read, at every N of the read's grid:
- Delta(N) = found_V(N) − found_H2(N), with a question bootstrap (2,000 resamples, seed 0) for its 95% interval.
- A point counts when |Delta| > 0.002 (step 4e's tie) and its interval is clear of 0.
- **MORE:** some point counts above and none below. **FEWER:** the reverse. **MIXED:** both. **TIE:** neither.

Per read, the space **FINDS_MORE** when it is MORE on at least one dataset and FEWER on none.

**GOES_FURTHER** if either read FINDS_MORE. Otherwise **STOPS**.

Reported beside, deciding nothing:
- V0, VO and H1 on both reads;
- the golds V finds that H2 does not, at each N;
- found(N) by the gold's ring from the seeds (1, 2, or beyond 2);
- the fit's diagnostics: residuals per family, the latent offsets' shares, the mean delta;
- the map's chain use: the hop gate's mean and the most used chains per dataset.

## 6. What follows

- **GOES_FURTHER on the miss read:** P5b, in its own file, lets the pools take the space's targets. It is graded as P1
  grades pools: coverage by size, against today's pools and the SOTA orders.
- **GOES_FURTHER on the low read:** P5b, in its own file, gives the models the targets' distances as inputs, with one
  screen fit per family.
  - The distances are fixed before any row is scored, so the MLP stays free of message passing.
  - The node table's fit goes in the cost table's index column.
- **STOPS:** P5's space closes in this form. The record says where it fell short, per dataset and read:
  - VO is MORE than H2 while V is not: the map;
  - neither is MORE: the space.
- Each later stage keeps the frozen encoder, and nothing finer than a node.

## 7. How it runs

- **Code:** `scripts/p5a_space.py`. It imports step 4e's carve runner (the pools, seeds and golds P1a to P1c read) and
  P2a's reader (zsp's top 5), unchanged. The self-test passes:
  - signed assignment recovers each displacement's offset and sign;
  - the frame is orthonormal;
  - the fit's loss falls on a toy graph, and the start's fit leaves every node at its start;
  - the chunked nearest-node search, the gold ranks and the chains nearest a gold equal brute force, with seeds
    excluded;
  - the fusion keeps each node's best score, and the depth cut leaves the fused top unchanged;
  - the rings equal a breadth-first search, and H2 comes by ring, each by cosine;
  - the M-step's marginal form equals the full sum over every chain;
  - EM's log-likelihood never falls, and on a toy problem the map picks each question's chain;
  - the verdict's labels and call come out as declared on made-up reads, in a temporary folder.
- **Development runs, no number.** `--limit Q` runs a stage on the first Q sampled questions and writes nothing. It
  logs only that every step ran, the checks and the fit's loss. Where a stage's inputs are not on disk, it builds a
  frame from this corpus, a short space and a zero map in memory.
  - These ran on the laptop: base on all six; train on squad, musique, metaqa and webqsp; read on musique, metaqa and
    webqsp (its 14,148 chains); read1 on musique and metaqa.
  - hotpotqa and 2wiki run the same code at their own sizes.
- **Host items** (light lane; the card's fraction from each table's size):

  | item | what | where |
  | --- | --- | --- |
  | `p5a-base` | the frame, and every corpus's start | 2 CPUs, 5 GB |
  | `p5a-train-<dataset>` (six) | the space and V0's offsets; on the five training datasets, the fit carve's evidence | the card (0.45 on 2wiki and hotpotqa, 0.25 on webqsp, 0.1 elsewhere); 2 CPUs; up to 10 GB |
  | `p5a-train-map` | EM for both maps | 2 CPUs, 2 GB |
  | `p5a-read1-<dataset>-<k>of<n>` (37, P2a's parts) | zsp's top 5 on the sample, through P2a's reader | 1 CPU each; P2a's memory, 4.5 to 6.5 GB |
  | `p5a-read-<dataset>` (six) | every order and both reads | the card (0.3 on 2wiki and hotpotqa, 0.2 on webqsp, 0.1 elsewhere); 2 CPUs; up to 12 GB |
  | `verdict` | run locally, on the fetched records | |

  - Records go to `outputs/p5a/`.
  - Each node table is kept at 16 bits: the starts and the spaces take about 3.6 GB for all six (2wiki 0.77 GB
    each). The host had 28.2 GB free at 06:52.
- **ETA:** the frame about 10 minutes, the spaces about 30 (2wiki and hotpotqa side by side on the card), the map a
  few minutes, the reads about 30. The call should land 1.5 to 2 hours after the start, if the card is free.
- No ranking model trains, nothing is adopted, test splits are never read, and the encoder stays frozen.

## 8. Results (filed 13 October 2026, about 10:00): STOPS

`python scripts/p5a_space.py verdict` on the fetched records (`outputs/p5a/verdict.json` and `verdict.md`). Every
check held: each read's pools and seeds equal P2a's reader's on all 2,000 sampled questions (all 1,503 on webqsp); zsp's
R@5 equals P2a's filed read on every sampled question (0 differ on all six); one script, one frame and one map; every
fit's loss is finite.

**The call: STOPS.** Neither read FINDS_MORE. V finds more golds than H2 only on metaqa:

| read | metaqa | squad | musique | hotpotqa | 2wiki | webqsp (zero-shot) |
| --- | --- | --- | --- | --- | --- | --- |
| miss (golds outside I_q) | MORE | FEWER | FEWER | FEWER | FEWER | FEWER |
| low (in I_q, below zsp's 5) | MORE | TIE | FEWER | FEWER | FEWER | FEWER |
| VO against H2, miss (beside) | MORE | MORE | MORE | FEWER | TIE | MIXED |
| VO against H2, low (beside) | MORE | MORE | MIXED | MORE | MORE | MORE |
| where it fell short, miss | - | the map | the map | the space | the space | the space |
| where it fell short, low | - | the map | the space | the map | the map | the map |

Examples (found(N), V against H2):
- miss, N = 100: metaqa 0.072 against 0.034; musique 0.024 against 0.062; hotpotqa 0.085 against 0.268; webqsp 0.006
  against 0.146.
- low, N = 10: metaqa 0.048 against 0.007; musique 0.036 against 0.183; 2wiki 0.128 against 0.191; webqsp 0.030
  against 0.041.

**What the reads say.**
- **The map collapsed.** On squad, musique, hotpotqa and 2wiki, one chain is the map's first choice for all 2,000
  questions; on metaqa two chains share them (1,365 and 635), and on webqsp one chain takes 1,425 of 1,503. V is
  close to one fixed chain per corpus, not a chain chosen by the question.
- **The space holds more than the map reads.** The oracle chain (VO: one chain per question, chosen with the golds, so
  not a result) is MORE than H2 on the low read on five of six datasets, and on the miss read on metaqa, squad and
  musique. On the low read, the record puts the shortfall on the map on four datasets (squad, hotpotqa, 2wiki, webqsp)
  and on the space only on musique.
- **Where the space reaches past the hops.** Beyond ring 2, H2 has only the dense list's remainder. In the miss read
  at N = 1,000, V finds more of those golds: metaqa 0.148 against 0.000 (1,262 golds), musique 0.038 against 0.000
  (130), 2wiki 0.025 against 0.000 (81), webqsp 0.006 against 0.002 (656). On the low read at N = 100, V is ahead on
  ring-2 golds on metaqa (0.308 against 0.050), webqsp (0.168 against 0.122) and hotpotqa (1.000 against 0.789, 19
  golds). On ring 1, H2 is ahead everywhere (squad's low read ties): V loses the hops' own neighbours.
- **The node fit.** Each node moved by a root mean square of 0.21 to 0.23 from its start. The fit lowered the loss
  (metaqa 1.253 to 1.137) where the start-only fit barely moved it (1.253 to 1.245). On the miss read, V is above V0
  at every N on 2wiki and up to N = 500 on metaqa, and below it at every N on musique (N = 1,000: 0.130 against
  0.197).

**What follows (section 6, STOPS).** P5's space closes in this form. The shortfall is mostly the map: offsets chosen
per corpus from the question alone, without the seed or the graph around it. The next offset stage is the user's
13 October direction: a learned offset g(question, seed, graph features), a function of all three, with no walk at
read time. It gets its own file, declared before any number, and is read against the same H2 and on the same two
reads. Nothing here is adopted, and nothing in P1-P3's files changes.
